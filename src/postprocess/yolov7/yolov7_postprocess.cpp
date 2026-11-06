#include "yolov7_postprocess.h"

#include <cmath>
#include <cstdlib>
#include <iterator>
#include <sstream>

#include "common_util.hpp"

// YOLOv7Result methods implementation
float YOLOv7Result::iou(const YOLOv7Result& other) const {
    // Calculate intersection coordinates
    float x_left = std::max(box[0], other.box[0]);
    float y_top = std::max(box[1], other.box[1]);
    float x_right = std::min(box[2], other.box[2]);
    float y_bottom = std::min(box[3], other.box[3]);

    // Check if there is intersection
    if (x_right < x_left || y_bottom < y_top) {
        return 0.0f;
    }

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float union_area = area() + other.area() - intersection_area;

    return intersection_area / union_area;
}

bool YOLOv7Result::is_invalid(int image_width, int image_height) const {
    return box[0] < 0 || box[1] < 0 || box[2] > image_width || box[3] > image_height;
}

// Constructor
YOLOv7PostProcess::YOLOv7PostProcess(const int input_w, const int input_h,
                                     const float obj_threshold, const float score_threshold,
                                     const float nms_threshold, const bool is_ort_configured) {
    input_width_ = input_w;
    input_height_ = input_h;
    object_threshold_ = obj_threshold;
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    is_ort_configured_ = is_ort_configured;

    // Initialize model-specific parameters for YOLOv7
    cpu_output_names_ = {"output"};
    npu_output_names_ = {"onnx::Reshape_491", "onnx::Reshape_525", "onnx::Reshape_559"};
    anchors_by_strides_ = {{8, {{12, 16}, {19, 36}, {40, 28}}},
                           {16, {{36, 75}, {76, 55}, {72, 146}}},
                           {32, {{142, 110}, {192, 243}, {459, 401}}}};
}

// Default constructor
YOLOv7PostProcess::YOLOv7PostProcess() {
    input_width_ = 640;
    input_height_ = 640;
    object_threshold_ = 0.3f;
    score_threshold_ = 0.4f;
    nms_threshold_ = 0.5f;
    is_ort_configured_ = false;

    // Initialize model-specific parameters for YOLOv7
    cpu_output_names_ = {"output"};
    npu_output_names_ = {"onnx::Reshape_491", "onnx::Reshape_525", "onnx::Reshape_559"};
    anchors_by_strides_ = {{8, {{12, 16}, {19, 36}, {40, 28}}},
                           {16, {{36, 75}, {76, 55}, {72, 146}}},
                           {32, {{142, 110}, {192, 243}, {459, 401}}}};
}

// Process model outputs
std::vector<YOLOv7Result> YOLOv7PostProcess::postprocess(const dxrt::TensorPtrs& outputs) {
    // First align the tensors based on model configuration
    auto aligned_outputs = align_tensors(outputs);

    // Select the decode path by tensor rank rather than the global ORT flag,
    // as YOLOv5PostProcess does: a single fused 3D [1, N, 5+C] tensor is the
    // CPU/ORT decode; multi-scale feature maps (4D [1, (5+C)*A, gh, gw] or 5D
    // [1, A, gh, gw, 5+C]) are the anchor-grid decode. A *_nodecode export run
    // with the ORT tail on emits only 5D maps, and selecting by the flag found
    // no 3D tensor and returned nothing.
    std::vector<YOLOv7Result> detections;
    const bool fused_layout =
        !aligned_outputs.empty() && aligned_outputs.front()->shape().size() == 3;
    if (fused_layout) {
        detections = decoding_cpu_outputs(aligned_outputs);
    } else {
        detections = decoding_npu_outputs(aligned_outputs);
    }

    // Apply Non-Maximum Suppression
    detections = apply_nms(detections);

    return detections;
}

// Align output tensors based on model configuration
dxrt::TensorPtrs YOLOv7PostProcess::align_tensors(const dxrt::TensorPtrs& outputs) const {
    dxrt::TensorPtrs aligned;

    // Prefer a fused 3D [1, N, 5+C] tensor if present (ORT/CPU-decoded output);
    // selection follows the tensors, not the ORT flag (see postprocess()).
    for (const auto& output : outputs) {
        if (output->shape().size() == 3) {
            aligned.push_back(output);
            return aligned;  // fused output does not require reordering
        }
    }
    {
        // Align raw multi-scale NPU feature maps by stride. Accept both 4D NCHW
        // [1, (5+C)*A, gh, gw] and 5D [1, A, gh, gw, 5+C] layouts (mirrors
        // YOLOv5PostProcess::align_tensors's match_5d branch -- some YOLOv7
        // exports emit the anchor axis un-collapsed as a 5D row-major tensor
        // instead of the NCHW-folded 4D layout).
        for (const auto& as : anchors_by_strides_) {
            const int64_t anchors = static_cast<int64_t>(as.second.size());
            const int64_t gw = input_width_ / as.first;
            const int64_t gh = input_height_ / as.first;
            for (const auto& output : outputs) {
                const auto& s = output->shape();
                const bool match_4d = (s.size() == 4 && s[2] == gw && s[3] == gh &&
                                       s[1] == (num_classes_ + 5) * anchors);
                const bool match_5d = (s.size() == 5 && s[1] == anchors && s[2] == gh &&
                                       s[3] == gw && s[4] == (num_classes_ + 5));
                if (match_4d || match_5d) {
                    aligned.push_back(output);
                    break;
                }
            }
        }
    }

    if (aligned.empty()) {
        std::cerr << "[DXAPP] [ERROR] Failed to align output tensors based on "
                     "anchors_by_strides."
                  << std::endl;
        return outputs;  // Fallback to original outputs if alignment fails
    }

    return aligned;
}

// Decode model outputs to detection results
std::vector<YOLOv7Result> YOLOv7PostProcess::decoding_npu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv7Result> detections;

    // YOLOv7 typically has 3 output tensors for different scales
    // YOLOv7s typically has 3 output tensors for different scales
    // Each output contains: [batch, anchors * (num_classes+5), grid_y, grid_x]
    // Where (num_classes+5) = [x, y, w, h, obj_conf, cls_conf_0, ..., cls_conf_(num_classes-1)]
    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        const float* output = static_cast<const float*>(outputs[output_idx]->data());
        auto stride = std::next(anchors_by_strides_.begin(), output_idx)->first;
        int grid_x_size = input_width_ / stride;
        int grid_y_size = input_height_ / stride;
        // Process each grid cell
        const auto& anchors = anchors_by_strides_.at(stride);
        const bool is_5d = (outputs[output_idx]->shape().size() == 5);
        const int num_fields = num_classes_ + 5;
        // Layout-aware flat index for (anchor, field, grid_y, grid_x), mirrors
        // YOLOv5PostProcess::decoding_npu_outputs's is_5d flat_index:
        //  - 4D NCHW [1, A*F, gh, gw] : ((anchor*F + field)*gh + gy)*gw + gx
        //  - 5D      [1, A, gh, gw, F]: ((anchor*gh + gy)*gw + gx)*F + field
        auto flat_index = [&](int anchor_i, int field, int gy, int gx) -> int {
            if (is_5d) {
                return ((anchor_i * grid_y_size + gy) * grid_x_size + gx) * num_fields + field;
            }
            return ((anchor_i * num_fields) + field) * grid_x_size * grid_y_size +
                   gy * grid_x_size + gx;
        };
        for (int anchor = 0; anchor < static_cast<int>(anchors.size()); ++anchor) {
            int anchor_width = anchors[anchor].first;
            int anchor_height = anchors[anchor].second;
            for (int grid_y = 0; grid_y < grid_y_size; ++grid_y) {
                for (int grid_x = 0; grid_x < grid_x_size; ++grid_x) {
                    int objectness_idx = flat_index(anchor, 4, grid_y, grid_x);
                    auto objectness_score = sigmoid(output[objectness_idx]);
                    if (objectness_score < object_threshold_) {
                        continue;
                    }
                    int max_cls = -1;
                    float max_cls_conf = score_threshold_;
                    for (int cls = 0; cls < num_classes_; ++cls) {
                        auto cls_conf_idx = flat_index(anchor, 5 + cls, grid_y, grid_x);
                        auto cls_conf = objectness_score * sigmoid(output[cls_conf_idx]);
                        if (cls_conf > max_cls_conf) {
                            max_cls_conf = cls_conf;
                            max_cls = cls;
                        }
                    }
                    if (max_cls == -1) {
                        continue;
                    }
                    YOLOv7Result result;
                    std::vector<float> box_temp{0.f, 0.f, 0.f, 0.f};
                    result.confidence = max_cls_conf;
                    result.class_id = max_cls;
                    result.class_name = dxapp::common::get_coco_class_name(max_cls);
                    for (int i = 0; i < 4; i++) {
                        int box_idx = flat_index(anchor, i, grid_y, grid_x);
                        box_temp[i] = output[box_idx];
                    }
                    box_temp[0] = (sigmoid(box_temp[0]) * 2.0f - 0.5 + grid_x) * stride;
                    box_temp[1] = (sigmoid(box_temp[1]) * 2.0f - 0.5 + grid_y) * stride;
                    box_temp[2] = std::pow(sigmoid(box_temp[2]) * 2.0f, 2.0f) * anchor_width;
                    box_temp[3] = std::pow(sigmoid(box_temp[3]) * 2.0f, 2.0f) * anchor_height;
                    result.box.emplace_back(box_temp[0] - box_temp[2] / 2.0f);
                    result.box.emplace_back(box_temp[1] - box_temp[3] / 2.0f);
                    result.box.emplace_back(box_temp[0] + box_temp[2] / 2.0f);
                    result.box.emplace_back(box_temp[1] + box_temp[3] / 2.0f);

                    detections.push_back(result);
                }
            }
        }
    }
    return detections;
}

// Decode model outputs to detection results
std::vector<YOLOv7Result> YOLOv7PostProcess::decoding_cpu_outputs(
    const dxrt::TensorPtrs& outputs) const {
    std::vector<YOLOv7Result> detections;

    // YOLOv7 typically has 1 output tensor
    // output tensor contains: [batch, number of detections, attribute_size]
    // Where attribute_size = [x, y, w, h, obj_conf, cls_conf_0, ..., cls_conf_(num_classes-1)]

    for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        const float* output = static_cast<const float*>(outputs[output_idx]->data());
        auto num_dets = outputs[output_idx]->shape()[1];
        const int attribute_size = static_cast<int>(outputs[output_idx]->shape()[2]);
        for (int i = 0; i < num_dets; ++i) {
            const float* det = output + i * attribute_size;
            auto objectness_score = det[4];
            if (objectness_score < object_threshold_) {
                continue;
            }
            // Find the class with highest confidence
            int max_cls = -1;
            float max_cls_conf = score_threshold_;
            for (int cls = 0; cls < num_classes_; ++cls) {
                float class_conf = objectness_score * det[5 + cls];
                if (class_conf > max_cls_conf) {
                    max_cls_conf = class_conf;
                    max_cls = cls;
                }
            }
            if (max_cls == -1) {
                continue;
            }

            auto conf = max_cls_conf;

            YOLOv7Result result;
            result.confidence = conf;
            result.class_id = max_cls;
            result.class_name = dxapp::common::get_coco_class_name(max_cls);

            // Convert center coordinates to corner coordinates
            result.box.emplace_back(det[0] - det[2] / 2.0f);  // x1
            result.box.emplace_back(det[1] - det[3] / 2.0f);  // y1
            result.box.emplace_back(det[0] + det[2] / 2.0f);  // x2
            result.box.emplace_back(det[1] + det[3] / 2.0f);  // y2

            detections.push_back(result);
        }
    }
    return detections;
}

// Apply Non-Maximum Suppression
std::vector<YOLOv7Result> YOLOv7PostProcess::apply_nms(
    const std::vector<YOLOv7Result>& detections) const {
    if (detections.empty()) {
        return {};
    }

    // Sort detections by confidence (descending)
    std::vector<YOLOv7Result> sorted_detections = detections;
    std::sort(
        sorted_detections.begin(), sorted_detections.end(),
        [](const YOLOv7Result& a, const YOLOv7Result& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(sorted_detections.size(), false);
    std::vector<YOLOv7Result> result;

    for (size_t i = 0; i < sorted_detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }

        result.push_back(sorted_detections[i]);

        // Suppress overlapping detections
        for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
            if (!suppressed[j]) {
                float iou_value = sorted_detections[i].iou(sorted_detections[j]);
                if (iou_value > nms_threshold_) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return result;
}

// Set thresholds
void YOLOv7PostProcess::set_thresholds(float obj_threshold, float score_threshold,
                                       float nms_threshold) {
    if (obj_threshold >= 0.0f && obj_threshold <= 1.0f) {
        object_threshold_ = obj_threshold;
    }
    if (score_threshold >= 0.0f && score_threshold <= 1.0f) {
        score_threshold_ = score_threshold;
    }
    if (nms_threshold >= 0.0f && nms_threshold <= 1.0f) {
        nms_threshold_ = nms_threshold;
    }
}

// Get configuration information
std::string YOLOv7PostProcess::get_config_info() const {
    std::ostringstream oss;
    oss << "YOLOv7 PostProcess Configuration:\n"
        << "  Input dimensions: " << input_width_ << "x" << input_height_ << "\n"
        << "  Object threshold: " << object_threshold_ << "\n"
        << "  Score threshold: " << score_threshold_ << "\n"
        << "  NMS threshold: " << nms_threshold_ << "\n"
        << "  Number of classes: " << num_classes_ << "\n"
        << "  Is Ort Configured: " << (is_ort_configured_ ? "Yes" : "No") << "\n";

    for (auto& as : anchors_by_strides_) {
        oss << "  Stride: " << as.first << " Anchors: ";
        for (auto& a : as.second) {
            oss << a.first << ", " << a.second << " | ";
        }
        oss << "\n";
    }
    for (auto& cpu_output_name : cpu_output_names_) {
        oss << "  CPU output name: " << cpu_output_name << "\n";
    }
    for (auto& npu_output_name : npu_output_names_) {
        oss << "  NPU output name: " << npu_output_name << "\n";
    }

    return oss.str();
}
