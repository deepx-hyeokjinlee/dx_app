#include "sfa3d_postprocess.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kNumClasses = 3;
constexpr float kXMin = 0.0f;
constexpr float kXMax = 50.0f;
constexpr float kYMin = -25.0f;
constexpr float kYMax = 25.0f;
constexpr float kVeloZMin = -2.73f;

inline float sigmoid(float x) {
    x = std::max(-50.0f, std::min(x, 50.0f));
    return 1.0f / (1.0f + std::exp(-x));
}

struct Peak {
    int cls{0};
    float score{0.0f};
    int row{0};
    int col{0};
};

// SFA3D's _nms + _topk (sfa/utils/evaluation_utils.py): a cell survives only
// if it equals the maximum of its 3x3 neighbourhood, and the top K survivors
// are taken over ALL classes together. Ties keep the lower flat index
// (class, row, col). Without the suppression every neighbour of a strong peak
// came back as another box. max_k is kept for the call sites' signature.
std::vector<Peak> topkHeatmap(const float* heatmap, int num_classes, int h, int w,
                              int k, int /*max_k*/) {
    const int hw = h * w;
    std::vector<std::pair<float, int>> scored;
    scored.reserve(static_cast<size_t>(num_classes) * hw / 4);
    for (int cls = 0; cls < num_classes; ++cls) {
        const float* map = heatmap + static_cast<size_t>(cls) * hw;
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                const float v = map[r * w + c];
                bool is_max = true;
                for (int dr = -1; dr <= 1 && is_max; ++dr) {
                    const int rr = r + dr;
                    if (rr < 0 || rr >= h) continue;
                    for (int dc = -1; dc <= 1; ++dc) {
                        const int cc = c + dc;
                        if (cc < 0 || cc >= w) continue;
                        if (map[rr * w + cc] > v) { is_max = false; break; }
                    }
                }
                if (is_max) scored.emplace_back(v, cls * hw + r * w + c);
            }
        }
    }
    const int take = std::min(k, static_cast<int>(scored.size()));
    std::partial_sort(scored.begin(), scored.begin() + take, scored.end(),
                      [](const auto& x, const auto& y) {
                          return x.first > y.first || (x.first == y.first && x.second < y.second);
                      });
    std::vector<Peak> peaks;
    peaks.reserve(static_cast<size_t>(take));
    for (int i = 0; i < take; ++i) {
        const int flat = scored[static_cast<size_t>(i)].second;
        peaks.push_back({flat / hw, scored[static_cast<size_t>(i)].first,
                         (flat % hw) / w, flat % w});
    }
    return peaks;
}

const float* tensorData(const dxrt::TensorPtr& tensor) {
    return static_cast<const float*>(tensor->data());
}

int tensorChannels(const dxrt::TensorPtr& tensor) {
    const auto shape = tensor->shape();
    if (shape.size() < 3) return -1;
    return static_cast<int>(shape[shape.size() - 3]);
}

bool isSfa3dFiveHeadLayout(const dxrt::TensorPtrs& outputs) {
    if (outputs.size() < 5) return false;
    return tensorChannels(outputs[0]) == kNumClasses &&
           tensorChannels(outputs[1]) == 2 &&
           tensorChannels(outputs[2]) == 2 &&
           tensorChannels(outputs[3]) == 1 &&
           tensorChannels(outputs[4]) == kNumClasses;
}

}  // namespace

SFA3DPostProcess::SFA3DPostProcess(int input_w, int input_h,
                                   float score_threshold, float nms_threshold)
    : input_width_(input_w),
      input_height_(input_h),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold) {}

SFA3DPostProcess::SFA3DPostProcess()
    : input_width_(608), input_height_(608),
      score_threshold_(0.3f), nms_threshold_(0.2f) {}

std::vector<SFA3DResult> SFA3DPostProcess::postprocess(
    const dxrt::TensorPtrs& outputs) {
    if (outputs.empty()) return {};

    // Multi-head layout: hm(3), offset(2), direction(2), z(1), dim(3).
    // Decoded as maudzung/SFA3D does (evaluation_utils.py): x = x_min +
    // (row + offset[1]) * res -- the BEV's rows run forward DOWN the raster --,
    // y = y_min + (col + offset[0]) * res, z = z_coor + z_min (the box BOTTOM),
    // yaw = -atan2(direction[0], direction[1]) in the velodyne frame. Measured
    // 2026-11-10 on 100 KITTI frames (cars, median abs error): yaw 0.8 deg
    // (+atan2: 6.8), z 0.026 m as the bottom.
    if (outputs.size() >= 5) {
        const dxrt::TensorPtr* hm_tensor = nullptr;
        const dxrt::TensorPtr* offset_tensor = nullptr;
        const dxrt::TensorPtr* direction_tensor = nullptr;
        const dxrt::TensorPtr* z_tensor = nullptr;
        const dxrt::TensorPtr* dim_tensor = nullptr;

        if (isSfa3dFiveHeadLayout(outputs)) {
            hm_tensor = &outputs[0];
            offset_tensor = &outputs[1];
            direction_tensor = &outputs[2];
            z_tensor = &outputs[3];
            dim_tensor = &outputs[4];
        } else {
            std::unordered_map<int, std::vector<const dxrt::TensorPtr*>> by_channels;
            for (size_t i = 0; i < outputs.size() && i < 5; ++i) {
                const int ch = tensorChannels(outputs[i]);
                if (ch > 0) {
                    by_channels[ch].push_back(&outputs[i]);
                }
            }
            auto hm_it = by_channels.find(kNumClasses);
            auto two_it = by_channels.find(2);
            auto one_it = by_channels.find(1);
            if (hm_it != by_channels.end() && hm_it->second.size() >= 2 &&
                two_it != by_channels.end() && two_it->second.size() >= 2 &&
                one_it != by_channels.end() && !one_it->second.empty()) {
                hm_tensor = hm_it->second[0];
                dim_tensor = hm_it->second[1];
                offset_tensor = two_it->second[0];
                direction_tensor = two_it->second[1];
                z_tensor = one_it->second[0];
            }
        }

        if (hm_tensor != nullptr && offset_tensor != nullptr &&
            direction_tensor != nullptr && z_tensor != nullptr &&
            dim_tensor != nullptr) {
            const auto& hm_shape = (*hm_tensor)->shape();
            const int h = static_cast<int>(hm_shape[hm_shape.size() - 2]);
            const int w = static_cast<int>(hm_shape[hm_shape.size() - 1]);
            const int hw = h * w;

            std::vector<float> heatmap(static_cast<size_t>(kNumClasses * hw));
            for (int cls = 0; cls < kNumClasses; ++cls) {
                const float* src = tensorData(*hm_tensor) + cls * hw;
                for (int j = 0; j < hw; ++j) {
                    heatmap[static_cast<size_t>(cls * hw + j)] = sigmoid(src[j]);
                }
            }

            const float* offset = tensorData(*offset_tensor);
            const float* direction = tensorData(*direction_tensor);
            const float* z_coord = tensorData(*z_tensor);
            const float* dims = tensorData(*dim_tensor);

            const float x_res = (kXMax - kXMin) / static_cast<float>(h);
            const float y_res = (kYMax - kYMin) / static_cast<float>(w);
            const float scale_x = static_cast<float>(input_width_) / std::max(w, 1);
            const float scale_y = static_cast<float>(input_height_) / std::max(h, 1);

            auto peaks = topkHeatmap(heatmap.data(), kNumClasses, h, w,
                                     max_detections_, max_detections_);
            std::vector<SFA3DResult> results;
            for (const auto& peak : peaks) {
                if (peak.score < score_threshold_) continue;
                const int row = peak.row;
                const int col = peak.col;
                const int idx = row * w + col;

                SFA3DResult det;
                det.class_id = peak.cls;
                det.confidence = peak.score;
                det.bev_x = (col + offset[idx]) * scale_x;
                det.bev_y = (row + offset[hw + idx]) * scale_y;
                det.x3d = kXMin + (row + offset[hw + idx]) * x_res;
                det.y3d = kYMin + (col + offset[idx]) * y_res;
                det.z3d = z_coord[idx] + kVeloZMin;
                det.dim_h = dims[idx];
                det.dim_w = dims[hw + idx];
                det.dim_l = dims[2 * hw + idx];
                det.yaw = -std::atan2(direction[idx], direction[hw + idx]);
                det.bev_w = (det.dim_w / y_res) * scale_x;
                det.bev_h = (det.dim_l / x_res) * scale_y;
                results.push_back(det);
            }
            std::sort(results.begin(), results.end(),
                      [](const SFA3DResult& a, const SFA3DResult& b) {
                          return a.confidence > b.confidence;
                      });
            if (static_cast<int>(results.size()) > max_detections_) {
                results.resize(static_cast<size_t>(max_detections_));
            }
            return results;
        }
    }

    // Fused single tensor fallback
    const auto shape = outputs[0]->shape();
    if (shape.size() < 3) return {};

    int c = static_cast<int>(shape[shape.size() - 3]);
    int h = static_cast<int>(shape[shape.size() - 2]);
    int w = static_cast<int>(shape[shape.size() - 1]);
    const float* data = tensorData(outputs[0]);
    const int hw = h * w;

    if (c < kNumClasses + 8) {
        std::vector<float> heatmap(static_cast<size_t>(hw));
        for (int j = 0; j < hw; ++j) {
            heatmap[static_cast<size_t>(j)] = sigmoid(data[j]);
        }
        auto peaks = topkHeatmap(heatmap.data(), 1, h, w, max_detections_, max_detections_);
        std::vector<SFA3DResult> results;
        const float x_res = (kXMax - kXMin) / static_cast<float>(h);
        const float y_res = (kYMax - kYMin) / static_cast<float>(w);
        const float scale_x = static_cast<float>(input_width_) / std::max(w, 1);
        const float scale_y = static_cast<float>(input_height_) / std::max(h, 1);
        for (const auto& peak : peaks) {
            if (peak.score < score_threshold_) continue;
            SFA3DResult det;
            det.class_id = 0;
            det.confidence = peak.score;
            det.bev_x = static_cast<float>(peak.col) * scale_x;
            det.bev_y = static_cast<float>(peak.row) * scale_y;
            det.x3d = kXMin + peak.row * x_res;
            det.y3d = kYMin + peak.col * y_res;
            det.bev_w = 10.0f;
            det.bev_h = 10.0f;
            results.push_back(det);
        }
        return results;
    }

    std::vector<float> heatmap(static_cast<size_t>(kNumClasses * hw));
    for (int cls = 0; cls < kNumClasses; ++cls) {
        for (int j = 0; j < hw; ++j) {
            heatmap[static_cast<size_t>(cls * hw + j)] =
                sigmoid(data[cls * hw + j]);
        }
    }
    const float* offset = data + kNumClasses * hw;
    const float* z_coord = data + (kNumClasses + 2) * hw;
    const float* dims = data + (kNumClasses + 3) * hw;
    const float* yaw_im = data + (kNumClasses + 6) * hw;
    const float* yaw_re = data + (kNumClasses + 7) * hw;

    const float x_res = (kXMax - kXMin) / static_cast<float>(h);
    const float y_res = (kYMax - kYMin) / static_cast<float>(w);
    const float scale_x = static_cast<float>(input_width_) / std::max(w, 1);
    const float scale_y = static_cast<float>(input_height_) / std::max(h, 1);

    auto peaks = topkHeatmap(heatmap.data(), kNumClasses, h, w,
                             max_detections_, max_detections_);
    std::vector<SFA3DResult> results;
    for (const auto& peak : peaks) {
        if (peak.score < score_threshold_) continue;
        const int idx = peak.row * w + peak.col;

        SFA3DResult det;
        det.class_id = peak.cls;
        det.confidence = peak.score;
        det.bev_x = (peak.col + offset[idx]) * scale_x;
        det.bev_y = (peak.row + offset[hw + idx]) * scale_y;
        det.x3d = kXMin + (peak.row + offset[hw + idx]) * x_res;
        det.y3d = kYMin + (peak.col + offset[idx]) * y_res;
        det.z3d = z_coord[idx] + kVeloZMin;
        det.dim_h = dims[idx];
        det.dim_w = dims[hw + idx];
        det.dim_l = dims[2 * hw + idx];
        det.yaw = -std::atan2(yaw_im[idx], yaw_re[idx]);
        det.bev_w = (det.dim_w / y_res) * scale_x;
        det.bev_h = (det.dim_l / x_res) * scale_y;
        results.push_back(det);
    }
    std::sort(results.begin(), results.end(),
              [](const SFA3DResult& a, const SFA3DResult& b) {
                  return a.confidence > b.confidence;
              });
    if (static_cast<int>(results.size()) > max_detections_) {
        results.resize(static_cast<size_t>(max_detections_));
    }
    return results;
}
