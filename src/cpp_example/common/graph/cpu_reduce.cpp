#include "common/graph/cpu_reduce.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace dxapp {
namespace graph {
namespace {

const double kDefaultScaleFactor = 1.0;
const double kSingularSy = 1e-6;
const int kDefaultClassIds[] = {24, 26, 28};
const char* const kDefaultClassNames[] = {
    "backpack", "handbag", "suitcase", "box", "package", "luggage"};

std::string LowerCopy(const std::string& text) {
    std::string lowered = text;
    for (std::size_t i = 0; i < lowered.size(); ++i) {
        lowered[i] = static_cast<char>(
            std::tolower(static_cast<unsigned char>(lowered[i])));
    }
    return lowered;
}

const StageData* FindShape(const std::vector<StageDataPtr>& inputs, Shape shape) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i] && inputs[i]->shape() == shape) return inputs[i].get();
    }
    return NULL;
}

std::vector<int> ClassIds(const StageParams& params) {
    std::map<std::string, std::vector<std::string> >::const_iterator listed =
        params.lists.find("class_ids");
    if (listed == params.lists.end()) {
        return std::vector<int>(
            kDefaultClassIds, kDefaultClassIds + sizeof(kDefaultClassIds) / sizeof(kDefaultClassIds[0]));
    }
    std::vector<int> ids;
    for (std::size_t i = 0; i < listed->second.size(); ++i) {
        ids.push_back(std::atoi(listed->second[i].c_str()));
    }
    return ids;
}

std::vector<std::string> ClassNames(const StageParams& params) {
    std::map<std::string, std::vector<std::string> >::const_iterator listed =
        params.lists.find("classes");
    if (listed == params.lists.end()) {
        const std::size_t count = sizeof(kDefaultClassNames) / sizeof(kDefaultClassNames[0]);
        std::vector<std::string> names;
        for (std::size_t i = 0; i < count; ++i) names.push_back(kDefaultClassNames[i]);
        return names;
    }
    std::vector<std::string> names;
    for (std::size_t i = 0; i < listed->second.size(); ++i) {
        names.push_back(LowerCopy(listed->second[i]));
    }
    return names;
}

bool IsPackage(const BoxItem& item, const std::vector<int>& ids,
               const std::vector<std::string>& names) {
    if (std::find(ids.begin(), ids.end(), item.class_id) != ids.end()) return true;
    const std::string name = LowerCopy(item.class_name);
    return std::find(names.begin(), names.end(), name) != names.end();
}

float RectIou(const cv::Rect2f& left, const cv::Rect2f& right) {
    const float x1 = std::max(left.x, right.x);
    const float y1 = std::max(left.y, right.y);
    const float x2 = std::min(left.x + left.width, right.x + right.width);
    const float y2 = std::min(left.y + left.height, right.y + right.height);
    const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    if (inter == 0.0f) return 0.0f;
    const float area_left = std::max(0.0f, left.width) * std::max(0.0f, left.height);
    const float area_right = std::max(0.0f, right.width) * std::max(0.0f, right.height);
    const float union_area = area_left + area_right - inter;
    return union_area > 0.0f ? inter / union_area : 0.0f;
}

bool MaskOn(const cv::Mat& plane, int y, int x) {
    if (plane.depth() == CV_8U) return plane.at<unsigned char>(y, x) > 0;
    if (plane.depth() == CV_32F) return plane.at<float>(y, x) > 0.0f;
    if (plane.depth() == CV_64F) return plane.at<double>(y, x) > 0.0;
    return false;
}

int CountOnMask(const cv::Mat& mask) {
    if (mask.empty()) return 0;
    cv::Mat plane = mask.channels() == 1 ? mask : cv::Mat();
    if (mask.channels() != 1) cv::extractChannel(mask, plane, 0);
    int count = 0;
    for (int y = 0; y < plane.rows; ++y) {
        for (int x = 0; x < plane.cols; ++x) {
            if (MaskOn(plane, y, x)) ++count;
        }
    }
    return count;
}

bool MedianUnderMask(const cv::Mat& depth, const cv::Mat& mask, double* median) {
    if (depth.empty() || mask.empty() || depth.rows != mask.rows || depth.cols != mask.cols) {
        return false;
    }
    if (depth.channels() != 1) return false;
    cv::Mat plane = mask.channels() == 1 ? mask : cv::Mat();
    if (mask.channels() != 1) cv::extractChannel(mask, plane, 0);
    if (plane.rows != depth.rows || plane.cols != depth.cols) return false;
    std::vector<double> samples;
    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            if (!MaskOn(plane, y, x)) continue;
            double value = 0.0;
            if (depth.depth() == CV_32F) {
                value = depth.at<float>(y, x);
            } else if (depth.depth() == CV_64F) {
                value = depth.at<double>(y, x);
            } else {
                return false;
            }
            if (std::isfinite(value)) samples.push_back(value);
        }
    }
    if (samples.empty()) return false;
    const std::size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(mid),
                     samples.end());
    if (samples.size() % 2 == 1) {
        *median = samples[mid];
        return true;
    }
    const double upper = samples[mid];
    std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(mid - 1),
                     samples.end());
    *median = (samples[mid - 1] + upper) / 2.0;
    return true;
}

const BoxItem* LargestBox(const std::vector<BoxItem>& items) {
    const BoxItem* best = NULL;
    float best_area = 0.0f;
    for (std::size_t i = 0; i < items.size(); ++i) {
        const float area = items[i].box.width * items[i].box.height;
        if (area > best_area) {
            best = &items[i];
            best_area = area;
        }
    }
    return best;
}

void AddNumber(RecordItem* item, const std::string& name, double value) {
    item->numbers.push_back(std::make_pair(name, value));
}

CpuReduction EmptyRecords() {
    CpuReduction out;
    out.result.data.reset(new RecordsData());
    return out;
}

CpuReduction Missing(const std::string& what) {
    CpuReduction out = EmptyRecords();
    out.error = "missing " + what + " input";
    return out;
}

CpuReduction ReduceHeadPose(const std::vector<StageDataPtr>& inputs, const cv::Size& frame_size) {
    const StageData* boxes_data = FindShape(inputs, Shape::kBoxes);
    if (boxes_data == NULL) return Missing("boxes");
    const BoxesData* boxes = static_cast<const BoxesData*>(boxes_data);
    const BoxItem* face = LargestBox(boxes->items);
    CpuReduction out = EmptyRecords();
    if (face == NULL || face->landmarks.size() < 5 || frame_size.width <= 0 ||
        frame_size.height <= 0) {
        return out;
    }
    // Face landmark order: left eye, right eye, nose, left mouth, right mouth.
    const float kModel[5][3] = {
        {-30.0f, -30.0f, -30.0f},
        {30.0f, -30.0f, -30.0f},
        {0.0f, 0.0f, 0.0f},
        {-25.0f, 30.0f, -20.0f},
        {25.0f, 30.0f, -20.0f},
    };
    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    for (int i = 0; i < 5; ++i) {
        object_points.push_back(cv::Point3d(kModel[i][0], kModel[i][1], kModel[i][2]));
        image_points.push_back(cv::Point2d(face->landmarks[i].x, face->landmarks[i].y));
    }
    const double focal = static_cast<double>(frame_size.width);
    const cv::Mat camera = (cv::Mat_<double>(3, 3) <<
        focal, 0.0, focal / 2.0,
        0.0, focal, static_cast<double>(frame_size.height) / 2.0,
        0.0, 0.0, 1.0);
    cv::Mat rotation;
    cv::Mat translation;
    const cv::Mat no_distortion = cv::Mat::zeros(4, 1, CV_64F);
#if CV_VERSION_MAJOR * 10000 + CV_VERSION_MINOR * 100 + CV_VERSION_REVISION >= 40501
    const bool solved = cv::solvePnP(
        object_points, image_points, camera, no_distortion,
        rotation, translation, false, cv::SOLVEPNP_SQPNP);
#else
    // SQPnP needs OpenCV 4.5.1 (Ubuntu 20.04 ships 4.2.0). ITERATIVE alone needs six
    // points for a non-planar model, so seed it with EPnP and refine.
    bool solved = cv::solvePnP(
        object_points, image_points, camera, no_distortion,
        rotation, translation, false, cv::SOLVEPNP_EPNP);
    if (solved) {
        solved = cv::solvePnP(
            object_points, image_points, camera, no_distortion,
            rotation, translation, true, cv::SOLVEPNP_ITERATIVE);
    }
#endif
    if (!solved) return out;
    cv::Mat matrix;
    cv::Rodrigues(rotation, matrix);
    const double sy = std::sqrt(
        matrix.at<double>(0, 0) * matrix.at<double>(0, 0) +
        matrix.at<double>(1, 0) * matrix.at<double>(1, 0));
    double pitch = 0.0;
    double yaw = 0.0;
    double roll = 0.0;
    if (sy < kSingularSy) {
        pitch = std::atan2(-matrix.at<double>(1, 2), matrix.at<double>(1, 1));
        yaw = std::atan2(-matrix.at<double>(2, 0), sy);
        roll = 0.0;
    } else {
        pitch = std::atan2(matrix.at<double>(2, 1), matrix.at<double>(2, 2));
        yaw = std::atan2(-matrix.at<double>(2, 0), sy);
        roll = std::atan2(matrix.at<double>(1, 0), matrix.at<double>(0, 0));
    }
    const double kRadToDeg = 180.0 / 3.14159265358979323846;
    RecordItem item;
    AddNumber(&item, "pitch", pitch * kRadToDeg);
    AddNumber(&item, "yaw", yaw * kRadToDeg);
    AddNumber(&item, "roll", roll * kRadToDeg);
    std::shared_ptr<RecordsData> records(new RecordsData());
    records->items.push_back(item);
    out.result.data = records;
    return out;
}

const BoxItem* BestOverlap(const BoxItem& detection, const std::vector<BoxItem>& segments,
                           const std::vector<int>& ids, const std::vector<std::string>& names) {
    const BoxItem* best = NULL;
    float best_iou = 0.0f;
    for (std::size_t i = 0; i < segments.size(); ++i) {
        if (!IsPackage(segments[i], ids, names)) continue;
        const float iou = RectIou(detection.box, segments[i].box);
        if (iou > best_iou) {
            best = &segments[i];
            best_iou = iou;
        }
    }
    return best;
}

CpuReduction ReduceVolume(const StageParams& params, const std::vector<StageDataPtr>& inputs) {
    const StageData* boxes_data = FindShape(inputs, Shape::kBoxes);
    const StageData* instances_data = FindShape(inputs, Shape::kInstances);
    const StageData* depth_data = FindShape(inputs, Shape::kDenseMap);
    if (boxes_data == NULL) return Missing("boxes");
    if (instances_data == NULL) return Missing("instances");
    if (depth_data == NULL) return Missing("densemap");
    const BoxesData* boxes = static_cast<const BoxesData*>(boxes_data);
    const BoxesData* instances = static_cast<const BoxesData*>(instances_data);
    const DenseMapData* depth = static_cast<const DenseMapData*>(depth_data);
    const std::vector<int> ids = ClassIds(params);
    const std::vector<std::string> names = ClassNames(params);
    std::map<std::string, double>::const_iterator scale_it = params.numeric.find("scale_factor");
    const double scale = scale_it == params.numeric.end() ? kDefaultScaleFactor : scale_it->second;

    std::shared_ptr<RecordsData> records(new RecordsData());
    for (std::size_t i = 0; i < boxes->items.size(); ++i) {
        const BoxItem& item = boxes->items[i];
        if (!IsPackage(item, ids, names)) continue;
        const BoxItem* segment = BestOverlap(item, instances->items, ids, names);
        const int area = segment == NULL ? 0 : CountOnMask(segment->mask);
        double median = 0.0;
        const bool has_median =
            segment != NULL && MedianUnderMask(depth->values, segment->mask, &median);
        RecordItem row;
        AddNumber(&row, "box_x", item.box.x);
        AddNumber(&row, "box_y", item.box.y);
        AddNumber(&row, "box_w", item.box.width);
        AddNumber(&row, "box_h", item.box.height);
        AddNumber(&row, "class_id", item.class_id);
        AddNumber(&row, "mask_area_px", area);
        if (has_median && area > 0) {
            AddNumber(&row, "median_depth", median);
            AddNumber(&row, "volume_proxy", area * median * scale);
        }
        row.text.push_back(std::make_pair(std::string("class_name"), item.class_name));
        records->items.push_back(row);
    }
    CpuReduction out;
    out.result.data = records;
    return out;
}

}  // namespace

CpuReduction ReduceCpu(const std::string& op, const StageParams& params,
                       const std::vector<StageDataPtr>& inputs,
                       const cv::Size& frame_size) {
    if (op == "headpose") return ReduceHeadPose(inputs, frame_size);
    if (op == "volume") return ReduceVolume(params, inputs);
    CpuReduction out = EmptyRecords();
    out.error = "unknown cpu op \"" + op + "\"";
    return out;
}

std::vector<std::string> CpuPackageClassNames() {
    return std::vector<std::string>(
        kDefaultClassNames,
        kDefaultClassNames + sizeof(kDefaultClassNames) / sizeof(kDefaultClassNames[0]));
}

std::vector<std::string> CpuPackageClassIds() {
    std::vector<std::string> ids;
    for (std::size_t i = 0; i < sizeof(kDefaultClassIds) / sizeof(kDefaultClassIds[0]); ++i) {
        ids.push_back(std::to_string(kDefaultClassIds[i]));
    }
    return ids;
}

}  // namespace graph
}  // namespace dxapp
