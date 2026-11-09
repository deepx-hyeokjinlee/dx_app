/**
 * @file yolov7-w6_1280x1280_nodecode_factory.hpp
 * @brief YOLOv7d6 Abstract Factory implementation
 */

#ifndef YOLOV7_W6_1280X1280_NODECODE_FACTORY_HPP
#define YOLOV7_W6_1280X1280_NODECODE_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/yolo_detection_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_yolov7_w6_1280x1280_nodecode {

class Yolov7Factory : public IDetectionFactory {
public:

    Yolov7Factory(float obj_threshold = 0.25f,
                  float score_threshold = 0.4f,
                  float nms_threshold = 0.45f)
        : obj_threshold_(obj_threshold),
          score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        // Four detection heads (P3-P6, strides 8/16/32/64) and no in-graph decoder:
        // the anchor maps need the W6 anchors, as the Python example's
        // YOLOV7_W6_ANCHORS. The YOLOv7 preset carries only the three P5 levels.
        auto post = std::make_unique<YOLOv7Postprocessor>(
            input_width, input_height,
            obj_threshold_, score_threshold_, nms_threshold_,
            is_ort_configured,
            num_classes_,
            class_names_
        );
        post->setAnchors({
            {8,  {{19, 27}, {44, 40}, {38, 94}}},
            {16, {{96, 68}, {86, 152}, {180, 137}}},
            {32, {{140, 301}, {303, 264}, {238, 542}}},
            {64, {{436, 615}, {739, 380}, {925, 792}}},
        });
        return PostprocessorPtr<DetectionResult>(std::move(post));
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        obj_threshold_ = config.get<float>("obj_threshold", obj_threshold_);
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
        class_names_ = config.get_string_list("class_names");
        num_classes_ = config.get<int>("num_classes", num_classes_);
    }

    std::string getModelName() const override {
        return "yolov7-w6_1280x1280_nodecode";
    }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float obj_threshold_;
    float score_threshold_;
    float nms_threshold_;
    int num_classes_{80};
    std::vector<std::string> class_names_;
};

}  // namespace v_yolov7_w6_1280x1280_nodecode
}  // namespace dxapp

#endif  // YOLOV7_W6_1280X1280_NODECODE_FACTORY_HPP
