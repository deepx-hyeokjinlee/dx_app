/**
 * @file yolov3-gluon_608x608_factory.hpp
 * @brief YOLOv3_gluon_416 Abstract Factory implementation
 * 
 */

#ifndef YOLOV3_GLUON_608X608_FACTORY_HPP
#define YOLOV3_GLUON_608X608_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/yolo_detection_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_yolov3_gluon_608x608 {

class Yolov3Factory : public IDetectionFactory {
public:

    Yolov3Factory(float obj_threshold = 0.25f,
                  float score_threshold = 0.4f,
                  float nms_threshold = 0.45f)
        : obj_threshold_(obj_threshold),
          score_threshold_(score_threshold),
          nms_threshold_(nms_threshold) {}

    // GluonCV's YOLOv3: float32 RGB / 255 (the runner converts a float input),
    // the image at the top-left of the letterbox, and the original YOLOv3 box
    // coding. COCO val2017 200 images: 44.40 mAP; centred 43.83; the
    // Ultralytics coding 32.97.
    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height, cv::COLOR_BGR2RGB,
                                                       114, /*pad_edge=*/true);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        auto post = std::make_unique<YOLOv5Postprocessor>(
            input_width, input_height,
            obj_threshold_, score_threshold_, nms_threshold_,
            is_ort_configured,
            num_classes_,
            class_names_
        );
        post->setYolov3BoxCoding(true);
        return post;
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
        return "yolov3-gluon_608x608";
    }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float obj_threshold_;
    float score_threshold_;
    float nms_threshold_;
    int num_classes_{80};
    std::vector<std::string> class_names_;
};

}  // namespace v_yolov3_gluon_608x608
}  // namespace dxapp

#endif  // YOLOV3_GLUON_608X608_FACTORY_HPP
