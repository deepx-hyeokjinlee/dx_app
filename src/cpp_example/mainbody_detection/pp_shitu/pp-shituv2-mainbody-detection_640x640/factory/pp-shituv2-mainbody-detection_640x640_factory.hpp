/**
 * @file pp-shituv2-mainbody-detection_640x640_factory.hpp
 * @brief PpShitu Abstract Factory implementation
 * 
 * Uses v3-native NanoDet postprocessor with DFL decoding.
 */

#ifndef PP_SHITUV2_MAINBODY_DETECTION_640X640_FACTORY_HPP
#define PP_SHITUV2_MAINBODY_DETECTION_640X640_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/nanodet_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

// Includes required by bodies spliced in from sibling variants.
#include "common/processors/yolo_detection_postprocessor.hpp"

#include <string>
#include <utility>

// Carried over from object_detection/nanodet: this family was added
// after the restructure, so it has no original factory of its own,
// and that family's postprocessing is what pp_shitu needs. Only the
// class name, include guard and identity differ.

namespace dxapp {
namespace v_pp_shituv2_mainbody_detection_640x640 {

class PpShituFactory : public IDetectionFactory {
public:

    PpShituFactory(float score_threshold = 0.35f,
                   float nms_threshold = 0.6f,
                   int reg_max = 7)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold),
          reg_max_(reg_max) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        // default: nanodet-plus-1.5x_224x224
        (void)is_ort_configured;
        return std::make_unique<NanoDetPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_, num_classes_, reg_max_,
            false, class_names_
        );
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
        num_classes_ = config.get<int>("num_classes", num_classes_);
        class_names_ = config.get_string_list("class_names");
        reg_max_ = config.get<int>("reg_max", reg_max_);
    }

    std::string getModelName() const override {
        return "pp-shituv2-mainbody-detection_640x640";
    }
    std::string getTaskType() const override { return "mainbody_detection"; }

private:
    float obj_threshold_;  // from a sibling variant in this family
    float score_threshold_;
    float nms_threshold_;
    int num_classes_{80};
    int reg_max_;
    std::vector<std::string> class_names_;
};

}  // namespace v_pp_shituv2_mainbody_detection_640x640
}  // namespace dxapp

#endif  // PP_SHITUV2_MAINBODY_DETECTION_640X640_FACTORY_HPP
