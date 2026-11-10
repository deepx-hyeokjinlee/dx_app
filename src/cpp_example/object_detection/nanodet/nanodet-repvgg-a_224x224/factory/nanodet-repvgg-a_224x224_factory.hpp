/**
 * @file nanodet-repvgg-a_224x224_factory.hpp
 * @brief NanoDetplusm_15 Abstract Factory implementation
 * 
 * Uses v3-native NanoDet postprocessor with DFL decoding.
 */

#ifndef NANODET_REPVGG_A_224X224_FACTORY_HPP
#define NANODET_REPVGG_A_224X224_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/nanodet_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

// Includes required by bodies spliced in from sibling variants.
#include "common/processors/yolo_detection_postprocessor.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_nanodet_repvgg_a_224x224 {

class NanodetFactory : public IDetectionFactory {
public:

    NanodetFactory(float score_threshold = 0.3f,
                   float nms_threshold = 0.45f,
                   int reg_max = 7)
        : score_threshold_(score_threshold),
          nms_threshold_(nms_threshold),
          reg_max_(reg_max) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        return std::make_unique<DetectionPreprocessor>(input_width, input_height, /*color_conversion=*/-1);
    }

    PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) override {
        // default: nanodet-plus-1.5x_224x224
        (void)is_ort_configured;
        return std::make_unique<NanoDetPostprocessor>(
            input_width, input_height,
            score_threshold_, nms_threshold_, num_classes_, reg_max_,
            false, class_names_, center_offset_
        );
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        obj_threshold_ = config.get<float>("obj_threshold", obj_threshold_);
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
        num_classes_ = config.get<int>("num_classes", num_classes_);
        class_names_ = config.get_string_list("class_names");
        reg_max_ = config.get<int>("reg_max", reg_max_);
        center_offset_ = config.get<float>("center_offset", center_offset_);
    }

    std::string getModelName() const override {
        return "nanodet-repvgg-a_224x224";
    }
    std::string getTaskType() const override { return "object_detection"; }

private:
    // YOLOXPostprocessor reads an objectness threshold; it was declared here and
    // never set (undefined value passed to the postprocessor). 0.25 is the
    // YOLOXPostprocessor default.
    float obj_threshold_{0.25f};
    float score_threshold_;
    float nms_threshold_;
    int num_classes_{80};
    int reg_max_;
    float center_offset_ = 0.5f;  // NanoDet-Plus: 0.0
    std::vector<std::string> class_names_;
};

}  // namespace v_nanodet_repvgg_a_224x224
}  // namespace dxapp

#endif  // NANODET_REPVGG_A_224X224_FACTORY_HPP
