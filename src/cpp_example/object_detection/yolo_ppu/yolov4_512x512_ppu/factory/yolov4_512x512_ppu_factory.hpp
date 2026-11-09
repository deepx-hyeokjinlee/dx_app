/**
 * @file yolov4_512x512_ppu_factory.hpp
 * @brief YOLOv11N-PPU Abstract Factory implementation
 *
 * YOLOv11N PPU uses anchor-free decoding (same as YOLOv8 PPU).
 */

#ifndef YOLOV4_512X512_PPU_FACTORY_HPP
#define YOLOV4_512X512_PPU_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/ppu_postprocessor.hpp"
#include "common/visualizers/detection_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_yolov4_512x512_ppu {

class YoloPpuFactory : public IDetectionFactory {
public:

    YoloPpuFactory(float obj_threshold = 0.25f,
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
        auto post = std::make_unique<YOLOv5PPUPostprocessor>(
            input_width, input_height,
            obj_threshold_, score_threshold_, nms_threshold_,
            is_ort_configured,
            class_names_
        );
        // YOLOv4's anchors: the PPU output carries only the anchor index. With
        // YOLOv5's defaults the boxes were sized wrong -- against yolov5-x on 163
        // COCO images, mean best IoU 0.602 -> 0.842, recall@0.75 0.019 -> 0.842.
        post->setAnchors({{8, {{12, 16}, {19, 36}, {40, 28}}},
                          {16, {{36, 75}, {76, 55}, {72, 146}}},
                          {32, {{142, 110}, {192, 243}, {459, 401}}}});
        return PostprocessorPtr<DetectionResult>(std::move(post));
    }

    VisualizerPtr<DetectionResult> createVisualizer() override {
        return std::make_unique<DetectionVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        obj_threshold_ = config.get<float>("obj_threshold", obj_threshold_);
        score_threshold_ = config.get<float>("score_threshold", score_threshold_);
        class_names_ = config.get_string_list("class_names");
        nms_threshold_ = config.get<float>("nms_threshold", nms_threshold_);
    }

    std::string getModelName() const override {
        return "yolov4_512x512_ppu";
    }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float obj_threshold_;
    float score_threshold_;
    float nms_threshold_;
    std::vector<std::string> class_names_;
};

}  // namespace v_yolov4_512x512_ppu
}  // namespace dxapp

#endif  // YOLOV4_512X512_PPU_FACTORY_HPP
