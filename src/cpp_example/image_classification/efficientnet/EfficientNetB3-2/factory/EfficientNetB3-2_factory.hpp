/**
 * @file EfficientNetB3-2_factory.hpp
 * @brief EfficientNetb2 Abstract Factory implementation for classification
 */

#ifndef EFFICIENTNETB3_2_FACTORY_HPP
#define EFFICIENTNETB3_2_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/classification_postprocessor.hpp"
#include "common/visualizers/classification_visualizer.hpp"
#include "common/config/model_config.hpp"

// Includes required by bodies spliced in from sibling variants.
#include "common/processors/simple_resize_preprocessor.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_EfficientNetB3_2 {

class EfficientnetFactory : public IClassificationFactory {
public:

    EfficientnetFactory(int num_classes = 1000, int top_k = 5)
        : num_classes_(num_classes), top_k_(top_k) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // default: efficientnet-b2_288x288
        // Measured (the compile JSON gives no crop), ImageNetV2 10k top-1:
        // 68.62 stretch -> 70.27 with pillow bilinear resize (short side 343) + CenterCrop(300).
        auto pre = std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
        pre->setCenterCrop(343, 0, 0, cv::INTER_LINEAR, /*pillow=*/true);
        return pre;
    }

    PostprocessorPtr<ClassificationResult> createPostprocessor(
        int /*input_width*/, int /*input_height*/) override {
        return std::make_unique<EfficientNetPostprocessor>(num_classes_, top_k_);
    }

    VisualizerPtr<ClassificationResult> createVisualizer() override {
        return std::make_unique<ClassificationResultVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        num_classes_ = config.get<int>("num_classes", num_classes_);
        top_k_ = config.get<int>("top_k", top_k_);
    }

    std::string getModelName() const override {
        return "EfficientNetB3-2";
    }
    std::string getTaskType() const override { return "classification"; }

private:
    int num_classes_;
    int top_k_;
};

}  // namespace v_EfficientNetB3_2
}  // namespace dxapp

#endif  // EFFICIENTNETB3_2_FACTORY_HPP
