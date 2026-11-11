/**
 * @file efficientformerv2-s1_224x224_factory.hpp
 * @brief Efficientformer_l3 Abstract Factory implementation for classification
 */

#ifndef EFFICIENTFORMERV2_S1_224X224_FACTORY_HPP
#define EFFICIENTFORMERV2_S1_224X224_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/classification_postprocessor.hpp"
#include "common/visualizers/classification_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_efficientformerv2_s1_224x224 {

class EfficientformerFactory : public IClassificationFactory {
public:

    EfficientformerFactory(int num_classes = 1000, int top_k = 5)
        : num_classes_(num_classes), top_k_(top_k) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // Measured (the compile JSON gives no crop), ImageNetV2 10k top-1:
        // 62.10 stretch -> 65.41 with pillow bicubic resize (short side 249) + CenterCrop(224).
        auto pre = std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
        pre->setCenterCrop(249, 0, 0, cv::INTER_CUBIC, /*pillow=*/true);
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
        return "efficientformerv2-s1_224x224";
    }
    std::string getTaskType() const override { return "classification"; }

private:
    int num_classes_;
    int top_k_;
};

}  // namespace v_efficientformerv2_s1_224x224
}  // namespace dxapp

#endif  // EFFICIENTFORMERV2_S1_224X224_FACTORY_HPP
