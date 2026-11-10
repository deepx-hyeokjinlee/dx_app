/**
 * @file inceptionv1_224x224_factory.hpp
 * @brief Inceptionv1 Abstract Factory implementation for classification
 */

#ifndef INCEPTIONV1_224X224_FACTORY_HPP
#define INCEPTIONV1_224X224_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/classification_postprocessor.hpp"
#include "common/visualizers/classification_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_inceptionv1_224x224 {

class InceptionFactory : public IClassificationFactory {
public:

    InceptionFactory(int num_classes = 1000, int top_k = 5)
        : num_classes_(num_classes), top_k_(top_k) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // The compile JSON's evaluation transform (q-lite-json/2_5_0/inceptionv1_224x224.json):
        // torchvision Resize(256, bilinear) + CenterCrop(224).
        auto pre = std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
        pre->setCenterCrop(256, 0, 0, cv::INTER_LINEAR, /*pillow=*/true);
        return pre;
    }

    PostprocessorPtr<ClassificationResult> createPostprocessor(
        int /*input_width*/, int /*input_height*/) override {
        return std::make_unique<EfficientNetPostprocessor>(num_classes_, top_k_, background_class_);
    }

    VisualizerPtr<ClassificationResult> createVisualizer() override {
        return std::make_unique<ClassificationResultVisualizer>();
    }

    void loadConfig(const dxapp::ModelConfig& config) override {
        num_classes_ = config.get<int>("num_classes", num_classes_);
        top_k_ = config.get<int>("top_k", top_k_);
        background_class_ = config.get<bool>("background_class", background_class_);
    }

    std::string getModelName() const override {
        return "inceptionv1_224x224";
    }
    std::string getTaskType() const override { return "classification"; }

private:
    int num_classes_;
    int top_k_;
    // A TF-slim export: 1001 outputs, row 0 is the background class.
    bool background_class_{true};
};

}  // namespace v_inceptionv1_224x224
}  // namespace dxapp

#endif  // INCEPTIONV1_224X224_FACTORY_HPP
