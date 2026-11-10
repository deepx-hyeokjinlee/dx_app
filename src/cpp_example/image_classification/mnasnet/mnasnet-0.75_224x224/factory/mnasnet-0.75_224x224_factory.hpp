/**
 * @file mnasnet-0.75_224x224_factory.hpp
 * @brief Mnasnet0_5 Abstract Factory implementation for classification
 */

#ifndef MNASNET_0_75_224X224_FACTORY_HPP
#define MNASNET_0_75_224X224_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/letterbox_preprocessor.hpp"
#include "common/processors/classification_postprocessor.hpp"
#include "common/visualizers/classification_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_mnasnet_0_75_224x224 {

class MnasnetFactory : public IClassificationFactory {
public:

    MnasnetFactory(int num_classes = 1000, int top_k = 5)
        : num_classes_(num_classes), top_k_(top_k) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // The compile JSON's evaluation transform (q-lite-json/2_5_0/mnasnet-0.75_224x224.json):
        // torchvision Resize(232, bilinear) + CenterCrop(224).
        auto pre = std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
        pre->setCenterCrop(232, 0, 0, cv::INTER_LINEAR, /*pillow=*/true);
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
        return "mnasnet-0.75_224x224";
    }
    std::string getTaskType() const override { return "classification"; }

private:
    int num_classes_;
    int top_k_;
};

}  // namespace v_mnasnet_0_75_224x224
}  // namespace dxapp

#endif  // MNASNET_0_75_224X224_FACTORY_HPP
