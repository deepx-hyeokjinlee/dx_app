/**
 * @file beit-b-p16_384x384_factory.hpp
 * @brief BeitFactory Abstract Factory implementation
 */

#ifndef BEIT_B_P16_384X384_FACTORY_HPP
#define BEIT_B_P16_384X384_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/simple_resize_preprocessor.hpp"
#include "common/processors/classification_postprocessor.hpp"
#include "common/visualizers/classification_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <string>
#include <utility>

namespace dxapp {
namespace v_beit_b_p16_384x384 {

class BeitFactory : public IClassificationFactory {
public:

    BeitFactory(int num_classes = 1000, int top_k = 5)
        : num_classes_(num_classes), top_k_(top_k) {}

    PreprocessorPtr createPreprocessor(int input_width, int input_height) override {
        // Measured (the compile JSON gives no crop), ImageNetV2 10k top-1:
        // 76.26 stretch -> 76.63 with opencv bilinear resize (short side 439) + CenterCrop(384).
        auto pre = std::make_unique<SimpleResizePreprocessor>(input_width, input_height);
        pre->setCenterCrop(439, 0, 0, cv::INTER_LINEAR, /*pillow=*/false);
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
        return "beit-b-p16_384x384";
    }
    std::string getTaskType() const override { return "classification"; }

private:
    int num_classes_;
    int top_k_;
};

}  // namespace v_beit_b_p16_384x384
}  // namespace dxapp

#endif  // BEIT_B_P16_384X384_FACTORY_HPP
