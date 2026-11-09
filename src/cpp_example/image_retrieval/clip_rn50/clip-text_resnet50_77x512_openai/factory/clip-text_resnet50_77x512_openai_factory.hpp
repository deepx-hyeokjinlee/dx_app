/**
 * @file clip-text_resnet50_77x512_openai_factory.hpp
 * @brief ClipFactory Abstract Factory implementation
 */

#ifndef CLIP_TEXT_RESNET50_77X512_OPENAI_FACTORY_HPP
#define CLIP_TEXT_RESNET50_77X512_OPENAI_FACTORY_HPP

#include "common/base/i_factory.hpp"
#include "common/processors/clip_text_postprocessor.hpp"
#include "common/visualizers/embedding_visualizer.hpp"
#include "common/config/model_config.hpp"

#include <memory>
#include <string>
#include <utility>

namespace dxapp {
namespace v_clip_text_resnet50_77x512_openai {

class ClipFactory : public IEmbeddingFactory {
public:

    ClipFactory() = default;

    /// Where clip_rn50_text.nxclip and clip_bpe_merges.txt live (ClipTextRunner
    /// sets it from --text-assets, the config, or the .dxnn directory).
    void setTextAssetsDir(const std::string& dir) {
        assets_dir_ = dir;
        assets_.reset();
    }

    // The model takes text: ClipTextPreprocessor::encode tokenizes and embeds it,
    // CLIPTextPostprocessor projects the hidden state at the end-of-text token.
    // (It used to be given an image resize and a generic embedding flatten.)
    PreprocessorPtr createPreprocessor(int /*input_width*/, int /*input_height*/) override {
        return std::make_unique<ClipTextPreprocessor>(assets());
    }

    PostprocessorPtr<EmbeddingResult> createPostprocessor(
        int /*input_width*/, int /*input_height*/) override {
        return std::make_unique<CLIPTextPostprocessor>(assets());
    }

    VisualizerPtr<EmbeddingResult> createVisualizer() override {
        return std::make_unique<EmbeddingVisualizer>();
    }

    std::string getModelName() const override {
        return "clip-text_resnet50_77x512_openai";
    }
    std::string getTaskType() const override { return "image_retrieval"; }

private:
    std::shared_ptr<const clip::ClipTextAssets> assets() {
        if (!assets_) assets_ = clip::ClipTextAssets::load(assets_dir_);
        return assets_;
    }

    std::string assets_dir_;
    std::shared_ptr<const clip::ClipTextAssets> assets_;
};

}  // namespace v_clip_text_resnet50_77x512_openai
}  // namespace dxapp

#endif  // CLIP_TEXT_RESNET50_77X512_OPENAI_FACTORY_HPP
