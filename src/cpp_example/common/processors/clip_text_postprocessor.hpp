/**
 * @file clip_text_postprocessor.hpp
 * @brief CLIP text encoder: text in (ClipTextPreprocessor), joint embedding out
 *        (CLIPTextPostprocessor).
 *
 * clip-text_resnet50_77x512_openai.dxnn is the text transformer only. Its input is
 * the [1,77,512] token+position embedding and its output the [1,77,512] hidden
 * states, so both ends need the CPU-side weights (clip_text_encoder.hpp):
 *
 *   ClipTextPreprocessor::encode   text -> 77 token ids -> [77,512] float input,
 *                                  and the end-of-text position in the context
 *   CLIPTextPostprocessor::process ln_final of the hidden state at that position,
 *                                  @ text_projection, L2-normalised -> 1024-d
 *
 * The end-of-text position is the arg-max of the token ids (CLIP's
 * text.argmax(dim=-1)). It cannot be recovered from the hidden states: every
 * one of the 77 rows is non-zero, so "the last non-zero row" is always row 76.
 * A [1,D] output (a graph that already ends in the projection) is L2-normalised
 * and passed through.
 */

#ifndef CLIP_TEXT_POSTPROCESSOR_HPP
#define CLIP_TEXT_POSTPROCESSOR_HPP

#include "common/base/i_processor.hpp"
#include "common/processors/clip_text_encoder.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dxapp {

class ClipTextPreprocessor : public IPreprocessor {
public:
    explicit ClipTextPreprocessor(std::shared_ptr<const clip::ClipTextAssets> assets)
        : assets_(std::move(assets)) {
        if (!assets_) throw std::invalid_argument("[DXAPP] [ERROR] ClipTextPreprocessor needs the CLIP text assets");
    }

    /// The image entry point of IPreprocessor. This model takes text.
    void process(const cv::Mat&, cv::Mat&, PreprocessContext&) override {
        throw std::logic_error(
            "[DXAPP] [ERROR] clip-text takes text, not an image: call encode() "
            "(the example's ClipTextRunner does, from -t / --text-file)");
    }

    /// Tokenize `text` and write the [context, width] float32 model input.
    /// `truncated` (optional) says whether the prompt was cut at 77 tokens.
    void encode(const std::string& text, cv::Mat& output, PreprocessContext& ctx,
                bool* truncated = nullptr) const {
        const auto ids = assets_->tokenizer.tokenize(text, truncated);
        const auto embedded = assets_->weights.embed(ids);
        output.create(assets_->weights.context(), assets_->weights.width(), CV_32F);
        std::memcpy(output.data, embedded.data(), embedded.size() * sizeof(float));
        ctx = PreprocessContext{};
        ctx.input_width = assets_->weights.width();
        ctx.input_height = assets_->weights.context();
        ctx.eot_position = clip::eot_position(ids);
        token_count_ = 0;
        for (int id : ids) token_count_ += (id != 0);
    }

    /// Non-padding tokens of the last encode(), SOT and EOT included.
    int lastTokenCount() const { return token_count_; }

    int getInputWidth() const override { return assets_->weights.width(); }
    int getInputHeight() const override { return assets_->weights.context(); }
    int getColorConversion() const override { return -1; }

private:
    std::shared_ptr<const clip::ClipTextAssets> assets_;
    mutable int token_count_{0};
};

class CLIPTextPostprocessor : public IPostprocessor<EmbeddingResult> {
public:
    explicit CLIPTextPostprocessor(std::shared_ptr<const clip::ClipTextAssets> assets)
        : assets_(std::move(assets)) {
        if (!assets_) throw std::invalid_argument("[DXAPP] [ERROR] CLIPTextPostprocessor needs the CLIP text assets");
    }

    std::vector<EmbeddingResult> process(const dxrt::TensorPtrs& outputs,
                                         const PreprocessContext& ctx) override {
        if (outputs.empty()) {
            throw std::runtime_error("[DXAPP] [ERROR] CLIPTextPostprocessor: no output tensors");
        }
        const auto& output = outputs[0];
        const auto shape = output->shape();
        const auto* data = static_cast<const float*>(output->data());
        if (!data || output->type() != dxrt::DataType::FLOAT) {
            throw std::runtime_error("[DXAPP] [ERROR] CLIPTextPostprocessor: expected a float32 output");
        }
        const auto& w = assets_->weights;
        EmbeddingResult result;
        if (shape.size() == 3) {
            if (static_cast<int>(shape[1]) != w.context() || static_cast<int>(shape[2]) != w.width()) {
                throw std::runtime_error(
                    "[DXAPP] [ERROR] CLIPTextPostprocessor: hidden states [1," +
                    std::to_string(shape[1]) + "," + std::to_string(shape[2]) +
                    "] do not match the text weights [1," + std::to_string(w.context()) +
                    "," + std::to_string(w.width()) + "]");
            }
            if (ctx.eot_position < 0 || ctx.eot_position >= w.context()) {
                throw std::runtime_error(
                    "[DXAPP] [ERROR] CLIPTextPostprocessor: no end-of-text position in the "
                    "context (the input must come from ClipTextPreprocessor::encode)");
            }
            result.embedding = w.project(data, ctx.eot_position);
        } else {
            std::size_t n = 1;
            for (auto d : shape) n *= static_cast<std::size_t>(d);
            result.embedding.assign(data, data + n);
            double sq = 0.0;
            for (float v : result.embedding) sq += static_cast<double>(v) * v;
            const float norm = static_cast<float>(std::sqrt(sq));
            if (norm > 0.0f) for (float& v : result.embedding) v /= norm;
        }
        result.dimension = static_cast<int>(result.embedding.size());
        return {result};
    }

    std::string getModelName() const override { return "CLIPText"; }

private:
    std::shared_ptr<const clip::ClipTextAssets> assets_;
};

}  // namespace dxapp

#endif  // CLIP_TEXT_POSTPROCESSOR_HPP
