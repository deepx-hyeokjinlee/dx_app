#ifndef CLIP_TEXT_POSTPROCESS_H
#define CLIP_TEXT_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief CLIP text encoder post-processing: the joint-space text embedding
 *
 * clip-text_resnet50_77x512_openai is the text transformer only: it emits the
 * [1, 77, 512] hidden states. The embedding is ln_final (eps 1e-5) of the hidden
 * state at the end-of-text position, @ text_projection, L2-normalised -- 1024-d,
 * the space of clip-img_resnet50_224x224_openai. ln_final and text_projection
 * come from the NXCLIPT1 weights file (scripts/export_clip_text_assets.py); the
 * end-of-text position is the arg-max of the token ids, which the caller's
 * tokenizer knows (it cannot be recovered from the hidden states: every row is
 * non-zero).
 *
 * A [1, D] output (a graph that already ends in the projection) is
 * L2-normalised unless normalize is false; it needs neither the weights nor a
 * position. Same decode as the Python CLIPTextPostprocessor.
 */
class CLIPTextPostProcess {
   private:
    bool normalize_{true};
    int width_{0};
    int context_{0};
    int embed_dim_{0};
    std::vector<float> ln_w_, ln_b_, projection_;

   public:
    /// weights_path: the NXCLIPT1 file; required for a [1, 77, 512] output.
    explicit CLIPTextPostProcess(bool normalize = true, const std::string& weights_path = "");
    ~CLIPTextPostProcess() = default;

    /// eot_position: index of the end-of-text token in the input ids.
    std::vector<float> postprocess(const dxrt::TensorPtrs& outputs, int eot_position = -1);

    bool get_normalize() const { return normalize_; }
    int get_embed_dim() const { return embed_dim_; }
};

#endif  // CLIP_TEXT_POSTPROCESS_H
