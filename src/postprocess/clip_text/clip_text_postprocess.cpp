#include "clip_text_postprocess.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::string format_shapes(const dxrt::TensorPtrs& outputs) {
    std::ostringstream s;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        s << (i ? ", " : "") << "(";
        const auto& shape = outputs[i]->shape();
        for (std::size_t d = 0; d < shape.size(); ++d) {
            s << (d ? ", " : "") << shape[d];
        }
        s << ")";
    }
    return s.str();
}

std::size_t element_count(const dxrt::TensorPtr& tensor) {
    std::size_t n = 1;
    for (int64_t d : tensor->shape()) n *= static_cast<std::size_t>(d);
    return n;
}

void l2_normalise(std::vector<float>& v) {
    double sq = 0.0;
    for (float x : v) sq += static_cast<double>(x) * x;
    const float norm = static_cast<float>(std::sqrt(sq));
    if (norm > 1e-12f) {
        for (float& x : v) x /= norm;
    }
}

}  // namespace

CLIPTextPostProcess::CLIPTextPostProcess(bool normalize, const std::string& weights_path)
    : normalize_(normalize) {
    if (weights_path.empty()) return;
    // NXCLIPT1: magic, u32 vocab/context/width/embed_dim, then float32 token
    // embedding, positional embedding, ln_final weight, ln_final bias, projection.
    std::ifstream in(weights_path, std::ios::binary);
    char magic[8];
    if (!in || !in.read(magic, 8) || std::memcmp(magic, "NXCLIPT1", 8) != 0) {
        throw std::runtime_error("[DXAPP] [ERROR] CLIPTextPostProcess - not an NXCLIPT1 weights file: " +
                                 weights_path);
    }
    uint32_t h[4];
    if (!in.read(reinterpret_cast<char*>(h), sizeof(h))) {
        throw std::runtime_error("[DXAPP] [ERROR] CLIPTextPostProcess - truncated header: " + weights_path);
    }
    context_ = static_cast<int>(h[1]);
    width_ = static_cast<int>(h[2]);
    embed_dim_ = static_cast<int>(h[3]);
    const std::size_t w = h[2];
    in.seekg(static_cast<std::streamoff>((static_cast<std::size_t>(h[0]) + h[1]) * w * sizeof(float)),
             std::ios::cur);
    const auto read = [&](std::vector<float>& v, std::size_t n) {
        v.resize(n);
        return static_cast<bool>(in.read(reinterpret_cast<char*>(v.data()),
                                         static_cast<std::streamsize>(n * sizeof(float))));
    };
    if (!read(ln_w_, w) || !read(ln_b_, w) || !read(projection_, w * h[3])) {
        throw std::runtime_error("[DXAPP] [ERROR] CLIPTextPostProcess - truncated weights: " + weights_path);
    }
}

std::vector<float> CLIPTextPostProcess::postprocess(const dxrt::TensorPtrs& outputs, int eot_position) {
    if (outputs.empty()) {
        throw std::runtime_error("[DXAPP] [ERROR] CLIPTextPostProcess - no output tensors");
    }
    const dxrt::TensorPtr& output = outputs[0];
    if (output->type() != dxrt::DataType::FLOAT) {
        throw std::runtime_error(
            "[DXAPP] [ERROR] CLIPTextPostProcess - expected a float32 feature tensor, got " +
            format_shapes(outputs));
    }
    const auto& shape = output->shape();
    const float* data = static_cast<const float*>(output->data());

    if (shape.size() != 3) {
        std::vector<float> embedding(data, data + element_count(output));
        if (normalize_) l2_normalise(embedding);
        return embedding;
    }
    if (width_ == 0) {
        throw std::runtime_error(
            "[DXAPP] [ERROR] CLIPTextPostProcess - a hidden-state output " + format_shapes(outputs) +
            " needs ln_final and text_projection: pass the NXCLIPT1 weights path");
    }
    if (static_cast<int>(shape[1]) != context_ || static_cast<int>(shape[2]) != width_) {
        throw std::runtime_error("[DXAPP] [ERROR] CLIPTextPostProcess - output " + format_shapes(outputs) +
                                 " does not match the weights [1, " + std::to_string(context_) + ", " +
                                 std::to_string(width_) + "]");
    }
    if (eot_position < 0 || eot_position >= context_) {
        throw std::runtime_error(
            "[DXAPP] [ERROR] CLIPTextPostProcess - eot_position " + std::to_string(eot_position) +
            " is not in [0, " + std::to_string(context_) + "): pass the end-of-text index of the token ids");
    }
    // ln_final, as ClipTextWeights::project in the C++ example tree.
    const float* x = data + static_cast<std::size_t>(eot_position) * width_;
    double mean = 0.0;
    for (int d = 0; d < width_; ++d) mean += x[d];
    mean /= width_;
    double var = 0.0;
    for (int d = 0; d < width_; ++d) var += (x[d] - mean) * (x[d] - mean);
    var /= width_;
    const double inv = 1.0 / std::sqrt(var + 1e-5);
    std::vector<float> embedding(static_cast<std::size_t>(embed_dim_), 0.0f);
    for (int d = 0; d < width_; ++d) {
        const float v = static_cast<float>((x[d] - mean) * inv) * ln_w_[static_cast<std::size_t>(d)] +
                        ln_b_[static_cast<std::size_t>(d)];
        const float* row = projection_.data() + static_cast<std::size_t>(d) * embed_dim_;
        for (int e = 0; e < embed_dim_; ++e) embedding[static_cast<std::size_t>(e)] += v * row[e];
    }
    l2_normalise(embedding);  // the joint embedding is compared by cosine, always
    return embedding;
}
