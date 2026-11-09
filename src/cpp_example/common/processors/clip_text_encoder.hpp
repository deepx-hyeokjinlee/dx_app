/**
 * @file clip_text_encoder.hpp
 * @brief The CPU half of the CLIP RN50 (OpenAI) text tower around the NPU transformer.
 *
 * clip-text_resnet50_77x512_openai.dxnn takes the [1,77,512] token+position embedding
 * and returns the transformer's [1,77,512] hidden states; the tokenizer, the embedding
 * lookup, ln_final and text_projection are outside the compiled graph. This header is
 * that outside:
 *
 *   text -> ClipTokenizer (OpenAI simple_tokenizer.py, MIT) -> 77 ids
 *        -> ClipTextWeights::embed   (token_embedding[id] + positional)
 *        -> NPU transformer
 *        -> ClipTextWeights::project (ln_final at the EOT position, @ text_projection,
 *           L2-normalised) -> 1024-d, the space clip-img_resnet50_224x224_openai lives in
 *
 * The weights are OpenAI's RN50 checkpoint (MIT). scripts/export_clip_text_assets.py
 * checks its sha256 and writes the two files read here, clip_rn50_text.nxclip and
 * clip_bpe_merges.txt, next to the .dxnn. python_example/common/processors/
 * clip_text_encoder.py is the same algorithm line for line.
 *
 * Tokenizer differences from the Python original, both disclosed: ftfy's mojibake
 * repair is not done, and the regex classes \p{L} / \p{N} are approximated -- ASCII
 * letters and every non-ASCII code point that is not whitespace count as letters,
 * ASCII digits as numbers. Plain-English prompts tokenize identically.
 */

#ifndef DXAPP_CLIP_TEXT_ENCODER_HPP
#define DXAPP_CLIP_TEXT_ENCODER_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dxapp {
namespace clip {

class ClipTokenizer
{
public:
  static constexpr int kContext = 77;

  /// `merges_path`: the BPE merges, one "a b" pair per line, in rank order
  /// (bpe_simple_vocab_16e6.txt decompressed, header line removed).
  bool load(const std::string & merges_path, std::string & error)
  {
    std::ifstream in(merges_path);
    if (!in) { error = "cannot read BPE merges: " + merges_path; return false; }
    std::vector<std::pair<std::string, std::string>> merges;
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') { line.pop_back(); }
      if (line.empty()) { continue; }
      const auto space = line.find(' ');
      if (space == std::string::npos) { continue; }
      merges.emplace_back(line.substr(0, space), line.substr(space + 1));
    }
    // simple_tokenizer.py keeps merges[1 : 49152 - 256 - 2 + 1] of the file,
    // whose first line is a version header; the exporter drops that header.
    const std::size_t wanted = 49152 - 256 - 2;
    if (merges.size() < wanted) {
      error = "BPE merges file has " + std::to_string(merges.size()) + " merges, expected " +
              std::to_string(wanted);
      return false;
    }
    merges.resize(wanted);

    std::vector<std::string> in_order;
    bytes_to_unicode(byte_encoder_, in_order);
    // vocab = list(bytes_to_unicode().values()): dict insertion order.
    std::vector<std::string> vocab;
    for (const auto & s : in_order) { vocab.push_back(s); }
    for (const auto & s : in_order) { vocab.push_back(s + "</w>"); }
    for (const auto & m : merges) { vocab.push_back(m.first + m.second); }
    vocab.push_back("<|startoftext|>");
    vocab.push_back("<|endoftext|>");
    encoder_.clear();
    for (std::size_t i = 0; i < vocab.size(); ++i) { encoder_[vocab[i]] = static_cast<int>(i); }
    ranks_.clear();
    for (std::size_t i = 0; i < merges.size(); ++i) {
      ranks_[merges[i].first + '\x01' + merges[i].second] = static_cast<int>(i);
    }
    sot_ = encoder_.at("<|startoftext|>");
    eot_ = encoder_.at("<|endoftext|>");
    return true;
  }

  int sot() const { return sot_; }
  int eot() const { return eot_; }
  std::size_t vocab_size() const { return encoder_.size(); }

  /// BPE ids of `text`, without SOT/EOT.
  std::vector<int> encode(const std::string & text) const
  {
    std::vector<int> ids;
    for (const std::string & piece : pre_tokenize(clean(text))) {
      std::string mapped;
      for (unsigned char b : piece) { mapped += byte_encoder_[b]; }
      for (const std::string & sub : bpe(mapped)) {
        const auto it = encoder_.find(sub);
        if (it != encoder_.end()) { ids.push_back(it->second); }
      }
    }
    return ids;
  }

  /// clip.tokenize(text, truncate=True): [SOT, ids..., EOT] zero-padded to
  /// 77; too long is cut and the last slot made EOT. `truncated` says so.
  std::vector<int> tokenize(const std::string & text, bool * truncated = nullptr) const
  {
    std::vector<int> ids{sot_};
    const std::vector<int> body = encode(text);
    ids.insert(ids.end(), body.begin(), body.end());
    ids.push_back(eot_);
    const bool cut = ids.size() > static_cast<std::size_t>(kContext);
    if (cut) { ids.resize(kContext); ids.back() = eot_; }
    if (truncated != nullptr) { *truncated = cut; }
    ids.resize(kContext, 0);
    return ids;
  }

private:
  /// GPT-2's reversible byte -> printable-unicode map, as UTF-8 strings:
  /// `by_byte[b]` for encoding, `in_order` = dict(zip(bs, cs)).values().
  static void bytes_to_unicode(std::vector<std::string> & by_byte, std::vector<std::string> & in_order)
  {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) { bs.push_back(b); }
    for (int b = 0xA1; b <= 0xAC; ++b) { bs.push_back(b); }
    for (int b = 0xAE; b <= 0xFF; ++b) { bs.push_back(b); }
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
        bs.push_back(b);
        cs.push_back(256 + n++);
      }
    }
    by_byte.assign(256, std::string());
    in_order.assign(bs.size(), std::string());
    for (std::size_t i = 0; i < bs.size(); ++i) {
      in_order[i] = utf8(cs[i]);
      by_byte[static_cast<std::size_t>(bs[i])] = in_order[i];
    }
  }

  static std::string utf8(int cp)
  {
    std::string out;
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
  }

  /// basic_clean + whitespace_clean + lower(): HTML entities (&amp; &lt; &gt;
  /// &quot; &#39;) unescaped, whitespace runs to one space, trimmed, ASCII
  /// lower-cased. (ftfy.fix_text is not reproduced.)
  static std::string clean(const std::string & text)
  {
    static const std::pair<const char *, const char *> kEntities[] = {
      {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&#39;", "'"}};
    std::string s = text;
    for (int pass = 0; pass < 2; ++pass) {  // html.unescape twice
      for (const auto & e : kEntities) {
        std::string::size_type at = 0;
        while ((at = s.find(e.first, at)) != std::string::npos) {
          s.replace(at, std::strlen(e.first), e.second);
          at += std::strlen(e.second);
        }
      }
    }
    std::string out;
    bool space = false;
    for (unsigned char c : s) {
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') {
        space = true;
        continue;
      }
      if (space && !out.empty()) { out += ' '; }
      space = false;
      out += static_cast<char>((c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c);
    }
    return out;
  }

  enum class Kind { kSpace, kLetter, kDigit, kOther };

  static Kind kind_of(const std::string & s, std::size_t i, std::size_t & len)
  {
    const auto c = static_cast<unsigned char>(s[i]);
    len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    if (i + len > s.size()) { len = 1; }
    if (c == ' ') { return Kind::kSpace; }
    if (c >= 0x80) { return Kind::kLetter; }
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) { return Kind::kLetter; }
    if (c >= '0' && c <= '9') { return Kind::kDigit; }
    return Kind::kOther;
  }

  /// The simple_tokenizer.py pattern, alternatives in its order:
  /// <|startoftext|> | <|endoftext|> | 's|'t|'re|'ve|'m|'ll|'d | \p{L}+ | \p{N} | [^\s\p{L}\p{N}]+
  static std::vector<std::string> pre_tokenize(const std::string & s)
  {
    static const char * kSpecial[] = {"<|startoftext|>", "<|endoftext|>"};
    static const char * kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
      bool matched = false;
      for (const char * sp : kSpecial) {
        if (s.compare(i, std::strlen(sp), sp) == 0) {
          out.emplace_back(sp);
          i += std::strlen(sp);
          matched = true;
          break;
        }
      }
      if (matched) { continue; }
      for (const char * c : kContractions) {
        if (s.compare(i, std::strlen(c), c) == 0) {
          out.emplace_back(c);
          i += std::strlen(c);
          matched = true;
          break;
        }
      }
      if (matched) { continue; }
      std::size_t len = 1;
      const Kind k = kind_of(s, i, len);
      if (k == Kind::kSpace) { ++i; continue; }
      if (k == Kind::kDigit) { out.push_back(s.substr(i, 1)); ++i; continue; }
      std::size_t j = i;
      while (j < s.size()) {
        std::size_t l = 1;
        if (kind_of(s, j, l) != k) { break; }
        j += l;
      }
      out.push_back(s.substr(i, j - i));
      i = j;
    }
    return out;
  }

  /// UTF-8 code-point boundaries of a mapped token.
  static std::vector<std::string> symbols(const std::string & token)
  {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < token.size();) {
      const auto c = static_cast<unsigned char>(token[i]);
      const std::size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
      out.push_back(token.substr(i, len));
      i += len;
    }
    return out;
  }

  std::vector<std::string> bpe(const std::string & token) const
  {
    std::vector<std::string> word = symbols(token);
    if (word.empty()) { return {}; }
    word.back() += "</w>";
    if (word.size() == 1) { return word; }
    while (word.size() > 1) {
      int best = std::numeric_limits<int>::max();
      std::size_t at = 0;
      for (std::size_t i = 0; i + 1 < word.size(); ++i) {
        const auto it = ranks_.find(word[i] + '\x01' + word[i + 1]);
        if (it != ranks_.end() && it->second < best) { best = it->second; at = i; }
      }
      if (best == std::numeric_limits<int>::max()) { break; }
      const std::string first = word[at], second = word[at + 1];
      // Merge EVERY occurrence of (first, second), left to right.
      std::vector<std::string> merged;
      for (std::size_t i = 0; i < word.size();) {
        if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
          merged.push_back(first + second);
          i += 2;
        } else {
          merged.push_back(word[i]);
          ++i;
        }
      }
      word.swap(merged);
    }
    return word;
  }

  std::vector<std::string> byte_encoder_;
  std::unordered_map<std::string, int> encoder_;
  std::unordered_map<std::string, int> ranks_;
  int sot_{0};
  int eot_{0};
};

/// The text tower's CPU-side weights, from scripts/export_clip_text_assets.py:
///   "NXCLIPT1"  u32 vocab  u32 context  u32 width  u32 embed_dim  (little-endian)
///   token_embedding [vocab*width]  positional [context*width]
///   ln_final weight [width]  ln_final bias [width]  text_projection [width*embed_dim]
/// all float32.
class ClipTextWeights
{
public:
  bool load(const std::string & path, std::string & error)
  {
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "cannot read CLIP text weights: " + path; return false; }
    char magic[8];
    if (!in.read(magic, 8) || std::memcmp(magic, "NXCLIPT1", 8) != 0) {
      error = "not an NXCLIPT1 file: " + path;
      return false;
    }
    std::uint32_t h[4];
    if (!in.read(reinterpret_cast<char *>(h), sizeof(h))) { error = "truncated header"; return false; }
    vocab_ = static_cast<int>(h[0]);
    context_ = static_cast<int>(h[1]);
    width_ = static_cast<int>(h[2]);
    embed_dim_ = static_cast<int>(h[3]);
    const auto read = [&](std::vector<float> & v, std::size_t n) {
      v.resize(n);
      return static_cast<bool>(in.read(reinterpret_cast<char *>(v.data()),
                                       static_cast<std::streamsize>(n * sizeof(float))));
    };
    const auto w = static_cast<std::size_t>(width_);
    if (!read(token_, static_cast<std::size_t>(vocab_) * w) ||
        !read(position_, static_cast<std::size_t>(context_) * w) || !read(ln_w_, w) ||
        !read(ln_b_, w) || !read(projection_, w * static_cast<std::size_t>(embed_dim_))) {
      error = "truncated CLIP text weights: " + path;
      return false;
    }
    return true;
  }

  int context() const { return context_; }
  int width() const { return width_; }
  int embed_dim() const { return embed_dim_; }
  int vocab() const { return vocab_; }

  /// token_embedding[id] + positional_embedding, [context*width].
  std::vector<float> embed(const std::vector<int> & ids) const
  {
    std::vector<float> out(static_cast<std::size_t>(context_) * width_, 0.0f);
    for (int t = 0; t < context_ && t < static_cast<int>(ids.size()); ++t) {
      const int id = std::min(std::max(ids[static_cast<std::size_t>(t)], 0), vocab_ - 1);  // C++14: no std::clamp
      const float * tok = token_.data() + static_cast<std::size_t>(id) * width_;
      const float * pos = position_.data() + static_cast<std::size_t>(t) * width_;
      float * dst = out.data() + static_cast<std::size_t>(t) * width_;
      for (int d = 0; d < width_; ++d) { dst[d] = tok[d] + pos[d]; }
    }
    return out;
  }

  /// ln_final (eps 1e-5) of the hidden state at `eot_position`, @
  /// text_projection, L2-normalised: the joint-space text embedding.
  std::vector<float> project(const float * hidden, int eot_position) const
  {
    const float * x = hidden + static_cast<std::size_t>(eot_position) * width_;
    double mean = 0.0;
    for (int d = 0; d < width_; ++d) { mean += x[d]; }
    mean /= width_;
    double var = 0.0;
    for (int d = 0; d < width_; ++d) { var += (x[d] - mean) * (x[d] - mean); }
    var /= width_;
    const double inv = 1.0 / std::sqrt(var + 1e-5);
    std::vector<float> normed(static_cast<std::size_t>(width_));
    for (int d = 0; d < width_; ++d) {
      normed[static_cast<std::size_t>(d)] =
        static_cast<float>((x[d] - mean) * inv) * ln_w_[static_cast<std::size_t>(d)] +
        ln_b_[static_cast<std::size_t>(d)];
    }
    std::vector<float> out(static_cast<std::size_t>(embed_dim_), 0.0f);
    for (int d = 0; d < width_; ++d) {
      const float v = normed[static_cast<std::size_t>(d)];
      const float * row = projection_.data() + static_cast<std::size_t>(d) * embed_dim_;
      for (int e = 0; e < embed_dim_; ++e) { out[static_cast<std::size_t>(e)] += v * row[e]; }
    }
    double sum = 0.0;
    for (float v : out) { sum += static_cast<double>(v) * v; }
    const float n = static_cast<float>(std::sqrt(sum));
    if (n > 1e-12f) { for (float & v : out) { v /= n; } }
    return out;
  }

private:
  int vocab_{0};
  int context_{0};
  int width_{0};
  int embed_dim_{0};
  std::vector<float> token_, position_, ln_w_, ln_b_, projection_;
};

/// The EOT position: the largest id in the sequence (EOT is the last vocab
/// entry), as CLIP's text.argmax(dim=-1).
inline int eot_position(const std::vector<int> & ids)
{
  return static_cast<int>(std::max_element(ids.begin(), ids.end()) - ids.begin());
}


/// File names scripts/export_clip_text_assets.py writes into the assets directory.
constexpr const char* kWeightsFile = "clip_rn50_text.nxclip";
constexpr const char* kMergesFile = "clip_bpe_merges.txt";

/// The tokenizer and the CPU-side weights, loaded once and shared by the
/// preprocessor (tokenize + embed) and the postprocessor (project).
struct ClipTextAssets {
    ClipTokenizer tokenizer;
    ClipTextWeights weights;

    /// Load both files from `dir`; throws std::runtime_error naming the file and
    /// the export command when either is missing or malformed.
    static std::shared_ptr<const ClipTextAssets> load(const std::string& dir) {
        auto assets = std::make_shared<ClipTextAssets>();
        const std::string sep = (!dir.empty() && dir.back() != '/') ? "/" : "";
        std::string error;
        if (!assets->weights.load(dir + sep + kWeightsFile, error) ||
            !assets->tokenizer.load(dir + sep + kMergesFile, error)) {
            throw std::runtime_error(
                "[DXAPP] [ERROR] CLIP text assets: " + error +
                ". Write them once with: python scripts/export_clip_text_assets.py --out " +
                (dir.empty() ? std::string("<dir of the .dxnn>") : dir));
        }
        if (assets->weights.context() != ClipTokenizer::kContext ||
            assets->weights.vocab() != static_cast<int>(assets->tokenizer.vocab_size())) {
            throw std::runtime_error(
                "[DXAPP] [ERROR] CLIP text assets disagree: weights context " +
                std::to_string(assets->weights.context()) + " / vocab " +
                std::to_string(assets->weights.vocab()) + ", tokenizer context " +
                std::to_string(ClipTokenizer::kContext) + " / vocab " +
                std::to_string(assets->tokenizer.vocab_size()));
        }
        return assets;
    }
};

}  // namespace clip
}  // namespace dxapp

#endif  // DXAPP_CLIP_TEXT_ENCODER_HPP
