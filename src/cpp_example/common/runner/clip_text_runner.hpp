/**
 * @file clip_text_runner.hpp
 * @brief Runner for the CLIP text tower: prompts in, joint-space embeddings out.
 *
 * The image runners cannot drive this model: its input is text. Prompts come from
 * -t / --text (repeatable) and --text-file (one per line); with neither, three
 * built-in prompts are used. Each prompt is tokenized and embedded on the CPU
 * (ClipTextPreprocessor), run through the NPU transformer, and projected into the
 * 1024-d CLIP RN50 space (CLIPTextPostprocessor). With two or more distinct prompts
 * the runner prints their cosine-similarity matrix.
 *
 * The CPU-side weights are read from --text-assets, else the config key
 * "text_assets", else the directory holding the .dxnn
 * (scripts/export_clip_text_assets.py writes them).
 *
 * --save writes embeddings.json (prompt, token count, truncation flag, vector) to
 * the run directory. DXAPP_VERIFY=1 dumps each result like every other runner.
 * Async mode submits every prompt with RunAsync before waiting for the first.
 */

#ifndef CLIP_TEXT_RUNNER_HPP
#define CLIP_TEXT_RUNNER_HPP

#include <dxrt/dxrt_api.h>

#include <chrono>
#include <cstdlib>
#include <cxxopts.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common/processors/clip_text_postprocessor.hpp"
#include "common/utility/common_util.hpp"
#include "common/utility/run_dir.hpp"
#include "common/utility/verify_serialize.hpp"

namespace dxapp {

/// Prompts used when none is given; chosen so the similarity matrix shows
/// something (two animals closer to each other than to a vehicle).
inline std::vector<std::string> clipTextDefaultPrompts() {
    return {"a photo of a dog", "a photo of a cat", "a photo of a car"};
}

template <typename FactoryT>
class ClipTextRunnerBase {
public:
    ClipTextRunnerBase(std::unique_ptr<FactoryT> factory, bool async_mode)
        : factory_(std::move(factory)), async_(async_mode) {}

    int run(int argc, char* argv[]) {
        DXRT_TRY_CATCH_BEGIN
        Args args = parse(argc, argv);
        if (!args.image.empty()) {
            dxapp::fatal_error("[DXAPP] [ERROR] " + factory_->getModelName() +
                               " takes text, not an image: give prompts with -t \"a photo of a dog\" "
                               "(repeatable) or --text-file <file, one prompt per line>.");
        }
        std::vector<std::string> prompts = collectPrompts(args);
        dxapp::resolveAndValidateModel(args.model, argv[0]);

        if (!args.config.empty()) {
            dxapp::ModelConfig config(args.config);
            factory_->loadConfig(config);
            if (args.assets.empty()) args.assets = config.get<std::string>("text_assets", "");
        }
        if (args.assets.empty()) args.assets = fs::path(args.model).parent_path().string();
        factory_->setTextAssetsDir(args.assets);

        dxrt::InferenceOption io;
        dxrt::InferenceEngine ie(args.model, io);
        if (!dxapp::minversionforRTandCompiler(&ie)) return -1;

        auto pre_base = factory_->createPreprocessor(0, 0);
        auto* pre = dynamic_cast<ClipTextPreprocessor*>(pre_base.get());
        if (pre == nullptr) {
            dxapp::fatal_error("[DXAPP] [ERROR] " + factory_->getModelName() +
                               ": the factory did not create a ClipTextPreprocessor");
        }
        auto post = factory_->createPostprocessor(0, 0);
        const auto in_shape = ie.GetInputs().front().shape();
        if (in_shape.size() != 3 || static_cast<int>(in_shape[1]) != pre->getInputHeight() ||
            static_cast<int>(in_shape[2]) != pre->getInputWidth()) {
            dxapp::fatal_error("[DXAPP] [ERROR] model input does not match the CLIP text weights ([1," +
                               std::to_string(pre->getInputHeight()) + "," +
                               std::to_string(pre->getInputWidth()) + "] expected)");
        }

        std::cout << "[DXAPP] [INFO] Task: " << factory_->getTaskType() << std::endl;
        std::cout << "[DXAPP] [INFO] Model loaded: " << args.model << std::endl;
        std::cout << "[DXAPP] [INFO] Text assets: " << args.assets << std::endl;
        std::cout << "[DXAPP] [INFO] Prompts: " << prompts.size()
                  << (async_ ? " (async)" : " (sync)") << std::endl;

        const int total = args.loop > 0 ? args.loop : static_cast<int>(prompts.size());
        std::vector<cv::Mat> inputs(static_cast<std::size_t>(total));
        std::vector<PreprocessContext> ctxs(static_cast<std::size_t>(total));
        std::vector<int> token_counts(static_cast<std::size_t>(total));
        std::vector<bool> truncated(static_cast<std::size_t>(total));
        std::vector<std::vector<float>> last(prompts.size());
        double t_pre = 0.0, t_inf = 0.0, t_post = 0.0;
        const auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        const auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> jobs;
        for (int i = 0; i < total; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            const auto t0 = std::chrono::high_resolution_clock::now();
            bool cut = false;
            pre->encode(prompts[k % prompts.size()], inputs[k], ctxs[k], &cut);
            truncated[k] = cut;
            token_counts[k] = pre->lastTokenCount();
            t_pre += ms(t0, std::chrono::high_resolution_clock::now());
            if (async_) jobs.push_back(ie.RunAsync(inputs[k].data, nullptr));
        }
        for (int i = 0; i < total; ++i) {
            const std::size_t k = static_cast<std::size_t>(i);
            const auto t1 = std::chrono::high_resolution_clock::now();
            dxrt::TensorPtrs outputs = async_ ? ie.Wait(jobs[k]) : ie.Run(inputs[k].data, nullptr, nullptr);
            const auto t2 = std::chrono::high_resolution_clock::now();
            auto results = post->process(outputs, ctxs[k]);
            const auto t3 = std::chrono::high_resolution_clock::now();
            t_inf += ms(t1, t2);
            t_post += ms(t2, t3);
            if (results.empty()) continue;
            verify::dumpVerifyJson(results, args.model, "embedding", ctxs[k].input_height,
                                   ctxs[k].input_width);
            const std::string& text = prompts[k % prompts.size()];
            last[k % prompts.size()] = results[0].embedding;
            double sq = 0.0;
            for (float v : results[0].embedding) sq += static_cast<double>(v) * v;
            std::cout << "[" << i << "] \"" << text << "\"  tokens=" << token_counts[k]
                      << (truncated[k] ? " (cut at 77)" : "") << "  eot=" << ctxs[k].eot_position
                      << "  dim=" << results[0].dimension << "  |e|=" << std::fixed
                      << std::setprecision(4) << std::sqrt(sq) << std::defaultfloat << std::endl;
        }
        const double wall = ms(start, std::chrono::high_resolution_clock::now());

        printSimilarity(prompts, last);
        if (args.save) save(args, argv[0], prompts, last, token_counts, truncated);

        std::cout << "\n[DXAPP] [INFO] " << total << " prompt(s): preprocess "
                  << std::fixed << std::setprecision(2) << t_pre / total << " ms, inference "
                  << t_inf / total << " ms, postprocess " << t_post / total << " ms (avg); "
                  << wall << " ms total" << std::defaultfloat << std::endl;
        DXRT_TRY_CATCH_END
        return 0;
    }

private:
    struct Args {
        std::string model, image, text_file, assets, config, save_dir;
        std::vector<std::string> texts;
        int loop = -1;
        bool save = false, no_display = false, verbose = false;
    };

    Args parse(int argc, char* argv[]) {
        Args a;
        const std::string name = factory_->getModelName() + (async_ ? " Text Async Example" : " Text Sync Example");
        cxxopts::Options options(name, name + " application usage ");
        options.add_options()
            ("m, model_path", "CLIP text encoder model file (.dxnn)", cxxopts::value<std::string>(a.model))
            ("t, text", "prompt to encode (repeatable)", cxxopts::value<std::vector<std::string>>(a.texts))
            ("text-file", "file with one prompt per line", cxxopts::value<std::string>(a.text_file))
            ("text-assets", "directory holding clip_rn50_text.nxclip and clip_bpe_merges.txt "
                            "(default: config text_assets, else the .dxnn directory)",
             cxxopts::value<std::string>(a.assets))
            ("i, image_path", "not accepted: this model takes text", cxxopts::value<std::string>(a.image))
            ("s, save", "write embeddings.json to the run directory",
             cxxopts::value<bool>(a.save)->default_value("false"))
            ("save-dir", "base directory for --save", cxxopts::value<std::string>(a.save_dir)->default_value("artifacts/cpp_example"))
            ("l, loop", "number of encodes (cycles through the prompts; default: one per prompt)",
             cxxopts::value<int>(a.loop)->default_value("-1"))
            ("no-display", "accepted for symmetry with the image examples (nothing is drawn)",
             cxxopts::value<bool>(a.no_display)->default_value("false"))
            ("config", "Model configuration JSON file path", cxxopts::value<std::string>(a.config))
            ("show-log", "Enable verbose log output", cxxopts::value<bool>(a.verbose)->default_value("false"))
            ("h, help", "print usage");
        auto cmd = options.parse(argc, argv);
        if (cmd.count("help")) {
            std::cout << options.help() << std::endl;
            std::exit(0);
        }
        return a;
    }

    static std::vector<std::string> collectPrompts(const Args& a) {
        std::vector<std::string> prompts = a.texts;
        if (!a.text_file.empty()) {
            std::ifstream in(a.text_file);
            if (!in) dxapp::fatal_error("[DXAPP] [ERROR] cannot read --text-file " + a.text_file);
            std::string line;
            while (std::getline(in, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) prompts.push_back(line);
            }
        }
        if (prompts.empty()) {
            prompts = clipTextDefaultPrompts();
            std::cout << "[DXAPP] [INFO] No prompt given (-t / --text-file). Using the built-in prompts." << std::endl;
        }
        return prompts;
    }

    static void printSimilarity(const std::vector<std::string>& prompts,
                                const std::vector<std::vector<float>>& e) {
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < e.size(); ++i) if (!e[i].empty()) idx.push_back(i);
        if (idx.size() < 2) return;
        std::cout << "\nCosine similarity:" << std::endl;
        for (std::size_t a : idx) {
            std::cout << "  ";
            for (std::size_t b : idx) {
                double dot = 0.0;
                for (std::size_t d = 0; d < e[a].size(); ++d) dot += static_cast<double>(e[a][d]) * e[b][d];
                std::cout << std::fixed << std::setprecision(4) << std::setw(8) << dot;
            }
            const std::string& p = prompts[a];
            std::cout << std::defaultfloat << "  " << (p.size() <= 48 ? p : p.substr(0, 45) + "...") << std::endl;
        }
    }

    void save(const Args& a, const char* argv0, const std::vector<std::string>& prompts,
              const std::vector<std::vector<float>>& e, const std::vector<int>& tokens,
              const std::vector<bool>& cut) {
        const std::string dir = makeRunDir(a.save_dir, factory_->getModelName() + (async_ ? "_async" : "_sync"),
                                           "text", std::to_string(prompts.size()) + "-prompts");
        fs::create_directories(dir);
        writeRunInfo(dir, argv0, a.model, "text");
        nlohmann::json out = {{"model", fs::path(a.model).filename().string()}, {"prompts", nlohmann::json::array()}};
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            if (e[i].empty()) continue;
            out["prompts"].push_back({{"text", prompts[i]},
                                      {"tokens", i < tokens.size() ? tokens[i] : 0},
                                      {"truncated", i < cut.size() ? static_cast<bool>(cut[i]) : false},
                                      {"embedding", e[i]}});
        }
        std::ofstream(dir + "/embeddings.json") << out.dump(1) << "\n";
        std::cout << "[DXAPP] [INFO] Saved " << dir << "/embeddings.json" << std::endl;
    }

    std::unique_ptr<FactoryT> factory_;
    bool async_;
};

/// One prompt at a time: encode, Run, project.
template <typename FactoryT>
class ClipTextRunner : public ClipTextRunnerBase<FactoryT> {
public:
    explicit ClipTextRunner(std::unique_ptr<FactoryT> factory)
        : ClipTextRunnerBase<FactoryT>(std::move(factory), false) {}
};

/// Every prompt submitted with RunAsync, then waited for in order.
template <typename FactoryT>
class AsyncClipTextRunner : public ClipTextRunnerBase<FactoryT> {
public:
    explicit AsyncClipTextRunner(std::unique_ptr<FactoryT> factory)
        : ClipTextRunnerBase<FactoryT>(std::move(factory), true) {}
};

}  // namespace dxapp

#endif  // CLIP_TEXT_RUNNER_HPP
