#include "multi_model_graph/graph_cli.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <dxrt/dxrt_api.h>
#include <opencv2/imgcodecs.hpp>

#include "common/graph/graph_config.hpp"
#include "common/graph/graph_error.hpp"
#include "common/graph/graph_runner_async.hpp"
#include "common/graph/graph_runner_sync.hpp"
#include "common/graph/graph_visualizer.hpp"
#include "common/graph/i_registry.hpp"
#include "common/graph/shape.hpp"
#include "common/graph/stage_graph.hpp"
#include "common/registry/static_model_registry.hpp"
#include "common/third_party/nlohmann_json.hpp"
#include "common/utility/dxnn_container.hpp"
#include "common/utility/progress.hpp"
#include "common/utility/repo_path.hpp"
#include "multi_model_graph/graph_cli_interrupt.hpp"
#include "multi_model_graph/graph_cli_output.hpp"
#include "multi_model_graph/graph_consumer.hpp"
#include "multi_model_graph/graph_report_writer.hpp"

namespace dxapp {
namespace graph {
namespace cli {
namespace {

/// Frames in flight (spec 4.5). Set by the 1/2/4/8/16 sweep on DX-M1: the
/// smallest count within 3 % of the best throughput (16 reached 199.9 FPS,
/// 8 reached 191.9, 4 reached 179.8).
const std::size_t kDefaultMaxInflight = 16;

/// The largest value any numeric option takes: INT_MAX, the same on every
/// platform (a long is 32 bits on Windows). No count or timeout that a run
/// can use is anywhere near it - 2147483647 ms is 24 days - so a larger
/// value is a typo, and it would overflow on its way to a duration.
const long kMaxOptionValue = 2147483647L;

// The report schema, the path rule, the build sequence and the executor
// adapters are shared with the dx_graph Python module; see
// graph_consumer.hpp.
using consumer::AliasNotes;
using consumer::AsyncFrameExecutor;
using consumer::CollectMissing;
using consumer::DefaultModelDir;
using consumer::FileExists;
using consumer::FormatMissingList;
using consumer::FrameExecutor;
using consumer::MissingArtifact;
using consumer::MissingHeadline;
using consumer::MissingModelsError;
using consumer::NamesStreams;
using consumer::NoFramesError;
using consumer::OpenStream;
using consumer::ParseInputOverrides;
using consumer::PortsText;
using consumer::PrepareGraph;
using consumer::ReportToJson;
using consumer::ResolveStreams;
using consumer::StreamInput;
using consumer::StreamReader;
using consumer::StreamUri;
using consumer::StuckMessage;
using consumer::SyncFrameExecutor;

// ---------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------

struct GraphCliArgs {
    std::string graph_path;
    std::vector<std::string> inputs;  ///< --input values: [<source>=]<uri>, in order
    std::string output;     ///< rendered result: one video, or images
    std::string report;     ///< FrameReport as JSON
    std::string model_dir;
    std::string consumes;   ///< --list-models filter
    std::string produces;   ///< --list-models filter
    std::size_t max_inflight;
    std::size_t max_jobs_per_stage;  ///< 0 = no limit
    std::size_t stall_timeout_ms;    ///< 0 = wait forever
    std::size_t frames;     ///< 0 = every frame the source yields
    bool check_only;
    bool list_models;
    bool display;           ///< show each rendered frame in a window
    bool help;
    bool report_timing;   ///< --report-timing: node_ms in --report
    bool json;            ///< --check / --list-models: JSON on stdout

    GraphCliArgs()
        : model_dir(DefaultModelDir()),
          max_inflight(kDefaultMaxInflight),
          max_jobs_per_stage(0),
          stall_timeout_ms(0),
          frames(0),
          check_only(false),
          list_models(false),
          display(false),
          help(false),
          report_timing(false),
          json(false) {}
};

void PrintUsage(const char* program) {
    std::printf(
        "Run a multi-model pipeline described entirely by a node-graph JSON\n"
        "file. Changing which models run is a text edit, never a rebuild.\n"
        "\n"
        "Usage: %s --graph <file.json> [options]\n"
        "       %s --check <file.json> [--model-dir <dir>] [--json]\n"
        "       %s --list-models [--consumes <roi|frame|either>]"
        " [--produces <shape>]\n"
        "                     [--model-dir <dir>] [--json]\n"
        "\n"
        "Options:\n"
        "  --graph <file>        the node-graph JSON to run\n"
        "  --input [<source>=]<uri>\n"
        "                        override a source node's \"uri\" (image/video"
        " path,\n"
        "                        camera:<N>, rtsp://...). Repeat it once per"
        " source;\n"
        "                        a graph with several sources needs <source>=\n"
        "  --output <file>       write the rendered result. .mp4, .avi or"
        " .mkv: one\n"
        "                        video. Any other extension: an image; a"
        " multi-frame\n"
        "                        source writes <stem>_<index><ext>, and a"
        " live source\n"
        "                        (camera:, rtsp://) then needs --frames N."
        " With\n"
        "                        several sources, one file per stream: the"
        " video\n"
        "                        <stem>_<source_id><ext>, or images, always"
        " numbered:\n"
        "                        <stem>_<source_id>_<index><ext>\n"
        "  --display             show each rendered frame in a window; q or"
        " ESC stops\n"
        "                        the run as Ctrl-C does\n"
        "  --report <file>       write every FrameReport as JSON, ordered by\n"
        "                        (node, parent_index, roi_index)\n"
        "  --report-timing       with --report: add each frame's \"node_ms\", the\n"
        "                        time per node. Off by default, so the sync and\n"
        "                        async reports stay byte-identical\n"
        "  --model-dir <dir>     where the .dxnn files live\n"
        "                        (default: %s)\n"
        "  --max-inflight <N>    frames in flight at once (default: %u, at"
        " least 1).\n"
        "                        multi_model_graph_async only\n"
        "  --max-jobs-per-stage <N>\n"
        "                        jobs one stage may have outstanding across"
        " frames\n"
        "                        (default: 0 = no limit)."
        " multi_model_graph_async only\n"
        "  --stall-timeout-ms <N>\n"
        "                        fail when work is outstanding and nothing"
        " completes\n"
        "                        for N ms (default: 0 = wait forever).\n"
        "                        multi_model_graph_async only\n"
        "  --frames <N>          stop after N frames (default: all); with"
        " several\n"
        "                        sources, N frames of each stream\n"
        "  --check [<file>]      validate the graph and list the model files"
        " it\n"
        "                        needs, without opening the NPU, then exit.\n"
        "                        Exit code reflects the graph itself; a"
        " missing\n"
        "                        .dxnn is reported but is not an error here;"
        " a .dxnn\n"
        "                        that cannot load here (not a .dxnn, or v9"
        " before\n"
        "                        DX-RT 3.5.0) is one (exit 1)\n"
        "  --list-models         print every model this build can put in a"
        " graph, with\n"
        "                        its published flag and the container version"
        " of its\n"
        "                        .dxnn in --model-dir (v8, v9, missing; and"
        " when this\n"
        "                        DX-RT cannot load it, why), then"
        " the\n"
        "                        alias_of names of other models\n"
        "  --consumes <what>     with --list-models: roi | frame | either\n"
        "  --produces <shape>    with --list-models: boxes, obboxes,"
        " instances,\n"
        "                        keypoints, labelmap, densemap, image,"
        " scores,\n"
        "                        vector, boxes3d\n"
        "  --json                with --check or --list-models: one JSON document\n"
        "                        on stdout instead of the tables (shapes: README)\n"
        "  --help                this message\n"
        "\n"
        "Sample graphs: src/cpp_example/multi_model_graph/\n"
        "Model list:    docs/graph_models.md\n"
        "Schema:        src/cpp_example/multi_model_graph/README.md\n",
        program, program, program, DefaultModelDir().c_str(),
        static_cast<unsigned>(kDefaultMaxInflight));
}

/// True on success. On failure *error explains which argument was wrong;
/// the caller prints usage and exits 2.
bool ParseArgs(int argc, char** argv, ExecutorKind kind, GraphCliArgs* args,
               std::string* error) {
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        const bool has_value = (i + 1 < argc) && argv[i + 1][0] != '\0';
        // A value that itself looks like an option is a missing value, not
        // a value: "--graph --check x" must not silently run on a graph
        // named "--check".
        const bool value_is_option =
            has_value && std::strncmp(argv[i + 1], "--", 2) == 0;

        if (option == "--help" || option == "-h") {
            args->help = true;
            return true;
        }
        if (option == "--list-models") {
            args->list_models = true;
            continue;
        }
        if (option == "--display") {
            args->display = true;
            continue;
        }
        if (option == "--report-timing") {
            args->report_timing = true;
            continue;
        }
        if (option == "--json") {
            args->json = true;
            continue;
        }
        if (option == "--check") {
            args->check_only = true;
            // --check takes the graph path as an optional value so both
            // "--check g.json" and "--graph g.json --check" work.
            if (has_value && !value_is_option) args->graph_path = argv[++i];
            continue;
        }

        // Repeatable, once per source node (SP2); which source each value
        // replaces is decided in Main, once the graph is read.
        if (option == "--input") {
            if (!has_value || value_is_option) {
                *error = option + " needs a value";
                return false;
            }
            args->inputs.push_back(argv[++i]);
            continue;
        }

        std::string* text = NULL;
        if (option == "--graph") text = &args->graph_path;
        else if (option == "--output") text = &args->output;
        else if (option == "--report") text = &args->report;
        else if (option == "--model-dir") text = &args->model_dir;
        else if (option == "--consumes") text = &args->consumes;
        else if (option == "--produces") text = &args->produces;
        if (text != NULL) {
            if (!has_value || value_is_option) {
                *error = option + " needs a value";
                return false;
            }
            *text = argv[++i];
            continue;
        }

        std::size_t* number = NULL;
        if (option == "--max-inflight") number = &args->max_inflight;
        else if (option == "--max-jobs-per-stage") number = &args->max_jobs_per_stage;
        else if (option == "--stall-timeout-ms") number = &args->stall_timeout_ms;
        else if (option == "--frames") number = &args->frames;
        if (number != NULL) {
            // Refused before its value is read: the sync executor runs one
            // frame at a time, so it would silently ignore these.
            if (kind == kSyncExecutor &&
                (option == "--max-inflight" || option == "--max-jobs-per-stage" ||
                 option == "--stall-timeout-ms")) {
                *error = option + " applies to multi_model_graph_async only: "
                                  "this binary runs one frame at a time";
                return false;
            }
            if (!has_value || value_is_option) {
                *error = option + " needs a value";
                return false;
            }
            const std::string raw = argv[++i];
            char* end = NULL;
            errno = 0;
            const long parsed = std::strtol(raw.c_str(), &end, 10);
            if (end == raw.c_str() || *end != '\0' || parsed < 0) {
                *error = option + " expects a non-negative integer, got \"" +
                         raw + "\"";
                return false;
            }
            // strtol saturates at LONG_MAX with ERANGE; either way the value
            // is past anything the option can mean.
            if (errno == ERANGE || parsed > kMaxOptionValue) {
                *error = option + " expects an integer from 0 to " +
                         std::to_string(kMaxOptionValue) + ", got \"" + raw + "\"";
                return false;
            }
            *number = static_cast<std::size_t>(parsed);
            if (option == "--max-inflight" && parsed == 0) {
                *error = "--max-inflight must be at least 1 (frames in flight)";
                return false;
            }
            continue;
        }

        *error = "unknown option \"" + option + "\"";
        return false;
    }

    if (!args->list_models && args->graph_path.empty()) {
        *error = "no graph: pass --graph <file.json>";
        return false;
    }
    if (!args->list_models && !args->consumes.empty()) {
        *error = "--consumes only applies to --list-models";
        return false;
    }
    if (!args->list_models && !args->produces.empty()) {
        *error = "--produces only applies to --list-models";
        return false;
    }
    if (args->report_timing && args->report.empty()) {
        *error = "--report-timing needs --report <file>";
        return false;
    }
    if (args->json && !args->list_models && !args->check_only) {
        *error = "--json applies to --check and --list-models";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------
// --list-models
// ---------------------------------------------------------------------

/// Every Shape, so --produces can be validated against the same names
/// ToString() prints. Listing the enumerators by hand is deliberate: a
/// new shape added to shape.hpp makes the -Wswitch warning in
/// PayloadToJson fire in graph_consumer.cpp, next to this file, which is
/// where the omission would be noticed.
const Shape kAllShapes[] = {
    Shape::kFrame,    Shape::kBoxes,    Shape::kObBoxes, Shape::kInstances,
    Shape::kKeypoints, Shape::kLabelMap, Shape::kDenseMap, Shape::kImage,
    Shape::kScores,   Shape::kVector,   Shape::kBoxes3d, Shape::kRecords};

/// Matches --consumes. Its value has already been validated.
bool ConsumesMatches(const std::string& wanted, InputContract contract) {
    if (wanted == "roi") return AcceptsRoi(contract);
    if (wanted == "frame" || wanted == "full_frame") {
        return contract == InputContract::kFullFrame ||
               contract == InputContract::kEither;
    }
    return contract == InputContract::kEither;  // "either"
}

/**
 * @brief Validate both --list-models filters before any row is examined.
 *
 * Up front, not inside the row loop, for two reasons. An unknown value
 * must be reported even when no row would have reached the test - an
 * in-loop check on an empty or heavily pre-filtered list reports nothing
 * and exits 0. And the two filters must behave the same way: a typo in
 * --produces used to print an empty table and exit 0, which reads as
 * "no model produces that" rather than "you misspelled a shape".
 */
bool ValidateFilters(const GraphCliArgs& args, std::string* error) {
    if (!args.consumes.empty() && args.consumes != "roi" &&
        args.consumes != "frame" && args.consumes != "full_frame" &&
        args.consumes != "either") {
        *error = "--consumes expects roi, frame or either, got \"" +
                 args.consumes + "\"";
        return false;
    }
    if (!args.produces.empty()) {
        const std::size_t count = sizeof(kAllShapes) / sizeof(kAllShapes[0]);
        std::string known;
        for (std::size_t i = 0; i < count; ++i) {
            if (args.produces == ToString(kAllShapes[i])) return true;
            if (i != 0) known += ", ";
            known += ToString(kAllShapes[i]);
        }
        *error = "--produces expects one of " + known + ", got \"" +
                 args.produces + "\"";
        return false;
    }
    return true;
}

/// This process's DX-RT version, the one every stage's engine would load on.
const std::string& RuntimeVersion() {
    static const std::string version = dxrt::Configuration::GetInstance().GetVersion();
    return version;
}

/// The file column of --list-models: the container version of the model's
/// .dxnn in model_dir ("v8", "v9"), "missing", or "invalid" for a file that
/// is not a .dxnn. A container this runtime cannot load says why:
/// "v9 (needs DX-RT >= 3.5.0)" (R12).
std::string ContainerColumn(const ModelInfo& info, const std::string& model_dir) {
    const std::string path = model_dir + "/" + info.dxnn_file;
    if (!FileExists(path)) return "missing";
    uint32_t version = 0;
    std::string error;
    if (!ReadDxnnContainerVersion(path, &version, &error)) return "invalid";
    const std::string requirement = ContainerRequirement(version, RuntimeVersion());
    return "v" + std::to_string(version) + (requirement.empty() ? "" : " (" + requirement + ")");
}

bool ListedByFilters(const ModelInfo& info, const GraphCliArgs& args) {
    if (!args.consumes.empty() && !ConsumesMatches(args.consumes, info.input_contract)) {
        return false;
    }
    return args.produces.empty() || args.produces == ToString(info.output_shape);
}

/// Exit code for PrintModels: 0 on success, 2 on a bad filter value.
int PrintModels(const IModelRegistry& registry, const GraphCliArgs& args) {
    // Every non-comment line starts with a model name, because
    // tests/cpp_example/test_graph_cli.py reads the first token of each
    // line back and requires docs/graph_models.md to carry it. Anything
    // that is not a model row is a header ("model ...") or starts with
    // "-"; nothing else is printed on stdout.
    std::string filter_error;
    if (!ValidateFilters(args, &filter_error)) {
        std::fprintf(stderr, "%s\n", filter_error.c_str());
        return 2;
    }

    // One row per model, then one per alias_of alias (R6), which a filter
    // keeps or drops with the model it names. A model's old names are
    // aliases too, but one per model would double the table; --check notes
    // them where a graph uses one.
    const std::vector<ModelInfo> models = registry.list();
    const std::vector<ModelAlias> aliases = registry.aliases();
    // Columns grow with the longest name and task; a space always follows.
    std::size_t model_width = 46;
    std::size_t task_width = 28;
    for (std::size_t i = 0; i < models.size(); ++i) {
        model_width = std::max(model_width, models[i].model_name.size() + 1);
        task_width = std::max(task_width, models[i].task.size() + 1);
    }
    for (std::size_t a = 0; a < aliases.size(); ++a) {
        model_width = std::max(model_width, aliases[a].name.size() + 1);
    }
    // The file column grows too, for "v9 (needs DX-RT >= 3.5.0)".
    std::vector<std::string> files(models.size());
    std::size_t file_width = 9;
    for (std::size_t i = 0; i < models.size(); ++i) {
        if (!ListedByFilters(models[i], args)) continue;
        files[i] = ContainerColumn(models[i], args.model_dir);
        file_width = std::max(file_width, files[i].size() + 1);
    }
    const int model_w = static_cast<int>(model_width);
    const int task_w = static_cast<int>(task_width);
    const int file_w = static_cast<int>(file_width);

    std::ostringstream out;
    out << std::left << std::setw(model_w) << "model" << std::setw(task_w) << "task"
        << std::setw(11) << "produces" << std::setw(11) << "consumes"
        << std::setw(11) << "input" << std::setw(10) << "published"
        << std::setw(file_w) << "file" << "ready\n";
    out << std::string(model_width + task_width + 11 * 3 + 10 + file_width + 5, '-') << "\n";
    for (std::size_t i = 0; i < models.size(); ++i) {
        const ModelInfo& info = models[i];
        if (!ListedByFilters(info, args)) continue;
        std::ostringstream size;
        size << info.input_width << "x" << info.input_height;
        out << std::left << std::setw(model_w) << info.model_name
            << std::setw(task_w) << (info.task.empty() ? "-" : info.task)
            << std::setw(11) << ToString(info.output_shape)
            << std::setw(11) << ToString(info.input_contract)
            << std::setw(11) << size.str()
            << std::setw(10) << (info.published ? "yes" : "no")
            << std::setw(file_w) << files[i]
            << (info.ready ? "yes" : "no  " + info.not_ready_reason)
            << (info.ports.empty() ? std::string() : "  ports: " + PortsText(info)) << "\n";
    }
    for (std::size_t a = 0; a < aliases.size(); ++a) {
        if (aliases[a].kind != ModelAlias::kAliasOf) continue;
        const ModelInfo* info = registry.find(aliases[a].variant);
        if (info == NULL || !ListedByFilters(*info, args)) continue;
        out << std::left << std::setw(model_w) << aliases[a].name << "alias of "
            << aliases[a].variant << "\n";
    }
    std::fputs(out.str().c_str(), stdout);
    return 0;
}

/// One --list-models --json row: the table's columns, typed.
nlohmann::json ModelRowJson(const ModelInfo& info, const std::string& model_dir) {
    nlohmann::json row = nlohmann::json::object();
    row["model"] = info.model_name;
    row["task"] = info.task;
    row["produces"] = ToString(info.output_shape);
    row["consumes"] = ToString(info.input_contract);
    nlohmann::json input = nlohmann::json::object();
    input["width"] = info.input_width;
    input["height"] = info.input_height;
    row["input"] = input;
    row["published"] = info.published;
    row["file"] = info.dxnn_file;
    row["container"] = ContainerColumn(info, model_dir);
    row["ready"] = info.ready;
    row["reason"] = info.not_ready_reason;
    nlohmann::json ports = nlohmann::json::array();
    for (std::size_t p = 0; p < info.ports.size(); ++p) {
        nlohmann::json port = nlohmann::json::object();
        port["name"] = info.ports[p].name;
        port["shape"] = ToString(info.ports[p].shape);
        ports.push_back(port);
    }
    row["ports"] = ports;
    row["alias_of"] = nullptr;
    return row;
}

/// --list-models --json: the table as one array. 0, or 2 on a bad filter.
int PrintModelsJson(const IModelRegistry& registry, const GraphCliArgs& args) {
    std::string filter_error;
    if (!ValidateFilters(args, &filter_error)) {
        std::fprintf(stderr, "%s\n", filter_error.c_str());
        return 2;
    }
    nlohmann::json rows = nlohmann::json::array();
    const std::vector<ModelInfo> models = registry.list();
    for (std::size_t i = 0; i < models.size(); ++i) {
        if (!ListedByFilters(models[i], args)) continue;
        rows.push_back(ModelRowJson(models[i], args.model_dir));
    }
    const std::vector<ModelAlias> aliases = registry.aliases();
    for (std::size_t a = 0; a < aliases.size(); ++a) {
        if (aliases[a].kind != ModelAlias::kAliasOf) continue;
        const ModelInfo* info = registry.find(aliases[a].variant);
        if (info == NULL || !ListedByFilters(*info, args)) continue;
        nlohmann::json row = ModelRowJson(*info, args.model_dir);
        row["model"] = aliases[a].name;
        row["alias_of"] = aliases[a].variant;
        rows.push_back(row);
    }
    std::printf("%s\n", rows.dump(2).c_str());
    return 0;
}

// ---------------------------------------------------------------------
// --check / the T1 resource stage
// ---------------------------------------------------------------------

/// The --check line of one resource a node needs (R9).
std::string ResourceLine(const NodeSpec& node, const ResourceInfo& resource,
                         const std::string& model_dir) {
    const std::string who = "node \"" + node.id + "\": ";
    if (resource.kind == ResourceInfo::kNote) return "note: " + who + resource.value;
    if (resource.kind == ResourceInfo::kCompanion) {
        return "resource: " + who + "companion " + resource.value +
               (FileExists(model_dir + "/" + resource.value) ? " [present]" : " [MISSING]");
    }
    // The gallery the run opens: the node's "gallery" param over the
    // registry's (config.json), read against the repository unless it is
    // absolute - the stage overlay's rule (GalleryAgainstRoot).
    const std::map<std::string, std::string>::const_iterator param =
        node.params.text.find("gallery");
    const std::string gallery = param != node.params.text.end() ? param->second : resource.value;
    const std::string path = ResolveRepoRelative(gallery);
    std::string status = "[MISSING]";
    if (FileExists(path)) {
        char magic[8] = {0};
        std::ifstream in(path.c_str(), std::ios::binary);
        in.read(magic, sizeof(magic));
        status = in.gcount() == static_cast<std::streamsize>(sizeof(magic)) &&
                         std::memcmp(magic, "DXGAL1", 6) == 0
                     ? "[present, DXGAL1]"
                     : "[present, but not a DXGAL1 gallery]";
    }
    return "resource: " + who + "gallery " + gallery + " " + status;
}

void PrintCheckSummary(const GraphSpec& spec, const IModelRegistry& registry,
                       const std::string& graph_path,
                       const std::string& model_dir) {
    std::size_t model_nodes = 0;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        if (!spec.nodes[i].is_source && !spec.nodes[i].is_cpu) ++model_nodes;
    }
    std::printf("OK: graph \"%s\" is valid (%u nodes, %u edges, %u models)\n",
                spec.name.empty() ? graph_path.c_str() : spec.name.c_str(),
                static_cast<unsigned>(spec.nodes.size()),
                static_cast<unsigned>(spec.edges.size()),
                static_cast<unsigned>(model_nodes));

    // An old name or an alias runs its variant (R6): say so, once per node
    // (the same lines a run prints on stderr).
    std::fputs(AliasNotes(spec, registry).c_str(), stdout);

    // Columns grow with the longest id and model name; a space always
    // follows. Shorter ones keep the historical widths (10, 30). The model
    // column names the variant that runs.
    std::size_t id_width = 10;
    std::size_t model_width = 30;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        id_width = std::max(id_width, spec.nodes[i].id.size() + 1);
        if (spec.nodes[i].is_cpu) {
            model_width = std::max(model_width, std::string("(cpu ").size() + spec.nodes[i].op.size() + 2);
            continue;
        }
        if (!spec.nodes[i].is_source) {
            const ModelInfo* info = registry.find(spec.nodes[i].model);
            const std::string& name = info != NULL ? info->model_name : spec.nodes[i].model;
            model_width = std::max(model_width, name.size() + 1);
        }
    }
    const int id_w = static_cast<int>(id_width);
    const int model_w = static_cast<int>(model_width);

    std::ostringstream table;
    table << "\n" << std::left << std::setw(id_w) << "node"
          << std::setw(model_w) << "model" << std::setw(11) << "produces"
          << std::setw(11) << "consumes" << "artifact\n";
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        table << std::left << std::setw(id_w) << node.id;
        if (node.is_source) {
            table << std::setw(model_w) << "(source)" << std::setw(11) << "frame"
                  << std::setw(11) << "-" << node.uri << "\n";
            continue;
        }
        if (node.is_cpu) {
            table << std::setw(model_w) << ("(cpu " + node.op + ")")
                  << std::setw(11) << "records" << std::setw(11) << "result" << "-\n";
            continue;
        }
        const ModelInfo* info = registry.find(node.model);
        if (info == NULL) continue;
        // A present file names its container version, as --list-models
        // does ("[present, v8]"); one this runtime cannot load also says why
        // (R12), and the error itself follows on stderr (ModelFileProblems).
        const std::string column = ContainerColumn(*info, model_dir);
        const std::string status = column == "missing" ? "[MISSING]" : "[present, " + column + "]";
        table << std::setw(model_w) << info->model_name
              << std::setw(11) << ToString(info->output_shape)
              << std::setw(11) << ToString(info->input_contract)
              << info->dxnn_file << "  " << status << "\n";
    }
    std::fputs(table.str().c_str(), stdout);

    // What each node needs besides its .dxnn (R9): galleries, companion
    // engines, notes. Reported, like a missing .dxnn, not enforced.
    std::ostringstream resources;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        if (node.is_source || node.is_cpu) continue;
        const ModelInfo* info = registry.find(node.model);
        if (info == NULL) continue;
        for (std::size_t r = 0; r < info->resources.size(); ++r) {
            resources << ResourceLine(node, info->resources[r], model_dir) << "\n";
        }
    }
    if (!resources.str().empty()) std::printf("\n%s", resources.str().c_str());

    // SP2: with several sources, say which nodes each stream runs. A
    // one-source graph prints nothing here, so its --check is unchanged.
    const std::vector<StreamSpec> streams = ListStreams(spec);
    if (streams.size() > 1) {
        std::size_t source_width = 10;
        for (std::size_t s = 0; s < streams.size(); ++s) {
            source_width = std::max(source_width, streams[s].source.size() + 1);
        }
        const int source_w = static_cast<int>(source_width);
        std::ostringstream text;
        text << "\nstreams: " << streams.size()
             << " - one per source node, read in turn, one frame from each\n";
        for (std::size_t s = 0; s < streams.size(); ++s) {
            text << "  " << std::left << std::setw(source_w) << streams[s].source << "-> ";
            bool first = true;
            for (std::size_t k = 0; k < streams[s].nodes.size(); ++k) {
                if (streams[s].nodes[k] == streams[s].source) continue;
                text << (first ? "" : ", ") << streams[s].nodes[k];
                first = false;
            }
            text << "\n";
        }
        std::fputs(text.str().c_str(), stdout);
    }

    const std::vector<MissingArtifact> missing =
        CollectMissing(spec, registry, model_dir);
    if (!missing.empty()) {
        std::printf("\nMISSING: %s\n",
                    MissingHeadline(missing.size(), model_dir).c_str());
        std::fputs(FormatMissingList(missing).c_str(), stdout);
    }
}

/**
 * @brief One MODEL_LOAD error per model node whose .dxnn is present but
 *        cannot load here: not a .dxnn at all (empty, short, no DXNN
 *        header), or a container this DX-RT cannot load (v9 before 3.5.0,
 *        R12).
 *
 * --check exits 1 on any of them: unlike a missing file, which a download
 * fixes and which --check only reports, the graph cannot run with these
 * files, and a run would stop on them when the engine opens.
 */
std::vector<GraphError> ModelFileProblems(const GraphSpec& spec,
                                          const IModelRegistry& registry,
                                          const std::string& model_dir) {
    std::vector<GraphError> problems;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        if (node.is_source || node.is_cpu) continue;
        const ModelInfo* info = registry.find(node.model);
        if (info == NULL) continue;
        const std::string path = model_dir + "/" + info->dxnn_file;
        if (!FileExists(path)) continue;  // missing: reported, not an error
        uint32_t version = 0;
        std::string header_error;
        if (!ReadDxnnContainerVersion(path, &version, &header_error)) {
            problems.push_back(GraphError(
                GraphErrorCode::kModelLoad, "node \"" + node.id + "\"",
                "model \"" + node.model + "\" (" + info->dxnn_file +
                    ") is not a loadable .dxnn: " + header_error,
                "download it again: ./setup.sh --models " +
                    (info->download_name.empty() ? node.model : info->download_name)));
            continue;
        }
        const std::string error = ContainerLoadError(path, RuntimeVersion());
        if (error.empty()) continue;
        problems.push_back(GraphError(GraphErrorCode::kModelLoad, "node \"" + node.id + "\"",
                                      "model \"" + node.model + "\" (" + info->dxnn_file +
                                          ") cannot be loaded here: " + error,
                                      ""));
    }
    return problems;
}

// ---------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------

/**
 * @brief Warn when a ROI edge's filter matched nothing its producer found.
 *
 * A class-name typo is otherwise a silent no-op: "classes": ["persons"]
 * passes --check, runs, exits 0, reports "1 frame, 0 failed", and hands
 * the consumer zero crops. Nothing in the run says the filter was the
 * reason, and "persons" is a plausible spelling of a COCO class that is
 * actually "person".
 *
 * Computed here, in the CLI, from the report the executors already
 * produce - deliberately NOT a new FrameReport field. A field there would
 * have to be set identically by both executors and compared by
 * operator==, which is a sync/async parity risk taken on behalf of a
 * warning.
 *
 * Only fires when the producer actually emitted boxes: a detector that
 * found nothing is a legitimate empty frame, not a misconfigured filter.
 * Per-parent, because a node with two ROI parents has one results vector
 * for both, so a non-empty vector does not mean every parent contributed.
 * Once per edge per run, so a video does not repeat it every frame.
 */
void WarnAboutEmptyRoiEdges(const GraphSpec& spec, const FrameReport& report,
                            std::set<std::size_t>* warned) {
    for (std::size_t e = 0; e < spec.edges.size(); ++e) {
        const EdgeSpec& edge = spec.edges[e];
        if (!edge.roi.present) continue;
        if (warned->count(e) != 0) continue;

        std::map<std::string, std::vector<StageResult> >::const_iterator
            consumer = report.roi_results.find(edge.to);
        if (consumer == report.roi_results.end()) continue;
        std::size_t from_this_parent = 0;
        for (std::size_t i = 0; i < consumer->second.size(); ++i) {
            if (consumer->second[i].origin.parent_node == edge.from) {
                ++from_this_parent;
            }
        }
        if (from_this_parent != 0) continue;

        std::map<std::string, StageResult>::const_iterator producer =
            report.node_results.find(edge.from);
        if (producer == report.node_results.end()) continue;
        const BoxesData* boxes =
            dynamic_cast<const BoxesData*>(producer->second.data.get());
        if (boxes == NULL || boxes->items.empty()) continue;

        // The producer found things and the consumer got none of them.
        warned->insert(e);

        std::ostringstream filter;
        if (!edge.roi.classes.empty()) {
            filter << " \"classes\": [";
            for (std::size_t c = 0; c < edge.roi.classes.size(); ++c) {
                if (c != 0) filter << ", ";
                filter << "\"" << edge.roi.classes[c] << "\"";
            }
            filter << "]";
        }
        if (edge.roi.min_score >= 0.f) filter << " \"min_score\"";
        if (edge.roi.min_area > 0.f) filter << " \"min_area\"";
        if (edge.roi.max >= 0) filter << " \"max\"";

        std::set<std::string> emitted;
        for (std::size_t i = 0; i < boxes->items.size(); ++i) {
            if (!boxes->items[i].class_name.empty()) {
                emitted.insert(boxes->items[i].class_name);
            }
        }
        std::ostringstream names;
        std::size_t shown = 0;
        for (std::set<std::string>::const_iterator it = emitted.begin();
             it != emitted.end(); ++it, ++shown) {
            if (shown == 12) {
                names << ", ... (" << (emitted.size() - shown) << " more)";
                break;
            }
            if (shown != 0) names << ", ";
            names << *it;
        }

        std::fprintf(stderr,
                     "warning: edge \"%s\"->\"%s\" produced no crops, "
                     "although \"%s\" found %u box%s\n",
                     edge.from.c_str(), edge.to.c_str(), edge.from.c_str(),
                     static_cast<unsigned>(boxes->items.size()),
                     boxes->items.size() == 1 ? "" : "es");
        if (!filter.str().empty()) {
            std::fprintf(stderr, "  the \"roi\" filter on this edge sets:%s\n",
                         filter.str().c_str());
        }
        std::fprintf(stderr, "  \"%s\" emitted these classes: %s\n",
                     edge.from.c_str(),
                     names.str().empty() ? "(unnamed)" : names.str().c_str());
        if (report.skipped_out_of_bounds != 0) {
            std::fprintf(stderr,
                         "  %d region(s) were dropped for falling outside "
                         "the frame\n", report.skipped_out_of_bounds);
        }
        std::fprintf(stderr,
                     "  -> check the class names against that list, or relax "
                     "\"min_score\"/\"min_area\"/\"max\"\n");
    }
}

std::string FormatNumber(double value) {
    char buffer[64];
    if (std::isfinite(value) && std::fabs(value) < 1e9 &&
        std::fabs(value - std::round(value)) < 1e-4) {
        std::snprintf(buffer, sizeof(buffer), "%.0f", value);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.2f", value);
    }
    return buffer;
}

/// One payload reduced to a count, plus the measurements of a records payload.
std::string DescribePayload(const StageData* data) {
    if (data == NULL) return "none";
    switch (data->shape()) {
        case Shape::kBoxes:
        case Shape::kObBoxes:
        case Shape::kInstances: {
            const BoxesData* boxes = dynamic_cast<const BoxesData*>(data);
            if (boxes == NULL) return ToString(data->shape());
            return std::string(ToString(data->shape())) + "=" +
                   std::to_string(boxes->items.size());
        }
        case Shape::kKeypoints: {
            const KeypointsData* poses = dynamic_cast<const KeypointsData*>(data);
            if (poses == NULL) return "keypoints";
            return "keypoints=" + std::to_string(poses->items.size());
        }
        case Shape::kScores: {
            const ScoresData* scores = dynamic_cast<const ScoresData*>(data);
            if (scores == NULL) return "scores";
            return "scores=" + std::to_string(scores->items.size());
        }
        case Shape::kVector: {
            const VectorData* vector = dynamic_cast<const VectorData*>(data);
            if (vector == NULL) return "vector";
            return "vector=" + std::to_string(vector->values.size());
        }
        case Shape::kBoxes3d: {
            const Boxes3dData* boxes = dynamic_cast<const Boxes3dData*>(data);
            if (boxes == NULL) return "boxes3d";
            return "boxes3d=" + std::to_string(boxes->items.size());
        }
        case Shape::kRecords: {
            const RecordsData* records = dynamic_cast<const RecordsData*>(data);
            if (records == NULL) return "records";
            std::ostringstream text;
            text << "records=" << records->items.size();
            const std::size_t kMaxItems = 3;
            const std::size_t shown = std::min(records->items.size(), kMaxItems);
            for (std::size_t i = 0; i < shown; ++i) {
                text << (i == 0 ? "" : " |");
                const RecordItem& item = records->items[i];
                for (std::size_t n = 0; n < item.numbers.size(); ++n) {
                    text << " " << item.numbers[n].first << "="
                         << FormatNumber(item.numbers[n].second);
                }
                for (std::size_t n = 0; n < item.text.size(); ++n) {
                    text << " " << item.text[n].first << "=" << item.text[n].second;
                }
            }
            if (records->items.size() > kMaxItems) {
                text << " | +" << (records->items.size() - kMaxItems);
            }
            return text.str();
        }
        case Shape::kFrame:
        case Shape::kImage:
        case Shape::kLabelMap:
        case Shape::kDenseMap:
            return ToString(data->shape());
    }
    return ToString(data->shape());
}

/// Sum countable payloads that share one shape. A records node keeps each
/// item, because the numbers are the result a video run is watched for.
std::string DescribePayloads(const std::vector<const StageData*>& payloads) {
    std::vector<const StageData*> present;
    for (std::size_t i = 0; i < payloads.size(); ++i) {
        if (payloads[i] != NULL) present.push_back(payloads[i]);
    }
    if (present.empty()) return "none";
    const Shape shape = present[0]->shape();
    bool same = true;
    for (std::size_t i = 1; i < present.size(); ++i) {
        if (present[i]->shape() != shape) same = false;
    }
    if (!same || shape == Shape::kRecords || present.size() == 1) {
        if (present.size() == 1) return DescribePayload(present[0]);
        std::ostringstream text;
        text << "rois=" << present.size();
        for (std::size_t i = 0; i < present.size(); ++i) {
            text << " " << DescribePayload(present[i]);
        }
        return text.str();
    }
    std::size_t count = 0;
    for (std::size_t i = 0; i < present.size(); ++i) {
        const std::string one = DescribePayload(present[i]);
        const std::size_t eq = one.rfind('=');
        if (eq == std::string::npos) return "rois=" + std::to_string(present.size()) + " " + ToString(shape);
        count += static_cast<std::size_t>(std::atoi(one.c_str() + eq + 1));
    }
    return std::string(ToString(shape)) + "=" + std::to_string(count);
}

/// One stdout line per frame, so a video run is readable without a window.
void PrintFrameLine(const GraphSpec& spec, const FrameReport& report, bool several) {
    std::ostringstream line;
    if (several) line << "stream " << report.stream << " ";
    line << "frame " << report.frame_index << ":";
    if (!report.error.empty()) line << " error";
    for (std::size_t n = 0; n < spec.nodes.size(); ++n) {
        const NodeSpec& node = spec.nodes[n];
        if (node.is_source) continue;
        std::map<std::string, StageResult>::const_iterator full =
            report.node_results.find(node.id);
        std::map<std::string, std::vector<StageResult> >::const_iterator rois =
            report.roi_results.find(node.id);
        if (full == report.node_results.end() && rois == report.roi_results.end()) continue;
        std::vector<const StageData*> payloads;
        if (full != report.node_results.end()) {
            payloads.push_back(full->second.data.get());
        } else {
            for (std::size_t i = 0; i < rois->second.size(); ++i) {
                payloads.push_back(rois->second[i].data.get());
            }
        }
        line << " " << node.id << " " << DescribePayloads(payloads);
    }
    std::printf("%s\n", line.str().c_str());
    std::fflush(stdout);
}

/// Where one stream's rendered frames go.
struct StreamSink {
    std::unique_ptr<VideoOutput> video;
    std::unique_ptr<DisplayWindow> window;
    bool single_frame;
    int failed;
    StreamSink() : single_frame(false), failed(0) {}
};

/// This frame's FrameReport::node_ms in graph declaration order, for [PROGRESS].
NodeMs ProgressNodeMs(const GraphSpec& spec, const FrameReport& report) {
    NodeMs out;
    for (std::size_t n = 0; n < spec.nodes.size(); ++n) {
        std::map<std::string, double>::const_iterator ms = report.node_ms.find(spec.nodes[n].id);
        if (ms != report.node_ms.end()) out.push_back(std::make_pair(ms->first, ms->second));
    }
    return out;
}

int RunInputLoop(StageGraph& graph, FrameExecutor& executor, const GraphSpec& spec,
                 const GraphCliArgs& args, const std::vector<StreamUri>& uris) {
    // Every stream is opened before the report file: a uri that will not
    // open is a GraphError, printed by Main, and nothing has been written.
    std::vector<StreamInput> inputs;
    for (std::size_t s = 0; s < uris.size(); ++s) {
        StreamInput input;
        input.source = uris[s].source;
        input.uri = uris[s].uri;
        input.input = OpenStream(spec, uris[s]);
        inputs.push_back(std::move(input));
    }
    StreamReader reader(std::move(inputs), args.frames);
    // One source: every line, file name and window name is the one the loop
    // printed before streams existed (SP2 decision 3).
    const bool several = NamesStreams(graph);
    // Opened before the first frame: an unwritable path fails now, not after
    // the whole input, and every exit from this scope - normal, a stop
    // request, a failed frame write, an exception - finalizes the file.
    ReportFileWriter report_file;
    std::string problem;
    if (!args.report.empty() && !report_file.Open(args.report, spec.name, &problem)) {
        std::fprintf(stderr, "%s\n", problem.c_str());
        return 1;
    }
    for (std::size_t s = 0; s < reader.size(); ++s) {
        const std::string description = reader.input(s).input->getDescription();
        if (several) {
            std::printf("input:  %s: %s\n", reader.input(s).source.c_str(), description.c_str());
        } else {
            std::printf("input:  %s\n", description.c_str());
        }
    }
    // Flushed even into a pipe: this line says the frame loop is starting,
    // which is what a wrapper waiting to send Ctrl-C needs to know.
    std::fflush(stdout);

    const OutputKind output_kind = ClassifyOutput(args.output);
    std::vector<StreamSink> sinks(reader.size());
    for (std::size_t s = 0; s < reader.size(); ++s) {
        const StreamInput& input = reader.input(s);
        sinks[s].single_frame = input.input->getTotalFrames() == 1;
        if (output_kind == OutputKind::kVideo) {
            sinks[s].video.reset(new VideoOutput(
                StreamVideoPath(args.output, input.source, several), input.input->getFPS()));
        }
        if (args.display) {
            sinks[s].window.reset(new DisplayWindow(WindowName(input.source, several)));
        }
    }
    int failed_frames = 0;
    std::set<std::size_t> warned_edges;

    // Under pipelining the report that comes out is not necessarily for the
    // frame just read, so every report is rendered onto ITS OWN frame, which
    // the executor hands back with it, and into ITS OWN stream's sink.
    auto handle = [&](const FrameReport& report, const cv::Mat& shown) -> bool {
        StreamSink& sink = sinks.at(graph.StreamIndex(report.stream));
        if (!report.error.empty()) {
            ++failed_frames;
            ++sink.failed;
            if (several) {
                std::fprintf(stderr, "stream %s frame %u: %s\n", report.stream.c_str(),
                             static_cast<unsigned>(report.frame_index), report.error.c_str());
            } else {
                std::fprintf(stderr, "frame %u: %s\n",
                             static_cast<unsigned>(report.frame_index), report.error.c_str());
            }
        }
        PrintFrameLine(spec, report, several);
        progress().tickNodes(ProgressNodeMs(spec, report));  // DXAPP_PROGRESS=1 only
        WarnAboutEmptyRoiEdges(spec, report, &warned_edges);
        if (report_file.is_open() &&
            !report_file.Append(ReportToJson(report, several, args.report_timing), &problem)) {
            std::fprintf(stderr, "%s\n", problem.c_str());
            return false;
        }
        if (output_kind != OutputKind::kNone || sink.window) {
            const cv::Mat rendered = RenderReport(shown, report);
            if (sink.video) {
                if (!sink.video->Write(rendered, &problem)) {
                    std::fprintf(stderr, "%s\n", problem.c_str());
                    return false;
                }
            } else if (output_kind == OutputKind::kImagePerFrame) {
                const std::string path = StreamImagePath(args.output, report.stream, several,
                                                         sink.single_frame, report.frame_index);
                if (!cv::imwrite(path, rendered)) {
                    std::fprintf(stderr, "could not write %s\n", path.c_str());
                    return false;
                }
                std::printf("output: %s\n", path.c_str());
            }
            if (sink.window && sink.window->Show(rendered)) RequestStop();  // q / ESC
        }
        return true;
    };

    std::size_t stream = 0;
    std::size_t index = 0;
    cv::Mat frame;
    FrameReport report;
    cv::Mat shown;
    // avg_fps counts from here (model load is out, pipeline fill is in): an input
    // shorter than --max-inflight returns every report in one burst at Finish().
    progress().start();
    while (!Interrupted() && reader.Next(&stream, &frame, &index)) {
        executor.Submit(graph, stream, frame, index);
        while (executor.TryNext(&report, &shown)) {
            if (!handle(report, shown)) {
                executor.Finish(graph);
                return 1;
            }
        }
    }
    executor.Finish(graph);
    while (executor.TryNext(&report, &shown)) {
        if (!handle(report, shown)) return 1;
    }

    std::size_t total = 0;
    for (std::size_t s = 0; s < reader.size(); ++s) total += reader.frames_read(s);
    // A stream that ended without a frame (spec R9): the others ran to their
    // end first, and their outputs, report and summary lines follow as in
    // any finished run; the exit code is 1. With one source there is nothing
    // else to finish, so the run ends here, as before streams existed.
    bool no_frames = false;
    if (total == 0 && Interrupted()) {
        // A stop request that came while the models loaded ends the run
        // before the loop reads anything: not any input's fault, and the
        // same graceful stop as one mid-run (the report is "frames": []).
        std::fprintf(stderr, "interrupted before the first frame\n");
    } else {
        // One that a stop request kept from its turn has not ended, and is
        // not blamed.
        for (std::size_t s = 0; s < reader.size(); ++s) {
            if (reader.frames_read(s) != 0 || !reader.ended(s)) continue;
            std::fprintf(stderr, "%s\n", NoFramesError(reader.input(s).uri).what());
            no_frames = true;
        }
        if (no_frames && !several) return 1;
    }

    for (std::size_t s = 0; s < reader.size(); ++s) {
        if (!sinks[s].video) continue;
        sinks[s].video->Close();  // finalizes the container
        if (sinks[s].video->frames_written() != 0) {
            std::printf("output: %s\n",
                        StreamVideoPath(args.output, reader.input(s).source, several).c_str());
        }
    }

    if (report_file.is_open()) {
        if (!report_file.Finish(&problem)) {
            std::fprintf(stderr, "%s\n", problem.c_str());
            return 1;
        }
        std::printf("report: %s\n", args.report.c_str());
    }

    if (several) {
        for (std::size_t s = 0; s < reader.size(); ++s) {
            const std::size_t read = reader.frames_read(s);
            std::printf("stream %s: %u frame%s, %d failed\n", reader.input(s).source.c_str(),
                        static_cast<unsigned>(read), read == 1 ? "" : "s", sinks[s].failed);
        }
    }
    progress().finish();  // the run averages, before the summary line
    std::printf("%u frame%s, %d failed\n", static_cast<unsigned>(total),
                total == 1 ? "" : "s", failed_frames);
    return failed_frames == 0 && !no_frames ? 0 : 1;
}

}  // namespace

int Main(int argc, char** argv, ExecutorKind kind) {
    GraphCliArgs args;
    std::string error;
    if (!ParseArgs(argc, argv, kind, &args, &error)) {
        std::fprintf(stderr, "%s\n\n", error.c_str());
        PrintUsage(argv[0]);
        return 2;
    }
    if (args.help) {
        PrintUsage(argv[0]);
        return 0;
    }

    // The constructor populates itself from the generated tables; this
    // one line is the entire cost of naming a concrete registry.
    StaticModelRegistry registry;

    if (args.list_models) {
        return args.json ? PrintModelsJson(registry, args) : PrintModels(registry, args);
    }

    // First Ctrl-C or SIGTERM: stop reading, finish in-flight frames, write
    // the report. A later Ctrl-C terminates. See graph_cli_interrupt.hpp.
    InstallInterruptHandler();

    try {
        if (args.check_only) {
            GraphSpec spec = ParseGraphFile(args.graph_path);
            ValidateGraph(spec, registry);                      // T0
            // T1 is reported, not enforced: the exit code answers "is this
            // graph valid?", which is a question a checkout with no models
            // downloaded must still be able to ask. A present file that
            // cannot load here (not a .dxnn, or R12) is the exception: exit 1.
            PrintCheckSummary(spec, registry, args.graph_path, args.model_dir);
            std::fflush(stdout);
            const std::vector<GraphError> problems =
                ModelFileProblems(spec, registry, args.model_dir);
            for (std::size_t i = 0; i < problems.size(); ++i) {
                std::fprintf(stderr, "%s\n", problems[i].what());
            }
            return problems.empty() ? 0 : 1;
        }

        // Usage errors that need the graph's uri or the environment, found
        // before any model loads or any device opens.
        if (args.display && !DisplayAvailable()) {
            std::fprintf(stderr, "--display needs a window system, and neither DISPLAY "
                                 "nor WAYLAND_DISPLAY is set: run it in a desktop session, "
                                 "or save the result with --output\n\n");
            PrintUsage(argv[0]);
            return 2;
        }
        if (args.display && !DisplayOpens()) {
            std::fprintf(stderr, "--display could not open a window with %s: no window "
                                 "system answered there; run it in a desktop session, or "
                                 "save the result with --output\n\n",
                         DisplayEnvironment().c_str());
            PrintUsage(argv[0]);
            return 2;
        }
        // Which uri every stream reads, and the usage errors that depend on
        // it, before any model loads or any device opens (SP2). A graph file
        // that does not parse throws here the GraphError PrepareGraph threw
        // before: same message, same exit 1.
        const GraphSpec early = ParseGraphFile(args.graph_path);
        std::map<std::string, std::string> overrides;
        const std::string binding = ParseInputOverrides(early, args.inputs, &overrides);
        if (!binding.empty()) {
            std::fprintf(stderr, "%s\n\n", binding.c_str());
            PrintUsage(argv[0]);
            return 2;
        }
        std::vector<StreamUri> streams;
        const std::string unresolved = ResolveStreams(early, overrides, &streams);
        if (!unresolved.empty()) {
            // ParseInputOverrides binds only source ids, so this is not
            // expected; if it ever happens it is a usage error all the same.
            std::fprintf(stderr, "%s\n\n", unresolved.c_str());
            PrintUsage(argv[0]);
            return 2;
        }
        if (!args.output.empty()) {
            // SP1's live-source refusal, for each stream's own uri.
            for (std::size_t s = 0; s < streams.size(); ++s) {
                const std::string problem =
                    OutputProblem(args.output, streams[s].uri, args.frames);
                if (!problem.empty()) {
                    std::fprintf(stderr, "%s\n\n", problem.c_str());
                    PrintUsage(argv[0]);
                    return 2;
                }
            }
            // Not a usage error: exit 1, as an unwritable --report. Checked
            // on the first file each stream writes; with one source that is
            // --output itself, as before streams existed.
            const bool several = streams.size() > 1;
            const bool video = ClassifyOutput(args.output) == OutputKind::kVideo;
            for (std::size_t s = 0; s < streams.size(); ++s) {
                const std::string path =
                    !several ? args.output
                    : video  ? StreamVideoPath(args.output, streams[s].source, several)
                             : StreamImagePath(args.output, streams[s].source, several,
                                               false, 0);
                const std::string directory = OutputDirectoryProblem(path);
                if (!directory.empty()) {
                    std::fprintf(stderr, "%s\n", directory.c_str());
                    return 1;
                }
            }
        }

        // T0, then T1 (fatal here), then Build - the same sequence the
        // dx_graph module runs.
        GraphSpec spec;
        StageGraph graph;
        PrepareGraph(args.graph_path, args.model_dir, registry, &spec, &graph);
        std::printf("graph:  %s (%u nodes, %u edges)\n",
                    spec.name.empty() ? args.graph_path.c_str()
                                      : spec.name.c_str(),
                    static_cast<unsigned>(spec.nodes.size()),
                    static_cast<unsigned>(spec.edges.size()));
        std::fflush(stdout);  // the graph is built: visible through a pipe

        std::unique_ptr<FrameExecutor> executor;
        if (kind == kAsyncExecutor) {
            AsyncOptions options;
            options.max_frames_in_flight = args.max_inflight;
            options.max_jobs_per_stage = args.max_jobs_per_stage;
            options.stall_timeout_ms = args.stall_timeout_ms;
            options.interrupted = [] { return Interrupted(); };
            options.on_stuck = [](const std::vector<std::string>& ids) {
                std::fprintf(stderr, "%s\n", StuckMessage(ids).c_str());
            };
            executor.reset(new AsyncFrameExecutor(options));
        } else {
            executor.reset(new SyncFrameExecutor());
        }
        return RunInputLoop(graph, *executor, spec, args, streams);  // T2
    } catch (const MissingModelsError& missing) {
        // Already the complete message, list and trailing newline included.
        std::fputs(missing.what(), stderr);
        return 1;
    } catch (const GraphError& graph_error) {
        std::fprintf(stderr, "%s\n", graph_error.what());
        return 1;
    } catch (const std::exception& other) {
        std::fprintf(stderr, "ERROR: %s\n", other.what());
        return 1;
    }
}

}  // namespace cli
}  // namespace graph
}  // namespace dxapp
