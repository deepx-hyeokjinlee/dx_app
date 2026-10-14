#include "common/graph/graph_config.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include "common/third_party/nlohmann_json.hpp"

namespace dxapp {
namespace graph {
namespace {

using nlohmann::json;

// Carried over from Task 6's review: GCC's -Wdangling-reference fired at the
// two RequireArray("nodes"/"edges", ..., "") call sites below (originally
// graph_config.cpp:189, :269). RequireKey/RequireArray return a reference
// into parent's own storage (root, a local that outlives every use of the
// returned reference within ParseGraphText) — never into detail — so the
// warning is a genuine false positive: GCC's heuristic fires whenever *any*
// call argument requires materializing a temporary, without checking which
// parameter the returned reference actually depends on. The "" literal at
// those two sites was exactly such a temporary (converted to a temporary
// std::string bound to the const std::string& detail parameter). Passing a
// stable named lvalue instead of the literal removes that temporary at the
// source, which satisfies the heuristic without weakening the function
// signatures (detail stays a plain const&, so every other call site that
// already passes an lvalue - a `detail.str()` result stored in a variable,
// or a forwarded reference parameter - remains warning-free, unlike a
// by-value parameter, which would newly warn at those additional sites: a
// by-value parameter still copy-constructs a temporary from any argument,
// lvalue or not, at the call site, so a fix at that granularity trades two
// warnings for four instead of clearing them).
const std::string kNoDetail;

std::string Where(const std::string& origin, const std::string& detail) {
    return origin + (detail.empty() ? "" : " " + detail);
}

void RequireObject(const json& node, const std::string& origin,
                   const std::string& detail) {
    if (!node.is_object()) {
        throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, detail),
                         "expected a JSON object", "");
    }
}

/// `how` completes the three-part message for a key that is simply
/// absent. A MISTYPED required key never reaches here - RejectUnknownKeys
/// runs first and answers it as the typo it is - so this is the
/// genuine-omission path, where there is nothing to suggest and the
/// useful thing to say is what the key is for.
const json& RequireKey(const json& parent, const char* key,
                       const std::string& origin, const std::string& detail,
                       const char* how = "") {
    json::const_iterator it = parent.find(key);
    if (it == parent.end()) {
        throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, detail),
                         std::string("missing required key \"") + key + "\"",
                         how);
    }
    return *it;
}

std::string RequireString(const json& parent, const char* key,
                          const std::string& origin, const std::string& detail,
                          const char* how = "") {
    const json& value = RequireKey(parent, key, origin, detail, how);
    if (!value.is_string()) {
        throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, detail),
                         std::string("\"") + key + "\" must be a string", "");
    }
    return value.get<std::string>();
}

double RequireNumber(const json& value, const char* key,
                     const std::string& origin, const std::string& detail) {
    if (!value.is_number()) {
        throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, detail),
                         std::string("\"") + key + "\" must be a number", "");
    }
    return value.get<double>();
}

const json& RequireArray(const json& parent, const char* key,
                         const std::string& origin, const std::string& detail) {
    const json& value = RequireKey(parent, key, origin, detail);
    if (!value.is_array()) {
        throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, detail),
                         std::string("\"") + key + "\" must be an array", "");
    }
    return value;
}

// --- The schema's key sets, in ONE place ---------------------------
//
// These tables are the only definition of "a key this parser accepts".
// They drive the rejection below AND the "accepted keys" list printed in
// its message, so a suggestion can never advertise a key the parser does
// not read - the drift that makes a helpful error worse than none.
//
// "prompt" is deliberately absent from kNodeKeys: RejectReserved() runs
// first and answers it with kGraphReserved, which says more than
// "unknown key" does. A node's "params" object is NOT key-checked - its
// keys are model parameter names, which are open by design.
// "$schema" is accepted and ignored. It is inert - no schema key is
// within edit distance 2 of it, so tolerating it cannot mask a typo for a
// real key - and rejecting it would make this validator fight the editor
// tooling that makes hand-editing a graph pleasant, which is the premise
// the whole example rests on. This tolerance is NOT extended to
// "_comment"-style keys: "name" already carries intent, and every extra
// tolerated key is somewhere a typo can hide.
const char* const kTopKeys[] = {"version", "name", "nodes", "edges",
                                "$schema"};
const char* const kNodeKeys[] = {"id", "type", "uri", "model", "op", "params",
                                 "track", "engine"};
const char* const kEngineKeys[] = {"buffers", "share"};
/// DX-RT's DXRT_TASK_MAX_LOAD_LIMIT: the largest I/O buffer count it accepts.
const int kMaxEngineBuffers = 100;
const char* const kTrackKeys[] = {"algo", "iou", "max_age"};
const char* const kEdgeKeys[] = {"from", "to", "roi", "port", "carry"};
const char* const kRoiKeys[] = {"classes", "min_score", "min_area", "pad",
                                "max", "align"};
/// Node types v1 defines. "fuse" is reserved and answered by
/// RejectReserved() before the type check runs, with a better message.
const char kSourceType[] = "source";
const char kCpuType[] = "cpu";
const char* const kCpuOps[] = {"headpose", "volume"};

/**
 * @brief Damerau-Levenshtein (optimal string alignment) distance.
 *
 * A transposition costs 1, not 2, which is the difference between a
 * useful suggestion and a misleading one: plain Levenshtein puts "rio"
 * two edits from BOTH "roi" and "to", and a tie broken by declaration
 * order then advises "to" for what is obviously "roi". Inputs are schema
 * keys, so the full matrix costs nothing.
 */
std::size_t EditDistance(const std::string& a, const std::string& b) {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    std::vector<std::vector<std::size_t> > d(
        n + 1, std::vector<std::size_t>(m + 1, 0));
    for (std::size_t i = 0; i <= n; ++i) d[i][0] = i;
    for (std::size_t j = 0; j <= m; ++j) d[0][j] = j;
    for (std::size_t i = 1; i <= n; ++i) {
        for (std::size_t j = 1; j <= m; ++j) {
            const std::size_t cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            d[i][j] = std::min(std::min(d[i - 1][j] + 1, d[i][j - 1] + 1),
                               d[i - 1][j - 1] + cost);
            if (i > 1 && j > 1 && a[i - 1] == b[j - 2] &&
                a[i - 2] == b[j - 1]) {
                d[i][j] = std::min(d[i][j], d[i - 2][j - 2] + 1);
            }
        }
    }
    return d[n][m];
}

/// 0-2 means "worth suggesting", 3 means "say nothing".
const std::size_t kNoSuggestion = 3;

/**
 * @brief How close a typo is to a known key, for ranking suggestions.
 *
 * Edit distance alone misses the commonest typo of all - a key the user
 * started spelling right and then finished wrong. "paddng" is three edits
 * from "pad", far enough that a distance rule stays silent, yet "pad" is
 * plainly what was meant. A shared prefix of at least three characters is
 * therefore treated as a near match, ranked just behind a genuine
 * one-or-two-edit hit.
 */
std::size_t Closeness(const std::string& key, const std::string& known) {
    const std::size_t distance = EditDistance(key, known);
    if (distance < kNoSuggestion) return distance;
    const std::size_t shorter = std::min(key.size(), known.size());
    if (shorter >= 3 && key.compare(0, shorter, known, 0, shorter) == 0) {
        return kNoSuggestion - 1;
    }
    return kNoSuggestion;
}

/// The known key nearest to `key`, or "" when nothing is close enough.
template <std::size_t N>
std::string NearestKey(const std::string& key,
                       const char* const (&known)[N]) {
    std::string nearest;
    std::size_t best = kNoSuggestion;
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t score = Closeness(key, known[i]);
        if (score < best) {
            best = score;
            nearest = known[i];
        }
    }
    return nearest;
}

/// Same idea over a runtime list, for model names. The registry holds 348
/// of them and a typo on a model name used to be answered with nothing but
/// "run --list-models" against a 348-row table - on the one string the
/// README tells a user to edit. Registry spelling already differs from the
/// model zoo's for 119 of those, so getting it slightly wrong is the
/// normal case, not an unusual one.
std::string Lower(const std::string& text) {
    std::string out(text);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<char>(
            std::tolower(static_cast<unsigned char>(out[i])));
    }
    return out;
}

std::string NearestName(const std::string& name,
                        const std::vector<std::string>& known) {
    // Case-folded, unlike the schema-key suggester. The registry spells
    // models in lower case and the model zoo does not - "YoloV8N" for
    // "yolov8n", "ResNet50" for "resnet50", "FastSAM-s" for "fastsam_s" -
    // so a user arriving from the zoo is three substitutions away from
    // the right name and would get no suggestion at all under a
    // case-sensitive rule. Folding makes that exact case a distance-0
    // match, which is the most useful answer available. Schema keys are
    // all lower case already, so nothing there needs this.
    const std::string folded = Lower(name);
    std::string nearest;
    std::size_t best = kNoSuggestion;
    for (std::size_t i = 0; i < known.size(); ++i) {
        const std::size_t score = Closeness(folded, Lower(known[i]));
        if (score < best) {
            best = score;
            nearest = known[i];
        }
    }
    return nearest;
}

template <std::size_t N>
std::string JoinKeys(const char* const (&known)[N]) {
    std::string all;
    for (std::size_t i = 0; i < N; ++i) {
        if (i != 0) all += ", ";
        all += known[i];
    }
    return all;
}

std::string Quote(const std::string& text) { return "\"" + text + "\""; }

/**
 * @brief The value errors of ONE JSON object, thrown as one GraphError.
 *
 * Unknown keys are rejected before this runs (RejectUnknownKeys), so a typo
 * is still answered as a typo. After that, every bad VALUE of one object is
 * reported at once, for the same reason every unknown key is: fixing one
 * per round trip reads as the tool fighting the user. One violation keeps
 * its own what/how - the exact text the parser printed before collection
 * existed - so no existing message moves.
 */
class Violations {
 public:
    void Add(const std::string& what, const std::string& how) {
        what_.push_back(what);
        how_.push_back(how);
    }
    bool empty() const { return what_.empty(); }
    void ThrowIfAny(const std::string& where) const {
        if (what_.empty()) return;
        if (what_.size() == 1) {
            throw GraphError(GraphErrorCode::kGraphSchema, where, what_[0], how_[0]);
        }
        std::ostringstream what;
        what << what_.size() << " invalid values: ";
        std::string how;
        for (std::size_t i = 0; i < what_.size(); ++i) {
            if (i != 0) what << "; ";
            what << what_[i];
            if (how_[i].empty()) continue;
            if (!how.empty()) how += "; ";
            how += how_[i];
        }
        throw GraphError(GraphErrorCode::kGraphSchema, where, what.str(), how);
    }

 private:
    std::vector<std::string> what_;
    std::vector<std::string> how_;
};

/// The only tracker the engine builds (stage_graph.cpp: IouTracker).
const char* const kTrackAlgos[] = {"iou"};
const long long kMaxCount = 2147483647LL;

bool ReadNumber(const json& value, const char* key, Violations* bad, double* out) {
    if (!value.is_number()) {
        bad->Add(Quote(key) + " must be a number", "");
        return false;
    }
    *out = value.get<double>();
    return true;
}

/// A whole number in [low, kMaxCount]. By value: 16.0 is 16 (spec R2).
bool ReadCount(const json& value, const char* key, long long low,
               const std::string& how, Violations* bad, int* out) {
    double number = 0.0;
    if (!ReadNumber(value, key, bad, &number)) return false;
    if (number != std::floor(number) || number < static_cast<double>(low)) {
        std::ostringstream what;
        what << Quote(key) << " must be a whole number of at least " << low
             << ", got " << value.dump();
        bad->Add(what.str(), how);
        return false;
    }
    if (number > static_cast<double>(kMaxCount)) {
        bad->Add(Quote(key) + " must be at most 2147483647, got " + value.dump(), how);
        return false;
    }
    *out = static_cast<int>(number);
    return true;
}

/// A number in [low, high] ((low, high] when low_open).
bool ReadRange(const json& value, const char* key, double low, bool low_open,
               double high, const std::string& range_text,
               const std::string& how, Violations* bad, float* out) {
    double number = 0.0;
    if (!ReadNumber(value, key, bad, &number)) return false;
    const bool above_low = low_open ? number > low : number >= low;
    if (!above_low || number > high) {
        bad->Add(Quote(key) + " must be " + range_text + ", got " + value.dump(), how);
        return false;
    }
    *out = static_cast<float>(number);
    return true;
}

/**
 * @brief Reject every key the schema does not define, naming them and
 *        saying what to write instead.
 *
 * A key this parser does not read used to be dropped in silence, which in
 * a product configured by editing JSON is worse than an error: "paddng"
 * for "pad" left the graph running and quietly doing something other than
 * what was asked, and nothing prompted the user to look. --check could not
 * catch it either, because its table prints nodes and models, not "roi"
 * options.
 *
 * EVERY unknown key in one object is reported in one message, not just
 * the first. nlohmann iterates an object in sorted key order, so reporting
 * only the first meant a file with "classez" and "paddng" hid "paddng"
 * behind the alphabet and cost a second round trip to find. A user fixing
 * typos one at a time concludes the tool is fighting them - and sorted
 * order also makes the collected list deterministic, so a test can pin it.
 *
 * The suggestion is a nearest match at edit distance 1 or 2 against the
 * SAME table the accepted-keys list is printed from, so a suggestion can
 * never advertise a key the parser does not read. Ties resolve to the
 * first candidate in declaration order. When nothing is close enough the
 * accepted keys are enumerated anyway - enumerating beats silence.
 */
template <std::size_t N>
void RejectUnknownKeys(const json& object, const char* const (&known)[N],
                       const std::string& origin, const std::string& detail) {
    std::vector<std::string> unknown;
    std::vector<std::string> nearest;
    for (json::const_iterator it = object.begin(); it != object.end(); ++it) {
        const std::string key = it.key();
        bool accepted = false;
        for (std::size_t i = 0; i < N && !accepted; ++i) {
            if (key == known[i]) accepted = true;
        }
        if (accepted) continue;
        unknown.push_back(key);
        nearest.push_back(NearestKey(key, known));
    }
    if (unknown.empty()) return;

    std::string what;
    if (unknown.size() == 1) {
        what = "unknown key " + Quote(unknown[0]);
    } else {
        std::ostringstream text;
        text << unknown.size() << " unknown keys: ";
        for (std::size_t i = 0; i < unknown.size(); ++i) {
            if (i != 0) text << ", ";
            text << Quote(unknown[i]);
        }
        what = text.str();
    }

    // "did you mean X?" for one key; "X for Y, Z for W?" for several, so
    // each suggestion stays attached to the key it is for.
    std::string how;
    std::string advice;
    for (std::size_t i = 0; i < unknown.size(); ++i) {
        if (nearest[i].empty()) continue;
        if (!advice.empty()) advice += ", ";
        advice += Quote(nearest[i]);
        if (unknown.size() > 1) advice += " for " + Quote(unknown[i]);
    }
    if (!advice.empty()) how = "did you mean " + advice + "? ";
    how += "accepted keys: " + JoinKeys(known);

    throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, detail), what,
                     how);
}

/**
 * @brief Reject a node "type" this release does not define.
 *
 * The keys were locked down in round 2; this is the same hole one level
 * down. A node carrying BOTH a "type" the parser does not know and a
 * "model" was silently treated as a model node - and the commonest thing
 * a hand-editor writes, {"type": "model", "model": "..."}, worked only by
 * that accident. A node with an unknown "type" and no "model" already
 * failed usefully ("neither a source nor a model"), so this closes the
 * remaining half.
 */
void RejectUnknownNodeType(const json& node, const std::string& origin,
                           const std::string& detail) {
    json::const_iterator type = node.find("type");
    if (type == node.end()) return;
    if (!type->is_string()) {
        throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, detail),
                         "\"type\" must be a string", "");
    }
    const std::string value = type->get<std::string>();
    if (value == kSourceType || value == kCpuType) return;

    std::string how;
    const std::size_t source_distance = Closeness(value, kSourceType);
    const std::size_t cpu_distance = Closeness(value, kCpuType);
    if (source_distance < kNoSuggestion && source_distance <= cpu_distance) {
        how = "did you mean \"source\"? ";
    } else if (cpu_distance < kNoSuggestion) {
        how = "did you mean \"cpu\"? ";
    }
    how += "accepted node types are \"source\" and \"cpu\"; a model node carries a "
           "\"model\" name and no \"type\"";
    throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, detail),
                     "unknown node type " + Quote(value), how);
}

/// A label for a node before its "id" has been read: the id when it is
/// there, else the model name, else the array index alone. Round 2 gave
/// every unknown-key message the node's own id; a MISSING required key
/// used to fall back to a bare index, which is the one case where the
/// reader has least to go on.
std::string NodeLabel(const json& node, std::size_t index) {
    std::ostringstream label;
    label << "node[" << index << "]";
    json::const_iterator id = node.find("id");
    if (id != node.end() && id->is_string()) {
        label << " " << Quote(id->get<std::string>());
        return label.str();
    }
    json::const_iterator model = node.find("model");
    if (model != node.end() && model->is_string()) {
        label << " (model " << Quote(model->get<std::string>()) << ")";
    }
    return label.str();
}

/// Same idea for an edge: whichever endpoints are readable, with "?" for
/// the one that is missing - which is exactly the case a mistyped "from"
/// or "to" produces.
std::string EdgeLabel(const json& edge, std::size_t index) {
    std::ostringstream label;
    label << "edge[" << index << "]";
    json::const_iterator from = edge.find("from");
    json::const_iterator to = edge.find("to");
    const bool has_from = from != edge.end() && from->is_string();
    const bool has_to = to != edge.end() && to->is_string();
    if (!has_from && !has_to) return label.str();
    label << " " << (has_from ? Quote(from->get<std::string>()) : "?") << "->"
          << (has_to ? Quote(to->get<std::string>()) : "?");
    return label.str();
}

void RejectReserved(const json& node, const std::string& origin,
                    const std::string& detail) {
    if (node.find("prompt") != node.end()) {
        throw GraphError(
            GraphErrorCode::kGraphReserved, Where(origin, detail),
            "\"prompt\" is reserved for text-conditioned models and is not "
            "supported in this release",
            "remove \"prompt\"; see README.md for the models this build runs");
    }
    json::const_iterator type = node.find("type");
    if (type != node.end() && type->is_string() &&
        type->get<std::string>() == "fuse") {
        throw GraphError(
            GraphErrorCode::kGraphReserved, Where(origin, detail),
            "node type \"fuse\" is reserved for multi-model fusion and is not "
            "supported in this release",
            "use a cpu node (\"op\": \"headpose\" or \"volume\") with "
            "\"carry\": \"result\" edges; a generic fuse is not a node type");
    }
}

RoiSpec ParseRoi(const json& edge, const std::string& origin,
                 const std::string& detail) {
    RoiSpec roi;
    json::const_iterator found = edge.find("roi");
    if (found == edge.end()) return roi;
    RequireObject(*found, origin, detail + " \"roi\"");
    roi.present = true;

    const json& node = *found;
    RejectUnknownKeys(node, kRoiKeys, origin, detail + " \"roi\"");
    // Read in kRoiKeys order, so several violations are listed in the
    // schema's own order.
    Violations bad;
    json::const_iterator it = node.find("classes");
    if (it != node.end()) {
        if (!it->is_array()) {
            bad.Add("\"classes\" must be an array", "");
        } else {
            bool reported = false;
            for (std::size_t i = 0; i < it->size(); ++i) {
                const json& entry = (*it)[i];
                if (entry.is_string()) {
                    roi.classes.push_back(entry.get<std::string>());
                } else if (!reported) {
                    bad.Add("\"classes\" entries must be strings", "");
                    reported = true;
                }
            }
        }
    }
    it = node.find("min_score");
    if (it != node.end()) {
        ReadRange(*it, "min_score", 0.0, false, 1.0, "between 0 and 1",
                  "\"min_score\" is a confidence as the producer reports it "
                  "(0.35, not 35)",
                  &bad, &roi.min_score);
    }
    it = node.find("min_area");
    if (it != node.end()) {
        const double kFloatMax = static_cast<double>(std::numeric_limits<float>::max());
        if (it->is_number() && it->get<double>() > kFloatMax) {
            // Read into a float: a larger value would overflow the cast.
            bad.Add("\"min_area\" must be at most " + json(kFloatMax).dump() + ", got " +
                        it->dump(),
                    "\"min_area\" is an area in pixels");
        } else {
            ReadRange(*it, "min_area", 0.0, false,
                      std::numeric_limits<double>::max(), "at least 0",
                      "\"min_area\" is an area in pixels", &bad, &roi.min_area);
        }
    }
    it = node.find("pad");
    if (it != node.end()) {
        ReadRange(*it, "pad", 0.0, false, 1.0, "between 0 and 1",
                  "\"pad\" grows each box by this fraction of its size on "
                  "every side (0.05 = 5 %)",
                  &bad, &roi.pad);
    }
    it = node.find("max");
    if (it != node.end()) {
        ReadCount(*it, "max", 1,
                  "\"max\" keeps at most this many boxes; leave it out for no "
                  "limit",
                  &bad, &roi.max);
    }
    it = node.find("align");
    if (it != node.end()) {
        if (!it->is_string()) {
            bad.Add("\"align\" must be a string", "");
        } else if (it->get<std::string>() != "face5") {
            bad.Add("unknown align mode \"" + it->get<std::string>() + "\"",
                    "the only supported value is \"face5\"");
        } else {
            roi.align = it->get<std::string>();
        }
    }
    bad.ThrowIfAny(Where(origin, detail));
    return roi;
}

}  // namespace

const NodeSpec* GraphSpec::FindNode(const std::string& id) const {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].id == id) return &nodes[i];
    }
    return NULL;
}

GraphSpec ParseGraphText(const std::string& json_text,
                         const std::string& origin) {
    json root;
    try {
        root = json::parse(json_text);
    } catch (const std::exception& error) {
        throw GraphError(GraphErrorCode::kGraphSchema, origin,
                         std::string("not valid JSON: ") + error.what(), "");
    }
    RequireObject(root, origin, "");

    GraphSpec spec;

    json::const_iterator version = root.find("version");
    if (version == root.end()) {
        throw GraphError(GraphErrorCode::kGraphSchema, origin,
                         "missing required key \"version\"",
                         "add \"version\": 1");
    }
    // The gate, so it is reported alone: nothing after it is read against
    // a schema the graph may not be written for.
    const double version_number =
        RequireNumber(*version, "version", origin, kNoDetail);
    if (version_number != std::floor(version_number)) {
        throw GraphError(GraphErrorCode::kGraphSchema, origin,
                         "\"version\" must be a whole number, got " +
                             version->dump(),
                         "add \"version\": 1");
    }
    if (version_number != 1.0) {
        std::ostringstream message;
        message << "graph version " << version->dump()
                << " is not supported by this build";
        throw GraphError(GraphErrorCode::kGraphVersion, origin, message.str(),
                         "supports: 1");
    }
    spec.version = 1;

    // After the version gate on purpose: a graph written for a later
    // version legitimately carries keys this build has never heard of, and
    // "graph version 2 is not supported" is the useful answer to it.
    RejectUnknownKeys(root, kTopKeys, origin, kNoDetail);

    json::const_iterator name = root.find("name");
    if (name != root.end()) {
        if (!name->is_string()) {
            Violations bad;
            bad.Add("\"name\" must be a string, got " + name->dump(),
                    "\"name\" is free text: put it in quotes");
            bad.ThrowIfAny(origin);
        }
        spec.name = name->get<std::string>();
    }

    const json& nodes = RequireArray(root, "nodes", origin, kNoDetail);
    std::set<std::string> seen;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const json& node = nodes[i];
        std::ostringstream detail;
        detail << "node[" << i << "]";
        RequireObject(node, origin, detail.str());
        RejectReserved(node, origin, detail.str());

        // BEFORE the required keys are read, not after. "di" for "id" is
        // a mistyped required key, and reading "id" first answered it with
        // a bare "missing required key" and no suggestion - leaving the
        // two most important keys in the schema as the two without one.
        // As an unknown key it gets the same treatment as every other
        // typo. The label is best-effort for the same reason: the id may
        // be the very thing that is missing.
        const std::string node_label = NodeLabel(node, i);
        RejectUnknownKeys(node, kNodeKeys, origin, node_label);
        RejectUnknownNodeType(node, origin, node_label);

        NodeSpec parsed;
        parsed.id = RequireString(node, "id", origin, node_label,
                                  "add \"id\": a name unique within this graph");
        if (!seen.insert(parsed.id).second) {
            throw GraphError(GraphErrorCode::kGraphSchema, origin,
                             "duplicate node id \"" + parsed.id + "\"",
                             "node ids must be unique");
        }

        json::const_iterator type = node.find("type");
        const bool declared_source =
            type != node.end() && type->is_string() &&
            type->get<std::string>() == kSourceType;
        const bool declared_cpu =
            type != node.end() && type->is_string() &&
            type->get<std::string>() == kCpuType;
        json::const_iterator model = node.find("model");
        json::const_iterator op = node.find("op");

        if (declared_source) {
            if (op != node.end()) {
                throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, node_label),
                                 "\"op\" belongs on a cpu node",
                                 "remove \"op\" from this source");
            }
            parsed.is_source = true;
            parsed.uri = RequireString(
                node, "uri", origin, node_label,
                "add \"uri\": an image or video path, camera:<N>, or "
                "rtsp://...");
        } else if (declared_cpu) {
            if (model != node.end() || node.find("uri") != node.end()) {
                throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, node_label),
                                 "a cpu node has no \"model\" and no \"uri\"",
                                 "keep \"type\": \"cpu\" and \"op\" only");
            }
            parsed.is_cpu = true;
            parsed.op = RequireString(
                node, "op", origin, node_label,
                "add \"op\": \"headpose\" or \"volume\"");
            const char* const* op_end = kCpuOps + sizeof(kCpuOps) / sizeof(kCpuOps[0]);
            if (std::find(kCpuOps, op_end, parsed.op) == op_end) {
                throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, node_label),
                                 "unknown cpu op " + Quote(parsed.op),
                                 "accepted \"op\" values: " + JoinKeys(kCpuOps));
            }
        } else if (model != node.end()) {
            if (op != node.end()) {
                throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, node_label),
                                 "\"op\" belongs on a cpu node",
                                 "remove \"op\", or set \"type\": \"cpu\" and drop \"model\"");
            }
            if (!model->is_string()) {
                throw GraphError(GraphErrorCode::kGraphSchema,
                                 Where(origin, node_label),
                                 "\"model\" must be a string", "");
            }
            parsed.model = model->get<std::string>();
        } else {
            throw GraphError(
                GraphErrorCode::kGraphSchema, Where(origin, node_label),
                "node \"" + parsed.id +
                    "\" is neither a source, a cpu node, nor a model",
                "add \"type\": \"source\" with a \"uri\", \"type\": \"cpu\" "
                "with an \"op\", or a \"model\" name");
        }

        json::const_iterator params = node.find("params");
        if (params != node.end()) {
            RequireObject(*params, origin, node_label + " \"params\"");
            Violations bad;
            bool switch_hinted = false;
            for (json::const_iterator it = params->begin();
                 it != params->end(); ++it) {
                // U-62: the kind is checked against the model in
                // ValidateGraph; the parser has no registry.
                const json& value = it.value();
                if (value.is_number()) {
                    parsed.params.numeric[it.key()] = value.get<double>();
                } else if (value.is_string()) {
                    parsed.params.text[it.key()] = value.get<std::string>();
                } else if (value.is_array() &&
                           std::all_of(value.begin(), value.end(),
                                       [](const json& e) { return e.is_string(); })) {
                    std::vector<std::string>& list = parsed.params.lists[it.key()];
                    for (json::const_iterator e = value.begin(); e != value.end(); ++e) {
                        list.push_back(e->get<std::string>());
                    }
                } else {
                    const bool hint = value.is_boolean() && !switch_hinted;
                    switch_hinted = switch_hinted || hint;
                    bad.Add(Quote(it.key()) +
                                " must be a number, a string or an array of strings, got " +
                                value.dump(),
                            hint ? "write a switch as 1 or 0" : "");
                }
            }
            bad.ThrowIfAny(Where(origin, detail.str() + " \"params\""));
        }

        json::const_iterator track = node.find("track");
        if (track != node.end()) {
            RequireObject(*track, origin, node_label + " \"track\"");
            RejectUnknownKeys(*track, kTrackKeys, origin,
                              node_label + " \"track\"");
            parsed.track.present = true;
            // In kTrackKeys order, like "roi".
            Violations bad;
            json::const_iterator field = track->find("algo");
            if (field != track->end()) {
                if (!field->is_string()) {
                    bad.Add("\"algo\" must be a string", "");
                } else {
                    const std::string algo = field->get<std::string>();
                    const char* const* known_end =
                        kTrackAlgos + sizeof(kTrackAlgos) / sizeof(kTrackAlgos[0]);
                    if (std::find(kTrackAlgos, known_end, algo) == known_end) {
                        bad.Add("unknown tracker " + Quote(algo),
                                "accepted \"algo\" values: " +
                                    JoinKeys(kTrackAlgos));
                    } else {
                        parsed.track.algo = algo;
                    }
                }
            }
            field = track->find("iou");
            if (field != track->end()) {
                ReadRange(*field, "iou", 0.0, true, 1.0,
                          "greater than 0 and at most 1",
                          "\"iou\" is the overlap a box needs to continue a "
                          "track (0.3 is typical)",
                          &bad, &parsed.track.iou);
            }
            field = track->find("max_age");
            if (field != track->end()) {
                ReadCount(*field, "max_age", 0,
                          "\"max_age\" is how many frames an unmatched track "
                          "survives",
                          &bad, &parsed.track.max_age);
            }
            bad.ThrowIfAny(Where(origin, detail.str()));
        }

        json::const_iterator engine = node.find("engine");
        if (engine != node.end()) {
            if (parsed.is_source || parsed.is_cpu) {
                throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, node_label),
                                 "\"engine\" belongs on a model node",
                                 "remove \"engine\" from this node");
            }
            RequireObject(*engine, origin, node_label + " \"engine\"");
            RejectUnknownKeys(*engine, kEngineKeys, origin, node_label + " \"engine\"");
            parsed.engine.present = true;
            Violations bad;
            json::const_iterator field = engine->find("buffers");
            if (field != engine->end()) {
                int buffers = 0;
                if (ReadCount(*field, "buffers", 1,
                              "\"buffers\" is how many inputs DX-RT keeps in flight for "
                              "this model (default 6); fewer use less NPU memory",
                              &bad, &buffers)) {
                    if (buffers > kMaxEngineBuffers) {
                        bad.Add("\"buffers\" must be at most " +
                                    std::to_string(kMaxEngineBuffers) + ", got " + field->dump(),
                                "DX-RT accepts 1 to " + std::to_string(kMaxEngineBuffers));
                    } else {
                        parsed.engine.buffers = buffers;
                    }
                }
            }
            field = engine->find("share");
            if (field != engine->end()) {
                if (!field->is_boolean()) {
                    bad.Add("\"share\" must be true or false, got " + field->dump(),
                            "\"share\": true lets nodes of the same model use one engine");
                } else {
                    parsed.engine.share = field->get<bool>();
                }
            }
            bad.ThrowIfAny(Where(origin, node_label + " \"engine\""));
        }

        if (parsed.is_cpu && parsed.track.present) {
            throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, node_label),
                             "a cpu node cannot track",
                             "put \"track\" on the detector that produces the boxes");
        }

        spec.nodes.push_back(parsed);
    }

    const json& edges = RequireArray(root, "edges", origin, kNoDetail);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const json& edge = edges[i];
        std::ostringstream detail;
        detail << "edge[" << i << "]";
        RequireObject(edge, origin, detail.str());

        // Same order as a node, and for the same reason: "form" for
        // "from" must be answered as the typo it is, not as a bare
        // "missing required key".
        const std::string edge_label = EdgeLabel(edge, i);
        RejectUnknownKeys(edge, kEdgeKeys, origin, edge_label);

        EdgeSpec parsed;
        parsed.from = RequireString(edge, "from", origin, edge_label,
                                    "add \"from\": the id of the producing node");
        parsed.to = RequireString(edge, "to", origin, edge_label,
                                  "add \"to\": the id of the consuming node");
        parsed.roi = ParseRoi(edge, origin, edge_label);
        json::const_iterator port = edge.find("port");
        if (port != edge.end()) {
            if (!port->is_string() || port->get<std::string>().empty()) {
                throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, edge_label),
                                 "\"port\" must be a non-empty string, got " + port->dump(),
                                 "\"port\" names one of the producer's outputs; "
                                 "--list-models shows them");
            }
            parsed.port = port->get<std::string>();
        }
        json::const_iterator carry = edge.find("carry");
        if (carry != edge.end()) {
            if (!carry->is_string() || carry->get<std::string>() != "result") {
                throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, edge_label),
                                 "\"carry\" must be \"result\"",
                                 "a result edge hands the producer payload to a cpu node");
            }
            parsed.carry_result = true;
        }
        if (parsed.carry_result && parsed.roi.present) {
            throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, edge_label),
                             "a result edge cannot also crop",
                             "remove \"roi\" from this edge");
        }
        if (parsed.carry_result && !parsed.port.empty()) {
            throw GraphError(GraphErrorCode::kGraphSchema, Where(origin, edge_label),
                             "a result edge carries the primary payload",
                             "remove \"port\" from this edge");
        }
        spec.edges.push_back(parsed);
    }

    return spec;
}

GraphSpec ParseGraphFile(const std::string& path) {
    std::ifstream stream(path.c_str());
    if (!stream.is_open()) {
        throw GraphError(GraphErrorCode::kGraphSchema, path,
                         "cannot open the graph file",
                         "check the path and read permissions");
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return ParseGraphText(buffer.str(), path);
}

namespace {

struct ResolvedNode {
    const NodeSpec* spec;
    const ModelInfo* info;  ///< NULL for a source
    Shape output_shape;
};

void RequireKnownEndpoint(const GraphSpec& spec, const std::string& id,
                          const std::string& edge_label) {
    if (spec.FindNode(id) == NULL) {
        throw GraphError(GraphErrorCode::kGraphSchema, edge_label,
                         "unknown node id \"" + id + "\"",
                         "every \"from\" and \"to\" must name a node in \"nodes\"");
    }
}

std::string EdgeLabel(const EdgeSpec& edge) {
    return "edge \"" + edge.from + "\"->\"" + edge.to + "\"";
}

const PortInfo* FindPort(const ModelInfo& info, const std::string& name) {
    for (std::size_t i = 0; i < info.ports.size(); ++i) {
        if (info.ports[i].name == name) return &info.ports[i];
    }
    return NULL;
}

/// "boxes, drivable, lane": the primary's name first, then the declared ports.
std::string OutputNames(const ModelInfo& info) {
    std::string names = ToString(info.output_shape);
    for (std::size_t i = 0; i < info.ports.size(); ++i) names += ", " + info.ports[i].name;
    return names;
}

const ParamInfo* FindParam(const ModelInfo& info, const std::string& name) {
    for (std::size_t i = 0; i < info.text_params.size(); ++i) {
        if (info.text_params[i].name == name) return &info.text_params[i];
    }
    return NULL;
}

/// The how of a non-number for a key the model reads as a number.
std::string NumbersOnlyHow(const ModelInfo& info) {
    if (info.text_params.empty()) {
        return "model " + Quote(info.model_name) + " reads only numbers";
    }
    std::string names;
    for (std::size_t i = 0; i < info.text_params.size(); ++i) {
        if (i != 0) names += ", ";
        names += info.text_params[i].name;
    }
    return "model " + Quote(info.model_name) + " reads text only for: " + names;
}

/// A params number as the user wrote it. By now it has been through a
/// double, and json(5.0) prints "5.0"; a whole number prints as "5".
json AsWritten(double value) {
    if (std::isfinite(value) && value == std::floor(value) && std::fabs(value) < 1e15) {
        return json(static_cast<long long>(value));
    }
    return json(value);
}

/**
 * @brief U-62: every params value against the kind this model reads it as.
 *
 * Numeric keys first, then text, then lists, each in key order. A how
 * shared by several violations is printed once.
 */
void CheckParams(const StageParams& params, const ModelInfo& info, Violations* bad) {
    const std::string model = Quote(info.model_name);
    std::set<std::string> hows;
    const auto add = [bad, &hows](const std::string& what, const std::string& how) {
        bad->Add(what, hows.insert(how).second ? how : std::string());
    };
    const auto wrong_kind = [&model](const std::string& key, const ParamInfo& param,
                                     const json& value) {
        return Quote(key) +
               (param.kind == ParamInfo::kTextList ? " must be an array of strings"
                                                   : " must be a string") +
               " for model " + model + ", got " + value.dump();
    };

    for (std::map<std::string, double>::const_iterator it = params.numeric.begin();
         it != params.numeric.end(); ++it) {
        const ParamInfo* param = FindParam(info, it->first);
        if (param != NULL) add(wrong_kind(it->first, *param, AsWritten(it->second)), "");
    }
    for (std::map<std::string, std::string>::const_iterator it = params.text.begin();
         it != params.text.end(); ++it) {
        const ParamInfo* param = FindParam(info, it->first);
        if (param == NULL) {
            add(Quote(it->first) + " must be a number, got " + json(it->second).dump(),
                NumbersOnlyHow(info));
        } else if (param->kind != ParamInfo::kText) {
            add(wrong_kind(it->first, *param, json(it->second)), "");
        }
    }
    for (std::map<std::string, std::vector<std::string> >::const_iterator it =
             params.lists.begin();
         it != params.lists.end(); ++it) {
        const ParamInfo* param = FindParam(info, it->first);
        if (param == NULL) {
            add(Quote(it->first) + " must be a number, got " + json(it->second).dump(),
                NumbersOnlyHow(info));
        } else if (param->kind != ParamInfo::kTextList) {
            add(wrong_kind(it->first, *param, json(it->second)), "");
        }
    }
}

}  // namespace

std::vector<StreamSpec> ListStreams(const GraphSpec& spec) {
    std::map<std::string, std::vector<std::string> > successors;
    for (std::size_t e = 0; e < spec.edges.size(); ++e) {
        successors[spec.edges[e].from].push_back(spec.edges[e].to);
    }
    std::vector<StreamSpec> streams;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        if (!spec.nodes[i].is_source) continue;
        std::set<std::string> reached;
        reached.insert(spec.nodes[i].id);
        std::vector<std::string> frontier(1, spec.nodes[i].id);
        while (!frontier.empty()) {
            const std::string current = frontier.back();
            frontier.pop_back();
            const std::vector<std::string>& next = successors[current];
            for (std::size_t k = 0; k < next.size(); ++k) {
                if (reached.insert(next[k]).second) frontier.push_back(next[k]);
            }
        }
        StreamSpec stream;
        stream.source = spec.nodes[i].id;
        for (std::size_t k = 0; k < spec.nodes.size(); ++k) {
            if (reached.count(spec.nodes[k].id) != 0) stream.nodes.push_back(spec.nodes[k].id);
        }
        streams.push_back(stream);
    }
    return streams;
}

bool WholeInteger(const std::string& text) {
    if (text.empty()) return false;
    const char* begin = text.c_str();
    char* end = NULL;
    errno = 0;
    const long parsed = std::strtol(begin, &end, 10);
    if (end == begin || *end != '\0' || errno == ERANGE) return false;
    return parsed >= static_cast<long>(std::numeric_limits<int>::min()) &&
           parsed <= static_cast<long>(std::numeric_limits<int>::max());
}

void CheckCpuParams(const NodeSpec& node, Violations* bad) {
    if (node.op == "headpose") {
        if (!node.params.numeric.empty() || !node.params.text.empty() ||
            !node.params.lists.empty()) {
            bad->Add("headpose takes no params", "remove \"params\"");
        }
        return;
    }
    for (std::map<std::string, double>::const_iterator it = node.params.numeric.begin();
         it != node.params.numeric.end(); ++it) {
        if (it->first != "scale_factor") {
            bad->Add("unknown param " + Quote(it->first),
                     "volume accepts \"scale_factor\", \"classes\" and \"class_ids\"");
            continue;
        }
        if (!std::isfinite(it->second) || it->second < 0.0) {
            bad->Add("\"scale_factor\" must be a finite number >= 0",
                     "1.0 leaves the proxy in mask-pixels times depth");
        }
    }
    for (std::map<std::string, std::string>::const_iterator it = node.params.text.begin();
         it != node.params.text.end(); ++it) {
        const std::string hint = it->first == "classes" || it->first == "class_ids"
            ? "write " + Quote(it->first) + " as an array of strings"
            : "volume accepts \"scale_factor\", \"classes\" and \"class_ids\"";
        bad->Add("unknown param " + Quote(it->first), hint);
    }
    for (std::map<std::string, std::vector<std::string> >::const_iterator it =
             node.params.lists.begin();
         it != node.params.lists.end(); ++it) {
        if (it->first != "classes" && it->first != "class_ids") {
            bad->Add("unknown param " + Quote(it->first),
                     "volume accepts \"scale_factor\", \"classes\" and \"class_ids\"");
            continue;
        }
        if (it->first != "class_ids") continue;
        for (std::size_t i = 0; i < it->second.size(); ++i) {
            if (!WholeInteger(it->second[i])) {
                bad->Add("\"class_ids\"[" + std::to_string(i) + "] must be an integer",
                         "write class ids as strings, for example \"24\"");
            }
        }
    }
}

void CheckCpuInputs(const NodeSpec& node, const std::vector<Shape>& got) {
    std::map<Shape, int> count;
    std::string listed;
    for (std::size_t i = 0; i < got.size(); ++i) {
        count[got[i]] += 1;
        if (!listed.empty()) listed += ", ";
        listed += ToString(got[i]);
    }
    if (listed.empty()) listed = "nothing";
    const std::string where = "node \"" + node.id + "\"";
    if (node.op == "headpose") {
        if (got.size() == 1 && got[0] == Shape::kBoxes) return;
        throw GraphError(GraphErrorCode::kGraphEdge, where,
                         "headpose needs one boxes result, got " + listed,
                         "connect one detector with \"carry\": \"result\"");
    }
    const bool volume_ok = got.size() == 3 && count[Shape::kBoxes] == 1 &&
                           count[Shape::kInstances] == 1 && count[Shape::kDenseMap] == 1;
    if (volume_ok) return;
    throw GraphError(GraphErrorCode::kGraphEdge, where,
                     "volume needs one boxes, one instances and one densemap result, got " +
                         listed,
                     "connect a detector, a segmenter and a depth model with "
                     "\"carry\": \"result\"");
}

std::map<std::string, std::string> SharedEngineOwners(const GraphSpec& spec,
                                                      const IModelRegistry& registry) {
    std::map<std::string, std::string> owner_of;
    std::map<std::string, std::string> first_of_variant;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        if (node.is_source || node.is_cpu || !node.engine.share) continue;
        const ModelInfo* info = registry.find(node.model);
        if (info == NULL) continue;
        std::map<std::string, std::string>::iterator first =
            first_of_variant.find(info->model_name);
        if (first == first_of_variant.end()) {
            first = first_of_variant.insert(std::make_pair(info->model_name, node.id)).first;
        }
        owner_of[node.id] = first->second;
    }
    return owner_of;
}

namespace {

bool SameParams(const StageParams& a, const StageParams& b) {
    return a.numeric == b.numeric && a.text == b.text && a.lists == b.lists;
}

std::string BuffersText(int buffers) {
    return buffers == 0 ? std::string("the default") : std::to_string(buffers);
}

}  // namespace

void ValidateGraph(const GraphSpec& spec, const IModelRegistry& registry) {
    // 1. Resolve every node against the registry.
    std::map<std::string, ResolvedNode> resolved;
    bool has_source = false;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        ResolvedNode entry;
        entry.spec = &node;
        entry.info = NULL;

        if (node.is_source) {
            has_source = true;
            entry.output_shape = Shape::kFrame;
        } else if (node.is_cpu) {
            entry.output_shape = Shape::kRecords;
        } else {
            const ModelInfo* info = registry.find(node.model);
            if (info == NULL) {
                // The hint is always a model key, but it is found among
                // every spelling a user may arrive with: the keys, their
                // aliases (R6) and the model zoo's names. Nothing is
                // substituted; the node still fails here.
                const std::vector<ModelInfo> all = registry.list();
                const std::vector<ModelAlias> aliases = registry.aliases();
                std::vector<std::string> names;
                std::vector<std::string> keys;
                for (std::size_t m = 0; m < all.size(); ++m) {
                    names.push_back(all[m].model_name);
                    keys.push_back(all[m].model_name);
                }
                for (std::size_t a = 0; a < aliases.size(); ++a) {
                    names.push_back(aliases[a].name);
                    keys.push_back(aliases[a].variant);
                }
                for (std::size_t m = 0; m < all.size(); ++m) {
                    if (all[m].download_name.empty()) continue;
                    names.push_back(all[m].download_name);
                    keys.push_back(all[m].model_name);
                }
                const std::string nearest_name = NearestName(node.model, names);
                std::string nearest;
                for (std::size_t n = 0; n < names.size() && nearest.empty(); ++n) {
                    if (names[n] == nearest_name) nearest = keys[n];
                }
                std::string how;
                if (!nearest.empty()) {
                    how = "did you mean \"" + nearest + "\"? ";
                }
                how += "run --list-models, or see docs/graph_models.md";
                throw GraphError(
                    GraphErrorCode::kModelUnknown, "node \"" + node.id + "\"",
                    // Split across adjacent string literals so the boundary
                    // guard's CONCRETE_REGISTRY_SNAKE pattern (any raw
                    // "*_registry" text, scanned deliberately unstripped of
                    // strings/comments) does not false-positive on this
                    // config file name, which is not a concrete registry
                    // symbol. Adjacent literal concatenation inserts no
                    // characters, so the compiled message text is unchanged
                    // from the unsplit form.
                    "unknown model \"" + node.model +
                        "\" - not in config/model" "_registry.json",
                    how);
            }
            if (info->task.empty()) {
                throw GraphError(
                    GraphErrorCode::kModelNoTask, "node \"" + node.id + "\"",
                    "model \"" + node.model + "\" has no task registered",
                    "./scripts/add_model.sh --model " + node.model +
                        " --task <task>");
            }
            if (!info->ready) {
                throw GraphError(
                    GraphErrorCode::kModelNotReady, "node \"" + node.id + "\"",
                    "model \"" + node.model + "\" is registered but cannot run: " +
                        info->not_ready_reason,
                    "see docs/graph_models.md for models usable in graphs");
            }
            if (info->output_shape == Shape::kBoxes3d) {
                throw GraphError(
                    GraphErrorCode::kGraphEdge, "node \"" + node.id + "\"",
                    "model \"" + node.model +
                        "\" consumes LiDAR point clouds, not camera frames",
                    "3D detection cannot be connected to a camera graph");
            }
            entry.info = info;
            entry.output_shape = info->output_shape;
        }
        resolved[node.id] = entry;
    }

    if (!has_source) {
        throw GraphError(GraphErrorCode::kGraphOrphan, spec.name.empty()
                             ? std::string("graph")
                             : "graph \"" + spec.name + "\"",
                         "no source node",
                         "add a node with \"type\": \"source\" and a \"uri\"");
    }

    // 1b. U-62: each params value against the kind this model reads it as.
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const NodeSpec& node = spec.nodes[i];
        if (node.is_source) continue;
        if (node.is_cpu) {
            Violations cpu_bad;
            CheckCpuParams(node, &cpu_bad);
            cpu_bad.ThrowIfAny("node \"" + node.id + "\" \"params\"");
            continue;
        }
        const ModelInfo& info = *resolved[node.id].info;
        Violations bad;
        CheckParams(node.params, info, &bad);
        bad.ThrowIfAny("node \"" + node.id + "\" \"params\"");
    }

    // 2. Edge endpoints, duplicates, and source in-degree.
    std::set<std::string> edge_keys;
    for (std::size_t i = 0; i < spec.edges.size(); ++i) {
        const EdgeSpec& edge = spec.edges[i];
        const std::string label = EdgeLabel(edge);
        RequireKnownEndpoint(spec, edge.from, label);
        RequireKnownEndpoint(spec, edge.to, label);

        const std::string key = edge.from + "\x1f" + edge.to;
        if (!edge_keys.insert(key).second) {
            throw GraphError(
                GraphErrorCode::kGraphSchema, label,
                "duplicate edge \"" + edge.from + "\" -> \"" + edge.to + "\"",
                "declare the connection once; add a second consumer node "
                "instead of a second edge");
        }

        if (resolved[edge.to].spec->is_source) {
            throw GraphError(
                GraphErrorCode::kGraphEdge, label,
                "\"" + edge.to + "\" is a source and cannot receive input",
                "sources have no incoming edges");
        }
    }

    // 3. Acyclicity, by Kahn's algorithm on node ids. Checked before shape
    // compatibility (step 4): a cycle makes "producer" and "consumer" an
    // ill-defined pair for every edge in the loop, so reporting the
    // structural defect first avoids an arbitrary/misleading shape error
    // surfacing instead of the cycle that actually breaks the graph.
    std::map<std::string, int> indegree;
    std::map<std::string, std::vector<std::string> > successors;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        indegree[spec.nodes[i].id] = 0;
    }
    for (std::size_t i = 0; i < spec.edges.size(); ++i) {
        successors[spec.edges[i].from].push_back(spec.edges[i].to);
        indegree[spec.edges[i].to] += 1;
    }

    std::vector<std::string> ready;
    for (std::map<std::string, int>::const_iterator it = indegree.begin();
         it != indegree.end(); ++it) {
        if (it->second == 0) ready.push_back(it->first);
    }
    std::size_t visited = 0;
    while (!ready.empty()) {
        const std::string current = ready.back();
        ready.pop_back();
        ++visited;
        const std::vector<std::string>& next = successors[current];
        for (std::size_t i = 0; i < next.size(); ++i) {
            if (--indegree[next[i]] == 0) ready.push_back(next[i]);
        }
    }
    if (visited != spec.nodes.size()) {
        std::string stuck;
        for (std::map<std::string, int>::const_iterator it = indegree.begin();
             it != indegree.end(); ++it) {
            if (it->second > 0) {
                if (!stuck.empty()) stuck += ", ";
                stuck += it->first;
            }
        }
        throw GraphError(GraphErrorCode::kGraphCycle, "graph",
                         "cycle detected among: " + stuck,
                         "a graph must be acyclic");
    }

    // has_roi_in/has_frame_in are needed both by the plain-edge branch below
    // (a per-crop node cannot hand off an image on a plain edge) and by step
    // 4b (a node fed both a frame and crops of that frame), so they are
    // computed once, before step 4's loop, and reused by both.
    std::map<std::string, bool> has_frame_in;
    std::map<std::string, bool> has_roi_in;
    for (std::size_t i = 0; i < spec.edges.size(); ++i) {
        const EdgeSpec& edge = spec.edges[i];
        if (edge.carry_result) continue;
        if (edge.roi.present) {
            has_roi_in[edge.to] = true;
        } else {
            has_frame_in[edge.to] = true;
        }
    }

    // 4. Shape and input-contract compatibility.
    std::map<std::string, std::vector<Shape> > carried;
    for (std::size_t i = 0; i < spec.edges.size(); ++i) {
        const EdgeSpec& edge = spec.edges[i];
        const std::string label = EdgeLabel(edge);
        const ResolvedNode& producer = resolved[edge.from];
        const ResolvedNode& consumer = resolved[edge.to];

        // U-08: the output this edge reads. Without "port" (or with the
        // primary's own name) it is the primary, and every message below
        // is byte-identical to the text it had before ports existed.
        Shape out_shape = producer.output_shape;
        std::string out_name = "\"" + edge.from + "\"";
        bool primary_output = true;
        if (!edge.port.empty()) {
            if (producer.spec->is_source) {
                throw GraphError(GraphErrorCode::kGraphEdge, label,
                                 "\"" + edge.from + "\" is a source; \"port\" selects one "
                                 "of a model's outputs",
                                 "remove \"port\" from this edge");
            }
            if (producer.info == NULL) {
                throw GraphError(GraphErrorCode::kGraphEdge, label,
                                 "\"" + edge.from + "\" is a cpu node and has no output ports",
                                 "remove \"port\" from this edge");
            }
            if (edge.port != ToString(producer.output_shape)) {
                const PortInfo* port = FindPort(*producer.info, edge.port);
                if (port == NULL) {
                    throw GraphError(GraphErrorCode::kGraphEdge, label,
                                     "\"" + edge.from + "\" has no output port \"" +
                                         edge.port + "\"",
                                     "its outputs: " + OutputNames(*producer.info));
                }
                out_shape = port->shape;
                out_name = "\"" + edge.from + "\" port \"" + edge.port + "\"";
                primary_output = false;
            }
        }

        if (consumer.spec->is_cpu || edge.carry_result) {
            if (!consumer.spec->is_cpu) {
                throw GraphError(GraphErrorCode::kGraphEdge, label,
                                 "\"carry\": \"result\" is accepted only by a cpu node",
                                 "point this edge at a node with \"type\": \"cpu\"");
            }
            if (!edge.carry_result) {
                throw GraphError(GraphErrorCode::kGraphEdge, label,
                                 "a cpu node accepts only \"carry\": \"result\" edges",
                                 "add \"carry\": \"result\" and remove \"roi\" or a plain frame edge");
            }
            if (producer.spec->is_source) {
                throw GraphError(GraphErrorCode::kGraphEdge, label,
                                 "\"" + edge.from + "\" is a source and produces a frame, not a result",
                                 "carry the result of a model node");
            }
            carried[edge.to].push_back(out_shape);
            continue;
        }
        const InputContract contract = consumer.info->input_contract;

        if (edge.roi.present) {
            if (!ProducesRoi(out_shape)) {
                throw GraphError(
                    GraphErrorCode::kGraphEdge, label,
                    out_name + " produces " +
                        ToString(out_shape) + ", not boxes",
                    "remove \"roi\" from this edge, or use a detector as "
                    "the source");
            }
            if (!AcceptsRoi(contract)) {
                throw GraphError(
                    GraphErrorCode::kGraphEdge, label,
                    "\"" + edge.to + "\" requires a full frame",
                    "remove \"roi\" from this edge");
            }
            if (!edge.roi.align.empty() && !primary_output) {
                throw GraphError(
                    GraphErrorCode::kGraphAlign, label,
                    "\"align\": \"face5\" needs five facial landmarks, and port \"" +
                        edge.port + "\" of \"" + edge.from + "\" carries none",
                    "remove \"align\", or crop from \"" + edge.from + "\"'s primary boxes");
            }
            if (!edge.roi.align.empty() && !producer.info->produces_landmarks) {
                // producer.info is never NULL here: a source producer would
                // already have failed the ProducesRoi() check above, since
                // ProducesRoi(kFrame) is false.
                throw GraphError(
                    GraphErrorCode::kGraphAlign, label,
                    "\"align\": \"face5\" needs five facial landmarks, and "
                    "model \"" + producer.info->model_name +
                        "\" produces no landmarks",
                    "remove \"align\", or use a face detector as the source");
            }
        } else {
            if (contract == InputContract::kRoi) {
                throw GraphError(
                    GraphErrorCode::kGraphEdge, label,
                    "\"" + edge.to + "\" requires a cropped region, not a "
                    "full frame",
                    "add \"roi\": {} to this edge and feed it from a detector");
            }
            // A plain edge carries the SOURCE FRAME from a source, or the
            // producer's own image from an image producer (payload hand-off:
            // the consumer runs on that image, and its results map back
            // through origin.inv_align - FrameView, roi_router.hpp).
            // Everything else would discard the producer's output and stays
            // refused. Demonstrated on hardware before hand-off existed:
            // "cam->od, od->seg" and "cam->od, cam->seg" gave "seg" a
            // byte-identical payload digest - both executors silently lost
            // the boxes producer's output, and the parity suite agreed
            // because both sides lost the same data; agreement is not
            // correctness. That reasoning still holds for anything that is
            // not a source and not an image, the same reasoning as the
            // frame+ROI case in step 4b below.
            if (out_shape != Shape::kFrame) {
                if (out_shape == Shape::kImage) {
                    // Payload hand-off: the consumer runs on the producer's
                    // output image, and its results map back through
                    // origin.inv_align (FrameView, roi_router.hpp). Only a
                    // node that runs ONCE per frame can hand off - a per-crop
                    // node would hand off one image per crop, which has no
                    // single full-frame meaning.
                    if (has_roi_in.count(edge.from) != 0) {
                        throw GraphError(
                            GraphErrorCode::kGraphEdge, label,
                            "\"" + edge.from + "\" runs once per crop, so a "
                            "plain edge would hand off one image per crop",
                            "feed \"" + edge.from + "\" from a plain edge, or "
                            "give \"" + edge.to + "\" an \"roi\" edge instead");
                    }
                } else if (ProducesRoi(out_shape)) {
                    throw GraphError(
                        GraphErrorCode::kGraphEdge, label,
                        out_name + " produces " +
                            ToString(out_shape) +
                            ", and a plain edge carries only a source frame "
                            "or an image - \"" + edge.from + "\"'s output "
                            "would be discarded",
                        "add \"roi\": {} to crop \"" + edge.from +
                            "\"'s boxes, or feed \"" + edge.to +
                            "\" from a source node");
                } else {
                    throw GraphError(
                        GraphErrorCode::kGraphEdge, label,
                        out_name + " produces " +
                            ToString(out_shape) +
                            ", and only an image can be handed off on a "
                            "plain edge",
                        "feed \"" + edge.to + "\" from a source node or from "
                        "a node that produces an image");
                }
            }
        }
    }

    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        if (!spec.nodes[i].is_cpu) continue;
        const std::vector<Shape> none;
        std::map<std::string, std::vector<Shape> >::const_iterator got = carried.find(spec.nodes[i].id);
        CheckCpuInputs(spec.nodes[i], got == carried.end() ? none : got->second);
    }

    // 4a. One full-frame input image per node IN EACH STREAM (SP2). A stream
    // is a source and everything reachable from it (ListStreams); a frame of
    // that stream brings a node one input per plain in-edge whose producer
    // is in the stream. Two sources feeding one node are two streams, each
    // bringing it one frame. Two images inside one stream - "cam -> sr ->
    // od" plus "cam -> od" - have no single full-frame meaning for that node
    // to run on, so they are refused rather than one picked arbitrarily (the
    // same reasoning as 4b, for two image inputs instead of a frame and a
    // crop). Consumers are visited in node-id order and producers in edge
    // declaration order, so a one-source graph gets exactly the message it
    // got before streams existed; with several sources it names the stream.
    // One exception: only nodes of some stream are counted, so a graph broken
    // twice - an unreachable node that also takes two images - is now reported
    // by step 5 as GRAPH_ORPHAN (its root cause) instead of GRAPH_EDGE here.
    const std::vector<StreamSpec> streams = ListStreams(spec);
    for (std::size_t s = 0; s < streams.size(); ++s) {
        const std::set<std::string> member(streams[s].nodes.begin(), streams[s].nodes.end());
        std::map<std::string, std::vector<std::string> > plain_from;
        for (std::size_t i = 0; i < spec.edges.size(); ++i) {
            const EdgeSpec& edge = spec.edges[i];
            if (!edge.roi.present && !edge.carry_result && member.count(edge.from) != 0) {
                plain_from[edge.to].push_back(edge.from);
            }
        }
        for (std::map<std::string, std::vector<std::string> >::const_iterator it =
                 plain_from.begin(); it != plain_from.end(); ++it) {
            if (it->second.size() < 2) continue;
            std::string names;
            for (std::size_t k = 0; k < it->second.size(); ++k) {
                names += (k == 0 ? "\"" : ", \"") + it->second[k] + "\"";
            }
            const std::string in_stream = streams.size() > 1
                ? " in stream \"" + streams[s].source + "\"" : std::string();
            throw GraphError(GraphErrorCode::kGraphEdge, "node \"" + it->first + "\"",
                             "receives full-frame input from more than one image" +
                                 in_stream + ": " + names,
                             "keep one plain edge into this node, or split it into one "
                             "node per input");
        }
    }

    // 4b. A node fed BOTH a whole frame and crops of that frame.
    //
    // Nothing above rejects it: each edge is individually legal (a
    // kEither-contract consumer accepts both), and the shape check in step
    // 4 looks at one edge at a time. But there is no defined answer to what
    // such a node should run on. Both executors resolve it the same way -
    // the frame wins and every crop is silently dropped - so sync/async
    // parity holds while the user's data disappears, which is the worst
    // possible combination: the parity suite reports agreement precisely
    // because both sides lose the same crops.
    //
    // Refused rather than given invented semantics. "Run the frame", "run
    // every crop", "run both and merge" are three different graphs, and the
    // user can express whichever they meant by splitting the node in two.
    // (has_frame_in/has_roi_in were computed once, before step 4's loop.)
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        const std::string& id = spec.nodes[i].id;
        if (has_frame_in.count(id) == 0 || has_roi_in.count(id) == 0) continue;
        throw GraphError(
            GraphErrorCode::kGraphEdge, "node \"" + id + "\"",
            "receives both a full frame and ROI crops",
            "split it into two nodes, one for the full-frame input and one "
            "for the ROI input");
    }

    // 5. Reachability from some source.
    std::set<std::string> reachable;
    std::vector<std::string> frontier;
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        if (spec.nodes[i].is_source) {
            reachable.insert(spec.nodes[i].id);
            frontier.push_back(spec.nodes[i].id);
        }
    }
    while (!frontier.empty()) {
        const std::string current = frontier.back();
        frontier.pop_back();
        const std::vector<std::string>& next = successors[current];
        for (std::size_t i = 0; i < next.size(); ++i) {
            if (reachable.insert(next[i]).second) frontier.push_back(next[i]);
        }
    }
    for (std::size_t i = 0; i < spec.nodes.size(); ++i) {
        if (reachable.find(spec.nodes[i].id) == reachable.end()) {
            throw GraphError(
                GraphErrorCode::kGraphOrphan, "node \"" + spec.nodes[i].id + "\"",
                "unreachable from any source",
                "connect it with an edge, or remove it");
        }
    }

    // 6. Shared engines: one engine, one configuration. The factory, the
    // preprocessor and the postprocessor of the one stage are set up from
    // the owner's params, and DX-RT's buffer pool is the owner's.
    const std::map<std::string, std::string> owners = SharedEngineOwners(spec, registry);
    for (std::map<std::string, std::string>::const_iterator it = owners.begin();
         it != owners.end(); ++it) {
        if (it->first == it->second) continue;
        const NodeSpec& node = *spec.FindNode(it->first);
        const NodeSpec& owner = *spec.FindNode(it->second);
        const std::string where = "node \"" + node.id + "\" \"engine\"";
        if (!SameParams(node.params, owner.params)) {
            throw GraphError(GraphErrorCode::kGraphSchema, where,
                             "shares the engine of node \"" + owner.id +
                                 "\", but its \"params\" differ",
                             "one engine runs one configuration: give both nodes the same "
                             "\"params\", or remove \"share\" from one of them");
        }
        if (node.engine.buffers != owner.engine.buffers) {
            throw GraphError(GraphErrorCode::kGraphSchema, where,
                             "shares the engine of node \"" + owner.id +
                                 "\", but its \"buffers\" differ (" +
                                 BuffersText(node.engine.buffers) + " vs " +
                                 BuffersText(owner.engine.buffers) + ")",
                             "set the same \"buffers\" on every node that shares the engine");
        }
    }
}

}  // namespace graph
}  // namespace dxapp
