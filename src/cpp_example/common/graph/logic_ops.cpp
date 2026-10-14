#include "common/graph/logic_ops.hpp"

namespace dxapp {
namespace graph {
namespace {

LogicParamInfo Param(const char* name, LogicParamInfo::Kind kind, const char* help) {
    LogicParamInfo param;
    param.name = name;
    param.kind = kind;
    param.help = help;
    return param;
}

LogicParamInfo Number(const char* name, double low, double high, double fallback,
                      const char* help) {
    LogicParamInfo param = Param(name, LogicParamInfo::kNumber, help);
    param.low = low;
    param.high = high;
    param.number_default = fallback;
    return param;
}

/// A number that must be greater than 0.
LogicParamInfo Positive(const char* name, double high, double fallback, const char* help) {
    LogicParamInfo param = Number(name, 0.0, high, fallback, help);
    param.low_open = true;
    return param;
}

LogicParamInfo Text(const char* name, const char* fallback, const char* help) {
    LogicParamInfo param = Param(name, LogicParamInfo::kText, help);
    param.text_default = fallback;
    return param;
}

LogicParamInfo Choice(const char* name, const char* fallback,
                      const std::vector<std::string>& choices, const char* help) {
    LogicParamInfo param = Param(name, LogicParamInfo::kChoice, help);
    param.text_default = fallback;
    param.choices = choices;
    return param;
}

LogicParamInfo Required(LogicParamInfo param) {
    param.required = true;
    param.text_default.clear();
    return param;
}

LogicPort Port(Shape shape, bool per_crop) {
    LogicPort port;
    port.shape = shape;
    port.per_crop = per_crop;
    return port;
}

/// The three box shapes, full frame.
std::vector<LogicPort> BoxPorts() {
    std::vector<LogicPort> ports;
    ports.push_back(Port(Shape::kBoxes, false));
    ports.push_back(Port(Shape::kObBoxes, false));
    ports.push_back(Port(Shape::kInstances, false));
    return ports;
}

std::vector<std::string> Names(const char* const* names, std::size_t count) {
    return std::vector<std::string>(names, names + count);
}

/// Every shape a graph node can hand on, as the user op's "input"/"output" choices.
const Shape kPassable[] = {Shape::kBoxes, Shape::kObBoxes, Shape::kInstances,
                           Shape::kKeypoints, Shape::kLabelMap, Shape::kDenseMap,
                           Shape::kImage, Shape::kScores, Shape::kVector,
                           Shape::kRecords};

LogicParamInfo AnchorParam() {
    const char* const anchors[] = {"bottom", "center"};
    return Choice("anchor", "bottom", Names(anchors, 2),
                  "the point of a box that is tested: its bottom centre (where a person "
                  "stands) or its centre");
}

std::vector<LogicOpInfo> BuildLogicOps() {
    std::vector<LogicOpInfo> ops;

    LogicOpInfo filter;
    filter.op = "filter";
    filter.summary = "keeps the boxes of the listed classes with at least a score and an area";
    filter.accepts = BoxPorts();
    filter.same_shape_out = true;
    filter.params.push_back(Param("classes", LogicParamInfo::kTextList,
        "class names to keep, as the detector names them; leave it out to keep every class"));
    filter.params.push_back(Number("min_score", 0.0, 1.0, 0.0,
        "the smallest confidence kept (0.5, not 50)"));
    filter.params.push_back(Number("min_area", 0.0, kLogicUnbounded, 0.0,
        "the smallest box area kept, in source-frame pixels"));
    ops.push_back(filter);

    LogicOpInfo zone;
    zone.op = "zone";
    zone.summary = "keeps the boxes whose anchor point is inside (or outside) a polygon; "
                   "with track ids, an event when a box enters or leaves";
    zone.accepts = BoxPorts();
    zone.same_shape_out = true;
    zone.events.push_back("enter");
    zone.events.push_back("exit");
    zone.params.push_back(Required(Param("polygon", LogicParamInfo::kPolygon,
        "x0, y0, x1, y1, ...: at least 3 points, each a fraction 0..1 of the frame's "
        "width and height")));
    const char* const modes[] = {"inside", "outside"};
    zone.params.push_back(Choice("mode", "inside", Names(modes, 2),
        "which boxes pass: those whose anchor is inside the polygon, or outside it"));
    zone.params.push_back(AnchorParam());
    ops.push_back(zone);

    LogicOpInfo line;
    line.op = "line";
    line.summary = "counts tracked boxes crossing a line, in each direction, with an event "
                   "per crossing";
    line.accepts = BoxPorts();
    line.needs_tracking = true;
    line.events.push_back("cross");
    line.numbers.push_back("in");
    line.numbers.push_back("out");
    line.numbers.push_back("total");
    line.params.push_back(Required(Param("line", LogicParamInfo::kLine,
        "x0, y0, x1, y1: the line from A to B, each a fraction 0..1 of the frame's width "
        "and height; \"in\" is a crossing from the left of A->B to its right")));
    line.params.push_back(AnchorParam());
    const char* const directions[] = {"both", "in", "out"};
    line.params.push_back(Choice("direction", "both", Names(directions, 3),
        "which crossings raise a \"cross\" event; the counts always keep both"));
    ops.push_back(line);

    LogicOpInfo count;
    count.op = "count";
    count.summary = "counts the boxes of each input (one record per input), in total and "
                    "per listed class";
    count.accepts = BoxPorts();
    count.max_inputs = 8;
    count.numbers.push_back("total");
    count.params.push_back(Param("classes", LogicParamInfo::kTextList,
        "count only these classes, one number each besides \"total\"; leave it out to "
        "count every box"));
    ops.push_back(count);

    LogicOpInfo dwell;
    dwell.op = "dwell";
    dwell.summary = "measures how long each tracked box has stayed, with an event when it "
                    "passes a threshold";
    dwell.accepts = BoxPorts();
    dwell.needs_tracking = true;
    dwell.events.push_back("dwell");
    dwell.numbers.push_back("track_id");
    dwell.numbers.push_back("seconds");
    dwell.params.push_back(Positive("threshold_s", kLogicUnbounded, 10.0,
        "seconds a box must stay before its \"dwell\" event"));
    dwell.params.push_back(Number("grace_s", 0.0, 60.0, 1.0,
        "seconds a box may be missing (an occlusion) and still be the same stay"));
    ops.push_back(dwell);

    LogicOpInfo condition;
    condition.op = "condition";
    condition.summary = "raises an event when a value passes a threshold: a box or crop "
                        "count, a records number, or the crops with or without a label";
    condition.accepts = BoxPorts();
    condition.accepts.push_back(Port(Shape::kRecords, false));
    condition.accepts.push_back(Port(Shape::kScores, true));
    condition.accepts.push_back(Port(Shape::kVector, true));
    condition.events.push_back("<event>");
    condition.numbers.push_back("value");
    condition.numbers.push_back("active");
    condition.params.push_back(Text("value", "count",
        "\"count\" (boxes or crops), a number the input writes (records), or "
        "\"with:<label>\" / \"without:<label>\" (crops with scores)"));
    const char* const comparators[] = {">", ">=", "<", "<=", "==", "!="};
    condition.params.push_back(Choice("comparator", ">", Names(comparators, 6),
        "how the value is compared with \"threshold\""));
    condition.params.push_back(Number("threshold", -kLogicUnbounded, kLogicUnbounded, 0.0,
        "the number the value is compared with"));
    const char* const reductions[] = {"max", "min", "sum"};
    condition.params.push_back(Choice("reduce", "max", Names(reductions, 3),
        "how several records' values become one: their largest, smallest or sum"));
    condition.params.push_back(Number("cooldown_s", 0.0, 86400.0, 0.0,
        "0: one event when the condition becomes true; more: an event at most this "
        "often while it holds"));
    condition.params.push_back(Text("event", "",
        "the event name (letters, digits, _ . : -); leave it out for the node id"));
    ops.push_back(condition);

    LogicOpInfo gallery;
    gallery.op = "gallery";
    gallery.summary = "names each embedding after its nearest gallery entry, or \"unknown\" "
                      "under a threshold, with an event when a track's name changes";
    gallery.accepts.push_back(Port(Shape::kVector, true));
    gallery.accepts.push_back(Port(Shape::kVector, false));
    gallery.events.push_back("match");
    gallery.events.push_back("unknown");
    gallery.numbers.push_back("track_id");
    gallery.numbers.push_back("score");
    gallery.numbers.push_back("parent_index");
    gallery.params.push_back(Required(Param("gallery", LogicParamInfo::kText,
        "a DXGAL1 .bin built from an image folder: python3 scripts/build_gallery_database.py "
        "--variant <model> --images <folder> --out <file>.bin")));
    gallery.params.push_back(Number("threshold", 0.0, 1.0, 0.5,
        "the smallest cosine similarity that names a match"));
    const char* const modes_of_events[] = {"known", "all", "none"};
    gallery.params.push_back(Choice("events", "known", Names(modes_of_events, 3),
        "\"match\" events only, \"unknown\" events too, or none"));
    ops.push_back(gallery);

    LogicOpInfo user;
    user.op = "user";
    user.summary = "your own logic: in a run it passes its input through; an exported "
                   "project gets a function stub to fill in";
    std::vector<std::string> shapes;
    for (std::size_t i = 0; i < sizeof(kPassable) / sizeof(kPassable[0]); ++i) {
        user.accepts.push_back(Port(kPassable[i], false));
        shapes.push_back(ToString(kPassable[i]));
    }
    user.same_shape_out = true;
    user.params.push_back(Required(Param("name", LogicParamInfo::kText,
        "the stub function's name in an exported project: letters, digits and _, not "
        "starting with a digit")));
    user.params.push_back(Required(Choice("input", "", shapes,
        "the kind of data the function receives")));
    user.params.push_back(Required(Choice("output", "", shapes,
        "the kind it returns; in a run the input passes through, so it equals \"input\"")));
    ops.push_back(user);
    return ops;
}

std::vector<LogicOpInfo> BuildCpuOps() {
    std::vector<LogicOpInfo> ops;

    LogicOpInfo headpose;
    headpose.op = "headpose";
    headpose.summary = "head pitch, yaw and roll (degrees) from the largest face's five "
                       "landmarks";
    headpose.accepts.push_back(Port(Shape::kBoxes, false));
    const char* const angles[] = {"pitch", "yaw", "roll"};
    headpose.numbers = Names(angles, 3);
    ops.push_back(headpose);

    LogicOpInfo volume;
    volume.op = "volume";
    volume.summary = "a volume proxy per package box from its instance mask and the depth "
                     "map under it";
    volume.accepts.push_back(Port(Shape::kBoxes, false));
    volume.accepts.push_back(Port(Shape::kInstances, false));
    volume.accepts.push_back(Port(Shape::kDenseMap, false));
    volume.min_inputs = 3;
    volume.max_inputs = 3;
    const char* const measures[] = {"box_x", "box_y", "box_w", "box_h", "class_id",
                                    "mask_area_px", "median_depth", "volume_proxy"};
    volume.numbers = Names(measures, 8);
    volume.params.push_back(Number("scale_factor", 0.0, kLogicUnbounded, 1.0,
        "multiplies the proxy; 1.0 leaves it in mask-pixels times depth"));
    volume.params.push_back(Param("classes", LogicParamInfo::kTextList,
        "package class names; leave it out for the defaults, [] to match none"));
    volume.params.push_back(Param("class_ids", LogicParamInfo::kTextList,
        "package class ids as strings; leave it out for the defaults, [] to match none"));
    ops.push_back(volume);
    return ops;
}

const LogicOpInfo* FindIn(const std::vector<LogicOpInfo>& ops, const std::string& op) {
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (ops[i].op == op) return &ops[i];
    }
    return NULL;
}

}  // namespace

const std::vector<LogicOpInfo>& LogicOps() {
    static const std::vector<LogicOpInfo> ops = BuildLogicOps();
    return ops;
}

const LogicOpInfo* FindLogicOp(const std::string& op) { return FindIn(LogicOps(), op); }

const std::vector<LogicOpInfo>& CpuOpInfos() {
    static const std::vector<LogicOpInfo> ops = BuildCpuOps();
    return ops;
}

const LogicOpInfo* FindCpuOpInfo(const std::string& op) { return FindIn(CpuOpInfos(), op); }

const char* ToString(LogicParamInfo::Kind kind) {
    switch (kind) {
        case LogicParamInfo::kNumber:   return "number";
        case LogicParamInfo::kText:     return "text";
        case LogicParamInfo::kChoice:   return "choice";
        case LogicParamInfo::kTextList: return "text_list";
        case LogicParamInfo::kPolygon:  return "polygon";
        case LogicParamInfo::kLine:     return "line";
    }
    return "unknown";
}

}  // namespace graph
}  // namespace dxapp
