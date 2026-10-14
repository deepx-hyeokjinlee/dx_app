#include "common/graph/logic_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "common/graph/graph_error.hpp"
#include "common/graph/roi_router.hpp"

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

namespace {

/// A track not seen for this long (stream seconds) is forgotten.
const double kForgetAfterS = 5.0;

double Number(const LogicConfig& c, const char* key) {
    std::map<std::string, double>::const_iterator it = c.numbers.find(key);
    return it == c.numbers.end() ? 0.0 : it->second;
}

std::string Text(const LogicConfig& c, const char* key) {
    std::map<std::string, std::string>::const_iterator it = c.text.find(key);
    return it == c.text.end() ? std::string() : it->second;
}

const std::vector<std::string>& List(const LogicConfig& c, const char* key) {
    static const std::vector<std::string> kNone;
    std::map<std::string, std::vector<std::string> >::const_iterator it = c.lists.find(key);
    return it == c.lists.end() ? kNone : it->second;
}

/// A polygon or line param in source pixels.
std::vector<cv::Point2f> Points(const LogicConfig& c, const char* key, const cv::Size& size) {
    std::vector<cv::Point2f> points;
    std::map<std::string, std::vector<double> >::const_iterator it = c.arrays.find(key);
    if (it == c.arrays.end()) return points;
    for (std::size_t i = 0; i + 1 < it->second.size(); i += 2) {
        points.push_back(cv::Point2f(static_cast<float>(it->second[i] * size.width),
                                     static_cast<float>(it->second[i + 1] * size.height)));
    }
    return points;
}

cv::Point2f Anchor(const cv::Rect2f& box, const std::string& anchor) {
    if (anchor == "center") {
        return cv::Point2f(box.x + box.width * 0.5f, box.y + box.height * 0.5f);
    }
    return cv::Point2f(box.x + box.width * 0.5f, box.y + box.height);  // "bottom"
}

bool Listed(const std::vector<std::string>& classes, const BoxItem& item) {
    return classes.empty() ||
           std::find(classes.begin(), classes.end(), item.class_name) != classes.end();
}

void AddNumber(RecordItem* item, const std::string& name, double value) {
    item->numbers.push_back(std::make_pair(name, value));
}

void AddText(RecordItem* item, const std::string& name, const std::string& value) {
    item->text.push_back(std::make_pair(name, value));
}

GraphEvent MakeEvent(const LogicConfig& c, const std::string& name, int track_id,
                     const LogicFrame& at) {
    GraphEvent event;
    event.node = c.id;
    event.event = name;
    event.track_id = track_id;
    event.time_s = at.time_s;
    return event;
}

/// "3" for a whole number, else two decimals.
std::string Short(double value) {
    char text[64];
    if (std::isfinite(value) && std::fabs(value - std::round(value)) < 1e-9) {
        std::snprintf(text, sizeof(text), "%.0f", value);
    } else {
        std::snprintf(text, sizeof(text), "%.2f", value);
    }
    return text;
}

cv::Point2f Apply(const cv::Matx23f& m, const cv::Point2f& p) {
    return cv::Point2f(m(0, 0) * p.x + m(0, 1) * p.y + m(0, 2),
                       m(1, 0) * p.x + m(1, 1) * p.y + m(1, 2));
}

/// The input's boxes in source pixels; null when it carries no boxes.
std::shared_ptr<BoxesData> SourceBoxes(const LogicInput& input) {
    const BoxesData* boxes =
        input.data ? dynamic_cast<const BoxesData*>(input.data.get()) : NULL;
    if (boxes == NULL) return std::shared_ptr<BoxesData>();
    std::shared_ptr<BoxesData> out(new BoxesData(*boxes));
    if (!input.origin.from_roi && IsIdentityAffine(input.origin.inv_align)) return out;
    for (std::size_t i = 0; i < out->items.size(); ++i) {
        BoxItem& item = out->items[i];
        item.box = RestoreBox(item.box, input.origin);
        for (std::size_t k = 0; k < item.landmarks.size(); ++k) {
            const cv::Point2f p = Apply(input.origin.inv_align,
                                        cv::Point2f(item.landmarks[k].x, item.landmarks[k].y));
            item.landmarks[k].x = p.x;
            item.landmarks[k].y = p.y;
        }
    }
    return out;
}

/// The first full-frame input that carries a payload, or NULL.
const LogicInput* FirstFullFrame(const std::vector<LogicInput>& inputs) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i].per_crop && inputs[i].data) return &inputs[i];
    }
    return NULL;
}

/// Forget every track not in `seen` and not seen for `after` seconds;
/// `on_forget` hears each one first.
template <typename OnForget>
void Forget(LogicState* s, const std::set<int>& seen, double now, double after,
            OnForget on_forget) {
    for (std::map<int, TrackMemo>::iterator it = s->tracks.begin(); it != s->tracks.end();) {
        if (seen.count(it->first) == 0 && now - it->second.last_s > after) {
            on_forget(it->first, it->second);
            it = s->tracks.erase(it);
        } else {
            ++it;
        }
    }
}

LogicOutput RunFilter(const LogicConfig& c, const std::vector<LogicInput>& inputs) {
    LogicOutput out;
    const LogicInput* input = FirstFullFrame(inputs);
    const std::shared_ptr<BoxesData> boxes =
        input == NULL ? std::shared_ptr<BoxesData>() : SourceBoxes(*input);
    if (!boxes) {
        out.result.data.reset(new BoxesData(Shape::kBoxes));
        out.error = "missing boxes input";
        return out;
    }
    const std::vector<std::string>& classes = List(c, "classes");
    const float min_score = static_cast<float>(Number(c, "min_score"));
    const double min_area = Number(c, "min_area");
    std::shared_ptr<BoxesData> kept(new BoxesData(boxes->box_shape));
    for (std::size_t i = 0; i < boxes->items.size(); ++i) {
        const BoxItem& item = boxes->items[i];
        if (!Listed(classes, item) || item.score < min_score) continue;
        if (static_cast<double>(item.box.width) * item.box.height < min_area) continue;
        kept->items.push_back(item);
    }
    out.result.data = kept;
    return out;
}

LogicOutput RunZone(const LogicConfig& c, const std::vector<LogicInput>& inputs,
                    const LogicFrame& at, LogicState* s) {
    LogicOutput out;
    const std::vector<cv::Point2f> polygon = Points(c, "polygon", at.frame_size);
    out.overlay.polygons.push_back(polygon);
    const LogicInput* input = FirstFullFrame(inputs);
    const std::shared_ptr<BoxesData> boxes =
        input == NULL ? std::shared_ptr<BoxesData>() : SourceBoxes(*input);
    if (!boxes) out.error = "missing boxes input";
    std::shared_ptr<BoxesData> kept(new BoxesData(boxes ? boxes->box_shape : Shape::kBoxes));
    const bool want_inside = Text(c, "mode") != "outside";
    const std::string anchor = Text(c, "anchor");
    std::set<int> seen;
    std::vector<std::pair<std::string, int> > changes;  // (enter|exit, track), in box order
    if (boxes && polygon.size() >= 3) {
        for (std::size_t i = 0; i < boxes->items.size(); ++i) {
            const BoxItem& item = boxes->items[i];
            const cv::Point2f p = Anchor(item.box, anchor);
            const bool inside = cv::pointPolygonTest(polygon, p, false) >= 0.0;
            const bool selected = inside == want_inside;
            if (selected) {
                kept->items.push_back(item);
                out.overlay.highlights.push_back(item.box);
            }
            if (item.track_id < 0) continue;
            seen.insert(item.track_id);
            TrackMemo& memo = s->tracks[item.track_id];
            if (selected && !memo.inside) changes.push_back(std::make_pair("enter", item.track_id));
            if (!selected && memo.inside) changes.push_back(std::make_pair("exit", item.track_id));
            memo.inside = selected;
            memo.point = p;
            memo.last_s = at.time_s;
        }
    }
    Forget(s, seen, at.time_s, kForgetAfterS, [&changes](int track, const TrackMemo& memo) {
        if (memo.inside) changes.push_back(std::make_pair("exit", track));
    });
    for (std::size_t i = 0; i < changes.size(); ++i) {
        GraphEvent event = MakeEvent(c, changes[i].first, changes[i].second, at);
        AddNumber(&event.data, "count", static_cast<double>(kept->items.size()));
        out.events.push_back(event);
    }
    out.overlay.alert = !kept->items.empty();
    OverlayText label;
    if (!polygon.empty()) label.at = polygon[0];
    label.text = c.id + ": " + std::to_string(kept->items.size());
    label.alert = out.overlay.alert;
    out.overlay.texts.push_back(label);
    out.result.data = kept;
    return out;
}

/// +1 when p is right of A->B on the image (y down), -1 left, 0 on the line.
int Side(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& p) {
    const double cross = static_cast<double>(b.x - a.x) * (p.y - a.y) -
                         static_cast<double>(b.y - a.y) * (p.x - a.x);
    return cross > 0.0 ? 1 : (cross < 0.0 ? -1 : 0);
}

/// The line through a movement p1 -> p2 meets segment A-B (endpoints included).
bool MovementMeetsSegment(const cv::Point2f& p1, const cv::Point2f& p2,
                          const cv::Point2f& a, const cv::Point2f& b) {
    return Side(p1, p2, a) * Side(p1, p2, b) <= 0;
}

LogicOutput RunLine(const LogicConfig& c, const std::vector<LogicInput>& inputs,
                    const LogicFrame& at, LogicState* s) {
    LogicOutput out;
    const std::vector<cv::Point2f> line = Points(c, "line", at.frame_size);
    if (line.size() == 2) out.overlay.lines.push_back(std::make_pair(line[0], line[1]));
    const LogicInput* input = FirstFullFrame(inputs);
    const std::shared_ptr<BoxesData> boxes =
        input == NULL ? std::shared_ptr<BoxesData>() : SourceBoxes(*input);
    if (!boxes) out.error = "missing boxes input";
    const std::string anchor = Text(c, "anchor");
    const std::string direction = Text(c, "direction");
    std::set<int> seen;
    if (boxes && line.size() == 2) {
        for (std::size_t i = 0; i < boxes->items.size(); ++i) {
            const BoxItem& item = boxes->items[i];
            if (item.track_id < 0) continue;
            seen.insert(item.track_id);
            const cv::Point2f p = Anchor(item.box, anchor);
            const int side = Side(line[0], line[1], p);
            std::map<int, TrackMemo>::iterator found = s->tracks.find(item.track_id);
            if (found == s->tracks.end()) {
                TrackMemo memo;
                memo.point = p;
                memo.side = side;
                memo.last_s = at.time_s;
                s->tracks[item.track_id] = memo;
                continue;
            }
            TrackMemo& memo = found->second;
            if (side != 0 && memo.side != 0 && side != memo.side &&
                MovementMeetsSegment(memo.point, p, line[0], line[1])) {
                const bool inward = memo.side < 0;  // from the left of A->B to its right
                if (inward) {
                    ++s->count_in;
                } else {
                    ++s->count_out;
                }
                if (direction == "both" || direction == (inward ? "in" : "out")) {
                    GraphEvent event = MakeEvent(c, "cross", item.track_id, at);
                    AddText(&event.data, "direction", inward ? "in" : "out");
                    AddNumber(&event.data, "in", static_cast<double>(s->count_in));
                    AddNumber(&event.data, "out", static_cast<double>(s->count_out));
                    out.events.push_back(event);
                }
            }
            if (side != 0) memo.side = side;  // on the line: keep the side it came from
            memo.point = p;
            memo.last_s = at.time_s;
        }
    }
    Forget(s, seen, at.time_s, kForgetAfterS, [](int, const TrackMemo&) {});
    std::shared_ptr<RecordsData> records(new RecordsData());
    RecordItem totals;
    AddNumber(&totals, "in", static_cast<double>(s->count_in));
    AddNumber(&totals, "out", static_cast<double>(s->count_out));
    AddNumber(&totals, "total", static_cast<double>(s->count_in + s->count_out));
    records->items.push_back(totals);
    out.result.data = records;
    if (line.size() == 2) {
        OverlayText label;
        label.at = (line[0] + line[1]) * 0.5f;
        label.text = c.id + " in " + std::to_string(s->count_in) + " out " +
                     std::to_string(s->count_out);
        label.alert = !out.events.empty();
        out.overlay.texts.push_back(label);
        out.overlay.alert = !out.events.empty();
    }
    return out;
}

LogicOutput RunCount(const LogicConfig& c, const std::vector<LogicInput>& inputs) {
    LogicOutput out;
    const std::vector<std::string>& classes = List(c, "classes");
    std::shared_ptr<RecordsData> records(new RecordsData());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].per_crop) continue;
        const BoxesData* boxes =
            inputs[i].data ? dynamic_cast<const BoxesData*>(inputs[i].data.get()) : NULL;
        std::size_t total = 0;
        std::vector<std::size_t> per_class(classes.size(), 0);
        if (boxes != NULL) {
            for (std::size_t b = 0; b < boxes->items.size(); ++b) {
                if (!Listed(classes, boxes->items[b])) continue;
                ++total;
                for (std::size_t k = 0; k < classes.size(); ++k) {
                    if (classes[k] == boxes->items[b].class_name) ++per_class[k];
                }
            }
        }
        RecordItem item;
        AddNumber(&item, "total", static_cast<double>(total));
        for (std::size_t k = 0; k < classes.size(); ++k) {
            AddNumber(&item, classes[k], static_cast<double>(per_class[k]));
        }
        AddText(&item, "node", inputs[i].from);
        records->items.push_back(item);
        OverlayText label;  // a line in the top-left panel
        label.text = c.id + " " + inputs[i].from + ": " + std::to_string(total);
        out.overlay.texts.push_back(label);
    }
    if (inputs.empty()) out.error = "missing boxes input";
    out.result.data = records;
    return out;
}

/// The op's own run; RunLogic sorts the inputs and checks the events.
LogicOutput RunOp(const LogicConfig& config, const std::vector<LogicInput>& inputs,
                  const LogicFrame& frame, LogicState* s) {
    if (config.op == "filter") return RunFilter(config, inputs);
    if (config.op == "zone") return RunZone(config, inputs, frame, s);
    if (config.op == "line") return RunLine(config, inputs, frame, s);
    if (config.op == "count") return RunCount(config, inputs);
    LogicOutput out;
    out.result.data.reset(new RecordsData());
    out.error = "logic op \"" + config.op + "\" cannot run";
    return out;
}

}  // namespace

LogicConfig MakeLogicConfig(const NodeSpec& node) {
    const LogicOpInfo* info = FindLogicOp(node.op);
    if (info == NULL) {
        throw std::invalid_argument("MakeLogicConfig: \"" + node.op + "\" is not a logic op");
    }
    LogicConfig config;
    config.id = node.id;
    config.op = node.op;
    for (std::size_t p = 0; p < info->params.size(); ++p) {
        const LogicParamInfo& param = info->params[p];
        switch (param.kind) {
            case LogicParamInfo::kNumber: {
                std::map<std::string, double>::const_iterator it =
                    node.params.numeric.find(param.name);
                config.numbers[param.name] =
                    it != node.params.numeric.end() ? it->second : param.number_default;
                break;
            }
            case LogicParamInfo::kText:
            case LogicParamInfo::kChoice: {
                std::map<std::string, std::string>::const_iterator it =
                    node.params.text.find(param.name);
                config.text[param.name] =
                    it != node.params.text.end() ? it->second : param.text_default;
                break;
            }
            case LogicParamInfo::kTextList: {
                std::map<std::string, std::vector<std::string> >::const_iterator it =
                    node.params.lists.find(param.name);
                if (it != node.params.lists.end()) config.lists[param.name] = it->second;
                break;
            }
            case LogicParamInfo::kPolygon:
            case LogicParamInfo::kLine: {
                std::map<std::string, std::vector<double> >::const_iterator it =
                    node.arrays.find(param.name);
                if (it != node.arrays.end()) config.arrays[param.name] = it->second;
                break;
            }
        }
    }
    return config;
}

std::string DropRepeatedKeys(RecordItem* data) {
    std::set<std::string> seen;
    std::string first;
    std::vector<std::pair<std::string, double> > numbers;
    for (std::size_t i = 0; i < data->numbers.size(); ++i) {
        if (seen.insert(data->numbers[i].first).second) {
            numbers.push_back(data->numbers[i]);
        } else if (first.empty()) {
            first = data->numbers[i].first;
        }
    }
    std::vector<std::pair<std::string, std::string> > text;
    for (std::size_t i = 0; i < data->text.size(); ++i) {
        if (seen.insert(data->text[i].first).second) {
            text.push_back(data->text[i]);
        } else if (first.empty()) {
            first = data->text[i].first;
        }
    }
    data->numbers.swap(numbers);
    data->text.swap(text);
    return first;
}

LogicOutput RunLogic(const LogicConfig& config, std::vector<LogicInput> inputs,
                     const LogicFrame& frame, LogicState* state) {
    std::stable_sort(inputs.begin(), inputs.end(),
                     [](const LogicInput& a, const LogicInput& b) { return a.edge < b.edge; });
    LogicState scratch;
    LogicOutput out = RunOp(config, inputs, frame, state != NULL ? state : &scratch);
    for (std::size_t e = 0; e < out.events.size(); ++e) {
        const std::string repeated = DropRepeatedKeys(&out.events[e].data);
        if (!repeated.empty() && out.error.empty()) {
            out.error = "event \"" + out.events[e].event + "\" repeats the data key \"" +
                        repeated + "\"";
        }
    }
    return out;
}

}  // namespace graph
}  // namespace dxapp
