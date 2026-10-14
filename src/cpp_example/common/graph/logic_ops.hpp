/**
 * @file logic_ops.hpp
 * @brief Logic nodes ("type": "logic"): the decisions around the models.
 *
 * A logic node runs on the CPU once per frame, after the nodes it reads
 * ("carry": "result" edges). Its op is one of eight: filter, zone, line,
 * count, dwell, condition, gallery, user. LogicOps() is the one description
 * of them: ValidateGraph checks params and inputs against it and
 * --list-ops prints it, so the two cannot disagree. headpose and volume are
 * cpu nodes (cpu_reduce.hpp); CpuOpInfos() describes them the same way for
 * --list-ops only.
 *
 * Coordinates in params are fractions of the source frame (0..1), so one
 * graph fits any resolution.
 */
#ifndef DXAPP_GRAPH_LOGIC_OPS_HPP
#define DXAPP_GRAPH_LOGIC_OPS_HPP

#include <string>
#include <vector>

#include "common/graph/shape.hpp"

namespace dxapp {
namespace graph {

/// A number param with no bound on that side (--list-ops leaves the bound out).
const double kLogicUnbounded = 1e9;

/// One param of a logic op: what --list-ops prints and ValidateGraph enforces.
struct LogicParamInfo {
    enum Kind {
        kNumber,    ///< a JSON number in [low, high] ((low, high] when low_open)
        kText,      ///< a string
        kChoice,    ///< one of `choices`
        kTextList,  ///< an array of strings
        kPolygon,   ///< [x0, y0, x1, y1, ...], at least 3 points, each 0..1
        kLine       ///< [x0, y0, x1, y1], two distinct points, each 0..1
    };
    std::string name;
    Kind kind;
    bool required;
    double low;
    double high;
    bool low_open;
    double number_default;              ///< kNumber, when not required
    std::string text_default;           ///< kText / kChoice, when not required
    std::vector<std::string> choices;   ///< kChoice
    std::string help;

    LogicParamInfo()
        : kind(kNumber), required(false), low(0.0), high(0.0), low_open(false),
          number_default(0.0) {}
};

/// One input a logic op accepts: a shape, as a full-frame payload or as
/// the results of a node that runs once per crop.
struct LogicPort {
    Shape shape;
    bool per_crop;
    LogicPort() : shape(Shape::kBoxes), per_crop(false) {}
};

struct LogicOpInfo {
    std::string op;
    std::string summary;
    std::vector<LogicPort> accepts;
    int min_inputs;
    int max_inputs;
    bool same_shape_out;               ///< outputs its input's shape; else records
    bool needs_tracking;               ///< its box inputs must carry track ids
    std::vector<std::string> events;   ///< names it raises ("<event>": the "event" param)
    std::vector<std::string> numbers;  ///< the records numbers it writes
    std::vector<LogicParamInfo> params;

    LogicOpInfo()
        : min_inputs(1), max_inputs(1), same_shape_out(false), needs_tracking(false) {}
};

/// filter, zone, line, count, dwell, condition, gallery, user - in that order.
const std::vector<LogicOpInfo>& LogicOps();
/// NULL when `op` is not a logic op.
const LogicOpInfo* FindLogicOp(const std::string& op);
/// headpose and volume, described the same way (--list-ops, condition values).
const std::vector<LogicOpInfo>& CpuOpInfos();
/// NULL when `op` is not a cpu op.
const LogicOpInfo* FindCpuOpInfo(const std::string& op);
/// "number", "text", "choice", "text_list", "polygon", "line".
const char* ToString(LogicParamInfo::Kind kind);

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_LOGIC_OPS_HPP
