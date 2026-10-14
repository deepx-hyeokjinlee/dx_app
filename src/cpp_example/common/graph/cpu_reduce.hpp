/**
 * @file cpu_reduce.hpp
 * @brief CPU reducers a graph runs after model nodes: head pose and volume.
 *
 * A cpu node does not load a .dxnn. Its inputs arrive on "carry": "result"
 * edges (the producer payloads, not crops), and both executors call ReduceCpu
 * once those parents of the frame have finished.
 */
#ifndef DXAPP_GRAPH_CPU_REDUCE_HPP
#define DXAPP_GRAPH_CPU_REDUCE_HPP

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common/graph/i_registry.hpp"
#include "common/graph/shape.hpp"

namespace dxapp {
namespace graph {

/// One cpu node's output for one frame. `error` is empty when the frame is
/// valid, including "no face" and "no package" (those are empty records).
struct CpuReduction {
    StageResult result;
    std::string error;
};

/**
 * @brief Run `op` ("headpose" or "volume") on `inputs`.
 *
 * `frame_size` is the stream frame. headpose builds a pinhole camera with
 * focal length equal to the frame width. volume ignores it. `inputs` are
 * matched by shape, not by order.
 *
 * headpose records: numbers pitch, yaw, roll (degrees). Empty when the
 * largest face has fewer than five landmarks or solvePnP fails.
 *
 * volume records, one per package box: numbers box_x, box_y, box_w, box_h,
 * class_id, mask_area_px, and median_depth plus volume_proxy when the
 * overlapping instance mask and the depth map share a size and the mask has
 * a finite sample; text class_name.
 */
CpuReduction ReduceCpu(const std::string& op, const StageParams& params,
                       const std::vector<StageDataPtr>& inputs,
                       const cv::Size& frame_size);

/// volume's package classes when its "classes" / "class_ids" are left out
/// (ids as text, as a graph writes them), for --list-ops.
std::vector<std::string> CpuPackageClassNames();
std::vector<std::string> CpuPackageClassIds();

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_CPU_REDUCE_HPP
