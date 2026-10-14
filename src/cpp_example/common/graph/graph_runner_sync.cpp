#include "common/graph/graph_runner_sync.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

#include "common/graph/cpu_reduce.hpp"
#include "common/graph/result_to_shape.hpp"
#include "common/graph/roi_router.hpp"

namespace dxapp {
namespace graph {

FrameReport SyncExecutor::RunFrame(StageGraph& graph, const cv::Mat& frame,
                                   std::size_t frame_index) {
    return RunFrame(graph, graph.OnlyStream("SyncExecutor::RunFrame"), frame, frame_index);
}

FrameReport SyncExecutor::RunFrame(StageGraph& graph, std::size_t stream,
                                   const cv::Mat& frame, std::size_t frame_index) {
    if (stream >= graph.streams().size()) {
        throw std::out_of_range("SyncExecutor::RunFrame: stream " + std::to_string(stream) +
                                " of a graph with " +
                                std::to_string(graph.streams().size()) + " stream(s)");
    }
    const StreamPlan& plan = graph.streams()[stream];
    FrameReport report;
    report.frame_index = frame_index;
    report.stream = plan.source_id;
    report.time_s = graph.FrameTime(stream, frame_index);

    if (frame.empty()) {
        report.error = "source produced an empty frame";
        return report;
    }

    // node index -> the crops waiting at its input
    std::map<std::size_t, std::vector<RoiCrop> > roi_inbox;
    // node index -> the view its full-frame input comes from
    std::map<std::size_t, FrameView> frame_inbox;
    // node index -> parent payloads on carry-result edges
    std::map<std::size_t, std::vector<StageDataPtr> > result_inbox;
    // node index -> the view it actually ran on (full-frame nodes only)
    std::map<std::size_t, FrameView> view;
    // node index -> why its hand-off input was unusable
    std::map<std::size_t, std::string> handoff_error;
    FrameView source_view;
    source_view.image = frame;

    const std::vector<std::size_t>& order = plan.order;
    std::vector<NodeRuntime>& nodes = graph.mutable_nodes();

    for (std::size_t o = 0; o < order.size(); ++o) {
        const std::size_t index = order[o];
        NodeRuntime& node = nodes[index];

        StageDataPtr produced;
        StagePorts ports;  // the producer's extra outputs (U-08)

        if (node.is_source) {
            std::shared_ptr<FrameData> data(new FrameData());
            data->image = frame;
            produced = data;
            StageResult result;
            result.data = produced;
            report.node_results[node.id] = result;
            view[index] = source_view;
        } else if (node.is_cpu) {
            const std::vector<StageDataPtr> empty_inputs;
            std::map<std::size_t, std::vector<StageDataPtr> >::const_iterator found =
                result_inbox.find(index);
            const std::vector<StageDataPtr>& inputs =
                found == result_inbox.end() ? empty_inputs : found->second;
            const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
            const CpuReduction reduced = ReduceCpu(node.op, node.params, inputs, frame.size());
            report.node_ms[node.id] = ElapsedMs(started, std::chrono::steady_clock::now());
            report.node_results[node.id] = reduced.result;
            if (!reduced.error.empty() && report.error.empty()) {
                report.error = std::string("node \"") + node.id + "\": " + reduced.error;
            }
            produced = reduced.result.data;
            view[index] = source_view;
        } else if (frame_inbox.count(index) != 0) {
            const FrameView& in = frame_inbox[index];
            view[index] = in;
            StageInput input;
            input.image = in.image;
            input.origin.inv_align = in.to_source;
            const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
            try {
                StageResult result = node.stage->run(input);
                report.node_ms[node.id] = ElapsedMs(started, std::chrono::steady_clock::now());
                report.node_results[node.id] = result;
                produced = result.data;
                ports = result.ports;
            } catch (const std::exception& error) {
                report.node_ms[node.id] = ElapsedMs(started, std::chrono::steady_clock::now());
                report.error = std::string("node \"") + node.id + "\": " +
                               error.what();
                continue;
            }
        } else {
            if (handoff_error.count(index) != 0) {
                report.error = handoff_error[index];
            }
            std::vector<RoiCrop>& crops = roi_inbox[index];
            std::vector<StageResult> results;
            const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
            for (std::size_t c = 0; c < crops.size(); ++c) {
                StageInput input;
                input.image = crops[c].image;
                input.origin = crops[c].ref;
                try {
                    results.push_back(node.stage->run(input));
                } catch (const std::exception& error) {
                    report.error = std::string("node \"") + node.id + "\": " +
                                   error.what();
                }
            }
            // No crops: 0.0, as the async executor reports it.
            report.node_ms[node.id] =
                crops.empty() ? 0.0 : ElapsedMs(started, std::chrono::steady_clock::now());
            // Canonical order: both executors sort a node's ROI results
            // into ByOrigin order before storing them, so Task 10's parity
            // test compares like-ordered vectors regardless of either
            // executor's internal delivery order (and regardless of how
            // many parents fed this node — see the roi_inbox append below).
            std::stable_sort(results.begin(), results.end(), ByOrigin);
            report.roi_results[node.id] = results;
            continue;  // ROI stages are leaves in every supported graph
        }

        if (!produced) continue;

        // Tracking runs once per producer, before any consumer crops.
        const BoxesData* boxes = NULL;
        std::shared_ptr<BoxesData> tracked;
        if (node.tracked && ProducesRoi(produced->shape())) {
            const BoxesData* source_boxes =
                dynamic_cast<const BoxesData*>(produced.get());
            if (source_boxes != NULL) {
                tracked.reset(new BoxesData(*source_boxes));
                ApplyTrackIds(tracked.get(), node.trackers[stream].get());
                boxes = tracked.get();
                // Keep the origin the stage reported (its inv_align maps a
                // hand-off view back to the source frame) and its ports.
                StageResult result = report.node_results[node.id];  // origin and ports
                result.data = tracked;
                report.node_results[node.id] = result;
                produced = tracked;
            }
        } else if (ProducesRoi(produced->shape())) {
            boxes = dynamic_cast<const BoxesData*>(produced.get());
        }

        const std::vector<std::size_t> outgoing = graph.OutgoingEdges(index);
        for (std::size_t e = 0; e < outgoing.size(); ++e) {
            const EdgeRuntime& edge = graph.edges()[outgoing[e]];
            const EdgeCargo cargo = SelectEdgeCargo(edge, produced, boxes, ports);
            if (edge.carry_result) {
                if (cargo.payload) result_inbox[edge.to].push_back(cargo.payload);
                continue;
            }
            if (!edge.roi.present) {
                if (node.is_source) {
                    frame_inbox[edge.to] = source_view;
                } else {
                    const std::string problem = HandOffProblem(cargo.payload);
                    if (problem.empty()) {
                        frame_inbox[edge.to] = HandOffView(
                            *static_cast<const ImageData*>(cargo.payload.get()), view[index]);
                    } else {
                        handoff_error[edge.to] = "node \"" + graph.nodes()[edge.to].id +
                                                 "\": \"" + node.id + "\" " + problem;
                    }
                }
                continue;
            }
            if (cargo.boxes == NULL) {
                // A port that is missing or not box-shaped breaks the
                // producer's ports contract: say so, as a plain edge does.
                if (!edge.port.empty()) {
                    handoff_error[edge.to] = "node \"" + graph.nodes()[edge.to].id +
                                             "\": \"" + node.id + "\" port \"" + edge.port +
                                             "\" handed off no boxes";
                }
                continue;
            }
            RouteStats stats;
            std::vector<RoiCrop> crops =
                RouteRois(*cargo.boxes, view[index], edge.roi, node.id, NULL, &stats);
            report.skipped_out_of_bounds += stats.clipped_away;
            // Append, not assign: a node can have more than one ROI parent
            // (nothing in ValidateGraph forbids it, and free composition
            // through JSON is the point of this project), and each parent
            // is processed in a separate iteration of this outer loop.
            // Overwriting here would silently drop every earlier parent's
            // crops. Each crop's ref.parent_node still identifies which
            // producer it came from, and the ROI-consumer branch above
            // sorts the merged inbox into ByOrigin order before storing it.
            std::vector<RoiCrop>& inbox = roi_inbox[edge.to];
            inbox.insert(inbox.end(), crops.begin(), crops.end());
        }
    }

    return report;
}

}  // namespace graph
}  // namespace dxapp
