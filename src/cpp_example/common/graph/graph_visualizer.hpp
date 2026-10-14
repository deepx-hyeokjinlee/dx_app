/**
 * @file graph_visualizer.hpp
 * @brief Draw every stage's result of one frame report onto one canvas.
 *
 * `ref.inv_align` alone maps a crop-local coordinate directly to the
 * source frame — `roi_router.cpp`'s `RouteRois` folds the crop's own
 * `ref.src_box` translation into `inv_align` at construction time (a
 * different composition per crop kind; see `RouteRois`'s own comment), so
 * nothing downstream ever adds `ref.src_box` a second time. `RestorePoint-
 * Warped`, below, applies exactly `ref.inv_align` to a point. `roi_router.
 * hpp`'s `RestoreBoxCorners`/`RestoreBox` apply the same `ref.inv_align`
 * to a box's four corners, then (`RestoreBox` only) take their axis-
 * aligned bounding box. Both are correct for a plain crop too
 * (`ref.inv_align` reduces to plain translation there) and for a
 * full-frame stage (`ref.from_roi == false`: identity `inv_align` passes
 * the input straight through; a hand-off child's `inv_align` carries its
 * image's scale back to the source frame).
 *
 * Fix round 3 was this file's structural correction: rounds 1-2 composed
 * `inv_align`-applied-then-`+src_box` in every one of these functions,
 * which was wrong for both warped-crop producers (OBB applies the
 * translation *before* the rotation, not after; face5 has no `src_box`
 * relationship at all — see `roi_router.cpp`'s `RouteRois` comment) but
 * untestable by every fixture in this suite, since every non-identity-
 * `inv_align` fixture put `ref.src_box` at the origin, where both
 * compositions agree by coincidence.
 *
 * Fix round 4 removed `RestoreAngleWarped`, round 2's scalar-angle
 * composition: it had zero non-test callers (`DrawObBoxes` never
 * reconstructed a `cx/cy/w/h/angle` tuple from it after round 3 switched
 * to drawing real corners, and no other consumer - including Task 13's
 * `--report` JSON serializer, checked directly - ever restores an OBB's
 * angle to a scalar). `DrawObBoxes` now rotates a box's four corners by
 * `item.angle` in the crop's own local frame, then restores each corner
 * through `RestorePointWarped` - rotate first, warp second, exactly the
 * order `item.angle`'s own crop-local meaning requires.
 */
#ifndef DXAPP_GRAPH_GRAPH_VISUALIZER_HPP
#define DXAPP_GRAPH_GRAPH_VISUALIZER_HPP

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common/graph/shape.hpp"
#include "common/graph/stage_graph.hpp"

namespace dxapp {
namespace graph {

/**
 * @brief Undo a crop's alignment/un-rotation, then its translation.
 *
 * `ref.inv_align` is the inverse of the affine transform recorded when the
 * crop was produced (identity for a plain crop, so this is a strict
 * generalization of a translation-only restore). `roi_local` is a point
 * expressed in that crop's own pixel space; the result is the same point
 * in source-frame coordinates. `ref.from_roi == false` (a full-frame
 * stage) applies `ref.inv_align` too, same contract as RestoreBox/
 * RestorePoint: that is the identity (`roi_local` returned unchanged)
 * unless the result came from a handed-off image (see `roi_router.hpp`'s
 * `FrameView`).
 */
cv::Point2f RestorePointWarped(const cv::Point2f& roi_local, const RoiRef& ref);

/**
 * @brief Composite every stage's result in `report` onto `source`.
 *
 * Draw order is fixed so two calls with the same report produce identical
 * pixels: dense layers first - images, copied opaquely as the backdrop,
 * then labelmaps and densemaps blended over them - then instance masks,
 * then boxes (a detector's and each instance's, with its label)/obboxes/
 * keypoints restored to source coordinates, then scores/vectors as text
 * anchored to their parent box. Both `report.node_results`
 * (full-frame stages) and `report.roi_results` (per-crop stages) are
 * drawn — a cascade's second-stage results are exactly the point of this
 * function, not an afterthought.
 *
 * Array-shaped ROI results (an image, a labelmap, a densemap, an instance
 * mask) land where their crop came from. A plain crop, a crop from a
 * scaled view and a full-frame result are rectangles on the canvas and are
 * resized into them. A crop whose `origin.inv_align` rotates or shears (an
 * OBB or face5 crop, before or after a hand-off) is resized to its
 * `crop_size` and warped through `inv_align`, and only the pixels the
 * warped crop covers are blended or copied.
 *
 * A result's ports (`StageResult::ports`) are drawn in the pass of each
 * port's shape, right after that result's primary payload, with the
 * result's own `origin`. A binary mask (`LabelMapData::binary_mask`) blends
 * only its non-zero pixels, in `mask_color` at alpha 0.5 (on a rotated or
 * aligned crop it blends its whole footprint). A non-spatial dense map
 * (`DenseMapData::spatial` false) is never drawn.
 *
 * Logic nodes (FrameReport::overlays) are drawn by pass 5 only, after all
 * of the above: their polygons tinted and outlined, their lines with an
 * arrow to the "in" side (the right of A->B), highlighted boxes, then their
 * labels, at a point or stacked in a panel from the top left. A logic
 * node's own payload is not drawn by the shape passes: a zone's boxes are
 * its detector's boxes, already drawn.
 *
 * An empty `source` returns it unchanged; a default (empty) `report`
 * draws nothing and returns a clone of `source`.
 */
cv::Mat RenderReport(const cv::Mat& source, const FrameReport& report);

/// The CLI's recent-events panel: `lines` bottom-left, oldest first, the
/// newest at the bottom. Kept out of RenderReport, which draws one report.
void DrawEventLog(cv::Mat* canvas, const std::vector<std::string>& lines);

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_GRAPH_VISUALIZER_HPP
