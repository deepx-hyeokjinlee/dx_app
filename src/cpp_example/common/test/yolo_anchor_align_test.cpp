/**
 * @file yolo_anchor_align_test.cpp
 * @brief YOLOv5/YOLOv7PostProcess::align_tensors on a NON-SQUARE input
 *        (survey item F16). Called from common_unit_test.cpp's main().
 *
 * 640x384 (W x H): stride-8/16/32 grids are 80x48, 40x24, 20x12, so the NCHW
 * anchor maps are [1,255,48,80], [1,255,24,40], [1,255,12,20]. align_tensors
 * compared shape[2] (H) with the grid WIDTH and shape[3] (W) with the grid
 * HEIGHT, so on a non-square input nothing matched and the maps came back
 * in arrival order. They arrive out of stride order here; aligned, the first
 * is stride 8 (H 48, W 80).
 */
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "yolov5_postprocess.h"
#include "yolov7_postprocess.h"

namespace {

dxrt::TensorPtr AnchorMapNCHW(const char* name, int gh, int gw, std::vector<float>& storage) {
    storage.assign(static_cast<std::size_t>(255) * gh * gw, 0.f);
    std::vector<int64_t> shape;
    shape.push_back(1);
    shape.push_back(255);
    shape.push_back(gh);
    shape.push_back(gw);
    return std::make_shared<dxrt::Tensor>(name, shape, dxrt::DataType::FLOAT, storage.data());
}

template <class PostProcess>
int Failures(const char* which) {
    std::vector<float> a, b, c;
    dxrt::TensorPtrs outputs;
    outputs.push_back(AnchorMapNCHW("s32", 12, 20, a));
    outputs.push_back(AnchorMapNCHW("s8", 48, 80, b));
    outputs.push_back(AnchorMapNCHW("s16", 24, 40, c));
    PostProcess post(640, 384, 0.25f, 0.3f, 0.45f, false);
    const dxrt::TensorPtrs aligned = post.align_tensors(outputs);
    const int expect[3][2] = {{48, 80}, {24, 40}, {12, 20}};
    int failures = 0;
    if (aligned.size() != 3) {
        std::printf("FAIL %s: align_tensors returned %zu maps, expected 3\n", which, aligned.size());
        return 1;
    }
    for (int i = 0; i < 3; ++i) {
        const auto& s = aligned[i]->shape();
        if (s[2] != expect[i][0] || s[3] != expect[i][1]) {
            std::printf("FAIL %s: aligned[%d] is [1,255,%ld,%ld], expected [1,255,%d,%d]\n",
                        which, i, static_cast<long>(s[2]), static_cast<long>(s[3]),
                        expect[i][0], expect[i][1]);
            ++failures;
        }
    }
    return failures;
}

}  // namespace

int YoloAnchorMapsAlignFailures(const char* which) {
    return std::string(which) == "yolov5" ? Failures<YOLOv5PostProcess>(which)
                                          : Failures<YOLOv7PostProcess>(which);
}
