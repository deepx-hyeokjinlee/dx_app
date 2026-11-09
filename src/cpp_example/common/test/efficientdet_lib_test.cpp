/**
 * @file efficientdet_lib_test.cpp
 * @brief EfficientDetPostProcess (src/postprocess/efficientdet, the class the
 *        Python binding wraps) on a yolov4-shaped output. Called from
 *        common_unit_test.cpp's main().
 *
 * yolov4 exports run with the ORT tail emit [1,N,1,4] boxes as NORMALIZED
 * [x1,y1,x2,y2] and [1,N,80] scores with no background column. One box at
 * (0.1,0.2)-(0.5,0.6) of a 400x400 input, class 16 at 0.9: read as TF
 * [ymin,xmin,ymax,xmax] it comes back transposed, (80,40)-(240,200).
 */
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "efficientdet_postprocess.h"

int EfficientDetYolov4LayoutFailures() {
    const int n = 3, classes = 80;
    std::vector<float> boxes(static_cast<std::size_t>(n) * 4, 0.f);
    std::vector<float> scores(static_cast<std::size_t>(n) * classes, 0.f);
    boxes[0] = 0.1f; boxes[1] = 0.2f; boxes[2] = 0.5f; boxes[3] = 0.6f;
    scores[16] = 0.9f;
    dxrt::TensorPtrs outputs;
    outputs.push_back(std::make_shared<dxrt::Tensor>(
        "boxes", std::vector<int64_t>{1, n, 1, 4}, dxrt::DataType::FLOAT, boxes.data()));
    outputs.push_back(std::make_shared<dxrt::Tensor>(
        "scores", std::vector<int64_t>{1, n, classes}, dxrt::DataType::FLOAT, scores.data()));
    EfficientDetPostProcess post(400, 400, 0.3f, 0.45f, classes, /*has_background=*/false,
                                 /*boxes_xyxy=*/true);
    const auto found = post.postprocess(outputs);
    if (found.size() != 1) {
        std::printf("FAIL efficientdet yolov4: %zu detections, expected 1\n", found.size());
        return 1;
    }
    const auto& b = found[0].box;
    const float want[4] = {40.f, 80.f, 200.f, 240.f};
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(b[i] - want[i]) > 1e-3f) { ++failures; }
    }
    if (found[0].class_id != 16) { ++failures; }
    if (failures) {
        std::printf("FAIL efficientdet yolov4: box (%.1f,%.1f,%.1f,%.1f) class %d, expected "
                    "(40,80,200,240) class 16\n", b[0], b[1], b[2], b[3], found[0].class_id);
    }
    return failures;
}
