/**
 * @file preprocessing.hpp
 * @brief Common preprocessing utilities (letterbox, color conversion, etc.)
 */

#ifndef DXAPP_PREPROCESSING_HPP
#define DXAPP_PREPROCESSING_HPP

#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

#include "../base/i_processor.hpp"

namespace dxapp {

/// round(size * gain), half to even like Python's round(), at least 1.
inline int letterboxScaledSize(int size, double gain) {
    return std::max(1, static_cast<int>(std::nearbyint(size * gain)));
}

/**
 * @brief Calculate letterbox padding for maintaining aspect ratio
 * @param img_width Original image width
 * @param img_height Original image height
 * @param target_width Target model input width
 * @param target_height Target model input height
 * @param ctx Output preprocessing context with padding info
 */
inline void calculateLetterboxParams(int img_width, int img_height,
                                     int target_width, int target_height,
                                     PreprocessContext& ctx) {
    ctx.original_width = img_width;
    ctx.original_height = img_height;
    ctx.input_width = target_width;
    ctx.input_height = target_height;

    // The Ultralytics letterbox, as the Python LetterboxPreprocessor: the scaled
    // size is ROUNDED (half to even, Python's round()) and the odd padding pixel
    // goes to the bottom/right via round(d - 0.1). Truncating instead shifted
    // the image a row and resized it a row short whenever the scale is not
    // exact (1536x1024 -> 640: 426 rows at top 107 instead of 427 at 106), and
    // every box moved -- yolov5-s up to 7.7 px against the Python example.
    const double gain = std::min(static_cast<double>(target_width) / img_width,
                                 static_cast<double>(target_height) / img_height);
    ctx.scale = static_cast<float>(gain);
    const int new_width = letterboxScaledSize(img_width, gain);
    const int new_height = letterboxScaledSize(img_height, gain);
    ctx.pad_x = static_cast<int>(std::nearbyint((target_width - new_width) / 2.0 - 0.1));
    ctx.pad_y = static_cast<int>(std::nearbyint((target_height - new_height) / 2.0 - 0.1));
}

/**
 * @brief Apply letterbox transformation to an image
 * @param image Input image (BGR format)
 * @param output Output image (resized with padding)
 * @param target_width Target width
 * @param target_height Target height
 * @param color_conversion OpenCV color conversion code
 * @param ctx Output preprocessing context
 * @param pad_value Padding pixel value (default 114 for YOLO models)
 */
inline void makeLetterboxImage(const cv::Mat& image, cv::Mat& output,
                               int target_width, int target_height,
                               int color_conversion,
                               PreprocessContext& ctx,
                               int pad_value = 114,
                               bool pad_edge = false) {
    calculateLetterboxParams(image.cols, image.rows, target_width, target_height, ctx);
    // pad_edge: the image at the top-left and all padding right/bottom -- the
    // compile JSONs' `pad_location: edge`, as automl EfficientDet and GluonCV
    // evaluate (Python LetterboxPreprocessor(pad_location="edge")). The
    // postprocessors un-letterbox through ctx.pad_x/pad_y, so they follow.
    ctx.pad_edge = pad_edge;
    if (pad_edge) {
        ctx.pad_x = 0;
        ctx.pad_y = 0;
    }

    // New dimensions after scaling (the same rule as calculateLetterboxParams)
    const double gain = std::min(static_cast<double>(target_width) / image.cols,
                                 static_cast<double>(target_height) / image.rows);
    const int new_width = letterboxScaledSize(image.cols, gain);
    const int new_height = letterboxScaledSize(image.rows, gain);

    // Resize to new dimensions
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));

    // Convert color space; a negative code keeps the camera's BGR (models
    // trained on cv2 images: YOLOX, NanoDet, ...), as SimpleResizePreprocessor
    // already does. cv::cvtColor would throw on it.
    cv::Mat converted;
    if (color_conversion >= 0) {
        cv::cvtColor(resized, converted, color_conversion);
    } else {
        converted = resized;
    }

    // Asymmetric padding (matching original: remainder goes to bottom/right)
    int top = ctx.pad_y;
    int bottom = target_height - new_height - top;
    int left = ctx.pad_x;
    int right = target_width - new_width - left;

    // Add padding
    cv::copyMakeBorder(converted, output,
                       top, bottom,
                       left, right,
                       cv::BORDER_CONSTANT, 
                       cv::Scalar(pad_value, pad_value, pad_value));
}

/**
 * @brief Simple resize preprocessing (no letterbox)
 * @param image Input image
 * @param output Output image
 * @param target_width Target width
 * @param target_height Target height
 * @param color_conversion OpenCV color conversion code
 * @param ctx Output preprocessing context
 */
inline void makeResizedImage(const cv::Mat& image, cv::Mat& output,
                             int target_width, int target_height,
                             int color_conversion,
                             PreprocessContext& ctx) {
    ctx.original_width = image.cols;
    ctx.original_height = image.rows;
    ctx.input_width = target_width;
    ctx.input_height = target_height;
    ctx.scale = 1.0f;
    ctx.pad_x = 0;
    ctx.pad_y = 0;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_width, target_height));
    cv::cvtColor(resized, output, color_conversion);
}

/**
 * @brief Scale coordinates from model space back to original image space
 * @param x X coordinate in model space
 * @param y Y coordinate in model space
 * @param ctx Preprocessing context with transformation info
 * @return Pair of (x, y) in original image space
 */
inline std::pair<float, float> scaleToOriginal(float x, float y, 
                                               const PreprocessContext& ctx) {
    float orig_x = x - ctx.pad_x;
    float orig_y = y - ctx.pad_y;

    if (ctx.scale_x > 0.0f && ctx.scale_y > 0.0f) {
        orig_x = orig_x / ctx.scale_x;
        orig_y = orig_y / ctx.scale_y;
    } else if (ctx.scale > 0.0f) {
        orig_x = orig_x / ctx.scale;
        orig_y = orig_y / ctx.scale;
    }
    
    // [0, w] x [0, h], as scaleBox and the Python tree clip (the Ultralytics rule).
    orig_x = std::max(0.0f, std::min(orig_x, static_cast<float>(ctx.original_width)));
    orig_y = std::max(0.0f, std::min(orig_y, static_cast<float>(ctx.original_height)));
    
    return {orig_x, orig_y};
}

/**
 * @brief Scale a bounding box from model space to original image space
 * @param box Bounding box [x1, y1, x2, y2] in model space
 * @param ctx Preprocessing context
 * @return Scaled bounding box in original image space
 */
inline std::vector<float> scaleBoxToOriginal(const std::vector<float>& box,
                                             const PreprocessContext& ctx) {
    if (box.size() < 4) return box;
    
    std::pair<float,float> p1 = scaleToOriginal(box[0], box[1], ctx);
    std::pair<float,float> p2 = scaleToOriginal(box[2], box[3], ctx);
    float x1 = p1.first, y1 = p1.second;
    float x2 = p2.first, y2 = p2.second;
    
    return {x1, y1, x2, y2};
}

}  // namespace dxapp

#endif  // DXAPP_PREPROCESSING_HPP
