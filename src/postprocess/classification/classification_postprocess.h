#ifndef CLASSIFICATION_POSTPROCESS_H
#define CLASSIFICATION_POSTPROCESS_H

#include <dxrt/dxrt_api.h>

#include <string>
#include <vector>

/**
 * @brief Classification result structure
 */
struct ClassificationResult {
    int class_id{-1};
    float confidence{0.0f};

    ClassificationResult() = default;

    ClassificationResult(int cls_id, float conf)
        : class_id(cls_id), confidence(conf) {}

    ~ClassificationResult() = default;
};

/**
 * @brief Generic classification post-processing class
 *
 * Handles softmax (skipped when the output already is a distribution) + top-K
 * for classification models.
 * Input tensor: [1, num_classes] or [1, 1, num_classes] (raw logits or probabilities)
 */
class ClassificationPostProcess {
   private:
    int top_k_{5};
    bool background_class_{false};

    std::vector<float> apply_softmax(const float* data, int num_classes) const;

   public:
    /**
     * @brief Constructor
     * @param top_k Number of top predictions to return (default: 5)
     * @param background_class The head has a leading background row (TF-slim
     *     exports: 1001 outputs, row 0 = background, rows 1..1000 = ImageNet
     *     0..999). It is softmaxed over all its rows, then row 0 is dropped and
     *     class ids are the ImageNet ids. Without it the ids were one off
     *     (inceptionv1: top-1 4 of 200 ImageNetV2 images right, 119 of 200 with
     *     the shift) and the C++ example dropped the last class instead.
     */
    explicit ClassificationPostProcess(int top_k = 5, bool background_class = false);
    ~ClassificationPostProcess() = default;

    std::vector<ClassificationResult> postprocess(const dxrt::TensorPtrs& outputs);

    int get_top_k() const { return top_k_; }
    bool get_background_class() const { return background_class_; }
};

#endif  // CLASSIFICATION_POSTPROCESS_H
