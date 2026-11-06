/**
 * @file ResNet50-9
 * @brief resnet async inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/ResNet50-9_factory.hpp"
#include "common/runner/async_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_ResNet50_9::ResnetFactory>();
    dxapp::AsyncClassificationRunner<dxapp::v_ResNet50_9::ResnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
