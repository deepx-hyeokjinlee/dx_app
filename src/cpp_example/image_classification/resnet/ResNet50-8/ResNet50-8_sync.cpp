/**
 * @file ResNet50-8
 * @brief resnet sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/ResNet50-8_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_ResNet50_8::ResnetFactory>();
    dxapp::SyncClassificationRunner<dxapp::v_ResNet50_8::ResnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
