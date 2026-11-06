/**
 * @file ResNet18-2
 * @brief resnet sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/ResNet18-2_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_ResNet18_2::ResnetFactory>();
    dxapp::SyncClassificationRunner<dxapp::v_ResNet18_2::ResnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
