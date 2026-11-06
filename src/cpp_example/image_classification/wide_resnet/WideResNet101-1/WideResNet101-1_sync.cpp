/**
 * @file WideResNet101-1
 * @brief wide_resnet sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/WideResNet101-1_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_WideResNet101_1::WideResnetFactory>();
    dxapp::SyncClassificationRunner<dxapp::v_WideResNet101_1::WideResnetFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
