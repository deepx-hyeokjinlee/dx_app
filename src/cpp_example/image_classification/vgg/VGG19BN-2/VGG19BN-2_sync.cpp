/**
 * @file VGG19BN-2
 * @brief vgg sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/VGG19BN-2_factory.hpp"
#include "common/runner/sync_classification_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_VGG19BN_2::VggFactory>();
    dxapp::SyncClassificationRunner<dxapp::v_VGG19BN_2::VggFactory> runner(std::move(factory));
    return runner.run(argc, argv);
}
