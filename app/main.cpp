#include <cassert>
#include <print>
#include "MNISTImporter.hpp"
#include "NeuralNetwork.hpp"

int main() {
    try {
        auto train_images = MNISTImporter::import_images("data/train-images.idx3-ubyte");
        auto train_labels = MNISTImporter::import_labels("data/train-labels.idx1-ubyte");
        auto t10k_images  = MNISTImporter::import_images("data/t10k-images.idx3-ubyte");
        auto t10k_labels  = MNISTImporter::import_labels("data/t10k-labels.idx1-ubyte");
        assert(train_images.size() == train_labels.size() && t10k_images.size() == t10k_labels.size());

        NeuralNetwork nn{1};

        for (int i = 0; i < train_images.size(); i++) {
            const auto& image = train_images[i];
            const auto& label = train_labels[i];

            f32 loss = nn.train({&image, 1}, {&label, 1}, 1e-6f);

            if (i % 1000 == 0) {
                std::println("Loss: {:.3f}", loss);
            }
        }

        // Test accuracy on fresh data
        u32 correct = 0;
        for (int i = 0; i < t10k_images.size(); i++) {
            const auto& image = t10k_images[i];
            const auto& label = t10k_labels[i];

            nn.forward({&image, 1});

            u8 prediction;
            nn.predict({&prediction, 1});

            if (prediction == label) {
                correct++;
            }
        }
        std::println(
            "Final Accuracy: ({}/{}) {:.2f}%", correct, t10k_images.size(), 100.0f * correct / t10k_images.size());
    } catch (const std::exception& err) {
        std::println(stderr, "{}", err.what());
        return -1;
    } catch (...) {
        std::println(stderr, "unknown exception");
        return -1;
    }

    return 0;
}
