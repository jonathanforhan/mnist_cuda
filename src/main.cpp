#include "MNISTImage.hpp"
#include "NeuralNetwork.hpp"
#include <print>

int main() {
    try {
        auto train_images = MNISTImage::import_set("data/train-labels.idx1-ubyte", "data/train-images.idx3-ubyte");
        auto t10k_images  = MNISTImage::import_set("data/t10k-labels.idx1-ubyte", "data/t10k-images.idx3-ubyte");

        NeuralNetwork nn{1};

        for (int i = 0; i < 50'000; i++) {
            nn.forward(train_images[i]);
            f32 loss = nn.backward(train_images[i].label, 0.01f);
            if (i % 5000 == 0) {
                std::println("Step {}: Loss = {:.4f}", i, loss);
            }
        }

        // Test accuracy on fresh data
        int correct = 0;
        for (int i = 50'000; i < 50100; i++) { // Use different images
            nn.forward(train_images[i]);
            if (nn.predict() == train_images[i].label)
                correct++;
        }
        std::println("Final Accuracy: {}/100 = {}%", correct, correct);
    } catch (const std::exception& err) {
        std::println(stderr, "%s", err.what());
        return -1;
    } catch (...) {
        std::println(stderr, "unknown exception");
        return -1;
    }

    return 0;
}
