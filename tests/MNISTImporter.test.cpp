#include "MNISTImporter.hpp"
#include <stdexcept>

void MNISTImporter_test() {
    auto train_images = MNISTImporter::import_images("data/train-images.idx3-ubyte");
    auto train_labels = MNISTImporter::import_labels("data/train-labels.idx1-ubyte");
    auto t10k_images  = MNISTImporter::import_images("data/t10k-images.idx3-ubyte");
    auto t10k_labels  = MNISTImporter::import_labels("data/t10k-labels.idx1-ubyte");
    if (!(train_images.size() == train_labels.size() && t10k_images.size() == t10k_labels.size())) {
        throw std::runtime_error{"imported mnist images were corrupted"};
    }
}