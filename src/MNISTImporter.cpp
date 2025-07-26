#include "MNISTImporter.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

static u32 swap_endianess(u32 x) {
    return ((x & 0xff) << 24) | ((x & 0xff00) << 8) | ((x & 0xff0000) >> 8) | ((x & 0xff000000) >> 24);
}

void MNISTImage::debug() const {
    for (int i = 0; i < std::size(pixels); i++) {
        if (pixels[i] > 200) {
            printf("@");
        } else if (pixels[i] > 150) {
            printf("%%");
        } else if (pixels[i] > 100) {
            printf("*");
        } else if (pixels[i] > 50) {
            printf(".");
        } else {
            printf(" ");
        }

        if (i && i % 28 == 0) {
            printf("\n");
        }
    }

    printf("\n");
}

std::vector<MNISTImage> MNISTImporter::import_images(const char* images_path) {
    std::vector<MNISTImage> images;

    std::ifstream ifs(images_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error{"cannot open file %s" + std::string(images_path)};
    }

    u32 magic, nimages, rows, cols;
    ifs.read((char*)(&magic), sizeof(magic));
    ifs.read((char*)(&nimages), sizeof(nimages));
    ifs.read((char*)(&rows), sizeof(rows));
    ifs.read((char*)(&cols), sizeof(cols));

    magic   = swap_endianess(magic);
    nimages = swap_endianess(nimages);
    rows    = swap_endianess(rows);
    cols    = swap_endianess(cols);

    if (magic != 2051) {
        throw std::runtime_error{"magic number != 2051, instead is " + std::to_string(magic)};
    } else if (rows * cols != sizeof(MNISTImage::pixels)) {
        throw std::runtime_error{"rows * cols must equal 28x28"};
    }

    images.resize(nimages);

    for (auto& image : images) {
        ifs.read((char*)image.pixels, rows * cols);
    }

    return images;
}

std::vector<u8> MNISTImporter::import_labels(const char* labels_path) {
    std::vector<u8> labels;

    std::ifstream ifs(labels_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error{"cannot open file " + std::string(labels_path)};
    }

    u32 magic, nlabels;
    ifs.read((char*)(&magic), sizeof(magic));
    ifs.read((char*)(&nlabels), sizeof(nlabels));

    magic   = swap_endianess(magic);
    nlabels = swap_endianess(nlabels);

    if (magic != 2049) {
        throw std::runtime_error{"magic number != 2049, instead is " + std::to_string(magic)};
    }

    labels.resize(nlabels);

    for (auto& label : labels) {
        static_assert(sizeof(u8) == sizeof(label));
        ifs.read((char*)&label, sizeof(u8));
    }

    return labels;
}
