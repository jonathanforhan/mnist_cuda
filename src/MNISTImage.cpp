#include "MNISTImage.hpp"

#include "log.hpp"
#include <cassert>
#include <fstream>

static u32 swap_endianess(u32 x) {
    return ((x & 0xff) << 24) | ((x & 0xff00) << 8) | ((x & 0xff0000) >> 8) | ((x & 0xff000000) >> 24);
}

void MNISTImage::debug() const {
    printf("label: %u\n", label);

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

std::vector<MNISTImage> MNISTImage::import_set(const char* labels_path, const char* images_path) {
    std::vector<MNISTImage> images;

    { // labels
        std::ifstream ifs(labels_path, std::ios::binary);
        if (!ifs.is_open()) {
            ELOG("cannot open file %s", labels_path);
            std::exit(EXIT_FAILURE);
        }

        u32 magic, nlabels;
        ifs.read((char*)(&magic), sizeof(magic));
        ifs.read((char*)(&nlabels), sizeof(nlabels));

        magic   = swap_endianess(magic);
        nlabels = swap_endianess(nlabels);

        if (magic != 2049) {
            ELOG("magic number != 2049, instead is %u", magic);
            std::exit(EXIT_FAILURE);
        }

        images.resize(nlabels);

        for (auto& image : images) {
            static_assert(sizeof(u8) == sizeof(image.label));
            ifs.read((char*)&image.label, sizeof(u8));
        }
    }

    { // images
        std::ifstream ifs(images_path, std::ios::binary);
        if (!ifs.is_open()) {
            ELOG("cannot open file %s", images_path);
            std::exit(EXIT_FAILURE);
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
            ELOG("magic number != 2051, instead is %u", magic);
            std::exit(EXIT_FAILURE);
        }

        for (auto& image : images) {
            assert(rows * cols == sizeof(image.pixels));
            ifs.read((char*)image.pixels, rows * cols);
        }
    }

    return images;
}
