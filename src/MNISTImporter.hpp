#pragma once

#include "types.hpp"
#include <vector>

#pragma pack(push, 1)

struct MNISTImage {
    u8 pixels[28 * 28];

    void debug() const;
};

#pragma pack(pop)

struct MNISTImporter {
    MNISTImporter() = delete;

    static std::vector<MNISTImage> import_images(const char* images_path);

    static std::vector<u8> import_labels(const char* labels_path);
};
