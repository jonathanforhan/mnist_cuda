#pragma once

#include "types.hpp"
#include <vector>

struct MNISTImage {
    u8 pixels[28 * 28];
    u8 label;

    void debug() const;

    static std::vector<MNISTImage> import_set(const char* labels_path, const char* images_path);
};
