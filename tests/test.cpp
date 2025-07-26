#include <print>
#include <stdexcept>

void Tensor_test();
void MNISTImporter_test();

int main() {
    try {
        Tensor_test();
        MNISTImporter_test();
    } catch (const std::exception& ex) {
        std::println(stderr, "{}", ex.what());
        return -1;
    }
    return 0;
}