#include "Tensor.hpp"
#include <cassert>
#include <vector>
#include "types.hpp"

void Tensor_test() {
    {
        Tensor<f32> tensor(10, 20, 30);
        assert(tensor.size() == 10 * 20 * 30);
        assert(tensor.size() == tensor.dim(0) * tensor.dim(1) * tensor.dim(2));
        assert(tensor.size_bytes() == 10 * 20 * 30 * sizeof(Tensor<f32>::data_type));
        assert(tensor.dim(0) == 10);
        assert(tensor.dim(1) == 20);
        assert(tensor.dim(2) == 30);
    }

    {
        std::vector<int> host_in{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

        Tensor<int> tensor1(2, 5);
        tensor1.from_host(host_in);

        Tensor<int> tensor2(10);
        tensor2.from_device(tensor1, tensor1.size_bytes());

        std::vector<int> host_out(10);
        tensor2.to_host(host_out);

        for (int i = 0; i < 10; i++) {
            assert(host_out[i] == host_in[i]);
        }
    }
}
