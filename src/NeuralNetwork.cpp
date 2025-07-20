#include "NeuralNetwork.hpp"

#include <array>
#include <cublas_v2.h>

NeuralNetwork::NeuralNetwork(size_t batch_size)
    : B{batch_size} {
    cublasCreate(&_cublas_handle);

    xavier_init(_W, _W.capacity());
    zero_init(_b, _b.capacity());
}

NeuralNetwork::~NeuralNetwork() noexcept {
    cublasDestroy(_cublas_handle);
}

void NeuralNetwork::forward(const MNISTImage& mnist_image) {
    Tensor<u8, 1> gpu_pixels{28 * 28};
    gpu_pixels.from_host(mnist_image.pixels);

    launch_normalize_mnist(gpu_pixels, _x, gpu_pixels.capacity());

    // y = alpha * A^T * x + beta * y
    const float alpha = 1.0f, beta = 0.0f;
    int m = _W.size()[0], n = _W.size()[1];

    cublasSgemv(_cublas_handle, CUBLAS_OP_T, m, n, &alpha, _W, m, _x, 1, &beta, _y, 1);
    add_bias(_y, _b, _y, 10);
    relu_activation(_y, _y, 10);
}

f32 NeuralNetwork::backward(int true_label, f32 learning_rate) {
    // Compute loss and output gradients
    softmax_cross_entropy(_y, true_label, _loss, _dy, 10);

    // A = alpha * x * y^T + A
    const float alpha = 1.0f;
    int m = _dW.size()[0], n = _dW.size()[1];

    compute_weight_gradients(_x, _dy, _dW, 784, 10);

    update_weights(_W, _dW, learning_rate, 784 * 10);
    update_weights(_b, _dy, learning_rate, 10);

    // Return loss
    std::array<f32, 1> loss_cpu;
    _loss.to_host(loss_cpu);
    return loss_cpu[0];
}

void NeuralNetwork::train_step(const MNISTImage& mnist_image, f32 learning_rate) {
    forward(mnist_image);
    f32 loss = backward(mnist_image.label, learning_rate);
}

int NeuralNetwork::predict() {
    // Get output back to CPU and find max
    std::vector<float> cpu_output(10);
    _y.to_host(std::span<float>(cpu_output));

    return std::max_element(cpu_output.begin(), cpu_output.end()) - cpu_output.begin();
}
