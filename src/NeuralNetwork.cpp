#include "NeuralNetwork.hpp"

#include <cublas_v2.h>
#include <cassert>
#include <cmath>
#include "kernels.h"

NeuralNetwork::NeuralNetwork(int batch_size, int hidden_size)
    : B{batch_size},
      H{hidden_size} {
    cublasCreate(&_cublas_handle);

    /* initialize */
    kaiming_init(_W1, _W1.size(), M);
    zero_init(_b1, _b1.size());

    xavier_init(_W2, _W2.size(), H, N);
    zero_init(_b2, _b2.size());

    zero_init(_z1, _z1.size());
    zero_init(_a1, _a1.size());
}

NeuralNetwork::~NeuralNetwork() noexcept {
    cublasDestroy(_cublas_handle);
}

void NeuralNetwork::forward(std::span<const MNISTImage> mnist_images) {
    /* when not training mnist_images may be less than B * M */

    _raw.from_host(std::span{(const u8*)mnist_images.data(), mnist_images.size_bytes()});
    normalize_u8_to_f32(_raw, _x, _x.size());

    const float alpha = 1.0f, beta = 0.0f;

    // z1 = W1 @ x + b1
    cublasSgemv(_cublas_handle, CUBLAS_OP_T, M, H, &alpha, _W1, M, _x, 1, &beta, _z1, 1);
    cublasSaxpy(_cublas_handle, H, &alpha, _b1, 1, _z1, 1);

    // a1 = ReLU(z1)
    relu(_z1, _a1, 1, H);

    // y = W2 @ a1 + b2
    cublasSgemv(_cublas_handle, CUBLAS_OP_T, H, N, &alpha, _W2, H, _a1, 1, &beta, _y, 1);
    cublasSaxpy(_cublas_handle, N, &alpha, _b2, 1, _y, 1);
}

f32 NeuralNetwork::backward(std::span<const u8> true_labels, f32 learning_rate) {
    assert(true_labels.size_bytes() == B);

    _labels.from_host(true_labels);

    softmax_cross_entropy(_y, _labels, _loss, _dy, 1, N);

    f32 loss;
    _loss.to_host({&loss, 1});

    if (loss < 1e-6f) {
        return loss;
    }

    float alpha = 1.0f, beta = 0.0f;

    // dW2 = a1 @ dy
    cublasSger(_cublas_handle, H, N, &alpha, _a1, 1, _dy, 1, _dW2, H);
    // db2 = dy
    cublasScopy(_cublas_handle, N, _dy, 1, _db2, 1);

    // da1 = W2.T @ dy
    cublasSgemv(_cublas_handle, CUBLAS_OP_N, H, N, &alpha, _W2, H, _dy, 1, &beta, _da1, 1);
    // dz1 = da1 @ ReLU'(z1)
    relu_backwards(_dz1, _z1, _da1, 1, H);

    // dW1 = x outer dz1
    cublasSger(_cublas_handle, M, H, &alpha, _x, 1, _dz1, 1, _dW1, M);
    // db1 = dz1
    cublasScopy(_cublas_handle, H, _dz1, 1, _db1, 1);

    return loss;
}

f32 NeuralNetwork::train(std::span<const MNISTImage> images, std::span<const u8> labels, f32 learning_rate) {
    assert(images.size_bytes() == B * M);
    assert(labels.size_bytes() == B);

    forward(images);
    f32 loss = backward(labels, learning_rate);

    return loss;
}

void NeuralNetwork::predict(std::span<u8> output) {
    // Get output back to CPU and find max
    std::vector<float> cpu_output(N);
    _y.to_host(std::span<float>(cpu_output));

    output[0] = (u8)(std::max_element(cpu_output.begin(), cpu_output.end()) - cpu_output.begin());
}
