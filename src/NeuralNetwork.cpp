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
    xavier_init(_W, _W.size(), M, N);
    zero_init(_b, _b.size());
}

NeuralNetwork::~NeuralNetwork() noexcept {
    cublasDestroy(_cublas_handle);
}

void NeuralNetwork::forward(std::span<const MNISTImage> mnist_images) {
    /* when not training mnist_images may be less than B * M */

    _raw.from_host(std::span{(const u8*)mnist_images.data(), mnist_images.size_bytes()});
    normalize_u8_to_f32(_raw, _x, _x.size());

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemv(_cublas_handle, CUBLAS_OP_T, M, N, &alpha, _W, M, _x, 1, &beta, _y, 1);
    cublasSaxpy(_cublas_handle, N, &alpha, _b, 1, _y, 1);
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
    cublasSger(_cublas_handle, M, N, &alpha, _x, 1, _dy, 1, _dW, M);
    cublasScopy(_cublas_handle, N, _dy, 1, _db, 1);

    const float neg_lr = -learning_rate;
    cublasSaxpy(_cublas_handle, _W.size(), &neg_lr, _dW, 1, _W, 1);
    cublasSaxpy(_cublas_handle, _b.size(), &neg_lr, _db, 1, _b, 1);

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
