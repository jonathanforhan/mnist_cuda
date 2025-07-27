#include "NeuralNetwork.hpp"

#include <cublas_v2.h>
#include <cassert>
#include <print>
#include "kernels.h"

NeuralNetwork::NeuralNetwork(int batch_size, int hidden_size)
    : B{batch_size},
      H{hidden_size} {
    cublasCreate(&_cublas_handle);

    /* initialize */
    kaiming_init(_W1, _W1.size(), M);
    zero_init(_dW1, _dW1.size());
    zero_init(_b1, _b1.size());
    zero_init(_db1, _db1.size());

    xavier_init(_W2, _W2.size(), H, N);
    zero_init(_dW2, _dW2.size());
    zero_init(_b2, _b2.size());
    zero_init(_db2, _db2.size());

    zero_init(_z1, _z1.size());
    zero_init(_a1, _a1.size());
    zero_init(_dz1, _dz1.size());
    zero_init(_da1, _da1.size());
}

NeuralNetwork::~NeuralNetwork() noexcept {
    cublasDestroy(_cublas_handle);
}

void NeuralNetwork::forward(std::span<const MNISTImage> mnist_images) {
    /* when not training mnist_images may be less than B * M */

    _raw.from_host(std::span{(const u8*)mnist_images.data(), mnist_images.size_bytes()});
    normalize_u8_to_f32(_raw, _x, _x.size());

    const float alpha = 1.0f, beta = 0.0f;

    // z1[H] = W1[H,M] @ x[M] + b1[H]
    cublasSgemv(_cublas_handle, CUBLAS_OP_N, H, M, &alpha, _W1, H, _x, 1, &beta, _z1, 1);
    cublasSaxpy(_cublas_handle, H, &alpha, _b1, 1, _z1, 1);

    // a1[H] = ReLU(z1[H])
    relu(_z1, _a1, 1, H);

    // y[N] = W2[N,H] @ a1[H] + b2[N]
    cublasSgemv(_cublas_handle, CUBLAS_OP_N, N, H, &alpha, _W2, N, _a1, 1, &beta, _y, 1);
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

    // dW2[N,H] = dy[N] @ a1[H]^T
    cublasSger(_cublas_handle, N, H, &alpha, _dy, 1, _a1, 1, _dW2, N);
    // db2[N] = dy[N]
    cublasScopy(_cublas_handle, N, _dy, 1, _db2, 1);

    // da1[H] = W2[N,H]^T @ dy[N]
    cublasSgemv(_cublas_handle, CUBLAS_OP_T, N, H, &alpha, _W2, N, _dy, 1, &beta, _da1, 1);
    // dz1[H] = da1[H] * ReLU'(z1[H])
    relu_backwards(_da1, _z1, _dz1, 1, H);

#if 0
    std::vector<float> z1_vals(H);
    _z1.to_host(std::span<float>(z1_vals.data(), H));
    int positive = 0;
    for (float val : z1_vals)
        if (val > 0)
            positive++;
    printf("Positive z1: %d/%d (%.1f%%)\n", positive, H, 100.0f * positive / H);
#endif

    // dW1[H,M] = dz1[H] @ x[M]^T
    cublasSger(_cublas_handle, H, M, &alpha, _dz1, 1, _x, 1, _dW1, H);
    // db1[H] = dz1[H]
    cublasScopy(_cublas_handle, H, _dz1, 1, _db1, 1);

    // update weights and biases
    const float neg_lr = -learning_rate;
    cublasSaxpy(_cublas_handle, _W1.size(), &neg_lr, _dW1, 1, _W1, 1);
    cublasSaxpy(_cublas_handle, _b1.size(), &neg_lr, _db1, 1, _b1, 1);
    cublasSaxpy(_cublas_handle, _W2.size(), &neg_lr, _dW2, 1, _W2, 1);
    cublasSaxpy(_cublas_handle, _b2.size(), &neg_lr, _db2, 1, _b2, 1);

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
    std::vector<f32> cpu_output(N);
    _y.to_host(cpu_output);

    output[0] = (u8)(std::max_element(cpu_output.begin(), cpu_output.end()) - cpu_output.begin());
}
