#pragma once

#include <cublas_v2.h>
#include <span>
#include "MNISTImporter.hpp"
#include "Tensor.hpp"

class NeuralNetwork {
public:
    NeuralNetwork(int batch_size = 32, int hidden_size = 128);

    ~NeuralNetwork() noexcept;

    NeuralNetwork(const NeuralNetwork&) = delete;

    NeuralNetwork& operator=(const NeuralNetwork&) = delete;

    NeuralNetwork(NeuralNetwork&&) noexcept = default;

    NeuralNetwork& operator=(NeuralNetwork&&) noexcept = default;

    void forward(std::span<const MNISTImage> mnist_images);

    f32 backward(std::span<const u8> true_labels, f32 learning_rate = 0.01f);

    f32 train(std::span<const MNISTImage> mnist_images, std::span<const u8> mnist_labels, f32 learning_rate = 0.01f);

    void predict(std::span<u8> output);

private:
    const int B; /* batch size */
    const int H; /* hidden layer size */
    const int M = 28 * 28;
    const int N = 10;

    Tensor<f32> _x{M};
    Tensor<f32> _y{N};
    Tensor<f32> _W{M, N};
    Tensor<f32> _b{N};

    Tensor<f32> _dy{N};
    Tensor<f32> _dW{M, N};
    Tensor<f32> _db{N};
    Tensor<f32> _loss{1};
    Tensor<u8> _labels{1};
    Tensor<u8> _raw{M};

    cublasHandle_t _cublas_handle;
};
