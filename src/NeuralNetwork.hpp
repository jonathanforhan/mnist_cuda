#pragma once

#include "MNISTImage.hpp"
#include "Tensor.hpp"
#include <cublas_v2.h>

void xavier_init(float* weights, int n);
void zero_init(float* data, int n);
void launch_normalize_mnist(const uint8_t* input, float* output, size_t n);
void linear_forward(const float* weights, const float* input, float* output, int input_size, int output_size);
void add_bias(const float* x, const float* bias, float* y, int n);
void relu_activation(const float* x, float* y, int n);
void softmax_cross_entropy(const float* logits, int true_label, float* loss, float* gradients, int n);
void compute_loss_and_gradients(const float* output, int true_label, float* loss, float* output_gradients, int n);
void compute_loss_and_gradients(const float* output, int true_label, float* loss, float* output_gradients, int n);
void compute_weight_gradients(const float* input,
                              const float* output_gradients,
                              float* weight_gradients,
                              int input_size,
                              int output_size);
void update_weights(float* weights, const float* gradients, float learning_rate, int n);

class NeuralNetwork {
public:
    NeuralNetwork();

    ~NeuralNetwork() noexcept;

    void forward(const MNISTImage& mnist_image);

    f32 backward(int true_label, f32 learning_rate = 0.01f);

    void train_step(const MNISTImage& mnist_image, f32 learning_rate = 0.01f);

    int predict();

private:
    Tensor<f32, 28 * 28> _x;     /* input */
    Tensor<f32, 28 * 28, 10> _W; /* weights */
    Tensor<f32, 10> _b;          /* biases */
    Tensor<f32, 10> _y;          /* output */

    Tensor<f32, 28 * 28, 10> _dW; /* weight gradients */
    Tensor<f32, 10> _db;          /* bias gradients */
    Tensor<f32, 10> _dy;          /* output gradients */
    Tensor<f32, 1> _loss;         /* loss tensor */

    cublasHandle_t _cublas_handle;
};
