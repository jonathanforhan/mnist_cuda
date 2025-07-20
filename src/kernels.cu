#include "tensor.hpp"
#include <curand_kernel.h>

// Convert uint8 MNIST pixels [0,255] to normalized float [0,1]
__global__ void normalize_mnist(const uint8_t* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] / 255.0f;
    }
}

// Host wrapper
void launch_normalize_mnist(const uint8_t* input, float* output, size_t n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    normalize_mnist<<<grid, block>>>(input, output, n);
}

// Matrix-vector multiplication: y = A * x
// A is m x n matrix, x is n-vector, y is m-vector
__global__ void matvec_kernel(const float* A, const float* x, float* y, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        float sum = 0.0f;
        for (int col = 0; col < n; col++) {
            sum += A[row * n + col] * x[col]; // Row-major indexing
        }
        y[row] = sum;
    }
}

// Host wrapper
void linear_forward(const float* weights, const float* input, float* output, int input_size, int output_size) {
    dim3 block(256);
    dim3 grid((output_size + block.x - 1) / block.x);
    matvec_kernel<<<grid, block>>>(weights, input, output, output_size, input_size);
}

// Add bias to each element: y = x + bias
__global__ void add_bias_kernel(const float* x, const float* bias, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] + bias[idx];
    }
}

// Host wrappers
void add_bias(const float* x, const float* bias, float* y, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    add_bias_kernel<<<grid, block>>>(x, bias, y, n);
}

// ReLU activation: y = max(0, x)
__global__ void relu_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

void relu_activation(const float* x, float* y, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    relu_kernel<<<grid, block>>>(x, y, n);
}

// Initialize weights with Xavier/Glorot initialization
__global__ void xavier_init_kernel(float* weights, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Xavier initialization: std = sqrt(2.0 / (fan_in + fan_out))
        // For simplicity, we'll use std = 0.1
        weights[idx] = curand_normal(&state) * 0.1f;
    }
}

// Initialize bias to zero
__global__ void zero_init_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 0.0f;
    }
}

// Host wrappers
void xavier_init(float* weights, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    xavier_init_kernel<<<grid, block>>>(weights, n, time(nullptr));
}

void zero_init(float* data, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    zero_init_kernel<<<grid, block>>>(data, n);
}

// Softmax + Cross-entropy loss (combined for numerical stability)
__global__ void softmax_cross_entropy_kernel(const float* logits,
                                             int true_label,
                                             float* loss,
                                             float* gradients,
                                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Softmax with numerical stability
        float max_logit = logits[0];
        for (int i = 1; i < n; i++) {
            max_logit = fmaxf(max_logit, logits[i]);
        }

        float exp_sum = 0.0f;
        for (int i = 0; i < n; i++) {
            exp_sum += expf(logits[i] - max_logit);
        }

        float softmax_output = expf(logits[idx] - max_logit) / exp_sum;

        // Cross-entropy gradient: softmax_output - true_label_one_hot
        gradients[idx] = softmax_output - (idx == true_label ? 1.0f : 0.0f);

        // Loss (only compute once)
        if (idx == 0) {
            *loss = -logf(expf(logits[true_label] - max_logit) / exp_sum);
        }
    }
}

// Host wrapper
void softmax_cross_entropy(const float* logits, int true_label, float* loss, float* gradients, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    softmax_cross_entropy_kernel<<<grid, block>>>(logits, true_label, loss, gradients, n);
}

// Compute softmax loss and gradients
__global__ void compute_loss_and_gradients_kernel(const float* output,
                                                  int true_label,
                                                  float* loss,
                                                  float* output_gradients,
                                                  int n) {
    // Find max for numerical stability
    float max_val = output[0];
    for (int i = 1; i < n; i++) {
        max_val = fmaxf(max_val, output[i]);
    }

    // Compute exp sum
    float exp_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        exp_sum += expf(output[i] - max_val);
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Softmax probability
        float prob = expf(output[idx] - max_val) / exp_sum;

        // Gradient = probability - true_label_one_hot
        output_gradients[idx] = prob - (idx == true_label ? 1.0f : 0.0f);

        // Store loss (only thread 0)
        if (idx == 0) {
            float true_prob = expf(output[true_label] - max_val) / exp_sum;
            *loss           = -logf(true_prob);
        }
    }
}

// Compute weight gradients: grad_W[i,j] = input[j] * output_grad[i]
__global__ void compute_weight_gradients_kernel(const float* input,
                                                const float* output_gradients,
                                                float* weight_gradients,
                                                int input_size,
                                                int output_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // output neuron
    int col = blockIdx.x * blockDim.x + threadIdx.x; // input neuron

    if (row < output_size && col < input_size) {
        int idx               = row * input_size + col;
        weight_gradients[idx] = input[col] * output_gradients[row];
    }
}

// Update weights: w = w - learning_rate * gradient
__global__ void update_weights_kernel(float* weights, const float* gradients, float learning_rate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

// Host wrappers
void compute_loss_and_gradients(const float* output, int true_label, float* loss, float* output_gradients, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    compute_loss_and_gradients_kernel<<<grid, block>>>(output, true_label, loss, output_gradients, n);
}

void compute_weight_gradients(const float* input,
                              const float* output_gradients,
                              float* weight_gradients,
                              int input_size,
                              int output_size) {
    dim3 block(16, 16);
    dim3 grid((input_size + block.x - 1) / block.x, (output_size + block.y - 1) / block.y);
    compute_weight_gradients_kernel<<<grid, block>>>(
        input, output_gradients, weight_gradients, input_size, output_size);
}

void update_weights(float* weights, const float* gradients, float learning_rate, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    update_weights_kernel<<<grid, block>>>(weights, gradients, learning_rate, n);
}