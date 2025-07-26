#include "kernels.h"

#include <curand_kernel.h>
#include <cmath>
#include <ctime>
#include "tensor.hpp"

__global__ void normalize_u8_to_f32_kernel(const u8* input, f32* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = input[idx] / 255.0f;
    }
}

void normalize_u8_to_f32(const u8* input, f32* output, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    normalize_u8_to_f32_kernel<<<grid, block>>>(input, output, n);
}

__global__ void init_kernel(f32* W, int n, int fan_in, int fan_out, ulong seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        f32 std_dev = sqrtf(2.0f / (fan_in + fan_out));
        W[idx]      = curand_normal(&state) * std_dev;
    }
}

void xavier_init(f32* W, int n, int fan_in, int fan_out) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    init_kernel<<<grid, block>>>(W, n, fan_in, fan_out, (ulong)time(nullptr));
}

void kaiming_init(f32* W, int n, int fan_in) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    init_kernel<<<grid, block>>>(W, n, fan_in, 0, (ulong)time(nullptr));
}

void zero_init(f32* v, int n) {
    (void)cudaMemset(v, 0, n * sizeof(*v));
}

__global__ void add_bias_kernel(const f32* x, const f32* b, f32* y, int batch_size, int features) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int feat_idx  = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && feat_idx < features) {
        int idx = batch_idx * features + feat_idx;
        y[idx]  = x[idx] + b[feat_idx];
    }
}

void add_bias(const f32* x, const f32* b, f32* y, int batch_size, int features) {
    dim3 block(16, 16);
    dim3 grid((features + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
    add_bias_kernel<<<grid, block>>>(x, b, y, batch_size, features);
}

__global__ void relu_kernel(const f32* x, f32* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

void relu(const f32* x, f32* y, int batch_size, int features) {
    int n = batch_size * features;
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    relu_kernel<<<grid, block>>>(x, y, n);
}

__global__ void relu_backwards_kernel(const f32* dy, const f32* x, f32* dx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        dx[idx] = (x[idx] > 0.0f) ? dy[idx] : 0.0f;
    }
}

void relu_backwards(const f32* dy, const f32* x, f32* dx, int batch_size, int features) {
    int n = batch_size * features;
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    relu_backwards_kernel<<<grid, block>>>(dy, x, dx, n);
}

__global__ void softmax_cross_entropy_kernel(const f32* logits,
                                             const u8* true_labels,
                                             f32* loss,
                                             f32* gradients,
                                             int batch_size,
                                             int num_classes) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
        const f32* batch_logits = logits + batch_idx * num_classes;
        f32* batch_gradients    = gradients + batch_idx * num_classes;
        u8 true_label           = true_labels[batch_idx];

        /* find max in batch */
        f32 max_logit = batch_logits[0];
        for (int i = 1; i < num_classes; i++) {
            max_logit = fmaxf(max_logit, batch_logits[i]);
        }

        /* compute exp sum */
        f32 exp_sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            exp_sum += expf(batch_logits[i] - max_logit);
        }

        /* compute softmax */
        for (int i = 0; i < num_classes; i++) {
            f32 prob           = expf(batch_logits[i] - max_logit) / exp_sum;
            batch_gradients[i] = prob - (i == true_label ? 1.0f : 0.0f);
        }

        /* compute loss */
        f32 true_prob   = expf(batch_logits[true_label] - max_logit) / exp_sum;
        loss[batch_idx] = -logf(true_prob);
    }
}

void softmax_cross_entropy(const f32* logits,
                           const u8* true_labels,
                           f32* loss,
                           f32* gradients,
                           int batch_size,
                           int num_classes) {
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);
    softmax_cross_entropy_kernel<<<grid, block>>>(logits, true_labels, loss, gradients, batch_size, num_classes);
}

__global__ void update_weights_momentum_kernel(f32* W,
                                               const f32* gradients,
                                               f32* momentums,
                                               f32 learning_rate,
                                               f32 beta,
                                               int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Update momentum: v = beta * v + grad
        momentums[idx] = beta * momentums[idx] + gradients[idx];

        // Update weights: w = w - lr * v
        W[idx] -= learning_rate * momentums[idx];
    }
}

void update_weights_momentum(f32* W, const f32* gradients, f32* momentums, f32 learning_rate, f32 beta, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    update_weights_momentum_kernel<<<grid, block>>>(W, gradients, momentums, learning_rate, beta, n);
}

__global__ void compute_bias_gradients_kernel(const f32* dy, f32* db, int batch_size, int features) {
    int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (feat_idx < features) {
        f32 sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += dy[i * features + feat_idx];
        }
        db[feat_idx] = sum / batch_size;
    }
}

void compute_bias_gradients(const f32* dy, f32* db, int batch_size, int features) {
    dim3 block(256);
    dim3 grid((features + block.x - 1) / block.x);
    compute_bias_gradients_kernel<<<grid, block>>>(dy, db, batch_size, features);
}

__global__ void dropout_forward_kernel(const f32* x, f32* y, f32* mask, f32 dropout_rate, int n, ulong seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        f32 rand_val = curand_uniform(&state);
        if (rand_val < dropout_rate) {
            mask[idx] = 0.0f;
            y[idx]    = 0.0f;
        } else {
            mask[idx] = 1.0f / (1.0f - dropout_rate);
            y[idx]    = x[idx] * mask[idx];
        }
    }
}

void dropout_forward(const f32* x, f32* y, f32* mask, f32 dropout_rate, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    dropout_forward_kernel<<<grid, block>>>(x, y, mask, dropout_rate, n, (ulong)time(nullptr));
}

__global__ void dropout_backward_kernel(const f32* dy, const f32* mask, f32* dx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        dx[idx] = dy[idx] * mask[idx];
    }
}

void dropout_backward(const f32* dy, const f32* mask, f32* dx, int n) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    dropout_backward_kernel<<<grid, block>>>(dy, mask, dx, n);
}
