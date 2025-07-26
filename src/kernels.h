#pragma once

#include "types.hpp"

/**
 * @brief Normalize from 0-255 to 0.0-1.0
 * @param input     u8[n]
 * @param output    f32[n]
 * @param n length of input/output tensor
 */
void normalize_u8_to_f32(const u8* input, f32* output, int n);

/**
 * @brief Initialize W using Xavier Glorot method
 * @param W weights     f32[n]
 * @param n length of W
 * @param fan_in number of inputs from previous layer
 * @param fan_out number of outputs to next layer
 */
void xavier_init(f32* W, int n, int fan_in, int fan_out);

/**
 * @brief Initialize W using Kaiming He method
 * @param W weights     f32[n]
 * @param n length of W
 * @param fan_in number of inputs from previous layer
 */
void kaiming_init(f32* W, int n, int fan_in);

/**
 * @brief Initialize W using
 * @param W weights     f32[n]
 * @param n length of W
 * @param fan_in number of inputs from previous layer
 */
void zero_init(f32* v, int n);

/**
 * @brief y = x + b
 * @param x input   f32[batch, features]
 * @param b bias    f32[features]
 * @param y output  f32[batch, features]
 * @param batch_size number of batches in x and y
 * @param features the dimension of the bias tensor
 */
void add_bias(const f32* x, const f32* b, f32* y, int batch_size, int features);

/**
 * @brief y = max(0, x)
 * @param x input   f32[batch_size, features]
 * @param y output  f32[batch_size, features]
 * @param batch_size number of batches in x and y
 * @param features the dimension of x and y tensors
 */
void relu(const f32* x, f32* y, int batch_size, int features);

/**
 * @brief dx = dy * ReLU'(x)
 * @param dy output gradient    f32[batch_size, features]
 * @param x  input              f32[batch_size, features]
 * @param dx input gradient     f32[batch_size, features]
 * @param batch_size number of batches
 * @param features the dimension of dy, x and dx tensors
 */
void relu_backwards(const f32* dy, const f32* x, f32* dx, int batch_size, int features);

/**
 * @brief Converts vector into probability distribution
 * @param logits unnormalized outputs   f32[batch_size, num_classes]
 * @param true_labels true values       u8[batch_size]
 * @param loss
 * @param gradients
 * @param batch_size
 * @param num_classes
 */
void softmax_cross_entropy(const f32* logits,
                           const u8* true_labels,
                           f32* loss,
                           f32* gradients,
                           int batch_size,
                           int num_classes);

/**
 * @brief Update the momentum of weights in grad descent to overcome local extrema
 * @param W weights     f32[n]
 * @param gradients     f32[n]
 * @param momentums     f32[n]
 * @param learning_rate
 * @param beta momentum coefficient
 * @param n dimension of weights, gradients and momentums
 */
void update_weights_momentum(f32* W, const f32* gradients, f32* momentums, f32 learning_rate, f32 beta, int n);

/**
 * @brief Get the derivative of the biases using the output gradients
 * @param dy output gradients   f32[batch_size, features]
 * @param db bias gradients     f32[batch_size, features]
 * @param batch_size
 * @param features dimension of dy.x and db.x
 */
void compute_bias_gradients(const f32* dy, f32* db, int batch_size, int features);

/**
 * @brief Dropout to prevent overfitting
 * @param x inputs  f32[n]
 * @param y outputs f32[n]
 * @param mask mask output based on dropout_rate
 * @param dropout_rate normalized value that determines rate of dropout
 * @param n dimension of x and y
 */
void dropout_forward(const f32* x, f32* y, f32* mask, f32 dropout_rate, int n);

/**
 * @brief Dropout to prevent overfitting
 * @param dy output gradients               f32[n]
 * @param mask mask from dropout_forward    f32[n]
 * @param dx input gradients                f32[n]
 * @param n dimensions of dy, dx, mask
 */
void dropout_backward(const f32* dy, const f32* mask, f32* dx, int n);
