#pragma once

#include <cuda_runtime.h>
#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

/**
 * @brief Stores GPU Tensor data
 * @tparam T Type of data
 *
 * @code
 * Tensor<float> tensor(10, 10, 20); // 10x10x20 GPU tensor
 * Tensor<float> tensor(10, 10, 20); // 10x10x20 GPU tensor
 * assert(tensor.size() == 10 * 10 * 20);
 * assert(tensor.size_bytes() == 10 * 10 * 20 * sizeof(Tensor<float>::data_type));
 * assert(tensor.dim(0) == 10);
 * assert(tensor.dim(1) == 10);
 * assert(tensor.dim(2) == 20);
 * @endcode
 */
template <typename T>
requires(std::is_arithmetic<T>::value)
class Tensor {
private:
    static constexpr int MAX_DIMENSIONS = 8;

public:
    using size_type = int;
    using data_type = T;

public:
    template <typename... Dimensions>
    requires(std::convertible_to<Dimensions, size_type> && ...)
    explicit Tensor(Dimensions... dimensions)
    requires(sizeof...(dimensions) > 0 && sizeof...(dimensions) < MAX_DIMENSIONS)
    {
        if (!((dimensions != 0) && ...)) {
            throw std::invalid_argument{"all dimensions must be positive"};
        }

        _size = (static_cast<size_type>(dimensions) * ...);
        _dims = {static_cast<size_type>(dimensions)...};

        if (auto err = cudaMalloc(&_data, size_bytes())) {
            throw std::runtime_error{"cudaMalloc failure " + std::to_string((int)err)};
        }
    }

    ~Tensor() noexcept { cudaFree(_data); }

    Tensor(const Tensor& other) = delete;

    Tensor& operator=(const Tensor& other) = delete;

    Tensor(Tensor&& other) noexcept
        : _data{other._data},
          _size{other._size},
          _dims{other._dims} {
        other._data = nullptr;
        other._size = 0;
        other._dims = {};
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            cudaFree(_data);

            _data = other._data;
            _size = other._size;
            _dims = other._dims;

            other._data = nullptr;
            other._size = 0;
            other._dims = {};
        }
        return *this;
    }

    /**
     * @brief number of members T in tensor
     * @return the product of the dimenions
     */
    size_type size() const noexcept { return _size; }

    /**
     * @brief number of bytes in tensor
     * @return product of the dimensions and sizeof(T)
     */
    size_type size_bytes() const noexcept { return _size * sizeof(T); }

    /**
     * @brief dimenion at ith index
     * @param i index of dimension queried
     * @return ith dimensions
     */
    size_type dim(int i) const noexcept { return i >= MAX_DIMENSIONS ? 0 : _dims[i]; }

    /**
     * @brief copy memory from host to device
     * @param host_data CPU data
     */
    void from_host(std::span<const data_type> host_data) {
        if (host_data.size_bytes() != size_bytes()) {
            throw std::invalid_argument{"cannot copy data with different sizes"};
        } else if (auto err = cudaMemcpy(_data, host_data.data(), size_bytes(), cudaMemcpyHostToDevice)) {
            throw std::runtime_error{"cudaMemcpy failure " + std::to_string((int)err)};
        }
    }

    /**
     * @brief copy memory from device to device
     * @param data GPU data pointer
     * @param bytes memory size in bytes of GPU data
     */
    void from_device(data_type* data, size_type bytes) {
        if (bytes != size_bytes()) {
            throw std::invalid_argument{"cannot copy data with different sizes"};
        } else if (auto err = cudaMemcpy(_data, data, size_bytes(), cudaMemcpyDeviceToDevice)) {
            throw std::runtime_error{"cudaMemcpy failure " + std::to_string((int)err)};
        }
    }

    /**
     * @brief copy memory from device to host
     * @param host_data output CPU data
     */
    void to_host(std::span<data_type> host_data) const {
        if (host_data.size_bytes() != size_bytes()) {
            throw std::invalid_argument{"cannot copy data with different sizes"};
        } else if (auto err = cudaMemcpy(host_data.data(), _data, size_bytes(), cudaMemcpyDeviceToHost)) {
            throw std::runtime_error{"cudaMemcpy failure " + std::to_string((int)err)};
        }
    }

    /** @brief internal data pointer */
    data_type* data() noexcept { return _data; }

    /** @brief internal data pointer */
    const data_type* data() const noexcept { return _data; }

    /** @brief internal data pointer cast */
    operator data_type*() noexcept { return _data; }

    /** @brief internal data pointer cast */
    operator const data_type*() const noexcept { return _data; }

private:
    data_type* _data;
    size_type _size;
    std::array<size_type, MAX_DIMENSIONS> _dims;
};
