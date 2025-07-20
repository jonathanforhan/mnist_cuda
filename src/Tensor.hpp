#pragma once

#include "log.hpp"
#include <array>
#include <cassert>
#include <cuda_runtime.h>
#include <span>
#include <stdexcept>

template <typename T, size_t... Dimensions>
class Tensor {
public:
    using size_type = std::array<size_t, sizeof...(Dimensions)>;
    using data_type = T;

public:
    Tensor() {
        if (auto err = cudaMalloc(&_data, capacity() * sizeof(data_type))) {
            ELOG("cudaFree failure %d", (int)err);
            throw std::runtime_error{"cudaMalloc failure"};
        }
    }

    ~Tensor() noexcept {
        if (auto err = cudaFree(_data)) {
            ELOG("cudaFree failure %d", (int)err);
            std::exit(EXIT_FAILURE);
        }
    }

    Tensor(const Tensor& other) {
        if (auto err = cudaMalloc(&_data, capacity() * sizeof(data_type))) {
            ELOG("cudaFree failure %d", (int)err);
            throw std::runtime_error{"cudaMalloc failure"};
        }

        if (auto err = cudaMemcpy(_data, other._data, capacity() * sizeof(data_type), cudaMemcpyDeviceToDevice)) {
            ELOG("cudaMemcpy failure %d", (int)err);
            throw std::runtime_error("cudaMemcpy failure");
        }
    }

    Tensor& operator=(const Tensor& other) {
        if (auto err = cudaMemcpy(_data, other._data, capacity() * sizeof(data_type), cudaMemcpyDeviceToDevice)) {
            ELOG("cudaMemcpy failure %d", (int)err);
            throw std::runtime_error("cudaMemcpy failure");
        }
        return *this;
    }

    Tensor(Tensor&& other) noexcept {
        this->_data = other._data;
        other._data = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        this->_data = other._data;
        other._data = nullptr;
        return *this;
    }

    constexpr size_t capacity() const { return (Dimensions * ...); }

    constexpr size_type size() const { return size_type{Dimensions...}; }

    void from_host(std::span<const data_type> host_data) {
        assert(host_data.size() == capacity());
        if (auto err = cudaMemcpy(_data, host_data.data(), capacity() * sizeof(data_type), cudaMemcpyHostToDevice)) {
            ELOG("cudaMemcpy failure %d host: %p device: %p", (int)err, host_data.data(), _data);
            throw std::runtime_error("cudaMemcpy failure");
        }
    }

    void to_host(std::span<data_type> host_data) const {
        assert(host_data.size() == capacity());
        if (auto err = cudaMemcpy(host_data.data(), _data, capacity() * sizeof(data_type), cudaMemcpyDeviceToHost)) {
            ELOG("cudaMemcpy failure %d host: %p device: %p", (int)err, host_data.data(), _data);
            throw std::runtime_error("cudaMemcpy failure");
        }
    }

    data_type* data() { return _data; }

    const data_type* data() const { return _data; }

private:
    data_type* _data;
};