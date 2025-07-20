#pragma once

#include "log.hpp"
#include <array>
#include <cassert>
#include <cuda_runtime.h>
#include <span>
#include <stdexcept>

template <typename T, size_t R>
requires(R != 0)
class Tensor {
public:
    using size_type = std::array<size_t, R>;
    using data_type = T;

public:
    template <typename... Dimensions>
    requires(sizeof...(Dimensions) == R)
    Tensor(Dimensions... dimensions) {
        _size     = size_type{static_cast<size_t>(dimensions)...};
        _capacity = (static_cast<size_t>(dimensions) * ...);

        if (auto err = cudaMalloc(&_data, capacity() * sizeof(data_type))) {
            ELOG("cudaMalloc failure %d", (int)err);
            throw std::runtime_error{"cudaMalloc failure"};
        }
    }

    ~Tensor() noexcept { cudaFree(_data); }

    Tensor(const Tensor& other) = delete;

    Tensor& operator=(const Tensor& other) = delete;

    Tensor(Tensor&& other) noexcept
        : _data{other._data},
          _size{other._size},
          _capacity{other._capacity} {
        other._data     = nullptr;
        other._size     = {};
        other._capacity = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            cudaFree(_data);

            _data     = other._data;
            _size     = other._size;
            _capacity = other._capacity;

            other._data     = nullptr;
            other._size     = {};
            other._capacity = 0;
        }
        return *this;
    }

    size_t capacity() const { return _capacity; }

    size_type size() const { return _size; }

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

    operator data_type*() { return _data; }

    operator const data_type*() const { return _data; }

private:
    data_type* _data;
    size_type _size;
    size_t _capacity;
};