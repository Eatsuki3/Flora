#ifndef FLORA_GPU_RUNTIME_CUH
#define FLORA_GPU_RUNTIME_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

#define FLORA_CUDA_CHECK(expr)                                             \
    do {                                                                   \
        cudaError_t flora_cuda_error__ = (expr);                           \
        if (flora_cuda_error__ != cudaSuccess) {                           \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",               \
                __FILE__, __LINE__,                                        \
                cudaGetErrorString(flora_cuda_error__));                   \
            std::abort();                                                  \
        }                                                                  \
    } while (0)

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) {
        reserve(count);
    }

    ~DeviceBuffer() {
        reset();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept {
        swap(other);
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }

    void reserve(std::size_t count) {
        if (count <= capacity_) {
            return;
        }

        T* next = nullptr;
        FLORA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&next), count * sizeof(T)));

        if (ptr_ != nullptr) {
            FLORA_CUDA_CHECK(cudaFree(ptr_));
        }

        ptr_ = next;
        capacity_ = count;
    }

    void reset() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        capacity_ = 0;
    }

    T* data() {
        return ptr_;
    }

    const T* data() const {
        return ptr_;
    }

    std::size_t capacity() const {
        return capacity_;
    }

    void swap(DeviceBuffer& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(capacity_, other.capacity_);
    }

private:
    T* ptr_ = nullptr;
    std::size_t capacity_ = 0;
};

template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(std::size_t count) {
        reserve(count);
    }

    ~PinnedBuffer() {
        reset();
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void reserve(std::size_t count) {
        if (count <= capacity_) { return; }
        reset();

        void* raw = nullptr;
        cudaError_t error = cudaHostAlloc(&raw, count * sizeof(T), cudaHostAllocPortable);

        if (error == cudaSuccess) {
            ptr_ = static_cast<T*>(raw);
            pinned_ = true;
        } else {
            cudaGetLastError();
            ptr_ = static_cast<T*>(std::malloc(count * sizeof(T)));
            pinned_ = false;

            if (ptr_ == nullptr) {
                std::fprintf(stderr, "[HOST ALLOC ERROR] count=%zu\n", count);
                std::abort();
            }
        }

        capacity_ = count;
    }

    void reset() {
        if (ptr_ == nullptr) {
            return;
        }

        if (pinned_) {
            cudaFreeHost(ptr_);
        } else {
            std::free(ptr_);
        }

        ptr_ = nullptr;
        capacity_ = 0;
        pinned_ = false;
    }

    T* data() {
        return ptr_;
    }

    const T* data() const {
        return ptr_;
    }

    std::size_t capacity() const {
        return capacity_;
    }

private:
    T* ptr_ = nullptr;
    std::size_t capacity_ = 0;
    bool pinned_ = false;
};

class GpuRuntime {
public:
    GpuRuntime() {
        FLORA_CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking));
        FLORA_CUDA_CHECK(cudaStreamCreateWithFlags(&transfer_stream_, cudaStreamNonBlocking));
        FLORA_CUDA_CHECK(cudaEventCreateWithFlags(&compute_ready_, cudaEventDisableTiming));
        FLORA_CUDA_CHECK(cudaEventCreateWithFlags(&transfer_ready_, cudaEventDisableTiming));
    }

    ~GpuRuntime() {
        if (compute_ready_ != nullptr) {
            cudaEventDestroy(compute_ready_);
        }

        if (transfer_ready_ != nullptr) {
            cudaEventDestroy(transfer_ready_);
        }

        if (compute_stream_ != nullptr) {
            cudaStreamDestroy(compute_stream_);
        }

        if (transfer_stream_ != nullptr) {
            cudaStreamDestroy(transfer_stream_);
        }
    }

    GpuRuntime(const GpuRuntime&) = delete;
    GpuRuntime& operator=(const GpuRuntime&) = delete;

    cudaStream_t computeStream() const {
        return compute_stream_;
    }

    cudaStream_t transferStream() const {
        return transfer_stream_;
    }

    cudaEvent_t computeReadyEvent() const {
        return compute_ready_;
    }

    cudaEvent_t transferReadyEvent() const {
        return transfer_ready_;
    }

    void synchronize() {
        FLORA_CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        FLORA_CUDA_CHECK(cudaStreamSynchronize(transfer_stream_));
    }

private:
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t transfer_stream_ = nullptr;
    cudaEvent_t compute_ready_ = nullptr;
    cudaEvent_t transfer_ready_ = nullptr;
};

#endif