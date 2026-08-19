#include "gpu_online.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace {

#define CUDA_CHECK(expr)                                                   \
    do {                                                                   \
        cudaError_t error__ = (expr);                                      \
        if (error__ != cudaSuccess) {                                      \
            std::fprintf(                                                  \
                stderr,                                                    \
                "[CUDA ERROR] %s:%d: %s\n",                                \
                __FILE__,                                                  \
                __LINE__,                                                  \
                cudaGetErrorString(error__));                              \
            std::abort();                                                  \
        }                                                                  \
    } while (0)

constexpr int kThreads = 256;
constexpr int kAlice = 1;

inline int gridFor(uint64_t count) {
    if (count == 0) { return 0; }
    uint64_t blocks = (count + static_cast<uint64_t>(kThreads) - 1) / static_cast<uint64_t>(kThreads);
    if (blocks > static_cast<uint64_t>(0x7fffffff)) {
        std::fprintf(stderr, "CUDA grid too large\n");
        std::abort();
    }
    return static_cast<int>(blocks);
}

template <typename T>
struct DeviceBuffer {
    T* pointer = nullptr;
    size_t capacity = 0;

    ~DeviceBuffer() {
        release();
    }

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void reserve(size_t count) {
        if (count <= capacity) { return; }

        T* replacement = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&replacement), count * sizeof(T)));
        if (pointer != nullptr) {
            CUDA_CHECK(cudaFree(pointer));
        }

        pointer = replacement;
        capacity = count;
    }

    void release() {
        if (pointer != nullptr) {
            cudaFree(pointer);
            pointer = nullptr;
        }

        capacity = 0;
    }

    void swap(DeviceBuffer& other) {
        std::swap(pointer, other.pointer);
        std::swap(capacity, other.capacity);
    }
};

struct HostPinnedBuffer {
    void* pointer = nullptr;
    size_t capacity_bytes = 0;
    bool pinned = false;

    ~HostPinnedBuffer() {
        release();
    }

    HostPinnedBuffer() = default;

    HostPinnedBuffer(const HostPinnedBuffer&) = delete;
    HostPinnedBuffer& operator=(const HostPinnedBuffer&) = delete;

    void reserve(size_t bytes) {
        if (bytes <= capacity_bytes) {
            return;
        }
        release();

        void* allocated = nullptr;
        cudaError_t error = cudaHostAlloc(&allocated, bytes, cudaHostAllocPortable);
        if (error == cudaSuccess) {
            pointer = allocated;
            pinned = true;
        } else {
            cudaGetLastError();
            pointer = std::malloc(bytes);
            pinned = false;

            if (pointer == nullptr) {
                std::fprintf(stderr, "Unable to allocate %zu host bytes\n", bytes);
                std::abort();
            }
        }
        capacity_bytes = bytes;
    }

    void release() {
        if (pointer == nullptr) { return; }

        if (pinned) {
            cudaFreeHost(pointer);
        } else {
            std::free(pointer);
        }

        pointer = nullptr;
        capacity_bytes = 0;
        pinned = false;
    }
};

__device__ __forceinline__ uint32_t gpuPrng(uint64_t seed, uint64_t index, uint32_t domain) {
    uint64_t value = (index * 3202034522624059713ULL + seed) ^ static_cast<uint64_t>(domain);

    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;

    return static_cast<uint32_t>(value);
}

__device__ __forceinline__ uint32_t gfMul(uint32_t x, uint32_t y) {

    constexpr uint32_t reduction = 0x8du;
    uint64_t product = 0;
    uint64_t wide_x = static_cast<uint64_t>(x);

#pragma unroll
    for (int bit = 0; bit < 32; ++bit) {
        uint64_t mask = static_cast<uint64_t>(0) - static_cast<uint64_t>((y >> bit) & 1u);
        product ^= (wide_x << bit) & mask;
    }

#pragma unroll
    for (int bit = 62; bit >= 32; --bit) {
        if ((product >> bit) & 1ULL) {
            product ^= 1ULL << bit;
            product ^= static_cast<uint64_t>(reduction) << (bit - 32);
        }
    }
    return static_cast<uint32_t>(product);
}

__device__ __forceinline__ uint32_t partyShare(uint32_t full_value, uint32_t mask, int party) {
    return party == kAlice ? mask : full_value ^ mask;
}

// ============================================================
// NeighborPass kernels
// ============================================================
// __global__ void neighborPrepareKernel(
//     const uint32_t* __restrict__ global_s, uint32_t* __restrict__ de_local,
//     uint64_t active_num, uint64_t offset, int B, int party, uint64_t nonce) {
//     uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
//     uint64_t total = active_num * static_cast<uint64_t>(B);

//     if (index >= total) { return; }

//     uint64_t tuple = index / static_cast<uint64_t>(B);
//     int field = static_cast<int>(index % static_cast<uint64_t>(B));

//     uint64_t left = tuple;
//     uint64_t right = tuple + offset;

//     uint32_t x_share;
//     uint32_t y_share;

//     if (field == 0) {
//         x_share = global_s[left * B];
//         y_share = global_s[right * B];
//     } else {
//         uint32_t right_flag = global_s[right * B];
//         uint32_t not_right_flag = right_flag ^ ((party == kAlice) ? 0xffffffffu : 0u);

//         x_share = not_right_flag;
//         y_share = global_s[left * B + field];
//     }

//     uint64_t triple_index = nonce + index;

//     uint32_t a_full = gpuPrng(0xa5d70001ULL, triple_index, 1);
//     uint32_t b_full = gpuPrng(0xa5d70001ULL, triple_index, 2);
//     uint32_t a_mask = gpuPrng(0xa5d70001ULL, triple_index, 3);
//     uint32_t b_mask = gpuPrng(0xa5d70001ULL, triple_index, 4);
//     uint32_t a_share = partyShare(a_full, a_mask, party);
//     uint32_t b_share = partyShare(b_full, b_mask, party);

//     de_local[index] = x_share ^ a_share;
//     de_local[total + index] = y_share ^ b_share;
// }

__global__ void neighborPrepareKernel(
    const uint32_t* __restrict__ global_s,
    const flora::preprocess::BooleanTripleRecord* __restrict__ triples,
    uint32_t* __restrict__ de_local,
    uint64_t active_num, uint64_t offset, int B, int party) {

    const uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    const uint64_t total = active_num * static_cast<uint64_t>(B);

    if (index >= total) { return; }

    const uint64_t tuple = index / static_cast<uint64_t>(B);
    const int field = static_cast<int>(index % static_cast<uint64_t>(B));

    const uint64_t left = tuple;
    const uint64_t right = tuple + offset;

    uint32_t x_share = 0;
    uint32_t y_share = 0;

    if (field == 0) {
        x_share = global_s[left * B];
        y_share = global_s[right * B];
    } else {
        const uint32_t right_flag = global_s[right * B];
        const uint32_t not_right_flag = right_flag ^ ((party == kAlice) ? 0xffffffffu : 0u);

        x_share = not_right_flag;
        y_share = global_s[left * B + field];
    }

    const auto triple = triples[index];

    de_local[index] = x_share ^ triple.a_share;
    de_local[total + index] = y_share ^ triple.b_share;
}

// __global__ void neighborFinishKernel(
//     const uint32_t* __restrict__ global_s,
//     uint32_t* __restrict__ global_s_temp,
//     const uint32_t* __restrict__ de_local,
//     const uint32_t* __restrict__ de_peer,
//     uint64_t active_num, uint64_t offset, int B, int party, uint64_t nonce) {

//     uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
//     uint64_t total = active_num * static_cast<uint64_t>(B);
//     if (index >= total) { return; }

//     uint32_t d = de_local[index] ^ de_peer[index];
//     uint32_t e = de_local[total + index] ^ de_peer[total + index];

//     uint64_t triple_index = nonce + index;

//     uint32_t a_full = gpuPrng(0xa5d70001ULL, triple_index, 1);
//     uint32_t b_full = gpuPrng(0xa5d70001ULL, triple_index, 2);
//     uint32_t c_full = a_full & b_full;
//     uint32_t a_mask = gpuPrng(0xa5d70001ULL, triple_index, 3);
//     uint32_t b_mask = gpuPrng(0xa5d70001ULL, triple_index, 4);
//     uint32_t c_mask = gpuPrng(0xa5d70001ULL, triple_index, 5);
//     uint32_t a_share = partyShare(a_full, a_mask, party);
//     uint32_t b_share = partyShare(b_full, b_mask, party);
//     uint32_t c_share = partyShare(c_full, c_mask, party);
//     uint32_t product_share = c_share ^ (d & b_share) ^ (e & a_share);

//     if (party == kAlice) {
//         product_share ^= d & e;
//     }

//     uint64_t tuple = index / static_cast<uint64_t>(B);
//     int field = static_cast<int>(index % static_cast<uint64_t>(B));
//     uint64_t right = tuple + offset;

//     if (field == 0) {
//         global_s_temp[right * B] = global_s[tuple * B] ^ global_s[right * B] ^ product_share;
//     } else {
//         global_s_temp[right * B + field] = global_s[right * B + field] ^ product_share;
//     }
// }

__global__ void neighborFinishKernel(
    const uint32_t* __restrict__ global_s, uint32_t* __restrict__ global_s_temp,
    const uint32_t* __restrict__ de_local, const uint32_t* __restrict__ de_peer,
    const flora::preprocess::BooleanTripleRecord* __restrict__ triples,
    uint64_t active_num, uint64_t offset, int B, int party) {

    const uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    const uint64_t total = active_num * static_cast<uint64_t>(B);

    if (index >= total) { return; }

    const uint32_t d = de_local[index] ^ de_peer[index];
    const uint32_t e = de_local[total + index] ^ de_peer[total + index];
    const auto triple = triples[index];

    uint32_t product_share = triple.c_share ^ (d & triple.b_share) ^ (e & triple.a_share);

    if (party == kAlice) {
        product_share ^= d & e;
    }

    const uint64_t tuple = index / static_cast<uint64_t>(B);
    const int field = static_cast<int>(index % static_cast<uint64_t>(B));
    const uint64_t right = tuple + offset;

    if (field == 0) {
        global_s_temp[right * B] = global_s[tuple * B] ^ global_s[right * B] ^ product_share;
    } else {
        global_s_temp[right * B + field] = global_s[right * B + field] ^ product_share;
    }
}



// ============================================================
// Filter kernels
// ============================================================

__global__ void filterExtractLastKernel(
    const uint32_t* __restrict__ path_shares,
    uint32_t* __restrict__ z_last,
    uint64_t path_num, int detect_length_k) {

    uint64_t path = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (path >= path_num) { return; }
    uint64_t width = static_cast<uint64_t>(detect_length_k + 1);
    z_last[path] = path_shares[path * width + static_cast<uint64_t>(detect_length_k)];
}

__global__ void zeroPhase1Kernel(const uint32_t* __restrict__ z_share,
    const flora::preprocess::GfZeroTestRecord* __restrict__ correlations,
    uint32_t* __restrict__ x_hat_local, uint64_t count) {

    const uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;

    if (index >= count) { return; }

    const auto correlation = correlations[index];

    x_hat_local[index] = z_share[index] ^ correlation.b_share;
}

__global__ void zeroPhase2Kernel(
    const uint32_t* __restrict__ x_hat_local, const uint32_t* __restrict__ x_hat_peer,
    const flora::preprocess::GfZeroTestRecord* __restrict__ correlations,
    uint32_t* __restrict__ w_local, uint64_t count) {

    const uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;

    if (index >= count) { return; }

    const uint32_t x_hat = x_hat_local[index] ^ x_hat_peer[index];
    const auto correlation = correlations[index];

    w_local[index] = correlation.c_share ^ gfMul(x_hat, correlation.r_share);
}


__global__ void zeroPhase3Kernel(
    const uint32_t* __restrict__ w_local,
    const uint32_t* __restrict__ w_peer,
    uint8_t* __restrict__ zero_bits, uint64_t count) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index >= count) { return; }
    uint32_t opened = w_local[index] ^ w_peer[index];
    zero_bits[index] = opened == 0 ? 1u : 0u;
}


__global__ void filterCompactKernel(
    const uint8_t* __restrict__ zero_bits,
    uint64_t* __restrict__ survivors,
    uint32_t* __restrict__ survivor_count,
    uint64_t path_num) {

    if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
    uint32_t output_position = 0;
    for (uint64_t path = 0; path < path_num; ++path) {
        if (zero_bits[path] == 0) {
            survivors[output_position] = path;
            ++output_position;
        }
    }

    *survivor_count = output_position;
}

__global__ void filterBuildBatchKernel(
    const uint32_t* __restrict__ path_shares,
    const uint64_t* __restrict__ survivors,
    uint32_t* __restrict__ z_batch,
    uint64_t survivor_count, int detect_length_k) {

    uint64_t survivor_position = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;

    if (survivor_position >= survivor_count) { return; }

    uint64_t path = survivors[survivor_position];
    uint64_t width = static_cast<uint64_t>(detect_length_k + 1);
    uint32_t first = path_shares[path * width];
    uint32_t last = path_shares[path * width + static_cast<uint64_t>(detect_length_k)];

    z_batch[survivor_position] = first;
    z_batch[survivor_count + survivor_position] = last ^ first;

    for (int position = 1; position < detect_length_k; ++position) {
        uint32_t value = path_shares[path * width + static_cast<uint64_t>(position)];
        uint64_t batch_index = static_cast<uint64_t>(position + 1) * survivor_count + survivor_position;
        z_batch[batch_index] = last ^ value;
    }
}

__global__ void filterFinalDecisionKernel(
    const uint8_t* __restrict__ second_zero_bits,
    const uint64_t* __restrict__ survivors,
    uint8_t* __restrict__ is_valid,
    uint8_t* __restrict__ is_cycle,
    uint64_t survivor_count, int detect_length_k) {

    uint64_t position = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (position >= survivor_count) { return; }
    uint64_t original_path = survivors[position];

    bool first_zero = second_zero_bits[position] != 0;
    bool cycle = second_zero_bits[survivor_count + position] != 0;
    bool repeat = false;

    for (int path_position = 1; path_position < detect_length_k; ++path_position) {
        uint64_t batch_index = static_cast<uint64_t>(path_position + 1) * survivor_count + position;
        repeat = repeat || second_zero_bits[batch_index] != 0;
    }

    bool valid = !first_zero && !repeat;
    is_valid[original_path] = valid ? 1u : 0u;
    is_cycle[original_path] = valid && cycle ? 1u : 0u;
}

} // namespace

struct GpuOnlineContext {
    cudaStream_t compute_stream = nullptr;
    cudaStream_t transfer_stream = nullptr;

    // Triples
    DeviceBuffer<flora::preprocess::BooleanTripleRecord>  neighbor_triples;
    uint64_t neighbor_triple_count = 0;
    DeviceBuffer<flora::preprocess::GfZeroTestRecord> zero_test_correlations;
    uint64_t zero_test_correlation_count = 0;

    // Neighbor
    DeviceBuffer<uint32_t> neighbor_global_s;
    DeviceBuffer<uint32_t> neighbor_global_s_temp;
    uint64_t neighbor_element_num = 0;
    int neighbor_B = 0;

    // opening
    DeviceBuffer<uint32_t> de_local;
    DeviceBuffer<uint32_t> de_peer;
    DeviceBuffer<uint32_t> w_local;
    DeviceBuffer<uint32_t> w_peer;

    HostPinnedBuffer host_send;
    HostPinnedBuffer host_receive;

    // Filter
    DeviceBuffer<uint32_t> path_shares;
    DeviceBuffer<uint32_t> z_last;
    DeviceBuffer<uint32_t> z_batch;

    DeviceBuffer<uint8_t> zero_bits;
    DeviceBuffer<uint64_t> survivors;
    DeviceBuffer<uint32_t> survivor_count;

    DeviceBuffer<uint8_t> is_valid;
    DeviceBuffer<uint8_t> is_cycle;

    uint64_t filter_path_num = 0;
    int filter_detect_length_k = 0;

    GpuOnlineContext() {
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &compute_stream,
            cudaStreamNonBlocking));

        CUDA_CHECK(cudaStreamCreateWithFlags(
            &transfer_stream,
            cudaStreamNonBlocking));
    }

    ~GpuOnlineContext() {
        if (compute_stream != nullptr) {
            cudaStreamDestroy(compute_stream);
        }

        if (transfer_stream != nullptr) {
            cudaStreamDestroy(transfer_stream);
        }
    }
};

extern "C" GpuOnlineContext* gpu_online_create() {
    return new GpuOnlineContext();
}

extern "C" void gpu_online_destroy(
    GpuOnlineContext* context) {

    delete context;
}

// ============================================================
// NeighborPass API
// ============================================================
extern "C" void gpu_neighbor_upload(GpuOnlineContext* context, const uint32_t* host_global_s, uint64_t element_num, int B) {
    if (context == nullptr || host_global_s == nullptr) {
        std::abort();
    }
    uint64_t word_count = element_num * static_cast<uint64_t>(B);

    context->neighbor_global_s.reserve(word_count);
    context->neighbor_global_s_temp.reserve(word_count);
    context->neighbor_element_num = element_num;
    context->neighbor_B = B;

    CUDA_CHECK(cudaMemcpyAsync(context->neighbor_global_s.pointer, host_global_s, word_count * sizeof(uint32_t), cudaMemcpyHostToDevice, context->transfer_stream));
    CUDA_CHECK(cudaStreamSynchronize(context->transfer_stream));
}

extern "C" void gpu_neighbor_upload_triples(GpuOnlineContext* context, const flora::preprocess::BooleanTripleRecord* host_triples, uint64_t count) {

    if (context == nullptr) {
        std::abort();
    }

    if (count > 0 && host_triples == nullptr) {
        std::abort();
    }

    context->neighbor_triples.reserve(count);
    context->neighbor_triple_count = count;

    if (count == 0) {
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(context->neighbor_triples.pointer, host_triples,
        count * sizeof(flora::preprocess::BooleanTripleRecord),
        cudaMemcpyHostToDevice, context->compute_stream));
}


// extern "C" void gpu_neighbor_prepare(GpuOnlineContext* context, uint64_t offset,
//     int party, uint64_t nonce, const void** host_send, void** host_receive, size_t* opening_bytes) {

//     uint64_t element_num = context->neighbor_element_num;
//     if (offset >= element_num) {
//         std::abort();
//     }
//     uint64_t active_num = element_num - offset;
//     uint64_t total = active_num * static_cast<uint64_t>(context->neighbor_B);
//     size_t bytes = 2ULL * total * sizeof(uint32_t);

//     context->de_local.reserve(2ULL * total);
//     context->de_peer.reserve(2ULL * total);
//     context->host_send.reserve(bytes);
//     context->host_receive.reserve(bytes);

//     neighborPrepareKernel<<<gridFor(total), kThreads, 0, context->compute_stream>>>(
//             context->neighbor_global_s.pointer, context->de_local.pointer,
//             active_num, offset, context->neighbor_B, party, nonce);

//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaMemcpyAsync(context->host_send.pointer, context->de_local.pointer, bytes, cudaMemcpyDeviceToHost, context->compute_stream));
//     CUDA_CHECK(cudaStreamSynchronize(context->compute_stream));

//     *host_send = context->host_send.pointer;
//     *host_receive = context->host_receive.pointer;
//     *opening_bytes = bytes;
// }

extern "C" void gpu_neighbor_prepare(GpuOnlineContext* context, uint64_t offset, int party, uint64_t nonce,
    const void** host_send, void** host_receive, size_t* opening_bytes) {

    (void)nonce;

    if (context == nullptr || host_send == nullptr || host_receive == nullptr || opening_bytes == nullptr) {
        std::abort();
    }

    const uint64_t element_num = context->neighbor_element_num;

    if (offset >= element_num) {
        std::abort();
    }

    const uint64_t active_num = element_num - offset;
    const uint64_t total = active_num * static_cast<uint64_t>(context->neighbor_B);

    if (context->neighbor_triple_count != total) {
        std::fprintf(
            stderr,
            "gpu_neighbor_prepare: expected %llu triples, "
            "got %llu\n",
            static_cast<unsigned long long>(total),
            static_cast<unsigned long long>(context->neighbor_triple_count));

        std::abort();
    }

    const size_t bytes = 2ULL * total * sizeof(uint32_t);

    context->de_local.reserve(2ULL * total);
    context->de_peer.reserve(2ULL * total);

    context->host_send.reserve(bytes);
    context->host_receive.reserve(bytes);

    neighborPrepareKernel<<<gridFor(total), kThreads, 0, context->compute_stream>>>(
            context->neighbor_global_s.pointer, context->neighbor_triples.pointer, context->de_local.pointer,
            active_num, offset, context->neighbor_B, party);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(context->host_send.pointer, context->de_local.pointer,
        bytes, cudaMemcpyDeviceToHost, context->compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(context->compute_stream));

    *host_send = context->host_send.pointer;
    *host_receive = context->host_receive.pointer;
    *opening_bytes = bytes;
}

// extern "C" void gpu_neighbor_finish(GpuOnlineContext* context, uint64_t offset, int party, uint64_t nonce) {

//     uint64_t element_num = context->neighbor_element_num;
//     uint64_t active_num = element_num - offset;
//     uint64_t total = active_num * static_cast<uint64_t>(context->neighbor_B);
//     size_t bytes = 2ULL * total * sizeof(uint32_t);

//     CUDA_CHECK(cudaMemcpyAsync(context->de_peer.pointer, context->host_receive.pointer, bytes, cudaMemcpyHostToDevice, context->compute_stream));

//     uint64_t prefix_words = offset * static_cast<uint64_t>(context->neighbor_B);

//     if (prefix_words > 0) {
//         CUDA_CHECK(cudaMemcpyAsync(context->neighbor_global_s_temp.pointer, context->neighbor_global_s.pointer,
//             prefix_words * sizeof(uint32_t), cudaMemcpyDeviceToDevice, context->compute_stream));
//     }

//     neighborFinishKernel<<<gridFor(total), kThreads, 0, context->compute_stream>>>(
//             context->neighbor_global_s.pointer, context->neighbor_global_s_temp.pointer,
//             context->de_local.pointer, context->de_peer.pointer,
//             active_num, offset, context->neighbor_B, party, nonce);

//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaStreamSynchronize(context->compute_stream));
//     context->neighbor_global_s.swap(context->neighbor_global_s_temp);
// }

extern "C" void gpu_neighbor_finish(GpuOnlineContext* context, uint64_t offset, int party, uint64_t nonce) {

    uint64_t element_num = context->neighbor_element_num;
    uint64_t active_num = element_num - offset;
    uint64_t total = active_num * static_cast<uint64_t>(context->neighbor_B);
    size_t bytes = 2ULL * total * sizeof(uint32_t);

    CUDA_CHECK(cudaMemcpyAsync(context->de_peer.pointer, context->host_receive.pointer, bytes, cudaMemcpyHostToDevice, context->compute_stream));

    uint64_t prefix_words = offset * static_cast<uint64_t>(context->neighbor_B);

    if (prefix_words > 0) {
        CUDA_CHECK(cudaMemcpyAsync(context->neighbor_global_s_temp.pointer, context->neighbor_global_s.pointer,
            prefix_words * sizeof(uint32_t), cudaMemcpyDeviceToDevice, context->compute_stream));
    }

    neighborFinishKernel<<<gridFor(total), kThreads, 0, context->compute_stream>>>(
        context->neighbor_global_s.pointer, context->neighbor_global_s_temp.pointer,
        context->de_local.pointer, context->de_peer.pointer,
        context->neighbor_triples.pointer, active_num, offset, context->neighbor_B, party);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(context->compute_stream));
    context->neighbor_global_s.swap(context->neighbor_global_s_temp);
    context->neighbor_triple_count = 0;
}

extern "C" void gpu_neighbor_download(GpuOnlineContext* context, uint32_t* host_global_s) {

    uint64_t word_count = context->neighbor_element_num * static_cast<uint64_t>(context->neighbor_B);
    CUDA_CHECK(cudaMemcpyAsync(host_global_s, context->neighbor_global_s.pointer,
        word_count * sizeof(uint32_t), cudaMemcpyDeviceToHost, context->transfer_stream));
    CUDA_CHECK(cudaStreamSynchronize(context->transfer_stream));
}

// ============================================================
// Filter API
// ============================================================
extern "C" void gpu_filter_upload_paths(GpuOnlineContext* context, const uint32_t* host_path_shares, uint64_t path_num, int detect_length_k) {

    uint64_t word_count = path_num * static_cast<uint64_t>(detect_length_k + 1);

    context->path_shares.reserve(word_count);
    context->z_last.reserve(path_num);
    context->filter_path_num = path_num;
    context->filter_detect_length_k = detect_length_k;

    if (word_count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(context->path_shares.pointer, host_path_shares,
            word_count * sizeof(uint32_t), cudaMemcpyHostToDevice, context->transfer_stream));
        CUDA_CHECK(cudaStreamSynchronize(context->transfer_stream));
    }
}

extern "C" void gpu_zero_test_upload_correlations(GpuOnlineContext* context, const flora::preprocess::GfZeroTestRecord* host_records, uint64_t count) {

    if (context == nullptr) {
        std::abort();
    }
    if (count > 0 && host_records == nullptr) {
        std::abort();
    }

    context->zero_test_correlations.reserve(count);
    context->zero_test_correlation_count = count;

    if (count == 0) { return; }

    CUDA_CHECK(cudaMemcpyAsync(
        context->zero_test_correlations.pointer, host_records,
        count * sizeof(flora::preprocess::GfZeroTestRecord),
        cudaMemcpyHostToDevice, context->compute_stream));
}

extern "C" void gpu_filter_extract_last(
    GpuOnlineContext* context) {

    uint64_t path_num = context->filter_path_num;
    if (path_num == 0) { return; }
    filterExtractLastKernel<<<gridFor(path_num), kThreads, 0, context->compute_stream>>>(
            context->path_shares.pointer, context->z_last.pointer, path_num, context->filter_detect_length_k);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void gpu_zero_test_phase1(
    GpuOnlineContext* context, int source, uint64_t element_count, int party, uint64_t nonce,
    const void** host_send, void** host_receive, size_t* opening_bytes) {

    (void)party;
    (void)nonce;
    if (context == nullptr || host_send == nullptr || host_receive == nullptr || opening_bytes == nullptr) {
        std::abort();
    }

    const uint32_t* z_source = nullptr;

    if (source == 0) {
        z_source = context->z_last.pointer;
    } else if (source == 1) {
        z_source = context->z_batch.pointer;
    } else {
        std::fprintf(stderr, "Bad zero-test source\n");
        std::abort();
    }

    if (context->zero_test_correlation_count != element_count) {
        std::fprintf(stderr,
            "gpu_zero_test_phase1: expected %llu "
            "correlations, got %llu\n",
            static_cast<unsigned long long>(element_count),
            static_cast<unsigned long long>(context->zero_test_correlation_count));
        std::abort();
    }

    context->de_local.reserve(element_count);
    context->de_peer.reserve(element_count);

    const size_t bytes = element_count * sizeof(uint32_t);
    context->host_send.reserve(bytes);
    context->host_receive.reserve(bytes);

    if (element_count > 0) {
        zeroPhase1Kernel<<<gridFor(element_count), kThreads, 0, context->compute_stream>>>(
                z_source, context->zero_test_correlations.pointer, context->de_local.pointer, element_count);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(context->host_send.pointer, context->de_local.pointer, bytes,
            cudaMemcpyDeviceToHost, context->compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(context->compute_stream));
    }

    *host_send = context->host_send.pointer;
    *host_receive = context->host_receive.pointer;
    *opening_bytes = bytes;
}

extern "C" void gpu_zero_test_phase2(GpuOnlineContext* context, uint64_t element_count, int party, uint64_t nonce,
    const void** host_send, void** host_receive, size_t* opening_bytes) {

    (void)party;
    (void)nonce;

    if (context == nullptr || host_send == nullptr || host_receive == nullptr || opening_bytes == nullptr) {
        std::abort();
    }

    if (context->zero_test_correlation_count != element_count) {
        std::fprintf(stderr, "gpu_zero_test_phase2: bad correlation count\n");
        std::abort();
    }

    const size_t x_hat_bytes = element_count * sizeof(uint32_t);

    context->w_local.reserve(element_count);
    context->w_peer.reserve(element_count);

    CUDA_CHECK(cudaMemcpyAsync( context->de_peer.pointer, context->host_receive.pointer, x_hat_bytes,
        cudaMemcpyHostToDevice, context->compute_stream));

    zeroPhase2Kernel<<<gridFor(element_count), kThreads, 0, context->compute_stream>>>(
            context->de_local.pointer, context->de_peer.pointer,
            context->zero_test_correlations.pointer, context->w_local.pointer, element_count);

    CUDA_CHECK(cudaGetLastError());
    const size_t w_bytes = element_count * sizeof(uint32_t);

    context->host_send.reserve(w_bytes);
    context->host_receive.reserve(w_bytes);

    CUDA_CHECK(cudaMemcpyAsync(context->host_send.pointer, context->w_local.pointer, w_bytes,
        cudaMemcpyDeviceToHost, context->compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(context->compute_stream));

    *host_send = context->host_send.pointer;
    *host_receive = context->host_receive.pointer;
    *opening_bytes = w_bytes;
}

extern "C" void gpu_zero_test_phase3(GpuOnlineContext* context, uint64_t element_count) {

    size_t bytes = element_count * sizeof(uint32_t);
    context->zero_bits.reserve(element_count);
    CUDA_CHECK(cudaMemcpyAsync(context->w_peer.pointer, context->host_receive.pointer, bytes, cudaMemcpyHostToDevice, context->compute_stream));

    zeroPhase3Kernel<<<gridFor(element_count), kThreads, 0, context->compute_stream>>>(
            context->w_local.pointer, context->w_peer.pointer, context->zero_bits.pointer, element_count);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(context->compute_stream));
    context->zero_test_correlation_count = 0;
}

extern "C" uint64_t gpu_online_filter_compact(GpuOnlineContext* context, uint64_t path_num) {

    if (context == nullptr) {
        std::fprintf(stderr, "gpu_online_filter_compact: null context\n");
        std::abort();
    }
    if (path_num > static_cast<uint64_t>(0xffffffffu)) {
        std::fprintf(stderr,
            "gpu_online_filter_compact: "
            "path_num exceeds uint32_t range: %llu\n",
            static_cast<unsigned long long>(path_num));
        std::abort();
    }
    context->survivors.reserve(path_num);
    context->survivor_count.reserve(1);

    filterCompactKernel<<<1, 1, 0, context->compute_stream>>>(
            context->zero_bits.pointer, context->survivors.pointer, context->survivor_count.pointer, path_num);

    CUDA_CHECK(cudaGetLastError());
    uint32_t host_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&host_count, context->survivor_count.pointer, sizeof(uint32_t),
        cudaMemcpyDeviceToHost, context->compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(context->compute_stream));

    if (static_cast<uint64_t>(host_count) > path_num) {
        std::fprintf(stderr,
            "gpu_online_filter_compact: "
            "invalid survivor count %u, path_num=%llu\n",
            host_count, static_cast<unsigned long long>(path_num));
        std::abort();
    }
    return static_cast<uint64_t>(host_count);
}

extern "C" void gpu_online_filter_build_zbatch(GpuOnlineContext* context, uint64_t survivor_count, int detect_length_k) {
    uint64_t total = survivor_count * static_cast<uint64_t>(detect_length_k + 1);
    context->z_batch.reserve(total);
    if (survivor_count == 0) { return; }

    filterBuildBatchKernel<<<gridFor(survivor_count), kThreads, 0, context->compute_stream>>>(
            context->path_shares.pointer, context->survivors.pointer, context->z_batch.pointer,
            survivor_count, detect_length_k);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void gpu_online_filter_final_decision(GpuOnlineContext* context, uint64_t survivor_count, int detect_length_k, uint64_t path_num) {
    context->is_valid.reserve(path_num);
    context->is_cycle.reserve(path_num);
    CUDA_CHECK(cudaMemsetAsync(context->is_valid.pointer, 0, path_num * sizeof(uint8_t), context->compute_stream));
    CUDA_CHECK(cudaMemsetAsync(context->is_cycle.pointer, 0, path_num * sizeof(uint8_t), context->compute_stream));

    if (survivor_count > 0) {
        filterFinalDecisionKernel<<<gridFor(survivor_count), kThreads, 0, context->compute_stream>>>(
                context->zero_bits.pointer, context->survivors.pointer,
                context->is_valid.pointer, context->is_cycle.pointer,
                survivor_count, detect_length_k);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(context->compute_stream));
}

extern "C" void gpu_online_filter_download_decision(GpuOnlineContext* context, uint8_t* host_is_valid, uint8_t* host_is_cycle, uint64_t path_num) {
    CUDA_CHECK(cudaMemcpyAsync(host_is_valid, context->is_valid.pointer, path_num * sizeof(uint8_t),
        cudaMemcpyDeviceToHost, context->transfer_stream));
    CUDA_CHECK(cudaMemcpyAsync(host_is_cycle, context->is_cycle.pointer, path_num * sizeof(uint8_t),
        cudaMemcpyDeviceToHost, context->transfer_stream));
    CUDA_CHECK(cudaStreamSynchronize(context->transfer_stream));
}