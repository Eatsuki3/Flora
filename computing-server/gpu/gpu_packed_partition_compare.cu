#include "gpu_packed_partition_compare.cuh"
#include "gpu_runtime.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kThreads = 256;
constexpr int kAlice = 1;

int blocksFor(uint64_t count) {
    if (count == 0) {
        return 0;
    }
    const uint64_t blocks = (count + static_cast<uint64_t>(kThreads) - 1ULL) / static_cast<uint64_t>(kThreads);
    if (blocks > static_cast<uint64_t>(0x7fffffff)) {
        std::fprintf(stderr, "gpu_packed_compare: CUDA grid too large\n");
        std::abort();
    }
    return static_cast<int>(blocks);
}

__device__ __forceinline__ uint32_t gpuPrng(uint64_t nonce, uint64_t index, uint32_t domain) {

    uint64_t value = nonce ^ (index * 0x9e3779b97f4a7c15ULL) ^ (static_cast<uint64_t>(domain) * 0xd6e8feb86659fd93ULL);
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;

    return static_cast<uint32_t>(value);
}

__device__ __forceinline__ uint32_t partyShare(uint32_t plaintext, uint32_t mask, int party) {
    return party == kAlice ? mask : plaintext ^ mask;
}

__device__ __forceinline__ uint32_t activeLaneMask(uint64_t word, uint64_t comparison_count) {

    const uint64_t begin = word * 32ULL;
    if (begin >= comparison_count) {
        return 0u;
    }
    const uint64_t remaining = comparison_count - begin;
    if (remaining >= 32ULL) {
        return 0xffffffffu;
    }
    return static_cast<uint32_t>((1ULL << remaining) - 1ULL);
}

// Key bit order: MSB -> LSB
// node_val[node_bits], end_node[node_bits], tie_breaker[tie_bits]
__device__ __forceinline__ uint32_t readLocalKeyBit(
    uint32_t node_share, uint32_t end_share, uint64_t tie_share,
    int node_bits, int tie_bits, int key_bit) {

    if (key_bit < node_bits) {
        const int source_bit = node_bits - 1 - key_bit;
        return (node_share >> source_bit) & 1u;
    }
    if (key_bit < 2 * node_bits) {
        const int position = key_bit - node_bits;
        const int source_bit = node_bits - 1 - position;
        return (end_share >> source_bit) & 1u;
    }

    const int position = key_bit - 2 * node_bits;
    const int source_bit = tie_bits - 1 - position;
    return static_cast<uint32_t>((tie_share >> source_bit) & 1ULL);
}

// Each CUDA thread constructs: 1 key bit × a group of up to 32 comparator,
// Each bit lane of packed_x[bit * packed_words + word] corresponds to one comparator.
__global__ void packKeyBitsKernel(
    const uint32_t* __restrict__ node_share, const uint32_t* __restrict__ end_share, const uint64_t* __restrict__ tie_share,
    const uint32_t* __restrict__ pivot_node_share, const uint32_t* __restrict__ pivot_end_share, const uint64_t* __restrict__ pivot_tie_share,
    uint32_t* __restrict__ packed_x, uint32_t* __restrict__ packed_y, uint32_t* __restrict__ packed_equal,
    uint64_t comparison_count, uint64_t packed_words,
    int node_bits, int tie_bits, int key_bits, int party) {

    const uint64_t packed_index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    const uint64_t total = packed_words * static_cast<uint64_t>(key_bits);

    if (packed_index >= total) { return; }

    const int key_bit = static_cast<int>(packed_index / packed_words);
    const uint64_t word = packed_index % packed_words;
    const uint64_t first_comparison = word * 32ULL;

    uint32_t x_word = 0u;
    uint32_t y_word = 0u;

    for (int lane = 0; lane < 32; ++lane) {
        const uint64_t comparison = first_comparison + static_cast<uint64_t>(lane);
        if (comparison >= comparison_count) { break; }

        const uint32_t x_bit = readLocalKeyBit(
                node_share[comparison], end_share[comparison], tie_share[comparison],
                node_bits, tie_bits, key_bit);
        const uint32_t y_bit = readLocalKeyBit(
                pivot_node_share[comparison], pivot_end_share[comparison], pivot_tie_share[comparison],
                node_bits, tie_bits, key_bit);

        x_word |= x_bit << lane;
        y_word |= y_bit << lane;
    }
    packed_x[packed_index] = x_word;
    packed_y[packed_index] = y_word;

    // E = NOT(x XOR y)
    const uint32_t public_one_share = (party == kAlice) ? activeLaneMask(word, comparison_count) : 0u;
    packed_equal[packed_index] = x_word ^ y_word ^ public_one_share;
}

// G[i] = x[i] AND NOT(y[i]) (represents if x > y on current key bit)
__global__ void buildInitialGtInputsKernel(
    const uint32_t* __restrict__ packed_x, const uint32_t* __restrict__ packed_y,
    uint32_t* __restrict__ and_x, uint32_t* __restrict__ and_y,
    uint64_t packed_words, uint64_t comparison_count, uint64_t count, int party) {

    const uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index >= count) { return; }
    const uint64_t word = index % packed_words;
    const uint32_t public_one_share = (party == kAlice) ? activeLaneMask(word, comparison_count) : 0u;

    and_x[index] = packed_x[index];
    and_y[index] = packed_y[index] ^ public_one_share;
}

// Prefix combine: For higher segment H and lower segmnt L, G = G_H OR (E_H AND G_L), E = E_H AND E_L
// G_H and E_H AND G_L are mutually exclusive in plaintext, so OR can be directly replaced with XOR.
// Each prefix item requires two AND operations: product[2*i] = E_H AND G_L, product[2*i + 1] = E_H AND E_L
__global__ void buildPrefixAndInputsKernel(
    const uint32_t* __restrict__ current_g, const uint32_t* __restrict__ current_e,
    uint32_t* __restrict__ and_x, uint32_t* __restrict__ and_y,
    uint64_t packed_words, int key_bits, int prefix_offset) {

    const uint64_t item = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    const uint64_t item_count = packed_words * static_cast<uint64_t>(key_bits);
    if (item >= item_count) { return; }
    const int key_bit = static_cast<int>(item / packed_words);
    const uint64_t word = item % packed_words;
    const uint64_t product_base = item * 2ULL;

    if (key_bit < prefix_offset) {
        and_x[product_base] = 0u;
        and_y[product_base] = 0u;
        and_x[product_base + 1ULL] = 0u;
        and_y[product_base + 1ULL] = 0u;
        return;
    }

    const uint64_t high_index = static_cast<uint64_t>(key_bit - prefix_offset) * packed_words + word;
    const uint64_t low_index = item;
    const uint32_t e_high = current_e[high_index];
    and_x[product_base] = e_high;
    and_y[product_base] = current_g[low_index];
    and_x[product_base + 1ULL] = e_high;
    and_y[product_base + 1ULL] = current_e[low_index];
}

__global__ void mergePrefixKernel(
    const uint32_t* __restrict__ current_g, const uint32_t* __restrict__ current_e, const uint32_t* __restrict__ products,
    uint32_t* __restrict__ next_g, uint32_t* __restrict__ next_e,
    uint64_t packed_words, int key_bits, int prefix_offset) {

    const uint64_t item = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    const uint64_t item_count = packed_words * static_cast<uint64_t>(key_bits);
    if (item >= item_count) { return; }

    const int key_bit = static_cast<int>(item / packed_words);

    if (key_bit < prefix_offset) {
        next_g[item] = current_g[item];
        next_e[item] = current_e[item];
        return;
    }

    const uint64_t high_index = static_cast<uint64_t>(key_bit - prefix_offset) * packed_words + item % packed_words;
    const uint64_t product_base = item * 2ULL;

    // G = G_H XOR (E_H AND G_L), E = E_H AND E_L
    next_g[item] = current_g[high_index] ^ products[product_base];
    next_e[item] = products[product_base + 1ULL];
}

__global__ void andPrepareKernel(const uint32_t* __restrict__ x, const uint32_t* __restrict__ y,
    const flora::preprocess::BooleanTripleRecord* __restrict__ triples,
    uint32_t* __restrict__ de_local, uint64_t count) {

    const uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;

    if (index >= count) { return; }

    const auto triple = triples[index];
    de_local[index] = x[index] ^ triple.a_share;
    de_local[count + index] = y[index] ^ triple.b_share;
}

__global__ void andFinishKernel(const uint32_t* __restrict__ de_local, const uint32_t* __restrict__ de_peer,
    const flora::preprocess::BooleanTripleRecord* __restrict__ triples,
    uint32_t* __restrict__ output, uint64_t count, int party) {

    const uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;

    if (index >= count) { return; }

    const uint32_t d = de_local[index] ^ de_peer[index];
    const uint32_t e = de_local[count + index] ^ de_peer[count + index];
    const auto triple = triples[index];
    uint32_t result = triple.c_share ^ (d & triple.b_share) ^ (e & triple.a_share);

    if (party == kAlice) {
        result ^= d & e;
    }
    output[index] = result;
}

} // namespace

struct GpuPackedCompareContext {
    enum class Phase {
        InitialGt,
        Prefix,
        Done
    };

    int party = 0;
    int node_bits = 0;
    int tie_bits = 0;
    int key_bits = 0;

    uint64_t comparison_count = 0;
    uint64_t packed_words = 0;
    uint64_t packed_item_count = 0;

    uint64_t batch_nonce = 0;
    uint64_t round_index = 0;

    int prefix_offset = 1;

    Phase phase = Phase::Done;

    GpuRuntime runtime;

    DeviceBuffer<uint32_t> node_share;
    DeviceBuffer<uint32_t> end_share;
    DeviceBuffer<uint64_t> tie_share;

    DeviceBuffer<uint32_t> pivot_node_share;
    DeviceBuffer<uint32_t> pivot_end_share;
    DeviceBuffer<uint64_t> pivot_tie_share;

    DeviceBuffer<uint32_t> packed_x;
    DeviceBuffer<uint32_t> packed_y;

    DeviceBuffer<uint32_t> current_g;
    DeviceBuffer<uint32_t> current_e;

    DeviceBuffer<uint32_t> next_g;
    DeviceBuffer<uint32_t> next_e;

    DeviceBuffer<uint32_t> and_x;
    DeviceBuffer<uint32_t> and_y;
    DeviceBuffer<uint32_t> and_output;

    DeviceBuffer<uint32_t> de_local;
    DeviceBuffer<uint32_t> de_peer;

    PinnedBuffer<uint32_t> host_send;
    PinnedBuffer<uint32_t> host_receive;

    DeviceBuffer<flora::preprocess::BooleanTripleRecord> and_triples;
    uint64_t uploaded_triple_count = 0;
    uint64_t current_and_count = 0;
    // uint64_t current_and_nonce = 0;
};

extern "C"
GpuPackedCompareContext* gpu_packed_compare_create(int party, int node_bits, int tie_bits) {

    if (party != 1 && party != 2) {
        std::fprintf(stderr, "gpu_packed_compare_create: party must be 1 or 2\n");
        std::abort();
    }
    if (node_bits <= 0 || node_bits > 32) {
        std::fprintf(stderr, "gpu_packed_compare_create: node_bits must be in [1,32]\n");
        std::abort();
    }
    if (tie_bits <= 0 || tie_bits > 64) {
        std::fprintf(stderr, "gpu_packed_compare_create: tie_bits must be in [1,64]\n");
        std::abort();
    }

    auto* context = new GpuPackedCompareContext();
    context->party = party;
    context->node_bits = node_bits;
    context->tie_bits = tie_bits;
    context->key_bits = 2 * node_bits + tie_bits;

    return context;
}

extern "C"
void gpu_packed_compare_destroy(GpuPackedCompareContext* context) {
    delete context;
}

extern "C"
uint64_t gpu_packed_compare_next_triple_count(const GpuPackedCompareContext* context) {

    if (context == nullptr || context->phase == GpuPackedCompareContext::Phase::Done) {
        return 0;
    }

    if (context->phase == GpuPackedCompareContext::Phase::InitialGt) {
        return context->packed_item_count;
    }
    return 2ULL * context->packed_item_count;
}

extern "C"
void gpu_packed_compare_upload_triples(GpuPackedCompareContext* context,
    const flora::preprocess::BooleanTripleRecord* host_triples, uint64_t count) {

    if (context == nullptr) {
        std::abort();
    }

    const uint64_t expected = gpu_packed_compare_next_triple_count(context);
    if (count != expected) {
        std::fprintf(stderr,
            "gpu_packed_compare_upload_triples: "
            "expected %llu, got %llu\n",
            static_cast<unsigned long long>(expected),
            static_cast<unsigned long long>(count));

        std::abort();
    }

    if (count > 0 && host_triples == nullptr) {
        std::abort();
    }
    context->and_triples.reserve(count);
    context->uploaded_triple_count = count;

    if (count == 0) { return; }

    cudaStream_t stream = context->runtime.computeStream();
    FLORA_CUDA_CHECK(cudaMemcpyAsync(context->and_triples.data(), host_triples,
        count * sizeof(flora::preprocess::BooleanTripleRecord),
        cudaMemcpyHostToDevice, stream));
}

extern "C"
void gpu_packed_compare_begin(
    GpuPackedCompareContext* context,
    const uint32_t* node_share, const uint32_t* end_share, const uint64_t* tie_share,
    const uint32_t* pivot_node_share, const uint32_t* pivot_end_share, const uint64_t* pivot_tie_share,
    uint64_t comparison_count, uint64_t batch_nonce) {

    if (context == nullptr) {
        std::abort();
    }

    context->comparison_count = comparison_count;
    context->packed_words = (comparison_count + 31ULL) / 32ULL;
    context->packed_item_count = context->packed_words * static_cast<uint64_t>(context->key_bits);
    context->batch_nonce = batch_nonce;
    context->round_index = 0;
    context->prefix_offset = 1;

    if (comparison_count == 0) {
        context->phase = GpuPackedCompareContext::Phase::Done;
        return;
    }

    context->phase = GpuPackedCompareContext::Phase::InitialGt;
    context->node_share.reserve(comparison_count);
    context->end_share.reserve(comparison_count);
    context->tie_share.reserve(comparison_count);
    context->pivot_node_share.reserve(comparison_count);
    context->pivot_end_share.reserve(comparison_count);
    context->pivot_tie_share.reserve(comparison_count);
    context->packed_x.reserve(context->packed_item_count);
    context->packed_y.reserve(context->packed_item_count);
    context->current_g.reserve(context->packed_item_count);
    context->current_e.reserve(context->packed_item_count);
    context->next_g.reserve(context->packed_item_count);
    context->next_e.reserve(context->packed_item_count);

    cudaStream_t stream = context->runtime.computeStream();

    FLORA_CUDA_CHECK(cudaMemcpyAsync(context->node_share.data(), node_share, 
        comparison_count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    FLORA_CUDA_CHECK(cudaMemcpyAsync(context->end_share.data(), end_share,
        comparison_count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    FLORA_CUDA_CHECK(cudaMemcpyAsync(context->tie_share.data(), tie_share,
        comparison_count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    FLORA_CUDA_CHECK(cudaMemcpyAsync(context->pivot_node_share.data(), pivot_node_share,
        comparison_count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    FLORA_CUDA_CHECK(cudaMemcpyAsync(context->pivot_end_share.data(), pivot_end_share,
        comparison_count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    FLORA_CUDA_CHECK(cudaMemcpyAsync(context->pivot_tie_share.data(), pivot_tie_share,
        comparison_count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

    packKeyBitsKernel<<<blocksFor(context->packed_item_count), kThreads, 0, stream>>>(
            context->node_share.data(),
            context->end_share.data(),
            context->tie_share.data(),
            context->pivot_node_share.data(),
            context->pivot_end_share.data(),
            context->pivot_tie_share.data(),
            context->packed_x.data(),
            context->packed_y.data(),
            context->current_e.data(),
            context->comparison_count,
            context->packed_words,
            context->node_bits,
            context->tie_bits,
            context->key_bits,
            context->party);

    FLORA_CUDA_CHECK(cudaGetLastError());
    FLORA_CUDA_CHECK(cudaStreamSynchronize(stream));
}

extern "C"
int gpu_packed_compare_has_round(const GpuPackedCompareContext* context) {
    return context != nullptr && context->phase != GpuPackedCompareContext::Phase::Done;
}

extern "C"
void gpu_packed_compare_prepare_round(GpuPackedCompareContext* context, const void** host_send,
    void** host_receive, size_t* opening_bytes) {

    if (context == nullptr || context->phase == GpuPackedCompareContext::Phase::Done) {
        std::abort();
    }

    cudaStream_t stream = context->runtime.computeStream();
    if (context->phase == GpuPackedCompareContext::Phase::InitialGt) {
        context->current_and_count = context->packed_item_count;
        context->and_x.reserve(context->current_and_count);
        context->and_y.reserve(context->current_and_count);
        context->and_output.reserve(context->current_and_count);

        buildInitialGtInputsKernel<<<blocksFor(context->current_and_count), kThreads, 0, stream>>>(
                context->packed_x.data(),
                context->packed_y.data(),
                context->and_x.data(),
                context->and_y.data(),
                context->packed_words,
                context->comparison_count,
                context->current_and_count,
                context->party);
    } else {
        // Each prefix item requires two AND operations.
        context->current_and_count = 2ULL * context->packed_item_count;
        context->and_x.reserve(context->current_and_count);
        context->and_y.reserve(context->current_and_count);
        context->and_output.reserve(context->current_and_count);

        buildPrefixAndInputsKernel<<<blocksFor(context->packed_item_count), kThreads, 0, stream>>>(
                context->current_g.data(),
                context->current_e.data(),
                context->and_x.data(),
                context->and_y.data(),
                context->packed_words,
                context->key_bits,
                context->prefix_offset);
    }

    if (context->uploaded_triple_count != context->current_and_count) {
        std::fprintf(stderr,
            "gpu_packed_compare_prepare_round: "
            "missing triples, expected=%llu uploaded=%llu\n",
            static_cast<unsigned long long>(context->current_and_count),
            static_cast<unsigned long long>(context->uploaded_triple_count));
        std::abort();
    }

    ++context->round_index;
    context->de_local.reserve(2ULL * context->current_and_count);
    context->de_peer.reserve(2ULL * context->current_and_count);

    andPrepareKernel<<<blocksFor(context->current_and_count), kThreads, 0, stream>>>(
            context->and_x.data(), context->and_y.data(),
            context->and_triples.data(), context->de_local.data(), context->current_and_count);

    FLORA_CUDA_CHECK(cudaGetLastError());
    const uint64_t opening_words = 2ULL * context->current_and_count;
    context->host_send.reserve(opening_words);
    context->host_receive.reserve(opening_words);

    const size_t bytes = opening_words * sizeof(uint32_t);

    FLORA_CUDA_CHECK(cudaMemcpyAsync(context->host_send.data(), context->de_local.data(),
        bytes, cudaMemcpyDeviceToHost, stream));
    FLORA_CUDA_CHECK(cudaStreamSynchronize(stream));

    *host_send = context->host_send.data();
    *host_receive = context->host_receive.data();
    *opening_bytes = bytes;
}

extern "C"
void gpu_packed_compare_finish_round(GpuPackedCompareContext* context) {

    if (context == nullptr || context->phase == GpuPackedCompareContext::Phase::Done) {
        std::abort();
    }

    cudaStream_t stream = context->runtime.computeStream();
    const size_t opening_bytes = 2ULL * context->current_and_count * sizeof(uint32_t);

    FLORA_CUDA_CHECK(cudaMemcpyAsync(context->de_peer.data(), context->host_receive.data(),
        opening_bytes, cudaMemcpyHostToDevice, stream));

    andFinishKernel<<<blocksFor(context->current_and_count), kThreads, 0, stream>>>(
            context->de_local.data(), context->de_peer.data(), context->and_triples.data(), context->and_output.data(),
            context->current_and_count, context->party);

    FLORA_CUDA_CHECK(cudaGetLastError());

    if (context->phase == GpuPackedCompareContext::Phase::InitialGt) {
        FLORA_CUDA_CHECK(cudaMemcpyAsync(
            context->current_g.data(), context->and_output.data(),
            context->packed_item_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));

        context->phase = GpuPackedCompareContext::Phase::Prefix;
    } else {
        mergePrefixKernel<<<blocksFor(context->packed_item_count), kThreads, 0, stream>>>(
                context->current_g.data(),
                context->current_e.data(),
                context->and_output.data(),
                context->next_g.data(),
                context->next_e.data(),
                context->packed_words,
                context->key_bits,
                context->prefix_offset);

        FLORA_CUDA_CHECK(cudaGetLastError());
        context->current_g.swap(context->next_g);
        context->current_e.swap(context->next_e);
        context->prefix_offset *= 2;

        if (context->prefix_offset >= context->key_bits) {
            context->phase = GpuPackedCompareContext::Phase::Done;
        }
    }
    FLORA_CUDA_CHECK(cudaStreamSynchronize(stream));
    context->uploaded_triple_count = 0;
}

extern "C"
uint64_t gpu_packed_compare_output_word_count(const GpuPackedCompareContext* context) {
    return context->packed_words;
}

extern "C"
void gpu_packed_compare_download_gt_share(GpuPackedCompareContext* context, uint32_t* output_words) {
    if (context == nullptr || output_words == nullptr || context->phase != GpuPackedCompareContext::Phase::Done) {
        std::abort();
    }
    if (context->comparison_count == 0) { return; }

    const uint64_t final_offset = static_cast<uint64_t>(context->key_bits - 1) * context->packed_words;
    cudaStream_t stream = context->runtime.computeStream();

    FLORA_CUDA_CHECK(cudaMemcpyAsync(output_words, context->current_g.data() + final_offset,
        context->packed_words * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    FLORA_CUDA_CHECK(cudaStreamSynchronize(stream));
}