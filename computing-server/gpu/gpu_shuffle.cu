#include "gpu_shuffle.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kThreads = 256;

int blocksFor(uint64_t count) {
    if (count == 0) {
        return 0;
    }

    uint64_t blocks = (count + static_cast<uint64_t>(kThreads) - 1ULL) / static_cast<uint64_t>(kThreads);
    if (blocks > static_cast<uint64_t>(0x7fffffff)) {
        std::fprintf(stderr, "gpu_shuffle: CUDA grid too large\n");
        std::abort();
    }

    return static_cast<int>(blocks);
}

void validateCompatible(const DeviceTupleStore& left, const DeviceTupleStore& right, const char* operation) {
    if (left.size() != right.size() || left.fieldCount() != right.fieldCount() ||
        left.maxDegree() != right.maxDegree() || left.detectLength() != right.detectLength()) {
        std::fprintf(stderr, "gpu_shuffle: incompatible stores in %s\n", operation);
        std::abort();
    }
}

__device__ __forceinline__ uint64_t mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ __forceinline__ uint32_t randomU32(uint64_t seed, uint64_t tuple, uint64_t field, uint64_t domain) {

    uint64_t input = seed ^ (tuple * 0xd6e8feb86659fd93ULL) ^
        (field * 0xa5a3564e27f8862fULL) ^ (domain * 0x9e3779b97f4a7c15ULL);

    return static_cast<uint32_t>(
        mix64(input));
}

__device__ __forceinline__ uint64_t randomU64(uint64_t seed, uint64_t tuple, uint64_t domain) {

    uint64_t low = static_cast<uint64_t>(randomU32(seed, tuple, 0, domain));
    uint64_t high = static_cast<uint64_t>(randomU32(seed, tuple, 1, domain));
    return (high << 32) | low;
}

__device__ __forceinline__ uint32_t reduceU32(uint32_t value, uint32_t range) {

    if (range == 0) {
        return value;
    }

    return value % range;
}

__device__ __forceinline__ uint64_t reduceU64(uint64_t value, uint64_t range) {

    if (range == 0) {
        return value;
    }

    return value % range;
}

__global__ void generateTupleMetadataKernel(
    uint32_t* __restrict__ node_id, uint32_t* __restrict__ d_share, uint64_t* __restrict__ tie_breaker,
    uint64_t tuple_count, uint32_t id_range, uint64_t tie_breaker_range, uint64_t seed) {

    uint64_t tuple = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (tuple >= tuple_count) { return; }

    node_id[tuple] = reduceU32(randomU32(seed, tuple, 0, 0x1001ULL), id_range);
    d_share[tuple] = reduceU32(randomU32(seed, tuple, 0, 0x1002ULL), id_range);
    tie_breaker[tuple] = reduceU64(randomU64(seed, tuple, 0x1003ULL), tie_breaker_range);
}

__global__ void generateTupleFieldsKernel(uint32_t* __restrict__ secret_vec, 
    uint64_t tuple_count, uint64_t field_count, uint32_t id_range, uint64_t seed) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    uint64_t total = tuple_count * field_count;

    if (index >= total) { return; }

    uint64_t field = index / tuple_count;
    uint64_t tuple = index % tuple_count;
    secret_vec[index] = reduceU32(randomU32(seed, tuple, field, 0x2001ULL), id_range);
}

__global__ void xorU32Kernel(const uint32_t* __restrict__ left, const uint32_t* __restrict__ right, uint32_t* __restrict__ output, uint64_t count) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        output[index] = left[index] ^ right[index];
    }
}

__global__ void xorU64Kernel(const uint64_t* __restrict__ left, const uint64_t* __restrict__ right, uint64_t* __restrict__ output, uint64_t count) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        output[index] = left[index] ^ right[index];
    }
}

__global__ void xor3U32Kernel(const uint32_t* __restrict__ first, const uint32_t* __restrict__ second,
    const uint32_t* __restrict__ third, uint32_t* __restrict__ output, uint64_t count) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        output[index] = first[index] ^ second[index] ^ third[index];
    }
}

__global__ void xor3U64Kernel(const uint64_t* __restrict__ first, const uint64_t* __restrict__ second,
    const uint64_t* __restrict__ third, uint64_t* __restrict__ output, uint64_t count) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        output[index] = first[index] ^ second[index] ^ third[index];
    }
}

__global__ void permuteU32Kernel(const uint32_t* __restrict__ input, uint32_t* __restrict__ output,
    const uint64_t* __restrict__ permutation, uint64_t tuple_count) {

    uint64_t tuple = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (tuple < tuple_count) {
        output[tuple] = input[permutation[tuple]];
    }
}

__global__ void permuteU64Kernel(const uint64_t* __restrict__ input, uint64_t* __restrict__ output,
    const uint64_t* __restrict__ permutation, uint64_t tuple_count) {

    uint64_t tuple = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (tuple < tuple_count) {
        output[tuple] = input[permutation[tuple]];
    }
}

__global__ void permuteFieldsKernel(const uint32_t* __restrict__ input, uint32_t* __restrict__ output,
    const uint64_t* __restrict__ permutation, uint64_t tuple_count, uint64_t field_count) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    uint64_t total = tuple_count * field_count;
    if (index >= total) { return; }

    uint64_t field = index / tuple_count;
    uint64_t output_tuple = index % tuple_count;
    uint64_t input_tuple = permutation[output_tuple];

    output[field * tuple_count + output_tuple] = input[field * tuple_count + input_tuple];
}

} // namespace

void gpuGenerateRandomTupleStore(
    GpuRuntime& runtime, DeviceTupleStore& output, uint64_t tuple_count,
    int max_degree, int detect_length_k,
    uint32_t id_range, uint64_t tie_breaker_range, uint64_t seed) {

    output.configure(tuple_count, max_degree, detect_length_k);
    output.setSize(tuple_count);

    if (tuple_count == 0) { return; }
    cudaStream_t stream = runtime.computeStream();

    generateTupleMetadataKernel<<<blocksFor(tuple_count), kThreads, 0, stream>>>(
            output.nodeId(), output.dShare(), output.tieBreaker(),
            tuple_count, id_range, tie_breaker_range, seed);

    uint64_t field_words = tuple_count * output.fieldCount();

    generateTupleFieldsKernel<<<blocksFor(field_words), kThreads, 0, stream>>>(
            output.secretVec(), tuple_count, output.fieldCount(), id_range, seed);
    FLORA_CUDA_CHECK(cudaGetLastError());
}

void gpuShuffleXor2(GpuRuntime& runtime, const DeviceTupleStore& left,const DeviceTupleStore& right, DeviceTupleStore& output) {

    validateCompatible(left, right, "gpuShuffleXor2");
    output.configure(left.size(), left.maxDegree(), left.detectLength());
    output.setSize(left.size());

    if (left.size() == 0) { return; }
    cudaStream_t stream = runtime.computeStream();
    uint64_t tuple_count = left.size();
    uint64_t field_words = tuple_count * left.fieldCount();

    xorU32Kernel<<<blocksFor(tuple_count), kThreads, 0, stream>>>(
            left.nodeId(), right.nodeId(), output.nodeId(), tuple_count);
    xorU32Kernel<<<blocksFor(tuple_count), kThreads, 0, stream>>>(
            left.dShare(), right.dShare(), output.dShare(), tuple_count);
    xorU64Kernel<<<blocksFor(tuple_count), kThreads, 0, stream>>>(
            left.tieBreaker(), right.tieBreaker(), output.tieBreaker(), tuple_count);
    xorU32Kernel<<<blocksFor(field_words), kThreads, 0, stream>>>(
            left.secretVec(), right.secretVec(), output.secretVec(), field_words);

    FLORA_CUDA_CHECK(cudaGetLastError());
}

void gpuShuffleXor3(GpuRuntime& runtime, const DeviceTupleStore& first, const DeviceTupleStore& second, const DeviceTupleStore& third, DeviceTupleStore& output) {

    validateCompatible(first, second, "gpuShuffleXor3(first, second)");
    validateCompatible(first, third, "gpuShuffleXor3(first, third)");

    output.configure(first.size(), first.maxDegree(), first.detectLength());
    output.setSize(first.size());
    if (first.size() == 0) { return; }

    cudaStream_t stream = runtime.computeStream();
    uint64_t tuple_count = first.size();
    uint64_t field_words = tuple_count * first.fieldCount();

    xor3U32Kernel<<<blocksFor(tuple_count), kThreads, 0, stream>>>(
            first.nodeId(), second.nodeId(), third.nodeId(), output.nodeId(), tuple_count);
    xor3U32Kernel<<<blocksFor(tuple_count), kThreads, 0, stream>>>(
            first.dShare(), second.dShare(), third.dShare(), output.dShare(), tuple_count);
    xor3U64Kernel<<<blocksFor(tuple_count), kThreads, 0, stream>>>(
            first.tieBreaker(), second.tieBreaker(), third.tieBreaker(), output.tieBreaker(), tuple_count);
    xor3U32Kernel<<<blocksFor(field_words), kThreads, 0, stream>>>(
            first.secretVec(), second.secretVec(), third.secretVec(), output.secretVec(), field_words);

    FLORA_CUDA_CHECK(cudaGetLastError());
}

void gpuShufflePermute(GpuRuntime& runtime, const DeviceTupleStore& input, DeviceTupleStore& output, const uint64_t* host_permutation) {

    output.configure(input.size(), input.maxDegree(), input.detectLength());
    output.setSize(input.size());

    if (input.size() == 0) { return; }

    DeviceBuffer<uint64_t> device_permutation(input.size());
    cudaStream_t transfer = runtime.transferStream();

    FLORA_CUDA_CHECK(cudaMemcpyAsync(device_permutation.data(), host_permutation,
        input.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, transfer));
    FLORA_CUDA_CHECK(cudaEventRecord(runtime.transferReadyEvent(), transfer));
    cudaStream_t compute = runtime.computeStream();
    FLORA_CUDA_CHECK(cudaStreamWaitEvent(compute, runtime.transferReadyEvent(), 0));

    permuteU32Kernel<<<blocksFor(input.size()), kThreads, 0, compute>>>(
            input.nodeId(), output.nodeId(), device_permutation.data(), input.size());
    permuteU32Kernel<<<blocksFor(input.size()), kThreads, 0, compute>>>(
            input.dShare(), output.dShare(), device_permutation.data(), input.size());
    permuteU64Kernel<<<blocksFor(input.size()), kThreads, 0, compute>>>(
            input.tieBreaker(), output.tieBreaker(), device_permutation.data(), input.size());
    uint64_t field_words = input.size() * input.fieldCount();
    permuteFieldsKernel<<<blocksFor(field_words), kThreads, 0, compute>>>(
            input.secretVec(), output.secretVec(), device_permutation.data(), input.size(), input.fieldCount());

    FLORA_CUDA_CHECK(cudaGetLastError());
    FLORA_CUDA_CHECK(cudaStreamSynchronize(compute));
}

void gpuShuffleCopy(GpuRuntime& runtime, const DeviceTupleStore& input, DeviceTupleStore& output) {

    output.configure(input.size(), input.maxDegree(), input.detectLength());
    output.setSize(input.size());

    if (input.size() == 0) { return; }
    cudaStream_t stream = runtime.computeStream();

    FLORA_CUDA_CHECK(cudaMemcpyAsync(output.nodeId(), input.nodeId(), 
        input.size() * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    FLORA_CUDA_CHECK(cudaMemcpyAsync(output.dShare(), input.dShare(),
        input.size() * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
    FLORA_CUDA_CHECK(cudaMemcpyAsync(output.tieBreaker(), input.tieBreaker(),
        input.size() * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
    FLORA_CUDA_CHECK(cudaMemcpyAsync(output.secretVec(), input.secretVec(),
        input.size() * input.fieldCount() * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
}