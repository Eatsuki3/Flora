#include "gpu_transform.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kThreads = 256;

int blocksFor(uint64_t count) {
    return static_cast<int>(
        (count + kThreads - 1) / kThreads);
}

__global__ void xor_u32_kernel(const uint32_t* __restrict__ left, const uint32_t* __restrict__ right,
    uint32_t* __restrict__ output, uint64_t count) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        output[index] = left[index] ^ right[index];
    }
}

__global__ void xor_u64_kernel(const uint64_t* __restrict__ left, const uint64_t* __restrict__ right,
    uint64_t* __restrict__ output, uint64_t count) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        output[index] = left[index] ^ right[index];
    }
}

__global__ void permute_u32_kernel(const uint32_t* __restrict__ input, uint32_t* __restrict__ output,
    const uint64_t* __restrict__ permutation, uint64_t count) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        output[index] = input[permutation[index]];
    }
}

__global__ void permute_u64_kernel(const uint64_t* __restrict__ input, uint64_t* __restrict__ output,
    const uint64_t* __restrict__ permutation, uint64_t count) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        output[index] =
            input[permutation[index]];
    }
}

__global__ void permute_fields_kernel(
    const uint32_t* __restrict__ input, uint32_t* __restrict__ output,
    const uint64_t* __restrict__ permutation, uint64_t tuple_count, uint64_t field_count) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    uint64_t total = tuple_count * field_count;
    if (index >= total) { return; }
    uint64_t field = index / tuple_count;
    uint64_t tuple = index % tuple_count;

    output[field * tuple_count + tuple] = input[field * tuple_count + permutation[tuple]];
}

__global__ void assign_tie_breaker_kernel(uint64_t* tie_breaker, uint64_t count, uint64_t public_mask_share, bool assign_index) {

    uint64_t index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (index < count) {
        tie_breaker[index] = assign_index ? index : public_mask_share;
    }
}

__global__ void explode_paths_kernel(
    const uint32_t* __restrict__ input_secret, uint32_t* __restrict__ output_node,
    uint32_t* __restrict__ output_d,  uint64_t* __restrict__ output_tie, uint32_t* __restrict__ output_secret,
    uint64_t input_count, int max_degree, int detect_length_k) {

    uint64_t output_count = input_count * static_cast<uint64_t>(max_degree);
    uint64_t output_index = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;

    if (output_index >= output_count) { return; }

    uint64_t input_tuple = output_index / static_cast<uint64_t>(max_degree);
    int degree = static_cast<int>(output_index % static_cast<uint64_t>(max_degree));
    int output_fields = detect_length_k + 1;

    output_node[output_index] = 0;
    output_d[output_index] = 0;
    output_tie[output_index] = 0;

    for (int path_index = 0; path_index < output_fields; ++path_index) {
        uint64_t input_field = static_cast<uint64_t>(degree) *
                static_cast<uint64_t>(output_fields) + static_cast<uint64_t>(path_index);
        output_secret[static_cast<uint64_t>(path_index) * output_count + output_index] =
            input_secret[input_field * input_count + input_tuple];
    }
}

__global__ void pad_paths_kernel(
    const uint32_t* __restrict__ input_node, const uint32_t* __restrict__ input_d, const uint64_t* __restrict__ input_tie, const uint32_t* __restrict__ input_secret,
    uint32_t* __restrict__ output_node, uint32_t* __restrict__ output_d, uint64_t* __restrict__ output_tie, uint32_t* __restrict__ output_secret,
    uint64_t tuple_count, int output_max_degree, int input_detect_length, int output_detect_length) {

    uint64_t tuple = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
    if (tuple >= tuple_count) { return; }
    output_node[tuple] = input_node[tuple];
    output_d[tuple] = input_d[tuple];
    output_tie[tuple] = input_tie[tuple];
    int input_width = input_detect_length + 1;
    int output_width = output_detect_length + 1;

    for (int degree = 0; degree < output_max_degree; ++degree) {
        for (int position = 0; position < output_width; ++position) {
            uint32_t value = 0;
            if (position < input_width) {
                value = input_secret[static_cast<uint64_t>(position) * tuple_count + tuple];
            }

            uint64_t output_field = static_cast<uint64_t>(degree) * static_cast<uint64_t>(output_width) + static_cast<uint64_t>(position);
            output_secret[output_field * tuple_count + tuple] = value;
        }
    }
}

void validateCompatible(const DeviceTupleStore& left, const DeviceTupleStore& right) {

    if (left.size() != right.size() || left.fieldCount() != right.fieldCount() ||
        left.maxDegree() != right.maxDegree() || left.detectLength() != right.detectLength()) {

        std::fprintf(stderr, "Incompatible DeviceTupleStore objects\n");
        std::abort();
    }
}

}  // namespace

void gpuXorTupleStores(GpuRuntime& runtime, const DeviceTupleStore& left, const DeviceTupleStore& right, DeviceTupleStore& output) {

    validateCompatible(left, right);

    output.configure(left.size(), left.maxDegree(), left.detectLength());
    output.setSize(left.size());

    uint64_t tuple_count = left.size();
    uint64_t field_words = tuple_count * left.fieldCount();

    if (tuple_count == 0) { return; }

    cudaStream_t stream = runtime.computeStream();

    xor_u32_kernel<<< blocksFor(tuple_count), kThreads, 0, stream>>>(
            left.nodeId(), right.nodeId(), output.nodeId(), tuple_count);
    xor_u32_kernel<<<blocksFor(tuple_count), kThreads, 0, stream>>>(
            left.dShare(), right.dShare(), output.dShare(), tuple_count);
    xor_u64_kernel<<<blocksFor(tuple_count), kThreads, 0, stream>>>(
            left.tieBreaker(), right.tieBreaker(), output.tieBreaker(), tuple_count);
    xor_u32_kernel<<<blocksFor(field_words), kThreads, 0, stream>>>(
            left.secretVec(), right.secretVec(), output.secretVec(), field_words);
    FLORA_CUDA_CHECK(cudaGetLastError());
}

void gpuPermuteTupleStore(GpuRuntime& runtime, const DeviceTupleStore& input, DeviceTupleStore& output, const uint64_t* host_permutation) {

    output.configure(input.size(), input.maxDegree(), input.detectLength());
    output.setSize(input.size());
    if (input.size() == 0) { return; }
    DeviceBuffer<uint64_t> permutation(input.size());
    cudaStream_t transfer = runtime.transferStream();

    FLORA_CUDA_CHECK(cudaMemcpyAsync(permutation.data(), host_permutation,
        input.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, transfer));
    FLORA_CUDA_CHECK(cudaEventRecord(runtime.transferReadyEvent(), transfer));
    cudaStream_t compute = runtime.computeStream();
    FLORA_CUDA_CHECK(cudaStreamWaitEvent(compute, runtime.transferReadyEvent(), 0));

    permute_u32_kernel<<<blocksFor(input.size()), kThreads, 0, compute>>>(
            input.nodeId(), output.nodeId(), permutation.data(), input.size());
    permute_u32_kernel<<< blocksFor(input.size()), kThreads, 0, compute>>>(
            input.dShare(), output.dShare(), permutation.data(), input.size());
    permute_u64_kernel<<<blocksFor(input.size()), kThreads, 0, compute>>>(
            input.tieBreaker(), output.tieBreaker(), permutation.data(), input.size());
    uint64_t field_words = input.size() * input.fieldCount();
    permute_fields_kernel<<<blocksFor(field_words), kThreads, 0, compute>>>(
            input.secretVec(), output.secretVec(), permutation.data(), input.size(), input.fieldCount());

    FLORA_CUDA_CHECK(cudaGetLastError());
    FLORA_CUDA_CHECK(cudaStreamSynchronize(compute));
}

void gpuAssignTieBreaker(GpuRuntime& runtime, DeviceTupleStore& store, uint64_t public_mask_share, bool assign_index) {

    if (store.size() == 0) { return; }

    assign_tie_breaker_kernel<<<blocksFor(store.size()), kThreads, 0, runtime.computeStream()>>>(
            store.tieBreaker(), store.size(), public_mask_share, assign_index);
    FLORA_CUDA_CHECK(cudaGetLastError());
}

void gpuExplodePaths(GpuRuntime& runtime, const DeviceTupleStore& input, DeviceTupleStore& output) {

    uint64_t output_count = input.size() * static_cast<uint64_t>(input.maxDegree());
    output.configure(output_count, 1, input.detectLength());
    output.setSize(output_count);
    if (output_count == 0) { return; }

    explode_paths_kernel<<< blocksFor(output_count), kThreads, 0, runtime.computeStream()>>>(
            input.secretVec(), output.nodeId(), output.dShare(), output.tieBreaker(), output.secretVec(),
            input.size(), input.maxDegree(), input.detectLength());
    FLORA_CUDA_CHECK(cudaGetLastError());
}

void gpuPadPaths(GpuRuntime& runtime, const DeviceTupleStore& input, DeviceTupleStore& output, int output_max_degree, int output_detect_length) {

    if (input.maxDegree() != 1) {
        std::fprintf(stderr, "gpuPadPaths requires input max_degree == 1\n");
        std::abort();
    }
    output.configure(input.size(), output_max_degree, output_detect_length);
    output.setSize(input.size());
    if (input.size() == 0) { return; }

    pad_paths_kernel<<<blocksFor(input.size()), kThreads, 0, runtime.computeStream()>>>(
            input.nodeId(), input.dShare(), input.tieBreaker(), input.secretVec(),
            output.nodeId(), output.dShare(), output.tieBreaker(), output.secretVec(),
            input.size(), output_max_degree, input.detectLength(), output_detect_length);
    FLORA_CUDA_CHECK(cudaGetLastError());
}