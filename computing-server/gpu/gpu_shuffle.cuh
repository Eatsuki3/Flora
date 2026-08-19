#ifndef FLORA_GPU_SHUFFLE_CUH
#define FLORA_GPU_SHUFFLE_CUH

#include "device_tuple_store.cuh"

#include <cstdint>


void gpuGenerateRandomTupleStore(
    GpuRuntime& runtime,
    DeviceTupleStore& output,
    uint64_t tuple_count,
    int max_degree,
    int detect_length_k,
    uint32_t id_range,
    uint64_t tie_breaker_range,
    uint64_t seed);

// output = left XOR right
void gpuShuffleXor2(
    GpuRuntime& runtime,
    const DeviceTupleStore& left,
    const DeviceTupleStore& right,
    DeviceTupleStore& output);

// output = first XOR second XOR third
void gpuShuffleXor3(
    GpuRuntime& runtime,
    const DeviceTupleStore& first,
    const DeviceTupleStore& second,
    const DeviceTupleStore& third,
    DeviceTupleStore& output);

/*
 * 使用原 CPU Shuffle 相同的 gather 语义：
 *
 * output[i] = input[permutation[i]]
 *
 * permutation 在 Host 上由：
 *
 * std::shuffle(
 *     permutation.begin(),
 *     permutation.end(),
 *     std::default_random_engine(seed));
 *
 * 生成。
 */
void gpuShufflePermute(
    GpuRuntime& runtime,
    const DeviceTupleStore& input,
    DeviceTupleStore& output,
    const uint64_t* host_permutation);

/*
 * output = input 的完整 GPU copy。
 */
void gpuShuffleCopy(
    GpuRuntime& runtime,
    const DeviceTupleStore& input,
    DeviceTupleStore& output);

#endif