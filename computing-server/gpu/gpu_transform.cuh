#ifndef FLORA_GPU_TRANSFORM_CUH
#define FLORA_GPU_TRANSFORM_CUH

#include "device_tuple_store.cuh"

#include <cstdint>

void gpuXorTupleStores(GpuRuntime& runtime, const DeviceTupleStore& left, const DeviceTupleStore& right, DeviceTupleStore& output);
void gpuPermuteTupleStore(GpuRuntime& runtime, const DeviceTupleStore& input, DeviceTupleStore& output, const uint64_t* host_permutation);
void gpuAssignTieBreaker(GpuRuntime& runtime, DeviceTupleStore& store, uint64_t public_mask_share, bool assign_index);
void gpuExplodePaths(GpuRuntime& runtime, const DeviceTupleStore& input, DeviceTupleStore& output);
void gpuPadPaths(GpuRuntime& runtime, const DeviceTupleStore& input, DeviceTupleStore& output,
    int output_max_degree, int output_detect_length);

#endif