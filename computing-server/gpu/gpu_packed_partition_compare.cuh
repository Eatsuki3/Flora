#ifndef FLORA_GPU_PACKED_PARTITION_COMPARE_CUH
#define FLORA_GPU_PACKED_PARTITION_COMPARE_CUH

#include <cstddef>
#include <cstdint>

#include "../preprocess/flora_preprocess_raw.hpp"

struct GpuPackedCompareContext;

extern "C" {

// Construct bit-packed GPU GMW comparator,
// node bits: actual plaintext bit width of node_val and end_node,
// tie bits: bit width of tie_breaker
GpuPackedCompareContext* gpu_packed_compare_create(int party, int node_bits, int tie_bits);

void gpu_packed_compare_destroy(GpuPackedCompareContext* context);

// Triples
uint64_t gpu_packed_compare_next_triple_count(const GpuPackedCompareContext* context);
void gpu_packed_compare_upload_triples(GpuPackedCompareContext* context, const flora::preprocess::BooleanTripleRecord* host_triples, uint64_t count);

// Start a batch of lexicographical comparisons:
// element key: (node_share[i], end_share[i], tie_share[i])
// pivot key: (pivot_node_share[i], pivot_end_share[i], pivot_tie_share[i])
// Each i may correspond to a distinct pivot. Therefore one batch can aggregate all partitions within the current quicksort wave.
void gpu_packed_compare_begin(GpuPackedCompareContext* context,
    const uint32_t* node_share, const uint32_t* end_share, const uint64_t* tie_share,
    const uint32_t* pivot_node_share, const uint32_t* pivot_end_share, const uint64_t* pivot_tie_share,
    uint64_t comparison_count, uint64_t batch_nonce);

// if comparator has Beaver AND communication round.
int gpu_packed_compare_has_round(const GpuPackedCompareContext* context);

void gpu_packed_compare_prepare_round(
    GpuPackedCompareContext* context,
    const void** host_send,
    void** host_receive,
    size_t* opening_bytes);

// host_receive must be filled before calling
void gpu_packed_compare_finish_round(GpuPackedCompareContext* context);

// After comparator completes, download packed local gt shares.
// Output format: Bit i % 32 of output_words[i / 32] = local XOR share of the i-th comparison result.
uint64_t gpu_packed_compare_output_word_count(const GpuPackedCompareContext* context);

void gpu_packed_compare_download_gt_share(GpuPackedCompareContext* context, uint32_t* output_words);

} // extern "C"

#endif