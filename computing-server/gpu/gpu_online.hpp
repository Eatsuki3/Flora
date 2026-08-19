#ifndef FLORA_GPU_ONLINE_HPP
#define FLORA_GPU_ONLINE_HPP

#include <cstddef>
#include <cstdint>

#include "../preprocess/flora_preprocess_raw.hpp"

struct GpuOnlineContext;

extern "C" {

GpuOnlineContext* gpu_online_create();

void gpu_online_destroy(GpuOnlineContext* context);

// ============================================================
// NeighborPass
// ============================================================

void gpu_neighbor_upload(GpuOnlineContext* context, const uint32_t* host_global_s, uint64_t element_num, int B);

// count = (element_num-offset) * B
void gpu_neighbor_upload_triples(GpuOnlineContext* context, const flora::preprocess::BooleanTripleRecord* host_triples, uint64_t count);

// Returns:
// host_send: D2H completed, ready to send to the peer directly.
// host_receive: CPU network layer writes the peer's opening message here.
// opening_bytes: number of bytes to send/receive.
void gpu_neighbor_prepare(GpuOnlineContext* context, uint64_t offset, int party, uint64_t nonce,
    const void** host_send, void** host_receive, size_t* opening_bytes);

// host_receive must be filled by the network layer.
void gpu_neighbor_finish(GpuOnlineContext* context, uint64_t offset, int party, uint64_t nonce);
void gpu_neighbor_download(GpuOnlineContext* context, uint32_t* host_global_s);

// ============================================================
// Filter
// ============================================================
void gpu_filter_upload_paths(GpuOnlineContext* context, const uint32_t* host_path_shares, uint64_t path_num, int detect_length_k);
void gpu_filter_extract_last(GpuOnlineContext* context);

// Every EQZ batch call once.
// count = element_count。
void gpu_zero_test_upload_correlations(GpuOnlineContext* context, const flora::preprocess::GfZeroTestRecord* host_records, uint64_t count);

// source=0 -> Round 1 (using z_last)
// source=1 -> Round 2 (using z_batch)
void gpu_zero_test_phase1(GpuOnlineContext* context, int source, uint64_t element_count, int party, uint64_t nonce,
    const void** host_send, void** host_receive, size_t* opening_bytes);

// host_receive must include the peer's de opening before calling.
// host_send: include W share.
// host_receive: use for receiving the peer's W share.
void gpu_zero_test_phase2(GpuOnlineContext* context, uint64_t element_count, int party, uint64_t nonce,
    const void** host_send, void** host_receive, size_t* opening_bytes);

// host_receive must include the peer's W share before calling.
void gpu_zero_test_phase3(GpuOnlineContext* context, uint64_t element_count);
uint64_t gpu_online_filter_compact(GpuOnlineContext* context, uint64_t path_num);
void gpu_online_filter_build_zbatch(GpuOnlineContext* context, uint64_t survivor_count, int detect_length_k);
void gpu_online_filter_final_decision(GpuOnlineContext* context, uint64_t survivor_count, int detect_length_k, uint64_t path_num);
void gpu_online_filter_download_decision(GpuOnlineContext* context, uint8_t* host_is_valid, uint8_t* host_is_cycle, uint64_t path_num);
} // extern "C"

#endif