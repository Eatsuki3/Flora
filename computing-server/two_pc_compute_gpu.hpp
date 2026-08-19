#ifndef TWO_PC_COMPUTE_GPU_HPP
#define TWO_PC_COMPUTE_GPU_HPP

#include "common/common.h"
#include "preprocess/flora_preprocess_raw.hpp"
#include "gpu/gpu_online.hpp"
#include "emp-sh2pc/emp-sh2pc/emp-sh2pc.h"
#include "emp-tool/emp-tool/emp-tool.h"
#include "gf.hpp"
#include <vector>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>

using namespace emp;

#ifndef CUDA_CHECK
#define CUDA_CHECK(val) { \
    cudaError_t err = val; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}
#endif

// Timer
struct PhaseTimer {
    std::atomic<uint64_t> local_comp_us{0};
    std::atomic<uint64_t> comm_us{0};
    std::atomic<uint64_t> serial_us{0};
    
    std::chrono::high_resolution_clock::time_point phase_start;

    PhaseTimer() {
        reset();
    }

    void reset() {
        local_comp_us = 0;
        comm_us = 0;
        serial_us = 0;
        phase_start = std::chrono::high_resolution_clock::now();
    }

    void add_comp(uint64_t us) { local_comp_us += us; }
    void add_comm(uint64_t us) { comm_us += us; }
    void add_serial(uint64_t us) { serial_us += us; }

    double comp_s() const { return local_comp_us.load() / 1000000.0; }
    double comm_s() const { return comm_us.load() / 1000000.0; }
    double serial_s() const { return serial_us.load() / 1000000.0; }
};

extern "C" {
    void gpu_memcpy_h2d_sync(void* dst, const void* src, size_t bytes);
    void gpu_memcpy_d2h_sync(void* dst, const void* src, size_t bytes);
    void gpu_memset_sync(void* dst, int val, size_t bytes);
    void* gpu_scratch_get(int slot, size_t bytes);
    void gpu_pool_destroy();
    void gpu_alloc_buffers(uint32_t** d_S, uint32_t** d_S_temp, const uint32_t* host_S, uint64_t size);
    void gpu_free_buffers(uint32_t* d_S, uint32_t* d_S_temp);
    void gpu_ks_tree_prep(
        const uint32_t* d_global_S, uint32_t* host_inA, uint32_t* host_inB,
        uint64_t active_num, uint64_t offset, int B, int max_degree, int party,
        uint64_t* pcie_us, uint64_t* kernel_us);
    
    void gpu_ks_tree_update(
        uint32_t** d_global_S_temp, uint32_t** d_global_S, const uint32_t* host_outC,
        uint64_t active_num, uint64_t offset, int B, int max_degree,
        uint64_t* pcie_us, uint64_t* kernel_us);
        
    void gpu_copy_back_final_S(uint32_t* host_S, const uint32_t* d_S, uint64_t size);
    
    void gpu_beaver_prep(
        const uint32_t* h_X, const uint32_t* h_Y, const uint32_t* h_a, const uint32_t* h_b,
        uint32_t* h_de_share, uint64_t n, uint64_t* pcie_us, uint64_t* kernel_us);
    
    void gpu_beaver_recon(
        const uint32_t* h_c, const uint32_t* h_a, const uint32_t* h_b,
        const uint32_t* h_de, uint32_t* h_Z, uint64_t n, int party,
        uint64_t* pcie_us, uint64_t* kernel_us);
    

    // interface for GPU filter phase, including zero-test, extract z_last,
    // reconstruct to lz, and final decision.
    void gpu_gf_zero_test_prep(
        const uint32_t* h_z_share, uint32_t* h_de_out,
        uint64_t m, int party, 
        uint64_t* pcie_us, uint64_t* kernel_us);

    void gpu_gf_zero_test_recon(
        const uint32_t* h_de_recon, uint32_t* d_W_sh,
        uint64_t m, int party,
        uint64_t* pcie_us, uint64_t* kernel_us);

    void gpu_filter_extract_z_last(
        const uint32_t* h_path_shares, uint32_t* h_z_last,
        uint64_t base, int K1, int K, uint64_t path_num,
        uint64_t* pcie_us, uint64_t* kernel_us);

    void gpu_recon_to_lz(
        const uint32_t* d_W_sh, const uint32_t* d_w_in, uint8_t* d_lz, uint64_t m,
        uint64_t* pcie_us, uint64_t* kernel_us);

    void gpu_final_decision(
        const uint8_t* d_bbatch, const uint64_t* d_surv, uint8_t* d_is_valid, uint8_t* d_is_cycle,
        uint64_t M, int K, uint64_t tuple_begin_index,
        uint64_t* pcie_us, uint64_t* kernel_us);
}

// =========================================================================

class TwoPcEngine {
   private:
    int party;
    string ip;
    int port;
    int emp_bit_len;
    int bit_len;

    int tie_breaker_bit;
    NetIO* nio = nullptr;
    NetIO* dealer_io = nullptr;
    bool semi_honest_initialized = false;

    GpuOnlineContext* gpu_online_context = nullptr;
    PhaseTimer* timer = nullptr;

   public:
    TwoPcEngine(int party_, string ip_, int port_, int bit_len_, int tie_breaker_bit_)
        : party(party_),
          ip(ip_),
          port(port_),
          emp_bit_len(bit_len_ + 1),
          bit_len(bit_len_),
          tie_breaker_bit(tie_breaker_bit_) {}

    ~TwoPcEngine() {
        if (gpu_online_context != nullptr) {
            gpu_online_destroy(gpu_online_context);
            gpu_online_context = nullptr;
        }
    }

    uint64_t getBytesSent() const { return nio->bytes_sent; }
    uint64_t getBytesRecv() const { return nio->bytes_recv; }
    uint64_t getRounds()    const { return nio->num_flush;  }
    void     resetCommStats()     { nio->reset_stats();     }
    NetIO* getIO() { return this->nio; }
    void setDealerIO(NetIO* io) { dealer_io = io; }
    void setTimer(PhaseTimer* t) { timer = t; }

    void receiveFilterCorrelations(uint64_t count, std::vector<flora::preprocess::GfZeroTestRecord>& records) {

        MYASSERT(dealer_io != nullptr);
        dealer_io->send_data_internal(reinterpret_cast<const char*>(&count), sizeof(count));
        dealer_io->flush();

        records.resize(static_cast<std::size_t>(count));
        if (count > 0) {
            dealer_io->recv_data_internal(reinterpret_cast<char*>(records.data()), count * sizeof(flora::preprocess::GfZeroTestRecord));
        }
    }

    void finishDealerStream() {
        MYASSERT(dealer_io != nullptr);

        const uint64_t finish = flora::preprocess::kFinishRequest;
        dealer_io->send_data_internal(reinterpret_cast<const char*>(&finish), sizeof(finish));
        dealer_io->flush();
    }

    GpuOnlineContext* getGpuOnlineContext() {
        if (gpu_online_context == nullptr) {
            gpu_online_context = gpu_online_create();
        }

        return gpu_online_context;
    }

    void exchangeRaw(const void* send_buffer,void* receive_buffer,size_t bytes) {
        nio->flush();
        if (party == ALICE) {
            nio->send_data_internal(static_cast<const char*>(send_buffer), bytes);
            nio->recv_data_internal(static_cast<char*>(receive_buffer), bytes);
        } else {
            nio->recv_data_internal(static_cast<char*>(receive_buffer), bytes);
            nio->send_data_internal(static_cast<const char*>(send_buffer), bytes);
        }
        nio->flush();
    }


    void initNetIOOnly() {
        MYASSERT(nio == nullptr);
        nio = new NetIO((party == ALICE) ? nullptr : ip.c_str(), port);
        semi_honest_initialized = false;
    }

    void terminateNetIOOnly() {
        MYASSERT(!semi_honest_initialized);
        if (nio != nullptr) {
            delete nio;
            nio = nullptr;
        }
    }

    void initEngine() {
        MYASSERT(nio == nullptr);
        nio = new NetIO((party == ALICE) ? nullptr : ip.c_str(), port);
        setup_semi_honest(nio, party);
        semi_honest_initialized = true;
    }

    void terminateEngine() {
        if(semi_honest_initialized){
            finalize_semi_honest();
            semi_honest_initialized = false;
        }
        if(nio != nullptr){
            delete nio;
            nio = nullptr;
        }
    }
    
    void obliviousFilter(vector<uint32_t>& path_shares, uint64_t tuple_begin_index, uint64_t path_num,
        int detect_length_k, vector<bool>& is_valid, vector<bool>& is_cycle) {

        if (path_num == 0) { return; }
        const uint64_t width = static_cast<uint64_t>(detect_length_k + 1);
        const uint64_t base = tuple_begin_index * width;
        MYASSERT(base + path_num * width <= path_shares.size());
        GpuOnlineContext* context = getGpuOnlineContext();
        
        // only upload current batch.
        gpu_filter_upload_paths(context, path_shares.data() + base, path_num, detect_length_k);
        gpu_filter_extract_last(context);

        const void* send_buffer = nullptr;
        void* receive_buffer = nullptr;
        size_t exchange_bytes = 0;

        // nonce must be unique for each round of zero-test, 
        // and should be different between the two parties.
        uint64_t nonce_first = 0x100000000ULL + tuple_begin_index * static_cast<uint64_t>(detect_length_k + 1);

        std::vector<flora::preprocess::GfZeroTestRecord> first_correlations;
        receiveFilterCorrelations(path_num, first_correlations);
        MYASSERT(first_correlations.size() == path_num);
        gpu_zero_test_upload_correlations(context, first_correlations.data(), path_num);

        // open x_hat = z XOR b for the first EQZ batch (first round).
        gpu_zero_test_phase1(context, 0, path_num, party, nonce_first, &send_buffer, &receive_buffer, &exchange_bytes);
        auto comm_begin = std::chrono::high_resolution_clock::now();
        exchangeRaw(send_buffer, receive_buffer, exchange_bytes);
        auto comm_end = std::chrono::high_resolution_clock::now();
        if (timer != nullptr) {
            timer->add_comm(std::chrono::duration_cast<std::chrono::microseconds>(comm_end - comm_begin).count());
        }

        // open W (first round).
        gpu_zero_test_phase2(context, path_num, party, nonce_first, &send_buffer, &receive_buffer, &exchange_bytes);
        comm_begin = std::chrono::high_resolution_clock::now();
        exchangeRaw(send_buffer, receive_buffer, exchange_bytes);
        comm_end = std::chrono::high_resolution_clock::now();
        if (timer != nullptr) {
            timer->add_comm(std::chrono::duration_cast<std::chrono::microseconds>(comm_end - comm_begin).count());
        }

        gpu_zero_test_phase3(context, path_num);
        uint64_t survivor_count = gpu_online_filter_compact(context, path_num);

        // Debug outputs.
        std::fprintf(
            stderr,
            "[GPU Filter][party=%d] tuple_begin=%llu path_num=%llu first-round survivors=%llu\n",
            party, static_cast<unsigned long long>(tuple_begin_index), static_cast<unsigned long long>(path_num),
            static_cast<unsigned long long>(survivor_count));
        // Debug outputs end.

        gpu_online_filter_build_zbatch(context, survivor_count, detect_length_k);

        if (survivor_count > 0) {
            uint64_t second_count = survivor_count * static_cast<uint64_t>(detect_length_k + 1);
            uint64_t nonce_second = 0x500000000ULL + tuple_begin_index * static_cast<uint64_t>(detect_length_k + 1);

            std::vector<flora::preprocess::GfZeroTestRecord> second_correlations;
            receiveFilterCorrelations(second_count, second_correlations);
            MYASSERT(second_correlations.size() == second_count);
            gpu_zero_test_upload_correlations(context, second_correlations.data(), second_count);

            // open d,e (second round).
            gpu_zero_test_phase1(context, 1, second_count, party, nonce_second, &send_buffer, &receive_buffer, &exchange_bytes);

            comm_begin = std::chrono::high_resolution_clock::now();
            exchangeRaw(send_buffer, receive_buffer, exchange_bytes);
            comm_end = std::chrono::high_resolution_clock::now();
            if (timer != nullptr) {
                timer->add_comm(std::chrono::duration_cast<std::chrono::microseconds>(comm_end - comm_begin).count());
            }

            // open W (second round).
            gpu_zero_test_phase2(context, second_count, party, nonce_second, &send_buffer, &receive_buffer, &exchange_bytes);

            comm_begin = std::chrono::high_resolution_clock::now();
            exchangeRaw(send_buffer, receive_buffer, exchange_bytes);
            comm_end = std::chrono::high_resolution_clock::now();
            if (timer != nullptr) {
                timer->add_comm(std::chrono::duration_cast<std::chrono::microseconds>(comm_end - comm_begin).count());
            }

            gpu_zero_test_phase3(context, second_count);
        }

        gpu_online_filter_final_decision(context, survivor_count, detect_length_k, path_num);

        vector<uint8_t> valid_bytes(path_num);
        vector<uint8_t> cycle_bytes(path_num);
        gpu_online_filter_download_decision(context, valid_bytes.data(), cycle_bytes.data(), path_num);

        MYASSERT(tuple_begin_index + path_num <= is_valid.size());

        MYASSERT(tuple_begin_index + path_num <= is_cycle.size());

        for (uint64_t index = 0; index < path_num; ++index) {
            is_valid[tuple_begin_index + index] = valid_bytes[index] != 0;
            is_cycle[tuple_begin_index + index] = cycle_bytes[index] != 0;
        }
    }

    void obliviousExtractPath(vector<uint32_t>& node_shares,
                              vector<uint64_t>& edge_tuple_indices,
                              uint64_t begin_index, uint64_t end_index,
                              uint64_t curr_vec_begin_index,
                              uint64_t processed_tuple_num) {
        auto t_start = std::chrono::high_resolution_clock::now();
        MYASSERT(node_shares.size() == (end_index - begin_index));
        MYASSERT(node_shares.size() >= (curr_vec_begin_index + processed_tuple_num));

        Integer* s1 = new Integer[processed_tuple_num];
        Integer* s2 = new Integer[processed_tuple_num];

        for (uint64_t i = 0; i < processed_tuple_num; i++) {
            s1[i] = Integer(emp_bit_len, node_shares[i + curr_vec_begin_index], ALICE);
            s2[i] = Integer(emp_bit_len, node_shares[i + curr_vec_begin_index], BOB);
        }

        string tuple_is_path_shares;
        Integer zero(emp_bit_len, 0, PUBLIC);
        for (uint64_t i = 0; i < processed_tuple_num; i++) {
            Integer node = s1[i] ^ s2[i];
            Bit node_equal_zero = (node == zero);
            bool current_tuple_is_path_share = getLSB(node_equal_zero.bit);
            tuple_is_path_shares.append(current_tuple_is_path_share ? "1" : "0");
        }
        auto t_setup = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_setup - t_start).count());
        }

        auto t_comm_start = std::chrono::high_resolution_clock::now();
        nio->flush();
        string tuple_is_path_another_shares(processed_tuple_num, 0);
        if (party == ALICE) {
            this->nio->send_data_internal(tuple_is_path_shares.c_str(), processed_tuple_num);
            this->nio->recv_data_internal((char*)tuple_is_path_another_shares.c_str(), processed_tuple_num);
        } else {
            this->nio->recv_data_internal((char*)tuple_is_path_another_shares.c_str(), processed_tuple_num);
            this->nio->send_data_internal(tuple_is_path_shares.c_str(), processed_tuple_num);
        }
        nio->flush();
        auto t_comm_end = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comm(std::chrono::duration_cast<std::chrono::microseconds>(t_comm_end - t_comm_start).count());
        }

        auto t_write_start = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < processed_tuple_num; i++) {
            bool current_tuple_is_path_s1 = tuple_is_path_another_shares[i] - '0';
            bool current_tuple_is_path_s2 = tuple_is_path_shares[i] - '0';
            bool current_tuple_is_path = current_tuple_is_path_s1 ^ current_tuple_is_path_s2;
            if (!current_tuple_is_path) {
                edge_tuple_indices.push_back(i + curr_vec_begin_index + begin_index);
            }
        }
        auto t_write_end = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_write_end - t_write_start).count());
        }

        delete[] s1; delete[] s2;
    }

};

#endif  // TWO_PC_COMPUTE_GPU_HPP