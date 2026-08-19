#ifndef TWO_PC_COMPUTE_GT_HPP
#define TWO_PC_COMPUTE_GT_HPP

#include "common/common.h"
#include "emp-sh2pc/emp-sh2pc/emp-sh2pc.h"
#include "emp-tool/emp-tool/emp-tool.h"
#include "gf.hpp"
#include "preprocess/flora_preprocess_raw.hpp"
#include <atomic>
#include <chrono>

using namespace emp;

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

// Functions for computing 2pc tasks
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
    
    PhaseTimer* timer = nullptr;

void getAndTriples(vector<uint32_t>& a, vector<uint32_t>& b, vector<uint32_t>& c, uint64_t n, int /* party */) {

        MYASSERT(dealer_io != nullptr);
        dealer_io->send_data_internal(reinterpret_cast<const char*>(&n), sizeof(n));
        dealer_io->flush();

        std::vector<flora::preprocess::BooleanTripleRecord> records(n);

        if (n > 0) {
            dealer_io->recv_data_internal(reinterpret_cast<char*>(records.data()),
                                          n * sizeof(flora::preprocess::BooleanTripleRecord));
        }

        a.resize(n);
        b.resize(n);
        c.resize(n);

        for (uint64_t i = 0; i < n; ++i) {
            a[i] = records[i].a_share;
            b[i] = records[i].b_share;
            c[i] = records[i].c_share;
        }
    }

    // reveal Beaver result between two parties
    void revealXor(NetIO* io, const vector<uint32_t>& mine,
                vector<uint32_t>& out, uint64_t n, int party) {
        vector<uint32_t> peer(n);
        if (party == ALICE) { io->send_data(mine.data(), n*4); io->flush(); io->recv_data(peer.data(), n*4); }
        else                { io->recv_data(peer.data(), n*4); io->send_data(mine.data(), n*4); io->flush(); }
        out.resize(n);
        for (uint64_t i = 0; i < n; i++) out[i] = mine[i] ^ peer[i];  
    }

   public:
    TwoPcEngine(int party_, string ip_, int port_, int bit_len_, int tie_breaker_bit_)
        : party(party_),
          ip(ip_),
          port(port_),
          emp_bit_len(bit_len_ + 1),
          bit_len(bit_len_),
          tie_breaker_bit(tie_breaker_bit_) {}

    uint64_t getBytesSent() const { return nio->bytes_sent; }
    uint64_t getBytesRecv() const { return nio->bytes_recv; }
    uint64_t getRounds()    const { return nio->num_flush;  }
    void     resetCommStats()     { nio->reset_stats();     }
    NetIO* getIO() { return this->nio; }
    void setDealerIO(NetIO* io) { dealer_io = io; }
    void setTimer(PhaseTimer* t) { timer = t; }
    void finishDealerStream() {
        MYASSERT(dealer_io != nullptr);

        uint64_t finish = flora::preprocess::kFinishRequest;

        dealer_io->send_data_internal(reinterpret_cast<const char*>(&finish), sizeof(finish));
        dealer_io->flush();
    }

    // GF-based zero-test do not need EMP Integer/Bit,
    // setup_semi_honest() will not be called, only NetIO is needed.
    void initNetIOOnly() {
        MYASSERT(nio == nullptr);

        nio = new NetIO(
            party == ALICE
                ? nullptr
                : ip.c_str(),
            port);

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
        nio = new NetIO(party == ALICE ? nullptr : ip.c_str(), port);
        setup_semi_honest(nio, party);
        semi_honest_initialized = true;
    }

    void terminateEngine() {
        if(semi_honest_initialized) {
            finalize_semi_honest();
            semi_honest_initialized = false;
        }
        if (nio!= nullptr){
            delete nio;
            nio = nullptr;
        }
    }

    void obliviousPartition(
        vector<uint32_t>& node_val_shares, vector<uint32_t>& end_node_shares,
        vector<uint64_t>& tie_breaker_shares, uint32_t pivot_val_share,
        uint32_t pivot_end_node_share, uint64_t pivot_tie_breaker_share,
        vector<uint64_t>& smaller_index, vector<uint64_t>& larger_index,
        uint64_t begin_index, uint64_t curr_vec_index, uint64_t process_num) {
        
        auto t_start = std::chrono::high_resolution_clock::now();
        MYASSERT(node_val_shares.size() == end_node_shares.size());
        MYASSERT(tie_breaker_shares.size() == node_val_shares.size());
        MYASSERT(tie_breaker_shares.size() >= process_num + curr_vec_index);

        Integer ps1(emp_bit_len, pivot_val_share, ALICE);
        Integer ps2(emp_bit_len, pivot_val_share, BOB);
        Integer pivot_node_val = ps1 ^ ps2;

        Integer pes1(emp_bit_len, pivot_end_node_share, ALICE);
        Integer pes2(emp_bit_len, pivot_end_node_share, BOB);
        Integer pivot_end_node = pes1 ^ pes2;

        Integer tbs1(this->tie_breaker_bit, pivot_tie_breaker_share, ALICE);
        Integer tbs2(this->tie_breaker_bit, pivot_tie_breaker_share, BOB);
        Integer pivot_tie_breaker = tbs1 ^ tbs2;

        Integer* node_s1 = new Integer[process_num];
        Integer* node_s2 = new Integer[process_num];

        for (uint64_t i = 0; i < process_num; i++) {
            node_s1[i] = Integer(emp_bit_len, node_val_shares[i + curr_vec_index], ALICE);
            node_s2[i] = Integer(emp_bit_len, node_val_shares[i + curr_vec_index], BOB);
        }

        Integer* node_vals = new Integer[process_num];
        for (uint64_t i = 0; i < process_num; i++) {
            node_vals[i] = node_s1[i] ^ node_s2[i];
        }

        Integer* end_s1 = new Integer[process_num];
        Integer* end_s2 = new Integer[process_num];

        for (uint64_t i = 0; i < process_num; i++) {
            end_s1[i] = Integer(emp_bit_len, end_node_shares[i + curr_vec_index], ALICE);
            end_s2[i] = Integer(emp_bit_len, end_node_shares[i + curr_vec_index], BOB);
        }

        Integer* end_nodes = new Integer[process_num];
        for (uint64_t i = 0; i < process_num; i++) {
            end_nodes[i] = end_s1[i] ^ end_s2[i];
        }

        Integer* tie_breaker_s1 = new Integer[process_num];
        Integer* tie_breaker_s2 = new Integer[process_num];
        for (uint64_t i = 0; i < process_num; i++) {
            tie_breaker_s1[i] = Integer(this->tie_breaker_bit, tie_breaker_shares[i + curr_vec_index], ALICE);
            tie_breaker_s2[i] = Integer(this->tie_breaker_bit, tie_breaker_shares[i + curr_vec_index], BOB);
        }

        Integer* tie_breakers = new Integer[process_num];
        for (uint64_t i = 0; i < process_num; i++) {
            tie_breakers[i] = tie_breaker_s1[i] ^ tie_breaker_s2[i];
        }
        auto t_setup = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_setup - t_start).count());
        }

        // timer start for communication
        auto t_comm_start = std::chrono::high_resolution_clock::now();
        string larger_than_shares(process_num, 0);
        string larger_than_another_shares(process_num, 0);

        for (uint64_t i = 0; i < process_num; i++) {
            Bit node_val_larger_than_pivot = (node_vals[i] > pivot_node_val);
            Bit node_val_equal = (node_vals[i] == pivot_node_val);
            Bit end_node_larger_than_pivot = (end_nodes[i] > pivot_end_node);
            Bit end_node_equal_pivot = (end_nodes[i] == pivot_end_node);
            Bit tie_breakers_larger_than_pivot = (tie_breakers[i] > pivot_tie_breaker);

            Bit larger_than_pivot =
                node_val_larger_than_pivot |
                (node_val_equal & end_node_larger_than_pivot) |
                (node_val_equal & end_node_equal_pivot & tie_breakers_larger_than_pivot);
            larger_than_shares[i] = getLSB(larger_than_pivot.bit) + '0';
        }

        nio->flush();
        if (party == ALICE) {
            this->nio->send_data_internal(larger_than_shares.c_str(), process_num);
            this->nio->recv_data_internal((char*)larger_than_another_shares.c_str(), process_num);
        } else {
            this->nio->recv_data_internal((char*)larger_than_another_shares.c_str(), process_num);
            this->nio->send_data_internal(larger_than_shares.c_str(), process_num);
        }
        nio->flush();
        auto t_comm_end = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comm(std::chrono::duration_cast<std::chrono::microseconds>(t_comm_end - t_comm_start).count());
        }

        auto t_unroll_start = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < process_num; i++) {
            bool larger_s1 = larger_than_shares[i] - '0';
            bool larger_s2 = larger_than_another_shares[i] - '0';
            bool larger_plain = larger_s1 ^ larger_s2;
            if (larger_plain) {
                larger_index.push_back(i + curr_vec_index + begin_index);
            } else {
                smaller_index.push_back(i + curr_vec_index + begin_index);
            }
        }
        auto t_unroll_end = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_unroll_end - t_unroll_start).count());
        }

        delete[] node_s1; delete[] node_s2; delete[] node_vals;
        delete[] end_s1; delete[] end_s2; delete[] end_nodes;
        delete[] tie_breaker_s1; delete[] tie_breaker_s2; delete[] tie_breakers;
    }

    void sortElemsDirectly(vector<uint32_t>& node_val_shares,
                           vector<uint32_t>& end_node_shares,
                           vector<uint64_t>& tie_breaker_shares,
                           uint64_t begin_index,
                           vector<uint64_t>& sorted_index) {
        auto t_start = std::chrono::high_resolution_clock::now();
        MYASSERT(sorted_index.empty());
        MYASSERT(node_val_shares.size() == end_node_shares.size());
        MYASSERT(node_val_shares.size() == tie_breaker_shares.size());

        Integer* node_s1 = new Integer[node_val_shares.size()];
        Integer* node_s2 = new Integer[node_val_shares.size()];

        for (uint64_t i = 0; i < node_val_shares.size(); i++) {
            node_s1[i] = Integer(emp_bit_len, node_val_shares[i], ALICE);
            node_s2[i] = Integer(emp_bit_len, node_val_shares[i], BOB);
        }

        Integer* node_vals = new Integer[node_val_shares.size()];
        for (uint64_t i = 0; i < node_val_shares.size(); i++) {
            node_vals[i] = node_s1[i] ^ node_s2[i];
        }

        Integer* end_s1 = new Integer[end_node_shares.size()];
        Integer* end_s2 = new Integer[end_node_shares.size()];

        for (uint64_t i = 0; i < end_node_shares.size(); i++) {
            end_s1[i] = Integer(emp_bit_len, end_node_shares[i], ALICE);
            end_s2[i] = Integer(emp_bit_len, end_node_shares[i], BOB);
        }

        Integer* end_nodes = new Integer[end_node_shares.size()];
        for (uint64_t i = 0; i < end_node_shares.size(); i++) {
            end_nodes[i] = end_s1[i] ^ end_s2[i];
        }

        Integer* tie_breaker_s1 = new Integer[tie_breaker_shares.size()];
        Integer* tie_breaker_s2 = new Integer[tie_breaker_shares.size()];
        for (uint64_t i = 0; i < tie_breaker_shares.size(); i++) {
            tie_breaker_s1[i] = Integer(this->tie_breaker_bit, tie_breaker_shares[i], ALICE);
            tie_breaker_s2[i] = Integer(this->tie_breaker_bit, tie_breaker_shares[i], BOB);
        }

        Integer* tie_breakers = new Integer[tie_breaker_shares.size()];
        for (uint64_t i = 0; i < tie_breaker_shares.size(); i++) {
            tie_breakers[i] = tie_breaker_s1[i] ^ tie_breaker_s2[i];
        }

        uint64_t elem_num = end_node_shares.size();
        vector<uint64_t> indices;
        for (uint64_t i = 0; i < elem_num; i++) {
            indices.push_back(i);
        }
        auto t_setup = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_setup - t_start).count());
        }

        // timer
        auto t_comm_start = std::chrono::high_resolution_clock::now();
        while (sorted_index.size() < elem_num) {
            uint64_t smallest_index = indices[0];
            uint64_t smallest_id_in_indices = 0;
            for (uint64_t i = 1; i < indices.size(); i++) {
                uint64_t current_index = indices[i];
                Bit current_node_val_smaller = (node_vals[current_index] < node_vals[smallest_index]);
                Bit node_val_equal = (node_vals[current_index] == node_vals[smallest_index]);
                Bit current_end_node_smaller = (end_nodes[current_index] < end_nodes[smallest_index]);
                Bit end_node_equal = (end_nodes[current_index] == end_nodes[smallest_index]);
                Bit tie_breaker_smaller = (tie_breakers[current_index] < tie_breakers[smallest_index]);

                Bit current_is_smaller =
                    current_node_val_smaller |
                    (node_val_equal & current_end_node_smaller) |
                    (node_val_equal & end_node_equal & tie_breaker_smaller);

                bool smaller_plain = current_is_smaller.reveal();
                if (smaller_plain) {
                    smallest_index = current_index;
                    smallest_id_in_indices = i;
                }
            }
            sorted_index.push_back(smallest_index);
            indices.erase(indices.begin() + smallest_id_in_indices);
        }
        auto t_comm_end = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comm(std::chrono::duration_cast<std::chrono::microseconds>(t_comm_end - t_comm_start).count());
        }

        auto t_unroll_start = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < sorted_index.size(); i++) {
            sorted_index[i] += begin_index;
        }
        auto t_unroll_end = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_unroll_end - t_unroll_start).count());
        }

        delete[] node_s1; delete[] node_s2; delete[] node_vals;
        delete[] end_s1; delete[] end_s2; delete[] end_nodes;
        delete[] tie_breaker_s1; delete[] tie_breaker_s2; delete[] tie_breakers;
    }

    uint64_t findMedian(vector<uint32_t>& node_val_shares,
                        vector<uint32_t>& end_node_shares,
                        vector<uint64_t>& tie_breaker_shares) {
        vector<uint64_t> sorted_index;
        sortElemsDirectly(node_val_shares, end_node_shares, tie_breaker_shares, 0, sorted_index);
        return sorted_index[sorted_index.size() / 2];
    }


    // reveal a vector of uint32_t between two parties, and record the communication statistics
    void exchange_u32(const vector<uint32_t>& out, vector<uint32_t>& in,
                    int party, uint64_t& bytes_sent, uint64_t& bytes_recv) {
        uint64_t n = out.size();
        in.assign(n, 0);
        size_t nbytes = n * sizeof(uint32_t);
        nio->flush();
        if (party == ALICE) {
            nio->send_data_internal((const char*)out.data(), nbytes);
            nio->recv_data_internal((char*)in.data(), nbytes);
        } else {
            nio->recv_data_internal((char*)in.data(), nbytes);
            nio->send_data_internal((const char*)out.data(), nbytes);
        }
        nio->flush();
        bytes_sent += nbytes;
        bytes_recv += nbytes;
    }

    // ============================================================================
    // TEST only: local generate correlated randomness for EQZ
    //
    // correlated randomness：
    //     r <- GF(2^ell)^*
    //     b <- GF(2^ell)
    //     c = r * b
    //
    // 2 computing servers hold XOR shares：
    //     r = r_0 XOR r_1
    //     b = b_0 XOR b_1
    //     c = c_0 XOR c_1
    // ============================================================================
    void genEqzCorrelationsForTestOnly(
            uint64_t m,
            int party,
            vector<uint32_t>& r_sh,
            vector<uint32_t>& b_sh,
            vector<uint32_t>& c_sh) {

        r_sh.resize(m);
        b_sh.resize(m);
        c_sh.resize(m);

        static thread_local std::mt19937 rng_shared(0xF117);

        for (uint64_t i = 0; i < m; ++i) {
            uint32_t r = 0;

            // r \in GF(2^ell)^* (r != 0)
            while (r == 0) {
                r = rng_shared();
            }

            uint32_t b = rng_shared();
            uint32_t c = gf_mul(r, b);

            uint32_t r_mask = rng_shared();
            uint32_t b_mask = rng_shared();
            uint32_t c_mask = rng_shared();

            if (party == ALICE) {
                r_sh[i] = r_mask;
                b_sh[i] = b_mask;
                c_sh[i] = c_mask;
            } else {
                r_sh[i] = r ^ r_mask;
                b_sh[i] = b ^ b_mask;
                c_sh[i] = c ^ c_mask;
            }
        }
    }

    void recvEqzCorrelationsFromDealer(uint64_t n, vector<uint32_t>& r_sh, vector<uint32_t>& b_sh, vector<uint32_t>& c_sh) {
        MYASSERT(dealer_io != nullptr);

        // 告诉 S2 Dealer 本次需要多少条。
        dealer_io->send_data_internal(reinterpret_cast<const char*>(&n), sizeof(n));

        dealer_io->flush();

        std::vector<
            flora::preprocess::GfZeroTestRecord
        > records(n);

        if (n > 0) {
            dealer_io->recv_data_internal(
                reinterpret_cast<char*>(records.data()),
                n * sizeof(flora::preprocess::GfZeroTestRecord));
        }

        r_sh.resize(n);
        b_sh.resize(n);
        c_sh.resize(n);

        for (uint64_t i = 0; i < n; ++i) {
            r_sh[i] = records[i].r_share;
            b_sh[i] = records[i].b_share;
            c_sh[i] = records[i].c_share;
        }
    }

    // GF-based zero-test for a batch of XOR shares, output a vector of bits (0/1)
    void GfZeroTestBatch(
            const vector<uint32_t>& z_share,
            vector<uint8_t>& out_bit,
            int party,
            uint64_t& bytes_sent,
            uint64_t& bytes_recv) {

        auto t_start = std::chrono::high_resolution_clock::now();

        const uint64_t m = z_share.size();
        out_bit.assign(m, 0);

        if (m == 0) {
            return;
        }

        // 1. get correlated randomness from dealer:
        // r != 0, b <- GF(2^ell), c = r * b
        vector<uint32_t> r_sh;
        vector<uint32_t> b_sh;
        vector<uint32_t> c_sh;

        // genEqzCorrelationsForTestOnly(m, party, r_sh, b_sh, c_sh);
        recvEqzCorrelationsFromDealer(m, r_sh, b_sh, c_sh);

        // 2. compute [x_hat] = [z] XOR [b]
        vector<uint32_t> x_hat_share(m);

        for (uint64_t i = 0; i < m; ++i) {
            x_hat_share[i] = z_share[i] ^ b_sh[i];
        }

        auto t_setup_done = std::chrono::high_resolution_clock::now();

        if (timer) {
            timer->add_comp(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    t_setup_done - t_start).count());
        }

        // 3. reveal x_hat = z XOR b
        auto t_comm1_start = std::chrono::high_resolution_clock::now();

        vector<uint32_t> peer_x_hat_share;

        exchange_u32(x_hat_share, peer_x_hat_share,
            party, bytes_sent, bytes_recv);

        auto t_comm1_end = std::chrono::high_resolution_clock::now();

        if (timer) {
            timer->add_comm(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    t_comm1_end - t_comm1_start).count());
        }

        // 4. local compute: [W] = [c] XOR (x_hat * [r])
        auto t_local_start = std::chrono::high_resolution_clock::now();

        vector<uint32_t> w_share(m);

        for (uint64_t i = 0; i < m; ++i) {
            uint32_t x_hat =
                x_hat_share[i] ^ peer_x_hat_share[i];

            w_share[i] =
                c_sh[i] ^ gf_mul(x_hat, r_sh[i]);
        }

        auto t_local_end = std::chrono::high_resolution_clock::now();

        if (timer) {
            timer->add_comp(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    t_local_end - t_local_start).count());
        }

        // 5. reconstruct W
        auto t_comm2_start = std::chrono::high_resolution_clock::now();

        vector<uint32_t> peer_w_share;

        exchange_u32(
            w_share, peer_w_share,
            party, bytes_sent, bytes_recv);

        auto t_comm2_end = std::chrono::high_resolution_clock::now();

        if (timer) {
            timer->add_comm(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    t_comm2_end - t_comm2_start).count());
        }

        // 6. output: out_bit[i] = (W == 0)
        auto t_final_start = std::chrono::high_resolution_clock::now();

        for (uint64_t i = 0; i < m; ++i) {
            uint32_t W =
                w_share[i] ^ peer_w_share[i];

            out_bit[i] =
                static_cast<uint8_t>(W == 0);
        }

        auto t_final_end = std::chrono::high_resolution_clock::now();

        if (timer) {
            timer->add_comp(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    t_final_end - t_final_start).count());
        }
    }

    void obliviousFilter(vector<uint32_t>& path_shares,
                        uint64_t tuple_begin_index, uint64_t path_num,
                        int detect_length_k, vector<bool>& is_valid,
                        vector<bool>& is_cycle) {
        auto t_start = std::chrono::high_resolution_clock::now();
        MYASSERT(path_shares.size() >= (uint64_t)(detect_length_k + 1) * (tuple_begin_index + path_num));
        const uint64_t K1 = (uint64_t)(detect_length_k + 1);
        uint64_t base = K1 * tuple_begin_index;
        uint64_t bs = 0, br = 0;

        vector<uint32_t> z_last(path_num);
        for (uint64_t i = 0; i < path_num; i++) {
            z_last[i] = path_shares[base + i * K1 + detect_length_k];
        }
        auto t_setup = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_setup - t_start).count());
        }

        vector<uint8_t> lz;
        GfZeroTestBatch(z_last, lz, party, bs, br);

        auto t_filter = std::chrono::high_resolution_clock::now();
        std::vector<uint64_t> surv;
        surv.reserve(path_num);
        for (uint64_t i = 0; i < path_num; i++) {
            if (lz[i]) {
                is_valid[i + tuple_begin_index] = false;
                is_cycle[i + tuple_begin_index] = false;
            } else {
                surv.push_back(i);
            }
        }
        uint64_t M = surv.size();

        int K = detect_length_k;
        uint64_t n_rep = (uint64_t)(K - 1);
        uint64_t total = M * (2 + n_rep);
        vector<uint32_t> zbatch(total);

        for (uint64_t t = 0; t < M; t++) {
            uint64_t i = surv[t];
            uint32_t v0 = path_shares[base + i * K1 + 0];
            uint32_t vk = path_shares[base + i * K1 + K];
            zbatch[0 * M + t] = v0;
            zbatch[1 * M + t] = vk ^ v0;
            for (int j = 1; j < K; j++) {
                uint32_t vj = path_shares[base + i * K1 + j];
                zbatch[(2 + (j - 1)) * M + t] = vk ^ vj;
            }
        }
        auto t_batch_end = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_batch_end - t_filter).count());
        }

        vector<uint8_t> bbatch;
        GfZeroTestBatch(zbatch, bbatch, party, bs, br);

        auto t_unroll_start = std::chrono::high_resolution_clock::now();
        for (uint64_t t = 0; t < M; t++) {
            uint64_t i = surv[t];
            bool first_zero = bbatch[0 * M + t];
            bool cycle      = bbatch[1 * M + t];
            bool repeat = false;
            for (int j = 1; j < K; j++) {
                repeat = repeat || (bool)bbatch[(2 + (j - 1)) * M + t];
            }

            bool valid = !first_zero && !repeat;
            is_valid[i + tuple_begin_index] = valid;
            is_cycle[i + tuple_begin_index] = valid ? cycle : false;
        }
        auto t_unroll_end = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_unroll_end - t_unroll_start).count());
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

        // timer
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

    void beaverBatchAnd(NetIO* io,
                        const vector<uint32_t>& X, const vector<uint32_t>& Y,
                        vector<uint32_t>& Z, uint64_t n, int party) {
        auto t_start = std::chrono::high_resolution_clock::now();
        Z.resize(n);
        vector<uint32_t> a, b, c;
        getAndTriples(a, b, c, n, party);

        vector<uint32_t> de_share(2*n);
        for (uint64_t i = 0; i < n; i++) {
            de_share[i]     = X[i] ^ a[i];  
            de_share[n + i] = Y[i] ^ b[i];  
        }
        auto t_setup = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_setup - t_start).count());
        }

        // timer
        auto t_comm_start = std::chrono::high_resolution_clock::now();
        vector<uint32_t> de;
        revealXor(io, de_share, de, 2*n, party);
        auto t_comm_end = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comm(std::chrono::duration_cast<std::chrono::microseconds>(t_comm_end - t_comm_start).count());
        }

        auto t_final_start = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < n; i++) {
            uint32_t d = de[i], e = de[n+i];
            Z[i] = c[i] ^ (d & b[i]) ^ (e & a[i]) ^ ((party == ALICE) ? (d & e) : 0u);
        }
        auto t_final_end = std::chrono::high_resolution_clock::now();
        if (timer) {
            timer->add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_final_end - t_final_start).count());
        }
    }
};

#endif  // TWO_PC_COMPUTE_GT_HPP