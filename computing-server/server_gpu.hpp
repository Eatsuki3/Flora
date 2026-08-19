#ifndef SERVER_GPU_HPP
#define SERVER_GPU_HPP

#include <sstream>
#include <fstream>
#include <memory>
#include <chrono>
#include <vector>
#include <thread>
#include <algorithm>
#include <map>
#include <mutex>
#include <atomic>
#include <netinet/tcp.h>

#include "preprocess/flora_preprocess_raw.hpp"
#include "preprocess/flora_dealer_triple_store_gpu.hpp"
#include "gpu/gpu_packed_partition_compare.cuh"
#include "gpu/tuple_bridge.hpp"
#include "gpu/gpu_shuffle_protocol.hpp"
#include "gpu/gpu_online.hpp"
#include "secret-share/secret_share.hpp"
#include "two_pc_compute_gpu.hpp" 
#include "gf.hpp"

static constexpr int GPU_FILTER_STREAMS = 1;

class NetServer {
   public:

    PhaseTimer timer;

    struct PartitionTaskInfo {
        uint32_t pivot_val_share;
        uint32_t pivot_end_node_share;
        uint64_t begin_index;
        uint64_t end_index;
        uint64_t partition_begin;
        uint64_t partition_end;
        bool run_partition;
        uint64_t pivot_index;
        uint64_t pivot_tie_breaker_share;
    };

    struct PartitionResult {
        vector<uint64_t> smaller_index;
        vector<uint64_t> larger_index;
    };

    struct PivotInfo {
        uint32_t pivot_val_share;
        uint32_t pivot_end_node_share;
        uint64_t pivot_index;
        uint64_t pivot_tie_breaker_share;
    };

    struct PartitionThreadsInfo {
        vector<int> threads;
        uint64_t pivot_index;
    };

    struct PackedPartitionReference {
        int thread_id;
        pair<uint64_t, uint64_t> partition_key;
        uint64_t absolute_index;
    };

   protected:
    unique_ptr<Server> compute_server;

    unique_ptr<ServerSecretShares> pristine_edges_first;
    unique_ptr<ServerSecretShares> pristine_edges_second;

    SERVER_ID server_id;
    int bit_len;
    uint32_t id_range;
    int max_degree;
    int detect_length_k;
    int total_detect_length;

    // parallelism
    int local_compute_parallel_num;
    int neighbor_pass_parallel_num;
    int sort_parallel_num;
    int filter_parallel_num;
    int extract_path_parallel_num;

    uint64_t sort_elem_threshold;
    int num_elem_to_find_median;

    string ip_1; string ip_2; string ip_3;

    uint64_t filter_batch_num;
    uint64_t neighbor_pass_batch_num;
    uint64_t extract_batch_num;
    uint64_t partition_batch_num;

    int tie_breaker_bit;
    uint64_t tie_breaker_range;
    int fd_1; int fd_2; int fd_3;
    string ip_no_port_1; string ip_no_port_2; string ip_no_port_3;
    int seed12_permute;
    int seed23_permute;
    int seed31_permute;
    int seedA;
    int seedB;
    std::unique_ptr<GpuShuffleProtocol> gpu_shuffle_protocol;
 

    int rand_median_seed = 100;
    int start_port = 12345;
    int reassign_randomness_seed = 2345;
    int tmp_max_degree;
    int dealer_start_port = 30000;
    static const int DEALER_PORT_GAP = 5000;

    std::unique_ptr<flora::preprocess::DealerTripleStore> dealer_triple_store;

    void setMaxDegreeToOne() {
        this->tmp_max_degree = this->max_degree;
        this->max_degree = 1;
        compute_server->set_max_degree(1);
    }

    void resetMaxDegree() {
        this->max_degree = this->tmp_max_degree;
        compute_server->set_max_degree(this->max_degree);
    }


    void printTimerStats(const string& phase_name) {
        auto phase_end = std::chrono::high_resolution_clock::now();
        double wall_clock = std::chrono::duration<double>(phase_end - timer.phase_start).count();
        
        double comp = timer.comp_s();
        double comm = timer.comm_s();
        double ser  = timer.serial_s();
        double sum_accumulated = comp + comm + ser;
        
        double comp_norm = 0;
        double comm_norm = 0;
        double ser_norm = 0;
 
        if (sum_accumulated > 0) {
            comp_norm = (comp / sum_accumulated) * wall_clock;
            comm_norm = (comm / sum_accumulated) * wall_clock;
            ser_norm  = (ser / sum_accumulated) * wall_clock;
        } else {
            comp_norm = wall_clock; 
        }
        
        cout << "\n┌────────────────────────────────────────────────────────" << endl;
        cout << "│  [TIMING PROFILE] Module: " << phase_name << endl;
        cout << "├────────────────────────────────────────────────────────" << endl;
        cout << "│  1. computation time:     " << comp_norm  << " s  \t(" << (wall_clock > 0 ? (comp_norm / wall_clock * 100) : 0) << "%)" << endl;
        cout << "│  2. communication time:   " << comm_norm  << " s  \t(" << (wall_clock > 0 ? (comm_norm / wall_clock * 100) : 0) << "%)" << endl;
        cout << "│  3. Serial time: " << ser_norm   << " s  \t(" << (wall_clock > 0 ? (ser_norm / wall_clock * 100) : 0) << "%)" << endl;
        cout << "├────────────────────────────────────────────────────────" << endl;
        cout << "│  End-to-end time:        " << wall_clock << " s" << endl;
        cout << "└────────────────────────────────────────────────────────" << endl;
        timer.reset();
    }

   public:
    NetServer(SERVER_ID server_id_, uint32_t id_range_, int bit_len_,
              int max_degree_, int detect_length_k_, int total_detect_length_,
              int local_compute_parallel_num_, int neighbor_pass_parallel_num_,
              int sort_parallel_num_, int filter_parallel_num_,
              int extract_path_parallel_num_, uint64_t sort_elem_threshold_,
              int num_elem_to_find_median_, int seed12_permute_,
              int seed23_permute_, int seed31_permute_, int seedA_, int seedB_,
              string ip_1_, string ip_2_, string ip_3_,
              uint64_t filter_batch_num_, uint64_t neighbor_pass_batch_num_,
              uint64_t extract_batch_num_, uint64_t partition_batch_num_,
              int tie_breaker_bit_)
        : server_id(server_id_),
          bit_len(bit_len_),
          id_range(id_range_),
          max_degree(max_degree_),
          detect_length_k(detect_length_k_),
          total_detect_length(total_detect_length_),
          local_compute_parallel_num(local_compute_parallel_num_),
          neighbor_pass_parallel_num(neighbor_pass_parallel_num_),
          sort_parallel_num(sort_parallel_num_),
          filter_parallel_num(filter_parallel_num_),
          extract_path_parallel_num(extract_path_parallel_num_),
          sort_elem_threshold(sort_elem_threshold_),
          num_elem_to_find_median(num_elem_to_find_median_),
          ip_1(ip_1_), ip_2(ip_2_), ip_3(ip_3_),
          filter_batch_num(filter_batch_num_),
          neighbor_pass_batch_num(neighbor_pass_batch_num_),
          extract_batch_num(extract_batch_num_),
          partition_batch_num(partition_batch_num_),
          seed12_permute(seed12_permute_),
          seed23_permute(seed23_permute_),
          seed31_permute(seed31_permute_),
          seedA(seedA_),
          seedB(seedB_),
          tie_breaker_bit(tie_breaker_bit_) {
        ip_no_port_1 = split(ip_1, ":")[0];
        ip_no_port_2 = split(ip_2, ":")[0];
        ip_no_port_3 = split(ip_3, ":")[0];

        MYASSERT(tie_breaker_bit > 0 && tie_breaker_bit < 64);
        tie_breaker_range = 1ULL << tie_breaker_bit;

        compute_server = make_unique<Server>(
            server_id_, id_range_, max_degree_, detect_length_k_,
            local_compute_parallel_num_, seed12_permute_, seed23_permute_,
            seed31_permute_, seedA_, seedB_, this->tie_breaker_range);

        gpu_shuffle_protocol = std::make_unique<GpuShuffleProtocol>(
                this->id_range, this->tie_breaker_range);
    }

    ~NetServer() {}

    enum class FloraDealerDomain {
        Sort,
        Neighbor,
        Filter
    };

    int sortDealerPortForS0() const {
        return dealer_start_port + 200;
    }

    int sortDealerPortForS1() const {
        return dealer_start_port + DEALER_PORT_GAP + 200;
    }

    int neighborDealerPortForS0() const {
        return dealer_start_port;
    }

    int neighborDealerPortForS1() const {
        return dealer_start_port + DEALER_PORT_GAP;
    }

    int filterDealerPortForS0(int thread_id) const {
        return dealer_start_port + 100 + thread_id;
    }

    int filterDealerPortForS1(int thread_id) const {
        return dealer_start_port + DEALER_PORT_GAP + 100 + thread_id;
    }

    NetIO* connectToDealer(int party, int port_for_s0, int port_for_s1) {
        const int port = (party == ALICE) ? port_for_s0 : port_for_s1;
        NetIO* dealer_io = new NetIO(ip_no_port_3.c_str(), port);
        dealer_io->sync();
        return dealer_io;
    }

    NetIO* connectToSortDealer(int party) {
        return connectToDealer(party, sortDealerPortForS0(), sortDealerPortForS1());
    }

    NetIO* connectToNeighborDealer(int party) {
        return connectToDealer(party, neighborDealerPortForS0(), neighborDealerPortForS1());
    }

    NetIO* connectToFilterDealer(int party, int thread_id) {
        return connectToDealer(party, filterDealerPortForS0(thread_id), filterDealerPortForS1(thread_id));
    }

    void loadDealerTripleFiles(
        const std::string& sort_s0_path, const std::string& sort_s1_path,
        const std::string& neighbor_s0_path, const std::string& neighbor_s1_path,
        const std::string& filter_s0_path, const std::string& filter_s1_path) {

        if (server_id != SERVER_ID::S3) { return; }

        dealer_triple_store = std::make_unique<flora::preprocess::DealerTripleStore>(
                    sort_s0_path, sort_s1_path, neighbor_s0_path, neighbor_s1_path, filter_s0_path, filter_s1_path);
    }

    template<typename Record> void sendDealerRecordPair(NetIO* io_to_s0, NetIO* io_to_s1,
        const std::vector<Record>& records_s0, const std::vector<Record>& records_s1) {

        MYASSERT(records_s0.size() == records_s1.size());
        const size_t bytes = records_s0.size() * sizeof(Record);
        auto comm_begin = std::chrono::high_resolution_clock::now();

        std::thread send_s0([&] {
            if (bytes > 0) {
                io_to_s0->send_data_internal(reinterpret_cast<const char*>(records_s0.data()), bytes);
            }
            io_to_s0->flush();
        });

        std::thread send_s1([&] {
            if (bytes > 0) {
                io_to_s1->send_data_internal(reinterpret_cast<const char*>(records_s1.data()),bytes);
            }
            io_to_s1->flush();
        });

        send_s0.join();
        send_s1.join();
        auto comm_end = std::chrono::high_resolution_clock::now();
        timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(comm_end - comm_begin).count());
    }

    template<typename Record> void requestDealerRecords(NetIO* dealer_io, uint64_t count, std::vector<Record>& records) {

        MYASSERT(dealer_io != nullptr);

        dealer_io->send_data_internal(reinterpret_cast<const char*>(&count), sizeof(count));
        dealer_io->flush();
        records.resize(static_cast<std::size_t>(count));

        if (count > 0) {
            dealer_io->recv_data_internal(reinterpret_cast<char*>(records.data()), count * sizeof(Record));
        }
    }

    void finishDealerStream(NetIO* dealer_io) {
        MYASSERT(dealer_io != nullptr);

        const uint64_t finish = flora::preprocess::kFinishRequest;
        dealer_io->send_data_internal(reinterpret_cast<const char*>(&finish), sizeof(finish));
        dealer_io->flush();
    }
    
    void serveDealerStream(int port_for_s0, int port_for_s1, FloraDealerDomain domain) {

        MYASSERT(server_id == SERVER_ID::S3);
        MYASSERT(dealer_triple_store != nullptr);

        NetIO* io_to_s0 = nullptr;
        NetIO* io_to_s1 = nullptr;

        std::thread listen_s0([&] {
            io_to_s0 = new NetIO(nullptr, port_for_s0);
            io_to_s0->sync();
        });

        std::thread listen_s1([&] {
            io_to_s1 = new NetIO(nullptr, port_for_s1);
            io_to_s1->sync();
        });

        listen_s0.join();
        listen_s1.join();

        MYASSERT(io_to_s0 != nullptr);
        MYASSERT(io_to_s1 != nullptr);

        while (true) {
            uint64_t count_s0 = 0;
            uint64_t count_s1 = 0;

            std::thread receive_s0([&] {
                io_to_s0->recv_data_internal(reinterpret_cast<char*>(&count_s0), sizeof(count_s0));
            });

            std::thread receive_s1([&] {
                io_to_s1->recv_data_internal(reinterpret_cast<char*>(&count_s1), sizeof(count_s1));
            });

            receive_s0.join();
            receive_s1.join();

            MYASSERT(count_s0 == count_s1);

            if (count_s0 == flora::preprocess::kFinishRequest) {
                break;
            }

            const uint64_t count = count_s0;

            switch (domain) {
            case FloraDealerDomain::Sort: {
                std::vector<flora::preprocess::BooleanTripleRecord> records_s0;
                std::vector<flora::preprocess::BooleanTripleRecord> records_s1;

                dealer_triple_store->takeSort(count, records_s0, records_s1);
                sendDealerRecordPair(io_to_s0, io_to_s1, records_s0, records_s1);
                break;
            }

            case FloraDealerDomain::Neighbor: {
                std::vector<flora::preprocess::BooleanTripleRecord> records_s0;
                std::vector<flora::preprocess::BooleanTripleRecord> records_s1;

                dealer_triple_store->takeNeighbor(count, records_s0, records_s1);
                sendDealerRecordPair(io_to_s0, io_to_s1, records_s0, records_s1);
                break;
            }

            case FloraDealerDomain::Filter: {
                std::vector<flora::preprocess::GfZeroTestRecord> records_s0;
                std::vector<flora::preprocess::GfZeroTestRecord> records_s1;

                dealer_triple_store->takeFilter(count, records_s0, records_s1);
                sendDealerRecordPair(io_to_s0, io_to_s1, records_s0, records_s1);
                break;
            }

            default:
                MYASSERT(0);
            }
        }
        delete io_to_s0;
        delete io_to_s1;
    }



    void loadSecretShare(string share_file1, string share_file2) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ifstream in1(share_file1, ios::binary);
        MYASSERT(in1.is_open());
        stringstream ss1;
        ss1 << in1.rdbuf();
        in1.close();
        string serialized_str1 = ss1.str();
        compute_server->loadFirstServerSecretShares(serialized_str1);
        serialized_str1.clear();

        uint64_t V_boundary = 0;
        ifstream in1_meta(share_file1 + "_meta");
        if (in1_meta.is_open()) {
            in1_meta >> V_boundary;
            in1_meta.close();
        } else {
            V_boundary = compute_server->getFirstServerSecretShares()->size();
        }

        ifstream in2(share_file2, ios::binary);
        MYASSERT(in2.is_open());
        stringstream ss2;
        ss2 << in2.rdbuf();
        in2.close();
        string serialized_str2 = ss2.str();
        compute_server->loadSecondServerSecretShares(serialized_str2);
        serialized_str2.clear();

        compute_server->assignTieBreakerIds();

        pristine_edges_first = make_unique<ServerSecretShares>();
        pristine_edges_second = make_unique<ServerSecretShares>();
        
        for (uint64_t i = 0; i < V_boundary; i++) {
            unique_ptr<SecretShareTuple> t1 = make_unique<SecretShareTuple>(
                *(compute_server->getFirstServerSecretShares()->svec[i].get()));
            unique_ptr<SecretShareTuple> t2 = make_unique<SecretShareTuple>(
                *(compute_server->getSecondServerSecretShares()->svec[i].get()));
            pristine_edges_first->insert(t1);
            pristine_edges_second->insert(t2);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    }

    ServerSecretShares* getFirstServerSecretShares() {
        return compute_server->getFirstServerSecretShares();
    }

    ServerSecretShares* getSecondServerSecretShares() {
        return compute_server->getSecondServerSecretShares();
    }

    
    void exchangeNodeIdShares(NetIO* io, const vector<uint32_t>& mine,
                            vector<uint32_t>& peer, int party) {
        uint64_t n = mine.size();
        peer.resize(n);
        if (party == ALICE) {
            io->send_data(mine.data(), n*4); io->flush();
            io->recv_data(peer.data(), n*4);
        } else {
            io->recv_data(peer.data(), n*4);
            io->send_data(mine.data(), n*4); io->flush();
        }
    }

    void obliviousPrefixNeighborPass(unique_ptr<ServerSecretShares>& ss,
        string another_server_ip, int party) {

        using Clock = std::chrono::high_resolution_clock;
        uint64_t element_num = ss->size();

        
        NetIO* dealer_io = connectToNeighborDealer(party);
        if (element_num == 0) {
            finishDealerStream(dealer_io);
            delete dealer_io;
            return;
        }
        const int B = max_degree + 1;
        // if (element_num == 0) { return; }

        auto pack_begin = Clock::now();
        vector<uint32_t> global_s;
        global_s.reserve(element_num * static_cast<uint64_t>(B));

        for (uint64_t index = 0; index < element_num; ++index) {
            ss->svec[index]-> extractAndAppendDAndNeighbos(global_s);
        }

        auto pack_end = Clock::now();
        timer.add_comp(std::chrono::duration_cast<
                        std::chrono::microseconds>(pack_end - pack_begin).count());

        GpuOnlineContext* context = gpu_online_create();

        gpu_neighbor_upload(context, global_s.data(), element_num, B);
        int neighbor_port = start_port + 50;
        NetIO* io = new NetIO(party == ALICE ? nullptr : another_server_ip.c_str(), neighbor_port);
        uint64_t nonce_cursor = 0x900000000ULL;

        for (uint64_t offset = 1; offset < element_num; offset <<= 1) {
            const void* send_buffer = nullptr;
            void* receive_buffer = nullptr;
            size_t opening_bytes = 0;
            uint64_t active_num = element_num - offset;
            uint64_t round_words = active_num * static_cast<uint64_t>(B);
            std::vector<flora::preprocess::BooleanTripleRecord> neighbor_triples;
        
            requestDealerRecords(dealer_io, round_words, neighbor_triples);

            MYASSERT(neighbor_triples.size() == round_words);
            gpu_neighbor_upload_triples(context, neighbor_triples.data(), round_words);
            const uint64_t nonce = nonce_cursor;

            nonce_cursor += round_words + 4096ULL;
            gpu_neighbor_prepare(context, offset, party, nonce,
                &send_buffer, &receive_buffer, &opening_bytes);
            auto comm_begin = Clock::now();
            io->flush();

            if (party == ALICE) {
                io->send_data_internal(static_cast<const char*>(send_buffer), opening_bytes);
                io->recv_data_internal(static_cast<char*>(receive_buffer), opening_bytes);
            } else {
                io->recv_data_internal(static_cast<char*>(receive_buffer), opening_bytes);
                io->send_data_internal(static_cast<const char*>(send_buffer), opening_bytes);
            }
            io->flush();

            auto comm_end = Clock::now();
            timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(
                        comm_end - comm_begin).count());
            gpu_neighbor_finish(context, offset, party, nonce);
        }

        gpu_neighbor_download(context, global_s.data());
        finishDealerStream(dealer_io);
        delete dealer_io;
        dealer_io = nullptr;
        delete io;
        io = nullptr;

        gpu_online_destroy(context);
        context = nullptr;
        auto write_begin = Clock::now();
        vector<uint32_t> updated_neighbors(max_degree);
        for (uint64_t tuple = 0; tuple < element_num; ++tuple) {
            for (int degree = 0; degree < max_degree; ++degree) {
                updated_neighbors[degree] =
                    global_s[tuple * static_cast<uint64_t>(B) +
                            1ULL + static_cast<uint64_t>(degree)];
            }
            ss->svec[tuple]-> updateNeighbors(updated_neighbors);
        }
        auto write_end = Clock::now();
        timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(write_end - write_begin).count());
    }

    uint64_t choosePackedPivotIndex(uint64_t begin_index, uint64_t end_index, uint64_t sort_round) const {

        MYASSERT(begin_index < end_index);
        const uint64_t element_count = end_index - begin_index;

        // two parties must derive the same pivot index based on public information.
        // The input data has been obfuscated by ObShuffle before sorting, 
        // so publicly selecting a position does not reveal the key at that position.
        uint64_t value = 0x9e3779b97f4a7c15ULL;
        value ^= begin_index + 0xbf58476d1ce4e5b9ULL;
        value ^= end_index * 0x94d049bb133111ebULL;
        value ^= sort_round * 0xd6e8feb86659fd93ULL;
        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27;
        value *= 0x94d049bb133111ebULL;
        value ^= value >> 31;

        return begin_index + value % element_count;
    }

    void packedGpuPartitionWave(unique_ptr<ServerSecretShares>& ss, vector<vector<PartitionTaskInfo>>& partition_info_per_thread,
        vector<map<pair<uint64_t, uint64_t>, PartitionResult>>& partition_res_per_thread,
        string another_server_ip, int party, uint64_t sort_round, uint64_t invocation, NetIO* dealer_io) {
            
        MYASSERT(dealer_io != nullptr);
        using Clock = std::chrono::high_resolution_clock;

        vector<uint32_t> node_shares;
        vector<uint32_t> end_shares;
        vector<uint64_t> tie_shares;

        vector<uint32_t> pivot_node_shares;
        vector<uint32_t> pivot_end_shares;
        vector<uint64_t> pivot_tie_shares;

        vector<PackedPartitionReference> references;

        uint64_t total_comparisons = 0;

        for (const auto& task_vec :
            partition_info_per_thread) {

            for (const auto& task : task_vec) {
                if (task.run_partition) {
                    total_comparisons += task.end_index - task.begin_index;
                }
            }
        }

        if (total_comparisons == 0) { return; }

        node_shares.reserve(total_comparisons);
        end_shares.reserve(total_comparisons);
        tie_shares.reserve(total_comparisons);

        pivot_node_shares.reserve(total_comparisons);
        pivot_end_shares.reserve(total_comparisons);
        pivot_tie_shares.reserve(total_comparisons);

        references.reserve(total_comparisons);

        auto pack_begin = Clock::now();

        // S1/S2 must traverse in the same order.
        // thread_id ascending, task in vector original order, tuple index ascending.
        for (int thread_id = 0; thread_id < sort_parallel_num; ++thread_id) {
            for (const auto& task :  partition_info_per_thread[thread_id]) {
                if (!task.run_partition) {  continue; }
                const pair<uint64_t, uint64_t> partition_key(task.partition_begin, task.partition_end);
                
                // init the partition result for this thread and partition if not exist.
                // Each thread maintains its own local result for the same partition.
                if (
                    partition_res_per_thread[thread_id].find(partition_key) == partition_res_per_thread[thread_id].end()) {
                    partition_res_per_thread[thread_id][partition_key] = PartitionResult{
                            .smaller_index = vector<uint64_t>(),
                            .larger_index = vector<uint64_t>(),
                        };
                }

                for (uint64_t tuple_index = task.begin_index; tuple_index < task.end_index; ++tuple_index) {
                    const SecretShareTuple& tuple = *ss->svec[tuple_index];
                    node_shares.push_back(tuple.getNodeValShare());
                    end_shares.push_back(tuple.getEndNodeShare());
                    tie_shares.push_back(tuple.get_tie_breaker_id());

                    pivot_node_shares.push_back(task.pivot_val_share);
                    pivot_end_shares.push_back(task.pivot_end_node_share);
                    pivot_tie_shares.push_back(task.pivot_tie_breaker_share);

                    references.push_back(
                        PackedPartitionReference{
                            .thread_id = thread_id,
                            .partition_key = partition_key,
                            .absolute_index = tuple_index,
                        });
                }
            }
        }

        MYASSERT(node_shares.size() == total_comparisons);
        MYASSERT(references.size() == total_comparisons);

        auto pack_end = Clock::now();
        timer.add_comp(
            std::chrono::duration_cast<std::chrono::microseconds>(pack_end - pack_begin).count());
        // emp_bit_len = bit_len + 1
        // The extra bit is a 0-sign bit, the actual data still has only bit_len bits.
        // Therefore, the packed comparator can directly compare the bit_len valid bits.
        GpuPackedCompareContext* context = gpu_packed_compare_create(party, this->bit_len, this->tie_breaker_bit);
        const uint64_t batch_nonce = 0x330000000ULL ^ (invocation * 0x100000ULL) ^ (sort_round * 0x1000ULL);
        
        auto gpu_begin = Clock::now();
        gpu_packed_compare_begin(
            context, node_shares.data(), end_shares.data(), tie_shares.data(),
            pivot_node_shares.data(), pivot_end_shares.data(), pivot_tie_shares.data(),
            total_comparisons, batch_nonce);
        auto gpu_end = Clock::now();
        timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_begin).count());
        
        // use a different port for packed partition to avoid conflict with:
        // sort:     start_port + tid
        // neighbor: start_port + 50
        const int packed_partition_port = start_port + 120;
        NetIO* io = new NetIO(party == ALICE ? nullptr : another_server_ip.c_str(), packed_partition_port);
        uint64_t and_rounds = 0;
        uint64_t opening_bytes_sent = 0;

        while (gpu_packed_compare_has_round(context)) {
            const uint64_t triple_count = gpu_packed_compare_next_triple_count(context);

            MYASSERT(triple_count > 0);
            std::vector<flora::preprocess::BooleanTripleRecord> sort_triples;

            requestDealerRecords(dealer_io, triple_count, sort_triples);
            MYASSERT(sort_triples.size() == triple_count);
            gpu_packed_compare_upload_triples(context, sort_triples.data(), triple_count);

            const void* send_buffer = nullptr;
            void* receive_buffer = nullptr;
            size_t opening_bytes = 0;

            const auto prepare_begin = Clock::now();
            gpu_packed_compare_prepare_round(context, &send_buffer, &receive_buffer, &opening_bytes);
            const auto prepare_end = Clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(prepare_end - prepare_begin).count());

            const auto comm_begin = Clock::now();
            io->flush();
            if (party == ALICE) {
                io->send_data_internal(static_cast<const char*>(send_buffer), opening_bytes);
                io->recv_data_internal(static_cast<char*>(receive_buffer), opening_bytes);
            } else {
                io->recv_data_internal(static_cast<char*>(receive_buffer), opening_bytes);
                io->send_data_internal(static_cast<const char*>(send_buffer), opening_bytes);
            }
            io->flush();

            const auto comm_end = Clock::now();
            timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(comm_end - comm_begin).count());

            const auto finish_begin = Clock::now();
            gpu_packed_compare_finish_round(context);
            const auto finish_end = Clock::now();

            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(finish_end - finish_begin).count());

            opening_bytes_sent += opening_bytes;
            ++and_rounds;
        }

        const uint64_t output_words = gpu_packed_compare_output_word_count(context);
        vector<uint32_t> local_gt_share(output_words);
        vector<uint32_t> peer_gt_share(output_words);
        gpu_packed_compare_download_gt_share(context, local_gt_share.data());

        // reveal the gt share to both parties, and then each party can compute the final gt result.
        auto result_comm_begin = Clock::now();
        const size_t result_bytes = output_words * sizeof(uint32_t);
        io->flush();

        if (party == ALICE) {
            io->send_data_internal(reinterpret_cast<const char*>(local_gt_share.data()), result_bytes);
            io->recv_data_internal(reinterpret_cast<char*>(peer_gt_share.data()), result_bytes);
        } else {
            io->recv_data_internal(reinterpret_cast<char*>(peer_gt_share.data()), result_bytes);
            io->send_data_internal(reinterpret_cast<const char*>(local_gt_share.data()), result_bytes);
        }
        io->flush();

        auto result_comm_end = Clock::now();
        timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(result_comm_end - result_comm_begin).count());
        delete io;
        auto scatter_begin = Clock::now();

        for (uint64_t comparison = 0; comparison < total_comparisons; ++comparison) {
            const uint64_t word = comparison / 32ULL;
            const uint32_t lane = static_cast<uint32_t>(comparison % 32ULL);
            const uint32_t opened_word = local_gt_share[word] ^ peer_gt_share[word];
            const bool larger =(opened_word >> lane) & 1u;
            const PackedPartitionReference& reference = references[comparison];
            PartitionResult& result = partition_res_per_thread[reference.thread_id][reference.partition_key];

            if (larger) {
                result.larger_index.push_back(reference.absolute_index);
            } else {
                // equal goes to smaller, and pivot itself also goes to smaller.
                result.smaller_index.push_back(reference.absolute_index);
            }
        }

        auto scatter_end = Clock::now();
        timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(scatter_end - scatter_begin).count());

        gpu_packed_compare_destroy(context);
        std::fprintf(stderr,
            "[GPU PACKED PARTITION][party=%d] sort-round=%llu comparisons=%llu packed-words=%llu and-rounds=%llu opening-sent=%llu bytes result-open=%llu bytes\n",
            party, static_cast<unsigned long long>(sort_round),
            static_cast<unsigned long long>(total_comparisons), static_cast<unsigned long long>(output_words), static_cast<unsigned long long>(and_rounds),
            static_cast<unsigned long long>(opening_bytes_sent), static_cast<unsigned long long>(result_bytes));
    }

    // obliviousPackedFullSort() is the main entry point for the oblivious packed full sort algorithm.
    // It takes a ServerSecretShares object, which contains the secret-shared tuples to be sorted, and performs the sorting in-place.
    // The function uses a recursive partitioning approach, where the tuples are partitioned based on a pivot value, and the partitions are sorted recursively until they contain only one element or are empty.
    void obliviousPackedFullSort(unique_ptr<ServerSecretShares>& ss, string another_server_ip, int party) {
        
        const uint64_t current_invocation = ++packed_sort_invocation;
        using Clock = std::chrono::high_resolution_clock;
        const uint64_t tuple_count = ss->size();

        NetIO* dealer_io = connectToSortDealer(party);
        if (tuple_count <= 1){
            finishDealerStream(dealer_io);
            delete dealer_io;
            dealer_io = nullptr;
            return;
        }
        // if (tuple_count <= 1) { return; }
        
        // partition_ranges use half-open intervals: [begin, end)
        vector<pair<uint64_t, uint64_t>> partition_ranges;

        partition_ranges.push_back(make_pair(0ULL, tuple_count));

        uint64_t sort_round = 0;
        uint64_t total_comparisons = 0;
        uint64_t total_partitions = 0;
        uint64_t worst_partition_size = 0;

        auto total_begin = Clock::now();

        while (!partition_ranges.empty()) {
            ++sort_round;

            vector<vector<PartitionTaskInfo>> tasks_per_bucket(sort_parallel_num);
            vector<map<pair<uint64_t, uint64_t>, PartitionResult>> results_per_bucket(sort_parallel_num);
            struct WavePartitionInfo {
                pair<uint64_t, uint64_t> range;
                uint64_t pivot_index;
                int bucket;
            };
            vector<WavePartitionInfo> wave_infos;
            wave_infos.reserve(partition_ranges.size());
            vector<uint64_t> bucket_load(sort_parallel_num, 0ULL);

            uint64_t round_comparisons = 0;
            uint64_t round_max_partition = 0;
            auto prepare_begin = Clock::now();

            for (const auto& range : partition_ranges) {
                const uint64_t begin_index = range.first;
                const uint64_t end_index = range.second;
                MYASSERT(begin_index < end_index);
                const uint64_t element_count = end_index - begin_index;

                if (element_count <= 1) { continue; }
                const uint64_t pivot_index = choosePackedPivotIndex(begin_index, end_index, sort_round);
                MYASSERT(pivot_index >= begin_index);
                MYASSERT(pivot_index < end_index);

                // partition results are distributed to multiple buckets to balance the load across threads.
                int target_bucket = 0;
                for (int bucket = 1; bucket < sort_parallel_num; ++bucket) {
                    if (bucket_load[bucket] < bucket_load[target_bucket]) {
                        target_bucket = bucket;
                    }
                }

                const SecretShareTuple& pivot_tuple = *ss->svec[pivot_index];
                PartitionTaskInfo task {
                    .pivot_val_share = pivot_tuple.getNodeValShare(),
                    .pivot_end_node_share = pivot_tuple.getEndNodeShare(),
                    .begin_index = begin_index,
                    .end_index = end_index,
                    .partition_begin = begin_index,
                    .partition_end = end_index,
                    .run_partition = true,
                    .pivot_index = pivot_index,
                    .pivot_tie_breaker_share = pivot_tuple.get_tie_breaker_id(),
                };
                tasks_per_bucket[target_bucket].push_back(task);

                wave_infos.push_back(WavePartitionInfo{
                        .range = range,
                        .pivot_index = pivot_index,
                        .bucket = target_bucket,
                    });
                bucket_load[target_bucket] += element_count;
                round_comparisons += element_count;
                round_max_partition = std::max(round_max_partition, element_count);
            }

            auto prepare_end = Clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(prepare_end - prepare_begin).count());

            if (wave_infos.empty()) { break; }
            total_comparisons += round_comparisons;
            total_partitions += wave_infos.size();
            worst_partition_size = std::max(worst_partition_size, round_max_partition);
            
            // one time packed GPU GMW batch for all element-vs-pivot comparisons in the current recursion level.
            packedGpuPartitionWave(ss, tasks_per_bucket, results_per_bucket, another_server_ip, party, sort_round, current_invocation, dealer_io);

            auto reorder_begin = Clock::now();
            vector<pair<uint64_t, uint64_t>> next_partition_ranges;
            next_partition_ranges.reserve(wave_infos.size() * 2ULL);
            
            // all current partition ranges are non-overlapping, so we can reorder them sequentially.
            for (const auto& info : wave_infos) {
                const uint64_t begin_index = info.range.first;
                const uint64_t end_index = info.range.second;
                const pair<uint64_t, uint64_t> partition_key = info.range;

                auto result_it = results_per_bucket[info.bucket].find(partition_key);
                MYASSERT(result_it != results_per_bucket[info.bucket].end());
                PartitionResult& result = result_it->second;
                MYASSERT(result.smaller_index.size() + result.larger_index.size() == end_index - begin_index);

                // The pivot index must be in the smaller partition because the comparator used is strictly greater than (key > pivot). Therefore, the pivot itself will always be included in the smaller partition.
                const auto pivot_it = std::find(result.smaller_index.begin(), result.smaller_index.end(), info.pivot_index);
                MYASSERT(pivot_it != result.smaller_index.end());

                // the output of resortByPartition() includes smaller_without_pivot, pivot, larger
                // so the final position of pivot: begin + smaller_count - 1
                const uint64_t smaller_count = result.smaller_index.size();
                MYASSERT(smaller_count >= 1);
                const uint64_t pivot_final_index = begin_index + smaller_count - 1ULL;

                ss->resortByPartition(result.smaller_index, result.larger_index, begin_index, end_index, info.pivot_index);

                // [begin, pivot_final)
                if (pivot_final_index > begin_index) {
                    const uint64_t left_size = pivot_final_index - begin_index;
                    if (left_size > 1) {
                        next_partition_ranges.push_back(make_pair(begin_index, pivot_final_index));
                    }
                }

                // [pivot_final + 1, end)
                const uint64_t right_begin = pivot_final_index + 1ULL;
                if (right_begin < end_index) {
                    const uint64_t right_size = end_index - right_begin;
                    if (right_size > 1) {
                        next_partition_ranges.push_back(make_pair(right_begin, end_index));
                    }
                }
            }
            partition_ranges.swap(next_partition_ranges);

            auto reorder_end = Clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(reorder_end - reorder_begin).count());

            std::fprintf(stderr,
                "[GPU PACKED FULL SORT][party=%d] round=%llu partitions=%llu comparisons=%llu max-partition=%llu next-partitions=%llu\n",
                party, static_cast<unsigned long long>(sort_round),  static_cast<unsigned long long>(wave_infos.size()),
                static_cast<unsigned long long>(round_comparisons), static_cast<unsigned long long>(round_max_partition),
                static_cast<unsigned long long>(partition_ranges.size()));
            
            if (sort_round > 256ULL) {
                std::fprintf(stderr, "[GPU PACKED FULL SORT] excessive recursion depth; aborting\n");
                std::abort();
            }
        }

        finishDealerStream(dealer_io);
        delete dealer_io;
        dealer_io = nullptr;

        auto total_end =  Clock::now();
        const double total_seconds = std::chrono::duration<double>(total_end - total_begin).count();
        std::fprintf(stderr,
            "[GPU PACKED FULL SORT][party=%d] DONE\n  tuples=%llu\n  waves=%llu\n  partitions=%llu\n  comparisons=%llu\n  initial-max-partition=%llu\n  wall=%.6f s\n",
            party, static_cast<unsigned long long>(tuple_count), static_cast<unsigned long long>(sort_round), static_cast<unsigned long long>(total_partitions),
            static_cast<unsigned long long>(total_comparisons), static_cast<unsigned long long>(worst_partition_size), total_seconds);
    }

    void finishEmptyFilterDealerStream(int filter_thread_id, int party) {

        NetIO* dealer_io = connectToFilterDealer(party, filter_thread_id);
        finishDealerStream(dealer_io);
        delete dealer_io;
    }

    void filterThread(int filter_thread_id, int port, string another_server_ip, int party,
                      unique_ptr<ServerSecretShares>& ss, uint64_t begin_index,
                      uint64_t end_index, vector<bool>& is_valid, vector<bool>& is_cycle,
                      uint64_t& out_bytes_sent, uint64_t& out_bytes_recv, uint64_t& out_rounds) {
        
        auto t_ext_start = std::chrono::high_resolution_clock::now();
        vector<uint32_t> path_shares;
        ss->extractSinglePathFromRange(path_shares, begin_index, end_index);
        auto t_ext_done = std::chrono::high_resolution_clock::now();
        timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_ext_done - t_ext_start).count());
        
        TwoPcEngine engine(party, another_server_ip, port, bit_len, this->tie_breaker_bit);
        engine.setTimer(&this->timer); 
        engine.initNetIOOnly();
        engine.resetCommStats();  
        NetIO* dealer_io = connectToFilterDealer(party, filter_thread_id);
        engine.setDealerIO(dealer_io); 

        is_valid = vector<bool>((end_index - begin_index), false);
        is_cycle = vector<bool>((end_index - begin_index), false);
        if (this->filter_batch_num == 0 || ((end_index - begin_index) <= this->filter_batch_num)) {
            engine.obliviousFilter(path_shares, 0, end_index - begin_index, this->detect_length_k, is_valid, is_cycle);
        } else {
            uint64_t begin_tuple_index = 0;
            uint64_t total_tuple_num = end_index - begin_index;
            while (begin_tuple_index < total_tuple_num) {
                uint64_t remained_tuple_num = total_tuple_num - begin_tuple_index;
                uint64_t processed_tuple_num = min(this->filter_batch_num, remained_tuple_num);
                engine.obliviousFilter(path_shares, begin_tuple_index, processed_tuple_num, detect_length_k, is_valid, is_cycle);
                begin_tuple_index += processed_tuple_num;
            }
            MYASSERT(begin_tuple_index == total_tuple_num);
        }

        out_bytes_sent = engine.getBytesSent();
        out_bytes_recv = engine.getBytesRecv();
        out_rounds     = engine.getRounds();

        engine.finishDealerStream();
    }

    void obliviousFilter(unique_ptr<ServerSecretShares>& ss,
                        string another_server_ip, int party,
                        vector<bool>& is_valid, vector<bool>& is_cycle) {
        MYASSERT(is_valid.empty());
        MYASSERT(is_cycle.empty());
        uint64_t total_tuple_num = ss->size();

        const int gpu_filter_parallel_num = GPU_FILTER_STREAMS;

        uint64_t tuple_per_elem = total_tuple_num / gpu_filter_parallel_num;
        vector<vector<bool>> is_valid_per_thread(gpu_filter_parallel_num);
        vector<vector<bool>> is_cycle_per_thread(gpu_filter_parallel_num);
        vector<uint64_t> thr_bytes_sent(gpu_filter_parallel_num, 0);
        vector<uint64_t> thr_bytes_recv(gpu_filter_parallel_num, 0);
        vector<uint64_t> thr_rounds(gpu_filter_parallel_num, 0);
        vector<thread> t_vec;

        cout << "[S1/S2] obliviousFilter START, total="
             << ss->size() << " party=" << party << endl << flush;

        for (int i = 0; i < gpu_filter_parallel_num; ++i) {
            uint64_t begin_index = static_cast<uint64_t>(i) * tuple_per_elem;
            uint64_t end_index = static_cast<uint64_t>(i + 1) * tuple_per_elem;
            if (begin_index == total_tuple_num){
                thread t(&NetServer::finishEmptyFilterDealerStream, this, i, party);
                t_vec.push_back(std::move(t));
                continue;
            }

            if ((i == gpu_filter_parallel_num - 1) || (end_index >= total_tuple_num))
                end_index = total_tuple_num;
            thread t(&NetServer::filterThread, this, i, start_port + i,
                     another_server_ip, party, ref(ss), begin_index, end_index,
                     ref(is_valid_per_thread[i]), ref(is_cycle_per_thread[i]),
                     ref(thr_bytes_sent[i]), ref(thr_bytes_recv[i]),
                     ref(thr_rounds[i]));
            t_vec.push_back(move(t));
        }
        for (auto& t : t_vec) t.join();

        auto t_merge_start = std::chrono::high_resolution_clock::now();
        uint64_t total_sent = 0, total_recv = 0;
        for (int i = 0; i < gpu_filter_parallel_num; i++) {
            total_sent += thr_bytes_sent[i];
            total_recv += thr_bytes_recv[i];
        }
        cout << "[COMM][filter] sent: " << total_sent << " B | recv: " << total_recv << " B" << endl;

        uint64_t cur_ptr = 0;
        is_valid = vector<bool>(total_tuple_num);
        is_cycle = vector<bool>(total_tuple_num);
        for (int i = 0; i < gpu_filter_parallel_num; i++) {
            MYASSERT(is_valid_per_thread[i].size() == is_cycle_per_thread[i].size());
            copy(is_valid_per_thread[i].begin(), is_valid_per_thread[i].end(), is_valid.begin() + cur_ptr);
            copy(is_cycle_per_thread[i].begin(), is_cycle_per_thread[i].end(), is_cycle.begin() + cur_ptr);
            cur_ptr += is_valid_per_thread[i].size();
        }
        MYASSERT(cur_ptr == total_tuple_num);
        auto t_merge_done = std::chrono::high_resolution_clock::now();
        timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_merge_done - t_merge_start).count());
    }

    void extractPathThread(int port, string another_server_ip, int party,
                           unique_ptr<ServerSecretShares>& ss,
                           uint64_t begin_index, uint64_t end_index,
                           vector<uint64_t>& edge_tuple_indices) {
        auto t_ext_start = std::chrono::high_resolution_clock::now();
        vector<uint32_t> node_shares;
        ss->extractNodeIdSharesOnly(node_shares, begin_index, end_index);
        auto t_ext_done = std::chrono::high_resolution_clock::now();
        timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_ext_done - t_ext_start).count());

        uint64_t total_tuple_num = end_index - begin_index;

        TwoPcEngine engine(party, another_server_ip, port, bit_len, this->tie_breaker_bit);
        engine.setTimer(&this->timer); 
        engine.initEngine();
        if (this->extract_batch_num == 0 || this->extract_batch_num >= total_tuple_num) {
            engine.obliviousExtractPath(node_shares, edge_tuple_indices,
                                        begin_index, end_index, 0, total_tuple_num);
        } else {
            uint64_t curr_begin_index = 0;
            while (curr_begin_index < total_tuple_num) {
                uint64_t remained_processed_num = total_tuple_num - curr_begin_index;
                uint64_t processed_num = min(remained_processed_num, this->extract_batch_num);
                engine.obliviousExtractPath(node_shares, edge_tuple_indices,
                                            begin_index, end_index,
                                            curr_begin_index, processed_num);
                curr_begin_index += processed_num;
            }
        }
        engine.terminateEngine();
    }

    void obliviousExtractPathAndEdges(unique_ptr<ServerSecretShares>& ss,
                                      string another_server_ip, int party,
                                      unique_ptr<ServerSecretShares>& edges,
                                      unique_ptr<ServerSecretShares>& paths) {
        MYASSERT(edges->size() == 0);
        MYASSERT(paths->size() == 0);
        uint64_t total_tuple_num = ss->size();
        uint64_t tuple_per_elem = total_tuple_num / this->extract_path_parallel_num;
        vector<vector<uint64_t>> edge_indices_per_thread(this->extract_path_parallel_num);
        vector<thread> t_vec;
        for (int i = 0; i < this->extract_path_parallel_num; i++) {
            uint64_t begin_index = static_cast<uint64_t>(i) * tuple_per_elem;
            uint64_t end_index = static_cast<uint64_t>(i + 1) * tuple_per_elem;
            if (begin_index >= total_tuple_num) {
                break;
            }
            if ((i == this->extract_path_parallel_num - 1) || (end_index >= total_tuple_num)) {
                end_index = total_tuple_num;
            }
            thread t(&NetServer::extractPathThread, this, start_port + i,
                     another_server_ip, party, ref(ss), begin_index, end_index,
                     ref(edge_indices_per_thread[i]));
            t_vec.push_back(move(t));
        }
        for (auto& t : t_vec) {
            t.join();
        }

        auto t_write_start = std::chrono::high_resolution_clock::now();
        uint64_t cur_thread_id = 0;
        while (edge_indices_per_thread[cur_thread_id].empty()) {
            cur_thread_id++;
            MYASSERT(cur_thread_id < static_cast<uint64_t>(this->extract_path_parallel_num));
        }
        uint64_t cur_edge_index = 0;

        bool all_edges_filled = false;
        for (uint64_t i = 0; i < total_tuple_num; i++) {
            if (!all_edges_filled && (i == edge_indices_per_thread[cur_thread_id][cur_edge_index])) {
                edges->insert(ss->svec[i]);
                cur_edge_index++;
                if (cur_edge_index >= edge_indices_per_thread[cur_thread_id].size()) {
                    cur_thread_id++;
                    if (cur_thread_id == static_cast<uint64_t>(this->extract_path_parallel_num)) {
                        all_edges_filled = true;
                        continue;
                    }
                    while (edge_indices_per_thread[cur_thread_id].empty()) {
                        cur_thread_id++;
                        if (cur_thread_id == static_cast<uint64_t>(this->extract_path_parallel_num)) {
                            all_edges_filled = true;
                            continue;
                        }
                    }
                    cur_edge_index = 0;
                }
            } else {
                paths->insert(ss->svec[i]);
            }
        }
        ss->clear();
        auto t_write_done = std::chrono::high_resolution_clock::now();
        timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_write_done - t_write_start).count());
    }

    void listenThread(string ip_addr, int* client_fd) {
        int listen_fd = socket(PF_INET, SOCK_STREAM, 0);
        int sock_opt = 1;
        MYASSERT(setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &sock_opt, sizeof(sock_opt)) >= 0);
        struct sockaddr_in servaddr = string_to_struct_addr(ip_addr);
        if (bind(listen_fd, (sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
            fprintf(stderr, "bind failed: %s\n", strerror(errno));
        }
        MYASSERT(listen(listen_fd, 100) >= 0);
        cout << "start listening on " << ip_addr << endl;
        struct sockaddr_in clientaddr;
        socklen_t clientaddrlen = sizeof(clientaddr);
        int fd = accept(listen_fd, (struct sockaddr*)&clientaddr, &clientaddrlen);
        MYASSERT(fd > 0);
        *client_fd = fd;
        int flag = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        int buf_sz = 16 * 1024 * 1024;
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf_sz, sizeof(buf_sz));
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf_sz, sizeof(buf_sz));

        string recv_msg;
        MYASSERT(recvMsg(*client_fd, recv_msg));
        close(listen_fd);
    }

    void connectThread(string ip, int* target_fd) {
        sleep(15);
        cout << "try to connect to " << ip << endl;
        *target_fd = connect_to_addr(ip);
        MYASSERT(*target_fd > 0);

        int flag = 1;
        setsockopt(*target_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        int buf_sz = 16 * 1024 * 1024;
        setsockopt(*target_fd, SOL_SOCKET, SO_SNDBUF, &buf_sz, sizeof(buf_sz));
        setsockopt(*target_fd, SOL_SOCKET, SO_RCVBUF, &buf_sz, sizeof(buf_sz));

        MYASSERT(sendMsg(*target_fd, ip));
    }

    void connectWithOtherServers() {
        if (server_id == SERVER_ID::S1) {
            thread t1(&NetServer::listenThread, this, ip_1, &fd_2);
            thread t2(&NetServer::connectThread, this, ip_3, &fd_3);
            t1.join();
            t2.join();
        } else if (server_id == SERVER_ID::S2) {
            thread t1(&NetServer::listenThread, this, ip_2, &fd_3);
            thread t2(&NetServer::connectThread, this, ip_1, &fd_1);
            t1.join();
            t2.join();
        } else if (server_id == SERVER_ID::S3) {
            thread t1(&NetServer::listenThread, this, ip_3, &fd_1);
            thread t2(&NetServer::connectThread, this, ip_2, &fd_2);
            t1.join();
            t2.join();
        } else {
            MYASSERT(0);
        }
    }

    void closeConnectionWithOtherServers() {
        if (server_id == SERVER_ID::S1) {
            close(fd_2);
            close(fd_3);
        } else if (server_id == SERVER_ID::S2) {
            close(fd_3);
            close(fd_1);
        } else if (server_id == SERVER_ID::S3) {
            close(fd_1);
            close(fd_2);
        } else {
            MYASSERT(0);
        }
    }

    void validateFds() {
        if (server_id == SERVER_ID::S1) {
            MYASSERT(fd_2 > 0);
            MYASSERT(fd_3 > 0);
        } else if (server_id == SERVER_ID::S2) {
            MYASSERT(fd_1 > 0);
            MYASSERT(fd_3 > 0);
        } else if (server_id == SERVER_ID::S3) {
            MYASSERT(fd_1 > 0);
            MYASSERT(fd_2 > 0);
        } else {
            MYASSERT(0);
        }
    }

    void incDetectLength() {
        this->detect_length_k++;
        compute_server->incDetectLength();
    }
    
    void recvServerSecretShareToPtrWorker(ServerSecretShares* ss, int fd,
                                          int curr_max_degree,
                                          int curr_detect_length_k) {
        MYASSERT(ss->size() == 0);
        string serialized_str;
        auto t0 = std::chrono::high_resolution_clock::now();
        bool success = recvMsg(fd, serialized_str);
        auto t1 = std::chrono::high_resolution_clock::now();
        MYASSERT(success);
        ss->fillServerSecretSharesFromStr(serialized_str, curr_max_degree,
                                          curr_detect_length_k,
                                          this->local_compute_parallel_num);
        auto t2 = std::chrono::high_resolution_clock::now();
        MYASSERT(ss->size() != 0);

        timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        timer.add_serial(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    }

    void reconCurrentSS(int curr_max_degree, int curr_detect_length_k) {
        if (server_id == SERVER_ID::S1) {
            unique_ptr<ServerSecretShares> s2_1 = make_unique<ServerSecretShares>();
            unique_ptr<ServerSecretShares> s2_2 = make_unique<ServerSecretShares>();
            unique_ptr<ServerSecretShares> s3_1 = make_unique<ServerSecretShares>();
            unique_ptr<ServerSecretShares> s3_2 = make_unique<ServerSecretShares>();

            thread recvS2_1(&NetServer::recvServerSecretShareWorker, this, ref(s2_1), fd_2, curr_max_degree, curr_detect_length_k);
            thread recvS3_1(&NetServer::recvServerSecretShareWorker, this, ref(s3_1), fd_3, curr_max_degree, curr_detect_length_k);
            recvS2_1.join();
            recvS3_1.join();

            thread recvS2_2(&NetServer::recvServerSecretShareWorker, this, ref(s2_2), fd_2, curr_max_degree, curr_detect_length_k);
            thread recvS3_2(&NetServer::recvServerSecretShareWorker, this, ref(s3_2), fd_3, curr_max_degree, curr_detect_length_k);
            recvS2_2.join();
            recvS3_2.join();

            MYASSERT(*getFirstServerSecretShares() == *s3_2.get());
            MYASSERT(*s2_1.get() == *getSecondServerSecretShares());
            MYASSERT(*s2_2.get() == *s3_1.get());

            vector<ServerSecretShares*> recon_vec = {
                getFirstServerSecretShares(), getSecondServerSecretShares(), s3_1.get()};

            SecretShareTupleHelper helper(this->id_range, this->tie_breaker_range);
            unique_ptr<ServerSecretShares> recon_ss = helper.reconServerShares(recon_vec, curr_max_degree, curr_detect_length_k);
            recon_ss->print();
        } else if (server_id == SERVER_ID::S2 || server_id == SERVER_ID::S3) {
            thread sendS1(&NetServer::sendServerSecretShareWorker, this, getFirstServerSecretShares(), fd_1);
            sendS1.join();
            thread sendS2(&NetServer::sendServerSecretShareWorker, this, getSecondServerSecretShares(), fd_1);
            sendS2.join();
        } else {
            MYASSERT(0);
        }
    }

    void reconWithTwoPCShare(unique_ptr<ServerSecretShares>& ss) {
        if (server_id == SERVER_ID::S1) {
            unique_ptr<ServerSecretShares> s2_ss = make_unique<ServerSecretShares>();
            thread recvSS(&NetServer::recvServerSecretShareWorker, this, ref(s2_ss), fd_2, this->max_degree, this->detect_length_k);
            recvSS.join();
            vector<ServerSecretShares*> xor_vec = {ss.get(), s2_ss.get()};
            unique_ptr<ServerSecretShares> recon_ss = compute_server->xorSecretShareTupleAndCreateDst(xor_vec);
            recon_ss->print();
            return;
        } else if (server_id == SERVER_ID::S2) {
            thread sendSS(&NetServer::sendServerSecretShareWorker, this, ss.get(), fd_1);
            sendSS.join();
            return;
        } else if (server_id == SERVER_ID::S3) {
            return;
        } else {
            MYASSERT(0);
        }
    }

    void waitForOtherServers() {
        string msg = "ack";
        string recv_msg1, recv_msg2;
        if (server_id == SERVER_ID::S1) {
            sendMsg(this->fd_2, msg);
            sendMsg(this->fd_3, msg);
            recvMsg(this->fd_2, recv_msg1);
            recvMsg(this->fd_3, recv_msg2);
        } else if (server_id == SERVER_ID::S2) {
            sendMsg(this->fd_1, msg);
            sendMsg(this->fd_3, msg);
            recvMsg(this->fd_1, recv_msg1);
            recvMsg(this->fd_3, recv_msg2);
        } else if (server_id == SERVER_ID::S3) {
            sendMsg(this->fd_1, msg);
            sendMsg(this->fd_2, msg);
            recvMsg(this->fd_1, recv_msg1);
            recvMsg(this->fd_2, recv_msg2);
        } else {
            MYASSERT(0);
        }
    }

    void shuffleWrapper(int curr_max_degree, int curr_detect_length_k) {
        using Clock = std::chrono::high_resolution_clock;
        if (this->server_id == SERVER_ID::S1) {
            auto compute_begin = Clock::now();

            // x1 = permute12(A XOR B XOR z12), x2 = permute31(x1 XOR z31), A  = A_hat, B  = B_hat
            unique_ptr<ServerSecretShares> x2 = gpu_shuffle_protocol->s1Step1(
                    *getFirstServerSecretShares(), *getSecondServerSecretShares(),
                    seed12_permute, seed31_permute, seedA, seedB);

            auto compute_end = Clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_begin).count());
            
            // CPU communication and serieal
            thread sendX2(&NetServer::sendServerSecretShareWorker, this, x2.get(), fd_2);
            sendX2.join();

            auto sync_begin = Clock::now();
            waitForOtherServers();
            auto sync_end = Clock::now();
            timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(sync_end - sync_begin).count());

            return;
        }

        if (this->server_id == SERVER_ID::S2) {
            auto compute_begin = Clock::now();

            //y1 = permute12(C XOR z12)
            unique_ptr<ServerSecretShares> y1 = gpu_shuffle_protocol->s2Step1(*getSecondServerSecretShares(), seed12_permute);
            auto compute_end = Clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_begin).count());
            unique_ptr<ServerSecretShares> x2 = make_unique<ServerSecretShares>();

            thread sendY1(&NetServer::sendServerSecretShareWorker, this, y1.get(), fd_3);
            thread recvX2(&NetServer::recvServerSecretShareWorker, this, ref(x2), fd_1, curr_max_degree, curr_detect_length_k);
            sendY1.join();
            recvX2.join();

            compute_begin = Clock::now();

            // x3 = permute23(x2 XOR z23), B  = B_hat, c1 = x3 XOR B_hat
            unique_ptr<ServerSecretShares> c1 = gpu_shuffle_protocol->s2Step2(*getFirstServerSecretShares(), *x2, seed23_permute, seedB);
            compute_end = Clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_begin).count());
            unique_ptr<ServerSecretShares> c2 = make_unique<ServerSecretShares>();

            thread sendC1(&NetServer::sendServerSecretShareWorker, this, c1.get(), fd_3);
            thread recvC2(&NetServer::recvServerSecretShareWorker, this, ref(c2), fd_3, curr_max_degree, curr_detect_length_k);
            sendC1.join();
            recvC2.join();

            compute_begin = Clock::now();

            // second = c1 XOR c2
            gpu_shuffle_protocol->s2Step3(*getSecondServerSecretShares(), *c1, *c2);
            compute_end = Clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_begin).count());

            auto sync_begin = Clock::now();
            waitForOtherServers();
            auto sync_end = Clock::now();
            timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(sync_end - sync_begin).count());

            return;
        }

        if (this->server_id == SERVER_ID::S3) {
            unique_ptr<ServerSecretShares> y1 = make_unique<ServerSecretShares>();
            thread recvY1(&NetServer::recvServerSecretShareWorker, this, ref(y1), fd_2, curr_max_degree, curr_detect_length_k);
            recvY1.join();
            auto compute_begin = Clock::now();

            // y2 = permute31(y1 XOR z31), y3 = permute23(y2 XOR z23), A  = A_hat, c2 = y3 XOR A_hat
            unique_ptr<ServerSecretShares> c2 = gpu_shuffle_protocol->s3Step1(*getSecondServerSecretShares(), *y1, seed31_permute, seed23_permute, seedA);
            auto compute_end = Clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_begin).count());
            unique_ptr<ServerSecretShares> c1 = make_unique<ServerSecretShares>();

            thread sendC2(&NetServer::sendServerSecretShareWorker, this, c2.get(), fd_2);
            thread recvC1(&NetServer::recvServerSecretShareWorker, this, ref(c1), fd_2, curr_max_degree, curr_detect_length_k);
            sendC2.join();
            recvC1.join();

            compute_begin = Clock::now();

            // first = c1 XOR c2
            gpu_shuffle_protocol->s3Step2(*getFirstServerSecretShares(), *c1, *c2);
            compute_end = Clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_begin).count());

            auto sync_begin = Clock::now();
            waitForOtherServers();
            auto sync_end = Clock::now();
            timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(sync_end - sync_begin).count());

            return;
        }
        MYASSERT(0);
    }

    unique_ptr<ServerSecretShares> computeTwoPCShare() {

        if (server_id == SERVER_ID::S1) {
            // S1: first XOR second
            return gpu_shuffle_protocol-> collapseS1ToTwoPc(*getFirstServerSecretShares(), *getSecondServerSecretShares());
        }

        if (server_id == SERVER_ID::S2) {
            // S2: second
            auto ss = make_unique<ServerSecretShares>();
            for (uint64_t index = 0; index < getSecondServerSecretShares()->size(); ++index) {
                auto tuple = make_unique<SecretShareTuple>(*getSecondServerSecretShares()-> svec[index]);
                ss->insert(tuple);
            }
            return ss;
        }

        if (server_id == SERVER_ID::S3) {
            return make_unique<ServerSecretShares>();
        }
        MYASSERT(0);
        return nullptr;
    }

    void sortWrapper(unique_ptr<ServerSecretShares>& ss) {
        if (server_id == SERVER_ID::S1) {
            this->obliviousPackedFullSort(ss,ip_no_port_1,ALICE);
            return;
        }
        if (server_id == SERVER_ID::S2) {
            this->obliviousPackedFullSort(ss,ip_no_port_1,BOB);
            return;
        }
        if (server_id == SERVER_ID::S3) {
            serveDealerStream(sortDealerPortForS0(), sortDealerPortForS1(), FloraDealerDomain::Sort);
            return;
        }

        MYASSERT(0);
    }

    void neighborPassWrapper(unique_ptr<ServerSecretShares>& ss) {
        if (server_id == SERVER_ID::S1) {
            this->obliviousPrefixNeighborPass(ss, ip_no_port_1, ALICE);
            return;
        } else if (server_id == SERVER_ID::S2) {
            this->obliviousPrefixNeighborPass(ss, ip_no_port_1, BOB);
            return;
        } else if (server_id == SERVER_ID::S3) {
            serveDealerStream(neighborDealerPortForS0(), neighborDealerPortForS1(), FloraDealerDomain::Neighbor);
            return;
        } else {
            MYASSERT(0);
        }
    }

    void extractWrapper(unique_ptr<ServerSecretShares>& ss, unique_ptr<ServerSecretShares>& edge_shares, unique_ptr<ServerSecretShares>& path_shares) {
        if (server_id == SERVER_ID::S1) {
            this->obliviousExtractPathAndEdges(ss, ip_no_port_1, ALICE, edge_shares, path_shares);
            return;
        } else if (server_id == SERVER_ID::S2) {
            this->obliviousExtractPathAndEdges(ss, ip_no_port_1, BOB, edge_shares, path_shares);
            return;
        } else if (server_id == SERVER_ID::S3) {
            return;
        } else {
            MYASSERT(0);
        }
    }

    void parsePathsWrapper(unique_ptr<ServerSecretShares>& path_shares, unique_ptr<ServerSecretShares>& parsed_path_shares) {
        if ((server_id == SERVER_ID::S1) || (server_id == SERVER_ID::S2)) {
            path_shares->extractPathTuples(parsed_path_shares);
            return;
        } else if (server_id == SERVER_ID::S3) {
            return;
        } else {
            MYASSERT(0);
        }
    }
    
    void moveFilteredPathShares(unique_ptr<ServerSecretShares>& ss, unique_ptr<ServerSecretShares>& cycle_shares, vector<bool>& is_valid, vector<bool>& is_cycle) {
        uint64_t tuple_num = ss->size();
        MYASSERT(is_valid.size() == tuple_num);
        MYASSERT(is_cycle.size() == tuple_num);
        unique_ptr<ServerSecretShares> filtered_ss = make_unique<ServerSecretShares>();
        MYASSERT(cycle_shares->size() == 0);
        for (uint64_t i = 0; i < tuple_num; i++) {
            if (is_valid[i] && is_cycle[i]) {
                cycle_shares->insert(ss->svec[i]);
            } else if (is_valid[i] && !is_cycle[i]) {
                filtered_ss->insert(ss->svec[i]);
            }
        }
        ss = move(filtered_ss);
    }

    void filterWrapper(unique_ptr<ServerSecretShares>& ss, unique_ptr<ServerSecretShares>& cycle_shares) {
        vector<bool> is_valid, is_cycle;
        if (server_id == SERVER_ID::S1) {
            this->obliviousFilter(ss, ip_no_port_1, ALICE, is_valid, is_cycle);
            moveFilteredPathShares(ss, cycle_shares, is_valid, is_cycle);
            return;
        } else if (server_id == SERVER_ID::S2) {
            this->obliviousFilter(ss, ip_no_port_1, BOB, is_valid, is_cycle);
            moveFilteredPathShares(ss, cycle_shares, is_valid, is_cycle);
            return;
        } else if (server_id == SERVER_ID::S3) {
            const int gpu_filter_parallel_num = GPU_FILTER_STREAMS;
            std::vector<std::thread> dealer_threads;
            dealer_threads.reserve(gpu_filter_parallel_num);

            for (int thread_id = 0; thread_id < GPU_FILTER_STREAMS; ++thread_id) {
                dealer_threads.emplace_back(&NetServer::serveDealerStream,
                    this, filterDealerPortForS0(thread_id), filterDealerPortForS1(thread_id), FloraDealerDomain::Filter);
            }
            for (auto& thread : dealer_threads) {
                thread.join();
            }
            cout << "[S3][Filter Dealer] all streams finished" << endl << flush;
            return;
        } else {
            MYASSERT(0);
        }
    }

    void padPathsAndPristineEdges(unique_ptr<ServerSecretShares>& path_shares) {
        for (uint64_t i = 0; i < path_shares->size(); i++) {
            path_shares->svec[i]->padPathTuple(this->max_degree, this->detect_length_k);
        }

        for (uint64_t i = 0; i < pristine_edges_first->size(); i++) {
            pristine_edges_first->svec[i]->padEdgeTuple(this->max_degree, this->detect_length_k);
            pristine_edges_second->svec[i]->padEdgeTuple(this->max_degree, this->detect_length_k);
        }

        unique_ptr<ServerSecretShares> reloaded_edges;
        if (server_id == SERVER_ID::S1) {
            unique_ptr<ServerSecretShares> copy_first = make_unique<ServerSecretShares>();
            unique_ptr<ServerSecretShares> copy_second = make_unique<ServerSecretShares>();
            copy_first->copyFromServerSecretShares(pristine_edges_first);
            copy_second->copyFromServerSecretShares(pristine_edges_second);

            vector<ServerSecretShares*> xor_vec = {copy_first.get(), copy_second.get()};
            reloaded_edges = compute_server->xorSecretShareTupleAndCreateDst(xor_vec);
        } 
        else if (server_id == SERVER_ID::S2) {
            reloaded_edges = make_unique<ServerSecretShares>();
            reloaded_edges->copyFromServerSecretShares(pristine_edges_second);
        } 
        else {
            reloaded_edges = make_unique<ServerSecretShares>();
        }

        uint64_t expected_total = path_shares->size() + reloaded_edges->size();
        for (uint64_t i = 0; i < reloaded_edges->size(); i++) {
            path_shares->insert(reloaded_edges->svec[i]);
        }
        MYASSERT(path_shares->size() == expected_total);
    }

    void reassignSecretShares(unique_ptr<ServerSecretShares>& ss) {

        using Clock = std::chrono::high_resolution_clock;

        if (server_id == SERVER_ID::S1) {
            ServerSecretShares* first = getFirstServerSecretShares();
            ServerSecretShares* second = getSecondServerSecretShares();

            int reassign_seed = this->reassign_randomness_seed + this->detect_length_k + this->max_degree;
            auto compute_begin = Clock::now();

            // second = random(seed), first = two_pc XOR second
            gpu_shuffle_protocol-> expandS1TwoPcToThreePc(*ss, *first, *second, reassign_seed);
            auto compute_end = Clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_begin).count());
            
            // S1 -> S3: first
            // S1 -> S2: second
            thread sendFirst(&NetServer::sendServerSecretShareWorker, this, first, fd_3);
            thread sendSecond(&NetServer::sendServerSecretShareWorker, this, second, fd_2);
            sendFirst.join();
            sendSecond.join();

            return;
        }
        if (server_id == SERVER_ID::S2) {
            ServerSecretShares* third = getSecondServerSecretShares();
            third->moveServerShares(ss.get());
            thread sendThird(&NetServer::sendServerSecretShareWorker, this, third, fd_3);
            ServerSecretShares* second = getFirstServerSecretShares();
            second->clear();
            thread recvSecond(&NetServer::recvServerSecretShareToPtrWorker, this, second, fd_1, this->max_degree, this->detect_length_k);
            sendThird.join();
            recvSecond.join();

            return;
        }
        if (server_id == SERVER_ID::S3) {
            ServerSecretShares* third = getFirstServerSecretShares();
            ServerSecretShares* first = getSecondServerSecretShares();
            third->clear();
            first->clear();
            thread recvFirst(&NetServer::recvServerSecretShareToPtrWorker, this, first, fd_1, this->max_degree, this->detect_length_k);
            thread recvThird(&NetServer::recvServerSecretShareToPtrWorker, this, third, fd_2, this->max_degree, this->detect_length_k);
            recvFirst.join();
            recvThird.join();

            return;
        }
        MYASSERT(0);
    }

    void sendServerSecretShareWorker(ServerSecretShares* ss, int fd) {
        auto t0 = std::chrono::high_resolution_clock::now();
        string serialized_str = ss->serializeParallel(this->local_compute_parallel_num);
        auto t1 = std::chrono::high_resolution_clock::now();
        bool success = sendMsg(fd, serialized_str);
        auto t2 = std::chrono::high_resolution_clock::now();
        MYASSERT(success);

        timer.add_serial(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    }

    void recvServerSecretShareWorker(unique_ptr<ServerSecretShares>& ss, int fd,
                                    int curr_max_degree, int curr_detect_length_k) {
        MYASSERT(ss->size() == 0);
        string serialized_str;
        auto t0 = std::chrono::high_resolution_clock::now();
        bool success = recvMsg(fd, serialized_str);
        auto t1 = std::chrono::high_resolution_clock::now();
        MYASSERT(success);
        unique_ptr<ServerSecretShares> deserialized_ss =
            make_unique<ServerSecretShares>(serialized_str, curr_max_degree,
                                            curr_detect_length_k,
                                            this->local_compute_parallel_num);
        auto t2 = std::chrono::high_resolution_clock::now();
        ss = move(deserialized_ss);
        MYASSERT(ss->size() != 0);

        timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        timer.add_serial(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    }

    // cycle detection main loop
    void run(string share_file1, string share_file2, bool debugging) {
        loadSecretShare(share_file1, share_file2);
        connectWithOtherServers();
        cout << "successfully connected to other servers." << endl;
        validateFds();

        timer.reset(); 

        while (this->detect_length_k <= this->total_detect_length) {
            cout << "\n==============================================" << endl;
            cout << "[Flora] current length k = " << this->detect_length_k << endl;
            cout << "Number of tuples: " << this->getFirstServerSecretShares()->size() << " tuples" << endl;
            cout << "==============================================" << endl;
            auto start = system_clock::now();
            
            if (debugging) {
                reconCurrentSS(this->max_degree, this->detect_length_k);
            }

            // Shuffle in sort
            timer.reset();
            shuffleWrapper(this->max_degree, this->detect_length_k);
            printTimerStats("ObShuffle-1");

            if (debugging) {
                reconCurrentSS(this->max_degree, this->detect_length_k);
            }

            // 3PC -> 2PC
            auto t_tc0 = std::chrono::high_resolution_clock::now();
            unique_ptr<ServerSecretShares> local_compute_ss = computeTwoPCShare();
            auto t_tc1 = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_tc1 - t_tc0).count());

            // Sort
            sortWrapper(local_compute_ss);
            printTimerStats("ObSort");

            if (debugging) {
                reconWithTwoPCShare(local_compute_ss);
            }

            // Neighbor passing
            neighborPassWrapper(local_compute_ss); 
            printTimerStats("Ob-PPS");

            if (debugging) {
                reconWithTwoPCShare(local_compute_ss);
            }

            // Parse
            auto t_ex0 = std::chrono::high_resolution_clock::now();
            this->setMaxDegreeToOne();
            unique_ptr<ServerSecretShares> parsed_shares = make_unique<ServerSecretShares>();
            parsePathsWrapper(local_compute_ss, parsed_shares);
            auto t_ex1 = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_ex1 - t_ex0).count());

            uint64_t total_num_of_parsed_tuples = parsed_shares->size();
            cout << "[INFO] Parse done. " << total_num_of_parsed_tuples << " tuples (d=1) in total." << endl;
            
            // 2PC -> 3PC
            reassignSecretShares(parsed_shares);
            printTimerStats("2PC -> 3PC");

            // Shuffle in filter
            shuffleWrapper(this->max_degree, this->detect_length_k);
            printTimerStats("ObShuffle-2");

            if (debugging) {
                reconCurrentSS(this->max_degree, this->detect_length_k);
            }

            auto t_tc2 = std::chrono::high_resolution_clock::now();
            local_compute_ss->clear();
            local_compute_ss = computeTwoPCShare();
            auto t_tc3 = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_tc3 - t_tc2).count());

            if (debugging) {
                reconWithTwoPCShare(local_compute_ss);
            }

            // Filter
            unique_ptr<ServerSecretShares> cycle_shares = make_unique<ServerSecretShares>();
            filterWrapper(local_compute_ss, cycle_shares);
            printTimerStats("Ob-Filter");

            uint64_t num_of_valid_paths = local_compute_ss->size();
            cout << "[INFO] Filter done. Valid paths: " << num_of_valid_paths
                 << " | dummys/ cycles/ Edges: " << total_num_of_parsed_tuples - num_of_valid_paths << endl;

            if (local_compute_ss->size() == 0 && server_id != SERVER_ID::S3) {
                cout << "[SUCCESS] tuple size = 0, detection done." << endl;
                cout << "Round " << this->detect_length_k << " has detected " << cycle_shares->size() << " cycles." << endl;
                break;
            }
            cout << "Round " << this->detect_length_k << " has detected " << cycle_shares->size() << " cycles." << endl;

            // Prepare for the next round of detection (k -> k+1)
            auto t_pad0 = std::chrono::high_resolution_clock::now();
            this->incDetectLength();
            this->resetMaxDegree();
            padPathsAndPristineEdges(local_compute_ss); 
            auto t_pad1 = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_pad1 - t_pad0).count());
            printTimerStats("Append edges and pad tuples");

            if (debugging) {
                reconWithTwoPCShare(local_compute_ss);
            }

            reassignSecretShares(local_compute_ss);
            auto t_tb0 = std::chrono::high_resolution_clock::now();
            compute_server->assignTieBreakerIds();
            auto t_tb1 = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_tb1 - t_tb0).count());
            printTimerStats("Assign ids");
            
            auto end = system_clock::now();
            cout << "[TIME] ──>  Round " << this->detect_length_k-1 << " takes: "
                 << duration_cast<std::chrono::duration<double>>(end - start).count() << " seconds.\n" << endl;
        }
        cout << "[Flora] Done." << endl;
        closeConnectionWithOtherServers();
    }

   private:
    int partition_thread_num_debug = 0; 
    uint64_t packed_sort_invocation = 0;
};

#endif  // SERVER_GPU_HPP