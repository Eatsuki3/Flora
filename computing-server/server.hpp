#ifndef COMPUTE_SERVER_HPP
#define COMPUTE_SERVER_HPP

// #define TEST_BEAVER      // test beaver triple correctness
// #define CHECK_RESULT     // check the neighbor passing result correctness

#include <sstream>
#include <memory>
#include <chrono>
#include <vector>
#include <thread>
#include <algorithm>
#include <map>
#include <mutex>
#include <atomic>
#include <math.h> // ceil, log2
#include <netinet/tcp.h>

#include "preprocess/flora_preprocess_raw.hpp"
#include "preprocess/flora_dealer_triple_store.hpp"
#include "secret-share/secret_share.hpp"
#include "two_pc_compute.hpp"
#include "gf.hpp"

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

    // print timer stats
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

   protected:
    unique_ptr<Server> compute_server;

    // pristine edges: only stored once in the first round (load), and then only read without participating in intermediate shuffling
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

    /*ip address of the three computing server*/
    string ip_1;
    string ip_2;
    string ip_3;

    /*when the batch processing num is set to 0, it has no effect of batching*/
    /*just all data is loaded into memory once with no batching*/
    uint64_t filter_batch_num;
    uint64_t neighbor_pass_batch_num;
    uint64_t extract_batch_num;
    uint64_t partition_batch_num;

    int tie_breaker_bit;
    uint64_t tie_breaker_range;

    /*fd of connection*/
    int fd_1;
    int fd_2;
    int fd_3;

    string ip_no_port_1;
    string ip_no_port_2;
    string ip_no_port_3;

    int rand_median_seed = 100;
    int start_port = 12345;
    // for simplicity fix the random
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

    // TEST only: generate Beaver triples for AND operation
    void getAndTriples(vector<uint32_t>& a, vector<uint32_t>& b,
                    vector<uint32_t>& c, uint64_t n, int party) {
        std::mt19937 gen(0xA5D7);    
        a.resize(n); b.resize(n); c.resize(n);
        for (uint64_t i = 0; i < n; i++) {
            uint32_t a_full = gen(), b_full = gen();
            uint32_t c_full = a_full & b_full;       
            uint32_t a_m = gen(), b_m = gen(), c_m = gen();
            if (party == ALICE) { a[i]=a_m;          b[i]=b_m;          c[i]=c_m; }
            else                { a[i]=a_full^a_m;   b[i]=b_full^b_m;   c[i]=c_full^c_m; }
        }
    }

    // TEST only: batched Beaver AND operation
    void beaverBatchAnd(NetIO* io,
                        const vector<uint32_t>& X, const vector<uint32_t>& Y,
                        vector<uint32_t>& Z, uint64_t n, int party) {
        Z.resize(n);
        vector<uint32_t> a, b, c;
        getAndTriples(a, b, c, n, party);

        vector<uint32_t> de_share(2*n);
        for (uint64_t i = 0; i < n; i++) {
            de_share[i]     = X[i] ^ a[i];  
            de_share[n + i] = Y[i] ^ b[i];  
        }
        vector<uint32_t> de;
        revealXor(io, de_share, de, 2*n, party);

        for (uint64_t i = 0; i < n; i++) {
            uint32_t d = de[i], e = de[n+i];
            Z[i] = c[i] ^ (d & b[i]) ^ (e & a[i]) ^ ((party == ALICE) ? (d & e) : 0u);
        }
    }
    
    // TEST only: test Beaver AND operation correctness
    void testBeaverAnd(NetIO* io, int party) {
        uint64_t n = 1000;
        std::mt19937 g(party == ALICE ? 111u : 222u);
        vector<uint32_t> X(n), Y(n);
        for (uint64_t i = 0; i < n; i++) { X[i] = g(); Y[i] = g(); }

        vector<uint32_t> Z;
        beaverBatchAnd(io, X, Y, Z, n, party);

        vector<uint32_t> x_rec, y_rec, z_rec;
        revealXor(io, X, x_rec, n, party);
        revealXor(io, Y, y_rec, n, party);
        revealXor(io, Z, z_rec, n, party);

        if (party == ALICE) {
            int err = 0;
            for (uint64_t i = 0; i < n; i++)
                if (z_rec[i] != (x_rec[i] & y_rec[i])) err++;
            cout << "[testBeaverAnd] errors: " << err << "/" << n
                << (err == 0 ? "  PASS" : "  FAIL") << endl;
        }
    }

    void revealXor(NetIO* io, const vector<uint32_t>& mine,
                vector<uint32_t>& out, uint64_t n, int party) {
        vector<uint32_t> peer(n);
        if (party == ALICE) { io->send_data(mine.data(), n*4); io->flush(); io->recv_data(peer.data(), n*4); }
        else                { io->recv_data(peer.data(), n*4); io->send_data(mine.data(), n*4); io->flush(); }
        out.resize(n);
        for (uint64_t i = 0; i < n; i++) out[i] = mine[i] ^ peer[i];  
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
          ip_1(ip_1_),
          ip_2(ip_2_),
          ip_3(ip_3_),
          filter_batch_num(filter_batch_num_),
          neighbor_pass_batch_num(neighbor_pass_batch_num_),
          extract_batch_num(extract_batch_num_),
          partition_batch_num(partition_batch_num_),
          tie_breaker_bit(tie_breaker_bit_) {
        ip_no_port_1 = split(ip_1, ":")[0];
        ip_no_port_2 = split(ip_2, ":")[0];
        ip_no_port_3 = split(ip_3, ":")[0];
        tie_breaker_range = (1 << tie_breaker_bit);

        compute_server = make_unique<Server>(
            server_id_, id_range_, max_degree_, detect_length_k_,
            local_compute_parallel_num_, seed12_permute_, seed23_permute_,
            seed31_permute_, seedA_, seedB_, this->tie_breaker_range);
    }

    ~NetServer() {}

    enum class FloraDealerDomain {
        Neighbor,
        Filter
    };

    // load triples
    void loadDealerTripleFiles(
            const std::string& neighbor_s0_path, const std::string& neighbor_s1_path,
            const std::string& filter_s0_path, const std::string& filter_s1_path) {

        if (server_id != SERVER_ID::S3) { return; }

        dealer_triple_store = std::make_unique<flora::preprocess::DealerTripleStore>(
                    neighbor_s0_path, neighbor_s1_path, filter_s0_path, filter_s1_path);

        cout << "[S3] Pre-generated triple files loaded." << endl;
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
        int port = (party == ALICE) ? port_for_s0 : port_for_s1;
        NetIO* io = new NetIO(ip_no_port_3.c_str(), port);
        io->sync();
        return io;
    }

    NetIO* connectToNeighborDealer(int party) {
        int port = (party == ALICE) ? neighborDealerPortForS0() : neighborDealerPortForS1();
        NetIO* io = new NetIO(ip_no_port_3.c_str(), port);
        io->sync();
        return io;
    }

    void serveNeighborDealerStream() {
        cout << "[S3][Neighbor Dealer] " << "listening for S0/S1 connections..." << endl;
        MYASSERT(server_id == SERVER_ID::S3);
        MYASSERT(dealer_triple_store != nullptr);

        NetIO* io_to_s0 = nullptr;
        NetIO* io_to_s1 = nullptr;

        // listen to S0 and S1 in parallel to avoid blocking
        thread listen_s0([&] {
            io_to_s0 = new NetIO(nullptr, neighborDealerPortForS0());
            io_to_s0->sync();
        });

        thread listen_s1([&] {
            io_to_s1 = new NetIO(nullptr, neighborDealerPortForS1());
            io_to_s1->sync();
        });

        listen_s0.join();
        listen_s1.join();

        cout << "[S3][Neighbor Dealer] " << "S0/S1 connected." << endl;

        while (true) {
            uint64_t count_s0 = 0;
            uint64_t count_s1 = 0;

            // receive the number of triples needed from both parties
            thread recv_s0([&] {
                io_to_s0->recv_data_internal(
                    reinterpret_cast<char*>(&count_s0),
                    sizeof(count_s0));
            });

            thread recv_s1([&] {
                io_to_s1->recv_data_internal(
                    reinterpret_cast<char*>(&count_s1),
                    sizeof(count_s1));
            });

            recv_s0.join();
            recv_s1.join();

            MYASSERT(count_s0 == count_s1);

            if (count_s0 == flora::preprocess::kFinishRequest) {
                cout << "[S3][Neighbor Dealer] " << "stream finished." << endl;
                break;
            }

            const uint64_t count = count_s0;

            vector<flora::preprocess::BooleanTripleRecord> records_s0;

            vector<flora::preprocess::BooleanTripleRecord> records_s1;

            dealer_triple_store->takeNeighbor(count, records_s0, records_s1);

            MYASSERT(records_s0.size() == count);
            MYASSERT(records_s1.size() == count);

            auto comm_begin = high_resolution_clock::now();
            thread send_s0([&] {
                if (count > 0) {
                    io_to_s0->send_data_internal(
                        reinterpret_cast<const char*>(records_s0.data()),
                        count * sizeof(flora::preprocess::BooleanTripleRecord));
                }
                io_to_s0->flush();
            });

            thread send_s1([&] {
                if (count > 0) {
                    io_to_s1->send_data_internal(
                        reinterpret_cast<const char*>(records_s1.data()),
                        count * sizeof(flora::preprocess::BooleanTripleRecord));
                }
                io_to_s1->flush();
            });

            send_s0.join();
            send_s1.join();
            auto comm_end = high_resolution_clock::now();
            timer.add_comm(duration_cast<microseconds>(comm_end - comm_begin).count());


            cout << "[S3][Neighbor Dealer] sent " << count << " triples." << endl;
        }

        delete io_to_s0;
        delete io_to_s1;
    }
    void serveDealerStream(int port_for_s0, int port_for_s1, FloraDealerDomain domain) {

        MYASSERT(server_id == SERVER_ID::S3);
        MYASSERT(dealer_triple_store != nullptr);

        NetIO* io_to_s0 = nullptr;
        NetIO* io_to_s1 = nullptr;

        thread listen_s0([&] {
            io_to_s0 = new NetIO(nullptr, port_for_s0);
            io_to_s0->sync();
        });

        thread listen_s1([&] {
            io_to_s1 = new NetIO(nullptr, port_for_s1);
            io_to_s1->sync();
        });

        listen_s0.join();
        listen_s1.join();

        while (true) {
            uint64_t count_s0 = 0;
            uint64_t count_s1 = 0;

            thread recv_s0([&] {
                io_to_s0->recv_data_internal(
                    reinterpret_cast<char*>(&count_s0),
                    sizeof(count_s0));
            });

            thread recv_s1([&] {
                io_to_s1->recv_data_internal(
                    reinterpret_cast<char*>(&count_s1),
                    sizeof(count_s1));
            });

            recv_s0.join();
            recv_s1.join();

            MYASSERT(count_s0 == count_s1);

            if (count_s0 == flora::preprocess::kFinishRequest) {
                break;
            }

            const uint64_t count = count_s0;

            if (domain == FloraDealerDomain::Neighbor) {
                std::vector<flora::preprocess::BooleanTripleRecord> records_s0;
                std::vector<flora::preprocess::BooleanTripleRecord> records_s1;

                dealer_triple_store->takeNeighbor(count, records_s0, records_s1);
                auto comm_begin = high_resolution_clock::now();
                thread send_s0([&] {
                    if (count > 0) {
                        io_to_s0->send_data_internal(
                            reinterpret_cast<const char*>(records_s0.data()),
                            count * sizeof(flora::preprocess::BooleanTripleRecord));
                    }
                    io_to_s0->flush();
                });

                thread send_s1([&] {
                    if (count > 0) {
                        io_to_s1->send_data_internal(
                            reinterpret_cast<const char*>(records_s1.data()),
                            count * sizeof(flora::preprocess::BooleanTripleRecord));
                    }
                    io_to_s1->flush();
                });

                send_s0.join();
                send_s1.join();
                auto comm_end = high_resolution_clock::now();
                timer.add_comm(duration_cast<microseconds>(comm_end - comm_begin).count());

            } else {
                std::vector<flora::preprocess::GfZeroTestRecord> records_s0;
                std::vector<flora::preprocess::GfZeroTestRecord> records_s1;

                dealer_triple_store->takeFilter(count, records_s0, records_s1);

                auto comm_begin = high_resolution_clock::now();
                thread send_s0([&] {
                    if (count > 0) {
                        io_to_s0->send_data_internal(
                            reinterpret_cast<const char*>(records_s0.data()),
                            count * sizeof(flora::preprocess::GfZeroTestRecord));
                    }
                    io_to_s0->flush();
                });

                thread send_s1([&] {
                    if (count > 0) {
                        io_to_s1->send_data_internal(
                            reinterpret_cast<const char*>(records_s1.data()),
                            count * sizeof(flora::preprocess::GfZeroTestRecord));
                    }
                    io_to_s1->flush();
                });

                send_s0.join();
                send_s1.join();
                auto comm_end = high_resolution_clock::now();
                timer.add_comm(duration_cast<microseconds>(comm_end - comm_begin).count());

            }
        }

        delete io_to_s0;
        delete io_to_s1;
    }

    // load secret shares from files
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
        cout << "Successfully loaded the first secret share file.\n";

        uint64_t V_boundary = 0;
        ifstream in1_meta(share_file1 + "_meta");
        if (in1_meta.is_open()) {
            in1_meta >> V_boundary;
            in1_meta.close();
        } else {
            V_boundary = compute_server->getFirstServerSecretShares()->size();
            cout << "[WARNING] Companion metadata not detected, falling back to full copy. If run to k>=3, it may trigger contamination.\n";
        }

        // 加载第二份额文件
        ifstream in2(share_file2, ios::binary);
        MYASSERT(in2.is_open());
        stringstream ss2;
        ss2 << in2.rdbuf();
        in2.close();
        string serialized_str2 = ss2.str();
        compute_server->loadSecondServerSecretShares(serialized_str2);
        serialized_str2.clear();
        cout << "Successfully loaded the second secret share file.\n";

        // 分配系统初始 tie_breaker 序列
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

    // Parallel prefix neighbor passing
    void obliviousPrefixNeighborPass(unique_ptr<ServerSecretShares>& ss,
                               string another_server_ip, int party) {
        using namespace std::chrono;
        
        cout << "\n========================================================" << endl;
        cout << " [PROFILE] Ob-PPS start" << endl;
        cout << " [PROFILE] T: " << ss->size() << endl;
        cout << " [PROFILE] d: " << max_degree << endl;
        cout << "========================================================" << endl;

        auto t_total_start = high_resolution_clock::now();

        uint64_t element_num = ss->size();
        uint64_t B = max_degree + 1; 
        
        vector<uint32_t> global_S;
        global_S.reserve(element_num * B);

        // 1. local data extraction
        cout << " -> [STG 1/4] local data extraction... " << flush;
        auto t_stage1_start = high_resolution_clock::now();
        for (uint64_t i = 0; i < element_num; i++) {
            ss->svec[i]->extractAndAppendDAndNeighbos(global_S); 
        }
        auto t_stage1_end = high_resolution_clock::now();
        cout << "finish! Time: " << duration_cast<milliseconds>(t_stage1_end - t_stage1_start).count() << " ms" << endl;
        timer.add_comp(duration_cast<microseconds>(t_stage1_end - t_stage1_start).count());

        // 2. load F and V into local memory, and prepare for the next stage
        cout << " -> [STG 2/4] load F and V ... " << flush;
        auto t_stage2_start = high_resolution_clock::now();
        int np_port = start_port + 50;   
        NetIO* io = new NetIO(party == ALICE ? nullptr : another_server_ip.c_str(), np_port);
        io -> reset_stats();

        // connect to the dealer for Beaver triples.
        NetIO* dealer_io = connectToNeighborDealer(party);

        // This engine does not require initEngine().
        // beaverBatchAnd() will use the existing normal io and dealer_io.
        TwoPcEngine beaver_engine(party, another_server_ip, np_port, bit_len, tie_breaker_bit);

        beaver_engine.setTimer(&this->timer);
        beaver_engine.setDealerIO(dealer_io);
            
        auto t_stage2_end = high_resolution_clock::now();
        cout << "finish!" << endl;
        timer.add_comp(duration_cast<microseconds>(t_stage2_end - t_stage2_start).count());
             
#ifdef CHECK_RESULT
        {
            vector<uint32_t> in_plain;
            revealXor(io, global_S, in_plain, element_num * B, party);
            if (party == ALICE) {
                cout << "[CHECK-INPUT] element_num=" << element_num << endl;
                for (uint64_t i = 0; i < element_num; i++) {
                    cout << "  in[" << i << "] F=" << in_plain[i*B] << " V=[";
                    for (int d = 0; d < max_degree; d++)
                        cout << in_plain[i*B+1+d] << (d+1<max_degree?",":"");
                    cout << "]\n";
                }
            }
        }
#endif


        // 3. prefix scan
        cout << "\n -> [STG 3/4] start prefix scan" << endl;
        auto t_stg3_start = high_resolution_clock::now();

        vector<uint32_t> global_S_temp(element_num * B, 0);
        uint64_t rounds = ceil(log2((double)element_num));
        
#ifdef TEST_BEAVER
        testBeaverAnd(io, party);
        delete io;
        return;
#endif  

        for (uint64_t k = 1; k <= rounds; ++k) {
            auto t_round_start = high_resolution_clock::now();

            uint64_t offset = (1ULL << (k - 1));
            if (offset >= element_num) break;
            uint64_t active_num = element_num - offset;

            cout << " ├─ [ROUND " << k << "/" << rounds << "] offset: " << offset
                 << " | active_num: " << active_num << flush;

            uint64_t total = active_num * B;
            vector<uint32_t> inA(total, 0), inB(total, 0), outC(total, 0);

            for (uint64_t a = 0; a < active_num; ++a) {
                uint64_t R = a + offset, L = a, m = a * B;
                uint32_t F_R = global_S[R * B];

                inA[m] = global_S[L * B];   // F_L
                inB[m] = F_R;               // F_R

                uint32_t not_FR = F_R ^ ((party == ALICE) ? 0xFFFFFFFFu : 0u);
                for (int d = 0; d < max_degree; ++d) {
                    inA[m + 1 + d] = not_FR;
                    inB[m + 1 + d] = global_S[L * B + 1 + d];   // V_L[d]
                }
            }
            auto t_round_prep = high_resolution_clock::now();
            timer.add_comp(duration_cast<microseconds>(t_round_prep - t_round_start).count());

            auto t_c_start = high_resolution_clock::now();
            if (total > 0) {
                beaver_engine.beaverBatchAnd(io,inA,inB,outC,total,party);
            }
            auto t_c_end = high_resolution_clock::now();
            auto t_mult_pure_ms = duration_cast<milliseconds>(t_c_end - t_c_start).count();

            auto t_round_up = high_resolution_clock::now();
            for (uint64_t a = 0; a < active_num; ++a) {
                uint64_t R = a + offset, L = a, m = a * B;
                global_S_temp[R * B] = global_S[L * B] ^ global_S[R * B] ^ outC[m];
                for (int d = 0; d < max_degree; ++d) {
                    global_S_temp[R * B + 1 + d] = global_S[R * B + 1 + d] ^ outC[m + 1 + d];
                }
            }

            for (uint64_t i = 0; i < offset; ++i) {
                for (uint64_t j = 0; j < B; ++j) {
                    global_S_temp[i * B + j] = global_S[i * B + j];
                }
            }
            global_S = global_S_temp;

            auto t_round_end = high_resolution_clock::now();
            cout << " | Mult time: " << t_mult_pure_ms << " ms"
                 << " | Round time: " << duration_cast<milliseconds>(t_round_end - t_round_start).count() << " ms" << endl;

            timer.add_comp(duration_cast<microseconds>(t_round_end - t_round_up).count());
        }

#ifdef CHECK_RESULT
        {
            vector<uint32_t> out_plain;
            revealXor(io, global_S, out_plain, element_num * B, party);
            if (party == ALICE) {
                cout << "[CHECK-OUTPUT] element_num=" << element_num << endl;
                for (uint64_t i = 0; i < element_num; i++) {
                    cout << "  out[" << i << "] F=" << out_plain[i*B] << " V=[";
                    for (int d = 0; d < max_degree; d++)
                        cout << out_plain[i*B+1+d] << (d+1<max_degree?",":"");
                    cout << "]\n";
                }
            }
        }
#endif

        uint64_t dealer_sent = dealer_io->bytes_sent;
        uint64_t dealer_recv = dealer_io->bytes_recv;
        cout << "[COMM] neighbor passing: "
             << " | total_sent=" << io->bytes_sent + dealer_sent << " B"
             << " total_recv=" << io->bytes_recv + dealer_recv << " B"
             << " rounds=" << io->num_flush << endl;

        beaver_engine.finishDealerStream();

        delete dealer_io;
        dealer_io = nullptr;

        delete io;
        io = nullptr;
        auto t_stage3_done = high_resolution_clock::now();
        
        // 4. local update
        cout << " -> [STG 4/4] local update... " << flush;
        auto t_stage4_start = high_resolution_clock::now();
        for(uint64_t i = 0; i < element_num; i++){
            vector<uint32_t> resolved_neighbors(max_degree, 0);
            for(int d=0; d < max_degree; ++d) {
                resolved_neighbors[d] = global_S[i * B + 1 + d];
            }
            ss->svec[i]->updateNeighbors(resolved_neighbors);
        }
        auto t_stage4_end = high_resolution_clock::now();
        cout << "finish! Time: " << duration_cast<milliseconds>(t_stage4_end - t_stage4_start).count() << " ms" << endl;
        timer.add_comp(duration_cast<microseconds>(t_stage4_end - t_stage4_start).count());

        auto t_total_end = high_resolution_clock::now();
        cout << "========================================================" << endl;
        cout << "  [PROFILE] Neighbor passing time: " << duration_cast<seconds>(t_total_end - t_total_start).count() << " s" << endl;
        cout << "========================================================\n" << endl;
    }

    void partitionWorker(unique_ptr<ServerSecretShares>& ss,
                         vector<PartitionTaskInfo>& tasks,
                         map<pair<uint64_t, uint64_t>, PartitionResult>& res,
                         int index, int port, string another_server_ip,
                         int party) {
        TwoPcEngine engine(party, another_server_ip, port, bit_len, this->tie_breaker_bit);
        engine.setTimer(&this->timer); 
        engine.initEngine();

        MYASSERT(!tasks.empty());
        bool run_partition = tasks[0].run_partition;

        for (auto const& task : tasks) {
            MYASSERT(task.run_partition == run_partition);
            pair<uint64_t, uint64_t> partition_key(task.partition_begin, task.partition_end);
            struct PartitionResult partition_res {
                .smaller_index = vector<uint64_t>(),
                .larger_index = vector<uint64_t>(),
            };
            res[partition_key] = partition_res;

            auto t_prep_start = std::chrono::high_resolution_clock::now();
            vector<uint32_t> node_val_shares, end_node_shares;
            ss->extractNodeValFromServerSecretShare(
                node_val_shares, end_node_shares, task.begin_index, task.end_index);
            vector<uint64_t> tie_breaker_shares;
            ss->extractTieBreakerFromServerSecretShare(
                tie_breaker_shares, task.begin_index, task.end_index);
            auto t_prep_done = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_prep_done - t_prep_start).count());

            if (run_partition) {
                uint64_t total_tuple_num = task.end_index - task.begin_index;
                if (this->partition_batch_num >= total_tuple_num || this->partition_batch_num == 0) {
                    engine.obliviousPartition(
                        node_val_shares, end_node_shares, tie_breaker_shares,
                        task.pivot_val_share, task.pivot_end_node_share,
                        task.pivot_tie_breaker_share,
                        res[partition_key].smaller_index,
                        res[partition_key].larger_index, task.begin_index, 0,
                        total_tuple_num);
                } else {
                    uint64_t curr_vec_index = 0;
                    while (curr_vec_index < total_tuple_num) {
                        uint64_t remained_tuple = total_tuple_num - curr_vec_index;
                        uint64_t process_num = min(remained_tuple, this->partition_batch_num);
                        engine.obliviousPartition(
                            node_val_shares, end_node_shares,
                            tie_breaker_shares, task.pivot_val_share,
                            task.pivot_end_node_share,
                            task.pivot_tie_breaker_share,
                            res[partition_key].smaller_index,
                            res[partition_key].larger_index, task.begin_index,
                            curr_vec_index, process_num);
                        curr_vec_index += process_num;
                    }
                }
            } else {
                vector<uint64_t> sorted_index;
                engine.sortElemsDirectly(node_val_shares, end_node_shares,
                                         tie_breaker_shares, task.begin_index, sorted_index);
                auto t20 = std::chrono::high_resolution_clock::now();
                ss->resortBySortResult(sorted_index, task.begin_index, task.end_index);
                auto t21 = std::chrono::high_resolution_clock::now();
                timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t21 - t20).count());
            }
        }
        engine.terminateEngine();
    }

    void medianWorker(unique_ptr<ServerSecretShares>& ss,
                      vector<pair<uint64_t, uint64_t>>& partition_vec,
                      uint64_t begin_index_in_vec, uint64_t end_index_in_vec,
                      vector<struct PivotInfo>& pivot_vec, int port,
                      string another_server_ip, int party, int tid) {
        MYASSERT(pivot_vec.size() == partition_vec.size());
        TwoPcEngine engine(party, another_server_ip, port, bit_len, this->tie_breaker_bit);
        engine.setTimer(&this->timer);
        engine.initEngine();

        int rand_times = 0;
        for (uint64_t curr_index = begin_index_in_vec; curr_index < end_index_in_vec; curr_index++) {
            auto t_prep_start = std::chrono::high_resolution_clock::now();
            vector<uint32_t> node_val_shares;
            vector<uint32_t> end_node_val_shares;
            vector<uint64_t> tie_breaker_shares;
            vector<uint64_t> indices;
            auto partition_range = partition_vec[curr_index];
            uint64_t num_elems = partition_range.second - partition_range.first;
            uint64_t search_rand_num = min(static_cast<uint64_t>(this->num_elem_to_find_median), num_elems);
            unsigned int rand_seed;
            for (uint64_t i = 0; i < search_rand_num; i++) {
                rand_seed = this->rand_median_seed + port + tid + rand_times;
                uint64_t rand_idx = rand_r(&rand_seed) % num_elems;
                rand_times++;
                while (find(indices.begin(), indices.end(), rand_idx) != indices.end()) {
                    unsigned int rand_seed_sub = this->rand_median_seed + port + tid + rand_times;
                    rand_idx = rand_r(&rand_seed_sub) % num_elems;
                    rand_times++;
                }
                indices.push_back(rand_idx);
            }
            sort(indices.begin(), indices.end());
            for (const auto& rand_idx : indices) {
                node_val_shares.push_back(ss->svec[rand_idx + partition_range.first]->getNodeValShare());
                end_node_val_shares.push_back(ss->svec[rand_idx + partition_range.first]->getEndNodeShare());
                tie_breaker_shares.push_back(ss->svec[rand_idx + partition_range.first]->get_tie_breaker_id());
            }
            auto t_prep_done = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_prep_done - t_prep_start).count());

            uint64_t median_id_in_indicies = engine.findMedian(node_val_shares, end_node_val_shares, tie_breaker_shares);
            
            auto t_write_start = std::chrono::high_resolution_clock::now();
            uint64_t median_id = indices[median_id_in_indicies];
            pivot_vec[curr_index].pivot_val_share = node_val_shares[median_id_in_indicies];
            pivot_vec[curr_index].pivot_end_node_share = end_node_val_shares[median_id_in_indicies];
            pivot_vec[curr_index].pivot_tie_breaker_share = tie_breaker_shares[median_id_in_indicies];
            pivot_vec[curr_index].pivot_index = median_id + partition_range.first;
            auto t_write_done = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_write_done - t_write_start).count());
        }
        engine.terminateEngine();
    }

    void obliviousSort(unique_ptr<ServerSecretShares>& ss,
                       string another_server_ip, int party) {
        vector<pair<uint64_t, uint64_t>> partition_vec;
        vector<pair<uint64_t, uint64_t>> smaller_partition_vec;

        if (ss->size() <= sort_elem_threshold) {
            smaller_partition_vec.push_back(make_pair(0, ss->size()));
        } else {
            partition_vec.push_back(make_pair(0, ss->size()));
        }

        int round = 0;
        while (!partition_vec.empty() || !smaller_partition_vec.empty()) {
            round++;
            uint64_t total_elem_of_partition = 0;
            for (const auto& partition_range : partition_vec) {
                MYASSERT(partition_range.second > partition_range.first);
                total_elem_of_partition += partition_range.second - partition_range.first;
            }

            uint64_t total_elem_of_sort = 0;
            for (const auto& sort_range : smaller_partition_vec) {
                MYASSERT(sort_range.second > sort_range.first);
                MYASSERT(sort_range.second - sort_range.first <= sort_elem_threshold);
                total_elem_of_sort += (sort_range.second - sort_range.first) * (sort_range.second - sort_range.first);
            }

            double partition_ratio = static_cast<double>(total_elem_of_partition) /
                                     static_cast<double>(total_elem_of_partition + total_elem_of_sort);

            int partition_thread_num = static_cast<int>(static_cast<double>(sort_parallel_num) * partition_ratio);
            int small_sort_thread_num = sort_parallel_num - partition_thread_num;

            MYASSERT(partition_thread_num <= sort_parallel_num);
            MYASSERT(partition_thread_num >= 0);

            if (partition_thread_num == 0 && total_elem_of_partition > 0) {
                partition_thread_num = 1;
                small_sort_thread_num = sort_parallel_num - partition_thread_num;
            } else if (small_sort_thread_num == 0 && total_elem_of_sort > 0) {
                small_sort_thread_num = 1;
                partition_thread_num = sort_parallel_num - partition_thread_num;
            }

            MYASSERT(partition_thread_num + small_sort_thread_num == sort_parallel_num);
            MYASSERT(partition_thread_num >= 0);
            MYASSERT(small_sort_thread_num >= 0);

            map<pair<uint64_t, uint64_t>, PartitionThreadsInfo> partition_per_thread_map;
            vector<vector<PartitionTaskInfo>> partition_info_per_thread(sort_parallel_num);

            vector<PivotInfo> pivot_vec(partition_vec.size());
            vector<thread> pivot_finding_threads;

            if (!partition_vec.empty()) {
                uint64_t pivot_task_num = partition_vec.size();
                uint64_t pivot_task_per_thread = pivot_task_num / sort_parallel_num;
                if (pivot_task_per_thread == 0) {
                    pivot_task_per_thread = 1;
                }
                for (int i = 0; i < sort_parallel_num; i++) {
                    uint64_t pivot_begin_index_in_vec = static_cast<uint64_t>(i) * pivot_task_per_thread;
                    uint64_t pivot_end_index_in_vec = static_cast<uint64_t>(i + 1) * pivot_task_per_thread;
                    if (pivot_begin_index_in_vec >= pivot_task_num) {
                        break;
                    }
                    if (pivot_end_index_in_vec >= pivot_task_num) {
                        pivot_end_index_in_vec = pivot_task_num;
                    }
                    if (i == sort_parallel_num - 1) {
                        pivot_end_index_in_vec = pivot_task_num;
                    }
                    if (pivot_begin_index_in_vec == pivot_end_index_in_vec) {
                        continue;
                    }
                    thread t(
                        &NetServer::medianWorker, this, ref(ss),
                        ref(partition_vec), pivot_begin_index_in_vec,
                        pivot_end_index_in_vec, ref(pivot_vec), start_port + i,
                        another_server_ip, party, i);
                    pivot_finding_threads.push_back(move(t));
                }
                for (auto& t : pivot_finding_threads) {
                    t.join();
                }
            }

            if (partition_thread_num > 0) {
                MYASSERT(partition_vec.size() > 0);
                uint64_t current_partition_index = 0;
                uint64_t current_begin = partition_vec[0].first;
                uint64_t current_end = partition_vec[0].second;

                uint64_t elem_per_thread_of_partition = total_elem_of_partition / partition_thread_num;
                for (int i = 0; i < partition_thread_num; i++) {
                    uint64_t begin = i * elem_per_thread_of_partition;
                    uint64_t end = (i + 1) * elem_per_thread_of_partition;
                    if (i == partition_thread_num - 1) {
                        end = total_elem_of_partition;
                    }

                    if (begin == end) {
                        continue;
                    }

                    uint64_t curr_ptr = 0;
                    while (curr_ptr < (end - begin)) {
                        MYASSERT(current_begin < current_end);
                        pair<uint64_t, uint64_t> partition_index_key = partition_vec[current_partition_index];
                        if ((current_begin + (end - begin - curr_ptr)) <= current_end) {
                            PartitionTaskInfo info {
                                .pivot_val_share = pivot_vec[current_partition_index].pivot_val_share,
                                .pivot_end_node_share = pivot_vec[current_partition_index].pivot_end_node_share,
                                .begin_index = current_begin,
                                .end_index = current_begin + (end - begin - curr_ptr),
                                .partition_begin = partition_index_key.first,
                                .partition_end = partition_index_key.second,
                                .run_partition = true,
                                .pivot_index = pivot_vec[current_partition_index].pivot_index,
                                .pivot_tie_breaker_share = pivot_vec[current_partition_index].pivot_tie_breaker_share,
                            };

                            if (partition_per_thread_map.find(partition_index_key) == partition_per_thread_map.end()) {
                                partition_per_thread_map[partition_index_key] = PartitionThreadsInfo{
                                        .threads = vector<int>(),
                                        .pivot_index = pivot_vec[current_partition_index].pivot_index,
                                    };
                            }
                            partition_per_thread_map[partition_index_key].threads.push_back(i);
                            partition_info_per_thread[i].push_back(info);

                            current_begin += (end - begin - curr_ptr);
                            curr_ptr += (end - begin - curr_ptr);

                            if ((current_begin == current_end) && (i != partition_thread_num - 1)) {
                                current_partition_index += 1;
                                MYASSERT(current_partition_index < partition_vec.size());
                                current_end = partition_vec[current_partition_index].second;
                                current_begin = partition_vec[current_partition_index].first;
                            }
                        } else {
                            PartitionTaskInfo info {
                                .pivot_val_share = pivot_vec[current_partition_index].pivot_val_share,
                                .pivot_end_node_share = pivot_vec[current_partition_index].pivot_end_node_share,
                                .begin_index = current_begin,
                                .end_index = current_end,
                                .partition_begin = partition_index_key.first,
                                .partition_end = partition_index_key.second,
                                .run_partition = true,
                                .pivot_index = pivot_vec[current_partition_index].pivot_index,
                                .pivot_tie_breaker_share = pivot_vec[current_partition_index].pivot_tie_breaker_share,
                            };

                            partition_info_per_thread[i].push_back(info);

                            if (partition_per_thread_map.find(partition_index_key) == partition_per_thread_map.end()) {
                                partition_per_thread_map[partition_index_key] = PartitionThreadsInfo{
                                        .threads = vector<int>(),
                                        .pivot_index = pivot_vec[current_partition_index].pivot_index,
                                    };
                            }
                            partition_per_thread_map[partition_index_key].threads.push_back(i);
                            curr_ptr += (current_end - current_begin);

                            if (curr_ptr == (end - begin) && i == partition_thread_num - 1) {
                                break;
                            }

                            current_partition_index += 1;
                            MYASSERT(current_partition_index < partition_vec.size());
                            current_end = partition_vec[current_partition_index].second;
                            current_begin = partition_vec[current_partition_index].first;
                        }
                    }
                }
            }

            if (small_sort_thread_num > 0) {
                uint64_t total_sort_range_num = smaller_partition_vec.size();
                uint64_t partition_range_for_sort_per_thread = total_sort_range_num / small_sort_thread_num;
                if (partition_range_for_sort_per_thread == 0) {
                    partition_range_for_sort_per_thread = 1;
                }
                for (int i = 0; i < small_sort_thread_num; i++) {
                    uint64_t begin_index_in_vec = static_cast<uint64_t>(i) * partition_range_for_sort_per_thread;
                    uint64_t end_index_in_vec = static_cast<uint64_t>(i + 1) * partition_range_for_sort_per_thread;
                    if (begin_index_in_vec >= total_sort_range_num) {
                        break;
                    }
                    if (end_index_in_vec >= total_sort_range_num) {
                        end_index_in_vec = total_sort_range_num;
                    }

                    if (i == small_sort_thread_num - 1) {
                        end_index_in_vec = total_sort_range_num;
                    }
                    if (begin_index_in_vec == end_index_in_vec) {
                        continue;
                    }
                    for (uint64_t curr_index_in_vec = begin_index_in_vec; curr_index_in_vec < end_index_in_vec; curr_index_in_vec++) {
                        PartitionTaskInfo info {
                            .begin_index = smaller_partition_vec[curr_index_in_vec].first,
                            .end_index = smaller_partition_vec[curr_index_in_vec].second,
                            .partition_begin = smaller_partition_vec[curr_index_in_vec].first,
                            .partition_end = smaller_partition_vec[curr_index_in_vec].second,
                            .run_partition = false,
                        };
                        MYASSERT(i + partition_thread_num < sort_parallel_num);
                        partition_info_per_thread[i + partition_thread_num].push_back(info);
                    }
                }
            }

            vector<map<pair<uint64_t, uint64_t>, PartitionResult>> partition_res_per_thread(sort_parallel_num);

            vector<thread> t_vec;
            for (int i = 0; i < sort_parallel_num; i++) {
                if (partition_info_per_thread[i].empty()) {
                    continue;
                }
                thread t(&NetServer::partitionWorker, this, ref(ss),
                         ref(partition_info_per_thread[i]),
                         ref(partition_res_per_thread[i]), i, start_port + i,
                         another_server_ip, party);
                t_vec.push_back(move(t));
            }

            for (auto& t : t_vec) {
                t.join();
            }

            partition_vec.clear();
            smaller_partition_vec.clear();

            uint64_t larger_elem_num_after_partition = 0;
            uint64_t smaller_elem_num_after_partition = 0;
            for (const auto& kv : partition_per_thread_map) {
                pair<uint64_t, uint64_t> partition_range = kv.first;
                vector<int> target_threads = kv.second.threads;
                uint64_t small_index_size = 0;
                uint64_t large_index_size = 0;
                for (const auto& tid : target_threads) {
                    MYASSERT(partition_res_per_thread[tid].find(partition_range) != partition_res_per_thread[tid].end());
                    small_index_size += partition_res_per_thread[tid][partition_range].smaller_index.size();
                    large_index_size += partition_res_per_thread[tid][partition_range].larger_index.size();
                }

                auto t330 = std::chrono::high_resolution_clock::now();
                vector<uint64_t> all_small_indices(small_index_size);
                vector<uint64_t> all_large_indices(large_index_size);
                uint64_t small_current_index = 0;
                uint64_t large_current_index = 0;
                for (const auto& tid : target_threads) {
                    copy(partition_res_per_thread[tid][partition_range].smaller_index.begin(),
                         partition_res_per_thread[tid][partition_range].smaller_index.end(),
                         all_small_indices.begin() + small_current_index);
                    copy(partition_res_per_thread[tid][partition_range].larger_index.begin(),
                         partition_res_per_thread[tid][partition_range].larger_index.end(),
                         all_large_indices.begin() + large_current_index);
                    small_current_index += partition_res_per_thread[tid][partition_range].smaller_index.size();
                    large_current_index += partition_res_per_thread[tid][partition_range].larger_index.size();
                }

                larger_elem_num_after_partition += all_large_indices.size();
                smaller_elem_num_after_partition += all_small_indices.size();

                MYASSERT(all_small_indices.size() + all_large_indices.size() == (partition_range.second - partition_range.first));

                ss->resortByPartition(
                    all_small_indices, all_large_indices, partition_range.first,
                    partition_range.second, kv.second.pivot_index);
                auto t331 = std::chrono::high_resolution_clock::now();
                timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t331 - t330).count());

                if (((partition_range.first + small_current_index) > 1) &&
                    (partition_range.first + 1 < partition_range.first + small_current_index - 1)) {
                    if (small_current_index - 1 > sort_elem_threshold) {
                        partition_vec.push_back(make_pair(
                            partition_range.first,
                            partition_range.first + small_current_index - 1));
                    } else {
                        smaller_partition_vec.push_back(make_pair(
                            partition_range.first,
                            partition_range.first + small_current_index - 1));
                    }
                }

                if ((partition_range.first + small_current_index + 1) < partition_range.second) {
                    if (partition_range.second - (partition_range.first + small_current_index) > sort_elem_threshold) {
                        partition_vec.push_back(make_pair(
                            partition_range.first + small_current_index,
                            partition_range.second));
                    } else {
                        smaller_partition_vec.push_back(make_pair(
                            partition_range.first + small_current_index,
                            partition_range.second));
                    }
                }
            }
        }
    }

    void finishEmptyFilterDealerStream(int filter_thread_id, int party) {

        NetIO* dealer_io = connectToDealer(party, 
                filterDealerPortForS0(filter_thread_id),
                filterDealerPortForS1(filter_thread_id));

        uint64_t finish = flora::preprocess::kFinishRequest;
        dealer_io->send_data_internal(reinterpret_cast<const char*>(&finish),sizeof(finish));
        dealer_io->flush();

        delete dealer_io;
    }


    void filterThread(int filter_thread_id, int port, string another_server_ip, int party,
                      unique_ptr<ServerSecretShares>& ss, uint64_t begin_index,
                      uint64_t end_index, vector<bool>& is_valid, vector<bool>& is_cycle,
                      uint64_t& out_bytes_sent, uint64_t& out_bytes_recv, uint64_t& out_rounds) {
        MYASSERT(begin_index < end_index);
        auto t_ext_start = std::chrono::high_resolution_clock::now();
        vector<uint32_t> path_shares;
        ss->extractSinglePathFromRange(path_shares, begin_index, end_index);
        auto t_ext_done = std::chrono::high_resolution_clock::now();
        timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_ext_done - t_ext_start).count());
        
        TwoPcEngine engine(party, another_server_ip, port, bit_len, this->tie_breaker_bit);

        engine.setTimer(&this->timer);
        engine.initNetIOOnly();
        engine.resetCommStats();

        NetIO* dealer_io = connectToDealer(party, filterDealerPortForS0(filter_thread_id), filterDealerPortForS1(filter_thread_id));
        engine.setDealerIO(dealer_io);
            
        const uint64_t local_tuple_num = end_index - begin_index;
        
        is_valid = vector<bool>(local_tuple_num, false);
        is_cycle = vector<bool>(local_tuple_num, false);

        if (this->filter_batch_num == 0 || (local_tuple_num <= this->filter_batch_num)) {
            engine.obliviousFilter(path_shares, 0, local_tuple_num,
                                   this->detect_length_k, is_valid, is_cycle);
        } else {
            uint64_t begin_tuple_index = 0;
            uint64_t total_tuple_num = end_index - begin_index;
            while (begin_tuple_index < total_tuple_num) {
                uint64_t remained_tuple_num = total_tuple_num - begin_tuple_index;
                uint64_t processed_tuple_num = min(this->filter_batch_num, remained_tuple_num);

                engine.obliviousFilter(path_shares, begin_tuple_index,
                                       processed_tuple_num, detect_length_k,
                                       is_valid, is_cycle);
                begin_tuple_index += processed_tuple_num;
            }
            MYASSERT(begin_tuple_index == total_tuple_num);
        }


        uint64_t dealer_sent = dealer_io->bytes_sent;
        uint64_t dealer_recv = dealer_io->bytes_recv;
        out_bytes_sent = engine.getBytesSent() + dealer_sent;
        out_bytes_recv = engine.getBytesRecv() + dealer_recv;
        out_rounds     = engine.getRounds();

        engine.finishDealerStream();
        delete dealer_io;
        dealer_io = nullptr;
        engine.terminateNetIOOnly();

    }

    void oblivousFilter(
            unique_ptr<ServerSecretShares>& ss,
            string another_server_ip,
            int party,
            vector<bool>& is_valid,
            vector<bool>& is_cycle) {

        MYASSERT(is_valid.empty());
        MYASSERT(is_cycle.empty());
        MYASSERT(filter_parallel_num > 0);

        const uint64_t total_tuple_num = ss->size();

        vector<vector<bool>> is_valid_per_thread(filter_parallel_num);
        vector<vector<bool>> is_cycle_per_thread(filter_parallel_num);
        vector<uint64_t> thr_bytes_sent(filter_parallel_num, 0);
        vector<uint64_t> thr_bytes_recv(filter_parallel_num, 0);
        vector<uint64_t> thr_rounds(filter_parallel_num, 0);
        vector<thread> t_vec;
        t_vec.reserve(filter_parallel_num);

        cout << "[S1/S2] oblivousFilter START, total="
            << total_tuple_num
            << " party=" << party
            << " threads=" << filter_parallel_num
            << endl
            << flush;

        // Every filter stream must establish a connection to S3.
        //
        // Use the following range formula to correctly handle:
        //     total_tuple_num < filter_parallel_num
        //
        // Some threads will get an empty range, but still need to send Finish.
        for (int i = 0; i < filter_parallel_num; ++i) {
            const uint64_t begin_index =
                total_tuple_num * static_cast<uint64_t>(i) /
                static_cast<uint64_t>(filter_parallel_num);

            const uint64_t end_index =
                total_tuple_num * static_cast<uint64_t>(i + 1) /
                static_cast<uint64_t>(filter_parallel_num);

            if (begin_index == end_index) {
                // No tuple in this stream, but still need to connect to S3 and send the finish signal.
                thread t(&NetServer::finishEmptyFilterDealerStream,this,i,party);
                t_vec.push_back(move(t));
                continue;
            }


            thread t(&NetServer::filterThread,
                this, i, start_port + i, another_server_ip,
                party, ref(ss),
                begin_index, end_index,
                ref(is_valid_per_thread[i]), ref(is_cycle_per_thread[i]),
                ref(thr_bytes_sent[i]), ref(thr_bytes_recv[i]),
                ref(thr_rounds[i]));

            t_vec.push_back(move(t));
        }

        for (auto& t : t_vec) {
            t.join();
        }

        auto t_merge_start = std::chrono::high_resolution_clock::now();

        uint64_t total_sent = 0;
        uint64_t total_recv = 0;
        uint64_t max_rounds = 0;
        uint64_t sum_rounds = 0;

        for (int i = 0; i < filter_parallel_num; ++i) {

            total_sent += thr_bytes_sent[i];
            total_recv += thr_bytes_recv[i];
            max_rounds = std::max(max_rounds, thr_rounds[i]);
            sum_rounds += thr_rounds[i];
        }

        cout << "[COMM][filter] sent="
            << total_sent << " B"
            << " recv=" << total_recv << " B"
            << " sent+recv="
            << total_sent + total_recv << " B"
            << endl;

        cout << "[COMM][filter] rounds:"
            << " max_over_threads="
            << max_rounds
            << " sum_over_threads="
            << sum_rounds
            << endl;

        is_valid.assign(total_tuple_num, false);
        is_cycle.assign(total_tuple_num, false);

        uint64_t cur_ptr = 0;

        for (int i = 0; i < filter_parallel_num; ++i) {

            MYASSERT(is_valid_per_thread[i].size() == is_cycle_per_thread[i].size());

            const uint64_t thread_result_size = is_valid_per_thread[i].size();

            MYASSERT(cur_ptr + thread_result_size <= total_tuple_num);

            copy(is_valid_per_thread[i].begin(),
                is_valid_per_thread[i].end(),
                is_valid.begin() + cur_ptr);

            copy(is_cycle_per_thread[i].begin(),
                is_cycle_per_thread[i].end(),
                is_cycle.begin() + cur_ptr);

            cur_ptr += thread_result_size;
        }

        MYASSERT(cur_ptr == total_tuple_num);

        auto t_merge_done = std::chrono::high_resolution_clock::now();

        timer.add_comp(
            std::chrono::duration_cast<std::chrono::microseconds>(
                    t_merge_done - t_merge_start).count());
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
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));   // NODELAY
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
        setsockopt(*target_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));  // ← 之前漏了这行!
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
        if (this->server_id == SERVER_ID::S1) {
            auto t0 = std::chrono::high_resolution_clock::now();
            unique_ptr<ServerSecretShares> x2 = compute_server->shuffle_s1_step1();
            auto t1 = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

            thread sendX2(&NetServer::sendServerSecretShareWorker, this, x2.get(), fd_2);
            sendX2.join();
            
            auto t2 = std::chrono::high_resolution_clock::now();
            waitForOtherServers();
            auto t3 = std::chrono::high_resolution_clock::now();
            timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count());
        } else if (this->server_id == SERVER_ID::S2) {
            auto t0 = std::chrono::high_resolution_clock::now();
            unique_ptr<ServerSecretShares> y1 = compute_server->shuffle_s2_step1();
            auto t1 = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

            thread sendY1(&NetServer::sendServerSecretShareWorker, this, y1.get(), fd_3);
            unique_ptr<ServerSecretShares> x2 = make_unique<ServerSecretShares>();
            thread recvX2(&NetServer::recvServerSecretShareWorker, this, ref(x2), fd_1, curr_max_degree, curr_detect_length_k);
            sendY1.join();
            recvX2.join();

            auto t2 = std::chrono::high_resolution_clock::now();
            unique_ptr<ServerSecretShares> c1 = compute_server->shuffle_s2_step2(*x2.get());
            auto t3 = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count());

            thread sendC1(&NetServer::sendServerSecretShareWorker, this, c1.get(), fd_3);
            unique_ptr<ServerSecretShares> c2 = make_unique<ServerSecretShares>();
            thread recvC2(&NetServer::recvServerSecretShareWorker, this, ref(c2), fd_3, curr_max_degree, curr_detect_length_k);
            sendC1.join();
            recvC2.join();

            auto t4 = std::chrono::high_resolution_clock::now();
            compute_server->shuffle_s2_step3(*c1.get(), *c2.get());
            auto t5 = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count());

            auto t6 = std::chrono::high_resolution_clock::now();
            waitForOtherServers();
            auto t7 = std::chrono::high_resolution_clock::now();
            timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(t7 - t6).count());
        } else if (this->server_id == SERVER_ID::S3) {
            unique_ptr<ServerSecretShares> y1 = make_unique<ServerSecretShares>();
            thread recvY1(&NetServer::recvServerSecretShareWorker, this, ref(y1), fd_2, curr_max_degree, curr_detect_length_k);
            recvY1.join();

            auto t0 = std::chrono::high_resolution_clock::now();
            unique_ptr<ServerSecretShares> c2 = compute_server->shuffle_s3_step1(*y1.get());
            auto t1 = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

            thread sendC2(&NetServer::sendServerSecretShareWorker, this, c2.get(), fd_2);
            unique_ptr<ServerSecretShares> c1 = make_unique<ServerSecretShares>();
            thread recvC1(&NetServer::recvServerSecretShareWorker, this, ref(c1), fd_2, curr_max_degree, curr_detect_length_k);
            sendC2.join();
            recvC1.join();

            auto t2 = std::chrono::high_resolution_clock::now();
            compute_server->shuffle_s3_step2(*c1.get(), *c2.get());
            auto t3 = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count());

            auto t4 = std::chrono::high_resolution_clock::now();
            waitForOtherServers();
            auto t5 = std::chrono::high_resolution_clock::now();
            timer.add_comm(std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count());
        } else {
            MYASSERT(0);
        }
    }

    unique_ptr<ServerSecretShares> computeTwoPCShare() {
        if (server_id == SERVER_ID::S1) {
            vector<ServerSecretShares*> xor_vec = {
                getFirstServerSecretShares(), getSecondServerSecretShares()};
            unique_ptr<ServerSecretShares> ss = compute_server->xorSecretShareTupleAndCreateDst(xor_vec);
            return ss;
        } else if (server_id == SERVER_ID::S2) {
            unique_ptr<ServerSecretShares> ss = make_unique<ServerSecretShares>(getSecondServerSecretShares());
            return ss;
        } else if (server_id == SERVER_ID::S3) {
            unique_ptr<ServerSecretShares> ss = make_unique<ServerSecretShares>();
            return ss;
        } else {
            MYASSERT(0);
        }
    }

    void sortWrapper(unique_ptr<ServerSecretShares>& ss) {
        if (server_id == SERVER_ID::S1) {
            this->obliviousSort(ss, ip_no_port_1, ALICE);
            return;
        } else if (server_id == SERVER_ID::S2) {
            this->obliviousSort(ss, ip_no_port_1, BOB);
            return;
        } else if (server_id == SERVER_ID::S3) {
            return;
        } else {
            MYASSERT(0);
        }
    }

    void neighborPassWrapper(unique_ptr<ServerSecretShares>& ss) {
        if (server_id == SERVER_ID::S1) {
            this->obliviousPrefixNeighborPass(ss, ip_no_port_1, ALICE);
            return;
        } else if (server_id == SERVER_ID::S2) {
            this->obliviousPrefixNeighborPass(ss, ip_no_port_1, BOB);
            return;
        } else if (server_id == SERVER_ID::S3) {
            this->serveNeighborDealerStream();
            return;
        } else {
            MYASSERT(0);
        }
    }

    void extractWrapper(unique_ptr<ServerSecretShares>& ss,
                        unique_ptr<ServerSecretShares>& edge_shares,
                        unique_ptr<ServerSecretShares>& path_shares) {
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

    void parsePathsWrapper(unique_ptr<ServerSecretShares>& path_shares,
                           unique_ptr<ServerSecretShares>& parsed_path_shares) {
        if ((server_id == SERVER_ID::S1) || (server_id == SERVER_ID::S2)) {
            path_shares->extractPathTuples(parsed_path_shares);
            return;
        } else if (server_id == SERVER_ID::S3) {
            return;
        } else {
            MYASSERT(0);
        }
    }

    void moveFilteredPathShares(unique_ptr<ServerSecretShares>& ss,
                                unique_ptr<ServerSecretShares>& cycle_shares,
                                vector<bool>& is_valid,
                                vector<bool>& is_cycle) {
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


    void filterWrapper(unique_ptr<ServerSecretShares>& ss,
                             unique_ptr<ServerSecretShares>& cycle_shares) {
        vector<bool> is_valid, is_cycle;
        if (server_id == SERVER_ID::S1) {
            this->oblivousFilter(ss, ip_no_port_1, ALICE, is_valid, is_cycle);
            moveFilteredPathShares(ss, cycle_shares, is_valid, is_cycle);
            return;
        } else if (server_id == SERVER_ID::S2) {
            this->oblivousFilter(ss, ip_no_port_1, BOB, is_valid, is_cycle);
            moveFilteredPathShares(ss, cycle_shares, is_valid, is_cycle);
            return;
        } else if (server_id == SERVER_ID::S3) {
            cout << "[S3][Filter Dealer] starting "
             << filter_parallel_num
             << " streams"
             << endl << flush;

            MYASSERT(dealer_triple_store != nullptr);

            vector<thread> dealer_threads;
            dealer_threads.reserve(filter_parallel_num);
            for (int i = 0; i < filter_parallel_num; ++i) {
                const int port_s0 = filterDealerPortForS0(i);
                const int port_s1 = filterDealerPortForS1(i);
                cout << "[S3][Filter Dealer] spawn stream="
                 << i
                 << " port_s0=" << port_s0
                 << " port_s1=" << port_s1
                 << endl << flush;

                dealer_threads.emplace_back(&NetServer::serveDealerStream,this,
                    port_s0,port_s1, FloraDealerDomain::Filter);
            }
            for (auto& t : dealer_threads) {t.join();}
            cout << "[S3][Filter Dealer] all streams finished" << endl << flush;
            return;
        } else {
            MYASSERT(0);
        }
    }

    void padPathsAndPristineEdges(unique_ptr<ServerSecretShares>& path_shares) {
        auto t_pad_start = std::chrono::high_resolution_clock::now();
        // pad paths and pristine edges to the new max_degree and detect_length_k
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

        // append reloaded_edges to path_shares
        uint64_t expected_total = path_shares->size() + reloaded_edges->size();
        for (uint64_t i = 0; i < reloaded_edges->size(); i++) {
            path_shares->insert(reloaded_edges->svec[i]); // 移动所有权插入本地
        }
        MYASSERT(path_shares->size() == expected_total);
        cout << "[PER] Append edges done. Tuple size:" << path_shares->size() << endl;
        auto t_pad_done = std::chrono::high_resolution_clock::now();
        timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_pad_done - t_pad_start).count());
    }

    void reassignSecretShares(unique_ptr<ServerSecretShares>& ss) {
        if (server_id == SERVER_ID::S1) {
            ServerSecretShares* s2 = getSecondServerSecretShares();
            compute_server->createRandomnessToDstWithSize(
                s2, this->reassign_randomness_seed + this->detect_length_k + this->max_degree, ss->size());
            ServerSecretShares* s1 = getFirstServerSecretShares();
            vector<ServerSecretShares*> xor_vec = {ss.get(), s2};
            s1->clear();

            compute_server->xorSecretShareTupleWithEmptyDst(s1, xor_vec);

            thread sendS1(&NetServer::sendServerSecretShareWorker, this, s1, fd_3);
            thread sendS2(&NetServer::sendServerSecretShareWorker, this, s2, fd_2);
            sendS1.join();
            sendS2.join();
            return;
        } else if (server_id == SERVER_ID::S2) {
            ServerSecretShares* s3 = getSecondServerSecretShares();
            s3->moveServerShares(ss.get());
            thread sendS3(&NetServer::sendServerSecretShareWorker, this, s3, fd_3);
            ServerSecretShares* s2 = getFirstServerSecretShares();
            s2->clear();
            thread recvS2(&NetServer::recvServerSecretShareToPtrWorker, this,
                          s2, fd_1, this->max_degree, this->detect_length_k);
            sendS3.join();
            recvS2.join();
            return;
        } else if (server_id == SERVER_ID::S3) {
            ServerSecretShares* s3 = getFirstServerSecretShares();
            ServerSecretShares* s1 = getSecondServerSecretShares();
            s3->clear();
            s1->clear();
            thread recvS1(&NetServer::recvServerSecretShareToPtrWorker, this, s1, fd_1, this->max_degree, this->detect_length_k);
            thread recvS3(&NetServer::recvServerSecretShareToPtrWorker, this, s3, fd_2, this->max_degree, this->detect_length_k);
            recvS1.join();
            recvS3.join();
            return;
        } else {
            MYASSERT(0);
        }
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
            auto t_tc0 = std::chrono::high_resolution_clock::now();

            // 3PC -> 2PC 
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

            uint64_t total_num_of_parsed_tuples = parsed_shares->size();
            cout << "[INFO] Parse done. " << total_num_of_parsed_tuples << " tuples (d=1) in total." << endl;
            auto t_ex1 = std::chrono::high_resolution_clock::now();
            timer.add_comp(std::chrono::duration_cast<std::chrono::microseconds>(t_ex1 - t_ex0).count());

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

            // If no active paths remain, the detection is done.
            if (local_compute_ss->size() == 0 && server_id != SERVER_ID::S3) {
                cout << "[SUCCESS] tuple size = 0, detection done" << endl;
                cout << "Round " << this->detect_length_k << " has detected " << cycle_shares->size() << " cycles." << endl;
                break;
            }
            cout << "Round " << this->detect_length_k << " has detected " << cycle_shares->size() << " cycles." << endl;

            // Prepare for the next round of detection (k -> k+1)
            this->incDetectLength(); // k++
            this->resetMaxDegree();

            padPathsAndPristineEdges(local_compute_ss); 
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
    
};

#endif  // COMPUTE_SERVER_HPP