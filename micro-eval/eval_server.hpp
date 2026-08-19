#include "common/common.h"
#include "computing-server/server.hpp"
#include "secret-share/secret_share.hpp"

const int kTotalDetectLength = 100;
const int kShuffleSeed = 1234;

class EvalServer : public NetServer {
   public:

    EvalServer(SERVER_ID server_id_, uint32_t id_range_, int bit_len_,
               int max_degree_, int detect_length_k_, int total_detect_length_,
               int local_compute_parallel_num_, int neighbor_pass_parallel_num_,
               int sort_parallel_num_, int filter_parallel_num_,
               int extract_path_parallel_num_, uint64_t sort_elem_threshold_,
               int num_elem_to_find_median_, int seed12_permute_,
               int seed23_permute_, int seed31_permute_, int seedA_, int seedB_,
               /*ip address of the three computing server*/
               string ip_1_, string ip_2_, string ip_3_,
               /*num of tuples processed in each batch*/
               uint64_t filter_batch_num_, uint64_t neighbor_pass_batch_num_,
               uint64_t extract_batch_num_, uint64_t partition_batch_num_,
               int tie_breaker_bit_)
        : NetServer(server_id_, id_range_, bit_len_, max_degree_,
                    detect_length_k_, total_detect_length_,
                    /*local_compute_parallel_num=*/local_compute_parallel_num_,
                    /*neighbor_pass_parallel_num=*/neighbor_pass_parallel_num_,
                    /*sort_parallel_num=*/sort_parallel_num_,
                    /*filter_parallel_num=*/filter_parallel_num_,
                    /*extract_path_parallel_num=*/extract_path_parallel_num_,
                    /*sort_elem_threshold=*/sort_elem_threshold_,
                    /*num_to_find_median=*/num_elem_to_find_median_,
                    /*seed12_permute=*/seed12_permute_,
                    /*seed23_permute=*/seed23_permute_,
                    /*seed31_permute=*/seed31_permute_, /*seedA=*/seedA_,
                    /*seedB=*/seedB_, ip_1_, ip_2_, ip_3_,
                    /*filter_batch_num=*/filter_batch_num_,
                    /*neighbor_pass_batch_num=*/neighbor_pass_batch_num_,
                    /*extract_batch_num=*/extract_batch_num_,
                    /*partition_batch_num=*/partition_batch_num_,
                    /*tie_breaker_bit=*/tie_breaker_bit_) {}


    void createRandomTuplesForNeighborPassingWorker(
        int max_degree, int detect_length_k, uint64_t nodes_num,
        uint64_t path_tuple_num, unique_ptr<ServerSecretShares> &s1,
        unique_ptr<ServerSecretShares> &s2, int parallel_num, int pid) {
        SecretShareTupleHelper helper((1 << kBitLen), kTieBreakerRange);
        MYASSERT(s1->size() == 0);
        MYASSERT(s2->size() == 0);

        uint64_t nodes_per_thread = nodes_num / parallel_num;
        uint64_t begin_nodes = nodes_per_thread * pid;
        uint64_t end_nodes = nodes_per_thread * (pid + 1);
        if (pid == parallel_num - 1) {
            end_nodes = nodes_num;
        }

        for (uint64_t i = begin_nodes; i < end_nodes; i++) {
            // node id cannot be zero so set to i+1
            uint32_t node_id = (i + 1) % (1 << kBitLen);
            if (node_id == 0) {
                node_id = 1;
            }
            vector<uint32_t> current_neighbors;

            // create neighbors for the node_id.
            int rand_degree = rand() % max_degree;
            for (int j = 0; j < rand_degree; j++) {
                uint32_t neighbor_val = rand() % (1 << kBitLen);
                current_neighbors.push_back(neighbor_val);
            }

            unique_ptr<SecretShareTuple> et = helper.formatEdgeTuple(
                node_id, current_neighbors, max_degree, detect_length_k);

            // assign tie_breaker_id
            et->set_tie_breaker_id(i);

            vector<unique_ptr<SecretShareTuple>> ets =
                helper.splitTupleIntoTwo(et);
            s1->insert(ets[0]);
            s2->insert(ets[1]);

            // create rand_tuple_num of the current node id.
            for (uint64_t j = 0; j < path_tuple_num; j++) {
                vector<uint32_t> path;
                for (int k = 0; k < detect_length_k; k++) {
                    path.push_back(rand() % (1 << kBitLen));
                }
                unique_ptr<SecretShareTuple> pt =
                    helper.formatPathTuple(path, max_degree, detect_length_k);
                vector<unique_ptr<SecretShareTuple>> pts =
                    helper.splitTupleIntoTwo(pt);
                s1->insert(pts[0]);
                s2->insert(pts[1]);
            }
        }
    }

    void createRandomTuplesForNeighborPassing(
        int max_degree, int detect_length_k, uint64_t nodes_num,
        uint64_t path_tuple_num, unique_ptr<ServerSecretShares> &s1,
        unique_ptr<ServerSecretShares> &s2, int parallel_num) {
        vector<unique_ptr<ServerSecretShares>> s1_vec;
        vector<unique_ptr<ServerSecretShares>> s2_vec;
        vector<thread> t_vec;
        for (int i = 0; i < parallel_num; i++) {
            unique_ptr<ServerSecretShares> s1_p =
                make_unique<ServerSecretShares>();
            unique_ptr<ServerSecretShares> s2_p =
                make_unique<ServerSecretShares>();
            s1_vec.push_back(move(s1_p));
            s2_vec.push_back(move(s2_p));
        }
        for (int i = 0; i < parallel_num; i++) {
            thread t(&EvalServer::createRandomTuplesForNeighborPassingWorker,
                     this, max_degree, detect_length_k, nodes_num,
                     path_tuple_num, ref(s1_vec[i]), ref(s2_vec[i]),
                     parallel_num, i);
            t_vec.push_back(move(t));
        }
        for (auto &t : t_vec) {
            t.join();
        }
        s1->mergeFromSSVec(s1_vec);
        s2->mergeFromSSVec(s2_vec);

        MYASSERT(s1->size() == nodes_num * (path_tuple_num + 1));
        MYASSERT(s2->size() == nodes_num * (path_tuple_num + 1));

        cout << "finish creating tuples for neighbor passing\n";
    }

    void createRandomTuplesForFilterWorker(int max_degree, int detect_length_k,
                                           uint64_t path_num,
                                           double valid_ratio,
                                           double cycle_ratio,
                                           unique_ptr<ServerSecretShares> &s1,
                                           unique_ptr<ServerSecretShares> &s2,
                                           int parallel_num, int pid) {
        SecretShareTupleHelper helper((1 << kBitLen), kTieBreakerRange);
        uint64_t valid_path_num = static_cast<uint64_t>(path_num * valid_ratio);
        uint64_t cycle_num = static_cast<uint64_t>(path_num * cycle_ratio);
        uint64_t tuple_per_thread = path_num / parallel_num;
        uint64_t begin_index = tuple_per_thread * pid;
        uint64_t end_index = tuple_per_thread * (pid + 1);
        if (pid == parallel_num - 1) {
            end_index = path_num;
        }

        for (uint64_t i = begin_index; i < end_index; i++) {
            vector<uint32_t> path;
            for (int i = 0; i < detect_length_k; i++) {
                uint32_t node = rand() % ((1 << kBitLen) - 1) + 1;
                while (find(path.begin(), path.end(), node) != path.end()) {
                    node = rand() % ((1 << kBitLen) - 1) + 1;
                }
                path.push_back(node);
            }

            bool is_valid = false;
            bool is_cycle = false;
            if (i <= valid_path_num) {
                is_valid = true;
            }
            if (i <= cycle_num) {
                is_cycle = true;
            }
            if (!is_valid) {
                int repeat_node_id = rand() % (detect_length_k) + 1;
                if (repeat_node_id == detect_length_k) {
                    path.push_back(0);
                } else {
                    path.push_back(path[repeat_node_id]);
                }
            } else if (is_cycle) {
                path.push_back(path[0]);
            } else {
                // valid but non-cycle
                uint32_t last_node = rand() % ((1 << kBitLen) - 1) + 1;
                // if last node repeats with existing nodes
                while (find(path.begin(), path.end(), last_node) !=
                       path.end()) {
                    last_node = rand() % ((1 << kBitLen) - 1) + 1;
                }
                path.push_back(last_node);
            }
            
            unique_ptr<SecretShareTuple> pt =
                helper.formatFilledPathTuple(path, max_degree, detect_length_k);
            pt->set_tie_breaker_id(i);

            vector<unique_ptr<SecretShareTuple>> pts =
                helper.splitTupleIntoTwo(pt);
            s1->insert(pts[0]);
            s2->insert(pts[1]);
        }
    }

    void createRandomTuplesForFilter(int max_degree, int detect_length_k,
                                     uint64_t path_num, double valid_ratio,
                                     double cycle_ratio,
                                     unique_ptr<ServerSecretShares> &s1,
                                     unique_ptr<ServerSecretShares> &s2,
                                     int parallel_num) {
        uint64_t valid_path_num = static_cast<uint64_t>(path_num * valid_ratio);
        uint64_t cycle_num = static_cast<uint64_t>(path_num * cycle_ratio);
        cout << "creating " << path_num << " tuples, " << valid_path_num
             << " tuples, cycle tuples: " << cycle_num << endl;

        vector<unique_ptr<ServerSecretShares>> s1_vec;
        vector<unique_ptr<ServerSecretShares>> s2_vec;
        vector<thread> t_vec;
        for (int i = 0; i < parallel_num; i++) {
            unique_ptr<ServerSecretShares> s1_p =
                make_unique<ServerSecretShares>();
            unique_ptr<ServerSecretShares> s2_p =
                make_unique<ServerSecretShares>();
            s1_vec.push_back(move(s1_p));
            s2_vec.push_back(move(s2_p));
        }
        for (int i = 0; i < parallel_num; i++) {
            thread t(&EvalServer::createRandomTuplesForFilterWorker, this,
                     max_degree, detect_length_k, path_num, valid_ratio,
                     cycle_ratio, ref(s1_vec[i]), ref(s2_vec[i]), parallel_num,
                     i);
            t_vec.push_back(move(t));
        }
        for (auto &t : t_vec) {
            t.join();
        }
        s1->mergeFromSSVec(s1_vec);
        s2->mergeFromSSVec(s2_vec);

        MYASSERT(s1->size() == path_num);
        MYASSERT(s2->size() == path_num);

        cout << "finish creating tuples for filter\n";
    }

    unique_ptr<ServerSecretShares> prepareSecretSharesForNeighborPass(
        int max_degree, int detect_length_k, uint64_t nodes_num,
        uint64_t path_num, int parallel_num) {
        // generate random data, and exchange shares
        unique_ptr<ServerSecretShares> local_share =
            make_unique<ServerSecretShares>();
        if (server_id == SERVER_ID::S1) {
            unique_ptr<ServerSecretShares> s2 =
                make_unique<ServerSecretShares>();
            createRandomTuplesForNeighborPassing(max_degree, detect_length_k,
                                                 nodes_num, path_num,
                                                 local_share, s2, parallel_num);
            sendServerSecretShareWorker(s2.get(), fd_2);
        } else if(server_id == SERVER_ID::S2) {
            recvServerSecretShareWorker(local_share, fd_1, max_degree,
                                        detect_length_k);
        } else{
            // S3 holds no input shares; it only serves triples.
            cout << "[S3] no input shares prepared for neighbor passing"
                 << endl;
        }
        return local_share;
    }

    unique_ptr<ServerSecretShares> prepareSecretSharesForFilter(
        int max_degree, int detect_length_k, uint64_t path_num,
        double valid_ratio, double cycle_ratio, int parallel_num) {
        // generate random data, and exchange shares
        unique_ptr<ServerSecretShares> local_share =
            make_unique<ServerSecretShares>();
        if (server_id == SERVER_ID::S1) {
            unique_ptr<ServerSecretShares> s2 =
                make_unique<ServerSecretShares>();
            createRandomTuplesForFilter(max_degree, detect_length_k, path_num,
                                        valid_ratio, cycle_ratio, local_share,
                                        s2, parallel_num);
            sendServerSecretShareWorker(s2.get(), fd_2);
        } else if (server_id == SERVER_ID::S2) {
            recvServerSecretShareWorker(local_share, fd_1, max_degree,
                                        detect_length_k);
        } else {
            // S3 holds no input shares; it only serves GF zero-test
            // correlations.
            cout << "[S3] no input shares prepared for filter" << endl;
        }
        return local_share;
    }

};