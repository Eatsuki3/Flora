// Ours:
#ifndef GRAPH_HOLDER_HPP
#define GRAPH_HOLDER_HPP

#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "secret-share/secret_share.hpp"

const uint64_t kTieBreakerRangeForGen = (1ULL << 25);

class GraphHolder {
   public:
    unordered_map<uint32_t, vector<uint32_t>> edges;

    vector<unique_ptr<SecretShareTuple>> edge_tuples;
    vector<unique_ptr<SecretShareTuple>> path_tuples;

    ServerSecretShares s1_vec;
    ServerSecretShares s2_vec;
    ServerSecretShares s3_vec;

    unordered_set<uint32_t> unique_nodes;

    int max_degree;
    int detect_length_k; 
    uint32_t id_range;
    unique_ptr<SecretShareTupleHelper> helper;

    GraphHolder(int max_degree_, int detect_length_k_, uint32_t id_range_)
        : max_degree(max_degree_), detect_length_k(detect_length_k_), id_range(id_range_) {
        helper =
            make_unique<SecretShareTupleHelper>(id_range, kTieBreakerRangeForGen);
    }

    void load_graph(string graph_file_name) {
        ifstream fs(graph_file_name);
        MYASSERT(fs.is_open());
        string line;
        uint32_t src, dst;
        uint32_t max_node_id = 0;
        while (getline(fs, line)) {
            stringstream ss(line);
            ss >> src >> dst;
            unique_nodes.insert(src);
            unique_nodes.insert(dst);
            MYASSERT(src <= id_range);
            MYASSERT(src > 0);
            MYASSERT(dst <= id_range);
            MYASSERT(dst > 0);
            max_node_id = max(max_node_id, src);
            max_node_id = max(max_node_id, dst);

            if (edges.find(src) == edges.end()) {
                edges[src] = vector<uint32_t>();
            }
            edges[src].push_back(dst);
        }
        MYASSERT(max_node_id < id_range);
        MYASSERT(unique_nodes.size() < id_range);
        MYASSERT(max_node_id == unique_nodes.size());

        for (const auto& node : unique_nodes) {
            if (edges.find(node) == edges.end()) {
                edges[node] = vector<uint32_t>();
            }
        }

        MYASSERT(edges.size() == unique_nodes.size());
        cout << "load graph finish\n";
    }

    void formatSecretShares() {
        for (auto it = edges.begin(); it != edges.end(); ++it) {
            uint32_t src = it->first;
            vector<uint32_t> src_edges = it->second;
            
            unique_ptr<SecretShareTuple> et = helper->formatEdgeTuple(
                src, src_edges, max_degree, detect_length_k);
            edge_tuples.push_back(move(et));
            
            for (const auto dst : src_edges) {
                unique_ptr<SecretShareTuple> pt = helper->formatPathTuple(
                    {src, dst}, max_degree, detect_length_k);
                path_tuples.push_back(move(pt));
            }
        }
        cout << "edge tuple count: " << edge_tuples.size() << endl;
        cout << "path tuple count: " << path_tuples.size() << endl;
        cout << "formatting finish" << endl;
    }

    void createSecretShares() {
        for (uint64_t i = 0; i < edge_tuples.size(); i++) {
            if (i % 100000 == 0) {
                cout << "formatting edge tuple: " << i << " "
                     << edge_tuples.size() << endl;
            }
            vector<unique_ptr<SecretShareTuple>> ss =
                helper->splitTuple(edge_tuples[i]);
            s1_vec.insert(ss[0]);
            s2_vec.insert(ss[1]);
            s3_vec.insert(ss[2]);
        }
        for (uint64_t i = 0; i < path_tuples.size(); i++) {
            if (i % 100000 == 0) {
                cout << "formatting path tuple: " << i << " "
                     << path_tuples.size() << endl;
            }
            vector<unique_ptr<SecretShareTuple>> ss =
                helper->splitTuple(path_tuples[i]);
            s1_vec.insert(ss[0]);
            s2_vec.insert(ss[1]);
            s3_vec.insert(ss[2]);
        }
        cout << "finish creating secret shares" << endl;
    }

    // ======================== [核心修改点：注入伴生元数据文件] ========================
    void dumpSecretShares(string file1, string file2, string file3) {
        // 保存 Server 1 密文
        string s1_str = s1_vec.serialize();
        ofstream out1(file1, ios::binary);
        MYASSERT(out1.is_open());
        out1 << s1_str;
        out1.close();
        s1_str.clear();

        // ⚡ 新增：写入第一份额的伴生元数据，记录 Edge 母本边界数（即 N = 节点数）
        ofstream out1_meta(file1 + "_meta");
        MYASSERT(out1_meta.is_open());
        out1_meta << edge_tuples.size() << "\n";
        out1_meta.close();

        // 保存 Server 2 密文
        string s2_str = s2_vec.serialize();
        ofstream out2(file2, ios::binary);
        MYASSERT(out2.is_open());
        out2 << s2_str;
        out2.close();
        s2_str.clear();

        // ⚡ 新增：写入第二份额伴生元数据
        ofstream out2_meta(file2 + "_meta");
        MYASSERT(out2_meta.is_open());
        out2_meta << edge_tuples.size() << "\n";
        out2_meta.close();

        // 保存 Server 3 密文
        string s3_str = s3_vec.serialize();
        ofstream out3(file3, ios::binary);
        MYASSERT(out3.is_open());
        out3 << s3_str;
        out3.close();
        s3_str.clear();

        // ⚡ 新增：写入第三份额伴生元数据
        ofstream out3_meta(file3 + "_meta");
        MYASSERT(out3_meta.is_open());
        out3_meta << edge_tuples.size() << "\n";
        out3_meta.close();

        cout << "finish dumping files and companion metadata files.\n";
    }

    void loadGraphAndDumpShares(string input, string outfile1, string outfile2,
                                string outfile3) {
        load_graph(input);
        formatSecretShares();
        createSecretShares();
        dumpSecretShares(outfile1, outfile2, outfile3);
    }
};

#endif // GRAPH_HOLDER_HPP
