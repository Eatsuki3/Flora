#include <unistd.h>
#include <cstdlib>
#include "graph_holder.hpp"

// id range
const uint32_t id_range_for_gen = (1 << 23);

int main(int argc, char* argv[]) {
    int max_degree = 200;
    int detect_length_k = 2; // defualt k=2
    string graph_file = "/home/ices/temp_test/flora-master/dataset-scripts/synthetic-graph_200.txt";
    string output1 = "out1_small_200";
    string output2 = "out2_small_200";
    string output3 = "out3_small_200";
    string out_dir = "./mydata/"; // outfile

    int opt;
    while ((opt = getopt(argc, argv, "d:i:o:k:")) != -1) {
        if (opt == 'd') {
            max_degree = atoi(optarg);
        } else if (opt == 'i') {
            graph_file = string(optarg);
        } else if (opt == 'o') {
            out_dir = string(optarg);
        } else if (opt == 'k') {
            detect_length_k = atoi(optarg);
        }
    }

    output1 = out_dir + output1;
    output2 = out_dir + output2;
    output3 = out_dir + output3;

    cout << "=========================================" << endl;
    cout << " Flora secret share generator" << endl;
    cout << "=========================================" << endl;
    cout << "input file path:  " << graph_file << endl;
    cout << "max degree (d):    " << max_degree << endl;
    cout << "detect length (k):    " << detect_length_k << endl;
    cout << "out file path:    " << out_dir << endl;
    cout << "=========================================" << endl;


    GraphHolder graph_holder(max_degree, detect_length_k, id_range_for_gen);
    graph_holder.loadGraphAndDumpShares(graph_file, output1, output2, output3);
    
    cout << "[OK] Flora secret shares generation finish!" << endl;
    return 0;
}