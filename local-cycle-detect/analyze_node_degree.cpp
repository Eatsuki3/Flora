#include <unistd.h>

#include "graphstore.hpp"

int main(int argc, char* argv[]) {
    string graph_file =
        "/home/ices/temp_test/flora-master/dataset-scripts/synthetic-graph.txt";

    int opt;
    while ((opt = getopt(argc, argv, "i:")) != -1) {
        if (opt == 'i') {
            graph_file = string(optarg);
        }
    }

    // std::string outfile = "output.txt";
    // std::ofstream fout(outfile);
    GraphStore gs(graph_file);

    gs.load_graph();
    gs.countNodeDegrees();
    return 0;
}