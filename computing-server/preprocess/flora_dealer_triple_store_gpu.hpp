#ifndef FLORA_DEALER_TRIPLE_STORE_HPP
#define FLORA_DEALER_TRIPLE_STORE_HPP

// USED FOR TEST
#define FLORA_ALLOW_TRIPLE_REWIND

#include <cstdint>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

#include "flora_preprocess_raw.hpp"

namespace flora::preprocess {

class DealerTripleStore {
private:
    std::ifstream sort_s0_;
    std::ifstream sort_s1_;

    std::ifstream neighbor_s0_;
    std::ifstream neighbor_s1_;

    std::ifstream filter_s0_;
    std::ifstream filter_s1_;

    std::mutex sort_mutex_;
    std::mutex neighbor_mutex_;
    std::mutex filter_mutex_;

    uint64_t sort_consumed_ = 0;
    uint64_t neighbor_consumed_ = 0;
    uint64_t filter_consumed_ = 0;

    static void openFile(
            std::ifstream& input,
            const std::string& path) {

        input.open(path, std::ios::binary);

        if (!input.is_open()) {
            throw std::runtime_error(
                "Cannot open triple file: " + path);
        }
    }

    template<typename Record>
    static void readRecords(
            std::ifstream& input,
            uint64_t count,
            std::vector<Record>& output,
            const char* description) {

        if (count > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(Record))) {
            throw std::runtime_error(
                std::string("Triple request too large: ") +
                description);
        }

        output.resize(count);
        if (count == 0) { return; }

        const uint64_t total_bytes = count * sizeof(Record);

        uint64_t completed_bytes = 0;
        bool rewound_without_progress = false;

        while (completed_bytes < total_bytes) {
            const uint64_t remaining_bytes = total_bytes - completed_bytes;

            input.read(
                reinterpret_cast<char*>(output.data()) + completed_bytes,
                static_cast<std::streamsize>(remaining_bytes));

            const std::streamsize got = input.gcount();

            if (got > 0) {
                completed_bytes += static_cast<uint64_t>(got);
                rewound_without_progress = false;
            }

            if (completed_bytes == total_bytes) {
                break;
            }

    #ifdef FLORA_ALLOW_TRIPLE_REWIND
            input.clear();
            input.seekg(0, std::ios::beg);

            if (!input.good()) {
                throw std::runtime_error(
                    std::string("Failed to rewind triple file: ") +
                    description);
            }

            if (got == 0) {
                if (rewound_without_progress) {
                    throw std::runtime_error(
                        std::string("Triple file is empty: ") +
                        description);
                }

                rewound_without_progress = true;
            }
            
    #else
            throw std::runtime_error(
                std::string("Triple file exhausted: ") +
                description +
                ", requested_bytes=" +
                std::to_string(total_bytes) +
                ", completed_bytes=" +
                std::to_string(completed_bytes));
    #endif
        }
    }

public:
    DealerTripleStore(
            const std::string& sort_s0_path,
            const std::string& sort_s1_path,
            const std::string& neighbor_s0_path,
            const std::string& neighbor_s1_path,
            const std::string& filter_s0_path,
            const std::string& filter_s1_path) {

        openFile(sort_s0_, sort_s0_path);
        openFile(sort_s1_, sort_s1_path);
        openFile(neighbor_s0_, neighbor_s0_path);
        openFile(neighbor_s1_, neighbor_s1_path);
        openFile(filter_s0_, filter_s0_path);
        openFile(filter_s1_, filter_s1_path);
    }

    void takeSort(
        uint64_t count,
        std::vector<BooleanTripleRecord>& s0,
        std::vector<BooleanTripleRecord>& s1) {

        std::lock_guard<std::mutex> lock(sort_mutex_);

        readRecords(sort_s0_, count, s0, "sort S0");
        readRecords(sort_s1_, count, s1, "sort S1");
        sort_consumed_ += count;
    }

    void takeNeighbor(
            uint64_t count,
            std::vector<BooleanTripleRecord>& s0,
            std::vector<BooleanTripleRecord>& s1) {

        std::lock_guard<std::mutex> lock(neighbor_mutex_);

        readRecords(neighbor_s0_, count, s0, "neighbor S0");
        readRecords(neighbor_s1_, count, s1, "neighbor S1");
        neighbor_consumed_ += count;
    }

    void takeFilter(
            uint64_t count,
            std::vector<GfZeroTestRecord>& s0,
            std::vector<GfZeroTestRecord>& s1) {

        std::lock_guard<std::mutex> lock(filter_mutex_);

        readRecords(filter_s0_, count, s0, "filter S0");
        readRecords(filter_s1_, count, s1, "filter S1");
        filter_consumed_ += count;
    }

    uint64_t sortConsumed() const {
        return sort_consumed_;
    }

    uint64_t neighborConsumed() const {
        return neighbor_consumed_;
    }

    uint64_t filterConsumed() const {
        return filter_consumed_;
    }
};

} // namespace flora::preprocess

#endif // FLORA_DEALER_TRIPLE_STORE_HPP