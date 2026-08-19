#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "../gf.hpp"
#include "flora_preprocess_raw.hpp"

using flora::preprocess::BooleanTripleRecord;
using flora::preprocess::GfZeroTestRecord;

namespace {

std::mt19937_64 makeGenerator() {
    std::random_device rd;

    std::seed_seq seed{
        rd(), rd(), rd(), rd(),
        rd(), rd(), rd(), rd()
    };

    return std::mt19937_64(seed);
}

uint32_t randomU32(std::mt19937_64& rng) {
    return static_cast<uint32_t>(rng());
}

void requireOutput(const std::ofstream& output, const std::string& path) {

    if (!output) {
        throw std::runtime_error("Failed to write file: " + path);
    }
}

// void generateNeighborTriples(uint64_t item_count, const std::string& s0_path, const std::string& s1_path, std::mt19937_64& rng) {
//     std::ofstream s0(s0_path, std::ios::binary | std::ios::trunc);
//     std::ofstream s1(s1_path, std::ios::binary | std::ios::trunc);

//     if (!s0.is_open() || !s1.is_open()) {
//         throw std::runtime_error("Cannot create Neighbor triple files");
//     }

//     for (uint64_t i = 0; i < item_count; ++i) {
//         uint32_t a = randomU32(rng);
//         uint32_t b = randomU32(rng);
//         uint32_t c = a & b;

//         uint32_t a0 = randomU32(rng);
//         uint32_t b0 = randomU32(rng);
//         uint32_t c0 = randomU32(rng);

//         BooleanTripleRecord record_s0{
//             a0,
//             b0,
//             c0
//         };

//         BooleanTripleRecord record_s1{
//             a ^ a0,
//             b ^ b0,
//             c ^ c0
//         };

//         s0.write(reinterpret_cast<const char*>(&record_s0),
//             sizeof(record_s0));

//         s1.write(reinterpret_cast<const char*>(&record_s1),
//             sizeof(record_s1));
//     }

//     requireOutput(s0, s0_path);
//     requireOutput(s1, s1_path);

//     std::cout
//         << "[PREPROCESS] Neighbor triples generated: "
//         << item_count << '\n'
//         << "  S0 shares: " << s0_path << '\n'
//         << "  S1 shares: " << s1_path << std::endl;
// }


void generateBooleanTriples(
    uint64_t item_count,
    const std::string& s0_path,
    const std::string& s1_path,
    const char* description,
    std::mt19937_64& rng) {

    std::ofstream s0(
        s0_path,
        std::ios::binary | std::ios::trunc);

    std::ofstream s1(
        s1_path,
        std::ios::binary | std::ios::trunc);

    if (!s0.is_open() || !s1.is_open()) {
        throw std::runtime_error(
            std::string("Cannot create ") +
            description +
            " triple files");
    }

    for (uint64_t i = 0; i < item_count; ++i) {
        const uint32_t a = randomU32(rng);
        const uint32_t b = randomU32(rng);
        const uint32_t c = a & b;

        const uint32_t a0 = randomU32(rng);
        const uint32_t b0 = randomU32(rng);
        const uint32_t c0 = randomU32(rng);

        const BooleanTripleRecord record_s0{
            a0,
            b0,
            c0
        };

        const BooleanTripleRecord record_s1{
            a ^ a0,
            b ^ b0,
            c ^ c0
        };

        s0.write(
            reinterpret_cast<const char*>(&record_s0),
            sizeof(record_s0));

        s1.write(
            reinterpret_cast<const char*>(&record_s1),
            sizeof(record_s1));
    }

    requireOutput(s0, s0_path);
    requireOutput(s1, s1_path);

    std::cout
        << "[PREPROCESS] "
        << description
        << " Boolean triples generated: "
        << item_count << '\n'
        << "  S0 shares: " << s0_path << '\n'
        << "  S1 shares: " << s1_path << '\n';
}


void generateFilterTriples(
        uint64_t item_count,
        const std::string& s0_path,
        const std::string& s1_path,
        std::mt19937_64& rng) {

    std::ofstream s0(
        s0_path,
        std::ios::binary | std::ios::trunc);

    std::ofstream s1(
        s1_path,
        std::ios::binary | std::ios::trunc);

    if (!s0.is_open() || !s1.is_open()) {
        throw std::runtime_error(
            "Cannot create Filter triple files");
    }

    for (uint64_t i = 0; i < item_count; ++i) {
        uint32_t r = 0;

        while (r == 0) {
            r = randomU32(rng);
        }

        uint32_t b = randomU32(rng);
        uint32_t c = gf_mul(r, b);

        uint32_t r0 = randomU32(rng);
        uint32_t b0 = randomU32(rng);
        uint32_t c0 = randomU32(rng);

        GfZeroTestRecord record_s0{
            r0,
            b0,
            c0
        };

        GfZeroTestRecord record_s1{
            r ^ r0,
            b ^ b0,
            c ^ c0
        };

        s0.write(
            reinterpret_cast<const char*>(&record_s0),
            sizeof(record_s0));

        s1.write(
            reinterpret_cast<const char*>(&record_s1),
            sizeof(record_s1));
    }

    requireOutput(s0, s0_path);
    requireOutput(s1, s1_path);

    std::cout
        << "[PREPROCESS] Filter correlations generated: "
        << item_count << '\n'
        << "  S0 shares: " << s0_path << '\n'
        << "  S1 shares: " << s1_path << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 10) {
        std::cerr
            << "Usage:\n  "
            << argv[0]
            << " <sort_count> <neighbor_count> <filter_count>\n"
            << "    <sort_s0.bin>     <sort_s1.bin>\n"
            << "    <neighbor_s0.bin> <neighbor_s1.bin>\n"
            << "    <filter_s0.bin>   <filter_s1.bin>\n";

        return 1;
    }

    try {
        const uint64_t sort_count = std::stoull(argv[1]);
        const uint64_t neighbor_count = std::stoull(argv[2]);
        const uint64_t filter_count = std::stoull(argv[3]);

        std::mt19937_64 rng = makeGenerator();

        generateBooleanTriples(
            sort_count,
            argv[4],
            argv[5],
            "Sort",
            rng);

        generateBooleanTriples(
            neighbor_count,
            argv[6],
            argv[7],
            "Neighbor",
            rng);

        generateFilterTriples(
            filter_count,
            argv[8],
            argv[9],
            rng);

    } catch (const std::exception& exception) {
        std::cerr
            << "[PREPROCESS ERROR] "
            << exception.what()
            << std::endl;

        return 1;
    }

    return 0;
}