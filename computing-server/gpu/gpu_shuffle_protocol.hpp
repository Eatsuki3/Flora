#ifndef FLORA_GPU_SHUFFLE_PROTOCOL_HPP
#define FLORA_GPU_SHUFFLE_PROTOCOL_HPP

#include "gpu_shuffle.cuh"
#include "tuple_bridge.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

class GpuShuffleProtocol {
public:
    GpuShuffleProtocol(uint32_t id_range, uint64_t tie_breaker_range)
        : id_range_(id_range), tie_breaker_range_(tie_breaker_range) {}

    static std::vector<uint64_t> makePermutation(uint64_t size, int seed) {

        std::vector<uint64_t> permutation(size);
        std::iota(permutation.begin(), permutation.end(), 0ULL);
        std::shuffle(permutation.begin(), permutation.end(), std::default_random_engine(seed));

        return permutation;
    }

    // x1 = permute12(A XOR B XOR z12), x2 = permute31(x1 XOR z31)
    // A = A_hat, B = B_hat
    std::unique_ptr<ServerSecretShares> s1Step1(
        ServerSecretShares& first, ServerSecretShares& second,
        int seed12, int seed31, int seedA, int seedB) {

        validatePair(first, second);

        const uint64_t size = first.size();
        const int max_degree = first.svec[0]->get_max_degree();
        const int detect_length = first.svec[0]->get_detect_length_k();

        DeviceTupleStore d_first;
        DeviceTupleStore d_second;
        DeviceTupleStore d_z12;
        DeviceTupleStore d_z31;
        DeviceTupleStore d_xor;
        DeviceTupleStore d_x1;
        DeviceTupleStore d_xor31;
        DeviceTupleStore d_x2;
        DeviceTupleStore d_a_hat;
        DeviceTupleStore d_b_hat;

        uploadTupleStore(runtime_, d_first, first);
        uploadTupleStore(runtime_, d_second, second);

        gpuGenerateRandomTupleStore(
            runtime_, d_z12, size, max_degree, detect_length,
            id_range_, tie_breaker_range_, static_cast<uint64_t>(seed12));
        
        // A XOR B XOR z12
        gpuShuffleXor3(runtime_, d_first, d_second, d_z12, d_xor);

        std::vector<uint64_t> permutation12 = makePermutation(size, seed12);

        // x1 = permute12(A XOR B XOR z12)
        gpuShufflePermute(runtime_, d_xor, d_x1, permutation12.data());

        gpuGenerateRandomTupleStore(
            runtime_, d_z31, size, max_degree, detect_length,
            id_range_, tie_breaker_range_, static_cast<uint64_t>(seed31));

        // x1 XOR z31
        gpuShuffleXor2(runtime_, d_x1, d_z31, d_xor31);
        std::vector<uint64_t> permutation31 = makePermutation(size, seed31);

        // x2 = permute31(x1 XOR z31)
        gpuShufflePermute(runtime_, d_xor31, d_x2, permutation31.data());
        
        // A = A_hat, B = B_hat
        gpuGenerateRandomTupleStore(
            runtime_, d_a_hat, size, max_degree, detect_length,
            id_range_, tie_breaker_range_, static_cast<uint64_t>(seedA));
        gpuGenerateRandomTupleStore(
            runtime_, d_b_hat, size, max_degree, detect_length,
            id_range_, tie_breaker_range_, static_cast<uint64_t>(seedB));

        runtime_.synchronize();

        downloadTupleStore(runtime_, d_a_hat, first);
        downloadTupleStore(runtime_, d_b_hat, second);
        auto x2 = std::make_unique<ServerSecretShares>();
        downloadTupleStore(runtime_, d_x2, *x2);

        return x2;
    }

    // y1 = permute12(C XOR z12)
    std::unique_ptr<ServerSecretShares> s2Step1(const ServerSecretShares& second, int seed12) {

        validateNotEmpty(second);

        const uint64_t size = second.size();
        const int max_degree = second.svec[0]->get_max_degree();
        const int detect_length = second.svec[0]->get_detect_length_k();

        DeviceTupleStore d_second;
        DeviceTupleStore d_z12;
        DeviceTupleStore d_xor;
        DeviceTupleStore d_y1;

        uploadTupleStore(runtime_, d_second, second);
        gpuGenerateRandomTupleStore(
            runtime_, d_z12, size, max_degree, detect_length,
            id_range_, tie_breaker_range_, static_cast<uint64_t>(seed12));

        gpuShuffleXor2(runtime_, d_second, d_z12, d_xor);
        std::vector<uint64_t> permutation12 = makePermutation(size, seed12);
        gpuShufflePermute(runtime_, d_xor, d_y1, permutation12.data());
        runtime_.synchronize();
        auto y1 = std::make_unique<ServerSecretShares>();
        downloadTupleStore(runtime_, d_y1, *y1);

        return y1;
    }

    // x3 = permute23(x2 XOR z23), B = B_hat, c1 = x3 XOR B_hat
    std::unique_ptr<ServerSecretShares> s2Step2(ServerSecretShares& first, const ServerSecretShares& x2, int seed23, int seedB) {

        validatePair(first, x2);
        const uint64_t size = x2.size();
        const int max_degree = x2.svec[0]->get_max_degree();
        const int detect_length = x2.svec[0]->get_detect_length_k();

        DeviceTupleStore d_x2;
        DeviceTupleStore d_z23;
        DeviceTupleStore d_xor;
        DeviceTupleStore d_x3;
        DeviceTupleStore d_b_hat;
        DeviceTupleStore d_c1;

        uploadTupleStore(runtime_, d_x2, x2);
        gpuGenerateRandomTupleStore(
            runtime_, d_z23, size, max_degree, detect_length,
            id_range_, tie_breaker_range_, static_cast<uint64_t>(seed23));
        gpuShuffleXor2(runtime_, d_x2, d_z23, d_xor);

        std::vector<uint64_t> permutation23 = makePermutation(size, seed23);
        gpuShufflePermute(runtime_, d_xor, d_x3, permutation23.data());

        gpuGenerateRandomTupleStore(
            runtime_, d_b_hat, size, max_degree, detect_length,
            id_range_, tie_breaker_range_, static_cast<uint64_t>(seedB));
        gpuShuffleXor2(runtime_, d_x3, d_b_hat, d_c1);
        runtime_.synchronize();

        downloadTupleStore(runtime_, d_b_hat, first);
        auto c1 = std::make_unique<ServerSecretShares>();
        downloadTupleStore(runtime_, d_c1, *c1);

        return c1;
    }

    // second = c1 XOR c2
    void s2Step3(ServerSecretShares& second, const ServerSecretShares& c1, const ServerSecretShares& c2) {
        xorToHost(c1, c2, second);
    }

    // y2 = permute31(y1 XOR z31), y3 = permute23(y2 XOR z23)
    // A = A_hat, c2 = y3 XOR A_hat
    std::unique_ptr<ServerSecretShares> s3Step1(ServerSecretShares& second, const ServerSecretShares& y1, int seed31, int seed23, int seedA) {

        validatePair(second, y1);

        const uint64_t size = y1.size();
        const int max_degree = y1.svec[0]->get_max_degree();
        const int detect_length = y1.svec[0]->get_detect_length_k();

        DeviceTupleStore d_y1;
        DeviceTupleStore d_z31;
        DeviceTupleStore d_y1_xor_z31;
        DeviceTupleStore d_y2;
        DeviceTupleStore d_z23;
        DeviceTupleStore d_y2_xor_z23;
        DeviceTupleStore d_y3;
        DeviceTupleStore d_a_hat;
        DeviceTupleStore d_c2;

        uploadTupleStore(runtime_, d_y1, y1);
        gpuGenerateRandomTupleStore(
            runtime_, d_z31, size, max_degree, detect_length,
            id_range_, tie_breaker_range_, static_cast<uint64_t>(seed31));
        gpuShuffleXor2(runtime_, d_y1, d_z31, d_y1_xor_z31);

        std::vector<uint64_t> permutation31 = makePermutation(size, seed31);
        gpuShufflePermute(runtime_, d_y1_xor_z31, d_y2, permutation31.data());
        gpuGenerateRandomTupleStore(
            runtime_, d_z23, size, max_degree, detect_length,
            id_range_, tie_breaker_range_, static_cast<uint64_t>(seed23));
        gpuShuffleXor2(runtime_, d_y2, d_z23, d_y2_xor_z23);

        std::vector<uint64_t> permutation23 = makePermutation(size, seed23);
        gpuShufflePermute(runtime_, d_y2_xor_z23, d_y3, permutation23.data());

        gpuGenerateRandomTupleStore(
            runtime_, d_a_hat, size, max_degree, detect_length,
            id_range_, tie_breaker_range_, static_cast<uint64_t>(seedA));
        gpuShuffleXor2(runtime_, d_y3, d_a_hat, d_c2);
        runtime_.synchronize();
        downloadTupleStore(runtime_, d_a_hat, second);
        auto c2 = std::make_unique<ServerSecretShares>();
        downloadTupleStore(runtime_, d_c2, *c2);

        return c2;
    }

    // first = c1 XOR c2
    void s3Step2(ServerSecretShares& first, const ServerSecretShares& c1, const ServerSecretShares& c2) {
        xorToHost(c1, c2, first);
    }

    // 3PC->2PC (S1)
    // S1: two_pc = first XOR second, S2: two_pc = second, S3: empty
    std::unique_ptr<ServerSecretShares> collapseS1ToTwoPc(const ServerSecretShares& first, const ServerSecretShares& second) {

        validatePair(first, second);
        auto output = std::make_unique<ServerSecretShares>();
        xorToHost(first, second, *output);
        return output;
    }

    // 2PC->3PC (S1)
    // random_share = GPU_Random(seed), first = two_pc XOR random_share, second = random_share
    void expandS1TwoPcToThreePc(const ServerSecretShares& two_pc, ServerSecretShares& first, ServerSecretShares& second, int seed) {

        using Clock = std::chrono::high_resolution_clock;
        const uint64_t size = two_pc.size();
        const int max_degree = two_pc.svec[0]->get_max_degree();
        const int detect_length = two_pc.svec[0]->get_detect_length_k();

        DeviceTupleStore d_two_pc;
        DeviceTupleStore d_random;
        DeviceTupleStore d_first;

        auto t0 = Clock::now();
        uploadTupleStore(runtime_, d_two_pc, two_pc);
        auto t1 = Clock::now();

        gpuGenerateRandomTupleStore(
            runtime_, d_random, size, max_degree, detect_length,
            id_range_, tie_breaker_range_, static_cast<uint64_t>(seed));
        gpuShuffleXor2(runtime_, d_two_pc, d_random, d_first);
        runtime_.synchronize();

        auto t2 = Clock::now();
        downloadTupleStore(runtime_, d_first, first);
        auto t3 = Clock::now();
        downloadTupleStore(runtime_, d_random, second);
        auto t4 = Clock::now();
        auto milliseconds = [](Clock::time_point begin, Clock::time_point end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };

        std::fprintf(
            stderr,
            "[2PC->3PC GPU] "
            "upload=%.3f ms "
            "kernel=%.3f ms "
            "download-first=%.3f ms "
            "download-second=%.3f ms "
            "total=%.3f ms\n",
            milliseconds(t0, t1),
            milliseconds(t1, t2),
            milliseconds(t2, t3),
            milliseconds(t3, t4),
            milliseconds(t0, t4));
    }

private:
    void xorToHost(const ServerSecretShares& left, const ServerSecretShares& right, ServerSecretShares& output) {

        validatePair(left, right);

        DeviceTupleStore d_left;
        DeviceTupleStore d_right;
        DeviceTupleStore d_output;

        uploadTupleStore(runtime_, d_left, left);
        uploadTupleStore(runtime_, d_right, right);

        gpuShuffleXor2(runtime_, d_left, d_right, d_output);
        runtime_.synchronize();
        downloadTupleStore(runtime_, d_output, output);
    }

    static void validateNotEmpty(const ServerSecretShares& store) {
        if (store.size() == 0) {
            std::fprintf(stderr, "GpuShuffleProtocol: empty store\n");
            std::abort();
        }
    }

    static void validatePair(const ServerSecretShares& left, const ServerSecretShares& right) {

        validateNotEmpty(left);
        validateNotEmpty(right);

        if (left.size() != right.size()) {
            std::fprintf(stderr,
                "GpuShuffleProtocol: size mismatch: "
                "%llu != %llu\n",
                static_cast<unsigned long long>(left.size()),
                static_cast<unsigned long long>(right.size()));
            std::abort();
        }

        if (left.svec[0]->get_max_degree() != right.svec[0]->get_max_degree() ||
            left.svec[0]->get_detect_length_k() != right.svec[0]->get_detect_length_k()) {

            std::fprintf(stderr, "GpuShuffleProtocol: tuple shape mismatch\n");
            std::abort();
        }
    }

private:
    uint32_t id_range_;
    uint64_t tie_breaker_range_;
    GpuRuntime runtime_;
};

#endif