#ifndef FLORA_TUPLE_BRIDGE_HPP
#define FLORA_TUPLE_BRIDGE_HPP

#include "device_tuple_store.cuh"
#include "../secret-share/secret_share.hpp"

#include <cstdint>
#include <memory>
#include <vector>

struct HostTupleSoA {
    uint64_t size = 0;
    uint64_t field_count = 0;
    int max_degree = 0;
    int detect_length_k = 0;

    std::vector<uint32_t> node_id;
    std::vector<uint32_t> d_share;
    std::vector<uint64_t> tie_breaker;

    // field-major:
    // secret_vec[field * size + tuple]
    std::vector<uint32_t> secret_vec;
};

inline HostTupleSoA packTupleStore(
    const ServerSecretShares& source) {

    HostTupleSoA packed;
    packed.size = source.size();

    if (packed.size == 0) {
        return packed;
    }

    packed.max_degree = source.svec[0]->get_max_degree();
    packed.detect_length_k = source.svec[0]->get_detect_length_k();
    packed.field_count = static_cast<uint64_t>(packed.max_degree) * static_cast<uint64_t>(packed.detect_length_k + 1);

    packed.node_id.resize(packed.size);
    packed.d_share.resize(packed.size);
    packed.tie_breaker.resize(packed.size);
    packed.secret_vec.resize(packed.size * packed.field_count);

    for (uint64_t tuple_index = 0; tuple_index < packed.size; ++tuple_index) {
        const SecretShareTuple& tuple = *source.svec[tuple_index];
        MYASSERT(tuple.get_max_degree() == packed.max_degree);
        MYASSERT(tuple.get_detect_length_k() == packed.detect_length_k);

        packed.node_id[tuple_index] = tuple.get_node_id();
        packed.d_share[tuple_index] = tuple.get_d_share();
        packed.tie_breaker[tuple_index] = tuple.get_tie_breaker_id();

        for (uint64_t field = 0; field < packed.field_count; ++field) {
            packed.secret_vec[field * packed.size + tuple_index] = tuple.get_secret_share_vec_by_index(field);
        }
    }
    return packed;
}

inline void unpackTupleStore(const HostTupleSoA& packed, ServerSecretShares& destination) {

    destination.clear();

    for (uint64_t tuple_index = 0; tuple_index < packed.size; ++tuple_index) {

        auto tuple = std::make_unique<SecretShareTuple>(
                packed.node_id[tuple_index], packed.d_share[tuple_index],
                packed.max_degree, packed.detect_length_k);

        tuple->set_tie_breaker_id(packed.tie_breaker[tuple_index]);

        for (uint64_t field = 0; field < packed.field_count; ++field) {
            tuple->set_secret_share_vec_by_index(field, packed.secret_vec[field * packed.size + tuple_index]);
        }
        destination.insert(tuple);
    }
}

inline void uploadTupleStore(GpuRuntime& runtime, DeviceTupleStore& destination, const ServerSecretShares& source) {

    HostTupleSoA packed = packTupleStore(source);
    destination.configure(packed.size, packed.max_degree, packed.detect_length_k);
    destination.setSize(packed.size);

    if (packed.size == 0) { return; }
    cudaStream_t stream = runtime.transferStream();

    FLORA_CUDA_CHECK(cudaMemcpyAsync(destination.nodeId(), packed.node_id.data(),
        packed.size * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    FLORA_CUDA_CHECK(cudaMemcpyAsync(destination.dShare(), packed.d_share.data(),
        packed.size * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    FLORA_CUDA_CHECK(cudaMemcpyAsync(destination.tieBreaker(), packed.tie_breaker.data(),
        packed.size * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    FLORA_CUDA_CHECK(cudaMemcpyAsync(destination.secretVec(), packed.secret_vec.data(),
        packed.size * packed.field_count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
    FLORA_CUDA_CHECK(cudaStreamSynchronize(stream));
}

inline void downloadTupleStore(GpuRuntime& runtime, const DeviceTupleStore& source, ServerSecretShares& destination) {

    HostTupleSoA packed;

    packed.size = source.size();
    packed.field_count = source.fieldCount();
    packed.max_degree = source.maxDegree();
    packed.detect_length_k = source.detectLength();

    packed.node_id.resize(packed.size);
    packed.d_share.resize(packed.size);
    packed.tie_breaker.resize(packed.size);
    packed.secret_vec.resize(
        packed.size * packed.field_count);

    if (packed.size > 0) {
        cudaStream_t stream = runtime.transferStream();

        FLORA_CUDA_CHECK(cudaMemcpyAsync(packed.node_id.data(), source.nodeId(),
            packed.size * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        FLORA_CUDA_CHECK(cudaMemcpyAsync(packed.d_share.data(), source.dShare(),
            packed.size * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        FLORA_CUDA_CHECK(cudaMemcpyAsync(packed.tie_breaker.data(), source.tieBreaker(),
            packed.size * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        FLORA_CUDA_CHECK(cudaMemcpyAsync(packed.secret_vec.data(), source.secretVec(),
            packed.size * packed.field_count * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        FLORA_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    unpackTupleStore(packed, destination);
}

#endif