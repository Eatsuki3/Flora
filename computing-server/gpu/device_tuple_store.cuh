#ifndef FLORA_DEVICE_TUPLE_STORE_CUH
#define FLORA_DEVICE_TUPLE_STORE_CUH

#include "gpu_runtime.cuh"

#include <cstdint>
#include <utility>

class DeviceTupleStore {
public:
    DeviceTupleStore() = default;

    void configure(uint64_t capacity, int max_degree, int detect_length_k) {

        const uint64_t fields = static_cast<uint64_t>(max_degree) * static_cast<uint64_t>(detect_length_k + 1);

        node_id_.reserve(capacity);
        d_share_.reserve(capacity);
        tie_breaker_.reserve(capacity);
        secret_vec_.reserve(capacity * fields);

        capacity_ = capacity;
        field_count_ = fields;
        max_degree_ = max_degree;
        detect_length_k_ = detect_length_k;

        if (size_ > capacity_) {
            size_ = capacity_;
        }
    }

    void setSize(uint64_t size) {
        if (size > capacity_) {
            std::fprintf(stderr, "DeviceTupleStore size exceeds capacity\n");
            std::abort();
        }

        size_ = size;
    }

    uint32_t* nodeId() {
        return node_id_.data();
    }

    const uint32_t* nodeId() const {
        return node_id_.data();
    }

    uint32_t* dShare() {
        return d_share_.data();
    }

    const uint32_t* dShare() const {
        return d_share_.data();
    }

    uint64_t* tieBreaker() {
        return tie_breaker_.data();
    }

    const uint64_t* tieBreaker() const {
        return tie_breaker_.data();
    }

    uint32_t* secretVec() {
        return secret_vec_.data();
    }

    const uint32_t* secretVec() const {
        return secret_vec_.data();
    }

    uint64_t size() const {
        return size_;
    }

    uint64_t capacity() const {
        return capacity_;
    }

    uint64_t fieldCount() const {
        return field_count_;
    }

    int maxDegree() const {
        return max_degree_;
    }

    int detectLength() const {
        return detect_length_k_;
    }

    void swap(DeviceTupleStore& other) noexcept {
        node_id_.swap(other.node_id_);
        d_share_.swap(other.d_share_);
        tie_breaker_.swap(other.tie_breaker_);
        secret_vec_.swap(other.secret_vec_);

        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        std::swap(field_count_, other.field_count_);
        std::swap(max_degree_, other.max_degree_);
        std::swap(detect_length_k_, other.detect_length_k_);
    }

private:
    DeviceBuffer<uint32_t> node_id_;
    DeviceBuffer<uint32_t> d_share_;
    DeviceBuffer<uint64_t> tie_breaker_;
    DeviceBuffer<uint32_t> secret_vec_;

    uint64_t size_ = 0;
    uint64_t capacity_ = 0;
    uint64_t field_count_ = 0;
    int max_degree_ = 0;
    int detect_length_k_ = 0;
};

#endif