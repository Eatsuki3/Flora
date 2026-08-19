#ifndef FLORA_PREPROCESS_RAW_HPP
#define FLORA_PREPROCESS_RAW_HPP

#include <cstdint>
#include <limits>

namespace flora{
namespace preprocess {

// S0/S1 send this flag for not request triples。
static constexpr uint64_t kFinishRequest = std::numeric_limits<uint64_t>::max();

// Neighbor passing & Sort: packed Boolean Beaver triple。
// c = a & b.
#pragma pack(push, 1)
struct BooleanTripleRecord {
    uint32_t a_share;
    uint32_t b_share;
    uint32_t c_share;
};
#pragma pack(pop)

static_assert(
    sizeof(BooleanTripleRecord) == 12,
    "BooleanTripleRecord must be 12 bytes");

// Filter: EQZ correlation。
// r != 0，c = gf_mul(r, b)。
#pragma pack(push, 1)
struct GfZeroTestRecord {
    uint32_t r_share;
    uint32_t b_share;
    uint32_t c_share;
};
#pragma pack(pop)

static_assert(
    sizeof(GfZeroTestRecord) == 12,
    "GfZeroTestRecord must be 12 bytes");

} // namespace preprocess
} // namespace flora

#endif // FLORA_PREPROCESS_RAW_HPP