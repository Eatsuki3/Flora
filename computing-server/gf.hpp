// =============================================================================
//  gf.hpp - finite field GF(2^32) arithmetic
//
//  Usage: Used in Randomized Zero-Test (R ⊗ Z) and Beaver triples.
//
//  Requirements:
//    1. All parties (S1 / S2 online, S3 dealer) must include the same gf.hpp file,
//    using the *same irreducible polynomial* and reduction logic, otherwise multiplication results will be inconsistent -> zero-test fails.
//    2. Field addition = bitwise XOR. The 2-out-of-2 XOR share is directly the additive secret sharing in this field, no share conversion needed.
//    3. gf_mul(x, 0) == 0, gf_mul(x, 1) == x.
//
//  Definition of the field:
//    GF(2^32) = GF(2)[x] / (P(x)),  P(x) = x^32 + x^7 + x^3 + x^2 + 1
//    The low 32 bits of P(x) (without the implicit x^32 term) = 0b10001101 = 0x8D
// =============================================================================

#ifndef GF_HPP
#define GF_HPP

#include <cstdint>

static constexpr uint32_t GF32_REDUCTION_POLY = 0x8Du;

// -----------------------------------------------------------------------------
//  gf_mul: multiplication in GF(2^32) (carry-less multiply + polynomial reduction).
//  Step: (1) carry-less multiply, result fits in 63 bits;
//        (2) reduce the high 32 bits modulo P(x), fold back to 32 bits.
// -----------------------------------------------------------------------------
static inline uint32_t gf_mul(uint32_t x, uint32_t y) {
    uint64_t z = 0;
    uint64_t xx = static_cast<uint64_t>(x);
    for (int i = 0; i < 32; ++i) {
        uint64_t mask = static_cast<uint64_t>(0) - static_cast<uint64_t>((y >> i) & 1u);
        z ^= (xx << i) & mask;
    }

    for (int i = 62; i >= 32; --i) {
        if ((z >> i) & 1ull) {
            z ^= (1ull << i);
            z ^= (static_cast<uint64_t>(GF32_REDUCTION_POLY) << (i - 32));
        }
    }

    return static_cast<uint32_t>(z & 0xFFFFFFFFu);
}

// -----------------------------------------------------------------------------
//  gf_add: addition in GF(2^32) is simply bitwise XOR.
//  This is also the 2-out-of-2 additive secret sharing in this field.
// -----------------------------------------------------------------------------
static inline uint32_t gf_add(uint32_t x, uint32_t y) {
    return x ^ y;
}

// -----------------------------------------------------------------------------
//  gf_pow: fast exponentiation in GF(2^32), used for computing inverses or self-checks. x^e in GF(2^32).
// -----------------------------------------------------------------------------
static inline uint32_t gf_pow(uint32_t x, uint64_t e) {
    uint32_t result = 1u; // 乘法单位元
    uint32_t base = x;
    while (e > 0) {
        if (e & 1ull) result = gf_mul(result, base);
        base = gf_mul(base, base);
        e >>= 1;
    }
    return result;
}

// -----------------------------------------------------------------------------
//  gf_inv: multiplicative inverse in GF(2^32), x^(2^32 - 2) (Fermat's little theorem generalized to GF(2^32)).
//  Only for unit testing self-checks (e.g., verify gf_mul(x, gf_inv(x)) == 1), not used in the protocol hot path.
//  Note: x must be non-zero.
// -----------------------------------------------------------------------------
static inline uint32_t gf_inv(uint32_t x) {
    // The order of GF(2^n)* is 2^n - 1, so x^{-1} = x^{2^n - 2}.
    // For n = 32: 2^32 - 2 = 0xFFFFFFFE.
    return gf_pow(x, 0xFFFFFFFEull);
}

#endif  // GF_HPP