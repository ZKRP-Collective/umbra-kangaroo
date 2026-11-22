#define _POSIX_C_SOURCE 200809L

#include "rck_portable.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if !defined(__x86_64__) && !defined(_M_X64)
#error "the IFMA microbenchmark is x86-64 only"
#endif

#include <immintrin.h>

#define RCK_IFMA_LANES 8U
#define RCK_IFMA_DUAL_BLOCKS 16U
#define RCK_IFMA_LIMBS 5U
#define RCK_IFMA_PRODUCT_LIMBS 10U
#define RCK_RADIX52_MASK UINT64_C(0x000fffffffffffff)
#define RCK_RADIX52_TOP_MASK UINT64_C(0x0000ffffffffffff)
#define RCK_SECP52_P0 UINT64_C(0x000ffffefffffc2f)
#define RCK_SECP52_PN UINT64_C(0x000fffffffffffff)
#define RCK_SECP52_P4 UINT64_C(0x0000ffffffffffff)
#define RCK_SECP_FOLD UINT64_C(0x00000001000003d1)

static uint64_t rng_state = UINT64_C(0x6a09e667f3bcc909);
static volatile uint64_t sink;

typedef struct {
    uint64_t point_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t point_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t jump_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t jump_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t inverse_dx[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    rck_point scalar_points[RCK_IFMA_LANES];
    rck_fe scalar_inverses[RCK_IFMA_LANES];
} ifma_dual_block;

static void fe_from52(
    rck_fe* out,
    const uint64_t value[RCK_IFMA_LIMBS]
);
static void fe_to52(
    uint64_t out[RCK_IFMA_LIMBS],
    const rck_fe* value
);

static uint64_t rng_next(void) {
    uint64_t value;
    rng_state += UINT64_C(0x9e3779b97f4a7c15);
    value = rng_state;
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static double seconds_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void scalar_mul52(
    uint64_t out[RCK_IFMA_PRODUCT_LIMBS],
    const uint64_t left[RCK_IFMA_LIMBS],
    const uint64_t right[RCK_IFMA_LIMBS]
) {
    __uint128_t accumulated[RCK_IFMA_PRODUCT_LIMBS] = {0};
    size_t i;
    size_t j;
    for (i = 0; i < RCK_IFMA_LIMBS; ++i) {
        for (j = 0; j < RCK_IFMA_LIMBS; ++j) {
            accumulated[i + j] +=
                (__uint128_t)left[i] * right[j];
        }
    }
    for (i = 0; i + 1 < RCK_IFMA_PRODUCT_LIMBS; ++i) {
        accumulated[i + 1] += accumulated[i] >> 52;
        out[i] = (uint64_t)accumulated[i] & RCK_RADIX52_MASK;
    }
    out[RCK_IFMA_PRODUCT_LIMBS - 1] =
        (uint64_t)accumulated[RCK_IFMA_PRODUCT_LIMBS - 1];
}

__attribute__((target("avx512f,avx512ifma")))
static void ifma_mul52(
    uint64_t out[RCK_IFMA_PRODUCT_LIMBS][RCK_IFMA_LANES],
    uint64_t left[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t right[RCK_IFMA_LIMBS][RCK_IFMA_LANES]
) {
    __m512i accumulated[RCK_IFMA_PRODUCT_LIMBS];
    const __m512i mask = _mm512_set1_epi64((long long)RCK_RADIX52_MASK);
    size_t i;
    size_t j;

    for (i = 0; i < RCK_IFMA_PRODUCT_LIMBS; ++i) {
        accumulated[i] = _mm512_setzero_si512();
    }
    for (i = 0; i < RCK_IFMA_LIMBS; ++i) {
        const __m512i a = _mm512_loadu_si512((const void*)left[i]);
        for (j = 0; j < RCK_IFMA_LIMBS; ++j) {
            const __m512i b =
                _mm512_loadu_si512((const void*)right[j]);
            accumulated[i + j] =
                _mm512_madd52lo_epu64(accumulated[i + j], a, b);
            accumulated[i + j + 1] =
                _mm512_madd52hi_epu64(accumulated[i + j + 1], a, b);
        }
    }
    for (i = 0; i + 1 < RCK_IFMA_PRODUCT_LIMBS; ++i) {
        const __m512i carry = _mm512_srli_epi64(accumulated[i], 52);
        accumulated[i] = _mm512_and_si512(accumulated[i], mask);
        accumulated[i + 1] = _mm512_add_epi64(accumulated[i + 1], carry);
    }
    for (i = 0; i < RCK_IFMA_PRODUCT_LIMBS; ++i) {
        _mm512_storeu_si512((void*)out[i], accumulated[i]);
    }
}

__attribute__((target("avx512f,avx512ifma")))
static void normalize52(__m512i value[6]) {
    const __m512i mask = _mm512_set1_epi64((long long)RCK_RADIX52_MASK);
    size_t index;
    for (index = 0; index < 5; ++index) {
        const __m512i carry = _mm512_srli_epi64(value[index], 52);
        value[index] = _mm512_and_si512(value[index], mask);
        value[index + 1] = _mm512_add_epi64(value[index + 1], carry);
    }
}

__attribute__((target("avx512f,avx512ifma")))
static void add_folded_high(
    __m512i out[6],
    const __m512i* high,
    size_t count
) {
    const __m512i fold = _mm512_set1_epi64((long long)RCK_SECP_FOLD);
    size_t index;
    for (index = 0; index < count; ++index) {
        out[index] =
            _mm512_madd52lo_epu64(out[index], high[index], fold);
        out[index + 1] =
            _mm512_madd52hi_epu64(out[index + 1], high[index], fold);
    }
    normalize52(out);
}

__attribute__((target("avx512f,avx512ifma")))
static void fold_256_from_six(__m512i out[6], __m512i input[6]) {
    const __m512i mask = _mm512_set1_epi64((long long)RCK_RADIX52_MASK);
    const __m512i top_mask =
        _mm512_set1_epi64((long long)RCK_RADIX52_TOP_MASK);
    __m512i high[2];
    size_t index;
    for (index = 0; index < 4; ++index) {
        out[index] = input[index];
    }
    out[4] = _mm512_and_si512(input[4], top_mask);
    out[5] = _mm512_setzero_si512();
    high[0] = _mm512_or_si512(
        _mm512_srli_epi64(input[4], 48),
        _mm512_and_si512(_mm512_slli_epi64(input[5], 4), mask)
    );
    high[1] = _mm512_srli_epi64(input[5], 48);
    add_folded_high(out, high, 2);
}

__attribute__((target("avx512f,avx512ifma")))
static void canonicalize52(__m512i value[6]) {
    static const uint64_t modulus[RCK_IFMA_LIMBS] = {
        RCK_SECP52_P0, RCK_SECP52_PN, RCK_SECP52_PN,
        RCK_SECP52_PN, RCK_SECP52_P4
    };
    const __m512i zero = _mm512_setzero_si512();
    __mmask8 equal = UINT8_C(0xff);
    __mmask8 greater = 0;
    __mmask8 borrow = 0;
    __mmask8 subtract;
    __m512i difference[RCK_IFMA_LIMBS];
    int index;

    for (index = (int)RCK_IFMA_LIMBS - 1; index >= 0; --index) {
        const __m512i p = _mm512_set1_epi64((long long)modulus[index]);
        const __mmask8 gt = _mm512_cmp_epu64_mask(
            value[index], p, _MM_CMPINT_GT
        );
        const __mmask8 eq = _mm512_cmp_epu64_mask(
            value[index], p, _MM_CMPINT_EQ
        );
        greater |= equal & gt;
        equal &= eq;
    }
    subtract = greater | equal |
        _mm512_cmp_epu64_mask(value[5], zero, _MM_CMPINT_NE);
    for (index = 0; index < (int)RCK_IFMA_LIMBS; ++index) {
        const __m512i p = _mm512_set1_epi64((long long)modulus[index]);
        const __m512i first = _mm512_sub_epi64(value[index], p);
        const __m512i borrow_value = _mm512_maskz_set1_epi64(borrow, 1);
        const __mmask8 first_borrow = _mm512_cmp_epu64_mask(
            value[index], p, _MM_CMPINT_LT
        );
        const __mmask8 second_borrow = _mm512_cmp_epu64_mask(
            first, borrow_value, _MM_CMPINT_LT
        );
        difference[index] = _mm512_sub_epi64(first, borrow_value);
        borrow = first_borrow | second_borrow;
    }
    for (index = 0; index < (int)RCK_IFMA_LIMBS; ++index) {
        value[index] = _mm512_mask_blend_epi64(
            subtract, value[index], difference[index]
        );
    }
}

__attribute__((target("avx512f,avx512ifma")))
static void ifma_reduce52(
    uint64_t out[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t product[RCK_IFMA_PRODUCT_LIMBS][RCK_IFMA_LANES]
) {
    const __m512i mask = _mm512_set1_epi64((long long)RCK_RADIX52_MASK);
    const __m512i top_mask =
        _mm512_set1_epi64((long long)RCK_RADIX52_TOP_MASK);
    __m512i input[RCK_IFMA_PRODUCT_LIMBS];
    __m512i high[RCK_IFMA_LIMBS];
    __m512i first[6];
    __m512i second[6];
    size_t index;

    for (index = 0; index < RCK_IFMA_PRODUCT_LIMBS; ++index) {
        input[index] = _mm512_loadu_si512((const void*)product[index]);
    }
    for (index = 0; index < 4; ++index) {
        first[index] = input[index];
    }
    first[4] = _mm512_and_si512(input[4], top_mask);
    first[5] = _mm512_setzero_si512();
    for (index = 0; index < RCK_IFMA_LIMBS; ++index) {
        high[index] = _mm512_or_si512(
            _mm512_srli_epi64(input[index + 4], 48),
            _mm512_and_si512(
                _mm512_slli_epi64(input[index + 5], 4), mask
            )
        );
    }
    add_folded_high(first, high, RCK_IFMA_LIMBS);
    fold_256_from_six(second, first);
    fold_256_from_six(first, second);
    canonicalize52(first);
    for (index = 0; index < RCK_IFMA_LIMBS; ++index) {
        _mm512_storeu_si512((void*)out[index], first[index]);
    }
}

__attribute__((target("avx512f,avx512ifma")))
static void ifma_field_mul52(
    uint64_t out[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t left[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t right[RCK_IFMA_LIMBS][RCK_IFMA_LANES]
) {
    uint64_t product[RCK_IFMA_PRODUCT_LIMBS][RCK_IFMA_LANES];
    ifma_mul52(product, left, right);
    ifma_reduce52(out, product);
}

__attribute__((target("avx512f,avx512ifma")))
static void ifma_field_sub52(
    uint64_t out[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t left[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t right[RCK_IFMA_LIMBS][RCK_IFMA_LANES]
) {
    const __m512i mask = _mm512_set1_epi64((long long)RCK_RADIX52_MASK);
    __m512i value[RCK_IFMA_LIMBS];
    __mmask8 borrow = 0;
    size_t index;
    for (index = 0; index < RCK_IFMA_LIMBS; ++index) {
        const __m512i a = _mm512_loadu_si512((const void*)left[index]);
        const __m512i b = _mm512_loadu_si512((const void*)right[index]);
        const __m512i first = _mm512_sub_epi64(a, b);
        const __m512i borrow_value = _mm512_maskz_set1_epi64(borrow, 1);
        const __mmask8 first_borrow = _mm512_cmp_epu64_mask(
            a, b, _MM_CMPINT_LT
        );
        const __mmask8 second_borrow = _mm512_cmp_epu64_mask(
            first, borrow_value, _MM_CMPINT_LT
        );
        value[index] = _mm512_and_si512(
            _mm512_sub_epi64(first, borrow_value), mask
        );
        borrow = first_borrow | second_borrow;
    }
    {
        __mmask8 correction_borrow = 0;
        for (index = 0; index < RCK_IFMA_LIMBS; ++index) {
            const uint64_t correction_digit = index == 0 ? RCK_SECP_FOLD :
                (index + 1 == RCK_IFMA_LIMBS ?
                 UINT64_C(0x000f000000000000) : 0);
            const __m512i correction = _mm512_maskz_set1_epi64(
                borrow, (long long)correction_digit
            );
            const __m512i first = _mm512_sub_epi64(value[index], correction);
            const __m512i borrow_value =
                _mm512_maskz_set1_epi64(correction_borrow, 1);
            const __mmask8 first_borrow = _mm512_cmp_epu64_mask(
                value[index], correction, _MM_CMPINT_LT
            );
            const __mmask8 second_borrow = _mm512_cmp_epu64_mask(
                first, borrow_value, _MM_CMPINT_LT
            );
            value[index] = _mm512_and_si512(
                _mm512_sub_epi64(first, borrow_value), mask
            );
            correction_borrow = first_borrow | second_borrow;
        }
    }
    for (index = 0; index < RCK_IFMA_LIMBS; ++index) {
        _mm512_storeu_si512((void*)out[index], value[index]);
    }
}

__attribute__((target("avx512f,avx512ifma")))
static void ifma_field_neg52(
    uint64_t out[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t value[RCK_IFMA_LIMBS][RCK_IFMA_LANES]
) {
    static const uint64_t modulus[RCK_IFMA_LIMBS] = {
        RCK_SECP52_P0, RCK_SECP52_PN, RCK_SECP52_PN,
        RCK_SECP52_PN, RCK_SECP52_P4
    };
    uint64_t p[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    __mmask8 zero = UINT8_C(0xff);
    size_t index;
    size_t lane;
    for (index = 0; index < RCK_IFMA_LIMBS; ++index) {
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            p[index][lane] = modulus[index];
            if (value[index][lane] != 0) {
                zero &= (__mmask8)~(UINT8_C(1) << lane);
            }
        }
    }
    ifma_field_sub52(out, p, value);
    for (index = 0; index < RCK_IFMA_LIMBS; ++index) {
        const __m512i result = _mm512_loadu_si512((const void*)out[index]);
        _mm512_storeu_si512(
            (void*)out[index],
            _mm512_mask_blend_epi64(zero, result, _mm512_setzero_si512())
        );
    }
}

__attribute__((target("avx512f,avx512ifma")))
static void ifma_dual_baseline52(
    uint64_t out_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t out_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t point_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t point_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t jump_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t jump_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t inverse_dx[RCK_IFMA_LIMBS][RCK_IFMA_LANES]
) {
    uint64_t signed_jump_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t negated_jump_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t dy[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t slope[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t temporary[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    size_t limb;
    size_t lane;

    ifma_field_neg52(negated_jump_y, jump_y);
    for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            signed_jump_y[limb][lane] = point_y[0][lane] & 1U ?
                negated_jump_y[limb][lane] : jump_y[limb][lane];
        }
    }
    ifma_field_sub52(dy, point_y, signed_jump_y);
    ifma_field_mul52(slope, dy, inverse_dx);
    ifma_field_mul52(out_x, slope, slope);
    ifma_field_sub52(out_x, out_x, point_x);
    ifma_field_sub52(out_x, out_x, jump_x);
    ifma_field_sub52(temporary, point_x, out_x);
    ifma_field_mul52(out_y, slope, temporary);
    ifma_field_sub52(out_y, out_y, point_y);
}

__attribute__((target("avx512f,avx512ifma")))
static void ifma_candidate_x52(
    uint64_t out_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t out_slope[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t point_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t point_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t signed_jump_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t jump_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t inverse_dx[RCK_IFMA_LIMBS][RCK_IFMA_LANES]
) {
    uint64_t dy[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    ifma_field_sub52(dy, point_y, signed_jump_y);
    ifma_field_mul52(out_slope, dy, inverse_dx);
    ifma_field_mul52(out_x, out_slope, out_slope);
    ifma_field_sub52(out_x, out_x, point_x);
    ifma_field_sub52(out_x, out_x, jump_x);
}

__attribute__((target("avx512f,avx512ifma")))
static void ifma_dual_full52(
    uint64_t out_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t out_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t point_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t point_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t jump_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t jump_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES],
    uint64_t inverse_dx[RCK_IFMA_LIMBS][RCK_IFMA_LANES]
) {
    uint64_t negative_jump_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t primary_jump_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t alternate_jump_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t primary_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t alternate_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t primary_slope[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t alternate_slope[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t selected_slope[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t temporary[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    size_t limb;
    size_t lane;

    ifma_field_neg52(negative_jump_y, jump_y);
    for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            const bool odd = (point_y[0][lane] & 1U) != 0;
            primary_jump_y[limb][lane] = odd ?
                negative_jump_y[limb][lane] : jump_y[limb][lane];
            alternate_jump_y[limb][lane] = odd ?
                jump_y[limb][lane] : negative_jump_y[limb][lane];
        }
    }
    ifma_candidate_x52(
        primary_x, primary_slope, point_x, point_y, primary_jump_y,
        jump_x, inverse_dx
    );
    ifma_candidate_x52(
        alternate_x, alternate_slope, point_x, point_y, alternate_jump_y,
        jump_x, inverse_dx
    );
    for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            const bool choose_alternate =
                (alternate_x[0][lane] & 3U) < (primary_x[0][lane] & 3U);
            out_x[limb][lane] = choose_alternate ?
                alternate_x[limb][lane] : primary_x[limb][lane];
            selected_slope[limb][lane] = choose_alternate ?
                alternate_slope[limb][lane] : primary_slope[limb][lane];
        }
    }
    ifma_field_sub52(temporary, point_x, out_x);
    ifma_field_mul52(out_y, selected_slope, temporary);
    ifma_field_sub52(out_y, out_y, point_y);
}

static void check_products(void) {
    unsigned trial;
    for (trial = 0; trial < 10000; ++trial) {
        uint64_t left[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        uint64_t right[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        uint64_t product[RCK_IFMA_PRODUCT_LIMBS][RCK_IFMA_LANES];
        size_t limb;
        size_t lane;
        for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
            for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
                left[limb][lane] = rng_next() & RCK_RADIX52_MASK;
                right[limb][lane] = rng_next() & RCK_RADIX52_MASK;
            }
        }
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            left[RCK_IFMA_LIMBS - 1][lane] &=
                RCK_RADIX52_TOP_MASK;
            right[RCK_IFMA_LIMBS - 1][lane] &=
                RCK_RADIX52_TOP_MASK;
        }
        ifma_mul52(product, left, right);
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            uint64_t expected[RCK_IFMA_PRODUCT_LIMBS];
            uint64_t a[RCK_IFMA_LIMBS];
            uint64_t b[RCK_IFMA_LIMBS];
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                a[limb] = left[limb][lane];
                b[limb] = right[limb][lane];
            }
            scalar_mul52(expected, a, b);
            for (limb = 0; limb < RCK_IFMA_PRODUCT_LIMBS; ++limb) {
                if (product[limb][lane] != expected[limb]) {
                    fprintf(
                        stderr,
                        "IFMA product mismatch: trial=%u lane=%zu limb=%zu\n",
                        trial,
                        lane,
                        limb
                    );
                    exit(1);
                }
            }
        }
        ifma_field_sub52(product, left, right);
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            rck_fe a;
            rck_fe b;
            rck_fe expected;
            uint64_t actual[RCK_IFMA_LIMBS];
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                actual[limb] = left[limb][lane];
            }
            fe_from52(&a, actual);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                actual[limb] = right[limb][lane];
            }
            fe_from52(&b, actual);
            rck_fe_sub(&expected, &a, &b);
            fe_to52(actual, &expected);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                if (product[limb][lane] != actual[limb]) {
                    fprintf(
                        stderr,
                        "IFMA subtraction mismatch: trial=%u lane=%zu limb=%zu "
                        "got=%013" PRIx64 " expected=%013" PRIx64 "\n",
                        trial,
                        lane,
                        limb,
                        product[limb][lane],
                        actual[limb]
                    );
                    exit(1);
                }
            }
        }
    }
}

static void fe_from52(
    rck_fe* out,
    const uint64_t value[RCK_IFMA_LIMBS]
) {
    size_t limb;
    rck_fe_set_u32(out, 0);
    for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
        uint64_t bits = value[limb];
        unsigned offset = (unsigned)(52U * limb);
        while (bits != 0) {
            const unsigned word = offset / 32U;
            const unsigned shift = offset % 32U;
            const unsigned width = 32U - shift;
            out->limb[word] |= (uint32_t)(bits << shift);
            bits >>= width;
            offset += width;
        }
    }
}

static void fe_to52(
    uint64_t out[RCK_IFMA_LIMBS],
    const rck_fe* value
) {
    size_t limb;
    for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
        const unsigned offset = (unsigned)(52U * limb);
        uint64_t bits = 0;
        unsigned bit;
        for (bit = 0; bit < 52; ++bit) {
            const unsigned position = offset + bit;
            if (position < 256U) {
                bits |= (uint64_t)(
                    (value->limb[position / 32U] >> (position % 32U)) & 1U
                ) << bit;
            }
        }
        out[limb] = bits;
    }
}

static void check_field_products(void) {
    unsigned trial;
    for (trial = 0; trial < 10000; ++trial) {
        uint64_t left[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        uint64_t right[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        uint64_t product[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        size_t limb;
        size_t lane;
        for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
            for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
                left[limb][lane] = rng_next() & RCK_RADIX52_MASK;
                right[limb][lane] = rng_next() & RCK_RADIX52_MASK;
            }
        }
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            left[RCK_IFMA_LIMBS - 1][lane] &=
                RCK_RADIX52_TOP_MASK >> 1;
            right[RCK_IFMA_LIMBS - 1][lane] &=
                RCK_RADIX52_TOP_MASK >> 1;
        }
        ifma_field_mul52(product, left, right);
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            rck_fe a;
            rck_fe b;
            rck_fe expected;
            uint64_t actual[RCK_IFMA_LIMBS];
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                actual[limb] = left[limb][lane];
            }
            fe_from52(&a, actual);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                actual[limb] = right[limb][lane];
            }
            fe_from52(&b, actual);
            rck_fe_mul(&expected, &a, &b);
            fe_to52(actual, &expected);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                if (product[limb][lane] != actual[limb]) {
                    fprintf(
                        stderr,
                        "IFMA field mismatch: trial=%u lane=%zu limb=%zu\n",
                        trial,
                        lane,
                        limb
                    );
                    exit(1);
                }
            }
        }
    }
}

static void check_dual_steps(void) {
    unsigned trial;
    rck_point jump;
    rck_point_mul_u32(&jump, &RCK_GENERATOR, 17);
    for (trial = 0; trial < 2000; ++trial) {
        uint64_t point_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        uint64_t point_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        uint64_t jump_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        uint64_t jump_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        uint64_t inverse_dx[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        uint64_t result_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        uint64_t result_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        uint64_t full_x[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        uint64_t full_y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
        size_t lane;
        size_t limb;
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            rck_point point;
            rck_fe dx;
            rck_fe inverse;
            uint64_t packed[RCK_IFMA_LIMBS];
            uint32_t scalar = (uint32_t)(2 + (rng_next() % 1000000U));
            if (scalar == 17U) {
                ++scalar;
            }
            rck_point_mul_u32(&point, &RCK_GENERATOR, scalar);
            rck_fe_sub(&dx, &point.x, &jump.x);
            rck_fe_inv(&inverse, &dx);
            fe_to52(packed, &point.x);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                point_x[limb][lane] = packed[limb];
            }
            fe_to52(packed, &point.y);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                point_y[limb][lane] = packed[limb];
            }
            fe_to52(packed, &jump.x);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                jump_x[limb][lane] = packed[limb];
            }
            fe_to52(packed, &jump.y);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                jump_y[limb][lane] = packed[limb];
            }
            fe_to52(packed, &inverse);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                inverse_dx[limb][lane] = packed[limb];
            }
        }
        ifma_dual_baseline52(
            result_x,
            result_y,
            point_x,
            point_y,
            jump_x,
            jump_y,
            inverse_dx
        );
        ifma_dual_full52(
            full_x,
            full_y,
            point_x,
            point_y,
            jump_x,
            jump_y,
            inverse_dx
        );
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            rck_point point;
            rck_fe dx;
            rck_fe inverse;
            rck_dual_result expected;
            rck_dual_result expected_full;
            uint64_t packed[RCK_IFMA_LIMBS];
            rck_fe actual_x;
            rck_fe actual_y;
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                packed[limb] = point_x[limb][lane];
            }
            fe_from52(&point.x, packed);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                packed[limb] = point_y[limb][lane];
            }
            fe_from52(&point.y, packed);
            point.infinity = 0;
            rck_fe_sub(&dx, &point.x, &jump.x);
            rck_fe_inv(&inverse, &dx);
            if (!rck_point_dual_step_preinverted(
                    &expected,
                    &point,
                    &jump,
                    &inverse,
                    RCK_SELECT_BASELINE
                )) {
                fputs("scalar dual step unexpectedly rejected\n", stderr);
                exit(1);
            }
            if (!rck_point_dual_step_preinverted(
                    &expected_full,
                    &point,
                    &jump,
                    &inverse,
                    RCK_SELECT_FULL_GATED2
                )) {
                fputs("scalar full dual step unexpectedly rejected\n", stderr);
                exit(1);
            }
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                packed[limb] = result_x[limb][lane];
            }
            fe_from52(&actual_x, packed);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                packed[limb] = result_y[limb][lane];
            }
            fe_from52(&actual_y, packed);
            if (!rck_fe_equal(&actual_x, &expected.chosen.x) ||
                !rck_fe_equal(&actual_y, &expected.chosen.y)) {
                fprintf(
                    stderr,
                    "IFMA dual mismatch: trial=%u lane=%zu\n", trial, lane
                );
                exit(1);
            }
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                packed[limb] = full_x[limb][lane];
            }
            fe_from52(&actual_x, packed);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                packed[limb] = full_y[limb][lane];
            }
            fe_from52(&actual_y, packed);
            if (!rck_fe_equal(&actual_x, &expected_full.chosen.x) ||
                !rck_fe_equal(&actual_y, &expected_full.chosen.y)) {
                fprintf(
                    stderr,
                    "IFMA full dual mismatch: trial=%u lane=%zu\n",
                    trial,
                    lane
                );
                exit(1);
            }
        }
    }
}


static void benchmark(unsigned iterations) {
    uint64_t left[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t right[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    uint64_t product[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
    rck_fe scalar_left[RCK_IFMA_LANES];
    rck_fe scalar_right[RCK_IFMA_LANES];
    rck_fe scalar_product;
    double vector_started;
    double scalar_started;
    double vector_seconds;
    double scalar_seconds;
    size_t limb;
    size_t lane;
    unsigned iteration;

    for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            left[limb][lane] = rng_next() & RCK_RADIX52_MASK;
            right[limb][lane] = rng_next() & RCK_RADIX52_MASK;
        }
    }
    for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
        uint64_t value[RCK_IFMA_LIMBS];
        left[RCK_IFMA_LIMBS - 1][lane] &= RCK_RADIX52_TOP_MASK >> 1;
        right[RCK_IFMA_LIMBS - 1][lane] &= RCK_RADIX52_TOP_MASK >> 1;
        for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
            value[limb] = left[limb][lane];
        }
        fe_from52(&scalar_left[lane], value);
        for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
            value[limb] = right[limb][lane];
        }
        fe_from52(&scalar_right[lane], value);
    }
    vector_started = seconds_now();
    for (iteration = 0; iteration < iterations; ++iteration) {
        ifma_field_mul52(product, left, right);
        for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
            for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
                left[limb][lane] = product[limb][lane];
            }
        }
    }
    vector_seconds = seconds_now() - vector_started;
    sink ^= product[0][0];

    scalar_started = seconds_now();
    for (iteration = 0; iteration < iterations; ++iteration) {
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            rck_fe_mul(
                &scalar_product, &scalar_left[lane], &scalar_right[lane]
            );
            scalar_left[lane] = scalar_product;
        }
    }
    scalar_seconds = seconds_now() - scalar_started;
    sink ^= scalar_product.limb[0];
    printf(
        "IFMA radix-52 secp256k1 multiplication: PASS\n"
        "iterations=%u lanes=%u\n"
        "vector=%.3f ns/field-product scalar=%.3f ns/field-product "
        "speedup=%.3fx sink=%" PRIu64 "\n",
        iterations,
        RCK_IFMA_LANES,
        vector_seconds * 1e9 / ((double)iterations * RCK_IFMA_LANES),
        scalar_seconds * 1e9 / ((double)iterations * RCK_IFMA_LANES),
        scalar_seconds / vector_seconds,
        sink
    );
}

static void prepare_dual_blocks(ifma_dual_block blocks[RCK_IFMA_DUAL_BLOCKS]) {
    rck_point jump;
    size_t block;
    rck_point_mul_u32(&jump, &RCK_GENERATOR, 17);
    for (block = 0; block < RCK_IFMA_DUAL_BLOCKS; ++block) {
        size_t lane;
        for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
            rck_fe dx;
            uint64_t packed[RCK_IFMA_LIMBS];
            size_t limb;
            uint32_t scalar =
                (uint32_t)(2 + (rng_next() % 1000000U));
            if (scalar == 17U) {
                ++scalar;
            }
            rck_point_mul_u32(
                &blocks[block].scalar_points[lane],
                &RCK_GENERATOR,
                scalar
            );
            rck_fe_sub(
                &dx,
                &blocks[block].scalar_points[lane].x,
                &jump.x
            );
            rck_fe_inv(&blocks[block].scalar_inverses[lane], &dx);
            fe_to52(
                packed,
                &blocks[block].scalar_points[lane].x
            );
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                blocks[block].point_x[limb][lane] = packed[limb];
            }
            fe_to52(
                packed,
                &blocks[block].scalar_points[lane].y
            );
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                blocks[block].point_y[limb][lane] = packed[limb];
            }
            fe_to52(packed, &jump.x);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                blocks[block].jump_x[limb][lane] = packed[limb];
            }
            fe_to52(packed, &jump.y);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                blocks[block].jump_y[limb][lane] = packed[limb];
            }
            fe_to52(packed, &blocks[block].scalar_inverses[lane]);
            for (limb = 0; limb < RCK_IFMA_LIMBS; ++limb) {
                blocks[block].inverse_dx[limb][lane] = packed[limb];
            }
        }
    }
}

static void benchmark_dual_steps(unsigned iterations) {
    ifma_dual_block blocks[RCK_IFMA_DUAL_BLOCKS];
    rck_point jump;
    double vector_started;
    double scalar_started;
    double vector_seconds;
    double scalar_seconds;
    uint64_t vector_checksum = 0;
    uint64_t scalar_checksum = 0;
    unsigned iteration;

    prepare_dual_blocks(blocks);
    rck_point_mul_u32(&jump, &RCK_GENERATOR, 17);
    vector_started = seconds_now();
    for (iteration = 0; iteration < iterations; ++iteration) {
        size_t offset;
        for (offset = 0; offset < RCK_IFMA_DUAL_BLOCKS; ++offset) {
            ifma_dual_block* block = &blocks[
                (offset + iteration) & (RCK_IFMA_DUAL_BLOCKS - 1U)
            ];
            uint64_t x[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
            uint64_t y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
            ifma_dual_baseline52(
                x,
                y,
                block->point_x,
                block->point_y,
                block->jump_x,
                block->jump_y,
                block->inverse_dx
            );
            vector_checksum ^= x[0][0] ^ y[0][RCK_IFMA_LANES - 1U];
        }
    }
    vector_seconds = seconds_now() - vector_started;
    sink ^= vector_checksum;

    scalar_started = seconds_now();
    for (iteration = 0; iteration < iterations; ++iteration) {
        size_t offset;
        for (offset = 0; offset < RCK_IFMA_DUAL_BLOCKS; ++offset) {
            ifma_dual_block* block = &blocks[
                (offset + iteration) & (RCK_IFMA_DUAL_BLOCKS - 1U)
            ];
            size_t lane;
            for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
                rck_dual_result result;
                if (!rck_point_dual_step_preinverted(
                        &result,
                        &block->scalar_points[lane],
                        &jump,
                        &block->scalar_inverses[lane],
                        RCK_SELECT_BASELINE
                    )) {
                    fputs("scalar dual benchmark rejected input\n", stderr);
                    exit(1);
                }
                scalar_checksum ^= result.chosen.x.limb[0] ^
                    result.chosen.y.limb[0];
            }
        }
    }
    scalar_seconds = seconds_now() - scalar_started;
    sink ^= scalar_checksum;
    printf(
        "IFMA 128-step preinverted baseline batch: PASS\n"
        "batches=%u steps=%u\n"
        "vector=%.3f ns/dual-step scalar=%.3f ns/dual-step speedup=%.3fx "
        "sink=%" PRIu64 "\n",
        iterations,
        iterations * RCK_IFMA_DUAL_BLOCKS * RCK_IFMA_LANES,
        vector_seconds * 1e9 /
            ((double)iterations * RCK_IFMA_DUAL_BLOCKS * RCK_IFMA_LANES),
        scalar_seconds * 1e9 /
            ((double)iterations * RCK_IFMA_DUAL_BLOCKS * RCK_IFMA_LANES),
        scalar_seconds / vector_seconds,
        sink
    );

    vector_started = seconds_now();
    for (iteration = 0; iteration < iterations; ++iteration) {
        size_t offset;
        for (offset = 0; offset < RCK_IFMA_DUAL_BLOCKS; ++offset) {
            ifma_dual_block* block = &blocks[
                (offset + iteration) & (RCK_IFMA_DUAL_BLOCKS - 1U)
            ];
            uint64_t x[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
            uint64_t y[RCK_IFMA_LIMBS][RCK_IFMA_LANES];
            ifma_dual_full52(
                x,
                y,
                block->point_x,
                block->point_y,
                block->jump_x,
                block->jump_y,
                block->inverse_dx
            );
            vector_checksum ^= x[0][0] ^ y[0][RCK_IFMA_LANES - 1U];
        }
    }
    vector_seconds = seconds_now() - vector_started;
    sink ^= vector_checksum;

    scalar_started = seconds_now();
    for (iteration = 0; iteration < iterations; ++iteration) {
        size_t offset;
        for (offset = 0; offset < RCK_IFMA_DUAL_BLOCKS; ++offset) {
            ifma_dual_block* block = &blocks[
                (offset + iteration) & (RCK_IFMA_DUAL_BLOCKS - 1U)
            ];
            size_t lane;
            for (lane = 0; lane < RCK_IFMA_LANES; ++lane) {
                rck_dual_result result;
                if (!rck_point_dual_step_preinverted(
                        &result,
                        &block->scalar_points[lane],
                        &jump,
                        &block->scalar_inverses[lane],
                        RCK_SELECT_FULL_GATED2
                    )) {
                    fputs("scalar full dual benchmark rejected input\n", stderr);
                    exit(1);
                }
                scalar_checksum ^= result.chosen.x.limb[0] ^
                    result.chosen.y.limb[0];
            }
        }
    }
    scalar_seconds = seconds_now() - scalar_started;
    sink ^= scalar_checksum;
    printf(
        "IFMA 128-step preinverted SOTA+ batch: PASS\n"
        "batches=%u steps=%u\n"
        "vector=%.3f ns/dual-step scalar=%.3f ns/dual-step speedup=%.3fx "
        "sink=%" PRIu64 "\n",
        iterations,
        iterations * RCK_IFMA_DUAL_BLOCKS * RCK_IFMA_LANES,
        vector_seconds * 1e9 /
            ((double)iterations * RCK_IFMA_DUAL_BLOCKS * RCK_IFMA_LANES),
        scalar_seconds * 1e9 /
            ((double)iterations * RCK_IFMA_DUAL_BLOCKS * RCK_IFMA_LANES),
        scalar_seconds / vector_seconds,
        sink
    );
}

int main(int argc, char** argv) {
    unsigned iterations = 200000;
    if (!__builtin_cpu_supports("avx512f") ||
        !__builtin_cpu_supports("avx512ifma")) {
        fputs("AVX-512 IFMA is unavailable on this CPU\n", stderr);
        return 77;
    }
    if (argc == 2) {
        char* end;
        unsigned long parsed = strtoul(argv[1], &end, 10);
        if (*argv[1] == '\0' || *end != '\0' || parsed == 0 ||
            parsed > UINT32_MAX) {
            fputs("usage: ifma52_bench [positive_iterations]\n", stderr);
            return 2;
        }
        iterations = (unsigned)parsed;
    } else if (argc != 1) {
        fputs("usage: ifma52_bench [positive_iterations]\n", stderr);
        return 2;
    }
    check_products();
    check_field_products();
    check_dual_steps();
    puts("IFMA radix-52 dual-step oracle checks: PASS");
    benchmark(iterations);
    benchmark_dual_steps(iterations / 32U == 0 ? 1U : iterations / 32U);
    return 0;
}
