#ifndef UMBRA_GFX906_FIELD_H
#define UMBRA_GFX906_FIELD_H

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "../portable/rck_portable.h"

/* Correctness-first 8x32 field backend for Vega20/gfx906. Optimized GCN
 * assembly may replace these leaf routines without changing the batch ABI. */

static __device__ __forceinline__ uint32_t umbra_sub_raw(
    uint32_t out[8], const uint32_t left[8], const uint32_t right[8]
) {
    uint64_t borrow = 0;
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const uint64_t difference =
            (uint64_t)left[i] - right[i] - borrow;
        out[i] = (uint32_t)difference;
        borrow = difference >> 63;
    }
    return (uint32_t)borrow;
}

static __device__ __forceinline__ int umbra_compare_p(const uint32_t value[8]) {
    const uint32_t p[8] = {
        0xfffffc2fU, 0xfffffffeU, 0xffffffffU, 0xffffffffU,
        0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU
    };
    #pragma unroll
    for (int i = 7; i >= 0; --i) {
        if (value[i] < p[i]) {
            return -1;
        }
        if (value[i] > p[i]) {
            return 1;
        }
    }
    return 0;
}

static __device__ __forceinline__ void umbra_normalize(uint32_t value[8]) {
    const uint32_t p[8] = {
        0xfffffc2fU, 0xfffffffeU, 0xffffffffU, 0xffffffffU,
        0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU
    };
    if (umbra_compare_p(value) >= 0) {
        (void)umbra_sub_raw(value, value, p);
    }
}

static __device__ __forceinline__ void umbra_fold_carry(
    uint32_t value[8], uint32_t carry
) {
    while (carry != 0) {
        uint64_t sum = (uint64_t)value[0] + 977U;
        value[0] = (uint32_t)sum;
        sum = (sum >> 32) + value[1] + 1U;
        value[1] = (uint32_t)sum;
        #pragma unroll
        for (unsigned i = 2; i < 8; ++i) {
            sum = (sum >> 32) + value[i];
            value[i] = (uint32_t)sum;
        }
        carry = (uint32_t)(sum >> 32);
    }
}

static __device__ __forceinline__ void umbra_fe_add(
    rck_fe* out, const rck_fe* left, const rck_fe* right
) {
    rck_fe result;
    uint64_t carry = 0;
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const uint64_t sum =
            (uint64_t)left->limb[i] + right->limb[i] + carry;
        result.limb[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    umbra_fold_carry(result.limb, (uint32_t)carry);
    umbra_normalize(result.limb);
    *out = result;
}

static __device__ __forceinline__ void umbra_fe_sub(
    rck_fe* out, const rck_fe* left, const rck_fe* right
) {
    const uint32_t p[8] = {
        0xfffffc2fU, 0xfffffffeU, 0xffffffffU, 0xffffffffU,
        0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU
    };
    rck_fe result;
    const uint32_t borrow =
        umbra_sub_raw(result.limb, left->limb, right->limb);
    if (borrow != 0) {
        uint64_t carry = 0;
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
            const uint64_t sum =
                (uint64_t)result.limb[i] + p[i] + carry;
            result.limb[i] = (uint32_t)sum;
            carry = sum >> 32;
        }
    }
    *out = result;
}

static __device__ __forceinline__ void umbra_add_u64_at(
    uint32_t limbs[18], unsigned position, uint64_t value
) {
    while (value != 0 && position < 18) {
        const uint64_t sum =
            (uint64_t)limbs[position] + (uint32_t)value;
        limbs[position] = (uint32_t)sum;
        value = (value >> 32) + (sum >> 32);
        ++position;
    }
}

static __device__ __forceinline__ void umbra_fe_mul(
    rck_fe* out, const rck_fe* left, const rck_fe* right
) {
    uint32_t product[18] = {0};
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        uint64_t carry = 0;
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
            const unsigned position = i + j;
            const uint64_t accumulated =
                (uint64_t)left->limb[i] * right->limb[j] +
                product[position] + carry;
            product[position] = (uint32_t)accumulated;
            carry = accumulated >> 32;
        }
        umbra_add_u64_at(product, i + 8, carry);
    }

    for (unsigned round = 0; round < 32; ++round) {
        int found_high = 0;
        for (int position = 17; position >= 8; --position) {
            const uint32_t high = product[position];
            if (high == 0) {
                continue;
            }
            found_high = 1;
            product[position] = 0;
            umbra_add_u64_at(
                product, (unsigned)position - 8, (uint64_t)high * 977U
            );
            umbra_add_u64_at(product, (unsigned)position - 7, high);
        }
        if (!found_high) {
            break;
        }
    }

    rck_fe result;
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        result.limb[i] = product[i];
    }
    umbra_normalize(result.limb);
    *out = result;
}

#endif
