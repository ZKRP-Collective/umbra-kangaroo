#include "rck_portable.h"

/* Field multiplications performed; incremented only with -DRCK_COUNT_MULS. */
uint64_t rck_mul_count;

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

const rck_fe RCK_FIELD_P = {{
    0xfffffc2fU, 0xfffffffeU, 0xffffffffU, 0xffffffffU,
    0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU
}};

static const rck_fe RCK_FIELD_P_MINUS_2 = {{
    0xfffffc2dU, 0xfffffffeU, 0xffffffffU, 0xffffffffU,
    0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU
}};

const rck_point RCK_GENERATOR = {
    {{
        0x16f81798U, 0x59f2815bU, 0x2dce28d9U, 0x029bfcdbU,
        0xce870b07U, 0x55a06295U, 0xf9dcbbacU, 0x79be667eU
    }},
    {{
        0xfb10d4b8U, 0x9c47d08fU, 0xa6855419U, 0xfd17b448U,
        0x0e1108a8U, 0x5da4fbfcU, 0x26a3c465U, 0x483ada77U
    }},
    0
};

uint32_t rck_add256_raw_portable(
    uint32_t out[8],
    const uint32_t left[8],
    const uint32_t right[8]
) {
    uint64_t carry = 0;
    size_t i;
    for (i = 0; i < 8; ++i) {
        const uint64_t sum =
            (uint64_t)left[i] + right[i] + carry;
        out[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    return (uint32_t)carry;
}

static int fe_compare(const rck_fe* left, const rck_fe* right) {
    int i;
    for (i = 7; i >= 0; --i) {
        if (left->limb[i] < right->limb[i]) {
            return -1;
        }
        if (left->limb[i] > right->limb[i]) {
            return 1;
        }
    }
    return 0;
}

static uint32_t sub256_raw(
    uint32_t out[8],
    const uint32_t left[8],
    const uint32_t right[8]
) {
    uint64_t borrow = 0;
    size_t i;
    for (i = 0; i < 8; ++i) {
        const uint64_t difference =
            (uint64_t)left[i] - right[i] - borrow;
        out[i] = (uint32_t)difference;
        borrow = difference >> 63;
    }
    return (uint32_t)borrow;
}

static void fe_normalize(rck_fe* value) {
    while (fe_compare(value, &RCK_FIELD_P) >= 0) {
        (void)sub256_raw(
            value->limb, value->limb, RCK_FIELD_P.limb
        );
    }
}

static void fold_2_256_carry(rck_fe* value, uint32_t carry) {
    while (carry != 0) {
        uint64_t sum;
        size_t i;

        sum = (uint64_t)value->limb[0] + 977U;
        value->limb[0] = (uint32_t)sum;
        sum = (sum >> 32) + value->limb[1] + 1U;
        value->limb[1] = (uint32_t)sum;
        for (i = 2; i < 8; ++i) {
            sum = (sum >> 32) + value->limb[i];
            value->limb[i] = (uint32_t)sum;
        }
        carry = (uint32_t)(sum >> 32);
    }
}

void rck_fe_set_u32(rck_fe* out, uint32_t value) {
    memset(out, 0, sizeof(*out));
    out->limb[0] = value;
}

int rck_fe_equal(const rck_fe* left, const rck_fe* right) {
    uint32_t difference = 0;
    size_t i;
    for (i = 0; i < 8; ++i) {
        difference |= left->limb[i] ^ right->limb[i];
    }
    return difference == 0;
}

int rck_fe_is_zero(const rck_fe* value) {
    static const rck_fe zero = {{0}};
    return rck_fe_equal(value, &zero);
}

void rck_fe_add(rck_fe* out, const rck_fe* left, const rck_fe* right) {
    rck_fe result;
    uint32_t carry;
#if defined(RCK_USE_X86_64_ASM) && (defined(__x86_64__) || defined(_M_X64))
    carry = rck_add256_raw_asm(result.limb, left->limb, right->limb);
#else
    carry = rck_add256_raw_portable(
        result.limb, left->limb, right->limb
    );
#endif
    fold_2_256_carry(&result, carry);
    fe_normalize(&result);
    *out = result;
}

void rck_fe_sub(rck_fe* out, const rck_fe* left, const rck_fe* right) {
    rck_fe result;
    const uint32_t borrow =
        sub256_raw(result.limb, left->limb, right->limb);
    if (borrow != 0) {
        (void)rck_add256_raw_portable(
            result.limb, result.limb, RCK_FIELD_P.limb
        );
    }
    *out = result;
}

void rck_fe_neg(rck_fe* out, const rck_fe* value) {
    if (rck_fe_is_zero(value)) {
        *out = *value;
        return;
    }
    (void)sub256_raw(out->limb, RCK_FIELD_P.limb, value->limb);
}

#if !defined(__SIZEOF_INT128__)
static void add_u64_at(uint32_t limbs[18], size_t position, uint64_t value) {
    uint64_t carry = value;
    while (carry != 0 && position < 18) {
        const uint64_t sum =
            (uint64_t)limbs[position] + (uint32_t)carry;
        limbs[position] = (uint32_t)sum;
        carry = (carry >> 32) + (sum >> 32);
        ++position;
    }
}

static void fe_mul_32(
    rck_fe* out,
    const rck_fe* left,
    const rck_fe* right
) {
    uint32_t product[18] = {0};
    size_t i;
    size_t j;
    unsigned reduction_round;
    rck_fe result;

    for (i = 0; i < 8; ++i) {
        uint64_t carry = 0;
        for (j = 0; j < 8; ++j) {
            const size_t position = i + j;
            const uint64_t accumulated =
                (uint64_t)left->limb[i] * right->limb[j] +
                product[position] + carry;
            product[position] = (uint32_t)accumulated;
            carry = accumulated >> 32;
        }
        add_u64_at(product, i + 8, carry);
    }

    for (reduction_round = 0; reduction_round < 32; ++reduction_round) {
        int found_high = 0;
        int position;
        for (position = 17; position >= 8; --position) {
            const uint32_t high = product[position];
            if (high == 0) {
                continue;
            }
            found_high = 1;
            product[position] = 0;
            add_u64_at(
                product,
                (size_t)position - 8,
                (uint64_t)high * 977U
            );
            add_u64_at(
                product,
                (size_t)position - 7,
                high
            );
        }
        if (!found_high) {
            break;
        }
    }

    memcpy(result.limb, product, sizeof(result.limb));
    fe_normalize(&result);
    *out = result;
}
#else
typedef __uint128_t rck_u128;

static void add_u128_limbs(
    uint64_t* limbs,
    size_t count,
    size_t position,
    rck_u128 value
) {
    while (value != 0 && position < count) {
        const rck_u128 sum =
            (rck_u128)limbs[position] + (uint64_t)value;
        limbs[position] = (uint64_t)sum;
        value = (value >> 64) + (sum >> 64);
        ++position;
    }
}

static void reduce_product_64(
    rck_fe* out,
    const uint64_t product[8]
) {
    static const uint64_t reduction_constant =
        UINT64_C(0x1000003d1);
    uint64_t reduced[6] = {0};
    rck_fe result;
    size_t i;
    for (i = 0; i < 4; ++i) {
        reduced[i] = product[i];
    }
    for (i = 0; i < 4; ++i) {
        add_u128_limbs(
            reduced,
            6,
            i,
            (rck_u128)product[i + 4] * reduction_constant
        );
    }
    while (reduced[4] != 0 || reduced[5] != 0) {
        const uint64_t high0 = reduced[4];
        const uint64_t high1 = reduced[5];
        reduced[4] = 0;
        reduced[5] = 0;
        add_u128_limbs(
            reduced,
            6,
            0,
            (rck_u128)high0 * reduction_constant
        );
        add_u128_limbs(
            reduced,
            6,
            1,
            (rck_u128)high1 * reduction_constant
        );
    }
    for (i = 0; i < 4; ++i) {
        result.limb[2 * i] = (uint32_t)reduced[i];
        result.limb[2 * i + 1] = (uint32_t)(reduced[i] >> 32);
    }
    fe_normalize(&result);
    *out = result;
}

void rck_mul256_raw_portable64(
    uint64_t out[8],
    const uint64_t left[4],
    const uint64_t right[4]
) {
    size_t i;
    size_t j;
    memset(out, 0, 8 * sizeof(*out));
    for (i = 0; i < 4; ++i) {
        uint64_t carry = 0;
        for (j = 0; j < 4; ++j) {
            const size_t position = i + j;
            const rck_u128 accumulated =
                (rck_u128)left[i] * right[j] +
                out[position] + carry;
            out[position] = (uint64_t)accumulated;
            carry = (uint64_t)(accumulated >> 64);
        }
        out[i + 4] = carry;
    }
}

static void fe_mul_64(
    rck_fe* out,
    const rck_fe* left,
    const rck_fe* right
) {
    uint64_t a[4];
    uint64_t b[4];
    uint64_t product[8];
    size_t i;

    for (i = 0; i < 4; ++i) {
        a[i] = (uint64_t)left->limb[2 * i] |
            ((uint64_t)left->limb[2 * i + 1] << 32);
        b[i] = (uint64_t)right->limb[2 * i] |
            ((uint64_t)right->limb[2 * i + 1] << 32);
    }
    #if defined(RCK_USE_X86_64_BMI2) && defined(__x86_64__)
    if (__builtin_cpu_supports("bmi2")) {
        rck_mul256_raw_bmi2(product, a, b);
    } else {
        rck_mul256_raw_portable64(product, a, b);
    }
    #else
    rck_mul256_raw_portable64(product, a, b);
    #endif

    reduce_product_64(out, product);
#if defined(RCK_COUNT_MULS)
    ++rck_mul_count;
#endif
}

#endif

void rck_fe_mul(rck_fe* out, const rck_fe* left, const rck_fe* right) {
#if defined(__SIZEOF_INT128__)
    fe_mul_64(out, left, right);
#else
    fe_mul_32(out, left, right);
#endif
}

void rck_fe_sqr(rck_fe* out, const rck_fe* value) {
    rck_fe_mul(out, value, value);
}

#if defined(RCK_FUSED_AFFINE_FORMULAS)
/*
 * The affine hot path keeps intermediate field elements in caller locals.
 * Addition/subtraction is already canonical and does not need a reduction;
 * only products cross the pseudo-Mersenne reduction boundary.  Keeping these
 * helpers local lets the compiler see the complete formula without changing
 * either the field representation or its aliasing guarantees.
 */
static void fe_sub_hot(
    rck_fe* out,
    const rck_fe* left,
    const rck_fe* right
) {
    rck_fe result;
    const uint32_t borrow =
        sub256_raw(result.limb, left->limb, right->limb);
    if (borrow != 0) {
        (void)rck_add256_raw_portable(
            result.limb, result.limb, RCK_FIELD_P.limb
        );
    }
    *out = result;
}

static void fe_mul_hot(
    rck_fe* out,
    const rck_fe* left,
    const rck_fe* right
) {
#if defined(__SIZEOF_INT128__)
    fe_mul_64(out, left, right);
#else
    fe_mul_32(out, left, right);
#endif
}

static void fe_sqr_hot(rck_fe* out, const rck_fe* value) {
    fe_mul_hot(out, value, value);
}

#define RCK_HOT_SUB fe_sub_hot
#define RCK_HOT_MUL fe_mul_hot
#define RCK_HOT_SQR fe_sqr_hot
#else
#define RCK_HOT_SUB rck_fe_sub
#define RCK_HOT_MUL rck_fe_mul
#define RCK_HOT_SQR rck_fe_sqr
#endif

void rck_fe_inv(rck_fe* out, const rck_fe* value) {
    rck_fe powers[16];
    rck_fe result;
    int nibble;
    unsigned power;
    rck_fe_set_u32(&result, 1);
    powers[0] = result;
    powers[1] = *value;
    for (power = 2; power < 16; ++power) {
        rck_fe_mul(&powers[power], &powers[power - 1], value);
    }
    for (nibble = 63; nibble >= 0; --nibble) {
        const unsigned word = (unsigned)nibble / 8U;
        const unsigned shift = ((unsigned)nibble % 8U) * 4U;
        const unsigned digit =
            (RCK_FIELD_P_MINUS_2.limb[word] >> shift) & 15U;
        unsigned square;
        for (square = 0; square < 4; ++square) {
            rck_fe_sqr(&result, &result);
        }
        if (digit != 0) {
            rck_fe_mul(&result, &result, &powers[digit]);
        }
    }
    *out = result;
}

int rck_fe_batch_inv(
    rck_fe* out,
    const rck_fe* values,
    rck_fe* prefix_scratch,
    size_t count
) {
    rck_fe accumulated;
    rck_fe inverse;
    size_t index;
    rck_fe_set_u32(&accumulated, 1);
    for (index = 0; index < count; ++index) {
        if (rck_fe_is_zero(&values[index])) {
            return 0;
        }
        prefix_scratch[index] = accumulated;
        rck_fe_mul(&accumulated, &accumulated, &values[index]);
    }
    rck_fe_inv(&inverse, &accumulated);
    for (index = count; index > 0; --index) {
        const size_t position = index - 1;
        rck_fe_mul(
            &out[position],
            &inverse,
            &prefix_scratch[position]
        );
        rck_fe_mul(&inverse, &inverse, &values[position]);
    }
    return 1;
}

void rck_point_neg(rck_point* out, const rck_point* point) {
    *out = *point;
    if (!point->infinity) {
        rck_fe_neg(&out->y, &point->y);
    }
}

void rck_point_double(rck_point* out, const rck_point* point) {
    rck_fe x_squared;
    rck_fe numerator;
    rck_fe denominator;
    rck_fe inverse;
    rck_fe slope;
    rck_fe x3;
    rck_fe y3;
    rck_fe temporary;

    if (point->infinity || rck_fe_is_zero(&point->y)) {
        memset(out, 0, sizeof(*out));
        out->infinity = 1;
        return;
    }

    rck_fe_sqr(&x_squared, &point->x);
    rck_fe_add(&numerator, &x_squared, &x_squared);
    rck_fe_add(&numerator, &numerator, &x_squared);
    rck_fe_add(&denominator, &point->y, &point->y);
    rck_fe_inv(&inverse, &denominator);
    rck_fe_mul(&slope, &numerator, &inverse);
    rck_fe_sqr(&x3, &slope);
    rck_fe_add(&temporary, &point->x, &point->x);
    rck_fe_sub(&x3, &x3, &temporary);
    rck_fe_sub(&temporary, &point->x, &x3);
    rck_fe_mul(&y3, &slope, &temporary);
    rck_fe_sub(&y3, &y3, &point->y);

    out->x = x3;
    out->y = y3;
    out->infinity = 0;
}

int rck_point_add_preinverted(
    rck_point* out,
    const rck_point* left,
    const rck_point* right,
    const rck_fe* inverse_dx
) {
    rck_fe dy;
    rck_fe slope;
    rck_fe x3;
    rck_fe y3;
    rck_fe temporary;

    if (left->infinity || right->infinity ||
        rck_fe_equal(&left->x, &right->x)) {
        return 0;
    }
    RCK_HOT_SUB(&dy, &left->y, &right->y);
    RCK_HOT_MUL(&slope, &dy, inverse_dx);
    RCK_HOT_SQR(&x3, &slope);
    RCK_HOT_SUB(&x3, &x3, &left->x);
    RCK_HOT_SUB(&x3, &x3, &right->x);
    RCK_HOT_SUB(&temporary, &left->x, &x3);
    RCK_HOT_MUL(&y3, &slope, &temporary);
    RCK_HOT_SUB(&y3, &y3, &left->y);

    out->x = x3;
    out->y = y3;
    out->infinity = 0;
    return 1;
}

void rck_point_add(
    rck_point* out,
    const rck_point* left,
    const rck_point* right
) {
    rck_fe dx;
    rck_fe inverse;
    if (left->infinity) {
        *out = *right;
        return;
    }
    if (right->infinity) {
        *out = *left;
        return;
    }
    if (rck_fe_equal(&left->x, &right->x)) {
        if (rck_fe_equal(&left->y, &right->y)) {
            rck_point_double(out, left);
        } else {
            memset(out, 0, sizeof(*out));
            out->infinity = 1;
        }
        return;
    }
    rck_fe_sub(&dx, &left->x, &right->x);
    rck_fe_inv(&inverse, &dx);
    if (!rck_point_add_preinverted(
            out,
            left,
            right,
            &inverse
        )) {
        memset(out, 0, sizeof(*out));
        out->infinity = 1;
    }
}

void rck_point_mul_u64(rck_point* out, const rck_point* point, uint64_t scalar) {
    rck_point result;
    rck_point addend = *point;
    memset(&result, 0, sizeof(result));
    result.infinity = 1;

    while (scalar != 0) {
        if (scalar & 1U) {
            rck_point_add(&result, &result, &addend);
        }
        scalar >>= 1;
        if (scalar != 0) {
            rck_point_double(&addend, &addend);
        }
    }
    *out = result;
}

void rck_point_mul_u32(rck_point* out, const rck_point* point, uint32_t scalar) {
    rck_point_mul_u64(out, point, scalar);
}

typedef struct {
    rck_fe x;
    rck_fe y;
    rck_fe z;
    uint32_t infinity;
} rck_jacobian;

static void jacobian_set_generator(rck_jacobian* out) {
    out->x = RCK_GENERATOR.x;
    out->y = RCK_GENERATOR.y;
    rck_fe_set_u32(&out->z, 1);
    out->infinity = 0;
}

static void jacobian_double(rck_jacobian* out, const rck_jacobian* point) {
    rck_fe a;
    rck_fe b;
    rck_fe c;
    rck_fe d;
    rck_fe e;
    rck_fe f;
    rck_fe temporary;
    rck_fe x3;
    rck_fe y3;
    rck_fe z3;
    if (point->infinity || rck_fe_is_zero(&point->y)) {
        *out = *point;
        out->infinity = 1;
        return;
    }
    rck_fe_sqr(&a, &point->x);
    rck_fe_sqr(&b, &point->y);
    rck_fe_sqr(&c, &b);
    rck_fe_add(&temporary, &point->x, &b);
    rck_fe_sqr(&d, &temporary);
    rck_fe_sub(&d, &d, &a);
    rck_fe_sub(&d, &d, &c);
    rck_fe_add(&d, &d, &d);
    rck_fe_add(&e, &a, &a);
    rck_fe_add(&e, &e, &a);
    rck_fe_sqr(&f, &e);
    rck_fe_add(&temporary, &d, &d);
    rck_fe_sub(&x3, &f, &temporary);
    rck_fe_sub(&temporary, &d, &x3);
    rck_fe_mul(&y3, &e, &temporary);
    rck_fe_add(&temporary, &c, &c);
    rck_fe_add(&temporary, &temporary, &temporary);
    rck_fe_add(&temporary, &temporary, &temporary);
    rck_fe_sub(&y3, &y3, &temporary);
    rck_fe_mul(&z3, &point->y, &point->z);
    rck_fe_add(&z3, &z3, &z3);
    out->x = x3;
    out->y = y3;
    out->z = z3;
    out->infinity = 0;
}

static void jacobian_add_generator(
    rck_jacobian* out,
    const rck_jacobian* point
) {
    rck_fe z_squared;
    rck_fe u2;
    rck_fe s2;
    rck_fe h;
    rck_fe hh;
    rck_fe i;
    rck_fe j;
    rck_fe r;
    rck_fe v;
    rck_fe temporary;
    rck_fe x3;
    rck_fe y3;
    rck_fe z3;
    if (point->infinity) {
        jacobian_set_generator(out);
        return;
    }
    rck_fe_sqr(&z_squared, &point->z);
    rck_fe_mul(&u2, &RCK_GENERATOR.x, &z_squared);
    rck_fe_mul(&s2, &point->z, &z_squared);
    rck_fe_mul(&s2, &RCK_GENERATOR.y, &s2);
    rck_fe_sub(&h, &u2, &point->x);
    rck_fe_sqr(&hh, &h);
    rck_fe_add(&i, &hh, &hh);
    rck_fe_add(&i, &i, &i);
    rck_fe_mul(&j, &h, &i);
    rck_fe_sub(&r, &s2, &point->y);
    rck_fe_add(&r, &r, &r);
    rck_fe_mul(&v, &point->x, &i);
    rck_fe_sqr(&x3, &r);
    rck_fe_sub(&x3, &x3, &j);
    rck_fe_add(&temporary, &v, &v);
    rck_fe_sub(&x3, &x3, &temporary);
    rck_fe_sub(&temporary, &v, &x3);
    rck_fe_mul(&y3, &r, &temporary);
    rck_fe_mul(&temporary, &point->y, &j);
    rck_fe_add(&temporary, &temporary, &temporary);
    rck_fe_sub(&y3, &y3, &temporary);
    rck_fe_add(&temporary, &point->z, &h);
    rck_fe_sqr(&z3, &temporary);
    rck_fe_sub(&z3, &z3, &z_squared);
    rck_fe_sub(&z3, &z3, &hh);
    out->x = x3;
    out->y = y3;
    out->z = z3;
    out->infinity = 0;
}

static int jacobian_batch_to_affine(
    rck_point* out,
    const rck_jacobian* jacobians,
    size_t count
) {
    rck_fe* z_values;
    rck_fe* inverses;
    rck_fe* prefixes;
    size_t index;
    if (count == 0) {
        return 1;
    }
    z_values = (rck_fe*)calloc(count, sizeof(*z_values));
    inverses = (rck_fe*)calloc(count, sizeof(*inverses));
    prefixes = (rck_fe*)calloc(count, sizeof(*prefixes));
    if (z_values == NULL || inverses == NULL || prefixes == NULL) {
        free(z_values);
        free(inverses);
        free(prefixes);
        return 0;
    }
    for (index = 0; index < count; ++index) {
        if (jacobians[index].infinity) {
            rck_fe_set_u32(&z_values[index], 1);
        } else {
            z_values[index] = jacobians[index].z;
        }
    }
    if (!rck_fe_batch_inv(inverses, z_values, prefixes, count)) {
        free(z_values);
        free(inverses);
        free(prefixes);
        return 0;
    }
    for (index = 0; index < count; ++index) {
        rck_fe z_squared;
        rck_fe z_cubed;
        if (jacobians[index].infinity) {
            memset(&out[index], 0, sizeof(out[index]));
            out[index].infinity = 1;
            continue;
        }
        rck_fe_sqr(&z_squared, &inverses[index]);
        rck_fe_mul(&out[index].x, &jacobians[index].x, &z_squared);
        rck_fe_mul(&z_cubed, &z_squared, &inverses[index]);
        rck_fe_mul(&out[index].y, &jacobians[index].y, &z_cubed);
        out[index].infinity = 0;
    }
    free(z_values);
    free(inverses);
    free(prefixes);
    return 1;
}

int rck_generator_mul_batch(
    rck_point* out,
    const uint64_t* scalars,
    size_t count
) {
    rck_jacobian* jacobians;
    size_t index;
    int bit;
    int status;
    if (count == 0) {
        return 1;
    }
    jacobians = (rck_jacobian*)calloc(count, sizeof(*jacobians));
    if (jacobians == NULL) {
        free(jacobians);
        return 0;
    }
    for (index = 0; index < count; ++index) {
        jacobians[index].infinity = 1;
        for (bit = 63; bit >= 0; --bit) {
            jacobian_double(&jacobians[index], &jacobians[index]);
            if ((scalars[index] >> bit) & UINT64_C(1)) {
                jacobian_add_generator(
                    &jacobians[index],
                    &jacobians[index]
                );
            }
        }
    }
    status = jacobian_batch_to_affine(out, jacobians, count);
    free(jacobians);
    return status;
}

int rck_generator_sequence_batch(
    rck_point* out,
    uint64_t first_scalar,
    size_t count
) {
    rck_jacobian* jacobians;
    rck_jacobian current;
    uint64_t scalar = first_scalar;
    size_t index;
    int bit;
    int status;
    if (count == 0) {
        return 1;
    }
    if (count - 1U > UINT64_MAX - first_scalar) {
        return 0;
    }
    jacobians = (rck_jacobian*)calloc(count, sizeof(*jacobians));
    if (jacobians == NULL) {
        return 0;
    }
    memset(&current, 0, sizeof(current));
    current.infinity = 1;
    for (bit = 63; bit >= 0; --bit) {
        jacobian_double(&current, &current);
        if ((scalar >> bit) & UINT64_C(1)) {
            jacobian_add_generator(&current, &current);
        }
    }
    for (index = 0; index < count; ++index) {
        jacobians[index] = current;
        if (index + 1U < count) {
            if (scalar == 1U) {
                jacobian_double(&current, &current);
            } else {
                jacobian_add_generator(&current, &current);
            }
            ++scalar;
        }
    }
    status = jacobian_batch_to_affine(out, jacobians, count);
    free(jacobians);
    return status;
}

static uint32_t score_x2(const rck_fe* x) {
#ifndef RCK_SCORE_LIMB
#define RCK_SCORE_LIMB 0
#endif
    return x->limb[RCK_SCORE_LIMB] & 3U;
}

static void finish_y(
    rck_fe* out,
    const rck_point* point,
    const rck_fe* x3,
    const rck_fe* slope
) {
    rck_fe temporary;
    RCK_HOT_SUB(&temporary, &point->x, x3);
    RCK_HOT_MUL(out, slope, &temporary);
    RCK_HOT_SUB(out, out, &point->y);
}

int rck_point_dual_step_preinverted(
    rck_dual_result* out,
    const rck_point* point,
    const rck_point* jump,
    const rck_fe* inverse_dx,
    rck_select_mode mode
) {
    rck_fe primary_jump_y;
    rck_fe primary_slope;
    rck_fe primary_x;
    rck_fe alternate_jump_y;
    rck_fe alternate_slope;
    rck_fe alternate_x;
    rck_fe temporary;
    uint32_t primary_score;
    int evaluate = 0;
    int select_alternate = 0;

    if (point->infinity || jump->infinity ||
        rck_fe_equal(&point->x, &jump->x)) {
        return 0;
    }

    primary_jump_y = jump->y;
    if (point->y.limb[0] & 1U) {
        rck_fe_neg(&primary_jump_y, &primary_jump_y);
    }
    RCK_HOT_SUB(&temporary, &point->y, &primary_jump_y);
    RCK_HOT_MUL(&primary_slope, &temporary, inverse_dx);
    RCK_HOT_SQR(&primary_x, &primary_slope);
    RCK_HOT_SUB(&primary_x, &primary_x, &point->x);
    RCK_HOT_SUB(&primary_x, &primary_x, &jump->x);
    primary_score = score_x2(&primary_x);

    if (mode == RCK_SELECT_LITE2) {
        evaluate = primary_score == 3U;
    } else if (mode == RCK_SELECT_MID2) {
        evaluate = primary_score >= 2U;
    } else if (mode == RCK_SELECT_FULL_GATED2) {
        evaluate = primary_score != 0U;
    }

    if (evaluate) {
        rck_fe_neg(&alternate_jump_y, &primary_jump_y);
        RCK_HOT_SUB(&temporary, &point->y, &alternate_jump_y);
        RCK_HOT_MUL(&alternate_slope, &temporary, inverse_dx);
        RCK_HOT_SQR(&alternate_x, &alternate_slope);
        RCK_HOT_SUB(&alternate_x, &alternate_x, &point->x);
        RCK_HOT_SUB(&alternate_x, &alternate_x, &jump->x);
        select_alternate =
            score_x2(&alternate_x) < primary_score;
    }

    memset(out, 0, sizeof(*out));
    out->evaluated_alternate = (uint32_t)evaluate;
    out->selected_alternate = (uint32_t)select_alternate;
    out->chosen.x = select_alternate ? alternate_x : primary_x;
    finish_y(
        &out->chosen.y,
        point,
        &out->chosen.x,
        select_alternate ? &alternate_slope : &primary_slope
    );
    out->chosen.infinity = 0;
    if (evaluate) {
        out->rejected_x =
            select_alternate ? primary_x : alternate_x;
    }
    return 1;
}

int rck_point_dual_step(
    rck_dual_result* out,
    const rck_point* point,
    const rck_point* jump,
    rck_select_mode mode
) {
    rck_fe dx;
    rck_fe inverse;
    if (point->infinity || jump->infinity ||
        rck_fe_equal(&point->x, &jump->x)) {
        return 0;
    }
    rck_fe_sub(&dx, &point->x, &jump->x);
    rck_fe_inv(&inverse, &dx);
    return rck_point_dual_step_preinverted(
        out,
        point,
        jump,
        &inverse,
        mode
    );
}
