#include "rck_portable.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng_state = UINT64_C(0x123456789abcdef0);

static uint64_t rng_next(void) {
    uint64_t z;
    rng_state += UINT64_C(0x9e3779b97f4a7c15);
    z = rng_state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static void fail(const char* message) {
    fprintf(stderr, "portable test failed: %s\n", message);
    exit(1);
}

static int point_equal(const rck_point* left, const rck_point* right) {
    if (left->infinity != right->infinity) {
        return 0;
    }
    return left->infinity ||
        (rck_fe_equal(&left->x, &right->x) &&
         rck_fe_equal(&left->y, &right->y));
}

static void random_fe(rck_fe* out) {
    size_t i;
    for (i = 0; i < 8; i += 2) {
        const uint64_t value = rng_next();
        out->limb[i] = (uint32_t)value;
        out->limb[i + 1] = (uint32_t)(value >> 32);
    }
    if (out->limb[7] == UINT32_MAX &&
        out->limb[6] == UINT32_MAX) {
        out->limb[7] -= 1;
    }
}

static void test_raw_assembly(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned trial;
    for (trial = 0; trial < 100000; ++trial) {
        uint32_t left[8];
        uint32_t right[8];
        uint32_t portable[8];
        uint32_t assembly[8];
        size_t i;
        uint32_t carry_portable;
        uint32_t carry_assembly;
        for (i = 0; i < 8; ++i) {
            left[i] = (uint32_t)rng_next();
            right[i] = (uint32_t)rng_next();
        }
        carry_portable =
            rck_add256_raw_portable(portable, left, right);
        carry_assembly =
            rck_add256_raw_asm(assembly, left, right);
        if (carry_portable != carry_assembly ||
            memcmp(portable, assembly, sizeof(portable)) != 0) {
            fail("x86_64 raw add differs from portable C");
        }
    }
#if defined(__SIZEOF_INT128__)
    if (__builtin_cpu_supports("bmi2")) {
        for (trial = 0; trial < 100000; ++trial) {
            uint64_t left[4];
            uint64_t right[4];
            uint64_t portable[8];
            uint64_t assembly[8];
            size_t i;
            for (i = 0; i < 4; ++i) {
                left[i] = rng_next();
                right[i] = rng_next();
            }
            rck_mul256_raw_portable64(portable, left, right);
            rck_mul256_raw_bmi2(assembly, left, right);
            if (memcmp(portable, assembly, sizeof(portable)) != 0) {
                fail("x86_64 BMI2 product differs from portable C");
            }
        }
    }
#endif
#endif
}

static void test_field(void) {
    unsigned trial;
    rck_fe one;
    rck_fe zero;
    rck_fe_set_u32(&one, 1);
    rck_fe_set_u32(&zero, 0);

    for (trial = 0; trial < 200; ++trial) {
        rck_fe a;
        rck_fe b;
        rck_fe sum;
        rck_fe square;
        rck_fe restored;
        rck_fe product;
        rck_fe inverse;
        random_fe(&a);
        random_fe(&b);

        rck_fe_add(&sum, &a, &b);
        rck_fe_sub(&restored, &sum, &b);
        if (!rck_fe_equal(&restored, &a)) {
            fail("(a+b)-b != a");
        }

        rck_fe_mul(&product, &a, &one);
        if (!rck_fe_equal(&product, &a)) {
            fail("a*1 != a");
        }
        rck_fe_mul(&product, &a, &zero);
        if (!rck_fe_is_zero(&product)) {
            fail("a*0 != 0");
        }
        rck_fe_sqr(&square, &a);
        rck_fe_mul(&product, &a, &a);
        if (!rck_fe_equal(&square, &product)) {
            fail("square differs from multiplication");
        }
        if (!rck_fe_is_zero(&a)) {
            rck_fe_inv(&inverse, &a);
            rck_fe_mul(&product, &a, &inverse);
            if (!rck_fe_equal(&product, &one)) {
                fail("a*inverse(a) != 1");
            }
        }
    }
    {
        rck_fe values[17];
        rck_fe inverses[17];
        rck_fe scratch[17];
        rck_fe product;
        unsigned index;
        for (index = 0; index < 17; ++index) {
            random_fe(&values[index]);
            if (rck_fe_is_zero(&values[index])) {
                rck_fe_set_u32(&values[index], index + 1);
            }
        }
        if (!rck_fe_batch_inv(inverses, values, scratch, 17)) {
            fail("batch inversion rejected nonzero values");
        }
        for (index = 0; index < 17; ++index) {
            rck_fe_mul(&product, &values[index], &inverses[index]);
            if (!rck_fe_equal(&product, &one)) {
                fail("batch inverse is incorrect");
            }
        }
    }
}

static void test_points_and_dual(void) {
    uint32_t scalar;
    rck_point jump;
    rck_point_mul_u32(&jump, &RCK_GENERATOR, 17);

    for (scalar = 2; scalar < 80; ++scalar) {
        rck_point point;
        rck_point expected_primary;
        rck_point expected_alternate;
        rck_point preinverted_add;
        rck_point signed_jump;
        rck_point opposite_jump;
            rck_dual_result result;
            rck_dual_result preinverted_result;
            rck_fe dx;
            rck_fe inverse_dx;
            rck_select_mode mode;

        if (scalar == 17) {
            continue;
        }
        rck_point_mul_u32(&point, &RCK_GENERATOR, scalar);
        signed_jump = jump;
        if (point.y.limb[0] & 1U) {
            rck_point_neg(&signed_jump, &signed_jump);
        }
        rck_point_neg(&opposite_jump, &signed_jump);
        rck_point_add(&expected_primary, &point, &signed_jump);
        rck_point_add(&expected_alternate, &point, &opposite_jump);
        rck_fe_sub(&dx, &point.x, &signed_jump.x);
        rck_fe_inv(&inverse_dx, &dx);
        if (!rck_point_add_preinverted(
                &preinverted_add,
                &point,
                &signed_jump,
                &inverse_dx
            ) ||
            !point_equal(&preinverted_add, &expected_primary)) {
            fail("preinverted point addition differs");
        }

        for (mode = RCK_SELECT_BASELINE;
             mode <= RCK_SELECT_FULL_GATED2;
             mode = (rck_select_mode)(mode + 1)) {
            const rck_point* expected;
            if (!rck_point_dual_step(
                    &result, &point, &jump, mode
                )) {
                fail("dual step rejected ordinary points");
            }
            rck_fe_sub(&dx, &point.x, &jump.x);
            rck_fe_inv(&inverse_dx, &dx);
            if (!rck_point_dual_step_preinverted(
                    &preinverted_result,
                    &point,
                    &jump,
                    &inverse_dx,
                    mode
                ) ||
                !point_equal(
                    &preinverted_result.chosen,
                    &result.chosen
                )) {
                fail("preinverted dual step differs");
            }
            expected = result.selected_alternate
                ? &expected_alternate
                : &expected_primary;
            if (!point_equal(&result.chosen, expected)) {
                fail("dual step differs from ordinary point addition");
            }
            if (result.evaluated_alternate) {
                const rck_point* rejected =
                    result.selected_alternate
                    ? &expected_primary
                    : &expected_alternate;
                if (!rck_fe_equal(&result.rejected_x, &rejected->x)) {
                    fail("shadow x does not match rejected point");
                }
            }
        }
    }
}

static void test_generator_batch(void) {
    uint64_t scalars[64];
    rck_point points[64];
    unsigned index;
    scalars[0] = 0;
    scalars[1] = 1;
    for (index = 2; index < 64; ++index) {
        scalars[index] = rng_next() & UINT64_C(0x000000ffffffffff);
    }
    if (!rck_generator_mul_batch(points, scalars, 64)) {
        fail("generator batch multiplication failed");
    }
    for (index = 0; index < 64; ++index) {
        rck_point expected;
        rck_point_mul_u64(&expected, &RCK_GENERATOR, scalars[index]);
        if (!point_equal(&points[index], &expected)) {
            fail("generator batch multiplication differs");
        }
    }
    if (!rck_generator_sequence_batch(points, UINT64_C(0x12345678), 64)) {
        fail("generator sequence batch failed");
    }
    for (index = 0; index < 64; ++index) {
        rck_point expected;
        rck_point_mul_u64(
            &expected,
            &RCK_GENERATOR,
            UINT64_C(0x12345678) + index
        );
        if (!point_equal(&points[index], &expected)) {
            fail("generator sequence batch differs");
        }
    }
}

int main(void) {
    test_raw_assembly();
    test_field();
    test_points_and_dual();
    test_generator_batch();
    puts("portable C + x86_64 assembly tests: PASS");
    return 0;
}
