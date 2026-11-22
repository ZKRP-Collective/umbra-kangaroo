#ifndef RCK_PORTABLE_H
#define RCK_PORTABLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t limb[8];
} rck_fe;

typedef struct {
    rck_fe x;
    rck_fe y;
    uint32_t infinity;
} rck_point;

typedef enum {
    RCK_SELECT_BASELINE = 0,
    RCK_SELECT_LITE2 = 1,
    RCK_SELECT_MID2 = 2,
    RCK_SELECT_FULL_GATED2 = 3
} rck_select_mode;

typedef struct {
    rck_point chosen;
    rck_fe rejected_x;
    uint32_t evaluated_alternate;
    uint32_t selected_alternate;
} rck_dual_result;

extern uint64_t rck_mul_count;
extern const rck_fe RCK_FIELD_P;
extern const rck_point RCK_GENERATOR;

uint32_t rck_add256_raw_portable(
    uint32_t out[8],
    const uint32_t left[8],
    const uint32_t right[8]
);
#if defined(__SIZEOF_INT128__)
void rck_mul256_raw_portable64(
    uint64_t out[8],
    const uint64_t left[4],
    const uint64_t right[4]
);
#endif

#if defined(__x86_64__) || defined(_M_X64)
uint32_t rck_add256_raw_asm(
    uint32_t out[8],
    const uint32_t left[8],
    const uint32_t right[8]
);
void rck_mul256_raw_bmi2(
    uint64_t out[8],
    const uint64_t left[4],
    const uint64_t right[4]
);
#endif

void rck_fe_set_u32(rck_fe* out, uint32_t value);
int rck_fe_equal(const rck_fe* left, const rck_fe* right);
int rck_fe_is_zero(const rck_fe* value);
void rck_fe_add(rck_fe* out, const rck_fe* left, const rck_fe* right);
void rck_fe_sub(rck_fe* out, const rck_fe* left, const rck_fe* right);
void rck_fe_neg(rck_fe* out, const rck_fe* value);
void rck_fe_mul(rck_fe* out, const rck_fe* left, const rck_fe* right);
void rck_fe_sqr(rck_fe* out, const rck_fe* value);
void rck_fe_inv(rck_fe* out, const rck_fe* value);
int rck_fe_batch_inv(
    rck_fe* out,
    const rck_fe* values,
    rck_fe* prefix_scratch,
    size_t count
);

void rck_point_neg(rck_point* out, const rck_point* point);
void rck_point_add(
    rck_point* out,
    const rck_point* left,
    const rck_point* right
);
int rck_point_add_preinverted(
    rck_point* out,
    const rck_point* left,
    const rck_point* right,
    const rck_fe* inverse_dx
);
void rck_point_double(rck_point* out, const rck_point* point);
void rck_point_mul_u32(rck_point* out, const rck_point* point, uint32_t scalar);
void rck_point_mul_u64(rck_point* out, const rck_point* point, uint64_t scalar);
int rck_generator_mul_batch(
    rck_point* out,
    const uint64_t* scalars,
    size_t count
);
int rck_generator_sequence_batch(
    rck_point* out,
    uint64_t first_scalar,
    size_t count
);

int rck_point_dual_step(
    rck_dual_result* out,
    const rck_point* point,
    const rck_point* jump,
    rck_select_mode mode
);
int rck_point_dual_step_preinverted(
    rck_dual_result* out,
    const rck_point* point,
    const rck_point* jump,
    const rck_fe* inverse_dx,
    rck_select_mode mode
);

#ifdef __cplusplus
}
#endif

#endif
