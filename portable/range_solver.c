#include "rck_portable.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(RCK_USE_OPENMP)
#include <omp.h>
#endif

#ifndef WALK_HISTORY
#define WALK_HISTORY 16U
#endif
#ifndef RCK_JUMP_COUNT
#define RCK_JUMP_COUNT 64U
#endif
#ifndef RCK_WALKER_LIMIT
#define RCK_WALKER_LIMIT 128U
#endif
#ifndef RCK_DP_BUFFER_CAPACITY
#define RCK_DP_BUFFER_CAPACITY 256U
#endif

#if RCK_JUMP_COUNT == 0 || \
    (RCK_JUMP_COUNT & (RCK_JUMP_COUNT - 1U)) != 0
#error "RCK_JUMP_COUNT must be a power of two"
#endif
#if RCK_WALKER_LIMIT < 8
#error "RCK_WALKER_LIMIT must be at least eight"
#endif
#if RCK_DP_BUFFER_CAPACITY == 0 || \
    (RCK_DP_BUFFER_CAPACITY & (RCK_DP_BUFFER_CAPACITY - 1U)) != 0
#error "RCK_DP_BUFFER_CAPACITY must be a power of two"
#endif

#if defined(RCK_OVERBOOKED_JUMPS)
#define RCK_JUMPS_PER_WALKER 2U
#ifndef RCK_SECOND_JUMP_SCORE_MIN
#define RCK_SECOND_JUMP_SCORE_MIN 1U
#endif
#if RCK_SECOND_JUMP_SCORE_MIN > 3
#error "RCK_SECOND_JUMP_SCORE_MIN must be between zero and three"
#endif
#else
#define RCK_JUMPS_PER_WALKER 1U
#endif
#define RCK_HOT_BATCH_LIMIT (RCK_WALKER_LIMIT * RCK_JUMPS_PER_WALKER)

typedef struct {
    rck_point point;
    int64_t distance;
    uint32_t tame;
    uint64_t history[WALK_HISTORY];
    uint32_t history_position;
    uint32_t history_size;
    uint32_t pending_escape;
    uint32_t pending_escape_index;
    int64_t pending_escape_delta;
} walker;

typedef struct {
#if defined(RCK_COMPACT_DP)
    uint64_t x_fingerprint[2];
#else
    rck_fe x;
#endif
    int64_t distance;
    uint32_t tame;
    uint32_t occupied;
} dp_record;

#if defined(RCK_COMPACT_DP)
_Static_assert(sizeof(dp_record) == 32, "compact DP record must be 32 bytes");
#endif

typedef struct {
    uint64_t steps;
    uint64_t distinguished;
    uint64_t shadows;
    uint64_t alternate_evaluations;
    uint64_t alternate_selections;
    uint64_t second_jump_evaluations;
    uint64_t second_jump_selections;
    uint64_t collisions;
    uint64_t loop_escapes;
} solve_stats;

#if defined(RCK_BUFFERED_DP)
typedef struct {
    rck_fe x;
    uint64_t hash;
    int64_t distance;
    uint32_t tame;
    uint32_t shadow;
} buffered_dp;

static buffered_dp dp_buffer[RCK_DP_BUFFER_CAPACITY];
static buffered_dp dp_buffer_scratch[RCK_DP_BUFFER_CAPACITY];
static size_t dp_buffer_count;
#endif

static uint64_t rng_state;

static uint64_t mix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) *
        UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) *
        UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static uint64_t rng_next(void) {
    rng_state += UINT64_C(0x9e3779b97f4a7c15);
    return mix64(rng_state);
}

static uint64_t rng_bounded(uint64_t bound) {
    return bound == 0 ? 0 : rng_next() % bound;
}

static int hex_nibble(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

static int fe_from_hex(rck_fe* out, const char* text) {
    size_t limb;
    if (strlen(text) != 64) {
        return 0;
    }
    memset(out, 0, sizeof(*out));
    for (limb = 0; limb < 8; ++limb) {
        uint32_t value = 0;
        size_t digit;
        const size_t offset = 64 - (limb + 1) * 8;
        for (digit = 0; digit < 8; ++digit) {
            const int nibble = hex_nibble(text[offset + digit]);
            if (nibble < 0) {
                return 0;
            }
            value = (value << 4) | (uint32_t)nibble;
        }
        out->limb[limb] = value;
    }
    return 1;
}

static void fe_pow_sqrt(rck_fe* out, const rck_fe* value) {
    static const uint32_t exponent[8] = {
        0xbfffff0cU, 0xffffffffU, 0xffffffffU, 0xffffffffU,
        0xffffffffU, 0xffffffffU, 0xffffffffU, 0x3fffffffU
    };
    rck_fe result;
    int bit;
    rck_fe_set_u32(&result, 1);
    for (bit = 253; bit >= 0; --bit) {
        rck_fe_sqr(&result, &result);
        if ((exponent[bit / 32] >> (bit % 32)) & 1U) {
            rck_fe_mul(&result, &result, value);
        }
    }
    *out = result;
}

static int point_on_curve(const rck_point* point) {
    rck_fe left;
    rck_fe right;
    rck_fe seven;
    rck_fe_sqr(&left, &point->y);
    rck_fe_sqr(&right, &point->x);
    rck_fe_mul(&right, &right, &point->x);
    rck_fe_set_u32(&seven, 7);
    rck_fe_add(&right, &right, &seven);
    return rck_fe_equal(&left, &right);
}

static int point_from_hex(rck_point* out, const char* text) {
    const size_t length = strlen(text);
    char coordinate[65];
    int prefix;
    memset(out, 0, sizeof(*out));
    coordinate[64] = '\0';
    if (length == 130 && text[0] == '0' && text[1] == '4') {
        memcpy(coordinate, text + 2, 64);
        if (!fe_from_hex(&out->x, coordinate)) {
            return 0;
        }
        memcpy(coordinate, text + 66, 64);
        if (!fe_from_hex(&out->y, coordinate)) {
            return 0;
        }
        return point_on_curve(out);
    }
    if (length != 66 || text[0] != '0' ||
        (text[1] != '2' && text[1] != '3')) {
        return 0;
    }
    prefix = text[1] - '0';
    memcpy(coordinate, text + 2, 64);
    if (!fe_from_hex(&out->x, coordinate)) {
        return 0;
    }
    {
        rck_fe rhs;
        rck_fe seven;
        rck_fe check;
        rck_fe_sqr(&rhs, &out->x);
        rck_fe_mul(&rhs, &rhs, &out->x);
        rck_fe_set_u32(&seven, 7);
        rck_fe_add(&rhs, &rhs, &seven);
        fe_pow_sqrt(&out->y, &rhs);
        rck_fe_sqr(&check, &out->y);
        if (!rck_fe_equal(&check, &rhs)) {
            return 0;
        }
        if ((int)(out->y.limb[0] & 1U) != (prefix & 1)) {
            rck_fe_neg(&out->y, &out->y);
        }
    }
    return 1;
}

static int point_equal(const rck_point* left, const rck_point* right) {
    return left->infinity == right->infinity &&
        (left->infinity ||
         (rck_fe_equal(&left->x, &right->x) &&
          rck_fe_equal(&left->y, &right->y)));
}

static uint64_t x_fingerprint(const rck_fe* x) {
    return (uint64_t)x->limb[0] |
        ((uint64_t)x->limb[1] << 32);
}

static int history_contains(const walker* value, const rck_fe* x) {
    const uint64_t fingerprint = x_fingerprint(x);
    uint32_t index;
    for (index = 0; index < value->history_size; ++index) {
        if (value->history[index] == fingerprint) {
            return 1;
        }
    }
    return 0;
}

static void history_push(walker* value, const rck_fe* x) {
    value->history[value->history_position] = x_fingerprint(x);
    value->history_position =
        (value->history_position + 1U) % WALK_HISTORY;
    if (value->history_size < WALK_HISTORY) {
        ++value->history_size;
    }
}

static uint64_t integer_sqrt_ceil(uint64_t value) {
    uint64_t low = 0;
    uint64_t high = UINT64_C(1) << 32;
    while (low + 1 < high) {
        const uint64_t middle = low + (high - low) / 2;
        if (middle <= value / middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low * low == value ? low : low + 1;
}

static uint64_t hash_x(const rck_fe* x) {
    const uint64_t high_sample =
        (uint64_t)x->limb[4] |
        ((uint64_t)x->limb[5] << 32);
    return mix64(
        x_fingerprint(x) ^ high_sample ^
        UINT64_C(0x6a09e667f3bcc909)
    );
}

#if defined(RCK_COMPACT_DP)
static void record_x_fingerprint(dp_record* record, const rck_fe* x) {
    record->x_fingerprint[0] =
        (uint64_t)x->limb[0] | ((uint64_t)x->limb[1] << 32);
    record->x_fingerprint[1] =
        (uint64_t)x->limb[6] | ((uint64_t)x->limb[7] << 32);
}

static int record_x_matches(const dp_record* record, const rck_fe* x) {
    return record->x_fingerprint[0] ==
               ((uint64_t)x->limb[0] | ((uint64_t)x->limb[1] << 32)) &&
           record->x_fingerprint[1] ==
               ((uint64_t)x->limb[6] | ((uint64_t)x->limb[7] << 32));
}
#endif

static int distinguished(const rck_fe* x, unsigned bits) {
    const uint32_t mask = bits == 0 ? 0U : (UINT32_C(1) << bits) - 1U;
    return bits == 0 || (x->limb[5] & mask) == 0;
}

static int verify_candidate(
    uint64_t shifted,
    uint64_t edge,
    uint64_t low,
    uint64_t high,
    const rck_point* target,
    uint64_t* solution
) {
    uint64_t offset;
    uint64_t candidate;
    rck_point check;
    if (shifted < edge) {
        return 0;
    }
    offset = shifted - edge;
    if (offset > high - low || low > UINT64_MAX - offset) {
        return 0;
    }
    candidate = low + offset;
    rck_point_mul_u64(&check, &RCK_GENERATOR, candidate);
    if (!point_equal(&check, target)) {
        return 0;
    }
    *solution = candidate;
    return 1;
}

static int recover_collision(
    const dp_record* old,
    int64_t distance,
    uint32_t tame,
    uint64_t edge,
    uint64_t low,
    uint64_t high,
    const rck_point* target,
    uint64_t* solution
) {
    int64_t candidates[2];
    size_t count = 0;
    size_t index;
    if (old->tame != tame) {
        const int64_t tame_distance = old->tame ?
            old->distance : distance;
        const int64_t wild_distance = old->tame ?
            distance : old->distance;
        candidates[count++] = wild_distance - tame_distance;
        candidates[count++] = wild_distance + tame_distance;
    } else if (!tame && old->distance != distance) {
        const int64_t sum = old->distance + distance;
        if ((sum & 1) == 0) {
            candidates[count++] = sum / 2;
        }
    }
    for (index = 0; index < count; ++index) {
        if (candidates[index] >= 0 &&
            verify_candidate(
                (uint64_t)candidates[index],
                edge,
                low,
                high,
                target,
                solution
            )) {
            return 1;
        }
    }
    return 0;
}

static int observe_insert(
    dp_record* table,
    size_t capacity,
    const rck_fe* x,
    int64_t distance,
    uint32_t tame,
    uint64_t edge,
    uint64_t low,
    uint64_t high,
    const rck_point* target,
    uint64_t* solution,
    solve_stats* stats
) {
    size_t slot;
    slot = (size_t)hash_x(x) & (capacity - 1);
    for (;;) {
        dp_record* record = &table[slot];
        if (!record->occupied) {
#if defined(RCK_COMPACT_DP)
            record_x_fingerprint(record, x);
#else
            record->x = *x;
#endif
            record->distance = distance;
            record->tame = tame;
            record->occupied = 1;
            return 0;
        }
#if defined(RCK_COMPACT_DP)
        if (record_x_matches(record, x)) {
#else
        if (rck_fe_equal(&record->x, x)) {
#endif
            ++stats->collisions;
            return recover_collision(
                record,
                distance,
                tame,
                edge,
                low,
                high,
                target,
                solution
            );
        }
        slot = (slot + 1) & (capacity - 1);
    }
}

#if defined(RCK_BUFFERED_DP)
static int flush_buffered_dp(
    dp_record* table,
    size_t capacity,
    uint64_t edge,
    uint64_t low,
    uint64_t high,
    const rck_point* target,
    uint64_t* solution,
    solve_stats* stats
) {
    size_t count = dp_buffer_count;
    unsigned pass;
    size_t index;
    dp_buffer_count = 0;
    for (pass = 0; pass < 4; ++pass) {
        size_t offsets[256] = {0};
        size_t total = 0;
        for (index = 0; index < count; ++index) {
            ++offsets[(dp_buffer[index].hash >> (8 * pass)) & 255U];
        }
        for (index = 0; index < 256; ++index) {
            const size_t current = offsets[index];
            offsets[index] = total;
            total += current;
        }
        for (index = 0; index < count; ++index) {
            const size_t bucket =
                (dp_buffer[index].hash >> (8 * pass)) & 255U;
            dp_buffer_scratch[offsets[bucket]++] = dp_buffer[index];
        }
        memcpy(dp_buffer, dp_buffer_scratch, count * sizeof(*dp_buffer));
    }
    for (index = 0; index < count; ++index) {
        const buffered_dp* value = &dp_buffer[index];
        if (observe_insert(
                table,
                capacity,
                &value->x,
                value->distance,
                value->tame,
                edge,
                low,
                high,
                target,
                solution,
                stats
            )) {
            return 1;
        }
    }
    return 0;
}
#endif

static int observe(
    dp_record* table,
    size_t capacity,
    const rck_fe* x,
    int64_t distance,
    uint32_t tame,
    unsigned dp_bits,
    uint32_t shadow,
    uint64_t edge,
    uint64_t low,
    uint64_t high,
    const rck_point* target,
    uint64_t* solution,
    solve_stats* stats
) {
    if (!distinguished(x, dp_bits)) {
        return 0;
    }
    ++stats->distinguished;
    stats->shadows += shadow;
#if defined(RCK_BUFFERED_DP)
    dp_buffer[dp_buffer_count].x = *x;
    dp_buffer[dp_buffer_count].hash = hash_x(x);
    dp_buffer[dp_buffer_count].distance = distance;
    dp_buffer[dp_buffer_count].tame = tame;
    dp_buffer[dp_buffer_count].shadow = shadow;
    ++dp_buffer_count;
    if (dp_buffer_count == RCK_DP_BUFFER_CAPACITY) {
        return flush_buffered_dp(
            table, capacity, edge, low, high, target, solution, stats
        );
    }
    return 0;
#else
    return observe_insert(
        table, capacity, x, distance, tame, edge, low, high, target, solution,
        stats
    );
#endif
}

static int evaluate_walk_step(
    rck_dual_result* out,
    size_t* chosen_jump,
    uint32_t* evaluated_second_jump,
    uint32_t* selected_second_jump,
    const rck_point* point,
    const rck_point* jumps,
    const size_t candidate_jumps[2],
    const rck_fe* const candidate_inverses[2],
    size_t candidate_count,
    rck_select_mode mode
) {
    rck_dual_result candidate;
    uint32_t best_score = UINT32_MAX;
    size_t candidate_index;
    int found = 0;
    *evaluated_second_jump = 0;
    *selected_second_jump = 0;
    for (candidate_index = 0;
         candidate_index < candidate_count;
         ++candidate_index) {
        const size_t jump_index = candidate_jumps[candidate_index];
        uint32_t score;
#if defined(RCK_OVERBOOKED_JUMPS)
        if (candidate_index != 0 &&
            best_score < RCK_SECOND_JUMP_SCORE_MIN) {
            break;
        }
#endif
        if (candidate_index != 0) {
            *evaluated_second_jump = 1;
        }
        if (!rck_point_dual_step_preinverted(
                &candidate,
                point,
                &jumps[jump_index],
                candidate_inverses[candidate_index],
                mode
            )) {
            continue;
        }
        score = candidate.chosen.x.limb[0] & 3U;
        if (!found || score < best_score) {
            *out = candidate;
            *chosen_jump = jump_index;
            best_score = score;
            *selected_second_jump = (uint32_t)(candidate_index != 0);
            found = 1;
        }
    }
    return found;
}

static int solve_range(
    const rck_point* target,
    uint64_t low,
    uint64_t high,
    uint64_t seed,
    rck_select_mode mode,
    uint32_t shadow_dp,
    uint64_t* solution,
    solve_stats* total_stats
) {
    const uint64_t range = high - low + 1;
    const uint64_t root = integer_sqrt_ceil(range);
    const uint64_t edge = range / 32 > 2 ? range / 32 : 2;
    const uint64_t tame_range = edge;
    const uint64_t wild_range = range + 2 * edge;
    const size_t jump_count = RCK_JUMP_COUNT;
    size_t walker_count = root / 2;
    unsigned dp_bits;
    unsigned restart;
    rck_point low_point;
    rck_point shifted_target;
    rck_point edge_point;

    if (walker_count < 8) {
        walker_count = 8;
    }
    if (walker_count > RCK_WALKER_LIMIT) {
        walker_count = RCK_WALKER_LIMIT;
    }
#if defined(RCK_DP_BITS)
    dp_bits = RCK_DP_BITS;
#else
    dp_bits = root < 256 ? 0U : root < 4096 ? 2U : 4U;
#endif

    rck_point_mul_u64(&low_point, &RCK_GENERATOR, low);
    rck_point_neg(&low_point, &low_point);
    rck_point_add(&shifted_target, target, &low_point);
    rck_point_mul_u64(&edge_point, &RCK_GENERATOR, edge);
    rck_point_add(&shifted_target, &shifted_target, &edge_point);

    memset(total_stats, 0, sizeof(*total_stats));
    for (restart = 0; restart < 4; ++restart) {
        walker* walkers;
        rck_point* jumps;
        rck_point* escape_jumps;
        uint64_t* jump_sizes;
        uint64_t* escape_sizes;
        dp_record* table;
        const uint64_t max_steps = 8 * root;
        size_t capacity = 1024;
        size_t tame_count = walker_count / 3;
        rck_fe denominators[RCK_HOT_BATCH_LIMIT];
        rck_fe inverses[RCK_HOT_BATCH_LIMIT];
        rck_fe prefixes[RCK_HOT_BATCH_LIMIT];
        size_t batch_jumps[RCK_WALKER_LIMIT];
#if defined(RCK_OVERBOOKED_JUMPS)
        size_t batch_secondary_jumps[RCK_WALKER_LIMIT];
#endif
        uint32_t batch_valid[RCK_WALKER_LIMIT];
        uint32_t batch_is_escape[RCK_WALKER_LIMIT];
        uint64_t setup_scalars[
            2 * RCK_JUMP_COUNT + RCK_WALKER_LIMIT
        ];
        rck_point setup_points[
            2 * RCK_JUMP_COUNT + RCK_WALKER_LIMIT
        ];
        rck_dual_result batch_results[RCK_WALKER_LIMIT];
        rck_point batch_escape_points[RCK_WALKER_LIMIT];
        uint32_t batch_result_kind[RCK_WALKER_LIMIT];
        uint32_t batch_evaluated_second_jump[RCK_WALKER_LIMIT];
        uint32_t batch_selected_second_jump[RCK_WALKER_LIMIT];
#if defined(RCK_USE_OPENMP) && defined(RCK_PARALLEL_BATCH_INV)
        rck_fe chunk_products[RCK_WALKER_LIMIT];
        rck_fe chunk_inverses[RCK_WALKER_LIMIT];
        rck_fe chunk_prefixes[RCK_WALKER_LIMIT];
        uint32_t chunk_valid[RCK_WALKER_LIMIT];
#endif
        uint64_t step;
        size_t index;

        while (capacity < (size_t)(max_steps >> dp_bits) * 4 +
               walker_count * 4) {
            capacity <<= 1;
        }
        walkers = (walker*)calloc(walker_count, sizeof(*walkers));
        jumps = (rck_point*)calloc(jump_count, sizeof(*jumps));
        escape_jumps =
            (rck_point*)calloc(jump_count, sizeof(*escape_jumps));
        jump_sizes = (uint64_t*)calloc(jump_count, sizeof(*jump_sizes));
        escape_sizes =
            (uint64_t*)calloc(jump_count, sizeof(*escape_sizes));
        table = (dp_record*)calloc(capacity, sizeof(*table));
        if (walkers == NULL || jumps == NULL || escape_jumps == NULL ||
            jump_sizes == NULL || escape_sizes == NULL || table == NULL) {
            free(walkers);
            free(jumps);
            free(escape_jumps);
            free(jump_sizes);
            free(escape_sizes);
            free(table);
            return -1;
        }

        rng_state = mix64(seed ^ restart);
#if defined(RCK_BUFFERED_DP)
        dp_buffer_count = 0;
#endif
        for (index = 0; index < jump_count; ++index) {
            uint64_t jump = 8 * root + rng_bounded(8 * root);
            jump &= ~UINT64_C(1);
            if (jump == 0) {
                jump = 2;
            }
            jump_sizes[index] = jump;
            setup_scalars[2 * index] = jump;
            {
                const uint64_t escape_min =
                    range / 1024 > 2 ? range / 1024 : 2;
                uint64_t escape =
                    escape_min + rng_bounded(escape_min);
                escape &= ~UINT64_C(1);
                if (escape == 0) {
                    escape = 2;
                }
                escape_sizes[index] = escape;
                setup_scalars[2 * index + 1] = escape;
            }
        }
        for (index = 0; index < walker_count; ++index) {
            uint64_t distance;
            walkers[index].tame = index < tame_count;
            if (walkers[index].tame) {
                distance = 1 + rng_bounded(tame_range - 1);
            } else {
                distance = rng_bounded(wild_range) & ~UINT64_C(1);
            }
            walkers[index].distance = (int64_t)distance;
            setup_scalars[2 * jump_count + index] = distance;
        }
        if (!rck_generator_mul_batch(
                setup_points,
                setup_scalars,
                2 * jump_count + walker_count
            )) {
            free(walkers);
            free(jumps);
            free(escape_jumps);
            free(jump_sizes);
            free(escape_sizes);
            free(table);
            return -1;
        }
        for (index = 0; index < jump_count; ++index) {
            jumps[index] = setup_points[2 * index];
            escape_jumps[index] = setup_points[2 * index + 1];
        }
        {
            rck_point negative_target;
            rck_point_neg(&negative_target, &shifted_target);
            for (index = 0; index < walker_count; ++index) {
                walkers[index].point =
                    setup_points[2 * jump_count + index];
                batch_valid[index] =
                    !walkers[index].tame &&
                    !walkers[index].point.infinity &&
                    !rck_fe_equal(
                        &walkers[index].point.x,
                        &negative_target.x
                    );
                if (batch_valid[index]) {
                    rck_fe_sub(
                        &denominators[index],
                        &walkers[index].point.x,
                        &negative_target.x
                    );
                } else {
                    rck_fe_set_u32(&denominators[index], 1);
                }
            }
            if (!rck_fe_batch_inv(
                    inverses,
                    denominators,
                    prefixes,
                    walker_count
                )) {
                free(walkers);
                free(jumps);
                free(escape_jumps);
                free(jump_sizes);
                free(escape_sizes);
                free(table);
                return -1;
            }
            for (index = 0; index < walker_count; ++index) {
                if (walkers[index].tame) {
                    continue;
                }
                if (batch_valid[index]) {
                    if (!rck_point_add_preinverted(
                            &walkers[index].point,
                            &walkers[index].point,
                            &negative_target,
                            &inverses[index]
                        )) {
                        free(walkers);
                        free(jumps);
                        free(escape_jumps);
                        free(jump_sizes);
                        free(escape_sizes);
                        free(table);
                        return -1;
                    }
                } else {
                    rck_point_add(
                        &walkers[index].point,
                        &walkers[index].point,
                        &negative_target
                    );
                }
            }
        }
        for (index = 0; index < walker_count; ++index) {
            if (!walkers[index].tame) {
                if (walkers[index].point.infinity &&
                    verify_candidate(
                        (uint64_t)walkers[index].distance,
                        edge,
                        low,
                        high,
                        target,
                        solution
                    )) {
                    free(walkers);
                    free(jumps);
                    free(escape_jumps);
                    free(jump_sizes);
                    free(escape_sizes);
                    free(table);
                    return 1;
                }
            }
            if (!walkers[index].point.infinity) {
                history_push(&walkers[index], &walkers[index].point.x);
            }
        }

        for (step = 0; step < max_steps;) {
            const size_t batch_count =
                max_steps - step < walker_count ?
                (size_t)(max_steps - step) : walker_count;
            const size_t inversion_count =
                batch_count * RCK_JUMPS_PER_WALKER;
            size_t batch;
            for (batch = 0; batch < batch_count; ++batch) {
                walker* current =
                    &walkers[(step + batch) % walker_count];
                const size_t denominator_index =
                    batch * RCK_JUMPS_PER_WALKER;
                size_t attempts = 0;
                size_t jump_index;
                batch_is_escape[batch] = current->pending_escape;
                batch_valid[batch] = !current->point.infinity;
                if (!batch_valid[batch]) {
                    rck_fe_set_u32(
                        &denominators[denominator_index],
                        1
                    );
#if defined(RCK_OVERBOOKED_JUMPS)
                    rck_fe_set_u32(
                        &denominators[denominator_index + 1],
                        1
                    );
                    batch_secondary_jumps[batch] = 0;
#endif
                    batch_jumps[batch] = 0;
                    continue;
                }
                if (current->pending_escape) {
                    jump_index = current->pending_escape_index;
                    batch_jumps[batch] = jump_index;
                    if (rck_fe_equal(
                            &current->point.x,
                            &escape_jumps[jump_index].x
                        )) {
                        batch_valid[batch] = 0;
                        rck_fe_set_u32(
                            &denominators[denominator_index],
                            1
                        );
#if defined(RCK_OVERBOOKED_JUMPS)
                        rck_fe_set_u32(
                            &denominators[denominator_index + 1],
                            1
                        );
                        batch_secondary_jumps[batch] = 0;
#endif
                        continue;
                    }
                    rck_fe_sub(
                        &denominators[denominator_index],
                        &current->point.x,
                        &escape_jumps[jump_index].x
                    );
#if defined(RCK_OVERBOOKED_JUMPS)
                    rck_fe_set_u32(
                        &denominators[denominator_index + 1],
                        1
                    );
                    batch_secondary_jumps[batch] = 0;
#endif
                    continue;
                }
                jump_index =
                    current->point.x.limb[0] & (jump_count - 1);
                while (rck_fe_equal(
                           &current->point.x,
                           &jumps[jump_index].x
                       ) &&
                       attempts < jump_count) {
                    jump_index = (jump_index + 1) & (jump_count - 1);
                    ++attempts;
                }
                if (attempts == jump_count) {
                    batch_valid[batch] = 0;
                    rck_fe_set_u32(
                        &denominators[denominator_index],
                        1
                    );
#if defined(RCK_OVERBOOKED_JUMPS)
                    rck_fe_set_u32(
                        &denominators[denominator_index + 1],
                        1
                    );
                    batch_secondary_jumps[batch] = 0;
#endif
                    batch_jumps[batch] = 0;
                    continue;
                }
                batch_jumps[batch] = jump_index;
                rck_fe_sub(
                    &denominators[denominator_index],
                    &current->point.x,
                    &jumps[jump_index].x
                );
#if defined(RCK_OVERBOOKED_JUMPS)
                attempts = 0;
                jump_index =
                    current->point.x.limb[2] & (jump_count - 1);
                if (jump_index == batch_jumps[batch]) {
                    jump_index = (jump_index + 1) & (jump_count - 1);
                }
                while (rck_fe_equal(
                           &current->point.x,
                           &jumps[jump_index].x
                       ) && attempts < jump_count) {
                    jump_index = (jump_index + 1) & (jump_count - 1);
                    ++attempts;
                }
                if (attempts == jump_count) {
                    rck_fe_set_u32(
                        &denominators[denominator_index + 1],
                        1
                    );
                    batch_secondary_jumps[batch] =
                        batch_jumps[batch];
                } else {
                    batch_secondary_jumps[batch] = jump_index;
                    rck_fe_sub(
                        &denominators[denominator_index + 1],
                        &current->point.x,
                        &jumps[jump_index].x
                    );
                }
#endif
            }
#if defined(RCK_USE_OPENMP) && defined(RCK_PARALLEL_BATCH_INV)
            {
                int inversion_ok = 1;
                int requested_threads = omp_get_max_threads();
                if (requested_threads > (int)inversion_count) {
                    requested_threads = (int)inversion_count;
                }
                if (requested_threads > (int)RCK_WALKER_LIMIT) {
                    requested_threads = (int)RCK_WALKER_LIMIT;
                }
#pragma omp parallel num_threads(requested_threads) shared(inversion_ok)
                {
                    const int thread = omp_get_thread_num();
                    const int thread_count = omp_get_num_threads();
                    const size_t begin =
                        inversion_count * (size_t)thread /
                        (size_t)thread_count;
                    const size_t end =
                        inversion_count * (size_t)(thread + 1) /
                        (size_t)thread_count;
                    rck_fe accumulated;
                    size_t position;
                    rck_fe_set_u32(&accumulated, 1);
                    chunk_valid[thread] = 1;
                    for (position = begin; position < end; ++position) {
                        if (rck_fe_is_zero(&denominators[position])) {
                            chunk_valid[thread] = 0;
                        }
                        prefixes[position] = accumulated;
                        rck_fe_mul(
                            &accumulated,
                            &accumulated,
                            &denominators[position]
                        );
                    }
                    chunk_products[thread] = accumulated;
#pragma omp barrier
#pragma omp single
                    {
                        int chunk;
                        for (chunk = 0; chunk < thread_count; ++chunk) {
                            if (!chunk_valid[chunk]) {
                                inversion_ok = 0;
                            }
                        }
                        if (inversion_ok) {
                            inversion_ok = rck_fe_batch_inv(
                                chunk_inverses,
                                chunk_products,
                                chunk_prefixes,
                                (size_t)thread_count
                            );
                        }
                    }
#pragma omp barrier
                    if (inversion_ok) {
                        rck_fe inverse = chunk_inverses[thread];
                        for (position = end; position > begin; --position) {
                            const size_t current = position - 1;
                            rck_fe_mul(
                                &inverses[current],
                                &inverse,
                                &prefixes[current]
                            );
                            rck_fe_mul(
                                &inverse,
                                &inverse,
                                &denominators[current]
                            );
                        }
                    }
#pragma omp barrier
#pragma omp for schedule(static)
                    for (batch = 0; batch < batch_count; ++batch) {
                        walker* current =
                            &walkers[(step + batch) % walker_count];
                        const size_t jump_index = batch_jumps[batch];
                        const size_t denominator_index =
                            batch * RCK_JUMPS_PER_WALKER;
                        batch_result_kind[batch] = 0;
                        if (!batch_valid[batch] || !inversion_ok) {
                            continue;
                        }
                        if (batch_is_escape[batch]) {
                            rck_point signed_escape =
                                escape_jumps[jump_index];
                            if (current->pending_escape_delta < 0) {
                                rck_point_neg(
                                    &signed_escape,
                                    &signed_escape
                                );
                            }
                            if (rck_point_add_preinverted(
                                    &batch_escape_points[batch],
                                    &current->point,
                                    &signed_escape,
                                    &inverses[denominator_index]
                                )) {
                                batch_result_kind[batch] = 2;
                            }
                        } else {
                            size_t candidate_jumps[2];
                            const rck_fe* candidate_inverses[2];
                            candidate_jumps[0] = jump_index;
                            candidate_inverses[0] =
                                &inverses[denominator_index];
#if defined(RCK_OVERBOOKED_JUMPS)
                            candidate_jumps[1] =
                                batch_secondary_jumps[batch];
                            candidate_inverses[1] =
                                &inverses[denominator_index + 1];
#endif
                            if (evaluate_walk_step(
                                    &batch_results[batch],
                                    &batch_jumps[batch],
                                    &batch_evaluated_second_jump[batch],
                                    &batch_selected_second_jump[batch],
                                    &current->point,
                                    jumps,
                                    candidate_jumps,
                                    candidate_inverses,
                                    RCK_JUMPS_PER_WALKER,
                                    mode
                                )) {
                                batch_result_kind[batch] = 1;
                            }
                        }
                    }
                }
                if (!inversion_ok) {
                    free(walkers);
                    free(jumps);
                    free(escape_jumps);
                    free(jump_sizes);
                    free(escape_sizes);
                    free(table);
                    return -1;
                }
            }
#else
            if (!rck_fe_batch_inv(
                    inverses,
                    denominators,
                    prefixes,
                    inversion_count
                )) {
                free(walkers);
                free(jumps);
                free(escape_jumps);
                free(jump_sizes);
                free(escape_sizes);
                free(table);
                return -1;
            }
#if defined(RCK_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (batch = 0; batch < batch_count; ++batch) {
                walker* current =
                    &walkers[(step + batch) % walker_count];
                const size_t jump_index = batch_jumps[batch];
                const size_t denominator_index =
                    batch * RCK_JUMPS_PER_WALKER;
                batch_result_kind[batch] = 0;
                if (!batch_valid[batch]) {
                    continue;
                }
                if (batch_is_escape[batch]) {
                    rck_point signed_escape =
                        escape_jumps[jump_index];
                    if (current->pending_escape_delta < 0) {
                        rck_point_neg(&signed_escape, &signed_escape);
                    }
                    if (rck_point_add_preinverted(
                            &batch_escape_points[batch],
                            &current->point,
                            &signed_escape,
                            &inverses[denominator_index]
                        )) {
                        batch_result_kind[batch] = 2;
                    }
                } else {
                    size_t candidate_jumps[2];
                    const rck_fe* candidate_inverses[2];
                    candidate_jumps[0] = jump_index;
                    candidate_inverses[0] =
                        &inverses[denominator_index];
#if defined(RCK_OVERBOOKED_JUMPS)
                    candidate_jumps[1] =
                        batch_secondary_jumps[batch];
                    candidate_inverses[1] =
                        &inverses[denominator_index + 1];
#endif
                    if (evaluate_walk_step(
                            &batch_results[batch],
                            &batch_jumps[batch],
                            &batch_evaluated_second_jump[batch],
                            &batch_selected_second_jump[batch],
                            &current->point,
                            jumps,
                            candidate_jumps,
                            candidate_inverses,
                            RCK_JUMPS_PER_WALKER,
                            mode
                        )) {
                        batch_result_kind[batch] = 1;
                    }
                }
            }
#endif
            for (batch = 0; batch < batch_count; ++batch) {
                walker* current =
                    &walkers[(step + batch) % walker_count];
                const size_t jump_index = batch_jumps[batch];
                const rck_dual_result* result = &batch_results[batch];
                int64_t primary_delta;
                int64_t chosen_delta;
                int64_t rejected_delta;
                if (batch_result_kind[batch] == 2) {
                    current->point = batch_escape_points[batch];
                    current->distance +=
                        current->pending_escape_delta;
                    current->pending_escape = 0;
                    history_push(current, &current->point.x);
                    if (observe(
                            table,
                            capacity,
                            &current->point.x,
                            current->distance,
                            current->tame,
                            dp_bits,
                            0,
                            edge,
                            low,
                            high,
                            target,
                            solution,
                            total_stats
                        )) {
                        free(walkers);
                        free(jumps);
                        free(escape_jumps);
                        free(jump_sizes);
                        free(escape_sizes);
                        free(table);
                        return 1;
                    }
                    continue;
                }
                if (batch_result_kind[batch] != 1) {
                    continue;
                }
                primary_delta = (current->point.y.limb[0] & 1U) ?
                    -(int64_t)jump_sizes[jump_index] :
                    (int64_t)jump_sizes[jump_index];
                chosen_delta = result->selected_alternate ?
                    -primary_delta : primary_delta;
                rejected_delta = -chosen_delta;
                ++total_stats->steps;
                total_stats->alternate_evaluations +=
                    result->evaluated_alternate;
                total_stats->alternate_selections +=
                    result->selected_alternate;
                total_stats->second_jump_evaluations +=
                    batch_evaluated_second_jump[batch];
                total_stats->second_jump_selections +=
                    batch_selected_second_jump[batch];

                if (shadow_dp && result->evaluated_alternate &&
                    observe(
                        table,
                        capacity,
                        &result->rejected_x,
                        current->distance + rejected_delta,
                        current->tame,
                        dp_bits,
                        1,
                        edge,
                        low,
                        high,
                        target,
                        solution,
                        total_stats
                    )) {
                    free(walkers);
                    free(jumps);
                    free(escape_jumps);
                    free(jump_sizes);
                    free(escape_sizes);
                    free(table);
                    return 1;
                }
                if (history_contains(current, &result->chosen.x)) {
                    chosen_delta =
                        (current->point.y.limb[0] & 1U) ?
                        -(int64_t)escape_sizes[jump_index] :
                        (int64_t)escape_sizes[jump_index];
                    current->pending_escape = 1;
                    current->pending_escape_index =
                        (uint32_t)jump_index;
                    current->pending_escape_delta = chosen_delta;
                    ++total_stats->loop_escapes;
                    continue;
                } else {
                    current->point = result->chosen;
                }
                current->distance += chosen_delta;
                if (current->point.infinity) {
                    continue;
                }
                history_push(current, &current->point.x);
                if (observe(
                        table,
                        capacity,
                        &current->point.x,
                        current->distance,
                        current->tame,
                        dp_bits,
                        0,
                        edge,
                        low,
                        high,
                        target,
                        solution,
                        total_stats
                    )) {
                    free(walkers);
                    free(jumps);
                    free(escape_jumps);
                    free(jump_sizes);
                    free(escape_sizes);
                    free(table);
                    return 1;
                }
            }
            step += batch_count;
        }
#if defined(RCK_BUFFERED_DP)
        if (flush_buffered_dp(
                table,
                capacity,
                edge,
                low,
                high,
                target,
                solution,
                total_stats
            )) {
            free(walkers);
            free(jumps);
            free(escape_jumps);
            free(jump_sizes);
            free(escape_sizes);
            free(table);
            return 1;
        }
#endif
        free(walkers);
        free(jumps);
        free(escape_jumps);
        free(jump_sizes);
        free(escape_sizes);
        free(table);
    }
    return 0;
}

static int parse_u64_hex(const char* text, uint64_t* value) {
    char* end;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 16);
    if (errno != 0 || *text == '\0' || *end != '\0') {
        return 0;
    }
    *value = (uint64_t)parsed;
    return 1;
}

int main(int argc, char** argv) {
    rck_point target;
    uint64_t low;
    uint64_t high;
    uint64_t solution = 0;
    uint64_t root;
    solve_stats stats;
    rck_select_mode mode = RCK_SELECT_FULL_GATED2;
    uint32_t shadow_dp = 1;
    const char* mode_name = "full-gated+shadow-dp";
    uint64_t seed = UINT64_C(0x52434b616e676172);
    int status;
    if (argc < 4 || argc > 6) {
        fprintf(
            stderr,
            "usage: %s PUBLIC_KEY_HEX LOW_HEX HIGH_HEX"
            " [baseline|gated|lite|lite-shadow|full-shadow] [SEED_HEX]\n",
            argv[0]
        );
        return 2;
    }
    if (argc >= 5) {
        if (strcmp(argv[4], "baseline") == 0) {
            mode = RCK_SELECT_BASELINE;
            shadow_dp = 0;
            mode_name = "baseline";
        } else if (strcmp(argv[4], "gated") == 0) {
            shadow_dp = 0;
            mode_name = "gated";
        } else if (strcmp(argv[4], "lite-shadow") == 0) {
            mode = RCK_SELECT_LITE2;
            mode_name = "lite-shadow";
        } else if (strcmp(argv[4], "lite") == 0) {
            mode = RCK_SELECT_LITE2;
            shadow_dp = 0;
            mode_name = "lite";
        } else if (strcmp(argv[4], "full-shadow") != 0) {
            fputs("unknown mode\n", stderr);
            return 2;
        }
    }
    if (argc == 6) {
        if (!parse_u64_hex(argv[5], &seed)) {
            fputs("invalid seed\n", stderr);
            return 2;
        }
    }
    if (!point_from_hex(&target, argv[1])) {
        fputs("invalid secp256k1 public key\n", stderr);
        return 2;
    }
    if (!parse_u64_hex(argv[2], &low) ||
        !parse_u64_hex(argv[3], &high) ||
        low > high || high - low == UINT64_MAX ||
        high - low + 1 > (UINT64_C(1) << 60)) {
        fputs("invalid or unsupported range\n", stderr);
        return 2;
    }
    root = integer_sqrt_ceil(high - low + 1);
    printf(
        "mode=%s range=[0x%" PRIx64
        ",0x%" PRIx64 "] sqrt=%" PRIu64
        " seed=0x%" PRIx64 "\n",
        mode_name,
        low,
        high,
        root,
        seed
    );
    status = solve_range(
        &target,
        low,
        high,
        seed,
        mode,
        shadow_dp,
        &solution,
        &stats
    );
    if (status < 0) {
        fputs("allocation failed\n", stderr);
        return 2;
    }
    printf(
        "steps=%" PRIu64 " K=%.6f dp=%" PRIu64
        " shadow=%" PRIu64 " collisions=%" PRIu64
        " alt-eval=%" PRIu64 " alt-selected=%" PRIu64
        " second-jump-eval=%" PRIu64
        " second-jump-selected=%" PRIu64
        " loop-escapes=%" PRIu64 " muls=%" PRIu64 "\n",
        stats.steps,
        (double)stats.steps / (double)root,
        stats.distinguished,
        stats.shadows,
        stats.collisions,
        stats.alternate_evaluations,
        stats.alternate_selections,
        stats.second_jump_evaluations,
        stats.second_jump_selections,
        stats.loop_escapes, rck_mul_count
    );
    if (!status) {
        puts("not found");
        return 1;
    }
    printf("private=%064" PRIx64 "\n", solution);
    return 0;
}
