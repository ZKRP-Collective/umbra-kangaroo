/*
 * Reference interval solver: Gaudry-Schost with the negation map, using the
 * Galbraith-Ruprai set shapes. It is the external baseline for the Umbra walk.
 *
 * The interval is centered so the shifted secret k' = k - low - N/2 lies in
 * [-N/2, N/2). Tame walks start at a*G with a uniform in [-N/2, N/2]. Wild
 * walks start at Q' + b*G with b uniform in [-N/4, N/4]. Each step is
 * P <- P + s(P)*J[x(P)], where s(P) = -1 when y(P) is odd. The step is a
 * function of the class {P, -P}, so walks merge on classes. A walk ends at
 * a distinguished point and restarts from a fresh random start. A walk also
 * restarts on a fruitless 2-cycle or when it runs past its length cap.
 *
 * usage: gs_solver PUBLIC_KEY_HEX LOW_HEX HIGH_HEX DP_BITS [SEED_HEX]
 */
#include "rck_portable.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GS_WALKERS 128U
#define GS_JUMPS 64U

typedef struct {
    rck_point point;
    int64_t coeff;       /* tame: P = coeff*G; wild: P = Q' + coeff*G */
    uint32_t wild;
    uint64_t length;
    uint64_t prev_fp;    /* x fingerprint two steps back */
    uint64_t last_fp;
    uint32_t escape;
} gs_walker;

typedef struct {
    rck_fe x;
    int64_t coeff;
    uint32_t wild;
    uint32_t used;
} gs_record;

static uint64_t rng_state;

static uint64_t mix64(uint64_t v) {
    v += UINT64_C(0x9e3779b97f4a7c15);
    v = (v ^ (v >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)) * UINT64_C(0x94d049bb133111eb);
    return v ^ (v >> 31);
}

static uint64_t rng_next(void) {
    rng_state += UINT64_C(0x9e3779b97f4a7c15);
    return mix64(rng_state);
}

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int fe_from_hex(rck_fe* out, const char* text) {
    size_t i;
    memset(out, 0, sizeof(*out));
    if (strlen(text) != 64) return 0;
    for (i = 0; i < 64; ++i) {
        const int n = hex_nibble(text[63 - i]);
        if (n < 0) return 0;
        out->limb[i / 8] |= (uint32_t)n << (4 * (i % 8));
    }
    return 1;
}

static void fe_pow(rck_fe* out, const rck_fe* base, const uint32_t e[8]) {
    rck_fe acc;
    int bit;
    rck_fe_set_u32(&acc, 1);
    for (bit = 255; bit >= 0; --bit) {
        rck_fe_sqr(&acc, &acc);
        if ((e[bit / 32] >> (bit % 32)) & 1U) rck_fe_mul(&acc, &acc, base);
    }
    *out = acc;
}

static int point_from_hex(rck_point* out, const char* text) {
    /* compressed SEC1 only; sqrt exponent (p+1)/4 */
    static const uint32_t sqrt_exp[8] = {
        0xBFFFFF0CU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU,
        0xFFFFFFFFU, 0xFFFFFFFFU, 0xFFFFFFFFU, 0x3FFFFFFFU
    };
    rck_fe rhs, seven, y, check;
    if (strlen(text) != 66 || text[0] != '0' ||
        (text[1] != '2' && text[1] != '3')) {
        return 0;
    }
    if (!fe_from_hex(&out->x, text + 2)) return 0;
    rck_fe_sqr(&rhs, &out->x);
    rck_fe_mul(&rhs, &rhs, &out->x);
    rck_fe_set_u32(&seven, 7);
    rck_fe_add(&rhs, &rhs, &seven);
    fe_pow(&y, &rhs, sqrt_exp);
    rck_fe_sqr(&check, &y);
    if (!rck_fe_equal(&check, &rhs)) return 0;
    if ((y.limb[0] & 1U) != (uint32_t)(text[1] - '2')) rck_fe_neg(&y, &y);
    out->y = y;
    out->infinity = 0;
    return 1;
}

#define GS_TAB_BITS 16U
#define GS_TABS 4U
static rck_point* start_table; /* [t][i] = i * 2^(16t) * G */
static rck_point tame_base;     /* -half * G */
static rck_point wild_base;     /* Q' - (half/2) * G */

static int build_start_table(void) {
    const size_t n = (size_t)1 << GS_TAB_BITS;
    uint64_t* scalars = malloc(n * sizeof(*scalars));
    size_t t, i;
    start_table = malloc(GS_TABS * n * sizeof(*start_table));
    if (!scalars || !start_table) return 0;
    for (t = 0; t < GS_TABS; ++t) {
        for (i = 0; i < n; ++i) scalars[i] = (uint64_t)i << (GS_TAB_BITS * t);
        scalars[0] = 1; /* placeholder; index 0 means "skip" */
        if (!rck_generator_mul_batch(start_table + t * n, scalars, n)) return 0;
    }
    free(scalars);
    return 1;
}

/* P = base + u*G for 0 <= u < 2^64 using at most four affine additions. */
static void start_point(rck_point* out, const rck_point* base, uint64_t u) {
    const size_t n = (size_t)1 << GS_TAB_BITS;
    size_t t;
    *out = *base;
    for (t = 0; t < GS_TABS; ++t) {
        const size_t digit = (size_t)((u >> (GS_TAB_BITS * t)) & (n - 1));
        if (digit) rck_point_add(out, out, &start_table[t * n + digit]);
    }
}

static void random_start(gs_walker* w, uint64_t half) {
    do {
        if (w->wild) {
            const uint64_t u = rng_next() % (half + 1);
            w->coeff = (int64_t)u - (int64_t)(half / 2);
            start_point(&w->point, &wild_base, u);
        } else {
            const uint64_t u = rng_next() % (2 * half + 1);
            w->coeff = (int64_t)u - (int64_t)half;
            start_point(&w->point, &tame_base, u);
        }
    } while (w->point.infinity);
    w->length = 0;
    w->prev_fp = w->last_fp = 0;
    w->escape = 0;
}

static uint64_t fingerprint(const rck_fe* x) {
    return (uint64_t)x->limb[0] | ((uint64_t)x->limb[1] << 32);
}

static int verify(const rck_point* target, uint64_t low, uint64_t high,
                  uint64_t half, int64_t shifted, uint64_t* solution) {
    rck_point check;
    int64_t offset = shifted + (int64_t)half;
    uint64_t candidate;
    if (offset < 0 || (uint64_t)offset > high - low) return 0;
    candidate = low + (uint64_t)offset;
    rck_point_mul_u64(&check, &RCK_GENERATOR, candidate);
    if (check.infinity || !rck_fe_equal(&check.x, &target->x) ||
        !rck_fe_equal(&check.y, &target->y)) {
        return 0;
    }
    *solution = candidate;
    return 1;
}

int main(int argc, char** argv) {
    rck_point target, shifted, jumps[GS_JUMPS], neg_jumps[GS_JUMPS], tmp;
    int64_t jump_size[GS_JUMPS];
    gs_walker* walkers;
    gs_record* table;
    rck_fe den[GS_WALKERS], inv[GS_WALKERS], pre[GS_WALKERS];
    uint64_t low, high, range, half, root, cap, mask, steps = 0, restarts = 0;
    uint64_t dps = 0, fruitless = 0, overlong = 0, solution = 0, max_steps;
    unsigned dp_bits;
    size_t i, capacity = UINT64_C(1) << 22;
    int solved = 0;

    if (argc < 5 || argc > 6) {
        fprintf(stderr, "usage: %s PUBKEY LOW_HEX HIGH_HEX DP_BITS [SEED_HEX]\n", argv[0]);
        return 2;
    }
    if (!point_from_hex(&target, argv[1])) {
        fputs("invalid public key\n", stderr);
        return 2;
    }
    low = strtoull(argv[2], NULL, 16);
    high = strtoull(argv[3], NULL, 16);
    dp_bits = (unsigned)strtoul(argv[4], NULL, 10);
    rng_state = argc == 6 ? strtoull(argv[5], NULL, 16) : 1;
    if (high <= low || high - low > (UINT64_C(1) << 62) || dp_bits > 24) {
        fputs("range must be nonempty and at most 2^62; dp_bits <= 24\n", stderr);
        return 2;
    }
    range = high - low + 1;
    half = range / 2;
    for (root = 1; root * root < range; ++root) {}
    mask = dp_bits ? (UINT64_C(1) << dp_bits) - 1 : 0;
    cap = 16 * (mask + 1);
    max_steps = 64 * root;

    /* Q' = Q - (low + N/2) G */
    rck_point_mul_u64(&tmp, &RCK_GENERATOR, low + half);
    rck_point_neg(&tmp, &tmp);
    rck_point_add(&shifted, &target, &tmp);
    if (shifted.infinity) {
        printf("private=%064" PRIx64 "\n", low + half);
        return 0;
    }

    if (!build_start_table()) return 2;
    rck_point_mul_u64(&tame_base, &RCK_GENERATOR, half);
    rck_point_neg(&tame_base, &tame_base);
    rck_point_mul_u64(&tmp, &RCK_GENERATOR, half / 2);
    rck_point_neg(&tmp, &tmp);
    rck_point_add(&wild_base, &shifted, &tmp);

    for (i = 0; i < GS_JUMPS; ++i) {
        jump_size[i] = (int64_t)(1 + rng_next() % (2 * root));
        rck_point_mul_u64(&jumps[i], &RCK_GENERATOR, (uint64_t)jump_size[i]);
        rck_point_neg(&neg_jumps[i], &jumps[i]);
    }

    walkers = calloc(GS_WALKERS, sizeof(*walkers));
    table = calloc(capacity, sizeof(*table));
    if (!walkers || !table) return 2;

    for (i = 0; i < GS_WALKERS; ++i) {
        walkers[i].wild = (uint32_t)(i & 1U);
        random_start(&walkers[i], half);
    }

    while (!solved && steps < max_steps) {
        size_t idx[GS_WALKERS];
        for (i = 0; i < GS_WALKERS; ++i) {
            idx[i] = (walkers[i].point.x.limb[0] + walkers[i].escape) & (GS_JUMPS - 1);
            rck_fe_sub(&den[i], &walkers[i].point.x, &jumps[idx[i]].x);
            if (rck_fe_is_zero(&den[i])) rck_fe_set_u32(&den[i], 1);
        }
        if (!rck_fe_batch_inv(inv, den, pre, GS_WALKERS)) return 2;
        for (i = 0; i < GS_WALKERS && !solved; ++i) {
            gs_walker* w = &walkers[i];
            const int neg = (int)(w->point.y.limb[0] & 1U);
            const rck_point* j = neg ? &neg_jumps[idx[i]] : &jumps[idx[i]];
            uint64_t fp;
            int restart = 0;
            if (rck_fe_equal(&w->point.x, &j->x)) {
                rck_point_add(&w->point, &w->point, j);
            } else if (!rck_point_add_preinverted(&w->point, &w->point, j, &inv[i])) {
                return 2;
            }
            w->coeff += neg ? -jump_size[idx[i]] : jump_size[idx[i]];
            ++steps;
            ++w->length;
            if (w->point.infinity) {
                if (w->wild && verify(&target, low, high, half, -w->coeff, &solution)) solved = 1;
                restart = 1;
            } else {
                fp = fingerprint(&w->point.x);
                w->escape = 0;
                if (fp == w->prev_fp) {
                    /* fruitless 2-cycle: next step uses the neighbouring jump */
                    ++fruitless;
                    w->escape = 1;
                } else if ((w->point.x.limb[5] & (uint32_t)mask) == 0) {
                    size_t slot = (size_t)mix64(fp ^ w->point.x.limb[6]) & (capacity - 1);
                    ++dps;
                    for (;;) {
                        gs_record* r = &table[slot];
                        if (!r->used) {
                            r->used = 1;
                            r->x = w->point.x;
                            r->coeff = w->coeff;
                            r->wild = w->wild;
                            break;
                        }
                        if (rck_fe_equal(&r->x, &w->point.x)) {
                            int64_t cand[2];
                            int n = 0, c;
                            if (r->wild != w->wild) {
                                const int64_t t = r->wild ? w->coeff : r->coeff;
                                const int64_t d = r->wild ? r->coeff : w->coeff;
                                cand[n++] = t - d;
                                cand[n++] = -t - d;
                            } else if (w->wild && ((r->coeff + w->coeff) & 1) == 0) {
                                cand[n++] = -(r->coeff + w->coeff) / 2;
                            }
                            for (c = 0; c < n && !solved; ++c) {
                                if (verify(&target, low, high, half, cand[c], &solution)) solved = 1;
                            }
                            break;
                        }
                        slot = (slot + 1) & (capacity - 1);
                    }
                    restart = 1;
                } else if (w->length >= cap) {
                    ++overlong;
                    restart = 1;
                }
                w->prev_fp = w->last_fp;
                w->last_fp = fp;
            }
            if (restart && !solved) {
                ++restarts;
                random_start(w, half);
            }
        }
    }

    printf("steps=%" PRIu64 " K=%.6f dp=%" PRIu64 " restarts=%" PRIu64
           " fruitless=%" PRIu64 " overlong=%" PRIu64 "\n",
           steps, (double)steps / (double)root, dps, restarts, fruitless, overlong);
    if (!solved) {
        puts("not found");
        return 1;
    }
    printf("private=%064" PRIx64 "\n", solution);
    free(walkers);
    free(table);
    return 0;
}
