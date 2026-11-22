#include "gfx906_field.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

namespace {

bool equal(const rck_fe& left, const rck_fe& right) {
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

rck_fe random_canonical(std::mt19937_64& rng) {
    rck_fe value;
    for (unsigned limb = 0; limb < 8; ++limb) {
        value.limb[limb] = static_cast<std::uint32_t>(rng());
    }
    value.limb[7] &= 0x7fffffffU;
    return value;
}

}  // namespace

int main() {
    std::mt19937_64 rng(0x90650U);
    for (unsigned test = 0; test < 10000; ++test) {
        const rck_fe left = random_canonical(rng);
        const rck_fe right = random_canonical(rng);
        rck_fe expected;
        rck_fe actual;

        rck_fe_add(&expected, &left, &right);
        umbra_fe_add(&actual, &left, &right);
        if (!equal(expected, actual)) {
            std::fprintf(stderr, "gfx906 host add mismatch at %u\n", test);
            return 1;
        }
        rck_fe_sub(&expected, &left, &right);
        umbra_fe_sub(&actual, &left, &right);
        if (!equal(expected, actual)) {
            std::fprintf(stderr, "gfx906 host sub mismatch at %u\n", test);
            return 1;
        }
        rck_fe_mul(&expected, &left, &right);
        umbra_fe_mul(&actual, &left, &right);
        if (!equal(expected, actual)) {
            std::fprintf(stderr, "gfx906 host mul mismatch at %u\n", test);
            return 1;
        }
    }
    std::printf("gfx906 field leaf host-oracle tests: PASS (10000 cases)\n");
    return 0;
}
