#include "gfx906_field.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kCases = 4096;

#define HIP_CHECK(call) do {                                                \
    const hipError_t error_ = (call);                                       \
    if (error_ != hipSuccess) {                                             \
        std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                     hipGetErrorString(error_));                            \
        std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
} while (0)

__global__ void field_kernel(
    rck_fe* sums,
    rck_fe* differences,
    rck_fe* products,
    const rck_fe* left,
    const rck_fe* right,
    std::size_t count
) {
    const std::size_t index =
        (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    umbra_fe_add(&sums[index], &left[index], &right[index]);
    umbra_fe_sub(&differences[index], &left[index], &right[index]);
    umbra_fe_mul(&products[index], &left[index], &right[index]);
}

bool equal(const rck_fe& left, const rck_fe& right) {
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

void fill_inputs(std::vector<rck_fe>& values, std::mt19937_64& rng) {
    for (rck_fe& value : values) {
        for (unsigned limb = 0; limb < 8; ++limb) {
            value.limb[limb] = static_cast<std::uint32_t>(rng());
        }
        /* Values below 2^255 are canonical and still exercise every limb. */
        value.limb[7] &= 0x7fffffffU;
    }
}

}  // namespace

int main() {
    static_assert(sizeof(rck_fe) == 32, "field ABI must remain 32 bytes");
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    if (std::strncmp(properties.gcnArchName, "gfx906", 6) != 0) {
        std::fprintf(
            stderr, "expected gfx906/MI50, found %s\n", properties.gcnArchName
        );
        return EXIT_FAILURE;
    }

    std::mt19937_64 rng(0x90650U);
    std::vector<rck_fe> left(kCases);
    std::vector<rck_fe> right(kCases);
    std::vector<rck_fe> sums(kCases);
    std::vector<rck_fe> differences(kCases);
    std::vector<rck_fe> products(kCases);
    fill_inputs(left, rng);
    fill_inputs(right, rng);

    rck_fe* device_left = nullptr;
    rck_fe* device_right = nullptr;
    rck_fe* device_sums = nullptr;
    rck_fe* device_differences = nullptr;
    rck_fe* device_products = nullptr;
    const std::size_t bytes = kCases * sizeof(rck_fe);
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_left), bytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_right), bytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_sums), bytes));
    HIP_CHECK(hipMalloc(
        reinterpret_cast<void**>(&device_differences), bytes
    ));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&device_products), bytes));
    HIP_CHECK(hipMemcpy(device_left, left.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_right, right.data(), bytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(
        field_kernel,
        dim3((kCases + 255) / 256),
        dim3(256),
        0,
        0,
        device_sums,
        device_differences,
        device_products,
        device_left,
        device_right,
        kCases
    );
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(sums.data(), device_sums, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        differences.data(), device_differences, bytes, hipMemcpyDeviceToHost
    ));
    HIP_CHECK(hipMemcpy(
        products.data(), device_products, bytes, hipMemcpyDeviceToHost
    ));

    for (std::size_t index = 0; index < kCases; ++index) {
        rck_fe expected;
        rck_fe_add(&expected, &left[index], &right[index]);
        if (!equal(expected, sums[index])) {
            std::fprintf(stderr, "gfx906 add mismatch at %zu\n", index);
            return EXIT_FAILURE;
        }
        rck_fe_sub(&expected, &left[index], &right[index]);
        if (!equal(expected, differences[index])) {
            std::fprintf(stderr, "gfx906 sub mismatch at %zu\n", index);
            return EXIT_FAILURE;
        }
        rck_fe_mul(&expected, &left[index], &right[index]);
        if (!equal(expected, products[index])) {
            std::fprintf(stderr, "gfx906 mul mismatch at %zu\n", index);
            return EXIT_FAILURE;
        }
    }

    HIP_CHECK(hipFree(device_products));
    HIP_CHECK(hipFree(device_differences));
    HIP_CHECK(hipFree(device_sums));
    HIP_CHECK(hipFree(device_right));
    HIP_CHECK(hipFree(device_left));
    std::printf("gfx906 HIP field differential tests: PASS (%zu cases)\n", kCases);
    return EXIT_SUCCESS;
}
