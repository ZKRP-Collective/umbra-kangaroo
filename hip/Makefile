HIPCC ?= hipcc
CC ?= cc
CXX ?= c++
OFFLOAD_ARCH ?= gfx906
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Werror -pedantic
HIPFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Werror
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Werror -Wno-unknown-pragmas

BUILD_DIR := build
PORTABLE_OBJ := $(BUILD_DIR)/rck_portable.o
FIELD_SMOKE := $(BUILD_DIR)/gfx906_field_smoke
CPU_FIELD_TEST := $(BUILD_DIR)/gfx906_field_cpu_test

.PHONY: all test cpu-test clean toolchain

all: $(FIELD_SMOKE)

toolchain:
	@command -v $(HIPCC) >/dev/null || { \
		echo "hipcc not found; install a gfx906-capable ROCm toolchain" >&2; \
		exit 1; \
	}

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(PORTABLE_OBJ): ../portable/rck_portable.c ../portable/rck_portable.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I../portable -c ../portable/rck_portable.c -o $@

$(FIELD_SMOKE): gfx906_field_smoke.hip gfx906_field.h $(PORTABLE_OBJ) | toolchain
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(OFFLOAD_ARCH) -I../portable \
		gfx906_field_smoke.hip $(PORTABLE_OBJ) -o $@

$(CPU_FIELD_TEST): gfx906_field_cpu_test.cpp gfx906_field.h \
		cpu_stub/hip/hip_runtime.h $(PORTABLE_OBJ)
	$(CXX) $(CXXFLAGS) -Icpu_stub -I../portable \
		gfx906_field_cpu_test.cpp $(PORTABLE_OBJ) -o $@

test: $(FIELD_SMOKE)
	$(FIELD_SMOKE)

cpu-test: $(CPU_FIELD_TEST)
	$(CPU_FIELD_TEST)

clean:
	rm -rf $(BUILD_DIR)
