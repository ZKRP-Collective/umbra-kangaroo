CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Werror -pedantic

BUILD_DIR := build
TEST_BIN := $(BUILD_DIR)/test_portable
SOLVER_BIN := $(BUILD_DIR)/range_solver
GS_BIN := $(BUILD_DIR)/gs_solver
IFMA_BENCH_BIN := $(BUILD_DIR)/ifma52_bench

.PHONY: all asm-bmi2 compact-dp openmp openmp-product-tree \
	openmp-product-tree-compact openmp-overbooked test test-fallback32 \
	openmp-overbooked-gate2 openmp-overbooked-gate3 openmp-fused \
	openmp-fused-compact openmp-buffered-dp ifma-bench sanitize clean

all: $(TEST_BIN) $(SOLVER_BIN) $(GS_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_BIN): rck_portable.c rck_portable.h test_portable.c x86_64/limb_ops.S | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DRCK_USE_X86_64_ASM \
		rck_portable.c test_portable.c x86_64/limb_ops.S \
		-o $(TEST_BIN)

$(SOLVER_BIN): rck_portable.c rck_portable.h range_solver.c x86_64/limb_ops.S | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DRCK_USE_X86_64_ASM \
		rck_portable.c range_solver.c x86_64/limb_ops.S \
		-o $(SOLVER_BIN)

$(GS_BIN): rck_portable.c rck_portable.h gs_solver.c x86_64/limb_ops.S | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DRCK_USE_X86_64_ASM \
		rck_portable.c gs_solver.c x86_64/limb_ops.S \
		-o $(GS_BIN)

test: $(TEST_BIN)
	$(TEST_BIN)

$(IFMA_BENCH_BIN): ifma52_bench.c rck_portable.c rck_portable.h x86_64/limb_ops.S | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DRCK_USE_X86_64_ASM \
		ifma52_bench.c rck_portable.c x86_64/limb_ops.S \
		-o $(IFMA_BENCH_BIN)

ifma-bench: $(IFMA_BENCH_BIN)
	$(IFMA_BENCH_BIN)

asm-bmi2: CFLAGS += -DRCK_USE_X86_64_BMI2
asm-bmi2: clean all test

compact-dp: CFLAGS += -DRCK_COMPACT_DP
compact-dp: clean all test

openmp: CFLAGS += -fopenmp -DRCK_USE_OPENMP
openmp: clean all test

openmp-product-tree: CFLAGS += -fopenmp -DRCK_USE_OPENMP -DRCK_PARALLEL_BATCH_INV
openmp-product-tree: clean all test

openmp-product-tree-compact: CFLAGS += -fopenmp -DRCK_USE_OPENMP \
	-DRCK_PARALLEL_BATCH_INV -DRCK_COMPACT_DP
openmp-product-tree-compact: clean all test

openmp-fused: CFLAGS += -fopenmp -DRCK_USE_OPENMP \
	-DRCK_PARALLEL_BATCH_INV -DRCK_FUSED_AFFINE_FORMULAS
openmp-fused: clean all test

openmp-fused-compact: CFLAGS += -fopenmp -DRCK_USE_OPENMP \
	-DRCK_PARALLEL_BATCH_INV -DRCK_COMPACT_DP -DRCK_FUSED_AFFINE_FORMULAS
openmp-fused-compact: clean all test

openmp-buffered-dp: CFLAGS += -fopenmp -DRCK_USE_OPENMP \
	-DRCK_PARALLEL_BATCH_INV -DRCK_COMPACT_DP -DRCK_BUFFERED_DP
openmp-buffered-dp: clean all test

openmp-overbooked: CFLAGS += -fopenmp -DRCK_USE_OPENMP \
	-DRCK_PARALLEL_BATCH_INV -DRCK_COMPACT_DP -DRCK_OVERBOOKED_JUMPS
openmp-overbooked: clean all test

openmp-overbooked-gate2: CFLAGS += -fopenmp -DRCK_USE_OPENMP \
	-DRCK_PARALLEL_BATCH_INV -DRCK_COMPACT_DP -DRCK_OVERBOOKED_JUMPS \
	-DRCK_SECOND_JUMP_SCORE_MIN=2U
openmp-overbooked-gate2: clean all test

openmp-overbooked-gate3: CFLAGS += -fopenmp -DRCK_USE_OPENMP \
	-DRCK_PARALLEL_BATCH_INV -DRCK_COMPACT_DP -DRCK_OVERBOOKED_JUMPS \
	-DRCK_SECOND_JUMP_SCORE_MIN=3U
openmp-overbooked-gate3: clean all test

test-fallback32: clean | $(BUILD_DIR)
	$(CC) $(CFLAGS) -U__SIZEOF_INT128__ -DRCK_USE_X86_64_ASM \
		rck_portable.c test_portable.c x86_64/limb_ops.S \
		-o $(TEST_BIN)
	$(TEST_BIN)

sanitize: CFLAGS := -O1 -g -std=c11 -Wall -Wextra -Werror -pedantic \
	-fsanitize=address,undefined -fno-omit-frame-pointer
sanitize: clean $(TEST_BIN)
	ASAN_OPTIONS=detect_leaks=0 $(TEST_BIN)

clean:
	rm -f $(TEST_BIN) $(SOLVER_BIN) $(GS_BIN)
