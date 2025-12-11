# SM100 Structured: Testing, Benchmarking, and Switching Guide

This guide explains how to test, benchmark, and switch between the original `sm100`
and refactored `sm100_structured` implementations.

## Overview

The `sm100_structured` module is a refactored version of the SM100 matmul kernel
with improved code organization. Both implementations are **functionally
equivalent** and produce identical results—the refactoring is purely
organizational.

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        sm100/matmul.mojo                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  blackwell_matmul_tma_umma_warp_specialized()                     │  │
│  │                                                                   │  │
│  │  @parameter                                                       │  │
│  │  if _USE_STRUCTURED:  ──────────────────┐                         │  │
│  │      # Forward to sm100_structured      │                         │  │
│  │  else:                                  │                         │  │
│  │      # Use original implementation      │                         │  │
│  └─────────────────────────────────────────│─────────────────────────┘  │
└────────────────────────────────────────────│────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     sm100_structured/                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ matmul.mojo     │  │ matmul_kernels  │  │ tile_scheduler.mojo     │  │
│  │ (entry point)   │──│ .mojo (GPU)     │──│ WorkIterator            │  │
│  └─────────────────┘  └─────────────────┘  │ SchedulerWorkIterator   │  │
│                                            └─────────────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │ ring_buffer.mojo│  │ tile_loader.mojo│  │ matmul_output.mojo      │  │
│  │ RingBuffer      │  │ TileLoaderTMA   │  │ copy_accum_to_gmem      │  │
│  │ OutputRingBuffer│  └─────────────────┘  │ multi_stage_store_C     │  │
│  └─────────────────┘                       └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Shared Dependencies

`sm100_structured` imports these from `sm100` (not duplicated):

- `sm100/config.mojo` → `MatmulConfig` (shared type for API compatibility)

## Directory Structure

```text
max/kernels/src/linalg/matmul/gpu/
├── sm100/                      # Original implementation
│   ├── config.mojo             # MatmulConfig (SHARED by both)
│   ├── matmul.mojo             # Entry point with _USE_STRUCTURED flag
│   ├── dispatch.mojo           # Dispatch logic
│   └── ...
│
└── sm100_structured/           # Refactored implementation
    ├── __init__.mojo           # Exports public API
    ├── matmul.mojo             # CPU-side entry points
    ├── matmul_kernels.mojo     # GPU kernel structs (BlackwellMatmulSM100Kernel)
    ├── matmul_output.mojo      # Output pipeline (copy_accum_to_gmem, etc.)
    ├── ring_buffer.mojo        # RingBuffer, OutputRingBuffer
    ├── tile_loader.mojo        # TileLoaderTMA
    ├── tile_scheduler.mojo     # TileScheduler, WorkIterator, SchedulerWorkIterator
    ├── tile_scheduler_splitk.mojo  # Split-K variants
    ├── tile_writer.mojo        # TileWriter for epilogue
    ├── pipeline.mojo           # ProducerConsumerPipeline
    └── DOCS/                    # Documentation
        ├── testing_and_switching.md      # This file
        ├── sm100_matmul_refactoring_review.md
        ├── sm100_matmul_refactoring_plan.md
        └── sm100_matmul_kernel_refactoring_comparison.md
```

Both implementations expose the same API and are functionally equivalent.

---

## 1. Running Tests

### Smoke Tests (Quick Validation)

```bash
# Run the SM100 smoke test suite
./bazelw run //max/kernels/test/gpu/linalg:test_matmul_sm100_smoke.mojo.test
```

This runs 10 quick tests covering:

- Basic shapes and dtypes (bf16, fp16)
- Split-K configurations
- 2SM (CTA group) configurations
- Cluster shapes (1x1, 2x2, 4x4, 8x2)
- SwapAB configurations
- Misaligned dimensions

### Full Test Suite

```bash
# Run all SM100 matmul tests
./bazelw test //max/kernels/test/gpu/linalg:test_matmul_sm100_splitk_2sm_bf16.mojo.test

# Run tile scheduler unit tests
./bazelw run //max/kernels/test/gpu/linalg:test_matmul_tile_scheduler_sm100.mojo
```

### Iterative Tutorial Tests

```bash
# Run the CLC+TMEM tutorial test (uses TileScheduler)
./bazelw run //max/kernels/test/gpu/linalg/matmul_blackwell_iterative:8_clc_tmem_ping.mojo.test
```

---

## 2. Benchmarking

### Quick Benchmark

```bash
# Run matmul benchmarks on SM100
./bazelw run //max/kernels/benchmarks/gpu/linalg:bench_matmul -- \
    --M=4096 --N=4096 --K=4096 \
    --dtype=bfloat16 \
    --num_iters=100
```

### Comparative Benchmark (Original vs Structured)

To compare both implementations, you need to temporarily modify the import in the
benchmark or test file. See "Switching Implementations" below.

---

## 3. Switching Implementations

### Option A: Environment Variable (Recommended)

**Use the `MODULAR_USE_STRUCTURED_SM100` environment variable:**

```bash
# Use original sm100 (default):
./bazelw run //max/kernels/test/gpu/linalg:test_matmul_sm100_smoke.mojo.test

# Use sm100_structured:
MODULAR_USE_STRUCTURED_SM100=1 ./bazelw run //max/kernels/test/gpu/linalg:test_matmul_sm100_smoke.mojo.test
```

This works for **all tests and code paths** without modifying any files.

The feature flag is implemented in `sm100/matmul.mojo`:

```mojo
comptime _USE_STRUCTURED = env_get_bool["MODULAR_USE_STRUCTURED_SM100", False]()

fn blackwell_matmul_tma_umma_warp_specialized[...]:
    @parameter
    if _USE_STRUCTURED:
        # Forward to sm100_structured implementation
        structured_impl[...](c_device, a_device, b_device, ctx)
        return
    # ... original implementation
```

### Option B: Direct Import (For Development)

To always use `sm100_structured` in a specific test file:

```mojo
# Change from:
from linalg.matmul.gpu.sm100.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)

# To:
from linalg.matmul.gpu.sm100_structured import (
    blackwell_matmul_tma_umma_warp_specialized,
)
```

### Option C: Make sm100_structured the Permanent Default

To permanently switch all code to use `sm100_structured`:

**File to modify:** `max/kernels/src/linalg/matmul/gpu/sm100/matmul.mojo`

Change line ~108:

```mojo
# From:
comptime _USE_STRUCTURED = env_get_bool["MODULAR_USE_STRUCTURED_SM100", False]()

# To:
comptime _USE_STRUCTURED = True
```

---

## 4. Verifying Correctness

After switching implementations, verify correctness:

```bash
# 1. Run smoke tests
./bazelw run //max/kernels/test/gpu/linalg:test_matmul_sm100_smoke.mojo.test

# 2. Run numerical accuracy tests
./bazelw test //max/kernels/test/gpu/linalg:test_matmul_sm100_splitk_2sm_bf16.mojo.test

# 3. Compare outputs (manual)
# Modify a test to run both implementations and compare results:
#   result_original = blackwell_matmul_tma_umma_warp_specialized(...)  # from sm100
#   result_structured = blackwell_matmul_tma_umma_warp_specialized(...) # from sm100_structured
#   assert allclose(result_original, result_structured)
```

---

## 5. Key Differences (Functional Equivalence)

| Aspect | sm100 | sm100_structured |
|--------|-------|------------------|
| API | `blackwell_matmul_tma_umma_warp_specialized` | Same |
| Functionality | Full SM100 matmul | Same |
| Performance | Baseline | Same (no perf changes) |
| Code organization | Monolithic kernel | Modular with iterators |
| Pipeline management | Explicit state variables | Encapsulated in iterators |

The refactoring is purely organizational—no algorithmic or performance changes.

---

## 6. Implementation Details

### Environment Variable Switch

The switching mechanism is implemented in `sm100/matmul.mojo`
(lines ~107-108 and ~3054-3070):

```mojo
# Feature flag definition (line ~108)
comptime _USE_STRUCTURED = env_get_bool["MODULAR_USE_STRUCTURED_SM100", False]()

# In blackwell_matmul_tma_umma_warp_specialized() (line ~3054)
fn blackwell_matmul_tma_umma_warp_specialized[...](
    c_device, a_device, b_device, ctx
) raises:
    @parameter
    if _USE_STRUCTURED:
        from ..sm100_structured import (
            blackwell_matmul_tma_umma_warp_specialized as structured_impl,
        )
        structured_impl[
            c_type, c_layout, a_type, a_layout, b_type, b_layout, transpose_b,
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            register_based_epilogue=register_based_epilogue,
            pdl_level=pdl_level,
            max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
        ](c_device, a_device, b_device, ctx)
        return
    
    # ... original implementation continues
```

### Why MatmulConfig is Shared

Both implementations must use the same `MatmulConfig` type for API compatibility.
If they used separate types, callers would get type mismatch errors when switching.

`sm100_structured/matmul.mojo` and `matmul_kernels.mojo` import from the shared location:

```mojo
from ..sm100.config import MatmulConfig  # Shared type
```

---

## 7. Troubleshooting

### Build Errors After Switching

If you see import errors after switching:

```bash
# Clean build cache
./bazelw clean --expunge

# Rebuild
./bazelw build //max/kernels/src/linalg:linalg
```

### Test Failures

If tests fail after switching to `sm100_structured`:

1. Verify you're importing from the correct module
2. Check that all dependent tests have been updated
3. Run with verbose output: `./bazelw run ... -- --verbose`

### Performance Regression

The structured implementation should have identical performance. If you observe
regression:

1. Verify you're comparing apples-to-apples (same shapes, dtypes, configs)
2. Run multiple iterations to account for variance
3. Check GPU thermals (throttling can affect results)

---

## Quick Reference

```bash
# Test with original sm100 (default):
./bazelw run //max/kernels/test/gpu/linalg:test_matmul_sm100_smoke.mojo.test

# Test with sm100_structured:
MODULAR_USE_STRUCTURED_SM100=1 ./bazelw run //max/kernels/test/gpu/linalg:test_matmul_sm100_smoke.mojo.test

# Build
./bazelw build //max/kernels/src/linalg:linalg

# Format
./bazelw run //:format
```

### Switching Summary

| Method | Command/Change |
|--------|----------------|
| Env var (temporary) | `MODULAR_USE_STRUCTURED_SM100=1 ./bazelw run ...` |
| Direct import | Change import to `from ...sm100_structured import ...` |
| Permanent default | Set `comptime _USE_STRUCTURED = True` in `sm100/matmul.mojo` |
