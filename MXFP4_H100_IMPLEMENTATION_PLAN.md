# MXFP4 on H100 (SM90): Step-by-Step Implementation Plan

## Goal

Enable MXFP4-quantized weight inference on H100 GPUs via a **dequant-to-FP8 + FP8 GEMM** strategy. MXFP4 weights (4-bit `float4_e2m1fn`, packed as `uint8`) are dequantized to `float8_e4m3fn` on-the-fly using a dedicated GPU kernel, then fed into the existing SM90 FP8 warp-specialized GEMM.

---

## Background & Key Concepts

### MXFP4 Format
- **Data**: 4-bit `float4_e2m1fn` values packed 2-per-byte in `uint8` tensors (same as NVFP4 storage)
- **Scales**: `float8_e8m0fnu` (8-bit exponent-only), one scale per block of **32 elements** (`MXFP4_SF_VECTOR_SIZE = 32`)
- **Scale layout**: 5D interleaved format using `SF_ATOM_M = (32, 4)`, `SF_ATOM_K = 4`, `SF_MN_GROUP_SIZE = 128`
- Defined in `max/kernels/src/linalg/fp4_utils.mojo:30-34`

### NVFP4 Format (existing, SM100-only)
- Same 4-bit data packing
- **Scales**: `float8_e4m3fn`, one scale per **16 elements** (`NVFP4_SF_VECTOR_SIZE = 16`)
- Full matmul hardware support only on B200 (SM100) via UMMA

### Strategy: Dequant MXFP4 → FP8, then FP8 GEMM
H100 has no native FP4 tensor core support, but has excellent FP8 GEMM. The plan:
1. Write a GPU kernel that unpacks FP4 values, multiplies by their E8M0 scale, and quantizes to FP8
2. Feed the resulting FP8 tensor into the existing SM90 FP8 GEMM path

---

## Phase 1: MXFP4-to-FP8 Dequantization Kernel

### Step 1.1: Create the dequant kernel file

**Create** `max/kernels/src/linalg/mxfp4_dequant.mojo`

This kernel converts packed MXFP4 (`uint8`) + E8M0 scales → `float8_e4m3fn`.

**Reference code for unpacking FP4:**
- `fp4_utils.mojo:57-91` — `cast_uint_to_fp4e2m1[]` extracts 4-bit nibbles from uint8 using lookup table `E2M1_TO_FLOAT32` (line 37-54)
- `fp4_utils.mojo:30` — `MXFP4_SF_VECTOR_SIZE = 32` (block size for MXFP4 scales)
- `fp4_utils.mojo:34` — `MXFP4_SF_DTYPE = DType.float8_e8m0fnu`

**Reference code for scale factor access:**
- `fp4_utils.mojo:215-237` — `get_scale_factor[]` reads from the 5D interleaved scale layout
- `fp4_utils.mojo:24-27` — Scale layout constants: `SF_ATOM_M = (32, 4)`, `SF_ATOM_K = 4`, `SF_MN_GROUP_SIZE = 128`

**Reference code for kernel launch pattern:**
- `fp4_quantization.mojo:95-185` — `quantize_dynamic_scaled_fp4fp8[]` shows the pattern: compute grid/block dims, then `ctx.enqueue_function[kernel, kernel](...)`
- `fp4_quantization.mojo:145-161` — Grid/block dim calculation using `ELEMENTS_PER_THREAD = 8`, SM count, etc.

**Algorithm per thread:**
```
1. Load 8 packed FP4 values (4 uint8 bytes → 8 float4_e2m1 values)
2. Convert each FP4 to float32 via E2M1_TO_FLOAT32 lookup table
3. Load the E8M0 scale for this block of 32 (using get_scale_factor)
4. Convert E8M0 to float32: scale = 2^(e8m0_value - 127)
5. Multiply: fp32_val = fp4_as_fp32 * scale
6. Quantize to FP8: fp8_val = fp32_val.cast[float8_e4m3fn]()
7. Store output
```

**Key design decisions:**
- The input `uint8` tensor has shape `[N, K/2]` (2 FP4 values per byte)
- The output `float8_e4m3fn` tensor has shape `[N, K]`
- The scale tensor uses the 5D interleaved layout (same as NVFP4/MXFP8)
- Use `ELEMENTS_PER_THREAD = 8` to match the existing quantization kernel pattern
- No need for shared memory — this is a pure elementwise kernel

**Function signature:**
```mojo
fn dequant_mxfp4_to_fp8[
    out_dtype: DType,           # float8_e4m3fn
    scales_dtype: DType,        # float8_e8m0fnu
    in_dtype: DType,            # uint8 (packed fp4)
    output_layout: Layout,
    scales_layout: Layout,
    input_layout: Layout,
    //,
    *,
    SF_VECTOR_SIZE: Int = 32,   # MXFP4_SF_VECTOR_SIZE
](
    ctx: DeviceContext,
    output: LayoutTensor[out_dtype, output_layout, MutAnyOrigin],
    input: LayoutTensor[in_dtype, input_layout, ImmutAnyOrigin],
    scales: LayoutTensor[scales_dtype, scales_layout, ImmutAnyOrigin],
    num_rows: Int,
    num_cols: Int,  # K (unpacked dimension)
) raises
```

### Step 1.2: Write a unit test for the dequant kernel

**Create** `max/kernels/test/gpu/linalg/test_mxfp4_dequant.mojo`

**Reference test patterns:**
- `test/gpu/numerics/test_e2m1_conversion.mojo:25-76` — Tests for FP4 conversion correctness
- `test/gpu/linalg/test_matmul_sm100_block_scaled_nvfp4.mojo:58-120` — Test setup pattern: allocate host/device buffers, init random data, copy to device, run kernel, copy back, compare

**Test cases:**
1. **Known values**: Pack known FP4 values (0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) with scale=1.0, verify exact FP8 output
2. **Scale application**: Same FP4 values with scale=2.0 (E8M0 = 128), verify doubled output
3. **Block boundary**: Values spanning multiple 32-element blocks with different scales
4. **Typical shapes**: M=128, K=256 (small) and M=4096, K=8192 (typical LLM)
5. **Padding**: K not aligned to 32 (handled by zeroing out-of-bounds)

**Add to BUILD.bazel:**
- `max/kernels/test/gpu/linalg/BUILD.bazel` — add target `test_mxfp4_dequant`

**Run:**
```bash
./bazelw test --config=remote-h100 //max/kernels/test/gpu/linalg:test_mxfp4_dequant
```

### Step 1.3: Validate dequant kernel correctness against CPU reference

Write a CPU reference implementation in the test file that:
1. Unpacks FP4 values using `cast_uint_to_fp4e2m1` from `fp4_utils.mojo:57`
2. Applies scales manually in float32
3. Casts to FP8
4. Compares against GPU kernel output using `assert_almost_equal` from `internal_utils`

---

## Phase 2: Integrate Dequant + FP8 GEMM Pipeline

### Step 2.1: Create the combined MXFP4 matmul function

**Create** `max/kernels/src/linalg/mxfp4_matmul_sm90.mojo`

This function orchestrates: dequant MXFP4→FP8, then call SM90 FP8 GEMM.

**Reference for SM90 FP8 GEMM call:**
- `sm90/dispatch.mojo:495-586` — `matmul_dispatch_sm90_fp8[]` is the entry point
- `sm90/dispatch.mojo:532-558` — Shows how to construct `MatmulConfig` and call `warp_specialize_gemm_with_multicasting[]`
- `sm90/matmul.mojo:1` — `warp_specialize_gemm_with_multicasting` function definition

**Reference for MatmulConfig:**
- `utils_gpu.mojo` — `MatmulConfig` struct with `block_tile_shape`, `mma_shape`, `cluster_shape`, `num_pipeline_stages`, etc.
- SM90 FP8 typical config: `block_tile_shape=Index(128, 128, 128)`, `mma_shape=Index(64, 128, 32)`, `cluster_shape=Index(2, 1, 1)`

**Function signature:**
```mojo
fn mxfp4_matmul_sm90[
    c_type: DType,           # bfloat16
    a_type: DType,           # uint8 (packed fp4) or bfloat16 (activations)
    b_type: DType,           # uint8 (packed fp4 weights)
    scales_dtype: DType,     # float8_e8m0fnu
    //,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: NDBuffer[mut=True, c_type, 2, _, _],       # output [M, N]
    a: NDBuffer[a_type, 2, _, _],                   # activations [M, K] (bf16)
    b: NDBuffer[b_type, 2, _, _],                   # weights [N, K/2] (packed fp4)
    a_scales: LayoutTensor[...],                     # activation scales (if quantized)
    b_scales: LayoutTensor[...],                     # weight scales [5D interleaved]
    ctx: DeviceContext,
) raises
```

**Logic:**
1. Allocate temp FP8 buffer for dequantized B: `ctx.enqueue_create_buffer[float8_e4m3fn](N * K)`
2. If activations are BF16: quantize A to FP8 using dynamic scaling (reuse `quantize_dynamic_scaled_fp4fp8` with `MXFP8_SF_VECTOR_SIZE=32` from `fp4_quantization.mojo:96`)
3. Run `dequant_mxfp4_to_fp8` on B weights
4. Call `matmul_dispatch_sm90_fp8` (or `warp_specialize_gemm_with_multicasting` directly) with both FP8 tensors
5. Apply scale correction in epilogue

### Step 2.2: Write an end-to-end test for MXFP4 matmul on H100

**Create** `max/kernels/test/gpu/linalg/test_mxfp4_matmul_sm90.mojo`

**Reference test pattern:**
- `test/gpu/linalg/test_matmul_sm90_fp8.mojo:33-57` — Shows how to call `test_matmul_sm90[]` with FP8 config
- `test/gpu/linalg/test_matmul_sm100_block_scaled_nvfp4.mojo:58-120` — Full testbed with scale allocation

**Test procedure:**
1. Generate random BF16 activation matrix A [M, K]
2. Generate random BF16 weight matrix W [N, K]
3. Compute reference: C_ref = A @ W.T in BF16 (using cuBLAS via `vendor_blas.matmul`)
4. Quantize W to MXFP4 format:
   - Pack to uint8 [N, K/2]
   - Compute E8M0 scales per block of 32
   - Use `convert_ref_scales_to_mxfp8_format` from `fp4_utils.mojo:293-356` for scale layout
5. Run `mxfp4_matmul_sm90(C, A, W_fp4, scales, ctx)`
6. Compare C vs C_ref with appropriate tolerance (MXFP4 is ~2% relative error due to quantization)

**Test shapes (matching Llama-style GEMM shapes):**
- Small: M=32, N=128, K=256
- Medium: M=512, N=2560, K=8192 (from `sm90/dispatch.mojo:619-621` Llama-405B shapes)
- Dynamic M: M=1..128 with static N=8192, K=2048

**Add to BUILD.bazel and run:**
```bash
./bazelw test --config=remote-h100 //max/kernels/test/gpu/linalg:test_mxfp4_matmul_sm90
```

---

## Phase 3: Dispatch Integration

### Step 3.1: Add MXFP4 path to the GPU matmul dispatch

**Modify** `max/kernels/src/linalg/matmul/gpu/__init__.mojo`

At line 512 (inside the `comptime if ctx.default_device_info == H100:` block), add a new branch before the existing SM90 dispatch that checks for MXFP4 inputs:

```
comptime if ctx.default_device_info == H100:
    # NEW: MXFP4 dequant-to-FP8 path
    comptime if a_type == DType.uint8 and b_type == DType.uint8:
        # Route to mxfp4_matmul_sm90 (requires scale tensors)
        ...

    # Existing FP8/BF16 path
    var status = matmul_dispatch_sm90[...](c, a, b, ctx)
```

**However**, the current `matmul` dispatch at line 72 only takes `c, a, b` (no scale tensors). The block-scaled matmul has a separate entry point. There are two options:

**Option A (Recommended): Create a separate top-level function** like `mxfp4_matmul` similar to how `fp4_quantization.mojo` has `naive_block_scaled_matmul` (line 440). This is the pattern used by `dynamic_block_scaled_matmul_fp4` in MOGG.

**Option B: Extend the dispatch with optional scale parameters.**

Go with Option A — create a dedicated function that handles the full MXFP4-on-H100 path.

### Step 3.2: Register in MOGG (graph compiler integration)

**Modify** `max/kernels/src/Mogg/MOGGKernelAPI/MOGGKernelAPI.mojo`

Add a new operation name like `"mo.matmul.dynamic.block.scaled.mxfp4.h100"` that routes to the new `mxfp4_matmul_sm90` function when running on H100.

**Reference:**
- Search for `"mo.matmul.dynamic.block.scaled"` in `MOGGKernelAPI.mojo` to see how the existing NVFP4 matmul is registered
- The registration pattern involves matching on op name, extracting tensor operands, and calling the kernel

### Step 3.3: Test dispatch integration

Write a test that verifies the dispatch correctly routes MXFP4 inputs to the new path on H100 vs the native path on B200.

```bash
./bazelw test --config=remote-h100 //max/kernels/test/gpu/linalg:test_mxfp4_matmul_sm90
```

---

## Phase 4: Python-Level Integration

### Step 4.1: Add `matmul_mxfp4` to Python NN kernels

**Modify** `max/python/max/nn/kernels.py`

Add a new function similar to `dynamic_block_scaled_matmul_fp4` (search for it in kernels.py) that:
- Takes input tensor (BF16), MXFP4 weight tensor (uint8), and E8M0 scale tensor
- Calls the MOGG op registered in Step 3.2
- Returns BF16 output

### Step 4.2: Add `matmul_mxfp4` to float8_ops.py

**Modify** `max/python/max/nn/float8_ops.py`

Add a new function `matmul_mxfp4()` similar to `matmul_float4()` (lines 30-76):

```python
def matmul_mxfp4(
    x: TensorValue,           # BF16 activations
    weight: TensorValue,       # uint8 packed MXFP4 weights
    weight_scale: TensorValue, # E8M0 weight scales
    float8_config: Float8Config,
) -> TensorValue:
    """MXFP4 matmul with H100 dequant-to-FP8 fallback."""
    ...
```

### Step 4.3: Wire into Float8Config

**Modify** `max/python/max/nn/float8_config.py`

Add `is_mxfp4` property or extend the existing config to recognize MXFP4 quantization format. Reference the existing `is_nvfp4` check at `float8_ops.py:51`.

### Step 4.4: Wire into Linear layer

**Modify** `max/python/max/nn/linear.py`

At the point where `matmul_float4` is called (line 472), add a branch:
```python
if float8_config.is_mxfp4:
    res = matmul_mxfp4(x, weight, weight_scale, float8_config)
elif float8_config.is_nvfp4:
    res = matmul_float4(x, weight, weight_scale, input_scale, weight_scale_2, float8_config)
```

### Step 4.5: Test Python-level integration

```bash
./bazelw test //max/tests/integration/graph:test_matmul  # graph-level test
```

---

## Phase 5: Performance Optimization & Benchmarking

### Step 5.1: Create benchmark for MXFP4 matmul on H100

**Create** `max/kernels/benchmarks/gpu/bench_mxfp4_matmul_sm90.mojo`

**Reference:**
- `benchmarks/gpu/bench_matmul.mojo:14-59` — Benchmark imports and setup
- `benchmarks/gpu/bench_matmul.mojo:62-150` — `verify_matmul` pattern

**Benchmark shapes (Llama-8B/405B typical):**
```
M=1,    N=8192,  K=8192   (single token decode)
M=32,   N=8192,  K=8192   (small batch)
M=512,  N=8192,  K=2048   (medium batch)
M=4096, N=8192,  K=8192   (large batch)
M=512,  N=2560,  K=8192   (Llama-405B shape)
M=512,  N=16384, K=2048   (Llama-405B shape)
```

**Compare against:**
1. BF16 GEMM (cuBLAS baseline) — this is the "no quantization" reference
2. FP8 GEMM (existing SM90 path) — this is the "FP8 quantization" reference
3. MXFP4 dequant-to-FP8 GEMM — our new path

**Run:**
```bash
./bazelw run --config=remote-h100 //max/kernels/benchmarks/gpu:bench_mxfp4_matmul_sm90
```

### Step 5.2: Optimize the dequant kernel

Based on benchmark results, potential optimizations:
1. **Vectorized loads**: Load 16 bytes (128 FP4 values) per thread instead of 4 bytes
2. **Shared memory caching**: Cache scale factors in SMEM (one scale per 32 elements means high reuse)
3. **Fused dequant + GEMM**: Instead of writing dequantized FP8 to global memory, fuse the dequant into the tile loader of the SM90 GEMM kernel

### Step 5.3: (Advanced) Fused dequant in tile loader

For maximum performance, fuse the MXFP4 dequant into the SM90 matmul's tile loading phase:

**Modify** `max/kernels/src/linalg/matmul/gpu/sm90/tile_loader.mojo`

The `TileLoader` trait (line 48-71) defines `load_tile()`. Create a new loader that:
1. Loads uint8 (packed FP4) via TMA into shared memory
2. Each thread unpacks and dequantizes in registers
3. Stores FP8 values into the SMEM tile expected by WGMMA

This avoids the round-trip through global memory but is significantly more complex.

**Reference for tile loading:**
- `sm90/tile_loader.mojo:110-147` — `TMABarrierHandler` and `CPAsyncBarrierHandler`
- `sm90/matmul_kernels.mojo:103-200` — `HopperMatmulSM90Kernel_SMem` shared memory structure

---

## Phase 6: Model-Level Integration

### Step 6.1: Add MXFP4 weight adapter

Ensure the weight loading pipeline can load MXFP4-quantized checkpoints (e.g., from HuggingFace `float4_e2m1fn` format or GGUF with MXFP4 quantization).

**Reference:**
- `max/python/max/nn/linear.py` — `Float8Linear` class
- `max/python/max/pipelines/architectures/` — model-specific weight adapters

### Step 6.2: End-to-end model test

Run a small model (e.g., TinyLlama) with MXFP4 weights on H100:

```bash
./bazelw run //max/python/max/entrypoints:pipelines -- generate \
    --model <mxfp4-quantized-model> \
    --device gpu \
    --prompt "Hello, world!"
```

### Step 6.3: Accuracy verification

Use the logit verification framework:
```bash
./bazelw run //max/tests/integration/tools:generate_llm_logits -- \
    --device gpu --framework max --pipeline <model> --encoding mxfp4 \
    --output /tmp/max-mxfp4-logits.json

./bazelw run //max/tests/integration/accuracy:verify -- \
    --eval-metric cos,kl,tol \
    --cos-dist-threshold 0.005 \
    --kl-div-threshold 0.05 \
    /tmp/max-mxfp4-logits.json /tmp/torch-bf16-logits.json
```

---

## File Summary

### New Files
| File | Purpose |
|------|---------|
| `max/kernels/src/linalg/mxfp4_dequant.mojo` | MXFP4→FP8 dequant GPU kernel |
| `max/kernels/src/linalg/mxfp4_matmul_sm90.mojo` | Combined dequant+FP8 GEMM for H100 |
| `max/kernels/test/gpu/linalg/test_mxfp4_dequant.mojo` | Dequant kernel unit tests |
| `max/kernels/test/gpu/linalg/test_mxfp4_matmul_sm90.mojo` | End-to-end MXFP4 matmul tests |
| `max/kernels/benchmarks/gpu/bench_mxfp4_matmul_sm90.mojo` | Performance benchmarks |

### Modified Files
| File | Change |
|------|--------|
| `max/kernels/src/Mogg/MOGGKernelAPI/MOGGKernelAPI.mojo` | Register MXFP4 H100 op |
| `max/python/max/nn/kernels.py` | Add `mxfp4_matmul` kernel binding |
| `max/python/max/nn/float8_ops.py` | Add `matmul_mxfp4()` function |
| `max/python/max/nn/float8_config.py` | Add `is_mxfp4` config |
| `max/python/max/nn/linear.py` | Route MXFP4 to new matmul |
| `max/kernels/test/gpu/linalg/BUILD.bazel` | Add test targets |
| `max/kernels/benchmarks/gpu/BUILD.bazel` | Add benchmark target |

### Key Existing Files Referenced
| File | What it provides |
|------|-----------------|
| `max/kernels/src/linalg/fp4_utils.mojo` | FP4 conversion functions, scale layout constants |
| `max/kernels/src/linalg/fp4_quantization.mojo` | NVFP4 quantization kernel (template) |
| `max/kernels/src/linalg/matmul/gpu/sm90/dispatch.mojo` | SM90 FP8 dispatch logic |
| `max/kernels/src/linalg/matmul/gpu/sm90/matmul.mojo` | `warp_specialize_gemm_with_multicasting` |
| `max/kernels/src/linalg/matmul/gpu/sm90/testbed.mojo` | SM90 matmul test helper |
| `max/kernels/src/linalg/matmul/gpu/__init__.mojo` | Top-level GPU matmul dispatch |
| `max/kernels/src/linalg/matmul/gpu/sm100/block_scaled_dispatch.mojo` | SM100 NVFP4 dispatch (reference pattern) |

---

## Dependency Graph

```
Phase 1 (Dequant Kernel)
  ├── Step 1.1: Write kernel         ← depends on fp4_utils.mojo
  ├── Step 1.2: Write unit test      ← depends on Step 1.1
  └── Step 1.3: CPU reference test   ← depends on Step 1.2
          │
Phase 2 (Integration)
  ├── Step 2.1: Combined function    ← depends on Step 1.3 + sm90/dispatch.mojo
  └── Step 2.2: E2E test            ← depends on Step 2.1
          │
Phase 3 (Dispatch)                    ← depends on Phase 2
  ├── Step 3.1: matmul dispatch
  ├── Step 3.2: MOGG registration
  └── Step 3.3: Dispatch test
          │
Phase 4 (Python)                      ← depends on Phase 3
  ├── Steps 4.1-4.4: Python wiring
  └── Step 4.5: Python test
          │
Phase 5 (Performance)                 ← can start after Phase 2
  ├── Step 5.1: Benchmarks
  ├── Step 5.2: Dequant optimization
  └── Step 5.3: Fused tile loader    ← advanced, optional
          │
Phase 6 (Model)                       ← depends on Phase 4
  ├── Step 6.1: Weight adapter
  ├── Step 6.2: E2E model test
  └── Step 6.3: Accuracy verification
```

**Phases 1-2 are the critical path.** Get the dequant kernel correct and tested first, then build up from there. Phase 5 (benchmarks) can run in parallel with Phases 3-4.
