# MXFP4 Implementation Comparison: vLLM vs SGLang

## Executive Summary

Both vLLM and SGLang implement MXFP4 (Microscaling FP4, E2M1 format) primarily for
MoE (Mixture of Experts) layers in models like DeepSeek-V3 and GPT-OSS. They share
a nearly identical architecture — SGLang's code is adapted directly from vLLM's — but
diverge in backend availability, kernel implementations, and some quantization paths.

---

## 1. Data Format Overview

| Property | Value (Both) |
|---|---|
| FP4 format | E2M1 (2 exponent bits, 1 mantissa bit) |
| Representable values | ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} |
| Max value | 6.0 |
| Packing | 2 × FP4 values per `uint8` byte |
| Block size (MXFP4) | 32 elements per scale |
| Block size (NVFP4) | 16 elements per scale |
| Scale format (MXFP4) | E8M0 (unsigned, 8-bit exponent, 0-bit mantissa) stored as `uint8` |
| Scale format (NVFP4) | E4M3 (UE4M3) stored as `float8_e4m3fn` |
| Scale interpretation (E8M0) | `2^(uint8_value - 127)` |
| Weight storage | `torch.uint8` (packed FP4) |
| Scale storage | `torch.uint8` (E8M0) or `torch.float8_e4m3fn` (E4M3) |

### MXFP4 vs NVFP4

Both codebases distinguish two FP4 variants:

- **MXFP4 (OCP MX standard)**: Block size = 32, scale format = E8M0. Used for
  MoE weights in DeepSeek/GPT-OSS style models.
- **NVFP4 (NVIDIA proprietary)**: Block size = 16, scale format = UE4M3, plus a
  per-tensor global scale (`alpha = 1 / (input_global_scale * weight_global_scale)`).
  Used for dense linear layers via ModelOpt/compressed-tensors checkpoints.

---

## 2. When and Where Upcasting Happens

### 2.1 MXFP4 MoE Path (Both vLLM and SGLang)

There are **four distinct upcasting strategies** depending on the backend:

#### Path A: Triton Kernels (GPT-OSS / OAI `triton_kernels`)
**When**: SM90 (Hopper) or SM100 (Blackwell) with `triton_kernels` package available.

- **Weights**: FP4 (E2M1) stays packed as `uint8`. During `process_weights_after_loading()`,
  weights are *swizzled* (layout-converted) into hardware-optimal TMA block layout via
  `triton_kernels.tensor.convert_layout()` with `FP4` dtype tag — but **not upcast**
  to a higher precision at load time.
- **Activations**: Passed as BF16/FP16 to the Triton MoE kernel. The kernel handles
  quantization internally.
- **Actual upcasting**: Happens **inside the Triton matmul kernel** (`triton_kernels.matmul_ogs`).
  The kernel reads FP4 values, dequantizes using the E8M0 scales, and accumulates in **FP32**.
  The output is **BF16**.
- **Scale handling**: Scales are swizzled into a hardware-specific layout
  (`make_default_matmul_mxfp4_w_scale_layout`) and passed via `PrecisionConfig`.

#### Path B: FlashInfer + TRT-LLM Gen (SM100/Blackwell)
**When**: Blackwell GPU with FlashInfer installed, using `trtllm_fp4_block_scale_moe`.

- **Weights**: FP4 stays packed as `uint8`. Scales are interleaved via
  `nvfp4_block_scale_interleave()` and permuted for coalesced access.
  Scales are `.view(torch.float8_e4m3fn)` after shuffling.
- **Activations**: Two sub-paths:
  - `precision="bf16"`: Activations stay **BF16**. The TRT-LLM kernel handles
    quantization internally, pipelined with GEMM.
  - `precision="default"`: Activations are quantized to **MXFP8** (FP8 E4M3 with
    E8M0 block scales) via `flashinfer.mxfp8_quantize()` before the kernel call.
- **Actual upcasting**: Inside the TRT-LLM fused MoE kernel. Accumulation is in **FP32**,
  output is **BF16**.

#### Path C: BF16 Fallback (No specialized backend)
**When**: No Triton kernels or FlashInfer available, no Marlin.

- **Weights**: Fully upcast from FP4 to **BF16** at load time via
  `triton_kernels.numerics_details.mxfp.upcast_from_mxfp()` with
  `target_dtype=torch.bfloat16`.
- **Activations**: Stay in original dtype (BF16).
- **Result**: Standard BF16 MoE computation with no FP4 at runtime. This is a
  **pure dequantization-at-load** strategy — no FP4 kernels are called at inference.

Both vLLM and SGLang code paths (lines ~720-739 in SGLang's `mxfp4.py`):
```python
from triton_kernels.numerics_details.mxfp import upcast_from_mxfp
w13_weight = upcast_from_mxfp(
    layer.w13_weight,
    layer.w13_weight_scale,
    target_dtype=torch.bfloat16,
    axis=-1,
)
```

#### Path D: Marlin Backend (vLLM only, limited SGLang support)
**When**: GPU supports Marlin (SM80+).

- **Weights**: Packed FP4 reprocessed via `prepare_moe_fp4_layer_for_marlin()`.
- **Activations**: Quantized to **FP8 E4M3** via Marlin's internal quantization.
- **GEMM**: `ops.marlin_gemm()` with `b_q_type=scalar_types.float4_e2m1f`.
  Marlin unpacks FP4 and applies scales during the GEMM.
- **Output**: BF16/FP16.

#### Path E: AMD ROCm / AITER (Both, AMD GFX950)
**When**: AMD MI3xx GPUs with AITER installed.

- **Weights**: FP4 packed, shuffled via `shuffle_weight_a16w4()` with
  `shuffle_scale_a16w4()` for E8M0 scales.
- **Activations (static MXFP4 checkpoints)**: Passed as BF16, padded to alignment.
  The AITER `fused_moe()` kernel handles quantization with `QuantType.per_1x32`.
- **Activations (dynamic quantization)**: Quantized via AITER's
  `dynamic_mxfp4_quant()`, then shuffled with `shuffle_weight()`.
- **Output**: BF16.

### 2.2 NVFP4 Dense Linear Path (Both)

For non-MoE linear layers using NVFP4 (block_size=16, UE4M3 scales):

1. **Activation quantization**: BF16/FP16 → FP4 packed `uint8` + swizzled E4M3 block scales.
   - SGLang uses `sglang.jit_kernel.nvfp4.scaled_fp4_quant()` (custom CUDA kernel) or
     `flashinfer.fp4_quantize()` on Blackwell.
   - vLLM uses `csrc/quantization/fp4/nvfp4_quant_kernels.cu` (`cvt_fp16_to_fp4`).
2. **GEMM**: CUTLASS SM100 FP4 GEMM (`cutlass::nv_float4_t<cutlass::float_e2m1_t>`)
   or FlashInfer's `mm_fp4()`.
3. **Output**: BF16, FP16, or FP32 depending on configuration.
4. **Alpha scaling**: `alpha = 1 / (input_global_scale * weight_global_scale)` applied
   as epilogue fusion in the CUTLASS kernel.

---

## 3. Kernel Backends Summary

### 3.1 vLLM Kernel Backends

| Backend | GPU | Kernel | Activation Format | Weight Format | Scale Format |
|---|---|---|---|---|---|
| **OAI Triton** (`triton_kernels.matmul_ogs`) | SM90, SM100 | Triton MoE GEMM | BF16 (quantized internally) | FP4 E2M1 (swizzled) | E8M0 (swizzled) |
| **FlashInfer TRTLLM** | SM100 | `flashinfer_scaled_fp4_mm` / `trtllm_fp4_block_scale_moe` | MXFP8 or BF16 | FP4 E2M1 (permuted) | E8M0→E4M3 (interleaved) |
| **FlashInfer CUTLASS** | SM100 | `flashinfer_scaled_fp4_mm` | MXFP8 | FP4 E2M1 (permuted) | E8M0→E4M3 (interleaved) |
| **FlashInfer BF16** | SM90, SM100 | FlashInfer fused MoE | BF16 | FP4 E2M1 | E8M0 |
| **Marlin** | SM80+ | `ops.marlin_gemm` | FP8 E4M3 | FP4 E2M1 (Marlin layout) | E8M0→E8M0FNU |
| **AMD CK/AITER** | GFX950 | `aiter.fused_moe` | BF16 (AITER quantizes) | FP4 E2M1 (shuffled) | E8M0 (shuffled) |
| **BF16 Fallback** | Any | Standard matmul | BF16 | **Upcast to BF16 at load** | N/A (applied at load) |
| **CUTLASS SM100** (dense NVFP4) | SM100+ | CUTLASS `GemmUniversal` with `OpClassBlockScaledTensorOp` | FP4 E2M1 (packed) | FP4 E2M1 (packed) | UE4M3 (swizzled) |

### 3.2 SGLang Kernel Backends

| Backend | GPU | Kernel | Activation Format | Weight Format | Scale Format |
|---|---|---|---|---|---|
| **OAI Triton** (`triton_kernels.matmul_ogs`) | SM90, SM100, SM120 | Triton MoE GEMM | BF16 (quantized internally) | FP4 E2M1 (swizzled) | E8M0 (swizzled) |
| **FlashInfer TRTLLM Gen** | SM100 | `trtllm_fp4_block_scale_moe` | MXFP8 or BF16 | FP4 E2M1 (permuted+shuffled) | E8M0→E4M3 (interleaved) |
| **AMD AITER** | GFX950 | `aiter.fused_moe` with `QuantType.per_1x32` | BF16 (AITER quantizes) | FP4 E2M1 (shuffled) | E8M0 (shuffled) |
| **BF16 Fallback** | Any | `upcast_from_mxfp` then standard matmul | BF16 | **Upcast to BF16 at load** | N/A |
| **CUTLASS FP4 MoE** (JIT) | SM100, SM120 | `cutlass_fp4_group_mm` via JIT compilation | FP4 (quantized by `scaled_fp4_experts_quant`) | FP4 E2M1 (packed) | UE4M3 (swizzled) |
| **CUTLASS FP4 Dense** (JIT) | SM100, SM120 | `cutlass_scaled_fp4_mm` via JIT compilation | FP4 (quantized by `scaled_fp4_quant`) | FP4 E2M1 (packed) | UE4M3 (swizzled) |
| **FlashInfer `mm_fp4`** (dense) | SM100+ | FlashInfer GEMM (trtllm/cutlass/cudnn backends) | FP4 | FP4 | UE4M3 |

---

## 4. Detailed Quantization Flow

### 4.1 MXFP4 Weight Quantization (Offline, E8M0 Scales, Block=32)

Both frameworks load MXFP4 checkpoints where weights are pre-quantized:

```
Original weights (FP16/BF16) [offline]
  │
  ├─ Reshape to blocks of 32: [..., -1, 32]
  ├─ Compute per-block max: block_max = max(|x|) per block
  ├─ Compute E8M0 scale: scale = clamp(127 + floor(log2(block_max)) - 2, 0, 254)
  ├─ Scale input: x_scaled = x / 2^(scale - 127)
  ├─ Round to nearest E2M1 value: x_fp4 = round_to_e2m1(x_scaled)
  ├─ Pack 2 FP4 values per byte: output[i] = (fp4_odd << 4) | fp4_even
  │
  └─ Stored: weight_packed (uint8), weight_scale (uint8 E8M0)
```

### 4.2 NVFP4 Activation Quantization (Online, E4M3 Scales, Block=16)

The CUDA kernel `cvt_fp16_to_fp4` (identical logic in both codebases):

```
Input activations (BF16/FP16)
  │
  ├─ Warp-level max: each pair of threads computes max over 16 elements
  ├─ Compute scale factor: SF = SFScaleVal × (vecMax / 6.0)
  ├─ Convert to E4M3: fp8SFVal = __nv_fp8_e4m3(SF)
  │   (or E8M0: fp8SFVal = __nv_cvt_float_to_e8m0(SF) when UE8M0_SF=true)
  ├─ Compute output scale: 1/fp32(fp8(SF × 1/SFScaleVal)) × 1/SFScaleVal
  ├─ Scale and convert: x × outputScale → fp32 → E2M1 via fp32_vec_to_e2m1()
  ├─ Pack 8 E2M1 values into uint32
  │
  └─ Output: packed FP4 (uint8), swizzled E4M3 block scales
```

Where `SFScaleVal = 448.0 * 6.0 / tensor_amax` is the global scale (precomputed).

### 4.3 Dequantization (Reference Path)

From vLLM's `reference_mxfp4.py` (used for testing, not inference):

```
Packed FP4 (uint8) + E8M0 scales (uint8)
  │
  ├─ Unpack: low_nibble = byte & 0x0F, high_nibble = (byte >> 4) & 0x0F
  ├─ Extract fields: sign = bit[3], exp = bits[2:1], mantissa = bit[0]
  ├─ Reconstruct FP16/BF16:
  │     new_exp = exp - 1 + half_exp_bias  (if not subnormal)
  │     result = (sign << 15) | (new_exp << mantissa_bits) | (mantissa << (mantissa_bits-1))
  │     → reinterpret as float16/bfloat16
  ├─ Apply block scale: result × 2^(e8m0_scale - 127)
  │
  └─ Output: FP16 or BF16 tensor
```

---

## 5. CUTLASS Kernel Configuration Details

Both codebases use CUTLASS 3.x with `OpClassBlockScaledTensorOp` for FP4 GEMM.

### SM100 (Blackwell) Configurations:

| M Range | Tile Shape (M×N×K) | Cluster | Schedule |
|---|---|---|---|
| M ≤ 128 | 128×256×256 | (1,4,1) | `KernelTmaWarpSpecialized1SmNvf4Sm100` |
| 128 < M ≤ 256 | 256×256×256 | (2,4,1) | `KernelTmaWarpSpecialized2SmNvf4Sm100` |
| M > 256 | 256×256×256 | (4,4,1) | `KernelTmaWarpSpecialized2SmNvf4Sm100` |

### SM120 (Blackwell Desktop) Configurations:

| M Range | Tile Shape | Cluster |
|---|---|---|
| M ≤ 256 | 128×128×128 | (1,1,1) |
| M > 256 | 256×128×128 | (1,1,1) |

### Element types in CUTLASS:
```cpp
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // FP4
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;  // FP4
using ElementSFA = cutlass::float_ue4m3_t;                      // Scale (UE4M3)
using ElementSFB = cutlass::float_ue4m3_t;                      // Scale (UE4M3)
using ElementAccumulator = float;                                 // FP32 accumulation
using ElementD = cutlass::bfloat16_t / cutlass::half_t / float;  // Output
```

---

## 6. GPT-OSS Specific Details

GPT-OSS models use the MXFP4 MoE path. Key observations:

### vLLM
- GPT-OSS model class loads MXFP4 checkpoints with `quant_method="mxfp4"`.
- MoE layers use `Mxfp4MoEMethod` which selects between Triton, FlashInfer,
  Marlin, AITER, or BF16 fallback.
- The primary kernel for GPT-OSS is `triton_kernels.matmul_ogs.matmul_ogs`
  (the OpenAI Triton MoE kernel).
- Backend selection via `get_mxfp4_backend()` which checks GPU capability and
  available packages.

### SGLang
- GPT-OSS model at `sglang/srt/models/gpt_oss.py` uses the same `Mxfp4Config`.
- MoE layers use `Mxfp4MoEMethod` (adapted from vLLM) with similar backend selection.
- The primary kernel path depends on `get_moe_runner_backend()`:
  - `MoeRunnerBackend.TRITON_KERNELS` → OAI Triton kernel (same as vLLM)
  - `MoeRunnerBackend.TRITON` → Standard Triton fused MoE
- SGLang additionally supports a CUTLASS FP4 grouped GEMM path via JIT-compiled
  kernels (`cutlass_moe_fp4` in `cutlass_moe.py`), which vLLM accesses through
  different abstractions.

### Activation Handling for GPT-OSS MoE
- **FlashInfer TRTLLM path**: Activations quantized to MXFP8 (FP8 E4M3 + E8M0
  scales) via `mxfp8_quantize()`, or kept as BF16.
- **Triton kernels path**: Activations passed as BF16; the Triton kernel handles
  any internal quantization.
- **CUTLASS FP4 MoE path** (SGLang): Activations quantized to NVFP4 (FP4 + UE4M3
  scales) via `scaled_fp4_experts_quant()` using a per-expert global scale.

---

## 7. Key Differences Between vLLM and SGLang

| Aspect | vLLM | SGLang |
|---|---|---|
| **Code origin** | Original implementation | Adapted from vLLM (stated in file headers) |
| **Marlin backend** | Full support for MXFP4 MoE | Limited (primarily uses Triton/FlashInfer) |
| **CUTLASS FP4 MoE** | Via FlashInfer/Marlin abstractions | Direct JIT-compiled CUTLASS kernels (`jit_kernel/nvfp4.py`) |
| **JIT compilation** | Not used for FP4 kernels | Uses `tvm_ffi` JIT to compile CUTLASS FP4 kernels on-the-fly |
| **SM120 support** | Limited mentions | Explicit SM120 CUTLASS configs with `StridedLayout` fallback |
| **FP4 GEMM backends** | Implicit in backend selection | Explicit `Fp4GemmRunnerBackend` enum (auto/cudnn/cutlass/trtllm) |
| **FlashInfer MoE precision** | Fixed modes | Configurable `flashinfer_mxfp4_moe_precision` server arg |
| **Compressed tensors** | `CompressedTensorsW4A16Mxfp4` class | `CompressedTensorsW4A4Fp4` (NVFP4) and `CompressedTensorsW4A4NvFp4Moe` |
| **AMD dynamic quant** | Via Quark's `qdq_mxfp4()` | Via AITER's `dynamic_mxfp4_quant()` + `e8m0_shuffle()` |
| **Backend enum** | `Mxfp4Backend` (7 variants) | `get_moe_runner_backend()` returns MoeRunnerBackend |

---

## 8. Upcasting Summary Table

| Scenario | Source Format | Upcast Target | When | Where |
|---|---|---|---|---|
| MXFP4 weight at load (fallback) | FP4 E2M1 + E8M0 | **BF16** | `process_weights_after_loading` | `upcast_from_mxfp()` |
| MXFP4 activation (FlashInfer default) | BF16 | **MXFP8** (FP8 E4M3 + E8M0) | Before MoE kernel | `mxfp8_quantize()` |
| MXFP4 activation (FlashInfer bf16) | BF16 | **No upcast** (stays BF16) | Before MoE kernel | Direct pass |
| NVFP4 activation quantization | BF16/FP16 | **FP4 E2M1** + UE4M3 scales | Before GEMM | `scaled_fp4_quant()` / `cvt_fp16_to_fp4` |
| NVFP4 GEMM accumulation | FP4 × FP4 | **FP32** | During GEMM | CUTLASS tensor core |
| NVFP4 GEMM output | FP32 accumulator | **BF16/FP16** | End of GEMM | CUTLASS epilogue |
| Marlin MXFP4 activation | BF16/FP16 | **FP8 E4M3** | Before Marlin GEMM | Marlin quantization |
| AMD AITER MoE | BF16 | **FP4** (internal) | During AITER MoE | AITER `fused_moe()` |
| Dequantization (reference/test) | FP4 E2M1 + E8M0 | **FP16 or BF16** | Testing only | Bit manipulation |

---

## 9. File Reference

### vLLM Key Files
- `vllm/model_executor/layers/quantization/mxfp4.py` — Main MXFP4 config and MoE method
- `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py` — Swizzle, dequant, backend utils
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py` — Marlin FP4 support
- `vllm/csrc/quantization/fp4/nvfp4_quant_kernels.cu` — CUDA FP4 quantization kernel
- `vllm/csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu` — CUTLASS FP4 GEMM kernel
- `tests/quantization/reference_mxfp4.py` — Pure-Python reference implementation

### SGLang Key Files
- `python/sglang/srt/layers/quantization/mxfp4.py` — Main MXFP4 config and MoE method
- `python/sglang/srt/layers/quantization/mxfp4_tensor.py` — MXFP4 quantize/dequantize utilities
- `python/sglang/srt/layers/quantization/fp4_utils.py` — FP4 GEMM backend selection
- `python/sglang/srt/layers/quantization/modelopt_quant.py` — NVFP4 linear + MoE methods
- `python/sglang/srt/layers/moe/cutlass_moe.py` — CUTLASS MoE (FP8 and FP4)
- `python/sglang/jit_kernel/nvfp4.py` — JIT-compiled NVFP4 kernels
- `python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_quant_kernels.cuh` — CUDA FP4 quantization
- `python/sglang/jit_kernel/csrc/gemm/nvfp4/nvfp4_scaled_mm_kernels.cuh` — CUTLASS FP4 GEMM
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` — Compressed tensors NVFP4
