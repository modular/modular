# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import os

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops

from .kernels import (
    _fused_qkv_ragged_matmul_scaled_float4,
    _fused_qkv_ragged_matmul_scaled_float8,
    _is_pre_sm100_nvidia_gpu,
    block_scales_interleave,
    convert_weights_to_fp8_fnuz_if_needed,
    dynamic_block_scaled_matmul,
    dynamic_block_scaled_matmul_mxfp4,
    dynamic_scaled_matmul,
    fused_qkv_ragged_matmul,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_matmul_ragged,
    matmul_static_scaled_float8,
    mxfp4_dequant,
    nvfp4_gemm,
    nvfp4_gemv,
    nvfp4_skeleton_gemm,
    quantize_dynamic_block_scaled,
    quantize_dynamic_block_scaled_mxfp4,
    quantize_dynamic_scaled_float8,
    quantize_static_scaled_float8,
    quantize_tensor_dynamic_scaled_float8,
)
from .kv_cache import KVCacheParams, PagedCacheValues
from .quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    WeightScaleSpec,
)


def _reshape_pre_interleaved_scales(
    weight_scale: TensorValue,
) -> TensorValue:
    """Reshape pre-interleaved FP4 scales from 2D to 5D TCGEN layout.

    Checkpoints with pre-interleaved scales store them in 5D TCGEN order
    but flattened to `[M, K//16]`.  The matmul kernel requires rank-5
    input: `(M//128, K//64, 32, 4, 4)`.
    """
    M = weight_scale.shape[0]
    n_blocks = weight_scale.shape[1]
    SF_ATOM_K = 4
    SF_MN_GROUP_SIZE = 128
    SF_ATOM_M0 = 32
    return weight_scale.reshape(
        [
            M // SF_MN_GROUP_SIZE,
            n_blocks // SF_ATOM_K,
            SF_ATOM_M0,
            SF_MN_GROUP_SIZE // SF_ATOM_M0,
            SF_ATOM_K,
        ]
    )


# Activation-row count (M) above which the pre-Blackwell NVFP4 path switches
# from the fused dequant-GEMV (decode) to a dequant + BF16 tensor-core GEMM
# (prefill / high-concurrency decode). Override via env var to tune.
_NVFP4_PRE_SM100_GEMM_M_THRESHOLD = int(
    os.environ.get("MAX_NVFP4_GEMM_M_THRESHOLD", "32")
)

# Opt-in: route the pre-Blackwell large-M GEMM through the multistage-skeleton
# NVFP4 path (repack + qmatmul_nvfp4_g16) instead of the bespoke nvfp4_gemm.
# The skeleton is bit-exact and ~2.5x faster at gemma4 shapes on L40S, but the
# graph route is gated behind this flag until validated token-identical in
# serving (it changes the live NVFP4 dispatch). Default off = unchanged.
_NVFP4_USE_SKELETON_GEMM = os.environ.get("MAX_NVFP4_SKELETON_GEMM", "0") == "1"


def _matmul_float4(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue | None,
    input_scale: TensorValue | None,
    weight_scale_2: TensorValue | None,
    scales_pre_interleaved: bool = False,
) -> TensorValue:
    """Computes x @ weight.T with modelopt NVFP4 quantization.

    On NVIDIA GPUs without native FP4 matmul (pre-Blackwell), the fused
    dequant-GEMV decodes the packed weight in registers instead
    (``input_scale`` is unused there — activations are never quantized on
    the fallback path).

    Args:
        x: The input tensor in bf16.
        weight: The weight tensor in uint8 (float4-e2m1x2).
        weight_scale: The weight scale tensor in f8e4m3fn.
        input_scale: The input scale factor in f32 (used with vLLM convention by kernel).
        weight_scale_2: Additional weight scale factor in f32.
        scales_pre_interleaved: If True, weight_scale is already in 5D
            TCGEN interleaved layout and `block_scales_interleave` is
            skipped.

    Returns:
        The output tensor in bf16.
    """
    if _NVFP4_USE_SKELETON_GEMM:
        # Load-time skeleton path: ``weight`` is already the combined buffer
        # (repacked 4-bit weights + bf16-folded block scales) produced by the
        # host repack in the weight adapter. There are no separate scales, and
        # no in-graph repack -- run the skeleton GEMM directly.
        return nvfp4_skeleton_gemm(x.cast(DType.bfloat16), weight)

    assert weight_scale is not None
    assert input_scale is not None
    assert weight_scale_2 is not None

    if _is_pre_sm100_nvidia_gpu():
        # Pre-Blackwell has no native FP4 tensor cores, so dispatch by the
        # number of activation rows M (= tokens in flight this step):
        #   * Small M (single / low-concurrency decode): fused dequant-GEMV.
        #     FP4 is decoded in registers, the BF16 weight is never
        #     materialized, and per-token DRAM traffic is the packed bytes
        #     only -- optimal when M is tiny and the matmul is bandwidth bound.
        #   * Large M (prefill or high-concurrency decode): dequantize the
        #     weight to BF16 once and run a BF16 tensor-core GEMM. The GEMV
        #     re-reads the packed weight once per M_TILE rows and uses scalar
        #     FMAs, so it does not scale with M; the GEMM amortizes a single
        #     dequant over all rows and uses the tensor cores. The dequant is
        #     transient (one weight at a time), so it never needs the ~3x VRAM
        #     a hoisted full-BF16 weight would.
        if scales_pre_interleaved:
            raise ValueError(
                "NVFP4 checkpoints with pre-interleaved (TCGEN 5D) scales"
                " are not supported on pre-Blackwell GPUs"
            )
        scales_f32 = weight_scale.to(weight.device).cast(
            DType.float32
        ) * weight_scale_2.to(weight.device)
        x_bf16 = x.cast(DType.bfloat16)

        out_type = TensorType(
            dtype=DType.bfloat16,
            shape=[x_bf16.shape[0], weight.shape[0]],
            device=x_bf16.device,
        )

        def _gemv() -> TensorValue:
            return nvfp4_gemv(x_bf16, weight, scales_f32)

        def _gemm() -> TensorValue:
            return nvfp4_gemm(x_bf16, weight, scales_f32)

        m = ops.shape_to_tensor(x_bf16.shape)[0]
        use_gemm = m > ops.constant(
            _NVFP4_PRE_SM100_GEMM_M_THRESHOLD,
            DType.int64,
            device=DeviceRef.CPU(),
        )
        return ops.cond(use_gemm, [out_type], _gemm, _gemv)[0].tensor

    x, x_scales = quantize_dynamic_block_scaled(
        x,
        tensor_sf=1.0 / input_scale,
        scales_type=DType.float8_e4m3fn,
        out_type=DType.uint8,  # fp4-e2m1fnX2
    )

    weight_scale = weight_scale.to(x.device)
    if scales_pre_interleaved:
        weight_scale = _reshape_pre_interleaved_scales(weight_scale)
    else:
        weight_scale = block_scales_interleave(weight_scale)

    res = dynamic_block_scaled_matmul(
        x,
        weight,
        x_scales,
        weight_scale,
        tensor_sf=weight_scale_2 * input_scale,
        out_type=DType.bfloat16,
    )
    return res


def _matmul_float4_mxfp4(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
) -> TensorValue:
    """Computes x @ weight.T with MXFP4 quantization.

    Args:
        x: The input tensor in bf16.
        weight: The weight tensor in uint8 (float4-e2m1x2).
        weight_scale: The weight scale tensor in float8_e8m0fnu.

    Returns:
        The output tensor in bf16.
    """
    x, x_scales = quantize_dynamic_block_scaled_mxfp4(
        x,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.uint8,  # fp4-e2m1fnX2
    )

    weight_scale = weight_scale.to(x.device)

    res = dynamic_block_scaled_matmul_mxfp4(
        x,
        weight,
        x_scales,
        weight_scale,
        out_type=DType.bfloat16,
    )
    return res


def _dequantize_mxfp8_weight(
    weight: TensorValue,
    weight_scale: TensorValue,
) -> TensorValue:
    """Dequantize an MXFP8 weight to bf16.

    ``weight`` is ``float8_e4m3fn`` ``[N, K]`` and ``weight_scale`` is
    ``float8_e8m0fnu`` ``[N, K // 32]`` (one power-of-two scale per 32-element
    K block). Returns ``weight * scale`` in bf16, with each scale broadcast
    across its 32-element block.
    """
    SF_VECTOR_SIZE = 32
    n_dim = weight.shape[0]
    k_dim = weight.shape[1]
    k_blocks = k_dim // SF_VECTOR_SIZE
    # E8M0 -> f32 yields the exact power-of-two value; E4M3 -> f32 is exact.
    w = weight.cast(DType.float32)
    scale = weight_scale.cast(DType.float32)
    scale = ops.broadcast_to(
        ops.reshape(scale, [n_dim, k_blocks, 1]),
        [n_dim, k_blocks, SF_VECTOR_SIZE],
    )
    scale = ops.reshape(scale, [n_dim, k_dim])
    return (w * scale).cast(DType.bfloat16)


def _matmul_float8_mxfp8(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
) -> TensorValue:
    """Computes ``x @ weight.T`` with MXFP8 quantization.

    MXFP8 sibling of :func:`_matmul_float4_mxfp4`: the activation is
    dynamically quantized to ``float8_e4m3fn`` with E8M0 block scales, the
    ``float8_e8m0fnu`` weight scales are interleaved into the rank-5 SF-atom
    layout, and the SM100 block-scaled tensor-core MMA
    (``UMMAKind.KIND_MXF8F6F4``) is used -- avoiding the naive CUDA-core
    blockwise-FP8 fallback.

    Args:
        x: The input tensor in bf16.
        weight: The weight tensor in ``float8_e4m3fn``, shape ``[N, K]``.
        weight_scale: The E8M0 (``float8_e8m0fnu``) weight scales, rank-2
            ``[N, K // 32]`` as loaded from the checkpoint.

    Returns:
        The output tensor in bf16.
    """
    x, x_scales = quantize_dynamic_block_scaled(
        x,
        sf_vector_size=32,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )

    weight_scale = weight_scale.to(x.device)
    # Lift the rank-2 [N, K/32] checkpoint scales into the rank-5 tcgen05
    # SF-atom layout the block-scaled MMA consumes.
    weight_scale = block_scales_interleave(weight_scale, sf_vector_size=32)

    return dynamic_block_scaled_matmul(
        x,
        weight,
        x_scales,
        weight_scale,
        sf_vector_size=32,
        out_type=DType.bfloat16,
    )


def _matmul_float8(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    input_scale: TensorValue | None,
    quant_config: QuantConfig,
) -> TensorValue:
    """Computes x @ weight.T with float8 quantization.

    Args:
        x: The input tensor.
        weight: The weight tensor.
        weight_scale: The weight scale tensor.
        input_scale: The input scale tensor (only required for static
            fp8 quantization).
        quant_config: The quantization configuration.

    Returns:
        The output tensor.
    """
    weight, weight_scale = convert_weights_to_fp8_fnuz_if_needed(
        weight, weight_scale
    )

    if input_scale is not None:
        # Static quantization: input scale was pre-computed.
        x = quantize_static_scaled_float8(x, input_scale, out_type=weight.dtype)
        if quant_config.weight_scale.is_tensor and weight_scale.rank >= 2:
            # Fused QKV: per-projection weight scales were broadcast to
            # rowwise [N, 1] in qkv_weight_scale. Route through
            # dynamic_scaled_matmul (colwise/rowwise) because
            # matmul_static_scaled_float8 requires scalar scales.
            weight_scale = weight_scale.to(x.device)
            # Broadcast scalar input_scale to [1, M] to match the
            # per-token layout that the colwise/rowwise kernel expects.
            a_scales = ops.broadcast_to(
                input_scale.reshape([1, 1]).to(x.device), [1, x.shape[0]]
            )
            colwise_input_spec = InputScaleSpec(
                granularity=ScaleGranularity.COLWISE,
                origin=quant_config.input_scale.origin,
                dtype=weight_scale.dtype,
            )
            rowwise_weight_spec = WeightScaleSpec(
                granularity=ScaleGranularity.ROWWISE,
                dtype=weight_scale.dtype,
            )
            return dynamic_scaled_matmul(
                x,
                weight,
                a_scales,
                weight_scale,
                colwise_input_spec,
                rowwise_weight_spec,
                out_type=DType.bfloat16,
            )
        return matmul_static_scaled_float8(x, weight, input_scale, weight_scale)
    elif (
        quant_config.input_scale.is_tensor
        and quant_config.weight_scale.is_tensor
    ):
        # GEX-3496 workaround for dynamic tensor-wise FP8: force
        # tensor-scale into row-scale format so we hit the
        # rowwise/colwise kernel path.
        n_out = weight.shape[0]
        if quant_config.weight_scale.is_tensor and weight_scale.rank < 2:
            weight_scale = ops.broadcast_to(
                weight_scale.reshape([1, 1]), [n_out, 1]
            )
        elif weight_scale.rank < 2:
            weight_scale = weight_scale.reshape([1, 1])
        weight_scale = weight_scale.to(x.device)

        colwise_input_spec = InputScaleSpec(
            granularity=ScaleGranularity.COLWISE,
            origin=quant_config.input_scale.origin,
            dtype=weight_scale.dtype,
        )
        rowwise_weight_spec = WeightScaleSpec(
            granularity=ScaleGranularity.ROWWISE,
            dtype=weight_scale.dtype,
        )

        x, x_scales = quantize_dynamic_scaled_float8(
            x,
            colwise_input_spec,
            rowwise_weight_spec,
            scales_type=weight_scale.dtype,
            out_type=weight.dtype,
        )

        # activation scales are [1xm] which but in the dynamic scaled matmul we expect just use x_scale[0,0]
        return dynamic_scaled_matmul(
            x,
            weight,
            x_scales,
            weight_scale,
            colwise_input_spec,
            rowwise_weight_spec,
            out_type=DType.bfloat16,
        )
    else:
        # Dynamic non-per-tensor (per-row/block).
        x, x_scale = quantize_dynamic_scaled_float8(
            x,
            quant_config.input_scale,
            quant_config.weight_scale,
            scales_type=weight_scale.dtype,
            out_type=weight.dtype,
        )
        weight_scale = weight_scale.to(x.device)

    return dynamic_scaled_matmul(
        x,
        weight,
        x_scale,
        weight_scale,
        quant_config.input_scale,
        quant_config.weight_scale,
        out_type=DType.bfloat16,
    )


def quantized_matmul(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    input_scale: TensorValue | None,
    quant_config: QuantConfig,
    weight_scale_2: TensorValue | None = None,
) -> TensorValue:
    """Single entry point for all quantized dense matmuls.

    Dispatches to the appropriate kernel based on the quantization format
    in ``quant_config``.

    Args:
        x: The input tensor.
        weight: The weight tensor.
        weight_scale: The weight scale tensor.
        input_scale: The input scale tensor (required for NVFP4 and
            static FP8).
        quant_config: The quantization configuration.
        weight_scale_2: Additional weight scale factor (NVFP4 only).

    Returns:
        The output tensor.
    """
    match quant_config.format:
        case QuantFormat.NVFP4:
            if not _NVFP4_USE_SKELETON_GEMM:
                # The load-time skeleton path folds the scales into the
                # combined weight buffer, so input_scale / weight_scale_2 are
                # intentionally absent there.
                assert input_scale is not None
                assert weight_scale_2 is not None
            return _matmul_float4(
                x,
                weight,
                weight_scale,
                input_scale,
                weight_scale_2,
                scales_pre_interleaved=quant_config.scales_pre_interleaved,
            )
        case QuantFormat.MXFP4:
            return _matmul_float4_mxfp4(
                x,
                weight,
                weight_scale,
            )
        case QuantFormat.MXFP8:
            return _matmul_float8_mxfp8(
                x,
                weight,
                weight_scale,
            )
        case (
            QuantFormat.COMPRESSED_TENSORS_FP8
            | QuantFormat.FBGEMM_FP8
            | QuantFormat.BLOCKSCALED_FP8
        ):
            return _matmul_float8(
                x,
                weight,
                weight_scale,
                input_scale,
                quant_config,
            )
        case _:
            raise ValueError(
                f"Unsupported quantization format for dense matmul: {quant_config.format}"
            )


def quantized_fused_qkv_matmul(
    kv_params: KVCacheParams,
    x: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    input_row_offsets: TensorValue,
    n_heads: int,
    quant_config: QuantConfig,
    weight_scale: TensorValue,
    input_scale: TensorValue | None = None,
    weight_scale_2: TensorValue | None = None,
    bias: TensorValue | None = None,
    _output_dim: int | None = None,
) -> TensorValue:
    """Single entry point for quantized fused QKV matmuls.

    Dispatches to the NVFP4 or FP8 fused QKV kernel based on the
    quantization format in ``quant_config``.

    Args:
        kv_params: KV cache parameters.
        x: The input tensor of shape ``[total_seq_len, hidden_dim]``.
        wqkv: The concatenated QKV weight tensor.
        kv_collection: The paged KV cache.
        layer_idx: The current layer index.
        input_row_offsets: Batch boundary offsets.
        n_heads: Number of attention heads.
        quant_config: The quantization configuration.
        weight_scale: The weight scale tensor.
        input_scale: The input scale tensor.
        weight_scale_2: Additional weight scale factor (NVFP4 only).
        bias: Optional bias tensor (FP8 only).
        _output_dim: Optional output dimension override for the FP8
            kernel. If not provided, defaults to
            ``n_heads * head_dim``.

    Returns:
        The query projection output tensor.
    """
    match quant_config.format:
        case QuantFormat.MXFP8:
            # No fused block-scaled MXFP8 QKV+KV-write kernel exists yet, so
            # dequantize the (constant) QKV weight to bf16 and use the
            # unquantized fused QKV ragged matmul. The dense MLP still uses the
            # MXFP8 block-scaled tensor-core path. bf16 weights are strictly
            # more accurate than the FP8 source, so accuracy does not regress.
            wqkv_bf16 = _dequantize_mxfp8_weight(
                wqkv, weight_scale.to(x.device)
            )
            return fused_qkv_ragged_matmul(
                kv_params,
                input=x,
                input_row_offsets=input_row_offsets,
                wqkv=wqkv_bf16,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=n_heads,
                bias=bias,
                _output_dim=_output_dim,
            )
        case QuantFormat.NVFP4:
            assert input_scale is not None
            assert weight_scale_2 is not None

            if _is_pre_sm100_nvidia_gpu():
                # No in-tree model reaches this branch on pre-Blackwell:
                # NVFP4 QKV projections route through StackedLinear and
                # _matmul_float4 (already fused). Keep a clear error rather
                # than dead fallback code.
                raise ValueError(
                    "quantized_fused_qkv_matmul has no NVFP4 path on"
                    " pre-Blackwell GPUs; route QKV through StackedLinear"
                    " (_matmul_float4) instead"
                )

            x, x_scales = quantize_dynamic_block_scaled(
                x,
                tensor_sf=1.0 / input_scale,
                scales_type=DType.float8_e4m3fn,
                out_type=DType.uint8,
            )

            weight_scale = weight_scale.to(x.device)
            if quant_config.scales_pre_interleaved:
                weight_scale = _reshape_pre_interleaved_scales(weight_scale)
            else:
                weight_scale = block_scales_interleave(weight_scale)

            return _fused_qkv_ragged_matmul_scaled_float4(
                kv_params,
                input=x,
                input_row_offsets=input_row_offsets,
                wqkv=wqkv,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=n_heads,
                input_scale=x_scales.to(x.device),
                weight_scale=weight_scale,
                tensor_sf=input_scale * weight_scale_2,
            )
        case (
            QuantFormat.COMPRESSED_TENSORS_FP8
            | QuantFormat.FBGEMM_FP8
            | QuantFormat.BLOCKSCALED_FP8
        ):
            # FP8 path (static or dynamic)
            if quant_config.is_static:
                assert input_scale is not None
                x = quantize_static_scaled_float8(
                    x, input_scale.to(DeviceRef.CPU())
                )
                x_scales = input_scale
            elif (
                quant_config.input_scale.is_tensor
                and quant_config.weight_scale.is_tensor
            ):
                # Workaround for GEX-3496: force tensor-scale into
                # row-scale format. Input: per-tensor dynamic quantization.
                # Weight: rowwise [total_dim, 1] from qkv_weight_scale.
                x, x_scales = quantize_tensor_dynamic_scaled_float8(
                    x,
                    quant_config.input_scale,
                    quant_config.weight_scale,
                    out_type=wqkv.dtype,
                    scales_type=weight_scale.dtype,
                )
                # x_scales = x_scales.reshape([1, 1])
                # Don't pass quant_config so the kernel infers
                # per-channel (1,1,-1) from [1,1] + [N,1] shapes.
                return _fused_qkv_ragged_matmul_scaled_float8(
                    kv_params,
                    input=x,
                    wqkv=wqkv,
                    bias=bias,
                    input_row_offsets=input_row_offsets,
                    kv_collection=kv_collection,
                    layer_idx=layer_idx,
                    n_heads=n_heads,
                    input_scale=x_scales.to(x.device),
                    weight_scale=weight_scale.to(x.device),
                    quant_config=None,
                    _output_dim=_output_dim,
                )
            else:
                x, x_scales = quantize_dynamic_scaled_float8(
                    x,
                    quant_config.input_scale,
                    quant_config.weight_scale,
                    scales_type=weight_scale.dtype,
                    out_type=wqkv.dtype,
                )

            return _fused_qkv_ragged_matmul_scaled_float8(
                kv_params,
                input=x,
                wqkv=wqkv,
                bias=bias,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=n_heads,
                input_scale=x_scales.to(x.device),
                weight_scale=weight_scale.to(x.device),
                quant_config=quant_config,
                _output_dim=_output_dim,
            )
        case _:
            raise ValueError(
                f"Unsupported quantization format for fused QKV matmul: {quant_config.format}"
            )


def quantized_grouped_matmul(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    expert_start_indices: TensorValue,
    expert_ids: TensorValue,
    usage_stats: TensorValue,
    quant_config: QuantConfig,
) -> TensorValue:
    """Single entry point for quantized grouped matmuls (MoE).

    Dispatches to the appropriate kernel based on the quantization format.
    Handles weight dequant (MXFP4) or input quantization + transpose (FP8)
    internally.

    Args:
        x: The input tensor in bf16.
        weight: The weight tensor in storage layout
            (MXFP4: ``[E, out, in//2]``, FP8: ``[E, in, out]``).
        weight_scale: The weight scale tensor in storage layout.
        expert_start_indices: Starting index of each expert's token group.
        expert_ids: Expert identifier for each token group.
        usage_stats: Per-expert usage statistics. The MXFP4 path passes it
            straight to ``grouped_matmul_ragged``, currently the FP8 path
            copies it to CPU.
        quant_config: The quantization configuration.

    Returns:
        The grouped matmul output tensor in bf16.
    """
    match quant_config.format:
        case QuantFormat.MXFP4:
            dequanted = mxfp4_dequant(
                weight, weight_scale, out_type=DType.bfloat16
            )
            return grouped_matmul_ragged(
                x,
                dequanted,
                expert_start_indices,
                expert_ids,
                usage_stats,
            )
        case (
            QuantFormat.COMPRESSED_TENSORS_FP8
            | QuantFormat.FBGEMM_FP8
            | QuantFormat.BLOCKSCALED_FP8
        ):
            assert quant_config.input_scale.block_size is not None
            input_block_size = quant_config.input_scale.block_size[1]

            weight_t = weight.transpose(1, 2)
            scale_t = weight_scale.transpose(1, 2)

            x_fp8, x_scales = quantize_dynamic_scaled_float8(
                x,
                quant_config.input_scale,
                quant_config.weight_scale,
                group_size_or_per_token=input_block_size,
                out_type=weight.dtype,
                scales_type=quant_config.weight_scale.dtype,
            )

            return grouped_dynamic_scaled_fp8_matmul(
                x_fp8,
                weight_t,
                x_scales,
                scale_t,
                expert_start_indices,
                expert_ids,
                usage_stats.to(DeviceRef.CPU()),
                quant_config.input_scale,
                quant_config.weight_scale,
            )
        case _:
            raise ValueError(
                f"Unsupported quantization format for grouped matmul: {quant_config.format}"
            )
