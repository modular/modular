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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue

from .kernels import (
    _fused_qkv_ragged_matmul_scaled_float4,
    _fused_qkv_ragged_matmul_scaled_float8,
    block_scales_interleave,
    convert_weights_to_fp8_fnuz_if_needed,
    dynamic_block_scaled_matmul_fp4,
    dynamic_scaled_matmul,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_matmul_ragged,
    matmul_static_scaled_float8,
    mxfp4_dequant,
    quantize_dynamic_block_scaled_fp4,
    quantize_dynamic_scaled_float8,
    quantize_static_scaled_float8,
)
from .kv_cache import KVCacheParams, PagedCacheValues
from .quant_config import QuantConfig, QuantFormat


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


def _matmul_float4(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    input_scale: TensorValue,
    weight_scale_2: TensorValue,
    scales_pre_interleaved: bool = False,
) -> TensorValue:
    """Computes x @ weight.T with modelopt NVFP4 quantization.

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
    x, x_scales = quantize_dynamic_block_scaled_fp4(
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

    res = dynamic_block_scaled_matmul_fp4(
        x,
        weight,
        x_scales,
        weight_scale,
        tensor_sf=weight_scale_2 * input_scale,
        out_type=DType.bfloat16,
    )
    return res


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
        x = quantize_static_scaled_float8(x, input_scale, out_type=weight.dtype)

        return matmul_static_scaled_float8(x, weight, input_scale, weight_scale)
    else:
        x, x_scales = quantize_dynamic_scaled_float8(
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
            x_scales,
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
    scales_pre_interleaved: bool = False,
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
        scales_pre_interleaved: If True, weight_scale is already in 5D
            TCGEN interleaved layout (NVFP4 only).

    Returns:
        The output tensor.
    """
    match quant_config.format:
        case QuantFormat.NVFP4:
            assert input_scale is not None
            assert weight_scale_2 is not None
            return _matmul_float4(
                x,
                weight,
                weight_scale,
                input_scale,
                weight_scale_2,
                scales_pre_interleaved=scales_pre_interleaved,
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
    scales_pre_interleaved: bool = False,
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
        scales_pre_interleaved: If True, weight_scale is already in 5D
            TCGEN interleaved layout (NVFP4 only).

    Returns:
        The query projection output tensor.
    """
    match quant_config.format:
        case QuantFormat.NVFP4:
            assert input_scale is not None
            assert weight_scale_2 is not None

            x, x_scales = quantize_dynamic_block_scaled_fp4(
                x,
                tensor_sf=1.0 / input_scale,
                scales_type=DType.float8_e4m3fn,
                out_type=DType.uint8,
            )

            weight_scale = weight_scale.to(x.device)
            if scales_pre_interleaved:
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
        usage_stats: Per-expert usage statistics (will be moved to CPU).
        quant_config: The quantization configuration.

    Returns:
        The grouped matmul output tensor in bf16.
    """
    cpu_usage_stats = usage_stats.to(DeviceRef.CPU())

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
                cpu_usage_stats,
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
                cpu_usage_stats,
                quant_config.input_scale,
                quant_config.weight_scale,
            )
        case _:
            raise ValueError(
                f"Unsupported quantization format for grouped matmul: {quant_config.format}"
            )
