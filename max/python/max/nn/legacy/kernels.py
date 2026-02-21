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
"""Helper functions for wrapping custom kv cache/attention related ops."""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Any

import numpy as np
from max.driver import accelerator_architecture_name
from max.dtype import DType
from max.graph import (
    BufferValue,
    BufferValueLike,
    DeviceRef,
    Dim,
    StaticDim,
    TensorType,
    TensorValue,
    TensorValueLike,
    Value,
    ops,
)
from max.graph.ops import assert_same_device
from max.graph.ops.quantized import repack_gguf_quantized_weights
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.nn.legacy.float8_config import (
    Float8Config,
    Float8InputScaleSpec,
    Float8WeightScaleSpec,
)

from .attention.mask_config import (
    AttentionMaskVariant,
    MHAMaskConfig,
    MHAMaskVariant,
    PositionalEncodingVariant,
)
from .kv_cache import KVCacheParams, PagedCacheValues, kernel_substring

_MHA_MASK_CONFIG_DICT = {
    MHAMaskVariant.CAUSAL_MASK: MHAMaskConfig(
        attention_mask_variant=AttentionMaskVariant.CAUSAL_MASK,
        positional_encoding_variant=PositionalEncodingVariant.NO_POS,
    ),
    MHAMaskVariant.CAUSAL_ALIBI_MASK: MHAMaskConfig(
        attention_mask_variant=AttentionMaskVariant.CAUSAL_MASK,
        positional_encoding_variant=PositionalEncodingVariant.ALIBI_POS,
    ),
    MHAMaskVariant.NULL_MASK: MHAMaskConfig(
        attention_mask_variant=AttentionMaskVariant.NULL_MASK,
        positional_encoding_variant=PositionalEncodingVariant.NO_POS,
    ),
    MHAMaskVariant.CHUNKED_CAUSAL_MASK: MHAMaskConfig(
        attention_mask_variant=AttentionMaskVariant.CHUNKED_CAUSAL_MASK,
        positional_encoding_variant=PositionalEncodingVariant.NO_POS,
    ),
    MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK: MHAMaskConfig(
        attention_mask_variant=AttentionMaskVariant.SLIDING_WINDOW_CAUSAL_MASK,
        positional_encoding_variant=PositionalEncodingVariant.NO_POS,
    ),
}

KEY_CACHE_INDEX = 0
VALUE_CACHE_INDEX = 1


def ceildiv(n: Dim, d: Dim) -> Dim:
    """Ceiling division.

    Args:
        n: The numerator.
        d: The denominator.

    Returns:
        The ceiling of dividing n by d.
    """
    return (n + d - 1) // d


def fused_qkv_padded_matmul(
    kv_params: KVCacheParams,
    input: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    valid_lengths: TensorValue,
    n_heads: int,
) -> TensorValue:
    """Computes fused query, key, and value projections with padded input.

    This is for non-ragged (padded batch) inputs where sequences may have
    different actual lengths but are padded to a uniform shape.

    Args:
        kv_params: KV cache parameters.
        input: Input tensor with shape [batch_size, seq_len, hidden_dim].
        wqkv: Weight tensor for Q, K, V projections.
        kv_collection: Paged KV cache collection.
        layer_idx: Layer index for cache lookup (must be uint32).
        valid_lengths: Buffer of shape [batch] containing the valid length for each
            sequence (must be uint32). K and V are only written to cache for
            positions within these lengths.
        n_heads: Number of attention heads.

    Returns:
        Query projections tensor. K and V projections are written to cache.

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    if input.dtype != wqkv.dtype:
        raise ValueError(
            "expected input and wqkv to have the same dtype, but got"
            f" {input.dtype} and {wqkv.dtype}, respectively."
        )

    input_rank_expected = 3
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input to have rank {input_rank_expected}, was {input.rank}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(
            f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        )

    if valid_lengths.dtype != DType.uint32:
        raise ValueError(
            f"expected valid_lengths to have dtype uint32, was {valid_lengths.dtype}"
        )

    if valid_lengths.rank != 1:
        raise ValueError(
            f"expected valid_lengths to have rank 1 [batch], was rank {valid_lengths.rank}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for fused_qkv_padded_matmul: {kv_params.cache_strategy}"
        )

    cache_strategy_str = kernel_substring(kv_params.cache_strategy)
    op_name = f"mo.fused_qkv_matmul.padded.{cache_strategy_str}"

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=[input, wqkv, *kv_collection, layer_idx, valid_lengths],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
                device=input.device,
            )
        ],
    )[0].tensor


def fused_qkv_ragged_matmul(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    bias: TensorValue | None = None,
    _output_dim: int | None = None,
) -> TensorValue:
    """Computes fused query, key, and value projections with ragged input.

    Args:
        kv_params: KVCacheParams object containing key-value cache parameters.
        input: TensorValue representing the input tensor with shape
            [total_seq_len, hidden_dim].
        input_row_offsets: TensorValue indicating the start and end of each
            request in the input tensor with shape [batch_size + 1].
        wqkv: The concatenated Q, K and V projection weights.
        kv_collection: PagedCacheValues object for managing key-value cache.
        layer_idx: TensorValue representing the layer index, expected to have
            dtype uint32.
        n_heads: Number of Query attention heads.
        bias: Optional bias vector concatenated as [q, k, v].
        _output_dim: Optional output dimension. If not provided, the output
            dimension will be [n_heads * head_dim].

    Returns:
        Query projection tensor.
    """
    if input.dtype != wqkv.dtype:
        raise ValueError(
            "expected input and wqkv to have the same dtype, but got"
            f" {input.dtype} and {wqkv.dtype}, respectively."
        )

    input_rank_expected = 2
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input to have rank {input_rank_expected}, was {input.rank}"
        )

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(
            f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for fused_qkv_ragged_matmul: {kv_params.cache_strategy}"
        )

    cache_strategy_str = kernel_substring(kv_params.cache_strategy)
    op_name = f"mo.fused_qkv_matmul.ragged.{cache_strategy_str}"
    values = [input, input_row_offsets, wqkv, *kv_collection, layer_idx]

    if bias is not None:
        op_name += ".bias"
        values.append(bias)

    output_dim = (
        _output_dim if _output_dim is not None else n_heads * kv_params.head_dim
    )

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [output_dim],
                device=input.device,
            )
        ],
    )[0].tensor


def fused_qkv_ragged_matmul_scaled_float8(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    input_scale: TensorValue,
    weight_scale: TensorValue,
    bias: TensorValue | None = None,
    float8_config: Float8Config | None = None,
    _output_dim: int | None = None,
) -> TensorValue:
    """Computes fused query, key, and value projections with scaled float8 input and weights.

    Args:
        kv_params: KVCacheParams object containing key-value cache parameters.
        input: TensorValue representing the input tensor with shape
            [M=total_seq_len, K=hidden_dim].
        input_row_offsets: TensorValue indicating the start and end of each
            batch in the input tensor with shape [batch_size + 1].
        wqkv: TensorValue representing the weight tensor with shape
            [N=(num_heads + 2 * num_kv_heads) * head_dim, K=hidden_dim].
        kv_collection: PagedCacheValues object for managing key-value cache.
        layer_idx: TensorValue representing the layer index, expected to have
            dtype uint32.
        n_heads: Number of attention heads.
        input_scale: TensorValue representing the input scale tensor. Shape
            varies depending on the quantization config.
        weight_scale: TensorValue representing the weight scale tensor. Shape
            varies depending on the quantization config.
        bias: Optional bias vector concatenated as [q, k, v].
        float8_config: Optional Float8Config object containing float8
            quantization parameters. If not provided, the quantization config
            will be inferred from the input and weight scale shapes.
        _output_dim: Optional output dimension. If not provided, the output
            dimension will be [n_heads * head_dim].

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    if input.dtype != wqkv.dtype:
        raise ValueError(
            "expected input and wqkv to have the same dtype, but got"
            f" {input.dtype} and {wqkv.dtype}, respectively."
        )

    input_rank_expected = 2
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input to have rank {input_rank_expected}, was {input.rank}"
        )

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(
            f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        )

    # Device check - all tensors must be on the same device
    tensors_to_check = [wqkv, input_row_offsets, input_scale, weight_scale]
    if bias is not None:
        tensors_to_check.append(bias)

    if not all(t.device == input.device for t in tensors_to_check):
        raise ValueError(
            f"expected all tensors to be on the same device as input ({input.device}), "
            f"but got:\n"
            f"  wqkv={wqkv.device}\n"
            f"  input_row_offsets={input_row_offsets.device}\n"
            f"  input_scale={input_scale.device}\n"
            f"  weight_scale={weight_scale.device}"
            + ("" if bias is None else f"\n  bias={bias.device}")
        )

    # layer_idx must be a scalar on CPU as it's used for indexing
    if layer_idx.device != DeviceRef.CPU():
        raise ValueError(
            f"expected layer_idx to be on CPU device, but got {layer_idx.device}"
        )

    # for per-tensor quantization, the scale is a scalar. We view it as a 1x1
    # rank-2 tensor so that we can use the same kernel for per-tensor and
    # per-channel quantization.
    if input_scale.shape in [[], [1]]:
        input_scale = input_scale.reshape([1, 1])

    if weight_scale.shape in [[], [1]]:
        weight_scale = weight_scale.reshape([1, 1])

    # Try to infer the quantization config
    if float8_config is not None:
        scales_granularity_mnk = float8_config.scales_granularity_mnk
    else:
        # with out float8_config, we either use per-tensor or per-channel quantization
        if (
            input_scale.shape[0] == 1
            and input_scale.shape[1] == 1
            and weight_scale.shape[0] == 1
            and weight_scale.shape[1] == 1
        ):
            scales_granularity_mnk = (-1, -1, -1)  # per-tensor quantization
        elif input_scale.shape[0] == 1 and weight_scale.shape[1] == 1:
            scales_granularity_mnk = (1, 1, -1)  # per-channel quantization
        else:
            raise ValueError(
                "Can not infer the quantization config from the input tensor shapes",
                "Please provide a float8_config",
            )

    assert kv_params.page_size is not None
    parameters: dict[str, int | str | DType] = {
        "kv_type": kv_params.dtype,
        "m_scale_granularity": scales_granularity_mnk[0],
        "n_scale_granularity": scales_granularity_mnk[1],
        "k_scale_granularity": scales_granularity_mnk[2],
    }

    op_name = "mo.fused_qkv_matmul.ragged.paged.scale"
    values = [
        input,
        input_row_offsets,
        wqkv,
        input_scale,
        weight_scale,
        *kv_collection,
        layer_idx,
    ]
    if bias is not None:
        op_name += ".bias"
        values.append(bias)

    output_dim = (
        _output_dim if _output_dim is not None else n_heads * kv_params.head_dim
    )

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=DType.bfloat16,
                shape=input.shape[:-1] + [output_dim],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def fused_qkv_ragged_matmul_scaled_float4(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    input_scale: TensorValue,
    weight_scale: TensorValue,
    tensor_sf: float | TensorValue,
    kv_scales: TensorValue | None = None,
    sf_vector_size: int = 16,
    _output_dim: int | None = None,
) -> TensorValue:
    """Computes fused query, key, and value projections with scaled float4 input and weights.

    Args:
        kv_params: KVCacheParams object containing key-value cache parameters.
        input: TensorValue representing the input tensor with shape
            [M=total_seq_len, K=hidden_dim].
        input_row_offsets: TensorValue indicating the start and end of each
            batch in the input tensor with shape [batch_size + 1].
        wqkv: TensorValue representing the weight tensor with shape
            [N=(num_heads + 2 * num_kv_heads) * head_dim, K=hidden_dim].
        kv_collection: PagedCacheValues object for managing key-value cache.
        layer_idx: TensorValue representing the layer index, expected to have
            dtype uint32.
        n_heads: Number of attention heads.
        input_scale: TensorValue representing the input scale tensor. Shape
            for blockwise scaling is 5D e.g., [2, 3, 32, 4, 4].
        weight_scale: TensorValue representing the weight scale tensor. Shape
            for blockwise scaling is 5D e.g., [2, 34, 32, 4, 4]
        tensor_sf: Buffer-wise scaling factor equal to weight_scale_2 * input_scale (pre-quantization, non-inverted).
        kv_scales: TBD, used in NVFP4 KV cache, see: https://github.com/NVIDIA/TensorRT-LLM/blob/0ffa77af51b272ba27424564ed253096d6f0f11a/tensorrt_llm/_torch/modules/linear.py#L690
        _output_dim: Optional output dimension. If not provided, the output
            dimension will be [n_heads * head_dim].

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    if input.dtype != wqkv.dtype:
        raise ValueError(
            "expected input and wqkv to have the same dtype, but got"
            f" {input.dtype} and {wqkv.dtype}, respectively."
        )

    input_rank_expected = 2
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input to have rank {input_rank_expected}, was {input.rank}"
        )

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(
            f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        )

    # Device check - all tensors must be on the same device
    tensors_to_check = [wqkv, input_row_offsets, input_scale, weight_scale]

    if not all(t.device == input.device for t in tensors_to_check):
        raise ValueError(
            f"expected all tensors to be on the same device as input ({input.device}), "
            f"but got:\n"
            f"  wqkv={wqkv.device}\n"
            f"  input_row_offsets={input_row_offsets.device}\n"
            f"  input_scale={input_scale.device}\n"
            f"  weight_scale={weight_scale.device}"
        )

    # layer_idx must be a scalar on CPU as it's used for indexing
    if layer_idx.device != DeviceRef.CPU():
        raise ValueError(
            f"expected layer_idx to be on CPU device, but got {layer_idx.device}"
        )

    # tensor_sf must be a scalar on CPU as it's used for per-tensor scaling
    if isinstance(tensor_sf, float):
        tensor_sf = ops.constant(
            tensor_sf, DType.float32, device=DeviceRef.CPU()
        )
    elif isinstance(tensor_sf, TensorValue):
        tensor_sf = (
            tensor_sf.cast(DType.float32).to(DeviceRef.CPU()).reshape(())
        )
    else:
        raise ValueError(
            "tensor_sf must be either float or a float32 CPU tensor of rank 0."
        )

    assert kv_params.page_size is not None
    parameters: dict[str, int | str | DType] = {
        "dtype": DType.uint8,
        "scale_type": DType.float8_e4m3fn,
        "kv_type": kv_params.dtype,
        "SF_VECTOR_SIZE": sf_vector_size,
    }

    op_name = "mo.fused_qkv_matmul.ragged.paged.scale.float4"
    values = [
        input,
        input_row_offsets,
        wqkv,
        input_scale,
        weight_scale,
        tensor_sf,
        *kv_collection,
        layer_idx,
    ]

    output_dim = (
        _output_dim if _output_dim is not None else n_heads * kv_params.head_dim
    )

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=DType.bfloat16,
                shape=input.shape[:-1] + [output_dim],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def unfused_qkv_ragged_matmul_gguf_quantized(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    n_heads: int,
    q_weight: TensorValue,
    k_weight: TensorValue,
    v_weight: TensorValue,
    quantization_encoding_q: QuantizationEncoding,
    quantization_encoding_k: QuantizationEncoding,
    quantization_encoding_v: QuantizationEncoding,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
) -> TensorValue:
    """Computes fused query, key, and value projections with ragged input and
    quantized weight matrices. A `quantization_config` must be provided.

    `input` and `input_row_offsets` are used together to implement the ragged
    tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """

    input_rank_expected = 2
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input to have rank {input_rank_expected}, was {input.rank}"
        )

    if input.dtype != DType.float32:
        raise ValueError(
            f"expected input to have dtype float32, was {input.dtype}"
        )

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(
            f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        )

    if kv_params.cache_strategy not in {"paged"}:
        raise ValueError(
            f"unsupported cache strategy for fused_qkv_ragged_matmul: {kv_params.cache_strategy}"
        )

    if (
        not quantization_encoding_q.is_gguf
        or not quantization_encoding_k.is_gguf
        or not quantization_encoding_v.is_gguf
    ):
        raise ValueError(
            f"expected quantization_encoding_q, quantization_encoding_k, and quantization_encoding_v to be gguf, was {quantization_encoding_q}, {quantization_encoding_k}, and {quantization_encoding_v}"
        )

    assert kv_params.page_size is not None
    parameters: dict[str, int | str | DType] = {
        "quantization_encoding_q": quantization_encoding_q.name,
        "quantization_encoding_k": quantization_encoding_k.name,
        "quantization_encoding_v": quantization_encoding_v.name,
    }

    cache_strategy_str = kernel_substring(kv_params.cache_strategy)
    return ops.inplace_custom(
        name=f"mo.unfused_qkv_matmul.ragged.{cache_strategy_str}.gguf_quantized",
        device=input.device,
        values=[
            input,
            input_row_offsets,
            repack_gguf_quantized_weights(q_weight, quantization_encoding_q),
            repack_gguf_quantized_weights(k_weight, quantization_encoding_k),
            repack_gguf_quantized_weights(v_weight, quantization_encoding_v),
            *kv_collection,
            layer_idx,
        ],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def fused_qkv_ragged_matmul_quantized(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    quantization_config: QuantizationConfig,
    perm_idx: TensorValue | None = None,
    bias: TensorValue | None = None,
) -> TensorValue:
    """Computes fused query, key, and value projections with ragged input and
    quantized weight matrices. A `quantization_config` must be provided.

    `input` and `input_row_offsets` are used together to implement the ragged
    tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """

    input_rank_expected = 2
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input to have rank {input_rank_expected}, was {input.rank}"
        )

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(
            f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for fused_qkv_ragged_matmul: {kv_params.cache_strategy}"
        )

    # In the group-wise quantization scheme, every `group_size` quantized weights
    # share the same scale. If `has_zp` is `True`, there is also a group-wise zero
    # point that need to be subtracted from the quantized weights.
    # Since the new extensibility API doesn't currently support `bool` type parameters,
    # we pass `has_zp` as an integer (`has_zp_int`).
    # For GPTQ, `has_zp_int` will always be 0.
    parameters: dict[str, int | str | DType] = {
        "group_size": quantization_config.group_size,
        "has_zp_int": 0,
    }
    if perm_idx:
        input = ops.gather(input, TensorValue(perm_idx), axis=1)
        perm_idx = perm_idx.to(input.type.device or DeviceRef.CPU())
        wqkv = ops.custom(
            "GPTQ_gpu_repack_b4_g128_desc_act",
            wqkv.device,
            list((wqkv, perm_idx)),
            out_types=[
                TensorType(
                    DType.uint8,
                    ((wqkv.shape[1], wqkv.shape[0])),
                    device=input.type.device or DeviceRef.CPU(),
                )
            ],
        )[0].tensor
    else:
        wqkv = ops.custom(
            "GPTQ_gpu_repack_b4_g128",
            wqkv.device,
            list((wqkv,)),
            out_types=[
                TensorType(
                    DType.uint8,
                    ((wqkv.shape[1], wqkv.shape[0])),
                    device=input.type.device or DeviceRef.CPU(),
                )
            ],
        )[0].tensor

    cache_strategy_str = kernel_substring(kv_params.cache_strategy)

    args = [input, input_row_offsets, wqkv, *kv_collection, layer_idx]
    if bias is not None:
        args.append(bias)
        bias_name_str = "bias."
    else:
        bias_name_str = ""

    op_name = f"mo.fused_qkv_matmul.ragged.{cache_strategy_str}.{bias_name_str}quantized"

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=args,
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def matmul_kv_cache_ragged(
    kv_params: KVCacheParams,
    hidden_states: TensorValue,
    input_row_offsets: TensorValue,
    weight: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
) -> None:
    """Computes key and value projections with ragged input.

    `hidden_states` and `input_row_offsets` are used together to
    implement the ragged tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`
    """
    if hidden_states.dtype != weight.dtype:
        raise ValueError(
            "expected hidden_states and weight to have the same dtype, but got"
            f" {hidden_states.dtype} and {weight.dtype}, respectively."
        )

    hidden_states_rank_expected = 2
    if hidden_states.rank != hidden_states_rank_expected:
        raise ValueError(
            "expected hidden_states to have rank "
            f"{hidden_states_rank_expected}, was {hidden_states.rank}"
        )

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for matmul_kv_cache_ragged: {kv_params.cache_strategy}"
        )

    cache_strategy_str = kernel_substring(kv_params.cache_strategy)
    op_name = f"mo.kv_matmul.ragged.{cache_strategy_str}"

    ops.inplace_custom(
        name=op_name,
        device=hidden_states.device,
        values=[
            hidden_states,
            input_row_offsets,
            weight,
            *kv_collection,
            layer_idx,
        ],
    )


def matmul_k_cache_ragged(
    kv_params: KVCacheParams,
    hidden_states: TensorValue,
    input_row_offsets: TensorValue,
    weight: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
) -> None:
    """Computes key projections with ragged input.

    `hidden_states` and `input_row_offsets` are used together to
    implement the ragged tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`
    """
    if hidden_states.dtype != weight.dtype:
        raise ValueError(
            "expected hidden_states and weight to have the same dtype, but got"
            f" {hidden_states.dtype} and {weight.dtype}, respectively."
        )

    hidden_states_rank_expected = 2
    if hidden_states.rank != hidden_states_rank_expected:
        raise ValueError(
            "expected hidden_states to have rank "
            f"{hidden_states_rank_expected}, was {hidden_states.rank}"
        )

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for matmul_kv_cache_ragged: {kv_params.cache_strategy}"
        )

    cache_strategy_str = kernel_substring(kv_params.cache_strategy)
    op_name = f"mo.k_matmul.ragged.{cache_strategy_str}"

    ops.inplace_custom(
        name=op_name,
        device=hidden_states.device,
        values=[
            hidden_states,
            input_row_offsets,
            weight,
            *kv_collection,
            layer_idx,
        ],
    )


def matmul_k_cache_ragged_scaled_float8(
    kv_params: KVCacheParams,
    hidden_states: TensorValue,
    input_row_offsets: TensorValue,
    weight: TensorValue,
    input_scale: TensorValue,
    weight_scale: TensorValue,
    kv_collection: PagedCacheValues,
    scales_granularity_mnk: tuple[int, int, int],
    layer_idx: TensorValue,
) -> None:
    """Computes key projections with ragged input with FP8 block scaling.

    Args:
        kv_params: KVCacheParams object containing key-value cache parameters.
        hidden_states: TensorValue representing the input tensor with shape
            [M=total_seq_len, K=hidden_dim].
        input_row_offsets: TensorValue indicating the start and end of each
            batch in the input tensor with shape [batch_size + 1].
        weight: TensorValue representing the weight tensor with shape
            [N=num_heads, K=hidden_dim].
        input_scale: TensorValue representing the input scale tensor with shape
            [ceildiv(K / BLOCK_SIZE_K), ceildiv(M / BLOCK_SIZE_M)].
        weight_scale: TensorValue representing the weight scale tensor with
            shape [ceildiv(N / BLOCK_SIZE_N), ceildiv(K / BLOCK_SIZE_K)].
        kv_collection: PagedCacheValues object for managing key-value cache.
        scales_granularity_mnk: tuple[int, int, int] representing the
            scaling (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K).
        layer_idx: TensorValue representing the layer index, expected to have
            dtype uint32.

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel,
            or when the cache strategy is not supported.
    """
    if hidden_states.dtype != weight.dtype:
        raise ValueError(
            "expected hidden_states and weight to have the same dtype, but got"
            f" {hidden_states.dtype} and {weight.dtype}, respectively."
        )

    hidden_states_rank_expected = 2
    if hidden_states.rank != hidden_states_rank_expected:
        raise ValueError(
            "expected hidden_states to have rank "
            f"{hidden_states_rank_expected}, was {hidden_states.rank}"
        )

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(
            f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for matmul_kv_cache_ragged: {kv_params.cache_strategy}"
        )

    cache_strategy_str = kernel_substring(kv_params.cache_strategy)
    op_name = f"mo.k_matmul.ragged.{cache_strategy_str}.scale"

    parameters: dict[str, bool | int | str | DType] = {
        "m_scale_granularity": scales_granularity_mnk[0],
        "n_scale_granularity": scales_granularity_mnk[1],
        "k_scale_granularity": scales_granularity_mnk[2],
    }

    ops.inplace_custom(
        name=op_name,
        device=hidden_states.device,
        values=[
            hidden_states,
            input_row_offsets,
            weight,
            input_scale,
            weight_scale,
            *kv_collection,
            layer_idx,
        ],
        parameters=parameters,
    )


def fused_qk_ragged_rope(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    freqs_cis: TensorValue,
    layer_idx: TensorValue,
    interleaved: bool = True,
    position_ids: TensorValue | None = None,
    mrope_section: list[int] | None = None,
) -> TensorValue:
    """Computes fused query-key attention with rotary positional encodings and ragged inputs.

    Args:
        kv_params: KV cache parameters
        input: [batch_size * seq_len, n_heads, head_dim]
        input_row_offsets: Ragged tensor offsets indicating where each batch starts and ends
        kv_collection: KV cache collection
        freqs_cis: tensor of shape (max_seq_len * 2, head_dim)
        layer_idx: Layer index for KV cache
        interleaved: Whether to use interleaved RoPE pattern
        position_ids: Optional ragged 2D array of position IDs. If None, defaults to
                     cache_length + token_idx for each token. When `num_sections` > 1,
                     `mrope_section` must be provided to indicate each section of the head_dim
                     to apply RoPE to. Shape: [num_sections, total_seq_len]
        mrope_section: Optional list of integers indicating the section of the head_dim to
        apply RoPE to. Must be used in conjunction with `position_ids`.

    `input` and `input_row_offsets` are used together to implement the ragged tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`. If `input`
    is not of the same dtype as `freqs_cis`, it will be cast to the dtype of `freqs_cis`
    for the computation, and cast back to the original dtype after the computation is
    finished.

    When `position_ids` and `mrope_section` are provided, it replaces the default position
    calculation (cache_length + token_idx) with explicit position values. This is useful for
    3D RoPE in models like Qwen2.5-VL that need custom position encoding.
    """

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(
            f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for fused_qk_ragged_rope: {kv_params.cache_strategy}"
        )

    parameters: dict[str, bool | int | str | DType] = {
        "interleaved": interleaved,
        "cache_dtype": kv_params.dtype,
    }

    if position_ids is not None:
        if position_ids.dtype != DType.uint32:
            raise ValueError(
                f"expected position_ids to have dtype uint32, was {position_ids.dtype}"
            )
        if position_ids.rank != 2:
            raise ValueError(
                f"expected position_ids to be 2D, got rank {position_ids.rank}"
            )
        if mrope_section is not None:
            if len(mrope_section) != position_ids.shape[0]:
                raise ValueError(
                    f"expected mrope_section to have length {position_ids.shape[0]}, "
                    f"was {len(mrope_section)}"
                )
            # multiplied by 2 because the kernel expects the section to be in terms of head_dim,
            # then calculate the prefix sum of the section
            mrope_section = [x * 2 for x in mrope_section]
            mrope_section = [
                sum(mrope_section[: i + 1]) for i in range(len(mrope_section))
            ]
            # convert mrope_section to a string, with each element separated by "_"
            parameters["mrope_section"] = "_".join(
                str(x) for x in mrope_section
            )
        else:
            parameters["mrope_section"] = ""

    cache_strategy_str = kernel_substring(kv_params.cache_strategy)

    if position_ids is not None:
        op_name = (
            f"mo.fused_qk_rope.ragged.{cache_strategy_str}.with_position_id"
        )
        values = [
            input,
            input_row_offsets,
            *kv_collection,
            freqs_cis,
            position_ids,
            layer_idx,
        ]
    else:
        op_name = f"mo.fused_qk_rope.ragged.{cache_strategy_str}"
        values = [
            input,
            input_row_offsets,
            *kv_collection,
            freqs_cis,
            layer_idx,
        ]

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
    )[0].tensor


def fused_qk_padded_rope(
    kv_params: KVCacheParams,
    input: TensorValue,
    kv_collection: PagedCacheValues,
    freqs_cis: TensorValue,
    layer_idx: TensorValue,
    valid_lengths: TensorValue,
    interleaved: bool = True,
) -> TensorValue:
    """Computes fused query-key RoPE with padded inputs and paged KV cache.

    This function applies Rotary Positional Embeddings (RoPE) to both Q and K tensors,
    where K is stored in the paged KV cache. This is the padded equivalent of
    fused_qk_ragged_rope.

    Args:
        kv_params: KV cache parameters.
        input: Query tensor of shape [batch, seq_len, n_heads, head_dim].
        kv_collection: Paged KV cache collection.
        freqs_cis: Frequency tensor of shape (max_seq_len * 2, head_dim).
        layer_idx: Layer index for KV cache (must be uint32 on CPU).
        valid_lengths: Buffer of shape [batch] containing the valid length for each
            sequence (must be uint32). RoPE is only applied to positions within
            these lengths.
        interleaved: Whether to use interleaved RoPE pattern.

    Returns:
        Query tensor with RoPE applied, same shape as input.

    Note:
        Unlike fused_qk_ragged_rope which requires ragged inputs, this function
        works with padded batch inputs where sequences may have different actual
        lengths but are padded to a uniform shape.
    """
    if layer_idx.dtype != DType.uint32:
        raise ValueError(
            f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        )

    if valid_lengths.dtype != DType.uint32:
        raise ValueError(
            f"expected valid_lengths to have dtype uint32, was {valid_lengths.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for fused_qk_padded_rope: {kv_params.cache_strategy}"
        )

    if input.rank != 4:
        raise ValueError(
            f"expected input to have rank 4 [batch, seq_len, n_heads, head_dim], was rank {input.rank}"
        )

    if valid_lengths.rank != 1:
        raise ValueError(
            f"expected valid_lengths to have rank 1 [batch], was rank {valid_lengths.rank}"
        )

    parameters: dict[str, bool | int | str | DType] = {
        "interleaved": interleaved,
    }

    # Use custom op that calls the Mojo fused_qk_rope kernel with paged cache
    op_name = "mo.fused_qk_rope.padded.paged"

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=[
            input,
            *kv_collection,
            freqs_cis,
            layer_idx,
            valid_lengths,
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
    )[0].tensor


def _validate_kv_cache_store_common(
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    key_or_value: int,
) -> None:
    if layer_idx.dtype != DType.uint32:
        raise ValueError(
            f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        )
    if layer_idx.rank != 0:
        raise ValueError(
            f"expected layer_idx to be a scalar (rank 0), was rank {layer_idx.rank}"
        )
    if kv_collection.kv_blocks.rank != 6:
        raise ValueError(
            f"expected kv_blocks to have rank 6, was {kv_collection.kv_blocks.rank}"
        )
    if kv_collection.cache_lengths.rank != 1:
        raise ValueError(
            f"expected cache_lengths to have rank 1, was {kv_collection.cache_lengths.rank}"
        )
    if kv_collection.lookup_table.rank != 2:
        raise ValueError(
            f"expected lookup_table to have rank 2, was {kv_collection.lookup_table.rank}"
        )
    if kv_collection.max_lengths.rank != 2:
        raise ValueError(
            f"expected max_lengths to have rank 2, was {kv_collection.max_lengths.rank}"
        )
    if key_or_value not in (KEY_CACHE_INDEX, VALUE_CACHE_INDEX):
        raise ValueError(
            "expected key_or_value to be KEY_CACHE_INDEX or VALUE_CACHE_INDEX, "
            f"was {key_or_value}"
        )


def kv_cache_store_paged_ragged(
    kv_collection: PagedCacheValues,
    x_cache: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
    *,
    key_or_value: int,
) -> None:
    """Stores key or value tensor into the paged KV cache (ragged inputs)."""
    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )
    if x_cache.rank != 3:
        raise ValueError(f"expected x_cache to have rank 3, was {x_cache.rank}")
    if input_row_offsets.rank != 1:
        raise ValueError(
            f"expected input_row_offsets to have rank 1, was {input_row_offsets.rank}"
        )
    _validate_kv_cache_store_common(kv_collection, layer_idx, key_or_value)

    parameters: dict[str, int | str | DType] = {
        "key_or_value": key_or_value,
    }

    ops.inplace_custom(
        "mo.kv_cache.store.paged.ragged",
        device=x_cache.device,
        values=[
            x_cache,
            kv_collection.kv_blocks,
            kv_collection.cache_lengths,
            kv_collection.lookup_table,
            input_row_offsets,
            kv_collection.max_lengths,
            layer_idx,
        ],
        parameters=parameters,
    )


def store_k_cache_ragged(
    kv_collection: PagedCacheValues,
    x_k: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
) -> None:
    kv_cache_store_paged_ragged(
        kv_collection,
        x_k,
        input_row_offsets,
        layer_idx,
        key_or_value=KEY_CACHE_INDEX,
    )


def store_v_cache_ragged(
    kv_collection: PagedCacheValues,
    x_v: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
) -> None:
    kv_cache_store_paged_ragged(
        kv_collection,
        x_v,
        input_row_offsets,
        layer_idx,
        key_or_value=VALUE_CACHE_INDEX,
    )


def kv_cache_store_paged_padded(
    kv_collection: PagedCacheValues,
    x_cache: TensorValue,
    valid_lengths: TensorValue,
    layer_idx: TensorValue,
    *,
    key_or_value: int,
) -> None:
    """Stores key or value tensor into the paged KV cache (padded inputs)."""
    if valid_lengths.dtype != DType.uint32:
        raise ValueError(
            f"expected valid_lengths to have dtype uint32, was {valid_lengths.dtype}"
        )
    if x_cache.rank != 4:
        raise ValueError(f"expected x_cache to have rank 4, was {x_cache.rank}")
    if valid_lengths.rank != 1:
        raise ValueError(
            f"expected valid_lengths to have rank 1, was {valid_lengths.rank}"
        )
    _validate_kv_cache_store_common(kv_collection, layer_idx, key_or_value)

    parameters: dict[str, int | str | DType] = {
        "key_or_value": key_or_value,
    }

    ops.inplace_custom(
        "mo.kv_cache.store.paged.padded",
        device=x_cache.device,
        values=[
            x_cache,
            kv_collection.kv_blocks,
            kv_collection.cache_lengths,
            kv_collection.lookup_table,
            valid_lengths,
            kv_collection.max_lengths,
            layer_idx,
        ],
        parameters=parameters,
    )


def store_k_cache_padded(
    kv_collection: PagedCacheValues,
    x_k: TensorValue,
    valid_lengths: TensorValue,
    layer_idx: TensorValue,
) -> None:
    kv_cache_store_paged_padded(
        kv_collection,
        x_k,
        valid_lengths,
        layer_idx,
        key_or_value=KEY_CACHE_INDEX,
    )


def store_v_cache_padded(
    kv_collection: PagedCacheValues,
    x_v: TensorValue,
    valid_lengths: TensorValue,
    layer_idx: TensorValue,
) -> None:
    kv_cache_store_paged_padded(
        kv_collection,
        x_v,
        valid_lengths,
        layer_idx,
        key_or_value=VALUE_CACHE_INDEX,
    )


def rope_ragged(
    input: TensorValue,
    input_row_offsets: TensorValue,
    start_pos: TensorValue,
    freqs_cis: TensorValue,
    *,
    interleaved: bool = True,
) -> TensorValue:
    """Apply RoPE to ragged input using the standard rope kernel."""
    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )
    if start_pos.dtype != DType.uint32:
        raise ValueError(
            f"expected start_pos to have dtype uint32, was {start_pos.dtype}"
        )
    if input.rank != 3:
        raise ValueError(f"expected input to have rank 3, was {input.rank}")
    if input_row_offsets.rank != 1:
        raise ValueError(
            f"expected input_row_offsets to have rank 1, was {input_row_offsets.rank}"
        )
    if start_pos.rank != 1:
        raise ValueError(
            f"expected start_pos to have rank 1, was {start_pos.rank}"
        )
    if freqs_cis.rank != 2:
        raise ValueError(
            f"expected freqs_cis to have rank 2, was {freqs_cis.rank}"
        )

    parameters: dict[str, bool | int | str | DType] = {
        "interleaved": interleaved,
    }

    return ops.custom(
        "mo.rope.ragged",
        device=input.device,
        values=[
            input,
            input_row_offsets,
            start_pos,
            freqs_cis,
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
    )[0].tensor


def _apply_rope_with_freqs_cis(
    input: TensorValue,
    freqs_cis: TensorValue,
    *,
    interleaved: bool = True,
) -> TensorValue:
    """Apply RoPE using per-token freqs_cis (no KV cache coupling)."""
    if freqs_cis.rank == 2:
        head_dim = input.shape[-1]
        freqs_cis = freqs_cis.reshape((freqs_cis.shape[0], head_dim // 2, 2))
    freqs_cis = ops.cast(freqs_cis, input.dtype)
    freqs_cis = ops.unsqueeze(freqs_cis, 1)  # [T, 1, D/2, 2]

    if interleaved:
        x_complex = ops.as_interleaved_complex(input)
        x_re = x_complex[..., 0]
        x_im = x_complex[..., 1]
    else:
        half_dim = input.shape[-1] // 2
        x_re = input[..., :half_dim]
        x_im = input[..., half_dim:]

    freqs_re = freqs_cis[..., 0]
    freqs_im = freqs_cis[..., 1]
    rope_re = (x_re * freqs_re) - (x_im * freqs_im)
    rope_im = (x_re * freqs_im) + (x_im * freqs_re)

    if interleaved:
        rope_complex = ops.stack([rope_re, rope_im], axis=-1)
    else:
        rope_complex = ops.concat((rope_re, rope_im), axis=-1)

    return ops.cast(ops.reshape(rope_complex, input.shape), input.dtype)


def _freqs_cis_from_position_ids(
    freqs_cis: TensorValue,
    position_ids: TensorValue,
    *,
    mrope_section: list[int] | None = None,
) -> TensorValue:
    """Build per-token freqs_cis from a freqs table + explicit position_ids."""
    if position_ids.dtype != DType.uint32:
        raise ValueError(
            f"expected position_ids to have dtype uint32, was {position_ids.dtype}"
        )
    if position_ids.rank == 1:
        position_ids = ops.unsqueeze(position_ids, 0)
    if position_ids.rank != 2:
        raise ValueError(
            f"expected position_ids to be 1D or 2D, got rank {position_ids.rank}"
        )

    freqs_by_section = ops.gather(input=freqs_cis, indices=position_ids, axis=0)
    if mrope_section is None:
        if position_ids.shape[0] != 1:
            raise ValueError(
                "mrope_section must be provided when position_ids has multiple sections"
            )
        return freqs_by_section[0]

    if len(mrope_section) != int(position_ids.shape[0]):
        raise ValueError(
            "expected mrope_section to have length "
            f"{position_ids.shape[0]}, was {len(mrope_section)}"
        )

    head_dim = freqs_cis.shape[-1]
    freqs_by_section = freqs_by_section.reshape(
        (position_ids.shape[0], position_ids.shape[1], head_dim // 2, 2)
    )
    freqs_t = freqs_by_section[0]

    h_offset = 1
    w_offset = 2
    step = 3
    h_length = mrope_section[h_offset] * step
    w_length = mrope_section[w_offset] * step

    h_indices = ops.range(
        h_offset,
        h_length,
        step,
        device=position_ids.device,
        dtype=DType.int64,
        out_dim=(h_length + 1) // step,
    )
    w_indices = ops.range(
        w_offset,
        w_length,
        step,
        device=position_ids.device,
        dtype=DType.int64,
        out_dim=(w_length + 1) // step,
    )

    total_seq_len = position_ids.shape[1]
    freqs_h_selected = ops.gather(
        input=freqs_by_section[h_offset], indices=h_indices, axis=1
    )
    h_indices_for_scatter = ops.tile(
        ops.unsqueeze(h_indices, 0), (total_seq_len, 1)
    )
    freqs_t = ops.scatter(
        input=freqs_t,
        updates=freqs_h_selected,
        indices=h_indices_for_scatter,
        axis=1,
    )

    freqs_w_selected = ops.gather(
        input=freqs_by_section[w_offset], indices=w_indices, axis=1
    )
    w_indices_for_scatter = ops.tile(
        ops.unsqueeze(w_indices, 0), (total_seq_len, 1)
    )
    freqs_t = ops.scatter(
        input=freqs_t,
        updates=freqs_w_selected,
        indices=w_indices_for_scatter,
        axis=1,
    )

    return ops.reshape(freqs_t, (total_seq_len, head_dim))


def rope_ragged_with_position_ids(
    input: TensorValue,
    freqs_cis: TensorValue,
    position_ids: TensorValue,
    *,
    mrope_section: list[int] | None = None,
    interleaved: bool = True,
) -> TensorValue:
    """Apply RoPE using explicit position_ids (no KV cache coupling)."""
    per_token_freqs = _freqs_cis_from_position_ids(
        freqs_cis,
        position_ids,
        mrope_section=mrope_section,
    )
    return _apply_rope_with_freqs_cis(
        input, per_token_freqs, interleaved=interleaved
    )


def flash_attention_padded_kv_cache(
    kv_params: KVCacheParams,
    q: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    valid_lengths: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    local_window_size: int = -1,
) -> TensorValue:
    """Computes flash attention with padded inputs and paged KV cache.

    Args:
        kv_params: KV cache parameters
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        kv_collection: Paged KV cache collection
        layer_idx: Layer index for cache lookup
        valid_lengths: Buffer of shape [batch] with dtype uint32 indicating
            actual (non-padded) sequence lengths for each batch element
        mask_variant: The mask variant to use for attention
        scale: Scaling factor for attention scores
        local_window_size: Local window size for sliding window attention

    Returns:
        Output tensor of shape [batch, seq_len, num_heads, head_dim]

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if valid_lengths.dtype != DType.uint32:
        raise ValueError(
            f"expected uint32 valid_lengths but got {valid_lengths.dtype}"
        )

    if valid_lengths.rank != 1:
        raise ValueError(
            f"expected valid_lengths to be rank 1, got {valid_lengths.rank}"
        )

    if valid_lengths.shape[0] != q.shape[0]:
        raise ValueError(
            f"valid_lengths batch size ({valid_lengths.shape[0]}) must match "
            f"q batch size ({q.shape[0]})"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for flash_attention_padded_kv_cache: "
            f"{kv_params.cache_strategy}"
        )

    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    parameters: dict[str, int | str | DType] = {
        "mask_str": mha_mask_config.attention_mask_variant.value,
        "score_mod_str": mha_mask_config.positional_encoding_variant.value,
        "local_window_size": local_window_size,
    }

    cache_strategy_str = kernel_substring(kv_params.cache_strategy)
    op_name = f"mo.mha.padded.{cache_strategy_str}"

    return ops.inplace_custom(
        op_name,
        device=q.device,
        values=[
            q,
            *kv_collection,
            layer_idx,
            valid_lengths,
            ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[TensorType(dtype=q.dtype, shape=q.shape, device=q.device)],
        parameters=parameters,
    )[0].tensor


def flash_attention_gpu(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    local_window_size: int = -1,
    valid_length: TensorValue | None = None,
) -> TensorValue:
    """Computes flash attention using GPU-optimized kernel.

    Args:
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        v: Value tensor of shape [batch, seq_len, num_heads, head_dim]
        mask_variant: The mask variant to use for attention
        scale: Scaling factor for attention scores
        local_window_size: Local window size for sliding window attention
        valid_length: Optional tensor of shape [batch] with dtype uint32.
            When provided, uses the padded kernel variant that respects
            the valid sequence lengths for each batch element.

    Returns:
        Output tensor of shape [batch, seq_len, num_heads, head_dim]
    """
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "q, k, v must have matching dtypes. Got "
            f"q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}"
        )

    expected_rank = 4
    for name, tensor in [("q", q), ("k", k), ("v", v)]:
        if tensor.rank != expected_rank:
            raise ValueError(
                f"{name} must be rank {expected_rank}, got {tensor.rank}"
            )

    # Validate head dimension matches across all inputs
    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        raise ValueError(
            "All inputs must have same head_dim. Got "
            f"q: {head_dim}, k: {k.shape[-1]}, v: {v.shape[-1]}"
        )

    # Validate valid_length if provided
    if valid_length is not None:
        if valid_length.dtype != DType.uint32:
            raise ValueError(
                f"valid_length must have dtype uint32, got {valid_length.dtype}"
            )

        if valid_length.rank != 1:
            raise ValueError(
                f"valid_length must be rank 1, got {valid_length.rank}"
            )

        if valid_length.shape[0] != q.shape[0]:
            raise ValueError(
                f"valid_length batch size ({valid_length.shape[0]}) must match "
                f"q batch size ({q.shape[0]})"
            )

    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    parameters: dict[str, int | str | DType] = {}
    parameters["mask_str"] = mha_mask_config.attention_mask_variant.value
    parameters["score_mod_str"] = (
        mha_mask_config.positional_encoding_variant.value
    )
    parameters["local_window_size"] = local_window_size

    op_name = "mo.mha.no_cache"
    values = [q, k, v]
    if valid_length is not None:
        op_name = "mo.mha.padded.no_cache"
        values.append(valid_length)
    values.append(
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU())
    )

    return ops.custom(
        op_name,
        values=values,
        out_types=[TensorType(dtype=q.dtype, shape=q.shape, device=q.device)],
        parameters=parameters,
        device=q.device,
    )[0].tensor


def flash_attention_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    local_window_size: int = -1,
    sink_weights: TensorValue | None = None,
) -> TensorValue:
    """Computes flash (self) attention provided the `!mo.opaque` KV Cache.

    Notably, this materializes the attention mask (dependent on MHAMaskVariant)
    within the kernel.
    `input` and `input_row_offsets` are used together to implement the ragged
    tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`

    Note that this is self attention and the KV sequence length is
    assumed to be equal to the Q sequence length.
    For KV sequence length != Q sequence length, use `cross_attention_ragged`.

    Args:
        kv_params: KVCacheParams object containing key-value cache parameters.
        input: TensorValue representing the input tensor with shape [total_seq_len, hidden_dim].
        input_row_offsets: TensorValue indicating the start and end of each batch in the input tensor with shape [batch_size + 1].
        kv_collection: PagedCacheValues object for managing key-value cache.
        layer_idx: TensorValue representing the layer index, expected to have dtype uint32.
        mask_variant: MHAMaskVariant specifying the type of attention mask to use.
        scale: float value used to scale the attention scores.
        local_window_size: int specifying the size of the local attention window, default is -1 for no local window.
        sink_weights: Optional tensor of shape [num_heads] containing learnable sink weights for each attention head.
    """
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )

    if input.dtype != kv_params.dtype:
        raise ValueError(
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for flash_attention_ragged: {kv_params.cache_strategy}"
        )

    if sink_weights is not None:
        if sink_weights.rank != 1:
            raise ValueError(
                f"expected sink_weights to have rank 1, got {sink_weights.rank}"
            )
        num_attention_heads = input.shape[1]
        if sink_weights.shape[0] != num_attention_heads:
            raise ValueError(
                f"expected sink_weights to have shape [{num_attention_heads}], "
                f"got {sink_weights.shape}"
            )

    cache_strategy_str = kernel_substring(kv_params.cache_strategy)
    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]

    # Select kernel based on whether sink_weights is provided
    op_name = f"mo.mha.ragged.{cache_strategy_str}"

    if sink_weights is not None:
        op_name += ".sink_weights"
    parameters: dict[str, int | str | DType] = {
        "mask_str": mha_mask_config.attention_mask_variant.value,
        "score_mod_str": mha_mask_config.positional_encoding_variant.value,
        "local_window_size": local_window_size,
    }

    values: MutableSequence[Value[Any]] = [
        input,
        input_row_offsets,
        *kv_collection,
        layer_idx,
        # NOTE: The scale argument to flash attention is constrained to float32.
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
    ]
    if sink_weights is not None:
        values.append(sink_weights)

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
    )[0].tensor


def flash_attention_ragged_gpu(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    input_row_offsets: TensorValue,
    max_seq_len: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    local_window_size: int = -1,
) -> TensorValue:
    """Computes flash attention for ragged inputs using GPU-optimized kernel
    without a KV cache.

    Args:
        q: Query tensor of shape [total_seq_len, num_heads, head_dim] (ragged)
        k: Key tensor of shape [total_seq_len, num_heads, head_dim] (ragged)
        v: Value tensor of shape [total_seq_len, num_heads, head_dim] (ragged)
        input_row_offsets: Buffer of shape [batch_size + 1] with dtype uint32.
            Indicates where each sequence starts and ends in the ragged tensors.
            The values should be a prefix sum (cumulative sum) of sequence lengths.
        mask_variant: The mask variant to use for attention
        scale: Scaling factor for attention scores
        local_window_size: Local window size for sliding window attention

    Returns:
        Output tensor of shape [total_seq_len, num_heads, head_dim]
    """
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "q, k, v must have matching dtypes. Got "
            f"q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}"
        )

    expected_rank = 3
    for name, tensor in [("q", q), ("k", k), ("v", v)]:
        if tensor.rank != expected_rank:
            raise ValueError(
                f"{name} must be rank {expected_rank}, got {tensor.rank}"
            )

    # Validate head dimension matches across all inputs
    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        raise ValueError(
            "All inputs must have same head_dim. Got "
            f"q: {head_dim}, k: {k.shape[-1]}, v: {v.shape[-1]}"
        )

    # Validate total sequence lengths match
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(
            "q, k, v must have same total sequence length. Got "
            f"q: {q.shape[0]}, k: {k.shape[0]}, v: {v.shape[0]}"
        )

    # Validate num_heads match
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError(
            "q, k, v must have same num_heads. Got "
            f"q: {q.shape[1]}, k: {k.shape[1]}, v: {v.shape[1]}"
        )

    # Validate input_row_offsets
    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            f"input_row_offsets must have dtype uint32, got {input_row_offsets.dtype}"
        )

    if input_row_offsets.rank != 1:
        raise ValueError(
            f"input_row_offsets must be rank 1, got {input_row_offsets.rank}"
        )

    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    parameters: dict[str, int | str | DType] = {}
    parameters["mask_str"] = mha_mask_config.attention_mask_variant.value
    parameters["score_mod_str"] = (
        mha_mask_config.positional_encoding_variant.value
    )
    parameters["local_window_size"] = local_window_size

    op_name = "mo.mha.ragged.no_cache"
    values = [q, k, v, input_row_offsets, max_seq_len]
    values.append(
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU())
    )

    return ops.custom(
        op_name,
        values=values,
        out_types=[
            TensorType(
                dtype=q.dtype,
                shape=q.shape,
                device=q.device,
            )
        ],
        parameters=parameters,
        device=q.device,
    )[0].tensor


def flare_mla_decode_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    qk_rope_dim: int = 64,
) -> TensorValue:
    """Computes flash (self) attention provided the `!mo.opaque` KV Cache.

    Notably, this materializes the attention mask (dependent on MHAMaskVariant)
    within the kernel.
    `input` and `input_row_offsets` are used together to implement the ragged
    tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`

    Note that this is self attention and the KV sequence length is
    assumed to be equal to the Q sequence length.
    For KV sequence length != Q sequence length, use `cross_attention_ragged`.
    """
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )

    # TODO: This check needs to be removed once FP8 KVCache is supported (KERN-2394).
    if input.dtype != kv_params.dtype:
        raise ValueError(
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for flash_attention_ragged: {kv_params.cache_strategy}"
        )

    if kv_collection.kv_blocks.shape[1] != 1:
        raise ValueError(
            f"expected kv_collection.kv_blocks.shape[1] to be 1, got {kv_collection.kv_blocks.shape[1]}"
        )

    assert kv_params.page_size is not None
    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    parameters: dict[str, int | str | DType] = {
        "mask_str": mha_mask_config.attention_mask_variant.value,
        "score_mod_str": mha_mask_config.positional_encoding_variant.value,
    }

    op_name = "mo.mla.decode.ragged.paged"

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=[
            input,
            input_row_offsets,
            *kv_collection,
            layer_idx,
            # NOTE: The scale argument to flash attention is constrained to float32.
            ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=[
                    input.shape[0],
                    input.shape[1],
                    input.shape[2] - qk_rope_dim,
                ],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def flare_mla_prefill_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    k: TensorValue,
    v: TensorValue,
    input_row_offsets: TensorValue,
    buffer_row_offsets: TensorValue,
    cache_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    qk_rope_dim: int = 64,
) -> TensorValue:
    """Performs MLA prefill. In the MLA prefill, we need to decompress
    the KV tensors, as we store the latent representations in the KV cache.
    We will decompress the KV tensors into a fixed size buffer to avoid
    out-of-memory errors. In case the total cache length is greater than
    the buffer size, we will process the attention calculation in chunks.

    This MLA prefill kernel will return the output tensor for this iteration
    and the softmax info tensor for this iteration. Such tensors will be used
    by the next iteration of the MLA prefill kernel to continue the attention
    calculation.

    Args:
        kv_params: KVCacheParams
        input: Input tensor
        k: Key tensor
        v: Value tensor
        input_row_offsets: Indicates where each batch starts and ends in `input`
        buffer_row_offsets: Indicates where each batch starts and ends in the buffer
        cache_offsets: Indicates where each batch starts and ends in the KV cache
        kv_collection: KV collection
        layer_idx: Layer index tensor
        mask_variant: Mask variant
        scale: Scale
        qk_rope_dim: QK rope dimension

    Returns:
        The output tensor for this iteration
    """
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for flare_mla_prefill_ragged: {kv_params.cache_strategy}"
        )

    assert kv_params.page_size is not None
    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    parameters: dict[str, int | str | DType] = {
        "mask_str": mha_mask_config.attention_mask_variant.value,
        "score_mod_str": mha_mask_config.positional_encoding_variant.value,
    }

    op_name = "mo.mla.prefill.ragged.paged"

    input_values: MutableSequence[Value[Any]] = [
        input,
        k,
        v,
        buffer_row_offsets,
        cache_offsets,
        input_row_offsets,
        *kv_collection,
        layer_idx,
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
    ]

    results = ops.inplace_custom(
        op_name,
        device=input.device,
        values=input_values,
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=[
                    input.shape[0],
                    input.shape[1],
                    input.shape[2] - qk_rope_dim,
                ],
                device=input.device,
            )
        ],
        parameters=parameters,
    )

    return results[0].tensor


def flare_mla_prefill_plan(
    kv_params: KVCacheParams,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    buffer_size: int,
    max_chunks: int = 16,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """This kernel plans how to process a batch of sequences with
    varying lengths using a fixed-size buffer.

    Each sequence in the batch has some existing cached tokens and new input
    tokens. The kernel divides the total tokens into chunks of buffer_size.

    For each chunk (iteration), it calculates:
        1. Buffer offsets for each sequence in each chunk
        2. Cache offsets for each sequence in each chunk
        3. Total buffer lengths for each processing iteration
    """

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for flare_mla_prefill_plan: {kv_params.cache_strategy}"
        )

    assert kv_params.page_size is not None

    buffer_size_tensor = ops.constant(
        buffer_size, DType.uint32, device=DeviceRef.CPU()
    )

    results = ops.inplace_custom(
        "mo.mla.prefill.ragged.plan",
        device=input_row_offsets.device,
        values=[
            input_row_offsets,
            *kv_collection,
            layer_idx,
            buffer_size_tensor,
        ],
        out_types=[
            TensorType(
                dtype=DType.uint32,
                shape=[max_chunks, input_row_offsets.shape[0]],
                device=input_row_offsets.device,
            ),  # buffer_row_offsets
            TensorType(
                dtype=DType.uint32,
                shape=[max_chunks, input_row_offsets.shape[0]],
                device=input_row_offsets.device,
            ),  # cache_offsets
            TensorType(
                dtype=DType.int32,
                shape=[max_chunks],
                device=input_row_offsets.device,
            ),  # buffer_lengths
        ],
    )

    return results[0].tensor, results[1].tensor, results[2].tensor


def _build_mla_mask_parameters(
    mask_variant: MHAMaskVariant,
) -> dict[str, int | str | DType]:
    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    return {
        "mask_str": mha_mask_config.attention_mask_variant.value,
        "score_mod_str": mha_mask_config.positional_encoding_variant.value,
    }


def _validate_mla_prefill_decode_graph_inputs(
    q: TensorValue,
    input_row_offsets: TensorValue,
    kv_params: KVCacheParams,
    layer_idx: TensorValue,
    *,
    op_name: str,
    tensor_name: str = "q",
    expected_dtype: DType | None = None,
) -> None:
    input_rank_expected = 3
    if q.rank != input_rank_expected:
        raise ValueError(
            f"expected {tensor_name} of rank {input_rank_expected} but got {q.rank}"
        )

    if expected_dtype is not None and q.dtype != expected_dtype:
        raise ValueError(
            f"expected {tensor_name} to be dtype: {expected_dtype}, got {q.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for {op_name}: {kv_params.cache_strategy}"
        )

    assert kv_params.page_size is not None


def _build_mla_prefill_decode_out_type(
    q: TensorValue,
    v_head_dim: int,
) -> TensorType:
    return TensorType(
        dtype=q.dtype,
        shape=[q.shape[0], q.shape[1], v_head_dim],
        device=q.device,
    )


def mla_prefill_graph(
    q: TensorValue,
    input_row_offsets: TensorValue,
    freqs_cis: TensorValue,
    kv_norm_gamma: TensorValue,
    buffer_row_offsets: TensorValue,
    cache_offsets: TensorValue,
    buffer_length: TensorValue,
    w_k: TensorValue,
    w_uv: TensorValue,
    kv_params: KVCacheParams,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    epsilon: float,
    v_head_dim: int,
    *,
    w_k_scale: TensorValue | None = None,
    w_uv_scale: TensorValue | None = None,
    float8_config: Float8Config | None = None,
) -> TensorValue:
    """
    This is a manually fused kernel that performs the following operations:
    - Apply RoPE to the query and the key cache (in-place).
    - Apply RMSNorm to the non-rope portion of the key cache (in-place).
    - Copy the KV latent values from PagedKVCache to a contiguous buffer.
    - Quantize the KV latent values to fp8.
    - Up-project the latent KV values to full K and V through two matmuls.
    - Perform MLA prefill.

    Args:
        q: Combined query tensor containing both nope and rope parts. Shape:
            [tot_seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim].
        input_row_offsets: Indicates where each request starts and ends in
            `input`. This is a 1D tensor of shape [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values for rotary position
            embeddings. Shape: [max_seq_len, qk_rope_head_dim].
        kv_a_proj_layernorm: RMSNorm gamma weights for normalizing the KV cache.
            Shape: [kv_lora_rank].
        buffer_row_offsets: Indicates where each request's KV latent values
            should be stored in the contiguous buffer. This is a 1D tensor of
            shape [num_batches + 1].
        cache_offsets: Indicates the starting token position in the KV cache
            from which to copy KV latent values for each request. This is a 1D
            tensor of shape [num_batches + 1].
        buffer_length: The total number of tokens in the KV cache. Scalar.
        w_k: Weight matrix for up-projecting latent KV values to full K.
            Shape: [num_heads * qk_nope_head_dim, kv_latent_dim].
        w_uv: Weight tensor for up-projecting latent KV values to full V.
            Shape: [num_heads, v_head_dim, kv_latent_dim].
        kv_params: KVCacheParams
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        mask_variant: Mask variant.
        scale: Scale for the attention calculation.
        epsilon: Small constant for numerical stability in RMSNorm.
        v_head_dim: Dimension of the V heads.
        w_k_scale: Optional FP8 scale tensor for `w_k`.
        w_uv_scale: Optional FP8 scale tensor for `w_uv`.
        float8_config: Optional FP8 config. When set, FP8 scales are required.

    Returns:
        Tensor of shape [total_seq_len, num_heads, v_head_dim].
    """

    _validate_mla_prefill_decode_graph_inputs(
        q,
        input_row_offsets,
        kv_params,
        layer_idx,
        op_name="mla_prefill_graph",
        expected_dtype=kv_params.dtype,
    )
    parameters = _build_mla_mask_parameters(mask_variant)

    input_values: MutableSequence[Value[Any]] = [
        q,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        buffer_row_offsets[0],  # one-shot prefill.
        cache_offsets[0],  # one-shot prefill.
        buffer_length[0],  # one-shot prefill.
        w_k,
        w_uv,
        *kv_collection,
        layer_idx,
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ops.constant(epsilon, dtype=DType.float32, device=DeviceRef.CPU()),
    ]

    op_name = "mo.mla.graph.prefill.paged"
    if float8_config is not None:
        assert w_k_scale is not None and w_uv_scale is not None
        assert float8_config.input_scale.block_size is not None
        assert float8_config.weight_scale.block_size is not None
        parameters.update(
            {
                "m_scale_granularity": float8_config.input_scale.block_size[0],
                "n_scale_granularity": float8_config.weight_scale.block_size[0],
                "k_scale_granularity": float8_config.weight_scale.block_size[1],
            }
        )
        op_name += ".fp8"
        input_values += [w_k_scale, w_uv_scale]
    else:
        assert w_k_scale is None and w_uv_scale is None, (
            "w_k_scale and w_uv_scale must be None when float8_config is not set"
        )

    return ops.inplace_custom(
        op_name,
        device=q.device,
        values=input_values,
        out_types=[_build_mla_prefill_decode_out_type(q, v_head_dim)],
        parameters=parameters,
    )[0].tensor


def mla_decode_graph(
    q: TensorValue,
    input_row_offsets: TensorValue,
    freqs_cis: TensorValue,
    kv_norm_gamma: TensorValue,
    w_uk: TensorValue,
    w_uv: TensorValue,
    kv_params: KVCacheParams,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    epsilon: float,
    v_head_dim: int,
    *,
    w_uk_scale: TensorValue | None = None,
    w_uv_scale: TensorValue | None = None,
    float8_config: Float8Config | None = None,
) -> TensorValue:
    """
    This is a manually fused kernel that performs the following operations:

    - Apply RoPE to the query and the key cache (in-place).
    - Apply RMSNorm to the non-rope portion of the key cache (in-place).
    - Project q_nope to kv_latent_dim through a fp8 batched matmul:
      q_nope_proj = q_nope_t @ w_uk
    - Concatenate q_nope_proj and q_rope:
      q_full = concat(q_nope_proj, q_rope, axis=2)
    - Perform MLA decode
    - Project raw_output to v_head_dim through another fp8 batched matmul:
      output = raw_output_t @ w_uv

    Args:
        q: Combined query tensor containing both nope and rope parts. Shape:
            [tot_seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim].
        input_row_offsets: Indicates where each request starts and ends in
            `input`. This is a 1D tensor of shape [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values for rotary position
            embeddings. Shape: [max_seq_len, qk_rope_head_dim].
        kv_a_proj_layernorm: RMSNorm gamma weights for normalizing the KV cache.
            Shape: [kv_lora_rank].
        w_uk: Weight matrix for projecting q_nope to kv_latent_dim. Shape:
            [num_heads, kv_latent_dim, qk_nope_head_dim].
        w_uv: Weight matrix for projecting MLA decode output to v_head_dim.
            Shape: [num_heads, v_head_dim, kv_latent_dim].
        kv_params: KVCacheParams
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        mask_variant: Mask variant.
        scale: Scale for the attention calculation.
        epsilon: Small constant for numerical stability in RMSNorm.
        v_head_dim: Dimension of the V heads.
        w_uk_scale: Optional FP8 scale tensor for `w_uk`.
        w_uv_scale: Optional FP8 scale tensor for `w_uv`.
        float8_config: Optional FP8 config. When set, FP8 scales are required.

    Returns:
        Tensor of shape [total_seq_len, num_heads, v_head_dim].
    """

    _validate_mla_prefill_decode_graph_inputs(
        q,
        input_row_offsets,
        kv_params,
        layer_idx,
        op_name="mla_decode_graph",
        expected_dtype=kv_params.dtype,
    )
    parameters = _build_mla_mask_parameters(mask_variant)

    input_values: MutableSequence[Value[Any]] = [
        q,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        w_uk,
        w_uv,
        *kv_collection,
        layer_idx,
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ops.constant(epsilon, dtype=DType.float32, device=DeviceRef.CPU()),
    ]

    op_name = "mo.mla.graph.decode.paged"
    if float8_config is not None:
        assert w_uk_scale is not None and w_uv_scale is not None
        assert float8_config.input_scale.block_size is not None
        assert float8_config.weight_scale.block_size is not None
        parameters.update(
            {
                "m_scale_granularity": float8_config.input_scale.block_size[0],
                "n_scale_granularity": float8_config.weight_scale.block_size[0],
                "k_scale_granularity": float8_config.weight_scale.block_size[1],
            }
        )
        op_name += ".fp8"
        input_values += [w_uk_scale, w_uv_scale]

    return ops.inplace_custom(
        op_name,
        device=q.device,
        values=input_values,
        out_types=[_build_mla_prefill_decode_out_type(q, v_head_dim)],
        parameters=parameters,
    )[0].tensor


def mla_prefill_decode_graph(
    q: TensorValue,
    input_row_offsets: TensorValue,
    freqs_cis: TensorValue,
    kv_norm_gamma: TensorValue,
    buffer_row_offsets: TensorValue,
    cache_offsets: TensorValue,
    buffer_length: TensorValue,
    w_k: TensorValue,
    w_uk: TensorValue,
    w_uv: TensorValue,
    kv_params: KVCacheParams,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    epsilon: float,
    v_head_dim: int,
    *,
    w_k_scale: TensorValue | None = None,
    w_uk_scale: TensorValue | None = None,
    w_uv_scale: TensorValue | None = None,
    float8_config: Float8Config | None = None,
) -> TensorValue:
    """Fused MLA prefill/decode kernel for FP8.

    Switches between prefill and decode based on the maximum sequence length in
    the batch. See `mla_prefill_graph` and `mla_decode_graph` for the dedicated
    paths.

    Args:
        q: Combined query tensor with nope+rope parts.
        input_row_offsets: Row offsets for the batch.
        freqs_cis: RoPE frequencies tensor.
        kv_norm_gamma: RMSNorm gamma for KV cache.
        buffer_row_offsets: One-shot prefill buffer row offsets.
        cache_offsets: One-shot prefill cache offsets.
        buffer_length: One-shot prefill buffer length tensor.
        w_k: Prefill K up-projection weights.
        w_uk: Decode query-projection weights.
        w_uv: Decode output-projection / prefill V-projection weights.
        kv_params: KV cache parameters.
        kv_collection: Paged KV cache values.
        layer_idx: Layer index (uint32).
        mask_variant: Attention mask variant.
        scale: Attention scale.
        epsilon: RMSNorm epsilon.
        v_head_dim: Value head dimension for output tensor shape.
        w_k_scale: Optional FP8 scale tensor for `w_k`.
        w_uk_scale: Optional FP8 scale tensor for `w_uk`.
        w_uv_scale: Optional FP8 scale tensor for `w_uv`.
        float8_config: Optional FP8 config. When set, FP8 scales are required.

    Returns:
        Tensor of shape [total_seq_len, num_heads, v_head_dim].
    """

    _validate_mla_prefill_decode_graph_inputs(
        q,
        input_row_offsets,
        kv_params,
        layer_idx,
        op_name="mla_prefill_decode_graph",
        expected_dtype=kv_params.dtype,
    )
    parameters = _build_mla_mask_parameters(mask_variant)

    input_values: MutableSequence[Value[Any]] = [
        q,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        buffer_row_offsets[0],  # one-shot prefill.
        cache_offsets[0],  # one-shot prefill.
        buffer_length[0],  # one-shot prefill.
        w_k,
        w_uk,
        w_uv,
        *kv_collection,
        layer_idx,
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ops.constant(epsilon, dtype=DType.float32, device=DeviceRef.CPU()),
    ]
    op_name = "mo.mla.graph.prefill.decode.paged"
    if float8_config is not None:
        assert (
            w_k_scale is not None
            and w_uk_scale is not None
            and w_uv_scale is not None
        )
        assert float8_config.input_scale.block_size is not None
        assert float8_config.weight_scale.block_size is not None
        parameters.update(
            {
                "m_scale_granularity": float8_config.input_scale.block_size[0],
                "n_scale_granularity": float8_config.weight_scale.block_size[0],
                "k_scale_granularity": float8_config.weight_scale.block_size[1],
            }
        )
        op_name += ".fp8"
        input_values += [w_k_scale, w_uk_scale, w_uv_scale]

    return ops.inplace_custom(
        op_name,
        device=q.device,
        values=input_values,
        out_types=[_build_mla_prefill_decode_out_type(q, v_head_dim)],
        parameters=parameters,
    )[0].tensor


def flare_mla_decompress_k_cache(
    kv_params: KVCacheParams,
    buffer_row_offsets_1d: TensorValue,
    cache_offsets_1d: TensorValue,
    buffer_length: TensorValue,
    weight: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    buffer_size: int,
) -> TensorValue:
    """This kernel decompresses the key cache by up-projecting latent representations
    into the KV space using a weight matrix.

    The process involves:

    1. Copying buffer_length latent vectors from the key cache into a contiguous
        buffer (k_latent)
    2. Computing k = k_latent @ weight.T to obtain the decompressed keys

    Returns:
        A tensor of shape [buffer_size, weight.shape[0]] containing the decompressed
        keys. Note that only the first buffer_length tokens are valid.
    """

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if cache_offsets_1d.dtype != DType.uint32:
        raise ValueError(
            f"expected uint32 cache_offsets but got {cache_offsets_1d.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for flare_mla_decompress_k_cache: {kv_params.cache_strategy}"
        )

    assert kv_params.page_size is not None

    results = ops.inplace_custom(
        "mo.mla.decompress.k.cache.ragged.paged",
        device=buffer_row_offsets_1d.device,
        values=[
            buffer_row_offsets_1d,
            cache_offsets_1d,
            buffer_length,
            weight,
            *kv_collection,
            layer_idx,
        ],
        out_types=[
            TensorType(
                dtype=kv_params.dtype,
                shape=[buffer_size, weight.shape[1]],
                device=buffer_row_offsets_1d.device,
            ),  # k_latent_buffer, only stores intermediate values
            TensorType(
                dtype=kv_params.dtype,
                shape=[buffer_size, weight.shape[0]],
                device=buffer_row_offsets_1d.device,
            ),  # k_buffer
        ],
    )

    return results[1].tensor


def kv_cache_get_max_seq_len(
    kv_params: KVCacheParams,
    kv_collection: PagedCacheValues,
) -> TensorValue:
    """This kernel returns the maximum sequence length."""

    assert kv_params.page_size is not None

    return ops.inplace_custom(
        "mo.kv_cache.get_max_seq_len.paged",
        device=DeviceRef.CPU(),
        values=[*kv_collection],
        out_types=[
            TensorType(dtype=DType.uint32, shape=[1], device=DeviceRef.CPU())
        ],
    )[0].tensor[0]


def cross_attention_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    kv_input_row_offsets: TensorValue,
    q_max_seq_len: TensorValue,
    scale: float,
    local_window_size: int = -1,
) -> TensorValue:
    """Computes cross attention provided the `!mo.opaque` KV Cache.

    Notably, this materializes the attention mask (dependent on MHAMaskVariant)
    within the kernel.
    `input` and `input_row_offsets` are used together to implement the ragged
    tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`

    attention, `kv_input_row_offsets` represents the KV sequence length.
    """
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )

    if input.dtype != kv_params.dtype:
        raise ValueError(
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for cross_attention_ragged: {kv_params.cache_strategy}"
        )

    if q_max_seq_len and (q_max_seq_len.dtype != DType.uint32):
        raise ValueError(
            f"expected q_max_seq_len to be uint32 but got {q_max_seq_len.dtype}"
        )

    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    parameters: dict[str, int | str | DType] = {
        "local_window_size": local_window_size,
        "mask_str": mha_mask_config.attention_mask_variant.value,
        "score_mod_str": mha_mask_config.positional_encoding_variant.value,
    }

    cache_strategy_str = kernel_substring(kv_params.cache_strategy)
    op_name = f"mo.cross_attention.ragged.{cache_strategy_str}"

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=[
            input,
            input_row_offsets,
            # Plumb in the query max sequence length for cross attention.
            # For self attention this is the same as the KV max seq len stored
            # on the kv_collection, but that isn't the case for cross attention.
            q_max_seq_len,
            kv_input_row_offsets,
            *kv_collection,
            layer_idx,
            # NOTE: The scale argument to flash attention is constrained to float32.
            ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
    )[0].tensor


def swish_glu(
    a: TensorValueLike, b0: TensorValueLike, b1: TensorValueLike
) -> TensorValue:
    """Computes swish(a@b0.t()) * (a@b1.t())"""
    a = TensorValue(a)
    b0 = TensorValue(b0)
    b1 = TensorValue(b1)
    a_rank_expected = 2
    if a.rank != a_rank_expected:
        raise ValueError(
            f"expected a to have rank {a_rank_expected}, was {a.rank}"
        )

    b0_rank_expected = 2
    if b0.rank != b0_rank_expected:
        raise ValueError(
            f"expected b0 to have rank {b0_rank_expected}, was {b0.rank}"
        )

    b1_rank_expected = 2
    if b1.rank != b1_rank_expected:
        raise ValueError(
            f"expected b1 to have rank {b1_rank_expected}, was {b1.rank}"
        )

    m = a.shape[0]
    n = b0.shape[0]
    if b0.shape[1] != a.shape[1]:
        raise ValueError(
            f"a.shape[1] == {a.shape[1]} != {b0.shape[1]} == b0.shape[1]"
        )

    if b0.shape != b1.shape:
        raise ValueError(f"b0.shape == {b0.shape} != {b1.shape} == b1.shape")

    if a.dtype != b0.dtype or a.dtype != b1.dtype:
        raise ValueError(
            "Element types of all arguments must be equal, but received"
            f" {a.dtype}, {b0.dtype}, and {b1.dtype}."
        )

    return ops.custom(
        "swishGLU",
        device=a.device,
        values=[a, b0, b1],
        out_types=[TensorType(dtype=a.dtype, shape=[m, n], device=a.device)],
    )[0].tensor


def kv_cache_ragged_radd(
    kv_params: KVCacheParams,
    a: TensorValue,
    kv_collection: PagedCacheValues,
    input_row_offsets: TensorValue,
    batch_offset: TensorValue,
    layer_idx: int,
) -> None:
    """This function adds a tensor to a slice of the KVCache, sliced on the batch dimension.

    This expects that the requests which should be sliced out are contiguous and
    in the front of the tensor, and we're only adding to the last requests in the batch.

    Args:
        a: The tensor to add to the KVCache.
        kv_collection: The KVCache collection to add to.
        input_row_offsets: The offsets of the input tensor.
        batch_offset: The batch to start applying the r-add to.
        layer_idx: The layer index to add to.
    """

    if a.rank != 2:
        raise ValueError(f"Expected a to have rank 2 but got {a.rank}")

    if input_row_offsets.rank != 1:
        raise ValueError(
            f"Expected input_row_offsets to have rank 1 but got {input_row_offsets.rank}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"Expected kv_params to have cache strategy PAGED but got {kv_params.cache_strategy}"
        )

    if kv_params.page_size is None:
        raise ValueError("Expected kv_params.page_size to be set")

    # slice input_row_offsets to the batch offset
    input_row_offsets = ops.slice_tensor(
        input_row_offsets,
        [(slice(batch_offset, None), Dim("input_row_offsets_slice_len"))],
    )

    op_name = (
        f"mo.kv_cache.ragged.{kernel_substring(kv_params.cache_strategy)}.radd"
    )

    ops.inplace_custom(
        op_name,
        device=input_row_offsets.device,
        values=[
            a,
            *kv_collection,
            input_row_offsets,
            batch_offset,
            ops.constant(layer_idx, DType.uint32, device=DeviceRef.CPU()),
        ],
    )


def rms_norm_key_cache(
    kv_params: KVCacheParams,
    kv_collection: PagedCacheValues,
    gamma: TensorValue,
    epsilon: float | np.floating[Any],
    layer_idx: TensorValue,
    total_seq_len: Dim,
    input_row_offsets: TensorValue,
    weight_offset: float | np.floating[Any],
    rms_norm_cols: int | None = None,
    multiply_before_cast: bool = True,
    per_head_norm: bool = True,
) -> None:
    """This function applies RMSNorm to the _new_ entries in the KVCache.

    When per_head_norm=True (default), RMSNorm is applied separately to each head.
    In this mode, gamma should have size [head_dim] and normalization occurs
    across the head_dim dimensions within each head.

    When per_head_norm=False, RMSNorm is applied per token across all heads.
    In this mode, gamma should have size [n_kv_heads * head_dim] and normalization
    occurs across all dimensions for each token.

    The size of the gamma tensor determines how many dimensions will be normalized.
    If gamma's size doesn't match the expected size based on per_head_norm setting,
    rms_norm_cols must be explicitly specified to confirm the intention to normalize
    only a subset of dimensions.

    Currently, the KVCacheT class itself isn't aware of the new cache entries
    until cache length increment, which happens after model forward.
    So use `input_row_offsets` to do this bookkeeping.
    """
    cache_strategy_str = kernel_substring(kv_params.cache_strategy)
    op_name = f"mo.rms_norm_kv_cache.ragged.{cache_strategy_str}"

    gamma_rank_expected = 1
    if gamma.rank != gamma_rank_expected:
        raise ValueError(
            f"expected gamma of rank {gamma_rank_expected} but got {gamma.rank}"
        )

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        )

    if gamma.shape[0] != kv_params.head_dim and per_head_norm:
        if rms_norm_cols is None:
            raise ValueError(
                "Size of gamma doesn't match head_dim. Please pass rms_norm_cols "
                "explicitly if you intend to apply RMSNorm to only a subset of "
                "head dimensions"
            )
        elif rms_norm_cols != gamma.shape[0]:
            raise ValueError(
                f"expected gamma of size {rms_norm_cols} but got {gamma.shape[0]}"
            )

    # TODO: Remove this check once FP8 KVCache is supported (KERN-2394).
    if gamma.dtype != kv_params.dtype:
        raise TypeError(
            f"expected gamma dtype {gamma.dtype} to match KV dtype {kv_params.dtype}"
        )

    parameters: dict[str, int | str | DType | bool] = {
        "multiply_before_cast": multiply_before_cast,
        "per_head_norm": per_head_norm,
    }
    if kv_params.cache_strategy == "paged":
        assert kv_params.page_size is not None

    ops.inplace_custom(
        op_name,
        device=input_row_offsets.device,
        values=[
            *kv_collection,
            gamma,
            ops.constant(epsilon, gamma.dtype, device=DeviceRef.CPU()),
            layer_idx,
            ops.cast(TensorValue(total_seq_len), DType.uint32),
            input_row_offsets,
            ops.constant(weight_offset, gamma.dtype, device=DeviceRef.CPU()),
        ],
        parameters=parameters,
    )


def moe_create_indices(
    topk_ids: TensorValue,
    num_local_experts: int,
) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue, TensorValue]:
    """Creates indices for the MoE layer.

    Args:
        topk_ids: The expert assignments for each token from the router.
        num_local_experts: The number of experts on this device.

    Returns:
        A tuple of five tensors:
        - token_expert_order: The reordered token indices, grouped by assigned expert.
        - expert_start_indices: The starting index for each expert's token group in
            the reordered sequence.
        - restore_token_order: The indices to restore original token ordering after
            expert computation.
        - expert_ids: ids of active experts selected for tokens
        - expert_usage_stats: The maximum number of tokens assigned to any expert,
            and the number of active experts.
    """

    results = ops.custom(
        "mo.moe.create.indices",
        device=topk_ids.device,
        values=[
            topk_ids,
        ],
        out_types=[
            TensorType(
                dtype=DType.uint32,
                shape=[topk_ids.shape[0]],
                device=topk_ids.device,
            ),  # token_expert_order
            TensorType(
                dtype=DType.uint32,
                shape=[num_local_experts + 1],
                device=topk_ids.device,
            ),  # expert_start_indices
            TensorType(
                dtype=DType.uint32,
                shape=[topk_ids.shape[0]],
                device=topk_ids.device,
            ),  # restore_token_order
            TensorType(
                dtype=DType.int32,
                shape=[num_local_experts],
                device=topk_ids.device,
            ),  # expert_ids
            TensorType(
                dtype=DType.uint32, shape=[2], device=topk_ids.device
            ),  # expert_usage_stats
        ],
    )

    return (
        results[0].tensor,
        results[1].tensor,
        results[2].tensor,
        results[3].tensor,
        results[4].tensor,
    )


def moe_router_group_limited(
    expert_scores: TensorValue,
    expert_bias: TensorValue,
    n_routed_experts: int,
    n_experts_per_tok: int,
    n_groups: int,
    topk_group: int,
    norm_weights: bool,
    routed_scaling_factor: float,
) -> tuple[TensorValue, TensorValue]:
    """Group limited MoE router.

    Reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L566.

    Args:
        expert_scores: The scores for each expert for each token. Shape:
            [num_tokens, n_routed_experts].
        expert_bias: The bias for each expert. Shape: [n_routed_experts].
        n_routed_experts: The total number of experts. Must be divisible by
            n_groups.
        n_experts_per_tok: The number of experts to be selected per token.
        n_groups: The total number of expert groups. Must be divisible by
            n_routed_experts.
        topk_group: The maximum number of expert groups that a token will be
            routed to.
        norm_weights: Whether to normalize the selected expert weights.
        routed_scaling_factor: The scaling factor for the routed expert weights.

    Returns:
        A tuple of two tensors:
        - expert_indices: The indices of the routed experts for each token.
            Shape: [num_tokens, n_experts_per_tok].
        - expert_weights: The weights of the routed experts for each token.
            Shape: [num_tokens, n_experts_per_tok].
    """

    parameters: dict[str, int | str | DType | bool] = {
        "n_routed_experts": n_routed_experts,
        "n_experts_per_tok": n_experts_per_tok,
        "n_groups": n_groups,
        "topk_group": topk_group,
        "norm_weights": norm_weights,
    }

    if expert_bias.rank != 1:
        raise ValueError(
            f"expected expert_bias of rank 1 but got {expert_bias.rank}"
        )
    if expert_bias.shape[0] != expert_scores.shape[1]:
        raise ValueError(
            f"expected expert_bias of shape [num_experts] but got {expert_bias.shape}"
        )

    results = ops.custom(
        "mo.moe.router.group.limited",
        device=expert_scores.device,
        values=[
            expert_scores,
            expert_bias,
            ops.constant(
                routed_scaling_factor, DType.float32, device=DeviceRef.CPU()
            ),
        ],
        out_types=[
            TensorType(
                dtype=DType.int32,
                shape=[expert_scores.shape[0], n_experts_per_tok],
                device=expert_scores.device,
            ),  # expert_indices
            TensorType(
                dtype=expert_scores.dtype,
                shape=[expert_scores.shape[0], n_experts_per_tok],
                device=expert_scores.device,
            ),  # expert_weights
        ],
        parameters=parameters,
    )

    return (results[0].tensor, results[1].tensor)


def grouped_matmul_ragged(
    hidden_states: TensorValue,
    weight: TensorValue,
    expert_start_indices: TensorValue,
    expert_ids: TensorValue,
    expert_usage_stats_host: TensorValue,
) -> TensorValue:
    """Grouped matmul used in MoE layer.

    `hidden_states` and `expert_start_indices` are used together to implement
    the ragged tensor. `expert_start_indices` indicates where each group starts
    and ends in `hidden_states`

    `expert_ids` is the id of the expert for each group in `hidden_states`

    `expert_usage_stats_host` is the maximum number of tokens assigned to any
    expert, and the number of active experts.

    """

    if weight.rank != 3:
        raise ValueError(f"expected weight of rank 3 but got {weight.rank}")

    if hidden_states.rank != 2:
        raise ValueError(
            f"expected hidden_states of rank 2 but got {hidden_states.rank}"
        )

    if (
        weight.shape[2] != hidden_states.shape[1]
        or weight.shape[0] != expert_ids.shape[0]
    ):
        raise ValueError(
            f"expected weight is of shape [num_experts, *, {hidden_states.shape[1]}] but got {weight.shape}"
        )

    output = ops.custom(
        "mo.grouped.matmul.ragged",
        device=hidden_states.device,
        values=[
            hidden_states,
            weight,
            expert_start_indices,
            expert_ids,
            expert_usage_stats_host[0],
            expert_usage_stats_host[1],
        ],
        out_types=[
            TensorType(
                dtype=hidden_states.dtype,
                shape=[hidden_states.shape[0], weight.shape[1]],
                device=hidden_states.device,
            ),
        ],
    )[0].tensor

    return output


def grouped_dynamic_scaled_nvfp4_matmul(
    hidden_states: TensorValue,
    weight: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    expert_start_indices: TensorValue,
    a_scale_offsets: TensorValue,
    expert_ids: TensorValue,
    expert_scales: TensorValue,
    expert_usage_stats_host: TensorValue,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Performs grouped NVFP4 matmul for MoE layers.

    Performs a grouped matmul with NVFP4 (4-bit) quantized inputs and weights.
    The inputs are packed as uint8 (2 NVFP4 values per byte) with float8_e4m3fn
    scaling factors. NVFP4 uses fixed 1D block scaling with 16 elements per
    scale factor along the K dimension.

    ``hidden_states`` and ``expert_start_indices`` together implement the ragged
    tensor representation for variable-length expert inputs.

    Args:
        hidden_states: The input activations with shape ``[total_tokens, K/2]``
            where K is the unpacked hidden dimension. Dtype must be uint8
            (packed NVFP4).
        weight: The expert weights with shape ``[num_experts, N, K/2]``.
            Dtype must be uint8 (packed NVFP4).
        a_scales: Scaling factors for inputs with shape
            ``[num_scale_rows, K_groups, 32, 4, 4]``. Dtype must be float8_e4m3fn.
        b_scales: Scaling factors for weights with shape
            ``[num_experts, N_groups, K_groups, 32, 4, 4]``. Dtype must be
            float8_e4m3fn.
        expert_start_indices: Indices indicating where each expert's tokens
            start in ``hidden_states``.
        a_scale_offsets: The offsets of the input scale tiles for each expert.
        expert_ids: The expert ID for each group.
        expert_scales: Per-expert scaling factors with shape ``[num_experts]``.
            Dtype must be float32. Multiplied with the matmul output in the
            epilogue.
        expert_usage_stats_host: A tensor containing [max_tokens_per_expert,
            num_active_experts].
        out_type: Output dtype. Defaults to bfloat16.
        tokens_padded_per_expert: If True, tokens per expert are padded for
            alignment. Defaults to False.

    Returns:
        The matmul result with shape ``[total_tokens, N]`` and dtype ``out_type``.
    """

    if weight.rank != 3:
        raise ValueError(f"expected weight of rank 3 but got {weight.rank}")

    if hidden_states.rank != 2:
        raise ValueError(
            f"expected hidden_states of rank 2 but got {hidden_states.rank}"
        )

    weight_k = weight.shape[2]
    hidden_k = hidden_states.shape[1]
    if weight_k != hidden_k or weight.shape[0] != expert_ids.shape[0]:
        raise ValueError(
            "expected weight is of shape [num_experts, *, "
            f"{hidden_k}] but got {weight.shape}"
        )

    if (hidden_states.dtype != DType.uint8) or (weight.dtype != DType.uint8):
        raise TypeError(
            "hidden_states and weight dtypes must be uint8 for NVFP4, but got "
            f"{hidden_states.dtype}, {weight.dtype}"
        )

    if (a_scales.dtype != b_scales.dtype) or (
        a_scales.dtype != DType.float8_e4m3fn
    ):
        raise TypeError(
            "a_scales and b_scales dtypes must be float8_e4m3fn for NVFP4, "
            f"but got {a_scales.dtype}, {b_scales.dtype}"
        )

    if expert_ids.dtype != DType.int32:
        raise TypeError(
            f"expert_ids dtype must be int32, but got {expert_ids.dtype}"
        )

    if expert_ids.rank != 1:
        raise ValueError(
            f"expected expert_ids of rank 1 but got {expert_ids.rank}"
        )
    if expert_start_indices.dtype != DType.uint32:
        raise TypeError(
            "expert_start_indices dtype must be uint32, but got"
            f" {expert_start_indices.dtype}"
        )
    if expert_start_indices.rank != 1:
        raise ValueError(
            "expected expert_start_indices of rank 1 but got"
            f" {expert_start_indices.rank}"
        )

    if a_scales.rank != 5 or b_scales.rank != 6:
        raise ValueError(
            "expected a_scales of rank 5 and b_scales of rank 6 but got"
            f" {a_scales.rank} and {b_scales.rank}"
        )

    if expert_scales.dtype != DType.float32:
        raise TypeError(
            f"expert_scales dtype must be float32, but got {expert_scales.dtype}"
        )
    if expert_scales.rank != 1:
        raise ValueError(
            f"expected expert_scales of rank 1 but got {expert_scales.rank}"
        )

    SF_ATOM_M = [32, 4]
    SF_ATOM_K = 4
    SF_VECTOR_SIZE = 16
    SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128
    SF_K_GROUP_SIZE = SF_ATOM_K * SF_VECTOR_SIZE

    a_scales_dim_1 = ceildiv(hidden_states.shape[1] * 2, Dim(SF_K_GROUP_SIZE))
    if (
        a_scales.shape[1] != a_scales_dim_1
        or a_scales.shape[2] != SF_ATOM_M[0]
        or a_scales.shape[3] != SF_ATOM_M[1]
        or a_scales.shape[4] != SF_ATOM_K
    ):
        raise ValueError(
            "a_scales shape must be "
            f"[*, {a_scales_dim_1}, {SF_ATOM_M[0]}, {SF_ATOM_M[1]}, {SF_ATOM_K}]"
            f" but got {a_scales.shape}"
        )

    b_scales_dim_1 = ceildiv(weight.shape[1], Dim(SF_MN_GROUP_SIZE))
    b_scales_dim_2 = ceildiv(weight.shape[2] * 2, Dim(SF_K_GROUP_SIZE))
    if (
        b_scales.shape[0] != weight.shape[0]
        or b_scales.shape[1] != b_scales_dim_1
        or b_scales.shape[2] != b_scales_dim_2
        or b_scales.shape[3] != SF_ATOM_M[0]
        or b_scales.shape[4] != SF_ATOM_M[1]
        or b_scales.shape[5] != SF_ATOM_K
    ):
        raise ValueError(
            "b_scales shape must be "
            f"[{weight.shape[0]}, {b_scales_dim_1}, {b_scales_dim_2}, "
            f"{SF_ATOM_M[0]}, {SF_ATOM_M[1]}, {SF_ATOM_K}] but got {b_scales.shape}"
        )

    output = ops.custom(
        "mo.grouped.matmul.dynamic.scaled.nvfp4",
        device=hidden_states.device,
        values=[
            hidden_states,
            weight,
            a_scales,
            b_scales,
            expert_start_indices,
            expert_ids,
            a_scale_offsets,
            expert_scales,
            expert_usage_stats_host[0],
            expert_usage_stats_host[1],
        ],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[hidden_states.shape[0], weight.shape[1]],
                device=hidden_states.device,
            ),
        ],
    )[0].tensor

    return output


def grouped_dynamic_scaled_fp8_matmul(
    hidden_states: TensorValue,
    weight: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    expert_start_indices: TensorValue,
    expert_ids: TensorValue,
    expert_usage_stats_host: TensorValue,
    input_scale_spec: Float8InputScaleSpec,
    weight_scale_spec: Float8WeightScaleSpec,
    out_type: DType = DType.bfloat16,
    tokens_padded_per_expert: bool = False,
) -> TensorValue:
    """Grouped blockwise scaled matmul used in MoE layer.

    Perform a grouped blockwise scaled matmul of two tensors with scaling factors.
    `hidden_states` and `expert_start_indices` are used together to implement
    the ragged tensor.

    Args:
        hidden_states: The first tensor to multiply. (2D tensor)
        weight: The second tensor to multiply, must be transposed. (3D tensor)
        a_scales: The scaling factors for the first tensor. (2D tensor)
        b_scales: The scaling factors for the second tensor. (3D tensor)
        expert_start_indices: indicates where each group starts and ends in `hidden_states`.
        expert_ids: The id of the expert for each group in `hidden_states`.
        expert_usage_stats_host: The maximum number of tokens assigned to any expert, and the number of active experts.
        input_scale_spec: The scaling granularity for the input tensor.
        weight_scale_spec: The scaling granularity for the weight tensor.
        tokens_padded_per_expert: If True, It's guaranteed that the number of tokens for each local expert will be
            padded, so that `a_scales` is aligned to 16 bytes. This is needed by the optimized grouped matmul kernel.
    Returns:
        The result of the matmul operation.
    """

    if weight.rank != 3:
        raise ValueError(f"expected weight of rank 3 but got {weight.rank}")

    if hidden_states.rank != 2:
        raise ValueError(
            f"expected hidden_states of rank 2 but got {hidden_states.rank}"
        )

    if (
        weight.shape[2] != hidden_states.shape[1]
        or weight.shape[0] != expert_ids.shape[0]
    ):
        raise ValueError(
            f"expected weight is of shape [num_experts, *, {hidden_states.shape[1]}] but got {weight.shape}"
        )

    if (hidden_states.dtype != weight.dtype) or (
        hidden_states.dtype != DType.float8_e4m3fn
    ):
        raise TypeError(
            f"hidden_states and weight dtypes must be float8_e4m3fn, but got {hidden_states.dtype}, {weight.dtype}"
        )

    if (a_scales.dtype != b_scales.dtype) or (a_scales.dtype != DType.float32):
        raise TypeError(
            f"a_scales and b_scales dtypes must be float32, but got {a_scales.dtype}, {b_scales.dtype}"
        )

    if expert_ids.dtype != DType.int32:
        raise TypeError(
            f"expert_ids dtype must be int32, but got {expert_ids.dtype}"
        )

    if expert_ids.rank != 1:
        raise ValueError(
            f"expected expert_ids of rank 1 but got {expert_ids.rank}"
        )
    if expert_start_indices.dtype != DType.uint32:
        raise TypeError(
            f"expert_start_indices dtype must be uint32, but got {expert_start_indices.dtype}"
        )
    if expert_start_indices.rank != 1:
        raise ValueError(
            f"expected expert_start_indices of rank 1 but got {expert_start_indices.rank}"
        )

    if a_scales.rank != 2 or b_scales.rank != 3:
        raise ValueError(
            f"expected a_scales of rank 2 and b_scales of rank 3 but got {a_scales.rank} and {b_scales.rank}"
        )

    if input_scale_spec.is_block and weight_scale_spec.is_block:
        # a_scale is of shape [ceildiv(K // BLOCK_SIZE), SeqLen-padded]
        # b_scale is of shape [num_of_experts, ceildiv(N // BLOCK_SIZE), ceildiv(K // BLOCK_SIZE)]
        if a_scales.rank != 2:
            raise ValueError(
                f"expected a_scales of rank 2 but got {a_scales.rank}"
            )
        if b_scales.rank != 3:
            raise ValueError(
                f"expected b_scales of rank 3 but got {b_scales.rank}"
            )

        if (
            input_scale_spec.block_size is None
            or weight_scale_spec.block_size is None
        ):
            raise ValueError(
                "both input block_size and weight block_size must be set for grouped blockwise scaling"
            )

        if (
            input_scale_spec.block_size[0] != 1
            or input_scale_spec.block_size[1] != 128
        ):
            raise ValueError(
                "grouped blockwise scaling only supports (1,128) granularity for input"
            )
        if (
            weight_scale_spec.block_size[0] != 128
            or weight_scale_spec.block_size[1] != 128
        ):
            raise ValueError(
                "grouped blockwise scaling only supports (128,128) granularity for weight"
            )
    else:
        raise ValueError("grouped FP8 matmul only supports blockwise scaling")

    output = ops.custom(
        "mo.grouped.matmul.dynamic.scaled.fp8",
        device=hidden_states.device,
        values=[
            hidden_states,
            weight,
            a_scales,
            b_scales,
            expert_start_indices,
            expert_ids,
            expert_usage_stats_host[0],
            expert_usage_stats_host[1],
        ],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[hidden_states.shape[0], weight.shape[1]],
                device=hidden_states.device,
            ),
        ],
        parameters={
            "input_scale_granularity": str(input_scale_spec.granularity),
            "weight_scale_granularity": str(weight_scale_spec.granularity),
            "m_scale_granularity": input_scale_spec.block_size[0],
            "n_scale_granularity": weight_scale_spec.block_size[0],
            "k_scale_granularity": weight_scale_spec.block_size[1],
            "tokens_padded_per_expert": tokens_padded_per_expert,
        },
    )[0].tensor

    return output


def batched_dynamic_scaled_fp8_matmul(
    a: TensorValue,
    b: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    input_scale_spec: Float8InputScaleSpec,
    weight_scale_spec: Float8WeightScaleSpec,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """
    Perform a batched blockwise scaled matmul of two tensors with scaling factors.

    Args:
        a: The first tensor to multiply (3D tensor).
        b: The second tensor to multiply, must be transposed (3D tensor).
        a_scales: The scaling factors for the first tensor (3D tensor).
        b_scales: The scaling factors for the second tensor (3D tensor).

    Returns:
        The result of the matmul operation.
    """
    if a.dtype != b.dtype:
        raise TypeError(
            f"a and b dtypes must match, but got {a.dtype}, {b.dtype}"
        )

    if a_scales.dtype != b_scales.dtype or a_scales.dtype != DType.float32:
        raise TypeError(
            f"a_scales and b_scales dtypes must be float32, but got {a_scales.dtype}, {b_scales.dtype}"
        )

    if a.rank != 3 or b.rank != 3:
        raise ValueError("A and B must be rank 3 tensors")

    if a_scales.rank != 3 or b_scales.rank != 3:
        raise ValueError("A_scales and B_scales must be rank 3 tensors")

    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "The batch dimension of b must match the batch dimension of a"
        )

    if a.shape[2] != b.shape[2]:
        raise ValueError("A and B K dimension does not match")

    if a.dtype != b.dtype or a.dtype != DType.float8_e4m3fn:
        raise TypeError(
            f"a and b dtypes must be float8_e4m3fn, but got {a.dtype}, {b.dtype}"
        )

    if input_scale_spec.is_block and weight_scale_spec.is_block:
        # a_scale is of shape [batch_size, ceildiv(K, BLOCK_SIZE), M-padded]
        # b_scale is of shape [batch_size, ceildiv(N, BLOCK_SIZE), ceildiv(K, BLOCK_SIZE)]
        if a_scales.shape[0] != b_scales.shape[0]:
            raise ValueError(
                "both a_scales and b_scales must have the same shape on the batch dimension"
            )

        if (
            input_scale_spec.block_size is None
            or weight_scale_spec.block_size is None
        ):
            raise ValueError(
                "both input scale_granularity and weight scale_granularity must be set for batched blockwise scaling"
            )

        if (
            input_scale_spec.block_size[0] != 1
            or input_scale_spec.block_size[1] != 128
        ):
            raise ValueError(
                "batched blockwise scaling only supports (1,128) granularity for input"
            )
        if (
            weight_scale_spec.block_size[0] != 128
            or weight_scale_spec.block_size[1] != 128
        ):
            raise ValueError(
                "batched blockwise scaling only supports (128,128) granularity for weight"
            )
    else:
        raise ValueError("unsupported FP8 scaling granularity")

    result = ops.custom(
        "mo.batched.matmul.dynamic.scaled.fp8",
        device=a.device,
        values=[a, b, a_scales, b_scales],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[a.shape[0], a.shape[1], b.shape[1]],
                device=a.device,
            )
        ],
        parameters={
            "input_scale_granularity": str(input_scale_spec.granularity),
            "weight_scale_granularity": str(weight_scale_spec.granularity),
            "m_scale_granularity": input_scale_spec.block_size[0],
            "n_scale_granularity": weight_scale_spec.block_size[0],
            "k_scale_granularity": weight_scale_spec.block_size[1],
        },
    )[0].tensor

    return result


def quantize_static_scaled_float8(
    x: TensorValue,
    scale: TensorValue,
    scale_is_inverted: bool = True,
    out_type: DType = DType.float8_e4m3fn,
) -> TensorValue:
    if scale.shape not in [[], [1]]:
        raise ValueError(
            f"expected scale to be a scalar, but got shape of {scale.shape}"
        )

    if x.dtype not in [DType.float16, DType.bfloat16, DType.float32]:
        raise ValueError(
            f"expected input dtype to be float16, bfloat16, or float32, but got {x.dtype}"
        )

    if x.rank != 2:
        raise ValueError(f"expected input rank to be 2, but got {x.rank}")

    if scale.device != DeviceRef.CPU():
        raise ValueError(f"expected scale to be on CPU, but got {scale.device}")

    return ops.custom(
        "mo.quantize_static_scaled_float8",
        device=x.device,
        values=[x, scale.reshape([])],
        parameters={"scale_is_inverted": scale_is_inverted},
        out_types=[TensorType(dtype=out_type, shape=x.shape, device=x.device)],
    )[0].tensor


def quantize_dynamic_scaled_float8(
    input: TensorValue,
    input_scale_spec: Float8InputScaleSpec,
    weight_scale_spec: Float8WeightScaleSpec,
    scale_ub: float = 1200.0,
    group_size_or_per_token: int = -1,
    out_type: DType = DType.float8_e4m3fn,
    scales_type: DType = DType.bfloat16,
) -> tuple[TensorValue, TensorValue]:
    """
    Dynamically quantize the input tensor to fp8.

    Args:
        input: The input tensor to quantize.
        scale_ub: The upper bound of the scale factor.
        group_size_or_per_token: The group size for quantization. When set to -1,
            the quantization is column-wise.
        out_type: The type of the output tensor.
        scales_type: The type of the scales tensor.

    Returns:
        The quantized tensor and the scales.
    """

    if input.rank != 2:
        raise ValueError("input must be rank 2 tensor")

    if out_type not in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
        raise ValueError("out_type must be float8_e4m3fn or float8_e4m3fnuz")

    if not isinstance(input.shape[1], StaticDim):
        raise ValueError(
            f"input.shape[1] must be a statically known dimension. Input shape received: {input.shape}"
        )

    if group_size_or_per_token == -1:
        if input_scale_spec.is_block or weight_scale_spec.is_block:
            assert input_scale_spec.block_size is not None
            group_size = input_scale_spec.block_size[1]
        else:
            group_size = int(input.shape[1])
    else:
        group_size = group_size_or_per_token

    a_scales_dim1 = input.shape[0]
    if input_scale_spec.is_block or weight_scale_spec.is_block:
        if not (input_scale_spec.is_block and weight_scale_spec.is_block):
            raise ValueError(
                "both input and weight must be blockwise scaled for blockwise scaling"
            )

        # For blockwise scaling pad the a_scales to 16 Bytes. This is required by NVIDIA SM90+ TMA instructions
        padding_size = 16 // scales_type.size_in_bytes
        a_scales_dim1 = (
            (input.shape[0] + padding_size - 1) // padding_size
        ) * padding_size

    result = ops.custom(
        "mo.quantize_dynamic_scaled_float8",
        device=input.device,
        values=[
            input,
            ops.constant(scale_ub, DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[input.shape[0], input.shape[1]],
                device=input.device,
            ),
            TensorType(
                dtype=scales_type,
                shape=[input.shape[1] // group_size, a_scales_dim1],
                device=input.device,
            ),
        ],
        parameters={
            "group_size_or_per_token": group_size,
        },
    )

    return result[0].tensor, result[1].tensor


def batched_quantize_dynamic_scaled_float8(
    input: TensorValue,
    input_scale_spec: Float8InputScaleSpec,
    weight_scale_spec: Float8WeightScaleSpec,
    scale_ub: float = 1200.0,
    group_size_or_per_token: int = -1,
    out_type: DType = DType.float8_e4m3fn,
    scales_type: DType = DType.bfloat16,
) -> tuple[TensorValue, TensorValue]:
    """
    Dynamically quantize the input tensor to fp8.

    Args:
        input: The input tensor to quantize. Shape: [batch_size, seq_len, hidden_size]
        scale_ub: The upper bound of the scale factor.
        group_size_or_per_token: The group size for quantization. When set to -1,
            the quantization is column-wise.
        out_type: The type of the output tensor.
        scales_type: The type of the scales tensor.

    Returns:
        The quantized tensor and the scales.
    """

    if input.rank != 3:
        raise ValueError("input must be rank 3 tensor")

    if out_type not in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
        raise ValueError("out_type must be float8_e4m3fn or float8_e4m3fnuz")

    if scales_type not in (DType.float32, DType.bfloat16, DType.float16):
        raise ValueError("scales_type must be float32, bfloat16, or float16")

    group_size = (
        group_size_or_per_token
        if group_size_or_per_token != -1
        else input.shape[2]
    )

    a_scales_dim1 = input.shape[1]
    if input_scale_spec.is_block or weight_scale_spec.is_block:
        if not (input_scale_spec.is_block and weight_scale_spec.is_block):
            raise ValueError(
                "both input and weight must be blockwise scaled for blockwise scaling"
            )

        # For blockwise scaling pad the a_scales to 16 Bytes. This is required by NVIDIA SM90+ TMA instructions
        padding_size = 16 // scales_type.size_in_bytes
        a_scales_dim1 = (
            (input.shape[1] + padding_size - 1) // padding_size
        ) * padding_size

    result = ops.custom(
        "mo.batched.quantize.dynamic.scaled.fp8",
        device=input.device,
        values=[
            input,
            ops.constant(scale_ub, DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[input.shape[0], input.shape[1], input.shape[2]],
                device=input.device,
            ),
            TensorType(
                dtype=scales_type,
                shape=[
                    input.shape[0],
                    input.shape[2] // group_size,
                    a_scales_dim1,
                ],
                device=input.device,
            ),
        ],
        parameters={
            "group_size_or_per_token": group_size_or_per_token,
        },
    )

    return result[0].tensor, result[1].tensor


def dynamic_scaled_matmul(
    a: TensorValue,
    b: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    input_scale_spec: Float8InputScaleSpec,
    weight_scale_spec: Float8WeightScaleSpec,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """
    Perform a matmul of two tensors with scaling factors. Currently only
    supports channel-wise scaling for weights and per-token scaling for inputs.

    Args:
        a: The first tensor to multiply.
        b: The second tensor to multiply, must be transposed.
        a_scales: The scaling factors for the first tensor.
        b_scales: The scaling factors for the second tensor.

    Returns:
        The result of the matmul operation.
    """

    if a.rank != 2 or b.rank != 2 or a_scales.rank != 2 or b_scales.rank != 2:
        raise ValueError("All arguments must be rank 2 tensors")

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            "The second dimension of b must match the second dimension of a"
        )

    if input_scale_spec.is_tensor and weight_scale_spec.is_tensor:
        if not (
            a_scales.shape[0]
            == a_scales.shape[1]
            == b_scales.shape[0]
            == b_scales.shape[1]
            == 1
        ):
            raise ValueError(
                "scaler tensors must be of shape [1, 1] for tensor scaling"
            )

    elif input_scale_spec.is_colwise and weight_scale_spec.is_rowwise:
        if a_scales.shape[0] != 1:
            raise ValueError("only per-token scaling is supported for a")

        if b_scales.shape[1] != 1:
            raise ValueError("only channel-wise scaling is supported for b")

    elif input_scale_spec.is_block or weight_scale_spec.is_block:
        if (
            input_scale_spec.block_size is None
            or weight_scale_spec.block_size is None
        ):
            raise ValueError(
                "both input and weight block size must be set for blockwise scaling"
            )
        if not (input_scale_spec.is_block and weight_scale_spec.is_block):
            raise ValueError(
                "both input and weight must be blockwise scaled for blockwise scaling"
            )

        if a_scales.dtype != b_scales.dtype or a_scales.dtype != DType.float32:
            raise TypeError(
                f"a_scales and b_scales dtypes must be float32, but got {a_scales.dtype}, {b_scales.dtype}"
            )

        # a_scale is of shape [ceildiv(K, BLOCK_SIZE), M-padded]
        # b_scale is of shape [ceildiv(N, BLOCK_SIZE), ceildiv(K, BLOCK_SIZE)]
        if a_scales.shape[0] != b_scales.shape[1]:
            raise ValueError(
                "both a_scales and b_scales must have the same shape on the K dimension."
                f" got a_scales.shape={a_scales.shape} and b_scales.shape={b_scales.shape}"
            )

    else:
        raise ValueError("unsupported FP8 scaling granularity")

    if (a.dtype != b.dtype) or (a_scales.dtype != b_scales.dtype):
        raise TypeError(
            f"a and b dtypes {a.dtype}, {b.dtype} must match, "
            f"as do a and b scales dtypes {a_scales.dtype}, {b_scales.dtype}"
        )

    result = ops.custom(
        "mo.matmul_dynamic_scaled_fp8",
        device=a.device,
        values=[a, b, a_scales, b_scales],
        out_types=[
            TensorType(
                dtype=out_type, shape=[a.shape[0], b.shape[0]], device=a.device
            )
        ],
        parameters={
            "input_scale_granularity": str(input_scale_spec.granularity),
            "weight_scale_granularity": str(weight_scale_spec.granularity),
            "m_scale_granularity": -1
            if input_scale_spec.block_size is None
            else input_scale_spec.block_size[0],
            "n_scale_granularity": -1
            if weight_scale_spec.block_size is None
            else weight_scale_spec.block_size[0],
            "k_scale_granularity": -1
            if weight_scale_spec.block_size is None
            else weight_scale_spec.block_size[1],
        },
    )[0].tensor

    return result


def dynamic_block_scaled_matmul_fp4(
    a: TensorValue,
    b: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    tensor_sf: TensorValue | float,
    sf_vector_size: int = 16,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """
    Perform a matmul of two FP4 tensors with 1D-block scaled scaling factors.

    Args:
        a: The first tensor to multiply.
        b: The second tensor to multiply, must be transposed.
        a_scales: The scaling factors for the first tensor.
        b_scales: The scaling factors for the second tensor.
        tensor_sf: Buffer-wise scaling factor equal to weight_scale_2 * input_scale (non-inverted).

    Returns:
        The result of the matmul operation.
    """

    if a.rank != 2 or b.rank != 2:
        raise ValueError("Both a and b must be rank 2 tensors")
    if a_scales.rank != 5 or b_scales.rank != 5:
        raise ValueError("Both a_scales and b_scales must be rank 5 tensors")

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            "The second dimension of b must match the second dimension of a"
        )

    if (a.dtype != b.dtype) or (a_scales.dtype != b_scales.dtype):
        raise TypeError(
            f"a and b dtypes {a.dtype}, {b.dtype} must match, "
            f"as do a and b scales dtypes {a_scales.dtype}, {b_scales.dtype}"
        )

    if a.dtype != DType.uint8:
        raise ValueError("A dtype must be uint8 (fp4-e2m1fnX2)")

    if a_scales.dtype != DType.float8_e4m3fn:
        raise ValueError("a_scales dtype must be float8_e4m3fn")

    if sf_vector_size != 16:
        raise ValueError("sf_vector_size must be 16 for NVFP4")

    SF_ATOM_M = [32, 4]
    SF_ATOM_K = 4
    SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128
    SF_K_GROUP_SIZE = SF_ATOM_K * sf_vector_size

    # scales tensor shape: [ceildiv(M, SF_MN_GROUP_SIZE), ceildiv(N, sf_vector_size * 4), SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K]
    # a_scales_dim_0 = (a.shape[0] + SF_MN_GROUP_SIZE - 1) // SF_MN_GROUP_SIZE
    a_scales_dim_1 = ceildiv(
        a.shape[1] * 2, Dim(SF_K_GROUP_SIZE)
    )  # each output element (uint8) is 2 fp4-e2m1fn values
    b_scales_dim_0 = ceildiv(b.shape[0], Dim(SF_MN_GROUP_SIZE))
    b_scales_dim_1 = ceildiv(
        b.shape[1] * 2, Dim(SF_K_GROUP_SIZE)
    )  # each output element (uint8) is 2 fp4-e2m1fn values
    scales_dim_2 = SF_ATOM_M[0]
    scales_dim_3 = SF_ATOM_M[1]
    scales_dim_4 = SF_ATOM_K

    if (
        a_scales.shape[1] != a_scales_dim_1
        or a_scales.shape[2] != scales_dim_2
        or a_scales.shape[3] != scales_dim_3
        or a_scales.shape[4] != scales_dim_4
    ):
        raise ValueError(
            f"a_scales shape must be {a_scales_dim_1, scales_dim_2, scales_dim_3, scales_dim_4}, but got {a_scales.shape}"
        )

    if (
        b_scales.shape[0] != b_scales_dim_0
        or b_scales.shape[1] != b_scales_dim_1
        or b_scales.shape[2] != scales_dim_2
        or b_scales.shape[3] != scales_dim_3
        or b_scales.shape[4] != scales_dim_4
    ):
        raise ValueError(
            f"b_scales shape must be {b_scales_dim_0, b_scales_dim_1, scales_dim_2, scales_dim_3, scales_dim_4}, but got {b_scales.shape}"
        )

    if a_scales.shape[1] != b_scales.shape[1]:
        raise ValueError(
            "a_scales and b_scales must have the same shape on the K dimension."
            f" got a_scales.shape={a_scales.shape} and b_scales.shape={b_scales.shape}"
        )

    result = ops.custom(
        "mo.matmul.dynamic.block.scaled",
        device=a.device,
        values=[
            a,
            b,
            a_scales,
            b_scales,
            ops.constant(tensor_sf, DType.float32, device=DeviceRef.CPU())
            if isinstance(tensor_sf, float)
            else tensor_sf,
        ],
        out_types=[
            TensorType(
                dtype=out_type, shape=[a.shape[0], b.shape[0]], device=a.device
            )
        ],
        parameters={
            "SF_VECTOR_SIZE": sf_vector_size,
        },
    )[0].tensor

    return result


def quantize_dynamic_block_scaled_fp4(
    input: TensorValue,
    tensor_sf: TensorValue | float,
    sf_vector_size: int = 16,
    scales_type: DType = DType.float8_e4m3fn,
    out_type: DType = DType.uint8,  # fp4-e2m1fnX2
) -> tuple[TensorValue, TensorValue]:
    """
    Dynamically quantize the input tensor to fp4-e2m1fn.

    Args:
        input: The input tensor to quantize. Shape: [seq_len, hidden_size]
        tensor_sf: The tensor-wise scale factor (inverted as per quantization kernel requirement).
        sf_vector_size: The block size for the scaling factors.
        out_type: The type of the output tensor.
        scales_type: The type of the scales tensor.

    Returns:
        The quantized tensor in [seq_len, hidden_size // 2] layout and the scales in [ceildiv(seq_len, 128), ceildiv(hidden_size, sf_vector_size * 4), 32, 4, 4] layout.
    """

    if input.rank != 2:
        raise ValueError("input tensor must be rank 2 tensor")

    if input.dtype != DType.bfloat16:
        raise ValueError("input tensor dtype must be bfloat16")

    if out_type not in (DType.uint8,):
        raise ValueError("out_type must be uint8 (fp4-e2m1fnX2)")

    if scales_type not in (DType.float8_e4m3fn,):
        raise ValueError("scales_type must be float8_e4m3fn for NVFP4")

    if sf_vector_size != 16:
        raise ValueError("sf_vector_size must be 16 for NVFP4")

    if int(input.shape[1]) % (sf_vector_size // 2) != 0:
        raise ValueError(
            "input.shape[1] must be a multiple of (sf_vector_size // 2)"
        )

    SF_ATOM_M = [32, 4]
    SF_ATOM_K = 4
    SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128
    SF_K_GROUP_SIZE = SF_ATOM_K * sf_vector_size

    # scales tensor shape: [ceildiv(M, SF_MN_GROUP_SIZE), ceildiv(N, sf_vector_size * 4), SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K]
    scales_dim_0 = ceildiv(input.shape[0], Dim(SF_MN_GROUP_SIZE))
    scales_dim_1 = ceildiv(input.shape[1], Dim(SF_K_GROUP_SIZE))
    scales_dim_2 = SF_ATOM_M[0]
    scales_dim_3 = SF_ATOM_M[1]
    scales_dim_4 = SF_ATOM_K

    result = ops.custom(
        "mo.quantize.dynamic.block.scaled",
        device=input.device,
        values=[
            input,
            ops.constant(tensor_sf, DType.float32, device=DeviceRef.CPU())
            if isinstance(tensor_sf, float)
            else tensor_sf,
        ],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[
                    input.shape[0],
                    input.shape[1] // 2,
                ],  # each output element (uint8) is 2 fp4-e2m1fn values
                device=input.device,
            ),
            TensorType(
                dtype=scales_type,
                shape=[
                    scales_dim_0,
                    scales_dim_1,
                    scales_dim_2,
                    scales_dim_3,
                    scales_dim_4,
                ],
                device=input.device,
            ),
        ],
        parameters={
            "SF_VECTOR_SIZE": sf_vector_size,
        },
    )

    return result[0].tensor, result[1].tensor


def block_scales_interleave(
    scales: TensorValue,
    sf_vector_size: int = 16,
    scales_type: DType = DType.float8_e4m3fn,
) -> TensorValue:
    """
    Interleave the block scales tensor in [M, N] layout to [ceildiv(M, 128), ceildiv(N, sf_vector_size * 4), 32, 4, 4] layout.

    Args:
        scales: The scales tensor to interleave in [M, N] layout.
        sf_vector_size: The block size for the scaling factors.

    Returns:
        The interleaved scales tensor in [ceildiv(M, 128), ceildiv(N, sf_vector_size * 4), 32, 4, 4] layout.
    """

    if scales.rank != 2:
        raise ValueError("Both a and b must be rank 2 tensors")

    if scales.dtype != DType.float8_e4m3fn:
        raise ValueError("scales dtype must be float8_e4m3fn")

    if sf_vector_size != 16:
        raise ValueError("sf_vector_size must be 16 for NVFP4")

    SF_ATOM_M = [32, 4]
    SF_ATOM_K = 4
    SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128

    # scales tensor shape: [ceildiv(M, SF_MN_GROUP_SIZE), ceildiv(N, sf_vector_size * 4), SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K]
    scales_dim_0 = ceildiv(scales.shape[0], Dim(SF_MN_GROUP_SIZE))
    scales_dim_1 = ceildiv(scales.shape[1], Dim(SF_ATOM_K))
    scales_dim_2 = SF_ATOM_M[0]
    scales_dim_3 = SF_ATOM_M[1]
    scales_dim_4 = SF_ATOM_K

    result = ops.custom(
        "mo.interleave.block.scales",
        device=scales.device,
        values=[scales],
        out_types=[
            TensorType(
                dtype=scales_type,
                shape=[
                    scales_dim_0,
                    scales_dim_1,
                    scales_dim_2,
                    scales_dim_3,
                    scales_dim_4,
                ],
                device=scales.device,
            ),
        ],
        parameters={
            "SF_VECTOR_SIZE": sf_vector_size,
        },
    )[0].tensor

    return result


def matmul_static_scaled_float8(
    input: TensorValue,
    weight: TensorValue,
    input_scale: TensorValue,
    weight_scale: TensorValue,
) -> TensorValue:
    if input_scale.shape not in [[], [1]]:
        raise ValueError(
            f"expected input_scale to be a scalar, but got shape of {input_scale.shape}"
        )
    if weight_scale.shape not in [[], [1]]:
        raise ValueError(
            f"expected weight_scale to be a scalar, but got shape of {weight_scale.shape}"
        )

    if input.dtype not in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
        raise ValueError(
            f"expected input dtype to be float8_e4m3fn or float8_e4m3fnuz, but got {input.dtype}"
        )
    if weight.dtype not in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
        raise ValueError(
            f"expected weight dtype to be float8_e4m3fn or float8_e4m3fnuz, but got {weight.dtype}"
        )

    if input.rank != 2:
        raise ValueError(f"expected input rank to be 2, but got {input.rank}")
    if weight.rank != 2:
        raise ValueError(f"expected weight rank to be 2, but got {weight.rank}")

    if input.shape[1] != weight.shape[1]:
        raise ValueError("K dimension does not match for matmul")

    if input_scale.device != DeviceRef.CPU():
        raise ValueError(
            f"expected input_scale to be on CPU, but got {input_scale.device}"
        )

    if weight_scale.device != DeviceRef.CPU():
        raise ValueError(
            f"expected weight_scale to be on CPU, but got {weight_scale.device}"
        )

    return ops.custom(
        "mo.matmul_static_scaled_float8",
        device=input.device,
        values=[
            input,
            weight,
            input_scale.reshape([]),
            weight_scale.reshape([]),
        ],
        out_types=[
            TensorType(
                dtype=DType.bfloat16,
                shape=[input.shape[0], weight.shape[0]],
                device=input.device,
            )
        ],
    )[0].tensor


def needs_fp8_fnuz_conversion() -> bool:
    """Check if we need to convert FP8 E4M3FN to FNUZ for AMD GPUs.

    Returns:
        True if running on AMD GPU with CDNA3 architecture, False otherwise.
    """
    try:
        return "gfx94" in accelerator_architecture_name()
    except Exception:
        return False


def normalize_e4m3fn_to_e4m3fnuz(
    weight: TensorValue,
    weight_scale: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Convert E4M3FN weights to E4M3FNUZ format for AMD GPUs.

    This conversion is necessary because AMD GPUs use the E4M3FNUZ format
    while NVIDIA GPUs use E4M3FN. The key differences are:
    1. The bit pattern 10000000 (-128) represents zero in E4M3FN but NaN in E4M3FNUZ
    2. For the same bit representation, E4M3FNUZ values are half of E4M3FN values

    Args:
        weight: The weight tensor in E4M3FN format.
        weight_scale: The weight scale factor.

    Returns:
        Tuple of (converted_weight, adjusted_weight_scale, adjusted_input_scale).
    """
    if weight.dtype != DType.float8_e4m3fn:
        raise ValueError(
            f"Expected weight dtype to be float8_e4m3fn, but got {weight.dtype}"
        )

    # Convert using custom op that takes float8_e4m3fn input and returns float8_e4m3fnuz
    # Then cast back to float8_e4m3fn to maintain dtype compatibility with kernels
    converted_weight_fnuz = ops.custom(
        "mo.convert_e4m3fn_to_e4m3fnuz",
        device=weight.device,
        values=[weight],
        out_types=[
            TensorType(
                dtype=DType.float8_e4m3fnuz,
                shape=weight.shape,
                device=weight.device,
            )
        ],
    )[0].tensor

    # Cast back to float8_e4m3fn to maintain kernel compatibility
    # The bit pattern has been converted, but we need FN dtype for the kernels
    # converted_weight = ops.cast(converted_weight_fnuz, DType.float8_e4m3fn)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    adjusted_weight_scale = weight_scale * ops.constant(
        2.0, weight_scale.dtype, device=weight_scale.device
    )

    return converted_weight_fnuz, adjusted_weight_scale


def convert_weights_to_fp8_fnuz_if_needed(
    weight: TensorValue,
    weight_scale: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Convert weights and scales to FP8 FNUZ format if needed for AMD GPUs.

    This utility function checks if FP8 FNUZ conversion is needed, currently onli AMD MI300 GPUs,
    and performs the conversion if required. This centralizes the conversion logic
    that was previously duplicated across multiple files.

    Args:
        weight: The weight tensor to potentially convert.
        weight_scale: The weight scale factor.

    Returns:
        Tuple of (weight, weight_scale) - converted if needed, original otherwise.
    """
    if needs_fp8_fnuz_conversion() and weight.dtype == DType.float8_e4m3fn:
        return normalize_e4m3fn_to_e4m3fnuz(weight, weight_scale)
    return weight, weight_scale


def merge_ragged_tensors(
    a: TensorValue,
    a_row_offsets: TensorValue,
    b: TensorValue,
    b_row_offsets: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Merges two ragged tensors into a single ragged tensor.

    Both ragged tensors must have the same batch size (same number of row
    offsets). This function interleaves the rows from each tensor based on
    their row offsets.

    Args:
        a: The first ragged tensor of shape [total_a_rows, ...].
        a_row_offsets: The row offsets of the first ragged tensor,indicating
            where each batch starts and ends in `a`.
        b: The second ragged tensor of shape [total_b_rows, ...].
        b_row_offsets: The row offsets of the second ragged tensor, indicating
            where each batch starts and ends in `b`.

    Returns:
        A tuple of two tensors:
            - The merged ragged tensor with shape
                [total_a_rows + total_b_rows, ...].
            - The merged row offsets with the same shape as input row offsets.

    Example:

    .. code-block:: python

        a = [1, 2, 3, 4, 5, 6]
        a_row_offsets = [0, 2, 6]
        b = [7, 8, 9, 10]
        b_row_offsets = [0, 3, 4]

        merged_tensor, merged_row_offsets = merge_ragged_tensors(
            a, a_row_offsets, b, b_row_offsets)

        merged_tensor = [1, 2, 7, 8, 9, 3, 4, 5, 6, 10]
        merged_row_offsets = [0, 5, 10]
    """

    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")

    if a_row_offsets.shape[0] != b_row_offsets.shape[0]:
        raise ValueError(
            "a_row_offsets and b_row_offsets must have the same shape"
        )

    c_shape = [a.shape[0] + b.shape[0]] + a.shape[1:]

    results = ops.custom(
        "mo.merge_ragged_tensors",
        device=a.device,
        values=[a, a_row_offsets, b, b_row_offsets],
        out_types=[
            TensorType(dtype=a.dtype, shape=c_shape, device=a.device),
            TensorType(
                dtype=DType.uint32, shape=a_row_offsets.shape, device=a.device
            ),
        ],
    )

    return results[0].tensor, results[1].tensor


def apply_penalties_to_logits(
    logits_buffer: BufferValue,
    frequency_data: TensorValue,
    frequency_offsets: TensorValue,
    *,
    frequency_penalty: TensorValueLike = 0.0,
    presence_penalty: TensorValueLike = 0.0,
    repetition_penalty: TensorValueLike = 1.0,
) -> None:
    """
    Applies penalties to the logits.

    Args:
        logits_buffer: The buffer to apply penalties to.
        frequency_data: 2d tensor of shape [unique_tokens, 2], where
            the first column indicates the token id and the second column
            indicates the frequency of the token.
        frequency_offsets: 1d tensor of shape [batch_size + 1], indicating
            start of each sequence's data.
        frequency_penalty: The frequency penalty to apply to the model's output.
            A positive value will penalize new tokens based on their frequency
            in the generated text: tokens will receive a penalty proportional
            to the count of appearances.
        presence_penalty: The presence penalty to apply to the model's output
            A positive value will penalize new tokens that have already appeared
            in the generated text at least once by applying a constant penalty.
        repetition_penalty: The repetition penalty to apply to the model's
            output. Values > 1 will penalize new tokens that have already
            appeared in prompt and generated text at least once by dividing the
            logits by the repetition penalty.
    """

    if logits_buffer.rank != 2:
        raise ValueError("logits_buffer must be a 2d buffer")

    if frequency_data.rank != 2:
        raise ValueError("frequency_data must be a 2d tensor")

    if frequency_offsets.rank != 1:
        raise ValueError("frequency_offsets must be a 1d tensor")

    if isinstance(frequency_penalty, float):
        frequency_penalty_tensor = ops.broadcast_to(
            ops.constant(
                frequency_penalty,
                dtype=DType.float32,
                device=logits_buffer.device,
            ),
            [logits_buffer.shape[0]],
        )
    else:
        frequency_penalty_tensor = TensorValue(frequency_penalty)
        if frequency_penalty_tensor.shape[0] != logits_buffer.shape[0]:
            raise ValueError(
                f"frequency_penalty tensor shape {frequency_penalty_tensor.shape} does not match logits_buffer shape {logits_buffer.shape}"
            )

    if isinstance(presence_penalty, float):
        presence_penalty_tensor = ops.broadcast_to(
            ops.constant(
                presence_penalty,
                dtype=DType.float32,
                device=logits_buffer.device,
            ),
            [logits_buffer.shape[0]],
        )
    else:
        presence_penalty_tensor = TensorValue(presence_penalty)
        if presence_penalty_tensor.shape[0] != logits_buffer.shape[0]:
            raise ValueError(
                f"presence_penalty tensor shape {presence_penalty_tensor.shape} does not match logits_buffer shape {logits_buffer.shape}"
            )

    if isinstance(repetition_penalty, float):
        repetition_penalty_tensor = ops.broadcast_to(
            ops.constant(
                repetition_penalty,
                dtype=DType.float32,
                device=logits_buffer.device,
            ),
            [logits_buffer.shape[0]],
        )
    else:
        repetition_penalty_tensor = TensorValue(repetition_penalty)
        if repetition_penalty_tensor.shape[0] != logits_buffer.shape[0]:
            raise ValueError(
                f"repetition_penalty tensor shape {repetition_penalty_tensor.shape} does not match logits_buffer shape {logits_buffer.shape}"
            )

    ops.inplace_custom(
        "sampler.apply_penalties",
        device=logits_buffer.device,
        values=[
            logits_buffer,
            frequency_data,
            frequency_offsets,
            frequency_penalty_tensor,
            presence_penalty_tensor,
            repetition_penalty_tensor,
        ],
    )


def update_frequency_data(
    frequency_data: BufferValue,
    frequency_offsets: TensorValue,
    tokens: TensorValue,
) -> None:
    """
    Updates the frequency data.

    Args:
        frequency_data: 2d tensor of shape [unique_tokens, 2], where
            the first column indicates the token id and the second column
            indicates the frequency of the token.
        frequency_offsets: 1d tensor of shape [batch_size + 1], indicating
            start of each sequence's data.
        tokens: The tokens to update the frequency data with.
    """

    if frequency_data.rank != 2:
        raise ValueError("frequency_data must be a 2d buffer")

    if frequency_offsets.rank != 1:
        raise ValueError("frequency_offsets must be a 1d tensor")

    if tokens.rank != 1:
        raise ValueError("tokens must be a 1d tensor")

    ops.inplace_custom(
        "sampler.update_frequency_data",
        device=frequency_data.device,
        values=[
            frequency_data,
            frequency_offsets,
            tokens,
        ],
    )


def scatter_set_constant(
    data: BufferValueLike,
    indices: TensorValueLike,
    fill_val: float,
) -> None:
    """
    Scatters values into a tensor at specified indices.
    """
    data = BufferValue(data)
    indices = TensorValue(indices)

    if data.rank != 2:
        raise ValueError(
            "scatter_set_constant currently only supports 2d tensors"
        )

    if indices.rank != 2:
        raise ValueError(
            "scatter_set_constant currently only supports 2d indices"
        )

    ops.inplace_custom(
        "mo.scatter_set_constant",
        device=data.device,
        values=[
            data,
            indices,
            ops.constant(fill_val, data.dtype, device=DeviceRef.CPU()),
        ],
    )


def scatter_nd_skip_oob_indices(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
) -> TensorValue:
    """
    Creates a new symbolic tensor where the updates are scattered into input at specified indices.

    This differs from scatter_nd in that it handles oob indices by skipping
    the update for that index. Oob indices are those which fall outside of
    the range [-dim, dim).

    Args:
        input: The input symbolic tensor to write elements to.
        updates: A symbolic tensor of elements to write to input.
        indices: A tensor of indices specifying where to write updates.
            Shape should be [num_updates, rank] for full indexing or
            [num_updates, k] for partial indexing where k < rank.

    Returns:
        A new symbolic tensor representing the result of the scatter_nd operation.
    """
    input = TensorValue(input)
    updates = TensorValue(updates)
    indices = TensorValue(indices)

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    if indices.dtype not in (DType.int32, DType.int64):
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'. Indices must be of type int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)

    return ops.custom(
        "mo.scatter_nd.skip_neg_indices",
        device=input.device,
        values=[input, updates, indices],
        out_types=[TensorType(input.dtype, input.shape, device=input.device)],
    )[0].tensor


def topk_fused_sampling(
    logits: TensorValue,
    top_k: TensorValueLike,
    *,
    temperature: TensorValueLike = 1.0,
    max_k: TensorValueLike | None = None,
    min_top_p: TensorValueLike | None = None,
    top_p: TensorValueLike = 1.0,
    seed: TensorValueLike = 0,
) -> TensorValue:
    """Performs top-k sampling with temperature scaling.

    Args:
        logits: Input logits tensor of shape [batch_size, vocab_size].
        top_k: Number of top tokens to consider for sampling. Can be a scalar
            (which will be expanded to batch_size) or a tensor of shape [batch_size].
        temperature: Temperature for scaling logits before sampling.
        max_k: Maximum value of k across the batch. Required when top_k is a tensor.
        top_p: Top-p (nucleus) sampling threshold. Can be a scalar or tensor.
        seed: Seed for the random number generator. Can be a scalar or tensor.
    Returns:
        Sampled tokens tensor of shape [batch_size, 1].

    Raises:
        ValueError: If input validation fails.
    """

    batch_size = logits.shape[0]
    device = logits.device
    max_k_tensor = max_k

    if isinstance(top_k, int):
        if top_k <= -1 or top_k > 255:
            raise ValueError(
                f"top_k must be greater than -1 and less than or equal to 255, got {top_k}"
            )

        if top_k == 0:
            top_k = -1

        max_k_tensor = ops.constant(
            top_k, dtype=DType.int64, device=DeviceRef.CPU()
        )
        top_k_tensor = ops.broadcast_to(
            ops.constant(top_k, dtype=DType.int64, device=device), [batch_size]
        )
    else:
        top_k_tensor = TensorValue(top_k)
        if max_k_tensor is None:
            raise ValueError(
                "max_k must be explicitly set when top_k is a tensor"
            )
        if top_k_tensor.shape[0] != batch_size:
            raise ValueError(
                f"top_k tensor shape {top_k_tensor.shape} does not match batch_size {batch_size}"
            )
        max_k_tensor = TensorValue(max_k_tensor)

    if isinstance(temperature, float):
        temperature_tensor = ops.broadcast_to(
            ops.constant(temperature, dtype=DType.float32, device=device),
            [batch_size],
        )
    else:
        temperature_tensor = TensorValue(temperature)
        if temperature_tensor.shape[0] != batch_size:
            raise ValueError(
                f"temperature tensor shape {temperature_tensor.shape} does not match batch_size {batch_size}"
            )

    # Handle top_p parameter - can be scalar or tensor
    min_top_p_tensor = min_top_p
    if isinstance(top_p, float | int):
        if top_p <= 0 or top_p > 1:
            raise ValueError(f"expected top_p to be in (0, 1], got {top_p}")
        top_p_tensor = ops.broadcast_to(
            ops.constant(top_p, dtype=DType.float32, device=device),
            [batch_size],
        )
        # Set min_top_p to the scalar value if provided, otherwise use top_p
        min_top_p_value = min_top_p if min_top_p is not None else top_p
        assert isinstance(min_top_p_value, float | int)
        min_top_p_tensor = ops.constant(
            min_top_p_value, dtype=DType.float32, device=DeviceRef.CPU()
        )
    else:
        top_p_tensor = TensorValue(top_p)
        if top_p_tensor.shape[0] != batch_size:
            raise ValueError(
                f"top_p tensor shape {top_p_tensor.shape} does not match batch_size {batch_size}"
            )
        # When top_p is a tensor, min_top_p must be provided
        if min_top_p is None:
            raise ValueError(
                "min_top_p must be explicitly set when top_p is a tensor"
            )
        min_top_p_tensor = TensorValue(min_top_p)

    # Handle seed parameter - can be scalar or tensor
    if isinstance(seed, int):
        seed_tensor = ops.broadcast_to(
            ops.constant(seed, dtype=DType.uint64, device=device), [batch_size]
        )
    else:
        seed_tensor = TensorValue(seed)
        if seed_tensor.shape[0] != batch_size:
            raise ValueError(
                f"seed tensor shape {seed_tensor.shape} does not match batch_size {batch_size}"
            )

    batch_shape = logits.shape[:-1]

    return ops.custom(
        "sampler.fused_token_sampling",
        device=logits.device,
        values=[
            top_k_tensor,
            max_k_tensor,
            temperature_tensor,
            top_p_tensor,
            min_top_p_tensor,
            seed_tensor,
            logits,
        ],
        out_types=[
            TensorType(
                dtype=DType.int64, shape=batch_shape + [1], device=device
            )
        ],
    )[0].tensor


def sgmv_kernel(  # noqa: ANN201
    input: TensorValue,
    lora: TensorValue,
    lora_ids: TensorValue,
    lora_ranks: TensorValue,
    input_row_offsets: TensorValue,
    max_lora_seq_len: int,
    lora_end_idx: TensorValue | None = None,
    bias: TensorValue | None = None,
):
    """
    Performs the SGMV kernel for LoRA. This is LoRA agnostic, meaning that
    we can perform LoRA A or B from this kernel call.
    Args:
        input: The input tensor
        lora: The LoRA tensor
        lora_ids: Ids of the LoRAs used for each sequence
        lora_ranks: The ranks of the LoRAs ihn the batch
        input_row_offsets: The sequence offsets that use LoRA
        max_lora_seq_len: The maximum sequence length of any given LoRA in the batch
        bias: The LoRA bias

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    if input.rank != 2:
        raise ValueError(f"expected input to have rank 2, was {input.rank}")

    if lora.rank != 3:
        raise ValueError(f"expected lora to have rank 3, was {lora.rank}")

    if input.dtype != lora.dtype:
        raise ValueError(
            f"expected input and lora to have the same dtype, but got "
            f"{input.dtype} and {lora.dtype}, respectively."
        )

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            f"expected input_row_offsets to have dtype uint32, was {input_row_offsets.dtype}"
        )

    M = input.shape[0] if not lora_end_idx else lora_end_idx.shape[0]

    out = ops.custom(
        "mo.lora_sgmv.ragged",
        device=input.device,
        values=[
            input,
            lora,
            input_row_offsets,
            lora_ids,
            ops.constant(
                max_lora_seq_len,
                DType.uint32,
                device=DeviceRef.CPU(),
            ),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=[M, lora.shape[1]],
                device=input.device,
            ),
        ],
    )[0].tensor

    return out


def sgmv_lora_kernel(
    input: TensorValue,
    lora_a: TensorValue,
    lora_b: TensorValue,
    lora_ids: TensorValue,
    lora_ranks: TensorValue,
    grouped_row_offsets: TensorValue,
    lora_end_idx: TensorValue,
    max_lora_seq_len: int,
    bias: TensorValue | None = None,
) -> TensorValue:
    """
    Computes the SGMV LoRA kernel for some number of LoRAs A and B given the input.

    out = Wx + xAB

    SGMV can be explained by two independent kernels:
        - shrink -> shrinks high-dimensional tensor to low-rank tensor
        - expand -> expands low-rank tensor to high-dimensional tensor

    where v = [0, ...] and y = (some output tensor)

    SGMV-shrink:
        v += xA

    SGMV-expand:
        y += vB

    Args:
        input: The input tensor
        lora_a: The LoRA tensor for A
        lora_b: The LoRA tensor for B
        lora_ids: Ids of the LoRAs used for each sequence
        lora_ranks: The ranks of the LoRAs ihn the batch
        grouped_row_offsets: The grouped sequence offsets that use LoRA
        max_lora_seq_len: The maximum sequence length of any given LoRA in the batch
        bias: The LoRA bias

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    if input.rank != 2:
        raise ValueError(f"expected input to have rank 2, was {input.rank}")

    if lora_a.rank != 3:
        raise ValueError(f"expected lora_a to have rank 3, was {lora_a.rank}")

    if lora_b.rank != 3:
        raise ValueError(f"expected lora_b to have rank 3, was {lora_b.rank}")

    if input.dtype != lora_a.dtype:
        raise ValueError(
            f"expected input and lora_a to have the same dtype, but got "
            f"{input.dtype} and {lora_a.dtype}, respectively."
        )

    if input.dtype != lora_b.dtype:
        raise ValueError(
            f"expected input and lora_b to have the same dtype, but got "
            f"{input.dtype} and {lora_b.dtype}, respectively."
        )

    if grouped_row_offsets.dtype != DType.uint32:
        raise ValueError(
            f"expected grouped_row_offsets to have dtype uint32, was {grouped_row_offsets.dtype}"
        )

    v = sgmv_kernel(
        input,
        lora_a,
        lora_ids,
        lora_ranks,
        grouped_row_offsets,
        max_lora_seq_len,
        lora_end_idx,
        bias,
    )

    output = sgmv_kernel(
        v,
        lora_b,
        lora_ids,
        lora_ranks,
        grouped_row_offsets,
        max_lora_seq_len,
        lora_end_idx,
        bias,
    )

    return output


def sgmv_lora_qkv_shrink(
    input: TensorValue,
    lora_a: TensorValue,
    lora_ids: TensorValue,
    lora_grouped_offsets: TensorValue,
    lora_end_idx: TensorValue,
    max_lora_seq_len: int,
    max_rank: int,
) -> TensorValue:
    """LoRA shrink grouped matmul with planar Q/K/V output.

    Performs the LoRA 'shrink' operation for routed tokens using SGMV (segmented
    grouped matrix-vector multiplication). Computes `[M, K] @ [G, 3*rank, K]^T`
    per active LoRA adapter, then permutes the flat `[M, 3*rank]` result into a
    planar layout `[3, M, rank]` representing separate Q, K, V projections.

    Args:
        input: Routed activation matrix with shape (M, K), where M is the total
            number of tokens and K is the hidden dimension.
        lora_a: Shrink weights for all LoRA adapters, shape (G, 3*rank, K) where
            G is the number of adapters and rank is the LoRA rank.
        lora_ids: Expert/adapter indices for each active group, shape (num_active,).
            Values in range [0, G). May use -1 to indicate inactive slots.
        lora_grouped_offsets: Inclusive prefix sums of tokens per active adapter,
            shape (num_active + 1,). Defines per-adapter [start, end) ranges in
            input. Must be non-decreasing with offsets[0] == 0.
        max_lora_seq_len: Upper bound on tokens for any active adapter. Used for
            kernel tuning and memory allocation.
        max_rank: The maximum LoRA rank, determines output shape.

    Returns:
        Output tensor with planar Q/K/V layout, shape (3, M, max_rank).

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    if input.rank != 2:
        raise ValueError(f"expected input to have rank 2, was {input.rank}")

    if lora_a.rank != 3:
        raise ValueError(f"expected lora_a to have rank 3, was {lora_a.rank}")

    if input.dtype != lora_a.dtype:
        raise ValueError(
            f"expected input and lora_a to have the same dtype, but got "
            f"{input.dtype} and {lora_a.dtype}, respectively."
        )

    if lora_grouped_offsets.dtype != DType.uint32:
        raise ValueError(
            f"expected lora_grouped_offsets to have dtype uint32, was {lora_grouped_offsets.dtype}"
        )

    return ops.custom(
        "mo.lora_sgmv.qkv_shrink.ragged",
        device=input.device,
        values=[
            input,
            lora_a,
            lora_grouped_offsets,
            lora_ids,
            ops.constant(
                max_lora_seq_len,
                DType.uint32,
                device=DeviceRef.CPU(),
            ),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=[3, lora_end_idx.shape[0], max_rank],
                device=input.device,
            ),
        ],
    )[0].tensor


def sgmv_qkv_lora_kernel(
    input: TensorValue,
    lora_a: TensorValue,
    lora_b_q: TensorValue,
    lora_b_kv: TensorValue,
    lora_ids: TensorValue,
    lora_ranks: TensorValue,
    input_row_offsets: TensorValue,
    lora_grouped_offsets: TensorValue,
    lora_end_idx: TensorValue,
    batch_seq_len: TensorValue,
    lora_ids_kv: TensorValue,
    lora_grouped_offsets_kv: TensorValue,
    kv_collection: PagedCacheValues,
    kv_params: KVCacheParams,
    layer_idx: TensorValue,
    max_lora_seq_len: int,
    max_rank: int,
    bias: TensorValue | None = None,
) -> TensorValue:
    """
    Computes the SGMV QKV LoRA kernel for Q, K, V projections with LoRA.

    Args:
        input: The input tensor.
        lora_a: The LoRA A tensor.
        lora_b_q: The LoRA B tensor for Q projection.
        lora_b_kv: The LoRA B tensor for K and V projections (stacked).
        lora_ids: IDs of the LoRAs used for each sequence.
        lora_ranks: The ranks of the LoRAs in the batch.
        input_row_offsets: The sequence offsets that use LoRA.
        lora_grouped_offsets: Grouped offsets for LoRA sequences.
        lora_end_idx: End index of LoRA tokens in the batch.
        batch_seq_len: Total sequence length of the batch.
        lora_ids_kv: LoRA IDs for KV projections (with offset for V portion).
        lora_grouped_offsets_kv: Grouped offsets for KV LoRA sequences.
        kv_collection: The KV cache.
        kv_params: The KV params.
        layer_idx: The layer index to retrieve the KV cache.
        max_lora_seq_len: The maximum sequence length of any given LoRA in the batch.
        max_rank: The maximum rank for the LoRAs.
        bias: Optional LoRA bias.

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    if input.rank != 2:
        raise ValueError(f"expected input to have rank 2, was {input.rank}")

    if lora_a.rank != 3:
        raise ValueError(f"expected lora_a to have rank 3, was {lora_a.rank}")

    if lora_b_q.rank != 3:
        raise ValueError(
            f"expected lora_b_q to have rank 3, was {lora_b_q.rank}"
        )

    if lora_b_kv.rank != 3:
        raise ValueError(
            f"expected lora_b_kv to have rank 3, was {lora_b_kv.rank}"
        )

    if input.dtype != lora_a.dtype:
        raise ValueError(
            f"expected input and lora_a to have the same dtype, but got "
            f"{input.dtype} and {lora_a.dtype}, respectively."
        )

    if input.dtype != lora_b_q.dtype:
        raise ValueError(
            f"expected input and lora_b_q to have the same dtype, but got "
            f"{input.dtype} and {lora_b_q.dtype}, respectively."
        )

    if input.dtype != lora_b_kv.dtype:
        raise ValueError(
            f"expected input and lora_b_kv to have the same dtype, but got "
            f"{input.dtype} and {lora_b_kv.dtype}, respectively."
        )

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            f"expected input_row_offsets to have dtype uint32, was {input_row_offsets.dtype}"
        )

    if lora_grouped_offsets.dtype != DType.uint32:
        raise ValueError(
            f"expected lora_grouped_offsets to have dtype uint32, was {lora_grouped_offsets.dtype}"
        )

    if lora_grouped_offsets_kv.dtype != DType.uint32:
        raise ValueError(
            f"expected lora_grouped_offsets_kv to have dtype uint32, was {lora_grouped_offsets_kv.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(
            f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"unsupported cache strategy for sgmv_qkv_lora_kernel: {kv_params.cache_strategy}"
        )

    if kv_params.page_size is None:
        raise ValueError("expected kv_params.page_size to be set")

    # shrink GMM:      [M, K] @ [G, 3*N, K]     // unchanged
    # transpose:       [M, 3, N] => [3, M, N]   // shall be fused into above
    v_qkv = sgmv_lora_qkv_shrink(
        input=input,
        lora_a=lora_a,
        lora_ids=lora_ids,
        lora_grouped_offsets=lora_grouped_offsets,
        lora_end_idx=lora_end_idx,
        max_lora_seq_len=max_lora_seq_len,
        max_rank=max_rank,
    )

    # slice for Q:     [0, M, N] (not materialized)
    # slice for KV:    [1:,M, N] (not materialized)
    # reshape and slices get fused into the input of the
    # grouped-matmuls.
    v_qkv = ops.reshape(v_qkv, [3 * lora_end_idx.shape[0], -1])

    # expand GMM-Q:    [M, N]  @ [G, Qdim, N]
    v_q = v_qkv[: lora_end_idx.shape[0], :]
    q_out = sgmv_kernel(
        v_q,
        lora_b_q,
        lora_ids,
        lora_ranks,
        lora_grouped_offsets,
        max_lora_seq_len,
        lora_end_idx=lora_end_idx,
        bias=bias,
    )

    v_kv = v_qkv[lora_end_idx.shape[0] :, :]
    # expand GMM-KV:   [2M, N] @ [2G, KVdim, N] // KV stacked in dim 0
    kv_out = sgmv_kernel(
        v_kv,
        lora_b_kv,
        lora_ids_kv,
        lora_ranks,
        lora_grouped_offsets_kv,
        max_lora_seq_len,
        bias=bias,
    )

    # write to cache:  write [2M, KVdim] directly w/o transforming to [M, 2*KVdim]
    kv_cache_ragged_2m_iadd(
        kv_params=kv_params,
        a=kv_out,
        kv_collection=kv_collection,
        input_row_offsets=input_row_offsets,
        lora_end_idx=lora_end_idx,
        batch_seq_len=batch_seq_len,
        layer_idx=layer_idx,
    )

    return q_out


def kv_cache_ragged_2m_iadd(
    kv_params: KVCacheParams,
    a: TensorValue,
    kv_collection: PagedCacheValues,
    input_row_offsets: TensorValue,
    lora_end_idx: TensorValue,
    batch_seq_len: TensorValue,
    layer_idx: TensorValue,
) -> None:
    """In-place add to paged KV cache with interleaved K/V layout.

    Performs an in-place addition of new key-value projections to paged KV cache.
    The input tensor `a` uses a "2M" layout where keys and values are interleaved:
    rows [0, m) contain keys and rows [m, 2m) contain values, where m is the number
    of tokens.

    Args:
        kv_params: KV cache configuration parameters. Must have cache_strategy
            set to PAGED and page_size must be defined.
        a: Input tensor with interleaved K/V data, shape (2*m, hidden_size) where
            m is the number of tokens. Rows [0, m) are keys, rows [m, 2m) are values.
        kv_collection: The paged KV cache collection containing cache blocks,
            cache lengths, lookup tables, and max lengths tensors.
        input_row_offsets: Ragged tensor offsets indicating where each batch starts and ends
        lora_end_idx: End index of LoRA token portion. Marks the boundary between
            LoRA sequences and base model sequences in the batch.
        batch_seq_len: Total sequence length in the batch. Used for indexing
            into the value portion of `a`.
        layer_idx: The transformer layer index to update in the KV cache.

    Raises:
        ValueError: If `a` does not have rank 2.
        ValueError: If `input_row_offsets` does not have rank 1.
        ValueError: If `kv_params.cache_strategy` is not PAGED.
        ValueError: If `kv_params.page_size` is None.
    """

    if a.rank != 2:
        raise ValueError(f"Expected a to have rank 2 but got {a.rank}")

    if input_row_offsets.rank != 1:
        raise ValueError(
            f"Expected input_row_offsets to have rank 1 but got {input_row_offsets.rank}"
        )

    if kv_params.cache_strategy != "paged":
        raise ValueError(
            f"Expected kv_params to have cache strategy PAGED but got {kv_params.cache_strategy}"
        )

    if kv_params.page_size is None:
        raise ValueError("Expected kv_params.page_size to be set")

    ops.inplace_custom(
        "mo.kv_cache.ragged.paged.2m_iadd",
        device=input_row_offsets.device,
        values=[
            a,
            *kv_collection,
            input_row_offsets,
            lora_end_idx,
            batch_seq_len,
            layer_idx,
        ],
    )


def spatial_merge(
    input: TensorValue,
    grid_thw: TensorValue,
    hidden_size: int,
    merge_size: int,
) -> TensorValue:
    """Performs spatial merge operation on ragged input tensors.

    This operation merges spatial dimensions of input patches according to
    the grid dimensions specified in grid_thw.

    Args:
        input: Input tensor of shape [total_patches_in_grid, hidden_size]
        grid_thw: Grid dimensions tensor of shape [batch_size, 3] containing
            [t, h, w] for each batch item, where:
            - t: temporal/frame dimension
            - h: height dimension
            - w: width dimension
        hidden_size: Hidden dimension size
        merge_size: Size of spatial merge blocks (typically 2)

    Returns:
        Output tensor of shape [total_patches_in_grid, hidden_size]

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    if input.rank != 2:
        raise ValueError(f"expected input to have rank 2, got {input.rank}")

    if grid_thw.dtype != DType.int64:
        raise ValueError(
            f"expected grid_thw to have dtype int64, got {grid_thw.dtype}"
        )

    if grid_thw.rank != 2:
        raise ValueError(
            f"expected grid_thw to have rank 2, got {grid_thw.rank}"
        )
    if grid_thw.shape[1] != 3:
        raise ValueError(
            f"expected grid_thw.shape[1] to be 3, got {grid_thw.shape[1]}"
        )

    if input.shape[1] != hidden_size:
        raise ValueError(
            f"expected input.shape[1] to match hidden_size ({hidden_size}), "
            f"got {input.shape[1]}"
        )

    return ops.custom(
        "mo.spatial_merge",
        device=input.device,
        values=[
            input,
            grid_thw,
            ops.constant(
                hidden_size, dtype=DType.int32, device=DeviceRef.CPU()
            ),
            ops.constant(merge_size, dtype=DType.int32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=[input.shape[0], hidden_size],
                device=input.device,
            )
        ],
    )[0].tensor


def sliced_add(
    x: TensorValue,
    y: TensorValue,
    lora_end_idx: TensorValue,
) -> TensorValue:
    """Adds tensors x and y element-wise for rows < lora_end_idx, otherwise copies x.

    This is used for LoRA where only some sequences have LoRA applied.
    For rows in [0, lora_end_idx): c = x + y
    For rows in [lora_end_idx, batch_seq_len): c = x

    Args:
        x: First input tensor.
        y: Second input tensor.
        lora_end_idx: End index of LoRA token portion (rows to apply add).
    """
    return ops.custom(
        "mo.sliced.add.ragged",
        device=x.device,
        values=[
            x,
            y,
            lora_end_idx,
        ],
        out_types=[
            TensorType(
                dtype=x.dtype,
                shape=x.shape,
                device=x.device,
            )
        ],
    )[0].tensor


def kv_cache_copy_pages_d2h(
    device_kv_collection: PagedCacheValues,
    device_page_ids: TensorValue,
    host_kv_blocks: BufferValue,
    host_page_ids: TensorValue,
    layer_idx: int,
    device_ref: DeviceRef,
) -> None:
    """Copy KV cache pages from GPU to CPU for a single layer.

    Performs async GPU->CPU copy of specified pages for layer-wise KV cache
    offloading.

    Args:
        device_kv_collection: Source KV cache on GPU.
        device_page_ids: Source page IDs to read from GPU.
        host_kv_collection: Destination KV cache on CPU.
        host_page_ids: Destination page IDs to write to CPU.
            Must have same length as device_page_ids.
        layer_idx: Which layer to copy.
        device_ref: Device for the GPU context.
    """
    ops.inplace_custom(
        name="mo.kv_cache.copy_pages_d2h",
        device=device_ref,
        values=[
            device_kv_collection.kv_blocks,
            host_kv_blocks,
            device_page_ids,
            host_page_ids,
            ops.constant(layer_idx, DType.uint32, device=DeviceRef.CPU()),
        ],
    )


def sleep(duration_sec: BufferValue, device_ref: DeviceRef) -> None:
    """Sleep for the given duration in seconds.

    This kernel is supported on CPUs and GPUs. However, the timing may be completely
    inaccurate on AMD GPUs due to limitation of current time.sleep(...) impl.

    Args:
        duration_sec: The duration to sleep in seconds.
    """
    # FIXME(GEX-3080): Convert duration_sec to a 0-d scalar instead of 1-d buffer.
    # We currently use 1-d buffer to prevent sleep op from being DCE'd away.
    if duration_sec.shape.static_dims != [1]:
        raise ValueError(
            f"Expected duration_sec to have shape [1] but got {duration_sec.shape.static_dims}"
        )
    if duration_sec.dtype != DType.float64:
        raise ValueError(
            f"Expected duration_sec to have DType.float64 but got {duration_sec.dtype}"
        )
    if duration_sec.device != DeviceRef.CPU():
        raise ValueError(
            f"Expected duration_sec to be on cpu but got {duration_sec.device}"
        )

    ops.inplace_custom(
        "mo.sleep",
        device=device_ref,
        values=[duration_sec],
        out_types=[],
    )


def lmcache_offload(
    output: BufferValue,
    paged_cache: TensorValue,
    slot_mapping: TensorValue,
    start_token: TensorValue,
    end_token: TensorValue,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dim: int,
    device_ref: DeviceRef,
) -> None:
    """Offload KV cache data from paged format to external contiguous format.

    Used by LMCache connector to copy KV data from MAX's paged cache layout
    to LMCache's contiguous KV_2LTD format for external storage.

    Args:
        output: Output buffer [kv_dim, num_layers, num_tokens, hidden_dim]
            where hidden_dim = num_kv_heads * head_dim.
        paged_cache: Input tensor [total_num_blocks, kv_dim, num_layers, page_size, num_kv_heads, head_dim].
        slot_mapping: Token to slot mapping [total_tokens].
        start_token: Starting token index scalar [1].
        end_token: Ending token index (exclusive) scalar [1].
        page_size: Number of tokens per page in the paged cache.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension of each attention head.
        kv_dim: KV dimension (2 for standard K/V, 1 for MLA).
        device_ref: Device reference for the operation.
    """
    parameters: dict[str, int | str | DType] = {
        "page_size": page_size,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "kv_dim": kv_dim,
    }
    ops.inplace_custom(
        name="mo.lmcache.offload",
        device=device_ref,
        values=[output, paged_cache, slot_mapping, start_token, end_token],
        parameters=parameters,
    )


def lmcache_onload(
    paged_cache: BufferValue,
    input_tensor: TensorValue,
    slot_mapping: TensorValue,
    start_token: TensorValue,
    end_token: TensorValue,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    kv_dim: int,
    device_ref: DeviceRef,
) -> None:
    """Onload KV cache data from external contiguous format to paged format.

    Used by LMCache connector to copy KV data from LMCache's contiguous
    KV_2LTD format into MAX's paged cache layout.

    Args:
        paged_cache: Output buffer [total_num_blocks, kv_dim, num_layers, page_size, num_kv_heads, head_dim].
        input_tensor: Input tensor [kv_dim, num_layers, num_tokens, hidden_dim]
            where hidden_dim = num_kv_heads * head_dim.
        slot_mapping: Token to slot mapping [total_tokens].
        start_token: Starting token index scalar [1].
        end_token: Ending token index (exclusive) scalar [1].
        page_size: Number of tokens per page in the paged cache.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension of each attention head.
        kv_dim: KV dimension (2 for standard K/V, 1 for MLA).
        device_ref: Device reference for the operation.
    """
    parameters: dict[str, int | str | DType] = {
        "page_size": page_size,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "kv_dim": kv_dim,
    }
    ops.inplace_custom(
        name="mo.lmcache.onload",
        device=device_ref,
        values=[
            paged_cache,
            input_tensor,
            slot_mapping,
            start_token,
            end_token,
        ],
        parameters=parameters,
    )
