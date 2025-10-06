# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
    TensorType,
    TensorValue,
    TensorValueLike,
    Value,
    ops,
)
from max.graph.ops.quantized import repack_gguf_quantized_weights
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.nn.float8_config import (
    Float8InputScaleSpec,
    Float8WeightScaleSpec,
)

from .attention.mask_config import (
    AttentionMaskVariant,
    MHAMaskConfig,
    MHAMaskVariant,
    PositionalEncodingVariant,
)
from .kv_cache import KVCacheParams, KVCacheStrategy, PagedCacheValues

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


def fused_qkv_ragged_matmul(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    bias: TensorValue | None = None,
) -> TensorValue:
    """Computes fused query, key, and value projections with ragged input.

    `input` and `input_row_offsets` are used together to implement the ragged
    tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`

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

    if kv_params.cache_strategy not in {
        KVCacheStrategy.PAGED,
    }:
        raise ValueError(
            f"unsupported cache strategy for fused_qkv_ragged_matmul: {kv_params.cache_strategy}"
        )

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    op_name = f"mo.fused_qkv_matmul.ragged.{cache_strategy_str}"
    values = [input, input_row_offsets, wqkv, *kv_collection, layer_idx]

    if bias:
        op_name += ".bias"
        values.append(bias)

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
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
) -> TensorValue:
    """Computes fused query, key, and value projections with ragged input.

    `input` and `input_row_offsets` are used together to implement the ragged
    tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`

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
    if not all(
        t.device == input.device
        for t in [wqkv, input_row_offsets, input_scale, weight_scale]
    ):
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

    # for per-tensor quantization, the scale is a scalar. We view it as a 1x1
    # rank-2 tensor so that we can use the same kernel for per-tensor and
    # per-channel quantization.
    if input_scale.shape in [[], [1]]:
        input_scale = input_scale.reshape([1, 1])

    if weight_scale.shape in [[], [1]]:
        weight_scale = weight_scale.reshape([1, 1])

    assert kv_params.page_size is not None
    parameters: dict[str, int | str | DType] = {
        "kv_type": kv_params.dtype,
    }

    op_name = "mo.fused_qkv_matmul.ragged.paged.scale"

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=[
            input,
            input_row_offsets,
            wqkv,
            input_scale,
            weight_scale,
            *kv_collection,
            layer_idx,
        ],
        out_types=[
            TensorType(
                dtype=DType.bfloat16,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
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

    if kv_params.cache_strategy not in {KVCacheStrategy.PAGED}:
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

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
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

    if kv_params.cache_strategy not in {
        KVCacheStrategy.PAGED,
    }:
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

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()

    args = [input, input_row_offsets, wqkv, *kv_collection, layer_idx]
    if bias:
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

    if kv_params.cache_strategy != KVCacheStrategy.PAGED:
        raise ValueError(
            f"unsupported cache strategy for matmul_kv_cache_ragged: {kv_params.cache_strategy}"
        )

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
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

    if kv_params.cache_strategy != KVCacheStrategy.PAGED:
        raise ValueError(
            f"unsupported cache strategy for matmul_kv_cache_ragged: {kv_params.cache_strategy}"
        )

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
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

    if kv_params.cache_strategy not in {
        KVCacheStrategy.PAGED,
    }:
        raise ValueError(
            f"unsupported cache strategy for fused_qk_ragged_rope: {kv_params.cache_strategy}"
        )

    parameters: dict[str, bool | int | str | DType] = {
        "interleaved": interleaved,
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

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()

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

    if kv_params.cache_strategy not in {
        KVCacheStrategy.PAGED,
    }:
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

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
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
        input_row_offsets: Tensor of shape [batch_size + 1] with dtype uint32.
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

    if kv_params.cache_strategy is not KVCacheStrategy.PAGED:
        raise ValueError(
            f"unsupported cache strategy for flash_attention_ragged: {kv_params.cache_strategy}"
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
    prev_output: TensorValue | None = None,
    prev_softmax_info: TensorValue | None = None,
) -> tuple[TensorValue, TensorValue]:
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
        prev_output: Optional. Previous output tensor
        prev_softmax_info: Optional. Previous softmax info tensor

    Returns:
        A tuple of two tensors:
            - The first tensor is the output tensor for this iteration
            - The second tensor is the softmax info tensor for this iteration
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

    if kv_params.cache_strategy is not KVCacheStrategy.PAGED:
        raise ValueError(
            f"unsupported cache strategy for flare_mla_prefill_ragged: {kv_params.cache_strategy}"
        )

    assert kv_params.page_size is not None
    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    parameters: dict[str, int | str | DType] = {
        "mask_str": mha_mask_config.attention_mask_variant.value,
        "score_mod_str": mha_mask_config.positional_encoding_variant.value,
    }

    is_init_str = ".init" if prev_output is None else ""
    op_name = f"mo.mla.prefill{is_init_str}.ragged.paged"

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
    if prev_output is not None:
        input_values.append(prev_output)
    if prev_softmax_info is not None:
        input_values.append(prev_softmax_info)

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
            ),
            TensorType(
                dtype=DType.float32,
                shape=[
                    input.shape[0],
                    input.shape[1],
                    2,
                ],
                device=input.device,
            ),
        ],
        parameters=parameters,
    )

    return results[0].tensor, results[1].tensor


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

    if kv_params.cache_strategy is not KVCacheStrategy.PAGED:
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
                shape=[max_chunks, input_row_offsets.shape[0] - 1],
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

    if kv_params.cache_strategy is not KVCacheStrategy.PAGED:
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

    if kv_params.cache_strategy not in {
        KVCacheStrategy.PAGED,
    }:
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

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
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
    in the front of the tensor, and we're only adding to the last requests in the batch
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

    if kv_params.cache_strategy != KVCacheStrategy.PAGED:
        raise ValueError(
            f"Expected kv_params to have cache strategy PAGED but got {kv_params.cache_strategy}"
        )

    if kv_params.page_size is None:
        raise ValueError("Expected kv_params.page_size to be set")

    # slice input_row_offests to the batch offset
    input_row_offsets = ops.slice_tensor(
        input_row_offsets,
        [(slice(batch_offset, None), Dim("input_row_offsets_slice_len"))],
    )

    op_name = (
        f"mo.kv_cache.ragged.{kv_params.cache_strategy.kernel_substring()}.radd"
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
    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
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

    if gamma.dtype != kv_params.dtype:
        raise TypeError(
            f"expected gamma dtype {gamma.dtype} to match KV dtype {kv_params.dtype}"
        )

    parameters: dict[str, int | str | DType | bool] = {
        "multiply_before_cast": multiply_before_cast,
        "per_head_norm": per_head_norm,
    }
    if kv_params.cache_strategy == KVCacheStrategy.PAGED:
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

    Returns:
        The result of the matmul operation.
    """

    if out_type != DType.bfloat16:
        raise ValueError(
            "Only bfloat16 is supported for batched blockwise scaled matmul"
        )

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
    else:
        raise ValueError("unsupported FP8 scaling granularity")

    # if (a_scales.shape[1] * a_scales.dtype.size_in_bytes) % 16 != 0:
    #     raise ValueError(
    #         "TMA expects total_num_tokens to be divisible by 16 bytes"
    #     )

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

    if a.rank != 3 or b.rank != 3 or a_scales.rank != 3 or b_scales.rank != 3:
        raise ValueError("All arguments must be rank 3 tensors")

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

    if out_type != DType.bfloat16:
        raise ValueError(
            "Only bfloat16 is supported for batched blockwise scaled matmul"
        )

    if a_scales.dtype != b_scales.dtype or a_scales.dtype != DType.float32:
        raise TypeError(
            f"a_scales and b_scales dtypes must be float32, but got {a_scales.dtype}, {b_scales.dtype}"
        )

    # if (a_scales.shape[1] * a_scales.dtype.size_in_bytes) % 16 != 0:
    #     raise ValueError(
    #         "TMA expects total_num_tokens to be divisible by 16 bytes"
    #     )

    if input_scale_spec.is_block and weight_scale_spec.is_block:
        # a_scale is of shape [batch_size, ceildiv(K // BLOCK_SIZE), M-padded]
        # b_scale is of shape [batch_size, ceildiv(N // BLOCK_SIZE), ceildiv(K // BLOCK_SIZE)]
        if a_scales.shape[0] != b_scales.shape[0]:
            raise ValueError(
                "both a_scales and b_scales must have the same shape on the batch dimension"
            )

    else:
        raise ValueError("unsupported FP8 scaling granularity")

    if (a.dtype != b.dtype) or (a_scales.dtype != b_scales.dtype):
        raise TypeError(
            f"a and b dtypes {a.dtype}, {b.dtype} must match, "
            f"as do a and b scales dtypes {a_scales.dtype}, {b_scales.dtype}"
        )

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

    group_size = (
        group_size_or_per_token
        if group_size_or_per_token != -1
        else input.shape[1]
    )

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
            "group_size_or_per_token": group_size_or_per_token,
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
        if not (input_scale_spec.is_block and weight_scale_spec.is_block):
            raise ValueError(
                "both input and weight must be blockwise scaled for blockwise scaling"
            )

        if a_scales.dtype != b_scales.dtype or a_scales.dtype != DType.float32:
            raise TypeError(
                f"a_scales and b_scales dtypes must be float32, but got {a_scales.dtype}, {b_scales.dtype}"
            )

        # a_scale is of shape [ceildiv(K // BLOCK_SIZE), M-padded]
        # b_scale is of shape [ceildiv(N // BLOCK_SIZE), ceildiv(K // BLOCK_SIZE)]
        if a_scales.shape[0] != b_scales.shape[1]:
            raise ValueError(
                "both a_scales and b_scales must have the same shape on the K dimension"
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


def topk_fused_sampling(
    logits: TensorValue,
    top_k: TensorValueLike,
    *,
    temperature: TensorValueLike = 1.0,
    max_k: TensorValueLike | None = None,
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
        if top_k <= 0 or top_k > 256:
            raise ValueError(
                f"top_k must be greater than 0 and less than or equal to 256, got {top_k}"
            )

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
    if isinstance(top_p, float | int):
        if top_p <= 0 or top_p > 1:
            raise ValueError(f"expected top_p to be in (0, 1], got {top_p}")
        top_p_tensor = ops.broadcast_to(
            ops.constant(top_p, dtype=DType.float32, device=device),
            [batch_size],
        )
    else:
        top_p_tensor = TensorValue(top_p)
        if top_p_tensor.shape[0] != batch_size:
            raise ValueError(
                f"top_p tensor shape {top_p_tensor.shape} does not match batch_size {batch_size}"
            )

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
    """
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
                shape=[input.shape[0], lora.shape[1]],
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
    """
    v = sgmv_kernel(
        input,
        lora_a,
        lora_ids,
        lora_ranks,
        grouped_row_offsets,
        max_lora_seq_len,
        bias,
    )

    output = sgmv_kernel(
        v,
        lora_b,
        lora_ids,
        lora_ranks,
        grouped_row_offsets,
        max_lora_seq_len,
        bias,
    )

    return output


def sgmv_qkv_lora_kernel(
    input: TensorValue,
    lora_a: TensorValue,
    lora_b: TensorValue,
    lora_ids: TensorValue,
    lora_ranks: TensorValue,
    input_row_offsets: TensorValue,
    lora_grouped_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    kv_params: KVCacheParams,
    layer_idx: TensorValue,
    max_lora_seq_len: int,
    max_rank: int,
    q_dim: int,
    kv_dim: int,
    bias: TensorValue | None = None,
) -> TensorValue:
    """
    Computes the SGMV QKV LoRA kernel for Q, K, V projections with LoRA.

    Args:
        input: The input tensor
        lora_a: The LoRA tensor for A
        lora_b: The LoRA tensor for B
        lora_ids: Ids of the LoRAs used for each sequence
        lora_ranks: The ranks of the LoRAs ihn the batch
        input_row_offsets: The sequence offsets that use LoRA
        kv_collection: The KV cache
        kv_params: The KV params
        layer_idx: The layer index to retrieve the KV cache
        max_lora_seq_len: The maximum sequence length of any given LoRA in the batch
        max_rank: The maximum rank for the LoRAs
        q_dim: The q dimension
        kv_dim: The kv dimension
        bias: Optional LoRA bias
    """
    if kv_params.cache_strategy != KVCacheStrategy.PAGED:
        raise ValueError("KV cache SGMV only supports Paged KV cache.")

    assert kv_params.page_size is not None

    v_qkv = sgmv_kernel(
        input,
        lora_a,
        lora_ids,
        lora_ranks,
        lora_grouped_offsets,
        max_lora_seq_len,
        bias,
    )

    q_out = sgmv_kernel(
        v_qkv[:, :max_rank],
        lora_b[:, :q_dim, :],
        lora_ids,
        lora_ranks,
        lora_grouped_offsets,
        max_lora_seq_len,
        bias,
    )

    k_out = sgmv_kernel(
        v_qkv[:, max_rank:],
        lora_b[:, q_dim : q_dim + kv_dim, :],
        lora_ids,
        lora_ranks,
        lora_grouped_offsets,
        max_lora_seq_len,
        bias,
    )

    v_out = sgmv_kernel(
        v_qkv[:, 2 * max_rank :],
        lora_b[:, q_dim + kv_dim :, :],
        lora_ids,
        lora_ranks,
        lora_grouped_offsets,
        max_lora_seq_len,
        bias,
    )

    ops.inplace_custom(
        "mo.kv_cache.ragged.paged.radd",
        device=input.device,
        values=[
            ops.concat([k_out, v_out], axis=1),
            *kv_collection,
            input_row_offsets,
            ops.constant(0, DType.uint32, DeviceRef.CPU()),
            layer_idx,
        ],
    )

    return q_out
