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

from dataclasses import dataclass
from enum import Enum
from typing import Optional, cast

import numpy as np
from max.dtype import DType
from max.graph import Dim, TensorType, TensorValue, TensorValueLike, ops
from max.graph.ops.quantized import repack_gguf_quantized_weights
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheCollection,
    PagedKVCacheCollectionFA3Fallback,
)


def fused_qkv_ragged_matmul(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection | PagedKVCacheCollection,
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
        msg = (
            "expected input and wqkv to have the same dtype, but got"
            f" {input.dtype} and {wqkv.dtype}, respectively."
        )
        raise ValueError(msg)

    input_rank_expected = 2
    if input.rank != input_rank_expected:
        msg = f"expected input to have rank {input_rank_expected}, was {input.rank}"
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = (
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy not in {
        KVCacheStrategy.CONTINUOUS,
        KVCacheStrategy.PAGED,
        KVCacheStrategy.PAGED_FA3_FALLBACK,
    }:
        msg = f"unsupported cache strategy for fused_qkv_ragged_matmul: {kv_params.cache_strategy}"
        raise ValueError(msg)

    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
    }
    if kv_params.cache_strategy in {
        KVCacheStrategy.PAGED,
        KVCacheStrategy.PAGED_FA3_FALLBACK,
    }:
        assert kv_params.page_size is not None
        parameters["page_size"] = int(kv_params.page_size)

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()

    if bias:
        op_name = f"mo.fused_qkv_matmul.ragged.{cache_strategy_str}.bias"

        return ops.inplace_custom(
            op_name,
            values=[
                input,
                input_row_offsets,
                wqkv,
                kv_collection,
                layer_idx,
                bias,
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

    op_name = f"mo.fused_qkv_matmul.ragged.{cache_strategy_str}"

    return ops.inplace_custom(
        op_name,
        values=[input, input_row_offsets, wqkv, kv_collection, layer_idx],
        out_types=[
            TensorType(
                dtype=input.dtype,
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
    kv_collection: ContinuousBatchingKVCacheCollection | PagedKVCacheCollection,
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
        msg = f"expected input to have rank {input_rank_expected}, was {input.rank}"
        raise ValueError(msg)

    if input.dtype != DType.float32:
        msg = f"expected input to have dtype float32, was {input.dtype}"
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = (
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy not in {
        KVCacheStrategy.CONTINUOUS,
    }:
        msg = f"unsupported cache strategy for fused_qkv_ragged_matmul: {kv_params.cache_strategy}"
        raise ValueError(msg)

    if (
        not quantization_encoding_q.is_gguf
        or not quantization_encoding_k.is_gguf
        or not quantization_encoding_v.is_gguf
    ):
        raise ValueError(
            f"expected quantization_encoding_q, quantization_encoding_k, and quantization_encoding_v to be gguf, was {quantization_encoding_q}, {quantization_encoding_k}, and {quantization_encoding_v}"
        )

    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "quantization_encoding_q": quantization_encoding_q.name,
        "quantization_encoding_k": quantization_encoding_k.name,
        "quantization_encoding_v": quantization_encoding_v.name,
    }

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    return ops.inplace_custom(
        name=f"mo.unfused_qkv_matmul.ragged.{cache_strategy_str}.gguf_quantized",
        values=[
            input,
            input_row_offsets,
            repack_gguf_quantized_weights(q_weight, quantization_encoding_q),
            repack_gguf_quantized_weights(k_weight, quantization_encoding_k),
            repack_gguf_quantized_weights(v_weight, quantization_encoding_v),
            kv_collection,
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
    kv_collection: ContinuousBatchingKVCacheCollection | PagedKVCacheCollection,
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
        msg = f"expected input to have rank {input_rank_expected}, was {input.rank}"
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = (
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy not in {
        KVCacheStrategy.CONTINUOUS,
        KVCacheStrategy.PAGED,
    }:
        msg = f"unsupported cache strategy for fused_qkv_ragged_matmul: {kv_params.cache_strategy}"
        raise ValueError(msg)

    # In the group-wise quantization scheme, every `group_size` quantized weights
    # share the same scale. If `has_zp` is `True`, there is also a group-wise zero
    # point that need to be substracted from the quantized weights.
    # Since the new extensibility API doesn't currently support `bool` type parameters,
    # we pass `has_zp` as an interger (`has_zp_int`).
    # For GPTQ, `has_zp_int` will always be 0.
    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "group_size": quantization_config.group_size,
        "has_zp_int": 0,
    }
    if perm_idx:
        input = ops.gather(input, TensorValue(perm_idx), axis=1)
        wqkv = ops.custom(
            "GPTQ_gpu_repack_b4_g128_desc_act",
            list((wqkv, perm_idx)),
            out_types=[
                TensorType(
                    DType.uint8,
                    ((wqkv.shape[1], wqkv.shape[0])),
                )
            ],
        )[0].tensor
    else:
        wqkv = ops.custom(
            "GPTQ_gpu_repack_b4_g128",
            list((wqkv,)),
            out_types=[
                TensorType(
                    DType.uint8,
                    ((wqkv.shape[1], wqkv.shape[0])),
                )
            ],
        )[0].tensor

    if kv_params.cache_strategy == KVCacheStrategy.PAGED:
        assert kv_params.page_size is not None
        parameters["page_size"] = int(kv_params.page_size)

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()

    args = [input, input_row_offsets, wqkv, kv_collection, layer_idx]
    if bias:
        args.append(bias)
        bias_name_str = "bias."
    else:
        bias_name_str = ""

    op_name = f"mo.fused_qkv_matmul.ragged.{cache_strategy_str}.{bias_name_str}quantized"

    return ops.inplace_custom(
        op_name,
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


def fused_qkv_matmul(
    kv_params: KVCacheParams,
    input: TensorValue,
    wqkv: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: TensorValue,
    n_heads: int,
) -> TensorValue:
    """Computes fused query, key and value projections."""
    if input.dtype != wqkv.dtype:
        msg = (
            "expected input and wqkv to have the same dtype, but got"
            f" {input.dtype} and {wqkv.dtype}, respectively."
        )
        raise ValueError(msg)

    input_rank_expected = 3
    if input.rank != input_rank_expected:
        msg = f"expected input to have rank {input_rank_expected}, was {input.rank}"
        raise ValueError(msg)

    wqkv_rank_expected = 2
    if wqkv.rank != wqkv_rank_expected:
        msg = (
            f"expected wqkv to have rank {wqkv_rank_expected}, was {wqkv.rank}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy != KVCacheStrategy.CONTINUOUS:
        msg = f"unsupported cache strategy for fused_qkv_matmul: {kv_params.cache_strategy}"
        raise ValueError(msg)

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    op_name = f"mo.fused_qkv_matmul.padded.{cache_strategy_str}"

    return ops.inplace_custom(
        op_name,
        values=[input, wqkv, kv_collection, layer_idx],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
                device=input.device,
            )
        ],
        parameters={
            "num_heads": kv_params.n_kv_heads_per_device,
            "head_dim": kv_params.head_dim,
        },
    )[0].tensor


def matmul_kv_cache_ragged(
    kv_params: KVCacheParams,
    hidden_states: TensorValue,
    input_row_offsets: TensorValue,
    weight: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: int | np.integer,
) -> None:
    """Computes key and value projections with ragged input.

    `hidden_states` and `input_row_offsets` are used together to
    implement the ragged tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`
    """
    if hidden_states.dtype != weight.dtype:
        msg = (
            "expected hidden_states and weight to have the same dtype, but got"
            f" {hidden_states.dtype} and {weight.dtype}, respectively."
        )
        raise ValueError(msg)

    hidden_states_rank_expected = 2
    if hidden_states.rank != hidden_states_rank_expected:
        msg = (
            "expected hidden_states to have rank "
            f"{hidden_states_rank_expected}, was {hidden_states.rank}"
        )
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = (
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )
        raise ValueError(msg)

    if kv_params.cache_strategy != KVCacheStrategy.CONTINUOUS:
        msg = f"unsupported cache strategy for matmul_kv_cache_ragged: {kv_params.cache_strategy}"
        raise ValueError(msg)

    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
    }
    if kv_params.cache_strategy == KVCacheStrategy.PAGED:
        assert kv_params.page_size is not None
        parameters["page_size"] = kv_params.page_size

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    op_name = f"mo.kv_matmul.ragged.{cache_strategy_str}"

    ops.inplace_custom(
        name=op_name,
        values=[
            hidden_states,
            input_row_offsets,
            weight,
            kv_collection,
            ops.constant(layer_idx, DType.uint32),
        ],
        parameters=parameters,
    )


def matmul_k_cache_ragged(
    kv_params: KVCacheParams,
    hidden_states: TensorValue,
    input_row_offsets: TensorValue,
    weight: TensorValue,
    kv_collection: PagedKVCacheCollection,
    layer_idx: int | np.integer,
) -> None:
    """Computes key projections with ragged input.

    `hidden_states` and `input_row_offsets` are used together to
    implement the ragged tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`
    """
    if hidden_states.dtype != weight.dtype:
        msg = (
            "expected hidden_states and weight to have the same dtype, but got"
            f" {hidden_states.dtype} and {weight.dtype}, respectively."
        )
        raise ValueError(msg)

    hidden_states_rank_expected = 2
    if hidden_states.rank != hidden_states_rank_expected:
        msg = (
            "expected hidden_states to have rank "
            f"{hidden_states_rank_expected}, was {hidden_states.rank}"
        )
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = (
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )
        raise ValueError(msg)

    if kv_params.cache_strategy != KVCacheStrategy.PAGED:
        msg = f"unsupported cache strategy for matmul_kv_cache_ragged: {kv_params.cache_strategy}"
        raise ValueError(msg)

    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
    }
    if kv_params.cache_strategy == KVCacheStrategy.PAGED:
        assert kv_params.page_size is not None
        parameters["page_size"] = kv_params.page_size

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    op_name = f"mo.k_matmul.ragged.{cache_strategy_str}"

    ops.inplace_custom(
        name=op_name,
        values=[
            hidden_states,
            input_row_offsets,
            weight,
            kv_collection,
            ops.constant(layer_idx, DType.uint32),
        ],
        parameters=parameters,
    )


def fused_qk_ragged_rope(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection | PagedKVCacheCollection,
    freqs_cis: TensorValue,
    layer_idx: TensorValue,
    interleaved: bool = True,
) -> TensorValue:
    """Computes fused query-key attention with rotary positional encodings and ragged inputs.

    `input` and `input_row_offsets` are used together to implement the ragged tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`
    """

    if input.dtype != freqs_cis.dtype:
        msg = (
            "expected input and freqs_cis to share a dtype, but got"
            f" {input.dtype} and {freqs_cis.dtype} respectively"
        )
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = (
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        msg = f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy not in {
        KVCacheStrategy.CONTINUOUS,
        KVCacheStrategy.PAGED,
        KVCacheStrategy.PAGED_FA3_FALLBACK,
    }:
        msg = f"unsupported cache strategy for fused_qk_ragged_rope: {kv_params.cache_strategy}"
        raise ValueError(msg)

    parameters: dict[str, bool | int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "interleaved": interleaved,
    }
    if kv_params.cache_strategy in {
        KVCacheStrategy.PAGED,
        KVCacheStrategy.PAGED_FA3_FALLBACK,
    }:
        assert kv_params.page_size is not None
        parameters["page_size"] = kv_params.page_size

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    op_name = f"mo.fused_qk_rope.ragged.{cache_strategy_str}"

    return ops.inplace_custom(
        op_name,
        values=[input, input_row_offsets, kv_collection, freqs_cis, layer_idx],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
    )[0].tensor


def fused_qk_rope(
    kv_params: KVCacheParams,
    input: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis_2d: TensorValue,
    layer_idx: TensorValue,
    interleaved: bool = True,
) -> TensorValue:
    """Computes fused query-key attention with rotary positional encodings."""
    input_rank_expected = 4
    if input.rank != input_rank_expected:
        msg = (
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )
        raise ValueError(msg)

    freqs_cis_rank_expected = 2
    if freqs_cis_2d.rank != freqs_cis_rank_expected:
        msg = (
            f"expected freqs_cis_2d of rank {freqs_cis_rank_expected} but got "
            f"{freqs_cis_2d.rank}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy != KVCacheStrategy.CONTINUOUS:
        msg = f"unsupported cache strategy for fused_qk_rope: {kv_params.cache_strategy}"
        raise ValueError(msg)

    parameters: dict[str, bool | int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "interleaved": interleaved,
    }
    if kv_params.cache_strategy == KVCacheStrategy.PAGED:
        assert kv_params.page_size is not None
        parameters["page_size"] = kv_params.page_size

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    op_name = f"mo.fused_qk_rope.padded.{cache_strategy_str}"

    return ops.inplace_custom(
        op_name,
        values=[input, kv_collection, freqs_cis_2d, layer_idx],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
    )[0].tensor


def flash_attention(
    kv_params: KVCacheParams,
    input: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: TensorValue,
    attention_mask: TensorValue,
    valid_lengths: TensorValue,
    scale: float,
) -> TensorValue:
    """Computes flash attention provided the mo.opaque KV Cache."""
    input_rank_expected = 4
    if input.rank != input_rank_expected:
        msg = (
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if attention_mask.dtype != input.dtype:
        msg = (
            f"expected attention mask dtype {attention_mask.dtype} to match "
            f"the input's dtype {input.dtype}"
        )
        raise ValueError(msg)

    if valid_lengths.dtype != DType.uint32:
        msg = f"expected uint32 valid_lengths but got {valid_lengths.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy != KVCacheStrategy.CONTINUOUS:
        msg = f"unsupported cache strategy for flash_attention: {kv_params.cache_strategy}"
        raise ValueError(msg)

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    op_name = f"mo.mha.padded.{cache_strategy_str}.tensor_mask.no_pos"

    return ops.inplace_custom(
        op_name,
        values=[
            input,
            kv_collection,
            layer_idx,
            attention_mask,
            valid_lengths,
            # NOTE: The scale argument to the flash attention kernel is
            # constrained to float32.
            ops.constant(scale, dtype=DType.float32),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters={
            "num_heads": kv_params.n_kv_heads_per_device,
            "head_dim": kv_params.head_dim,
        },
    )[0].tensor


def flash_attention_with_causal_mask(
    kv_params: KVCacheParams,
    input: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: TensorValue,
    valid_lengths: TensorValue,
    scale: float,
) -> TensorValue:
    """Computes flash attention provided the mo.opaque KV Cache.
    Notably, materializes the causal mask within the kernel."""

    if input.shape[0] != valid_lengths.shape[0]:
        msg = (
            "expected batch size of input, to equal length of valid_lengths"
            f" got batch size of input ({input.shape[0]}), length of"
            f" valid_lengths ({valid_lengths.shape[0]})"
        )
        raise ValueError(msg)

    if input.dtype != kv_params.dtype:
        msg = (
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if valid_lengths.dtype != DType.uint32:
        msg = f"expected uint32 valid_lengths but got {valid_lengths.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy != KVCacheStrategy.CONTINUOUS:
        msg = f"unsupported cache strategy for flash_attention_with_causal_mask: {kv_params.cache_strategy}"
        raise ValueError(msg)

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    op_name = f"mo.mha.padded.{cache_strategy_str}.causal_mask.no_pos"

    return ops.inplace_custom(
        op_name,
        values=[
            input,
            kv_collection,
            layer_idx,
            valid_lengths,
            # NOTE: The scale argument to flash attention is constrained to float32.
            ops.constant(scale, dtype=DType.float32),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters={
            "num_heads": kv_params.n_kv_heads_per_device,
            "head_dim": kv_params.head_dim,
        },
    )[0].tensor


@dataclass
class MHAMaskConfig:
    attention_mask_variant: AttentionMaskVariant
    positional_encoding_variant: PositionalEncodingVariant


class AttentionMaskVariant(str, Enum):
    NULL_MASK = "null_mask"
    CAUSAL_MASK = "causal_mask"
    TENSOR_MASK = "tensor_mask"


class PositionalEncodingVariant(str, Enum):
    NO_POS = "no_pos"
    ALIBI_POS = "alibi_pos"


class MHAMaskVariant(str, Enum):
    CAUSAL_MASK = 0
    CAUSAL_ALIBI_MASK = 1
    NULL_MASK = 2


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
}


def flash_attention_ragged_paged_fa3_fallback(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedKVCacheCollectionFA3Fallback,
    context_lengths: TensorValue,
    layer_idx: TensorValue,
) -> TensorValue:
    """Computes flash attention provided the `!mo.opaque` KV Cache. using the FA3 fallback kernel."""
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        msg = (
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )
        raise ValueError(msg)

    if input.dtype != kv_params.dtype:
        msg = (
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        raise ValueError(msg)

    # TODO(austin): remove this cast.
    input_row_offsets_cast = input_row_offsets.cast(DType.int32)
    assert kv_params.page_size is not None, (
        "Expected page size to be set for PAGED_FA3_FALLBACK"
    )
    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "page_size": kv_params.page_size,
    }
    context_lengths_cast = context_lengths.cast(DType.int32)

    op_name = "mo.mha.ragged.paged_fa3_fallback.causal_mask.no_pos"
    return ops.inplace_custom(
        op_name,
        values=[
            input,
            input_row_offsets_cast,
            context_lengths_cast,
            kv_collection,
            layer_idx,
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
    )[0].tensor


def flash_attention_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection
    | PagedKVCacheCollection
    | PagedKVCacheCollectionFA3Fallback,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    context_lengths: Optional[TensorValue] = None,
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
    if kv_params.cache_strategy == KVCacheStrategy.PAGED_FA3_FALLBACK:
        assert context_lengths is not None, (
            "context_lengths must be provided for PAGED_FA3_FALLBACK"
        )
        return flash_attention_ragged_paged_fa3_fallback(
            kv_params,
            input,
            input_row_offsets,
            cast(PagedKVCacheCollectionFA3Fallback, kv_collection),
            context_lengths,
            layer_idx,
        )
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        msg = (
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )
        raise ValueError(msg)

    if input.dtype != kv_params.dtype:
        msg = (
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy not in {
        KVCacheStrategy.CONTINUOUS,
        KVCacheStrategy.PAGED,
    }:
        msg = f"unsupported cache strategy for flash_attention_ragged: {kv_params.cache_strategy}"
        raise ValueError(msg)

    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
    }
    if kv_params.cache_strategy == KVCacheStrategy.PAGED:
        assert kv_params.page_size is not None
        parameters["page_size"] = kv_params.page_size

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    op_name = f"mo.mha.ragged.{cache_strategy_str}.{str(mha_mask_config.attention_mask_variant.value)}.{str(mha_mask_config.positional_encoding_variant.value)}"

    return ops.inplace_custom(
        op_name,
        values=[
            input,
            input_row_offsets,
            kv_collection,
            layer_idx,
            # NOTE: The scale argument to flash attention is constrained to float32.
            ops.constant(scale, dtype=DType.float32),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
    )[0].tensor


def flare_mla_decode_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedKVCacheCollection,
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
        msg = (
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )
        raise ValueError(msg)

    if input.dtype != kv_params.dtype:
        msg = (
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy is not KVCacheStrategy.PAGED:
        msg = f"unsupported cache strategy for flash_attention_ragged: {kv_params.cache_strategy}"
        raise ValueError(msg)

    assert kv_params.page_size is not None
    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "page_size": kv_params.page_size,
    }

    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    op_name = f"mo.mla.decode.ragged.paged.{str(mha_mask_config.attention_mask_variant.value)}.{str(mha_mask_config.positional_encoding_variant.value)}"

    return ops.inplace_custom(
        op_name,
        values=[
            input,
            input_row_offsets,
            kv_collection,
            layer_idx,
            # NOTE: The scale argument to flash attention is constrained to float32.
            ops.constant(scale, dtype=DType.float32),
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


def cross_attention_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection | PagedKVCacheCollection,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    kv_input_row_offsets: TensorValue,
    q_max_seq_len: TensorValue,
    scale: float,
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
        msg = (
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )
        raise ValueError(msg)

    if input.dtype != kv_params.dtype:
        msg = (
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )
        raise ValueError(msg)

    if layer_idx.dtype != DType.uint32:
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy not in {
        KVCacheStrategy.CONTINUOUS,
        KVCacheStrategy.PAGED,
    }:
        msg = f"unsupported cache strategy for cross_attention_ragged: {kv_params.cache_strategy}"
        raise ValueError(msg)

    if q_max_seq_len and (q_max_seq_len.dtype != DType.uint32):
        msg = (
            "expected q_max_seq_len to be uint32 but got {q_max_seq_len.dtype}"
        )
        raise ValueError(msg)

    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
    }
    if kv_params.cache_strategy == KVCacheStrategy.PAGED:
        assert kv_params.page_size is not None
        parameters["page_size"] = kv_params.page_size

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    op_name = f"mo.cross_attention.ragged.{cache_strategy_str}.{str(mha_mask_config.attention_mask_variant.value)}.{str(mha_mask_config.positional_encoding_variant.value)}"

    return ops.inplace_custom(
        op_name,
        values=[
            input,
            input_row_offsets,
            # Plumb in the query max sequence length for cross attention.
            # For self attention this is the same as the KV max seq len stored
            # on the kv_collection, but that isn't the case for cross attention.
            q_max_seq_len,
            kv_input_row_offsets,
            kv_collection,
            layer_idx,
            # NOTE: The scale argument to flash attention is constrained to float32.
            ops.constant(scale, dtype=DType.float32),
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
        msg = f"expected a to have rank {a_rank_expected}, was {a.rank}"
        raise ValueError(msg)

    b0_rank_expected = 2
    if b0.rank != b0_rank_expected:
        msg = f"expected b0 to have rank {b0_rank_expected}, was {b0.rank}"
        raise ValueError(msg)

    b1_rank_expected = 2
    if b1.rank != b1_rank_expected:
        msg = f"expected b1 to have rank {b1_rank_expected}, was {b1.rank}"
        raise ValueError(msg)

    m = a.shape[0]
    n = b0.shape[0]
    if b0.shape[1] != a.shape[1]:
        msg = f"a.shape[1] == {a.shape[1]} != {b0.shape[1]} == b0.shape[1]"
        raise ValueError(msg)

    if b0.shape != b1.shape:
        msg = f"b0.shape == {b0.shape} != {b1.shape} == b1.shape"
        raise ValueError(msg)

    if a.dtype != b0.dtype or a.dtype != b1.dtype:
        msg = (
            "Element types of all arguments must be equal, but received"
            f" {a.dtype}, {b0.dtype}, and {b1.dtype}."
        )
        raise ValueError(msg)

    return ops.custom(
        "swishGLU",
        values=[a, b0, b1],
        out_types=[
            TensorType(
                dtype=a.dtype,
                shape=[m, n],
                device=a.device,
            )
        ],
    )[0].tensor


def rms_norm_key_cache(
    kv_params: KVCacheParams,
    kv_collection: ContinuousBatchingKVCacheCollection | PagedKVCacheCollection,
    gamma: TensorValue,
    epsilon: float | np.floating,
    layer_idx: int | np.integer,
    total_seq_len: Dim,
    input_row_offsets: TensorValue,
    rms_norm_cols: Optional[int] = None,
) -> None:
    """Computes RMSNorm on the _new_ entries in the KVCache.

    This function applies RMSNorm to either all dimensions or a subset of
    dimensions in each head of the key cache. The size of the gamma tensor
    determines how many dimensions will be normalized. If gamma's size doesn't
    match head_dim, rms_norm_cols must be explicitly specified to confirm the
    intention to normalize only a subset of dimensions.

    Currently, the KVCacheT class itself isn't aware of the new cache entries
    until cache length increment, which happens after model forward.
    So use `input_row_offsets` to do this bookkeeping.
    """
    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    op_name = f"mo.rms_norm_kv_cache.ragged.{cache_strategy_str}"

    gamma_rank_expected = 1
    if gamma.rank != gamma_rank_expected:
        msg = (
            f"expected gamma of rank {gamma_rank_expected} but got {gamma.rank}"
        )
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        raise ValueError(msg)

    if gamma.shape[0] != kv_params.head_dim:
        if rms_norm_cols is None:
            msg = (
                "Size of gamma doesn't match head_dim. Please pass rms_norm_cols "
                "explicitly if you intend to apply RMSNorm to only a subset of "
                "head dimensions"
            )
            raise ValueError(msg)
        elif rms_norm_cols != gamma.shape[0]:
            msg = f"expected gamma of size {rms_norm_cols} but got {gamma.shape[0]}"
            raise ValueError(msg)

    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
    }
    if kv_params.cache_strategy == KVCacheStrategy.PAGED:
        assert kv_params.page_size is not None
        parameters["page_size"] = kv_params.page_size

    ops.inplace_custom(
        op_name,
        values=[
            kv_collection,
            gamma,
            ops.constant(epsilon, gamma.dtype),
            ops.constant(layer_idx, DType.uint32),
            ops.cast(TensorValue(total_seq_len), DType.uint32),
            input_row_offsets,
        ],
        parameters=parameters,
    )
