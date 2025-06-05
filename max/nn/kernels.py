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
from typing import Optional

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferValue,
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

from .attention.mask_config import (
    AttentionMaskVariant,
    MHAMaskConfig,
    MHAMaskVariant,
    PositionalEncodingVariant,
)
from .kv_cache import (
    ContinuousBatchingKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
    PagedKVCacheCollection,
)

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
    }:
        msg = f"unsupported cache strategy for fused_qkv_ragged_matmul: {kv_params.cache_strategy}"
        raise ValueError(msg)

    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
    }
    if kv_params.cache_strategy == KVCacheStrategy.PAGED:
        assert kv_params.page_size is not None
        parameters["page_size"] = int(kv_params.page_size)

    cache_strategy_str = kv_params.cache_strategy.kernel_substring()
    op_name = f"mo.fused_qkv_matmul.ragged.{cache_strategy_str}"
    values = [input, input_row_offsets, wqkv, kv_collection, layer_idx]

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
        parameters=parameters,
    )[0].tensor


def fused_qkv_ragged_matmul_scaled_float8(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedKVCacheCollection,
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
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "page_size": int(kv_params.page_size),
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
            kv_collection,
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

    if kv_params.cache_strategy not in {KVCacheStrategy.PAGED}:
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

    assert kv_params.page_size is not None
    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "quantization_encoding_q": quantization_encoding_q.name,
        "quantization_encoding_k": quantization_encoding_k.name,
        "quantization_encoding_v": quantization_encoding_v.name,
        "page_size": kv_params.page_size,
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
    # point that need to be subtracted from the quantized weights.
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
        device=input.device,
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
    kv_collection: PagedKVCacheCollection,
    layer_idx: TensorValue,
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
    op_name = f"mo.kv_matmul.ragged.{cache_strategy_str}"

    ops.inplace_custom(
        name=op_name,
        device=hidden_states.device,
        values=[
            hidden_states,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
        ],
        parameters=parameters,
    )


def matmul_k_cache_ragged(
    kv_params: KVCacheParams,
    hidden_states: TensorValue,
    input_row_offsets: TensorValue,
    weight: TensorValue,
    kv_collection: PagedKVCacheCollection,
    layer_idx: TensorValue,
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
        device=hidden_states.device,
        values=[
            hidden_states,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
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

    Args:
        input: [batch_size * seq_len, n_heads, head_dim]
        input_row_offsets:
        freqs_cis: tensor of shape (max_seq_len * 2, head_dim)
        layer_idx:
        interleaved:

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
    }:
        msg = f"unsupported cache strategy for fused_qk_ragged_rope: {kv_params.cache_strategy}"
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
    op_name = f"mo.fused_qk_rope.ragged.{cache_strategy_str}"

    return ops.inplace_custom(
        op_name,
        device=input.device,
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
        device=input.device,
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
    op_name = f"mo.mha.padded.{cache_strategy_str}.tensor_mask"
    parameters: dict[str, bool | int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "score_mod_str": PositionalEncodingVariant.NO_POS.value,
    }
    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=[
            input,
            kv_collection,
            layer_idx,
            attention_mask,
            valid_lengths,
            # NOTE: The scale argument to the flash attention kernel is
            # constrained to float32.
            ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
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
    op_name = f"mo.mha.padded.{cache_strategy_str}"
    parameters: dict[str, bool | int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "mask_str": MHAMaskVariant.CAUSAL_MASK.value,
        "score_mod_str": PositionalEncodingVariant.NO_POS.value,
    }
    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=[
            input,
            kv_collection,
            layer_idx,
            valid_lengths,
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


def causal_flash_attention_gpu(
    q: TensorValue, k: TensorValue, v: TensorValue, scale: float
) -> TensorValue:
    """Computes causal flash attention using GPU-optimized kernel.
    Args:
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        v: Value tensor of shape [batch, seq_len, num_heads, head_dim]
        scale: Scaling factor for attention scores
    """
    if q.dtype != k.dtype or q.dtype != v.dtype:
        msg = (
            "q, k, v must have matching dtypes. Got "
            f"q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}"
        )
        raise ValueError(msg)

    expected_rank = 4
    for name, tensor in [("q", q), ("k", k), ("v", v)]:
        if tensor.rank != expected_rank:
            msg = f"{name} must be rank {expected_rank}, got {tensor.rank}"
            raise ValueError(msg)

    # Validate head dimension matches across all inputs
    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        msg = (
            "All inputs must have same head_dim. Got "
            f"q: {head_dim}, k: {k.shape[-1]}, v: {v.shape[-1]}"
        )
        raise ValueError(msg)

    return ops.custom(
        "causal_flash_attention_gpu",
        device=q.device,
        values=[
            q,
            k,
            v,
            ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=q.dtype,
                shape=q.shape,
                device=q.device,
            )
        ],
    )[0].tensor


def null_mask_flash_attention_gpu(
    q: TensorValue, k: TensorValue, v: TensorValue, scale: float
) -> TensorValue:
    """Computes flash attention using GPU-optimized kernel.
    Args:
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        v: Value tensor of shape [batch, seq_len, num_heads, head_dim]
        scale: Scaling factor for attention scores
    """
    if q.dtype != k.dtype or q.dtype != v.dtype:
        msg = (
            "q, k, v must have matching dtypes. Got "
            f"q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}"
        )
        raise ValueError(msg)

    expected_rank = 4
    for name, tensor in [("q", q), ("k", k), ("v", v)]:
        if tensor.rank != expected_rank:
            msg = f"{name} must be rank {expected_rank}, got {tensor.rank}"
            raise ValueError(msg)

    # Validate head dimension matches across all inputs
    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        msg = (
            "All inputs must have same head_dim. Got "
            f"q: {head_dim}, k: {k.shape[-1]}, v: {v.shape[-1]}"
        )
        raise ValueError(msg)

    return ops.custom(
        "no_mask_flash_attention_gpu",
        device=q.device,
        values=[
            q,
            k,
            v,
            ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=q.dtype,
                shape=q.shape,
                device=q.device,
            )
        ],
    )[0].tensor


def flash_attention_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: ContinuousBatchingKVCacheCollection | PagedKVCacheCollection,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    local_window_size: int = -1,
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
    op_name = f"mo.mha.ragged.{cache_strategy_str}"

    parameters["mask_str"] = mha_mask_config.attention_mask_variant.value
    parameters["score_mod_str"] = (
        mha_mask_config.positional_encoding_variant.value
    )
    parameters["local_window_size"] = local_window_size

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=[
            input,
            input_row_offsets,
            kv_collection,
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
    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "page_size": kv_params.page_size,
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
            kv_collection,
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
    kv_collection: PagedKVCacheCollection,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    qk_rope_dim: int = 64,
    prev_output: Optional[TensorValue] = None,
    prev_softmax_info: Optional[TensorValue] = None,
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
        msg = f"unsupported cache strategy for flare_mla_prefill_ragged: {kv_params.cache_strategy}"
        raise ValueError(msg)

    assert kv_params.page_size is not None
    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "page_size": kv_params.page_size,
        "mask_str": mha_mask_config.attention_mask_variant.value,
        "score_mod_str": mha_mask_config.positional_encoding_variant.value,
    }

    is_init_str = ".init" if prev_output is None else ""
    op_name = f"mo.mla.prefill{is_init_str}.ragged.paged"

    input_values: MutableSequence[Value] = [
        input,
        k,
        v,
        buffer_row_offsets,
        cache_offsets,
        input_row_offsets,
        kv_collection,
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
    kv_collection: PagedKVCacheCollection,
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
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if input_row_offsets.dtype != DType.uint32:
        msg = f"expected uint32 input_row_offsets but got {input_row_offsets.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy is not KVCacheStrategy.PAGED:
        msg = f"unsupported cache strategy for flare_mla_prefill_plan: {kv_params.cache_strategy}"
        raise ValueError(msg)

    assert kv_params.page_size is not None
    parameters: dict[str, int | str | DType] = {
        "dtype": kv_params.dtype,
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "page_size": kv_params.page_size,
    }

    buffer_size_tensor = ops.constant(
        buffer_size, DType.uint32, device=DeviceRef.CPU()
    )

    results = ops.inplace_custom(
        "mo.mla.prefill.ragged.plan",
        device=input_row_offsets.device,
        values=[
            input_row_offsets,
            kv_collection,
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
        parameters=parameters,
    )

    return results[0].tensor, results[1].tensor, results[2].tensor


def flare_mla_decompress_k_cache(
    kv_params: KVCacheParams,
    buffer_row_offsets_1d: TensorValue,
    cache_offsets_1d: TensorValue,
    buffer_length: TensorValue,
    weight: TensorValue,
    kv_collection: PagedKVCacheCollection,
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
        msg = f"expected uint32 layer_idx but got {layer_idx.dtype}"
        raise ValueError(msg)

    if cache_offsets_1d.dtype != DType.uint32:
        msg = f"expected uint32 cache_offsets but got {cache_offsets_1d.dtype}"
        raise ValueError(msg)

    if kv_params.cache_strategy is not KVCacheStrategy.PAGED:
        msg = f"unsupported cache strategy for flare_mla_decompress_k_cache: {kv_params.cache_strategy}"
        raise ValueError(msg)

    assert kv_params.page_size is not None
    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "page_size": kv_params.page_size,
    }

    results = ops.inplace_custom(
        "mo.mla.decompress.k.cache.ragged.paged",
        device=buffer_row_offsets_1d.device,
        values=[
            buffer_row_offsets_1d,
            cache_offsets_1d,
            buffer_length,
            weight,
            kv_collection,
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
        parameters=parameters,
    )

    return results[1].tensor


def kv_cache_get_max_seq_len(
    kv_collection: PagedKVCacheCollection,
) -> TensorValue:
    """This kernel returns the maximum sequence length."""
    return ops.inplace_custom(
        "mo.kv_cache.get_max_seq_len.paged",
        device=DeviceRef.CPU(),
        values=[kv_collection],
        out_types=[
            TensorType(
                dtype=DType.uint32,
                shape=[1],
                device=DeviceRef.CPU(),
            )
        ],
    )[0].tensor[0]


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

    mha_mask_config = _MHA_MASK_CONFIG_DICT[mask_variant]
    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
        "local_window_size": local_window_size,
        "mask_str": mha_mask_config.attention_mask_variant.value,
        "score_mod_str": mha_mask_config.positional_encoding_variant.value,
    }
    if kv_params.cache_strategy == KVCacheStrategy.PAGED:
        assert kv_params.page_size is not None
        parameters["page_size"] = kv_params.page_size

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
            kv_collection,
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
        device=a.device,
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
    layer_idx: TensorValue,
    total_seq_len: Dim,
    input_row_offsets: TensorValue,
    weight_offset: float | np.floating,
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

    if gamma.dtype != kv_params.dtype:
        msg = f"expected gamma dtype {gamma.dtype} to match KV dtype {kv_params.dtype}"
        raise TypeError(msg)

    parameters: dict[str, int | str | DType] = {
        "num_heads": kv_params.n_kv_heads_per_device,
        "head_dim": kv_params.head_dim,
    }
    if kv_params.cache_strategy == KVCacheStrategy.PAGED:
        assert kv_params.page_size is not None
        parameters["page_size"] = kv_params.page_size

    ops.inplace_custom(
        op_name,
        device=input_row_offsets.device,
        values=[
            kv_collection,
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
        A tuple of four tensors:
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
                dtype=DType.uint32,
                shape=[num_local_experts],
                device=topk_ids.device,
            ),  # expert_ids
            TensorType(
                dtype=DType.uint32,
                shape=[2],
                device=topk_ids.device,
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
        msg = f"expected weight of rank 3 but got {weight.rank}"
        raise ValueError(msg)

    if hidden_states.rank != 2:
        msg = f"expected hidden_states of rank 2 but got {hidden_states.rank}"
        raise ValueError(msg)

    if (
        weight.shape[2] != hidden_states.shape[1]
        or weight.shape[0] != expert_ids.shape[0]
    ):
        msg = f"expected weight is of shape [num_experts, *, {hidden_states.shape[1]}] but got {weight.shape}"
        raise ValueError(msg)

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


def quantize_static_scaled_float8(
    x: TensorValue,
    scale: TensorValue,
    scale_is_inverted: bool = True,
) -> TensorValue:
    if scale.shape not in [[], [1]]:
        msg = f"expected scale to be a scalar, but got shape of {scale.shape}"
        raise ValueError(msg)

    if x.dtype not in [DType.float16, DType.bfloat16, DType.float32]:
        msg = f"expected input dtype to be float16, bfloat16, or float32, but got {x.dtype}"
        raise ValueError(msg)

    if x.rank != 2:
        msg = f"expected input rank to be 2, but got {x.rank}"
        raise ValueError(msg)

    if scale.device != DeviceRef.CPU():
        msg = f"expected scale to be on CPU, but got {scale.device}"
        raise ValueError(msg)

    return ops.custom(
        "mo.quantize_static_scaled_float8",
        device=x.device,
        values=[x, scale.reshape([])],
        parameters={"scale_is_inverted": scale_is_inverted},
        out_types=[
            TensorType(
                dtype=DType.float8_e4m3fn, shape=x.shape, device=x.device
            )
        ],
    )[0].tensor


def quantize_dynamic_scaled_float8(
    input: TensorValue,
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
        msg = "input must be rank 2 tensor"
        raise ValueError(msg)

    if out_type != DType.float8_e4m3fn:
        msg = "out_type must be float8_e4m3fn"
        raise ValueError(msg)

    group_size = (
        group_size_or_per_token
        if group_size_or_per_token != -1
        else input.shape[1]
    )

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
                shape=[input.shape[0], input.shape[1] // group_size],
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
        msg = "All arguments must be rank 2 tensors"
        raise ValueError(msg)

    if a.shape[1] != b.shape[1]:
        msg = "The second dimension of b must match the second dimension of a"
        raise ValueError(msg)

    if a_scales.shape[1] != 1:
        msg = "only per-token scaling is supported for a"
        raise ValueError(msg)

    if b_scales.shape[1] != 1:
        msg = "only channel-wise scaling is supported for b"
        raise ValueError(msg)

    if (a.dtype != b.dtype) or (a_scales.dtype != b_scales.dtype):
        msg = (
            f"a and b dtypes {a.dtype}, {b.dtype} must match, "
            f"as do a and b scales dtypes {a_scales.dtype}, {b_scales.dtype}"
        )
        raise TypeError(msg)

    result = ops.custom(
        "mo.matmul_dynamic_scaled_fp8",
        device=a.device,
        values=[a, b, a_scales, b_scales],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[a.shape[0], b.shape[0]],
                device=a.device,
            )
        ],
    )[0].tensor

    return result


def matmul_static_scaled_float8(
    input: TensorValue,
    weight: TensorValue,
    input_scale: TensorValue,
    weight_scale: TensorValue,
) -> TensorValue:
    if input_scale.shape not in [[], [1]]:
        msg = f"expected input_scale to be a scalar, but got shape of {input_scale.shape}"
        raise ValueError(msg)
    if weight_scale.shape not in [[], [1]]:
        msg = f"expected weight_scale to be a scalar, but got shape of {weight_scale.shape}"
        raise ValueError(msg)

    if input.dtype != DType.float8_e4m3fn:
        msg = f"expected input dtype to be float8_e4m3fn, but got {input.dtype}"
        raise ValueError(msg)
    if weight.dtype != DType.float8_e4m3fn:
        msg = (
            f"expected weight dtype to be float8_e4m3fn, but got {weight.dtype}"
        )
        raise ValueError(msg)

    if input.rank != 2:
        msg = f"expected input rank to be 2, but got {input.rank}"
        raise ValueError(msg)
    if weight.rank != 2:
        msg = f"expected weight rank to be 2, but got {weight.rank}"
        raise ValueError(msg)

    if input.shape[1] != weight.shape[1]:
        raise ValueError("K dimension does not match for matmul")

    if input_scale.device != DeviceRef.CPU():
        msg = f"expected input_scale to be on CPU, but got {input_scale.device}"
        raise ValueError(msg)

    if weight_scale.device != DeviceRef.CPU():
        msg = (
            f"expected weight_scale to be on CPU, but got {weight_scale.device}"
        )
        raise ValueError(msg)

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
        msg = "a and b must have the same dtype"
        raise ValueError(msg)

    if a_row_offsets.shape[0] != b_row_offsets.shape[0]:
        msg = "a_row_offsets and b_row_offsets must have the same shape"
        raise ValueError(msg)

    c_shape = [a.shape[0] + b.shape[0]] + a.shape[1:]

    results = ops.custom(
        "mo.merge_ragged_tensors",
        device=a.device,
        values=[a, a_row_offsets, b, b_row_offsets],
        out_types=[
            TensorType(
                dtype=a.dtype,
                shape=c_shape,
                device=a.device,
            ),
            TensorType(
                dtype=DType.uint32,
                shape=a_row_offsets.shape,
                device=a.device,
            ),
        ],
    )

    return results[0].tensor, results[1].tensor


def apply_penalties_to_logits(
    logits_buffer: BufferValue,
    frequency_data: TensorValue,
    frequency_offsets: TensorValue,
    *,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
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

    ops.inplace_custom(
        "sampler.apply_penalties",
        device=logits_buffer.device,
        values=[
            logits_buffer,
            frequency_data,
            frequency_offsets,
            ops.constant(
                frequency_penalty,
                DType.float32,
                device=DeviceRef.CPU(),
            ),
            ops.constant(
                presence_penalty,
                DType.float32,
                device=DeviceRef.CPU(),
            ),
            ops.constant(
                repetition_penalty,
                DType.float32,
                device=DeviceRef.CPU(),
            ),
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
    data: BufferValue,
    indices: TensorValue,
    fill_val: float,
) -> None:
    """
    Scatters values into a tensor at specified indices.
    """

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
    top_k: int,
    temperature: float,
    *,
    top_p: float = 1.0,
    seed: int = 0,
) -> TensorValue:
    """Performs top-k sampling with temperature scaling.

    Args:
        logits: Input logits tensor of shape [batch_size, vocab_size].
        top_k: Number of top tokens to consider for sampling.
        temperature: Temperature for scaling logits before sampling.
        seed: Seed for the random number generator.
    Returns:
        Sampled tokens tensor of shape [batch_size, 1].

    Raises:
        ValueError: If input validation fails.
    """

    if top_k <= 0:
        raise ValueError(f"expected top_k to be positive, got {top_k}")

    if temperature <= 0:
        raise ValueError(
            f"expected temperature to be positive, got {temperature}"
        )

    if top_p <= 0 or top_p > 1:
        raise ValueError(f"expected top_p to be in (0, 1], got {top_p}")

    batch_shape = logits.shape[:-1]
    device = logits.device

    return ops.custom(
        "sampler.fused_token_sampling",
        device=logits.device,
        values=[
            ops.constant(top_k, dtype=DType.int64, device=DeviceRef.CPU()),
            ops.constant(
                temperature, dtype=DType.float32, device=DeviceRef.CPU()
            ),
            ops.constant(top_p, dtype=DType.float32, device=DeviceRef.CPU()),
            ops.constant(seed, dtype=DType.uint64, device=DeviceRef.CPU()),
            logits,
        ],
        out_types=[
            TensorType(
                dtype=DType.int64,
                shape=batch_shape + [1],
                device=device,
            )
        ],
    )[0].tensor
