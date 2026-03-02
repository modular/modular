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

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields

from max.dtype import DType
from max.graph import BufferValue, TensorType, TensorValue, ops
from max.nn.kv_cache import KVCacheParams

from .attention.mask_config import MHAMaskVariant


@dataclass
class PagedKVCacheTensorsNoOpaque:
    blocks: BufferValue
    cache_lengths: TensorValue
    lookup_table: TensorValue
    max_lengths: TensorValue
    kv_scales: BufferValue | None = None

    def __iter__(self) -> Iterator[TensorValue]:
        for field in fields(self):
            yield getattr(self, field.name)


def rope_no_opaque(
    input: TensorValue,
    input_row_offsets: TensorValue,
    start_pos: TensorValue,
    freqs_cis: TensorValue,
    interleaved: bool = True,
) -> TensorValue:
    """Applies RoPE (Rotary Position Embedding) to input tensor without using opaque KV cache types.

    Args:
        input: Input tensor of shape [total_seq_len, num_heads, head_dim]
        input_row_offsets: Ragged tensor offsets indicating where each batch starts and ends
        start_pos: Starting positions for each batch element
        freqs_cis: Frequency tensor for rotary embeddings
        interleaved: Whether to use interleaved RoPE pattern

    Returns:
        Output tensor with RoPE applied, same shape as input
    """
    # Input validation
    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if start_pos.dtype != DType.uint32:
        raise ValueError(
            f"expected start_pos to have dtype uint32, was {start_pos.dtype}"
        )

    # Validate tensor ranks
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input to have rank {input_rank_expected}, was {input.rank}"
        )

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

    # Set up parameters for the kernel
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


def store_k_cache(
    kv_collection: PagedKVCacheTensorsNoOpaque,
    x_k: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
) -> None:
    """Stores key tensor into the KV cache.

    Args:
        kv_collection: The KV cache collection containing cache tensors
        x_k: Key tensor to store, shape [total_seq_len, num_heads, head_dim]
        input_row_offsets: Ragged tensor offsets indicating where each batch starts and ends
        layer_idx: Layer index for the cache, must be uint32
    """
    _store_cache_common(
        kv_collection=kv_collection,
        x_cache=x_k,
        input_row_offsets=input_row_offsets,
        layer_idx=layer_idx,
        key_or_value=0,  # 0 for key cache
    )


def store_v_cache(
    kv_collection: PagedKVCacheTensorsNoOpaque,
    x_v: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
) -> None:
    """Stores value tensor into the KV cache.

    Args:
        kv_collection: The KV cache collection containing cache tensors
        x_v: Value tensor to store, shape [total_seq_len, num_heads, head_dim]
        input_row_offsets: Ragged tensor offsets indicating where each batch starts and ends
        layer_idx: Layer index for the cache, must be uint32
    """
    _store_cache_common(
        kv_collection=kv_collection,
        x_cache=x_v,
        input_row_offsets=input_row_offsets,
        layer_idx=layer_idx,
        key_or_value=1,  # 1 for value cache
    )


def _store_cache_common(
    kv_collection: PagedKVCacheTensorsNoOpaque,
    x_cache: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
    key_or_value: int,
) -> None:
    """Common implementation for storing key or value tensors into KV cache.

    Args:
        kv_collection: The KV cache collection containing cache tensors
        x_cache: Buffer to store (key or value)
        input_row_offsets: Ragged tensor offsets
        layer_idx: Layer index for the cache
        key_or_value: 0 for key cache, 1 for value cache
    """

    _validate_inputs(x_cache, input_row_offsets, layer_idx, kv_collection)

    # Set up parameters for the kernel
    parameters: dict[str, int | str | DType] = {
        "key_or_value": key_or_value,
    }

    ops.inplace_custom(
        "mo.kv_cache.store.paged.ragged",
        device=x_cache.device,
        values=[
            x_cache,
            kv_collection.blocks,
            kv_collection.cache_lengths,
            kv_collection.lookup_table,
            input_row_offsets,
            kv_collection.max_lengths,
            layer_idx,
        ],
        parameters=parameters,
    )


def _validate_inputs(
    x_cache: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
    kv_collection: PagedKVCacheTensorsNoOpaque,
) -> None:
    # Input validation
    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected input_row_offsets to have dtype uint32, was"
            f" {input_row_offsets.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(
            f"expected layer_idx to have dtype uint32, was {layer_idx.dtype}"
        )

    # Validate tensor ranks
    x_cache_rank_expected = 3
    if x_cache.rank != x_cache_rank_expected:
        raise ValueError(
            f"expected x_cache to have rank {x_cache_rank_expected}, was {x_cache.rank}"
        )

    if input_row_offsets.rank != 1:
        raise ValueError(
            f"expected input_row_offsets to have rank 1, was {input_row_offsets.rank}"
        )

    if kv_collection.blocks.rank != 6:
        raise ValueError(
            f"expected blocks to have rank 6, was {kv_collection.blocks.rank}"
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

    # Validate scalar layer_idx
    if layer_idx.rank != 0:
        raise ValueError(
            f"expected layer_idx to be a scalar (rank 0), was rank {layer_idx.rank}"
        )


def store_k_scale_cache(
    kv_collection: PagedKVCacheTensorsNoOpaque,
    x_k_scale: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
    quantization_granularity: int,
) -> None:
    """Stores key scale tensor into the KV cache."""
    _validate_inputs(x_k_scale, input_row_offsets, layer_idx, kv_collection)
    if kv_collection.kv_scales is None:
        raise ValueError(
            "kv_collection.kv_scales is None, expected a buffer value"
        )
    ops.inplace_custom(
        "mo.kv_cache.store_k_scales.paged.ragged",
        device=x_k_scale.device,
        values=[
            x_k_scale,
            kv_collection.blocks,
            kv_collection.cache_lengths,
            kv_collection.lookup_table,
            input_row_offsets,
            kv_collection.max_lengths,
            kv_collection.kv_scales,
            layer_idx,
        ],
        parameters={
            "quantization_granularity": quantization_granularity,
        },
    )


def flash_attention_ragged_no_opaque(
    kv_params: KVCacheParams,
    input: TensorValue,
    layer_idx: TensorValue,
    kv_collection: PagedKVCacheTensorsNoOpaque,
    input_row_offsets: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float = 1.0,
) -> TensorValue:
    # TODO: implement
    raise NotImplementedError(
        "flash_attention_ragged_no_opaque not implemented"
    )
