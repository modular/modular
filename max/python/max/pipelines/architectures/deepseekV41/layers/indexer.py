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
"""Sparse-attention indexer for DeepSeek-V4.1-Flash.

The indexer cheaply scores every past KV position for each query, so that
attention can be restricted to the highest-scoring positions.

Index keys and queries are MXFP4: 128 E2M1 values packed into 64 bytes, with
one UE8M0 scale per 32 values packed into a 32-bit word. A layer's keys live
in an MXFP4 cache leaf with in-page scales: each page holds every key's
value bytes, then every key's scale word. Each page holds
``page_size // ratio`` keys, so key ``j`` of a request sits at slot
``j % keys_per_page`` of the block its token page maps to.

Selection is causal: a query at position ``p`` sees the ``(p + 1) // ratio``
keys whose groups have closed. Candidate blocks group ``block_size``
consecutive keys; see :func:`candidate_keep_from_block_max`.
"""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.kernels import top_k_per_row
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm

from .ragged import V41RaggedRows as RaggedRows


class DeepseekV41Indexer(Module):
    """Holds the index query, weight and key projections.

    .. code-block:: text

        score[t, s] = sum_h relu(q[t, h, :] . k[s, :]) * weight[t, h]
        weight[t, h] = weights_proj(x)[t, h] * head_dim ** -0.5

    RoPE is interleaved over each head's trailing ``rope_head_dim`` dims.
    Keys are projected from a compressed pre-RoPE KV latent, not from ``x``.
    ``wk``/``k_norm`` exist only when ``owns_k``, since a layer may reuse
    another layer's index keys.

    Selection scores every head at once, so under tensor parallelism every
    weight is replicated and each device selects the same keys.
    """

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        main_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        rms_norm_eps: float,
        dtype: DType,
        device: DeviceRef,
        *,
        owns_k: bool,
    ) -> None:
        """Initializes the indexer.

        Args:
            owns_k: Whether this layer produces its own index keys (creates
                ``wk``/``k_norm``).

        Raises:
            ValueError: If ``index_head_dim`` is not 128 or ``rope_head_dim``
                is not 64, the MXFP4 kernels' head layout.
        """
        super().__init__()
        if index_head_dim != 128 or rope_head_dim != 64:
            raise ValueError(
                "the MXFP4 indexer kernels need index_head_dim 128 and "
                f"rope_head_dim 64, got {index_head_dim} and {rope_head_dim}"
            )
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        # Scaled by `head_dim` only; there is no `n_heads**-0.5` factor.
        self.softmax_scale = self.head_dim**-0.5
        self.rms_norm_eps = rms_norm_eps
        self.owns_k = owns_k

        self.wq_b = Linear(
            q_lora_rank, self.n_heads * self.head_dim, dtype, device
        )
        # Uses the model `dtype` rather than a fixed bf16.
        self.weights_proj = Linear(hidden_size, self.n_heads, dtype, device)

        if owns_k:
            self.wk = Linear(main_head_dim, self.head_dim, dtype, device)
            self.k_norm = RMSNorm(self.head_dim, DType.float32, rms_norm_eps)

    def __call__(
        self, qr: TensorValue, x: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Projects the index queries and per-head weights.

        Args:
            qr: ``[..., T, q_lora_rank]`` normalized low-rank query.
            x: ``[..., T, hidden_size]`` raw hidden state.

        Returns:
            ``([T, n_heads * head_dim] queries, [T, n_heads] weights)``,
            before RoPE, quantization and weight scaling.
        """
        return (
            ops.reshape(self.wq_b(qr), [-1, self.n_heads * self.head_dim]),
            ops.reshape(self.weights_proj(x), [-1, self.n_heads]),
        )

    def shard_heads(self, num_devices: int) -> None:
        """Replicates ``wq_b`` and ``weights_proj`` on every device."""
        self.wq_b.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.weights_proj.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )


def candidate_blocks_active(
    width: int, compress_ratio: int, *, block_size: int, topk_blocks: int
) -> bool:
    """Returns whether the candidate keep step can drop a reachable block.

    False when ``ceil(width / (block_size * compress_ratio)) <=
    topk_blocks`` (every block is kept, so the pass can be skipped) or when
    any size is not positive (the mechanism is off).
    """
    if block_size <= 0 or topk_blocks <= 0 or compress_ratio <= 0:
        return False
    group = block_size * compress_ratio
    return -(-width // group) > topk_blocks


def _host_f32(value: float) -> TensorValue:
    return ops.constant([value], DType.float32, DeviceRef.CPU())


def _cache_values(leaf: PagedCacheValues) -> list[Value[Any]]:
    return [
        leaf.kv_blocks,
        leaf.page_stride,
        leaf.cache_lengths,
        leaf.lookup_table,
        leaf.max_prompt_length,
        leaf.max_cache_length,
    ]


def rope_cos_sin_table(freqs_cis_base: TensorValue) -> TensorValue:
    """Returns ``[L, rope_dim]`` float32 rows of cosines then sines.

    Args:
        freqs_cis_base: ``[L, rope_dim // 2, 2]`` interleaved cos/sin.
    """
    length, half = freqs_cis_base.shape[0], freqs_cis_base.shape[1]
    return ops.reshape(
        ops.transpose(freqs_cis_base, 1, 2), [length, int(half) * 2]
    )


def quantize_queries(
    queries: TensorValue,
    weights: TensorValue,
    positions: TensorValue,
    rope_table: TensorValue,
    *,
    n_heads: int,
    weight_scale: float,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """RoPEs and quantizes index queries to MXFP4 and scales the weights.

    Each head's trailing 64 channels rotate as interleaved pairs.

    Args:
        queries: ``[T, n_heads * 128]`` bf16 projected queries.
        weights: ``[T, n_heads]`` bf16 per-head weights.
        positions: ``[T]`` token positions.
        rope_table: From :func:`rope_cos_sin_table`.
        n_heads: Index heads.
        weight_scale: Factor folded into the weights.

    Returns:
        ``(packed [T, n_heads, 64] uint8, scales [T, n_heads * 4] uint8,
        weights [T, n_heads] float32)``.
    """
    t = queries.shape[0]
    device = queries.device
    packed, scales, scaled = ops.custom(
        "mo.csa.indexer.q_rope_quant",
        device=device,
        values=[
            queries,
            ops.cast(positions, DType.int64),
            rope_table,
            weights,
            _host_f32(weight_scale),
        ],
        out_types=[
            TensorType(DType.uint8, [t, n_heads * 64], device),
            TensorType(DType.uint8, [t, n_heads * 4], device),
            TensorType(DType.float32, [t, n_heads], device),
        ],
        parameters={"heads": n_heads},
    )
    return (
        ops.reshape(packed.tensor, [t, n_heads, 64]),
        scales.tensor,
        scaled.tensor,
    )


def index_key_slots(
    leaf: PagedCacheValues, rows: RaggedRows, *, ratio: int
) -> TensorValue:
    """Returns each row's ``[T]`` int64 slot in a ratio-pooled key leaf.

    Row ``t`` at position ``p`` writes key ``j = p // ratio``: slot
    ``block * keys_per_page + j % keys_per_page``, with ``block`` the
    request's physical block for page ``j // keys_per_page``.
    """
    keys_per_page = int(leaf.kv_blocks.shape[3])
    key = ops.floor_div(ops.cast(rows.positions, DType.int32), ratio)
    block = ops.gather_nd(
        leaf.lookup_table,
        ops.stack(
            [
                ops.cast(rows.bid, DType.int32),
                ops.floor_div(key, keys_per_page),
            ],
            axis=-1,
        ),
    )
    return ops.cast(block, DType.int64) * keys_per_page + ops.cast(
        key % keys_per_page, DType.int64
    )


def store_index_keys(
    leaf: PagedCacheValues,
    keys: TensorValue,
    positions: TensorValue,
    rope_table: TensorValue,
    norm_weight: TensorValue,
    slots: TensorValue,
    layer_idx: TensorValue,
    *,
    ratio: int,
    epsilon: float,
) -> None:
    """RMSNorms, RoPEs, quantizes and stores the group-closing rows' keys.

    The norm output and the rotated key each round to bf16. The trailing 64
    channels rotate at the group's first position.

    Args:
        leaf: The layer's index-key leaf.
        keys: ``[T, 128]`` bf16 projected keys before the norm.
        positions: ``[T]`` token positions.
        rope_table: From :func:`rope_cos_sin_table`.
        norm_weight: ``[128]`` RMSNorm weight.
        slots: From :func:`index_key_slots`.
        layer_idx: Host uint32 layer index within the leaf.
        ratio: Compression ratio, 1 or 2.
        epsilon: RMSNorm epsilon.
    """
    ops.inplace_custom(
        "mo.csa.indexer.k_store",
        device=keys.device,
        values=[
            keys,
            ops.cast(positions, DType.int64),
            rope_table,
            ops.cast(norm_weight, DType.bfloat16),
            slots,
            _host_f32(epsilon),
            leaf.kv_blocks,
            leaf.page_stride,
            layer_idx,
        ],
        parameters={"ratio": ratio},
    )


def select_index_keys(
    leaf: PagedCacheValues,
    q: tuple[TensorValue, TensorValue, TensorValue],
    offsets: TensorValue,
    layer_idx: TensorValue,
    *,
    top_k: int,
    ratio: int,
) -> TensorValue:
    """Returns ``[T, top_k]`` int32 visible key ids, unordered, ``-1``-padded.

    Args:
        leaf: The index-key leaf.
        q: :func:`quantize_queries`'s output.
        offsets: ``[B + 1]`` uint32 ragged row offsets.
        layer_idx: Host uint32 layer index within the leaf.
        top_k: Keys selected per row.
        ratio: Tokens per key.
    """
    packed, scales, weights = q
    t, n_heads = packed.shape[0], int(packed.shape[1])
    return ops.inplace_custom(
        "mo.csa.indexer.mxfp4",
        device=packed.device,
        values=[
            packed,
            scales,
            weights,
            offsets,
            *_cache_values(leaf),
            layer_idx,
        ],
        out_types=[TensorType(DType.int32, [t, top_k], packed.device)],
        parameters={"num_heads": n_heads, "top_k": top_k, "kpool": ratio},
    )[0].tensor


def select_index_keys_block_max(
    leaf: PagedCacheValues,
    q: tuple[TensorValue, TensorValue, TensorValue],
    offsets: TensorValue,
    layer_idx: TensorValue,
    *,
    top_k: int,
    ratio: int,
    block_size: int,
    num_blocks: int,
) -> tuple[TensorValue, TensorValue]:
    """:func:`select_index_keys` plus each row's per-block maximum score.

    Returns:
        ``(ids [T, top_k], block_max [T, num_blocks] float32)``; a block
        with no visible key is ``-inf``.
    """
    packed, scales, weights = q
    t, n_heads = packed.shape[0], int(packed.shape[1])
    device = packed.device
    ids, block_max = ops.inplace_custom(
        "mo.csa.indexer.mxfp4.block_max",
        device=device,
        values=[
            packed,
            scales,
            weights,
            offsets,
            *_cache_values(leaf),
            layer_idx,
        ],
        out_types=[
            TensorType(DType.int32, [t, top_k], device),
            TensorType(DType.float32, [t, num_blocks], device),
        ],
        parameters={
            "num_heads": n_heads,
            "top_k": top_k,
            "kpool": ratio,
            "block_size": block_size,
        },
    )
    return ids.tensor, block_max.tensor


def select_index_keys_masked(
    leaf: PagedCacheValues,
    q: tuple[TensorValue, TensorValue, TensorValue],
    offsets: TensorValue,
    keep: TensorValue,
    layer_idx: TensorValue,
    *,
    top_k: int,
    ratio: int,
    block_size: int,
) -> TensorValue:
    """:func:`select_index_keys` among the keys of each row's kept blocks.

    Args:
        keep: ``[T, num_blocks]`` bool, block ``b`` holding keys
            ``[b * block_size, (b + 1) * block_size)``.
    """
    packed, scales, weights = q
    t, n_heads = packed.shape[0], int(packed.shape[1])
    return ops.inplace_custom(
        "mo.csa.indexer.mxfp4.masked",
        device=packed.device,
        values=[
            packed,
            scales,
            weights,
            ops.cast(keep, DType.uint8),
            offsets,
            *_cache_values(leaf),
            layer_idx,
        ],
        out_types=[TensorType(DType.int32, [t, top_k], packed.device)],
        parameters={
            "num_heads": n_heads,
            "top_k": top_k,
            "kpool": ratio,
            "block_size": block_size,
        },
    )[0].tensor


# Block maxima rows are padded to a 16-byte multiple so every row of the
# top-k input stays vector aligned.
_BLOCK_MAX_ALIGN = 4


def candidate_num_blocks(max_seq_len: int, ratio: int, block_size: int) -> int:
    """Candidate blocks covering the most keys a request can hold.

    Rounded up to a multiple of four; the extra blocks hold no key.
    """
    blocks = -(-(max_seq_len // ratio) // block_size)
    return -(-blocks // _BLOCK_MAX_ALIGN) * _BLOCK_MAX_ALIGN


def candidate_keep_from_block_max(
    block_max: TensorValue,
    positions: TensorValue,
    *,
    ratio: int,
    block_size: int,
    topk_blocks: int,
) -> TensorValue:
    """Keeps each row's ``topk_blocks`` best blocks of keys.

    The block of the newest visible key, ``((p + 1) // ratio - 1) //
    block_size``, is forced in; ``-inf`` blocks are never kept. Among blocks
    tied with the ``k``-th best, the lowest ids fill the remaining picks.

    Args:
        block_max: ``[T, num_blocks]`` float32 per-block maximum scores.
        positions: ``[T]`` query positions.
        ratio: Tokens per key.
        block_size: Keys per block.
        topk_blocks: Blocks kept per row.

    Returns:
        ``[T, num_blocks]`` bool keep mask.
    """
    nblocks = int(block_max.shape[-1])
    device = block_max.device
    pos = ops.cast(positions, DType.int32)
    last = ops.floor_div(ops.floor_div(pos + 1, ratio) - 1, block_size)
    block_iota = ops.unsqueeze(
        ops.range(0, nblocks, dtype=DType.int32, device=device), 0
    )
    forced = ops.where(
        ops.unsqueeze(last, -1) == block_iota, float("inf"), block_max
    )
    k = min(topk_blocks, nblocks)
    # Picks come in descending order with ties by ascending block, so the
    # last pick is the k-th best value and the tied picks are the lowest ids.
    top_vals, top_ids = top_k_per_row(
        forced,
        ops.broadcast_to(
            ops.constant(k, DType.int64, device), [forced.shape[0]]
        ),
        k,
    )
    kth = top_vals[:, k - 1 : k]
    last_tied = ops.max(
        ops.where(top_vals == kth, ops.cast(top_ids, DType.int32), -1),
        axis=-1,
    )
    keep = (forced > kth) | ((forced == kth) & (block_iota <= last_tied))
    return keep & (forced > float("-inf"))


def key_positions(ids: TensorValue, ratio: int) -> TensorValue:
    """Maps key ids to the raw position of each key's group-closing token.

    ``-1`` stays ``-1``.
    """
    if ratio == 1:
        return ids
    return ops.where(ids >= 0, ids * ratio + (ratio - 1), -1)
