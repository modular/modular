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
consecutive keys; see :func:`select_index_keys_block_topk`.
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
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm
from max.nn.quant_config import QuantConfig

from .mxfp8_matmul import mxfp8_linear
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
        wq_b_quant_config: QuantConfig | None = None,
    ) -> None:
        """Initializes the indexer.

        Args:
            owns_k: Whether this layer produces its own index keys (creates
                ``wk``/``k_norm``).
            wq_b_quant_config: An MXFP8 config makes ``wq_b``
                ``float8_e4m3fn`` with block scales; ``None`` keeps it in
                ``dtype``.

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
            q_lora_rank,
            self.n_heads * self.head_dim,
            dtype if wq_b_quant_config is None else DType.float8_e4m3fn,
            device,
            quant_config=wq_b_quant_config,
        )
        # Uses the model `dtype` rather than a fixed bf16.
        self.weights_proj = Linear(hidden_size, self.n_heads, dtype, device)

        if owns_k:
            self.wk = Linear(main_head_dim, self.head_dim, dtype, device)
            self.k_norm = RMSNorm(self.head_dim, DType.float32, rms_norm_eps)

    def __call__(
        self,
        qr: TensorValue,
        x: TensorValue,
        qr_scales: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Projects the index queries and per-head weights.

        Args:
            qr: ``[..., T, q_lora_rank]`` normalized low-rank query, or its
                ``float8_e4m3fn`` quantization when ``qr_scales`` is given.
            x: ``[..., T, hidden_size]`` raw hidden state.
            qr_scales: The E8M0 block scales of a quantized ``qr``.

        Returns:
            ``([T, n_heads * head_dim] queries, [T, n_heads] weights)``,
            before RoPE, quantization and weight scaling.
        """
        return (
            ops.reshape(
                mxfp8_linear(self.wq_b, qr, qr_scales),
                [-1, self.n_heads * self.head_dim],
            ),
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
        parameters={
            "num_heads": n_heads,
            "top_k": top_k,
            "kpool": ratio,
            "scores_budget_bytes": INDEXER_SCORES_BUDGET_BYTES,
        },
    )[0].tensor


# The indexer ops score query rows in windows that fit this transient budget
# of bfloat16 scores, whatever the batch or context length; every indexer op
# takes it as a parameter. The candidate ops' ``[rows, blocks]`` scratch holds
# as many rows as one score window, so it takes at most a quarter of it.
INDEXER_SCORES_BUDGET_BYTES = 512 << 20


def candidate_window_rows(rows: int, num_blocks: int, block_size: int) -> int:
    """Rows of the candidate ops' window scratch."""
    budget = INDEXER_SCORES_BUDGET_BYTES // max(1, num_blocks * block_size * 2)
    return max(1, min(rows, budget))


def select_index_keys_block_topk(
    leaf: PagedCacheValues,
    q: tuple[TensorValue, TensorValue, TensorValue],
    offsets: TensorValue,
    positions: TensorValue,
    layer_idx: TensorValue,
    *,
    top_k: int,
    ratio: int,
    block_size: int,
    topk_blocks: int,
) -> tuple[TensorValue, TensorValue]:
    """:func:`select_index_keys` plus each row's ``topk_blocks`` best blocks.

    A block of ``block_size`` keys scores the maximum of its visible keys.
    The block of the newest visible key, ``((p + 1) // ratio - 1) //
    block_size``, is always kept, blocks past it or with no visible key
    never are, and which blocks tied with the last kept value are kept is
    unspecified.

    Args:
        positions: ``[T]`` query positions.
        topk_blocks: Blocks kept per row.

    Returns:
        ``(ids [T, top_k], kept [T, topk_blocks] int32)``; kept block ids
        come in no particular order, with ``-1`` in unused slots.
    """
    packed, scales, weights = q
    t, n_heads = packed.shape[0], int(packed.shape[1])
    device = packed.device
    ids, kept = ops.inplace_custom(
        "mo.csa.indexer.mxfp4.block_max",
        device=device,
        values=[
            packed,
            scales,
            weights,
            ops.cast(positions, DType.int32),
            offsets,
            *_cache_values(leaf),
            layer_idx,
        ],
        out_types=[
            TensorType(DType.int32, [t, top_k], device),
            TensorType(DType.int32, [t, topk_blocks], device),
        ],
        parameters={
            "num_heads": n_heads,
            "top_k": top_k,
            "kpool": ratio,
            "block_size": block_size,
            "scores_budget_bytes": INDEXER_SCORES_BUDGET_BYTES,
        },
    )
    return ids.tensor, kept.tensor


def select_index_keys_masked(
    leaf: PagedCacheValues,
    q: tuple[TensorValue, TensorValue, TensorValue],
    offsets: TensorValue,
    kept: TensorValue,
    layer_idx: TensorValue,
    *,
    top_k: int,
    ratio: int,
    block_size: int,
) -> TensorValue:
    """:func:`select_index_keys` among the keys of each row's kept blocks.

    Args:
        kept: ``[T, kept]`` int32 kept block ids, ``-1`` for none; block
            ``b`` holds keys ``[b * block_size, (b + 1) * block_size)``.
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
            kept,
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
            "scores_budget_bytes": INDEXER_SCORES_BUDGET_BYTES,
        },
    )[0].tensor


def candidate_num_blocks(max_seq_len: int, ratio: int, block_size: int) -> int:
    """Candidate blocks covering the most keys a request can hold."""
    return -(-(max_seq_len // ratio) // block_size)


def candidate_keep_blocks(
    max_seq_len: int,
    position_capacity: int,
    compress_ratio: int,
    *,
    block_size: int,
    topk_blocks: int,
) -> int:
    """Returns the candidate keep step's blocks per row, or 0 to skip it.

    The graph gates the keep step and the memory planner sizes it here. The
    gate reads ``max_seq_len``: a row past it, in the speculative slack,
    never commits a token, so blocks only such a row reaches are no reason
    to run the pass. The blocks cover ``position_capacity``, since the
    kernel sizes its scratch from the cached lengths, which reach into the
    slack when the pass does run.
    """
    if not candidate_blocks_active(
        max_seq_len,
        compress_ratio,
        block_size=block_size,
        topk_blocks=topk_blocks,
    ):
        return 0
    return candidate_num_blocks(position_capacity, compress_ratio, block_size)


def key_positions(ids: TensorValue, ratio: int) -> TensorValue:
    """Maps key ids to the raw position of each key's group-closing token.

    ``-1`` stays ``-1``.
    """
    if ratio == 1:
        return ids
    return ops.where(ids >= 0, ids * ratio + (ratio - 1), -1)
