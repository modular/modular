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
"""Graph ops of the MXFP4 sparse-attention indexer, ModuleV3 wiring.

Thin bindings of the ``mo.csa.indexer.*`` ops (registered in
``Kernels/lib/graph_compiler/csa_indexer.mojo``) over this stack's
``PagedCacheValues`` leaves. Index keys live in the MXFP4 index
leaf -- one 68-byte record per entry (64 E2M1 code bytes then four
UE8M0 scale bytes), in ``indexer_k_store``'s split-record page layout --
so the scorer reads them straight out of the pages and the candidate
axis never materializes.

- :func:`indexer_q_rope_quant` rotates and quantizes queries to MXFP4 and
  scales the per-head weights.
- :func:`indexer_k_store` normalizes, rotates, quantizes and stores the
  group-closing rows' index keys.
- :func:`indexer_select` scores and selects each row's top-k visible
  entries; :func:`indexer_select_block_topk` also keeps each row's best
  candidate blocks, and :func:`indexer_select_masked` restricts the
  selection to kept blocks.

None of the ops synchronize with the host, so they capture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType, TensorValue, Value, ops
from max.nn.legacy.kv_cache import PagedCacheValues

__all__ = [
    "INDEXER_SCORES_BUDGET_BYTES",
    "indexer_k_store",
    "indexer_q_rope_quant",
    "indexer_select",
    "indexer_select_block_topk",
    "indexer_select_masked",
]


# The select ops score query rows in windows that fit this transient budget of
# bfloat16 scores, whatever the batch or context length. Every select op takes
# it as a parameter, and the memory planner reserves the same figure.
INDEXER_SCORES_BUDGET_BYTES = 512 << 20


def _host_f32(value: float) -> Value[Any]:
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


def indexer_q_rope_quant_rows(
    queries: TensorValue,
    weights: TensorValue,
    positions: TensorValue,
    rope_table: TensorValue,
    *,
    n_heads: int,
    weight_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """RoPEs and quantizes index queries to MXFP4, scaling the weights.

    Args:
        queries: ``[T, n_heads * 128]`` bf16 projected queries, pre-RoPE.
        weights: ``[T, n_heads]`` bf16 per-head weights, unscaled.
        positions: ``[T]`` int64 token positions, below ``rope_table``'s
            row count.
        rope_table: ``[P, 64]`` fp32 cosines (first 32) then sines per
            absolute position, one row per position.
        n_heads: Index heads per token.
        weight_scale: Host factor folded into the returned weights.

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
        Tensor.from_graph_value(ops.reshape(packed.tensor, [t, n_heads, 64])),
        Tensor.from_graph_value(scales.tensor),
        Tensor.from_graph_value(scaled.tensor),
    )


def indexer_k_store_rows(
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
    """RMSNorms, RoPEs, quantizes and stores index keys into ``leaf``.

    Rows whose slot is negative, and rows that do not close a group of
    ``ratio`` positions, store nothing.

    Args:
        leaf: The packed MXFP4 index-key leaf (uint8 blocks, 72-byte
            rows holding 68-byte records).
        keys: ``[T, 128]`` bf16 projected keys, before the norm.
        positions: ``[T]`` int64 token positions.
        rope_table: ``[P, 64]`` fp32 cosines then sines per position.
        norm_weight: ``[128]`` bf16 RMSNorm weights.
        slots: ``[T]`` int64 destination slots,
            ``block * keys_per_page + offset``; negative skips the row.
        layer_idx: Host uint32 layer index within the leaf.
        ratio: Compression group size, 1 or 2.
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
            ops.cast(slots, DType.int64),
            _host_f32(epsilon),
            leaf.kv_blocks,
            leaf.page_stride,
            layer_idx,
        ],
        parameters={"ratio": ratio},
    )


def _select(
    op_name: str,
    q: tuple[TensorValue, TensorValue, TensorValue],
    offsets: TensorValue,
    leaf: PagedCacheValues,
    layer_idx: TensorValue,
    *,
    top_k: int,
    ratio: int,
    extra_values: Sequence[TensorValue] = (),
    extra_out_types: Sequence[TensorType] = (),
    parameters: Mapping[str, bool | int] | None = None,
) -> tuple[Tensor, ...]:
    """Calls one select op; ``extra_values`` sit after the weights.

    The registered kernels declare their extra input (the block top-k
    op's positions, the masked op's kept block ids) before
    ``input_row_offsets``, so it joins the leading values, not the tail.
    """
    packed, scales, weights = q
    t, n_heads = packed.shape[0], int(packed.shape[1])
    results = ops.inplace_custom(
        op_name,
        device=packed.device,
        values=[
            packed,
            scales,
            weights,
            *extra_values,
            offsets,
            *_cache_values(leaf),
            layer_idx,
        ],
        out_types=[
            TensorType(DType.int32, [t, top_k], packed.device),
            *extra_out_types,
        ],
        parameters={
            "num_heads": n_heads,
            "top_k": top_k,
            "kpool": ratio,
            "scores_budget_bytes": INDEXER_SCORES_BUDGET_BYTES,
            **(parameters or {}),
        },
    )
    return tuple(Tensor.from_graph_value(r.tensor) for r in results)


def indexer_select_rows(
    leaf: PagedCacheValues,
    q: tuple[TensorValue, TensorValue, TensorValue],
    offsets: TensorValue,
    layer_idx: TensorValue,
    *,
    top_k: int,
    ratio: int,
) -> Tensor:
    """Selects each row's top-k visible entries, ``-1``-padded, unordered."""
    return _select(
        "mo.csa.indexer.mxfp4",
        q,
        offsets,
        leaf,
        layer_idx,
        top_k=top_k,
        ratio=ratio,
    )[0]


def indexer_select_block_topk_rows(
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
) -> tuple[Tensor, Tensor]:
    """:func:`indexer_select` plus each row's ``topk_blocks`` best blocks.

    A block of ``block_size`` entries scores the maximum of its visible
    entries. The block of the newest visible entry, ``((p + 1) // ratio - 1)
    // block_size``, is always kept, blocks past it or with no visible entry
    never are, and which blocks tied with the last kept value are kept is
    unspecified.

    Args:
        positions: ``[T]`` token positions.

    Returns:
        ``(ids [T, top_k], kept [T, topk_blocks] int32)``; kept block ids
        come in no particular order, ``-1`` in unused slots.
    """
    device = q[0].device
    ids, kept = _select(
        "mo.csa.indexer.mxfp4.block_max",
        q,
        offsets,
        leaf,
        layer_idx,
        top_k=top_k,
        ratio=ratio,
        extra_values=[ops.cast(positions, DType.int32)],
        extra_out_types=[
            TensorType(DType.int32, [q[0].shape[0], topk_blocks], device)
        ],
        parameters={"block_size": block_size},
    )
    return ids, kept


def indexer_select_masked_rows(
    leaf: PagedCacheValues,
    q: tuple[TensorValue, TensorValue, TensorValue],
    offsets: TensorValue,
    kept: TensorValue,
    layer_idx: TensorValue,
    *,
    top_k: int,
    ratio: int,
    block_size: int,
) -> Tensor:
    """:func:`indexer_select` among the entries of each row's kept blocks.

    A row that sees at most ``top_k`` entries selects all of them, so the
    kept blocks restrict only longer rows.

    Args:
        kept: ``[T, kept]`` int32 kept block ids, ``-1`` for none; block
            ``b`` holds entries ``[b * block_size, (b + 1) * block_size)``.
    """
    return _select(
        "mo.csa.indexer.mxfp4.masked",
        q,
        offsets,
        leaf,
        layer_idx,
        top_k=top_k,
        ratio=ratio,
        extra_values=[kept],
        parameters={"block_size": block_size},
    )[0]


indexer_q_rope_quant = F.functional(indexer_q_rope_quant_rows)
indexer_k_store = F.functional(indexer_k_store_rows)
indexer_select = F.functional(indexer_select_rows)
indexer_select_block_topk = F.functional(indexer_select_block_topk_rows)
indexer_select_masked = F.functional(indexer_select_masked_rows)
