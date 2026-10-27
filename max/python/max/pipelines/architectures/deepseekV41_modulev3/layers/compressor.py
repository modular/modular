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
"""Pools raw token projections into pre-RoPE bf16 latents.

The pooling itself runs on main's registered V4.1 ops, exactly as the V2
pipeline runs them (``...deepseekV41.layers.compressor``):

* ratio 1 -- ``mo.csa.compress``: one RMS-normalized latent per row;
* ratio 2 -- ``mo.csa.compress.paged``: adjacent token pairs pool as
  ``sum_i kv_i * softmax_i(score_i)`` in float32, RMS-normalize without a
  bf16 rounding in between, then cast to bf16. A request's last raw rows
  save into its per-request ``c2a_state`` scratch ring (one float32
  ``2 * head_dim`` row per ring slot), and a request starting at an odd
  position pools its first pair against the row its own previous forward
  saved at ``start - 1`` -- the ratio-2 source carry,
  preserved across multi-token prefill and incremental decode.

The module owns the checkpoint's bf16 leaves
(``attn.compressor.{wkv,norm,wgate}.weight``); the pooling math runs in
fp32 against their promoted values, the same one-rounding promotion the
HF reference's fp32-declared linears perform.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from max import functional as F
from max.dtype import DType
from max.graph import TensorType, TensorValue, ops
from max.nn import Module
from max.nn.legacy.kv_cache import PagedCacheValues
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh

from .norm import RMSNorm

if TYPE_CHECKING:
    from ..model_config import DeepseekV41Config
    from .attention import RaggedRows

__all__ = [
    "Compressor",
    "compressed_visibility",
    "validate_chunk",
    "workspace_bytes",
]

_KERNEL_HEAD_DIM = 512
"""The only latent width the pooling ops support."""


def workspace_bytes(compress_ratio: int, head_dim: int = 512) -> int:
    """Returns the pooling op's scratch bytes per token row.

    The group's KV and score rows in float32, the bf16 latent staging,
    four int32 group maps and a presence flag per group slot.
    """
    return (
        2 * compress_ratio * head_dim * 4
        + head_dim * 2
        + 4 * 4
        + (compress_ratio)
    )


def compressed_visibility(
    pos_q: Tensor, pos_k: Tensor, compress_ratio: int
) -> Tensor:
    """Whether entry ``pos_k`` (an *entry* index) is visible at ``pos_q``.

    A query at token position ``pos_q`` sees entry ``j`` once every
    token of its group is at or before the query: ``j <
    (pos_q + 1) // ratio`` -- the HF reference's ``compress_lens``
    mask (``model.py:562-567``), in one expression. ``pos_q``
    broadcasts against ``pos_k``.
    """
    return pos_k < (pos_q + 1) // compress_ratio


def validate_chunk(
    starts: Sequence[int], lengths: Sequence[int], *, has_carry: Sequence[bool]
) -> None:
    """Host-side check: every odd-start request must own its predecessor.

    A chunk starting at odd position ``P`` pools its first pair against
    the raw row at ``P - 1``, which lives in the request's own state
    ring; a request with no such row (a fresh sequence, whose ring holds
    nothing it wrote) must be rejected before launch -- never silently
    published from stale ring bytes.
    """
    for i, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        if length == 0:
            continue
        if start % 2 == 1 and not has_carry[i]:
            raise ValueError(
                f"chunk for request {i} starts at odd position {start} "
                "without its predecessor row in the state ring; an "
                "odd-start pair cannot be pooled"
            )


def _ring_slots(
    state: PagedCacheValues, bid: TensorValue, positions: TensorValue
) -> TensorValue:
    """Each row's ``[T]`` int32 slot in its request's state ring.

    ``block * ring_rows + position % ring_rows``, where the block is
    column zero of the ring leaf's lookup table.
    """
    ring_rows = int(state.kv_blocks.shape[3])
    first_col = ops.constant(0, DType.int32, bid.device).broadcast_to(bid.shape)
    block = ops.gather_nd(state.lookup_table, ops.stack([bid, first_col], -1))
    return ops.cast(block, DType.int32) * ring_rows + ops.cast(
        positions % ring_rows, DType.int32
    )


def _ring_save_slots(
    ring_rows: int,
    bid: TensorValue,
    local: TensorValue,
    lengths: TensorValue,
    slots: TensorValue,
) -> TensorValue:
    """``slots`` with ``-1`` on rows outside their request's live
    ``ring_rows`` rows, so the ring save copies only the rows that
    survive a wrap.

    The live upper bound matters as much as the wrap bound: a captured
    decode pads its row axis past ``row_offsets[-1]``, and a padded
    row's ``local`` is at or past its request's length, so the wrap
    bound alone would let the padded rows wrap the ring and overwrite
    the valid carry with padding garbage.
    """
    length = ops.gather(lengths, bid, axis=0)
    keep = (local >= length - ring_rows) & (local < length)
    return ops.where(keep, slots, -1)


def _compress_pool(
    raw: TensorValue,
    positions: TensorValue,
    offsets: TensorValue,
    norm_weight: TensorValue,
    bid: TensorValue,
    local: TensorValue,
    lengths: TensorValue,
    *,
    ratio: int,
    epsilon: float,
    state: PagedCacheValues | None = None,
    layer_idx: TensorValue | None = None,
) -> TensorValue:
    """The ``mo.csa.compress`` / ``mo.csa.compress.paged`` invocation.

    A free function wrapped in :data:`compress_pool_op` so the
    experimental-tensor call traces it per device. ``state``, ``layer_idx``
    and the scalars arrive as keyword-only closure parameters, the
    :class:`~...layers.engram` lookup pattern.
    """
    if offsets.dtype != DType.uint32:
        raise ValueError("CSA2 offsets must be uint32")
    device = raw.device
    rows = raw.shape[0]
    out_types = [
        TensorType(DType.bfloat16, [rows, _KERNEL_HEAD_DIM], device),
        TensorType(
            DType.uint8,
            [rows * workspace_bytes(ratio, _KERNEL_HEAD_DIM)],
            device,
        ),
    ]
    values = [raw, ops.cast(positions, DType.int32), offsets]
    if ratio == 1:
        if state is not None or layer_idx is not None:
            raise ValueError("ratio 1 takes no state ring")
        return ops.custom(
            "mo.csa.compress",
            device=device,
            values=[*values, norm_weight],
            out_types=out_types,
            parameters={"epsilon": epsilon},
        )[0].tensor
    if state is None or layer_idx is None:
        raise ValueError("ratio 2 requires the state ring and layer index")
    ring_rows = int(state.kv_blocks.shape[3])
    slots = _ring_slots(state, bid, positions)
    save_slots = _ring_save_slots(ring_rows, bid, local, lengths, slots)
    return ops.inplace_custom(
        "mo.csa.compress.paged",
        device=device,
        values=[
            *values,
            ops.cast(slots, DType.int32),
            ops.cast(save_slots, DType.int32),
            norm_weight,
            state.kv_blocks,
            state.page_stride,
            layer_idx,
        ],
        out_types=out_types,
        parameters={"epsilon": epsilon},
    )[0].tensor


compress_pool_op = F.functional(_compress_pool)


class _Weight(Module[..., None]):
    """A bare ``.weight`` leaf: the checkpoint path
    ``...compressor.{wkv,norm,wgate}.weight`` maps attribute -> sub-module
    -> ``weight``, so each parameter is a one-attribute module.
    """

    def __init__(self, shape: Sequence[int], dtype: DType) -> None:
        super().__init__()
        self.weight = Tensor.zeros(shape, dtype=dtype)


class Compressor(Module[..., tuple[Tensor, Tensor]]):
    """Pools raw token projections into pre-RoPE bf16 latents.

    Constructor signature: ``(config, ratio, mesh)``; parameters
    ``wkv`` and ``norm`` at both ratios plus ``wgate`` at ratio 2,
    exactly the names the checkpoint's
    ``attn.compressor.{wkv,norm,wgate}.weight`` paths load into. All
    weights are bf16 leaves; the pooling math runs in fp32 against
    their promoted values.

    Args:
        config: The resolved model config (``hidden_size``, ``head_dim``
            and ``rms_norm_eps`` are the only fields read).
        ratio: 2 (gated pairs) or 1 (plain normalization).
        mesh: The TP mesh the module is placed on.
    """

    wkv: _Weight
    """``[head_dim, hidden]`` bf16 value projection (``.weight``)."""
    wgate: _Weight
    """``[head_dim, hidden]`` bf16 gate-score projection (``.weight``); ratio 2 only."""
    norm: RMSNorm
    """``[head_dim]`` bf16 learned per-element scale (``.weight``)."""

    def __init__(
        self, config: DeepseekV41Config, ratio: int, mesh: DeviceMesh
    ) -> None:
        super().__init__()
        if ratio not in (1, 2):
            raise ValueError(f"compress ratio must be 1 or 2, got {ratio}")
        if config.head_dim != _KERNEL_HEAD_DIM:
            raise ValueError(
                f"CSA2 compressor head_dim must be {_KERNEL_HEAD_DIM}, got "
                f"{config.head_dim}"
            )
        self.config = config
        self.ratio = ratio
        self.mesh = mesh
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.wkv = _Weight([self.head_dim, self.hidden_size], DType.bfloat16)
        self.norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if ratio == 2:
            self.wgate = _Weight(
                [self.head_dim, self.hidden_size], DType.bfloat16
            )

    def project(self, h: Tensor) -> Tensor:
        """Raw fp32 projections ``[T, head_dim * ratio]`` for pooling.

        ``wkv(h)`` first and, at ratio 2, ``wgate(h)`` second -- the
        exact layout the pooling op consumes (values then gate scores,
        both fp32; the checkpoint keeps both bf16 and the math runs
        against the promoted fp32 weights, the same one-rounding
        promotion the HF reference's fp32-declared linears perform).
        """
        h32 = F.cast(h, DType.float32)
        kv = h32 @ F.cast(self.wkv.weight, DType.float32).T
        if self.ratio == 1:
            return kv
        gate = h32 @ F.cast(self.wgate.weight, DType.float32).T
        return F.concat([kv, gate], axis=-1)

    def forward(
        self,
        raw: Tensor,
        rows: RaggedRows,
        state: PagedCacheValues | None = None,
        layer_idx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Pools raw rows into masked pre-RoPE latents.

        Args:
            raw: fp32 projections ``[T, head_dim * ratio]``
                (:meth:`project`'s output): the ``wkv`` row, then (ratio
                2) the ``wgate`` score row.
            rows: The forward's ragged bookkeeping (``bid``, ``local``,
                ``positions`` and per-request ``lengths`` tensors), as
                the trunk computes them.
            state: Ratio 2 only: this layer's ``c2a_state`` scratch ring
                leaf, as graph values (``to_graph_values()``). Like
                ``layer_idx``, dropped at ratio 1, so a caller can pass
                both the same way at every layer.
            layer_idx: Ratio 2 only: host uint32 index of this layer in
                ``state``.

        Returns:
            ``(latent, valid)``: ``[T, head_dim]`` bf16 pre-RoPE latents
            (rows that close no group zero) and the ``[T]`` bool
            publication mask. A row publishes exactly when it closes a
            group: at ratio 1 every row, at ratio 2 each row at an odd
            position. Whether an odd-position row's pair actually pools
            is the ring's contract (:func:`validate_chunk` guards it on
            the host); the pooled latent itself is zero on rows that
            close no group.
        """
        if raw.dtype != DType.float32 or raw.shape[-1] != self.head_dim * (
            self.ratio
        ):
            raise ValueError(
                "CSA2 raw projections have the wrong dtype or width"
            )
        latent = compress_pool_op(
            raw,
            rows.positions,
            rows.offsets,
            F.cast(self.norm.weight, DType.float32),
            rows.bid,
            rows.local,
            rows.lengths,
            ratio=self.ratio,
            epsilon=self.norm.eps,
            state=None if self.ratio == 1 else state,
            layer_idx=None if self.ratio == 1 else layer_idx,
        )
        valid = (rows.positions + 1) % self.ratio == 0
        return latent, valid
