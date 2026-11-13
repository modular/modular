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
"""Mistral Large 4's DSpark drafter: three dense MLA layers and its head.

The drafter is in the target checkpoint (``mtp_dspark.*``) and drafts three
tokens per target step:

* the five target taps of every context row are concatenated, projected to
  ``hidden`` and normalized (:class:`DSparkDrafter`'s ``target_projection`` +
  ``target_norm``), and each layer turns that stream into its own 576-wide MLA
  latent key -- 512 normalized content dims and a 64-wide rotated rope key,
  with no value half;
* a block of ``depth`` rows -- the anchor token's embedding followed by
  ``depth - 1`` learned mask rows -- runs **non-causally** over the last
  ``window`` context latents plus the block's own latents;
* the block rows are scored with the *target's* LM head and soft-capped, and
  :class:`DSparkMarkovHead` walks a gated recurrent chain across the rows that
  adds a rank-512 token-conditioned bias to each row's logits.

Three details are part of the trained parameter and are easy to get wrong:
the joint projection's concat order is ``[state | markov_embedding(prev) |
hidden]``, its output chunks are ``gate | candidate | emitted`` in that order,
and the soft cap applies to the shared-head logits **before** the recurrent
bias is added and never to the sum. The reference is Mistral's DSpark write-up
(§4-§9), which this module transcribes.

Attention is absorbed MLA over the latents, in the activation dtype with
float32 scores, softmax and accumulation, and it is *not* causal: every block
row sees every valid key, the block's own rows included. The key set is a
context window (the last ``window`` latents up to and including the anchor)
plus the ``depth`` block latents.

Two entries read that key set, and they differ only in where the window's rows
come from:

* :meth:`DSparkAttention.forward` takes the whole pool as one ``[n_ctx,
  latent]`` tensor and selects the window with a boolean ``[rows, keys]`` mask.
  That is the dense entry: a caller holding a whole prompt's taps (the parity
gate) runs it, and the mask is bounded on *both* ends so such a caller cannot
  leak the tokens after its own anchor into the block.
* :meth:`DSparkAttention.forward_paged` *materializes* the window instead --
  :func:`window_reads` names the ``window`` cache slots each request's block
  reads and one gather copies exactly those rows out of the drafter's paged
  leaf (:func:`gather_window_rows`); the block's own rows are multiplied
  separately and the two *score* rectangles are what meet, so no tensor ever
  holds both key sets. A request whose context is shorter than the window
  fills the rectangle by gathering its anchor row again and masking the
  repeats, so the attended key set is the dense path's. This is the serving
  entry: the block driver holds no dense pool, only the per-layer latents
  :meth:`DSparkDrafter.write_context` wrote in earlier steps.

Neither path materializes a key it must not read. The dense policy is
inherited rather than enforced: a masked key is dropped by an exactly-zero
softmax probability, which does not clear a NaN in it (``0 * NaN`` is
``NaN``), so a partly-materialized pool must leave its dead rows out. The
paged path cannot violate that -- it reads allocated cache rows only -- and a
request shorter than the window fills the rectangle by gathering its own
anchor row again, which the same kind of mask then drops: a key gathered
twice would otherwise carry twice the weight, not the same weight.

The drafter owns no LM head and no token embedding: both are the target's and
arrive as call arguments, never as parameters.

Its FFN takes the target's dense-FFN tensor-parallel split (:class:`DSparkMLP`):
gate and up are column-parallel, down is row-parallel and one all-reduce
completes the layer's sum. Its attention takes the target's head-axis split
(:class:`DSparkAttention`): ``wq`` and ``wkv_b`` shard on their head-major
output rows and ``wo`` on its head-major input columns, so each rank owns a
contiguous block of heads and one further all-reduce completes the
out-projection; the latent projection and its norm stay whole, because the
latent is the contraction's K on both sides of that matmul. At eight ranks
that is one eighth of the drafter's projection weights read per rank per step
instead of all of them. Both splits follow the mesh's name -- on a ``tp`` axis
only -- because a data-parallel rank is a replica whose requests belong to it
alone, and nothing there would complete the partial sums those row-parallel
projections leave.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Embedding, Linear, Module, ModuleList, RMSNorm
from max.nn._graph_ops import KEY_CACHE_INDEX, kv_cache_gather_rows_ragged
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.mesh_axis import TP
from max.nn.kernels import store_k_cache_ragged
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    NamedMapping,
    Replicated,
)
from numpy.typing import NDArray

__all__ = [
    "BLOCK_POSITION_OFFSET_TRTLLM",
    "BLOCK_POSITION_OFFSET_VLLM",
    "DSparkAttention",
    "DSparkBackbone",
    "DSparkLayer",
    "DSparkMLP",
    "DSparkMarkovHead",
    "WindowReads",
    "cos_sin_at",
    "device_scalar",
    "gather_window_rows",
    "negative_infinity",
    "rotate_interleaved",
    "splits_over_tensor_parallel",
    "store_context_rows",
    "window_reads",
    "window_slots",
]

_gather_rows = F.functional(kv_cache_gather_rows_ragged)
"""``kv_cache_gather_rows_ragged`` with the ModuleV3 ``Tensor`` boundary.

The op is a pure copy out of one paged cache leaf, addressed by logical slots,
so it takes this module's tensors and its distributed ``PagedCacheValues``
directly (``F.functional`` lowers the record to the graph op's per-device
form). It reads caches that hold one head per slot, which is what an MLA
leaf is."""

BLOCK_POSITION_OFFSET_TRTLLM = 1
"""The block's first row sits at ``anchor_position + 1``.

TensorRT-LLM's ML4 drafter computes ``query_position_ids = ctx_len_now + j``
with ``ctx_len_now`` the number of context rows *including* the anchor's own,
so row ``j`` is rotated at the position of the token it predicts. SGLang's
generic DSpark draft resolves the same layout: its block positions are
``prefix_lens + arange(block)`` and the anchor is the token committed at
``prefix_lens - 1``.
"""

BLOCK_POSITION_OFFSET_VLLM = 0
"""The block's first row sits at the anchor's own position (vLLM's layout)."""


def device_scalar(
    value: int, like: Tensor, dtype: DType | None = None
) -> Tensor:
    """A ``[]`` scalar constant on ``like``'s device, to compare against it.

    A bare Python number beside a device tensor is built on the CPU, and an
    elementwise op then refuses two devices. The dtype defaults to ``like``'s:
    an elementwise comparison needs both sides in one dtype, and an anchor
    tensor's integer dtype is the caller's choice, not this helper's.

    Args:
        value: The constant.
        like: The tensor the constant is compared with or gathered from.
        dtype: The constant's dtype; ``like.dtype`` when unset.

    Returns:
        The scalar, on ``like``'s device.
    """
    return F.constant(
        value, like.dtype if dtype is None else dtype, device=like.mapping
    )


def cos_sin_at(
    inv_freq: NDArray[np.floating], positions: Tensor, dtype: DType
) -> Tensor:
    """Returns ``[rows, width // 2, 2]`` cos/sin at ``positions``.

    The angle is the float32 product of the position and the inverse
    frequency, and the table is rounded through the activation dtype: the
    engine's rope tables live in the activation dtype, so this reproduces the
    rounding a precomputed table would carry.

    Args:
        inv_freq: The ``width // 2`` inverse frequencies, plain YaRN.
        positions: ``[rows]`` integer positions.
        dtype: The activation dtype the table rounds through.

    Returns:
        The cos/sin pairs, float32, in the layout
        :func:`rotate_interleaved` reads.
    """
    inv = F.constant(inv_freq, DType.float32, device=positions.mapping)
    angles = F.outer(F.cast(positions, DType.float32), inv)
    cos_sin = F.stack([F.cos(angles), F.sin(angles)], axis=-1)
    return F.cast(F.cast(cos_sin, dtype), DType.float32)


def window_slots(
    anchor_positions: Tensor, window: int
) -> tuple[Tensor, Tensor]:
    """Returns each block's context window as slots, and which are real.

    Request ``b`` reads slots ``[max(0, anchor + 1 - window), anchor]``, so a
    context shorter than the window reads only the slots it has. The window is a
    fixed ``window`` slots wide whatever the context, because a batch's keys are
    one rectangular tensor; the slots past the anchor are clamped onto the
    anchor's own row, which is a row the request *has* written and which is
    finite -- never an unmaterialized page, whose bytes could be a NaN that an
    exactly-zero softmax probability leaves alone (``0 * NaN`` is ``NaN``).
    Those clamped entries come back masked out by the second return value: a key
    gathered twice is not neutral, it carries twice the weight.

    Args:
        anchor_positions: ``[batch]`` each block's anchor position.
        window: The drafter's sliding window.

    Returns:
        ``(slots [batch, window] int32, keep [batch, window] bool)``, ascending
        within a request: the slots to gather, and which of them are in the
        window rather than a repeat of the anchor.
    """
    batch = anchor_positions.shape[0]
    device = anchor_positions.mapping
    zero = device_scalar(0, anchor_positions, DType.int32)
    anchors = F.reshape(anchor_positions.cast(DType.int32), [batch, 1])
    # The window's first slot, floored at zero; a Python ``-window`` beside the
    # device tensor would be built on the host and mix devices.
    first = anchors - device_scalar(window - 1, anchor_positions, DType.int32)
    first = F.where(first > zero, first, F.broadcast_to(zero, [batch, 1]))
    offsets = F.arange(
        0, window, 1, out_dim=window, dtype=DType.int32, device=device
    )
    wanted = first + F.reshape(offsets, [1, window])
    keep = wanted <= anchors
    return F.where(keep, wanted, F.broadcast_to(anchors, [batch, window])), keep


@dataclass(frozen=True)
class WindowReads:
    """One step's window addressing, shared by every layer of the block.

    A block reads the same rectangle of slots once per layer; only the layer it
    reads from changes. Everything about the addressing that does not name a
    layer is therefore built once per step (:func:`window_reads`) instead of
    once per layer: the slots, the mask over them, the ragged offsets the
    gather needs, and the block-visibility row every layer's mask ends with.
    """

    anchor: Tensor
    """``[batch]`` int32 each block's anchor position, clamped to the leaf."""

    slots: Tensor
    """``[batch, window]`` int32 the cache slots each block's window reads."""

    keep: Tensor
    """``[batch, window]`` bool which of those slots are in the window."""

    row_offsets: Tensor
    """``[batch + 1]`` uint32 the gather's ragged offsets for ``slots``."""

    own_visible: Tensor
    """``[batch, depth]`` bool: a block row sees every block row of its request.

    Constant for a step -- the block is block diagonal by construction, so no
    query row masks a sibling -- but built once here rather than in each layer's
    attention.
    """


def window_reads(
    anchor_positions: Tensor,
    cache_lengths: Tensor,
    window: int,
    depth: int,
) -> WindowReads:
    """Returns one step's window addressing, clamped to the leaf's own rows.

    The slots are :func:`window_slots`' for the anchor, and the anchor is the
    caller's -- derived from the leaf's lengths rather than read from them, so
    an off-by-N caller could name a row the request never wrote. The gather
    resolves a slot through the request's page table and guards nothing, so
    that would be an unallocated page; clamp the anchor to the last row the
    leaf holds and the gathered slots stay inside it.

    Args:
        anchor_positions: ``[batch]`` each block's anchor position.
        cache_lengths: ``[batch]`` the leaf's rows per request, in the cache's
            dtype (uint32).
        window: The drafter's sliding window.
        depth: Rows per block.

    Returns:
        The step's :class:`WindowReads`.
    """
    batch = anchor_positions.shape[0]
    device = anchor_positions.mapping
    anchors = anchor_positions.cast(DType.int32)
    # ``cache_lengths - 1`` is the last written row; a request with no rows at
    # all would give -1, so floor the clamp at zero as well. A zero-length
    # request is refused earlier, by the block driver's own staging. The two
    # vectors come from different tensors, so their lengths are different
    # symbolic dims: rebind the leaf's to the anchor's before comparing them.
    lengths = F.cast(
        F.rebind(cache_lengths, anchor_positions.shape), DType.int32
    )
    last = lengths - device_scalar(1, anchor_positions, DType.int32)
    zero = device_scalar(0, anchor_positions, DType.int32)
    anchors = F.max(F.min(anchors, last), F.broadcast_to(zero, [batch]))
    slots, keep = window_slots(anchors, window)
    # One row of ``slots`` per request, so the ragged offsets are the row
    # indices themselves.
    row_offsets = F.arange(
        0, batch + 1, 1, out_dim=batch + 1, dtype=DType.uint32, device=device
    )
    own_visible = F.full([batch, depth], True, dtype=DType.bool, device=device)
    return WindowReads(
        anchor=anchors,
        slots=slots,
        keep=keep,
        row_offsets=row_offsets,
        own_visible=own_visible,
    )


def gather_window_rows(
    kv: PagedCacheValues,
    slots: Tensor,
    row_offsets: Tensor,
    layer_idx: int,
) -> Tensor:
    """Copies the slots' latent rows out of one layer of the drafter's cache.

    Row ``r`` of ``slots`` addresses the request ``r``, and the copy is exact in
    the cache's dtype, so the result is the same latent the dense path would
    have been handed.

    Args:
        kv: The drafter's paged leaf, at the cache lengths its slots resolve in.
        slots: ``[batch, window]`` from :func:`window_slots`.
        row_offsets: ``[batch + 1]`` uint32 those rows' ragged offsets, built
            once per step by :func:`window_reads`.
        layer_idx: Which of the drafter's layers to read.

    Returns:
        ``[batch, window, latent_dim]``.

    Raises:
        ValueError: If the leaf is not replicated over the mesh.
    """
    # The op is a per-device copy with no sharding rule, so its result claims no
    # placement; a copy of a replicated cache's rows is placed like that cache.
    # A sharded or data-parallel leaf would need each rank to copy different
    # rows, which this read does not express.
    if not all(isinstance(p, Replicated) for p in kv.kv_blocks.placements):
        raise ValueError(
            "the drafter's window gather copies rows out of a replicated leaf,"
            f" and this cache is placed {kv.kv_blocks.placements}"
        )
    rows = _gather_rows(
        kv,
        slots,
        row_offsets,
        F.constant(layer_idx, DType.uint32, device=CPU()),
        key_or_value=KEY_CACHE_INDEX,
    )
    return rows.rebind_mapping(kv.kv_blocks.mapping)


def store_context_rows(
    kv: PagedCacheValues,
    latent: Tensor,
    row_offsets: Tensor,
    layer_idx: int,
) -> None:
    """Writes ``[rows, latent_dim]`` latents into one layer of the cache.

    Row ``r`` of request ``b`` lands at slot ``cache_lengths[b] + r -
    row_offsets[b]``, which is the position the row was built at, so the caller
    hands the leaf the cache lengths of the step that produced the rows (the
    pre-iteration lengths, under the block driver).

    Args:
        kv: The drafter's paged leaf.
        latent: ``[rows, latent_dim]`` from ``DSparkDrafter.context_latents``.
        row_offsets: ``[batch + 1]`` uint32 ragged offsets of ``latent``.
        layer_idx: Which of the drafter's layers to write.
    """
    store_k_cache_ragged(
        kv,
        F.unsqueeze(latent, axis=1),
        row_offsets,
        F.constant(layer_idx, DType.uint32, device=CPU()),
    )


def rotate_interleaved(x: Tensor, cos_sin: Tensor) -> Tensor:
    """Rotates ``x``'s last axis as GPT-J interleaved pairs.

    Pair ``(2i, 2i + 1)`` is one complex number, and the rotation runs in
    float32 with one rounding back to ``x``'s dtype.

    Args:
        x: ``[rows, ..., width]``, every ``width`` value in a pair.
        cos_sin: ``[rows, width // 2, 2]`` from :func:`_cos_sin`.

    Returns:
        ``x``'s shape and dtype, rotated.
    """
    width = int(x.shape[-1])
    pairs = F.reshape(x, list(x.shape[:-1]) + [width // 2, 2])
    real = F.cast(pairs[..., 0], DType.float32)
    imag = F.cast(pairs[..., 1], DType.float32)
    # The table broadcasts over the head axes between the token and pair
    # axes: one (cos, sin) per token and pair, shared by every head.
    values_rank = pairs.rank - 1
    broadcast = [cos_sin.shape[0]] + [1] * (values_rank - 2) + [width // 2]
    cos = F.reshape(cos_sin[..., 0], broadcast)
    sin = F.reshape(cos_sin[..., 1], broadcast)
    rotated = F.stack(
        [
            F.cast(real * cos - imag * sin, x.dtype),
            F.cast(real * sin + imag * cos, x.dtype),
        ],
        axis=-1,
    )
    return F.reshape(rotated, list(x.shape))


def negative_infinity(like: Tensor) -> Tensor:
    """A replicated ``-inf`` for ``like``'s mesh, to mask a score with.

    The constant is whole on every rank: a select's false branch broadcasts
    over whatever axes its condition drops, so it must not carry the score's
    placement -- under tensor parallelism that is a shard of the head axis,
    which a rank-0 constant cannot be split along.

    Args:
        like: The scores the constant masks.

    Returns:
        A ``float32`` rank-0 constant, replicated on ``like``'s mesh.
    """
    return F.constant(
        float("-inf"),
        DType.float32,
        device=DeviceMapping.replicated(like.mesh),
    )


def splits_over_tensor_parallel(mesh: DeviceMesh) -> bool:
    """Whether the drafter's weights split over ``mesh``'s single axis.

    The drafter is built on one mesh axis: ``tp`` beside its target's tensor
    parallelism, ``dp`` when the target runs data parallel
    (``dspark_mistral_large4_modulev3.model``). Only the first splits anything:
    a data-parallel mesh's ranks are replicas that hold their own requests, so
    no collective completes a row-parallel partial sum and the FFN's down
    projection would land on every rank as a partial the next one cannot
    finish.

    Args:
        mesh: The mesh the drafter's weights were built on.

    Returns:
        Whether the projections shard along ``TP``.
    """
    return mesh.num_devices > 1 and TP in mesh.axis_names


class DSparkAttention(Module[..., Tensor]):
    """Absorbed MLA over the drafter's own latent keys.

    One 576-wide latent per row per layer, ``[content 512 | rope key 64]``,
    and no value half: the value comes from the same latent through this
    layer's ``wkv_b`` slice, which is what makes MLA a latent cache rather
    than a K/V cache.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        scale: float,
        norm_eps: float,
        rope_inv_freq: NDArray[np.floating],
    ) -> None:
        """Builds the layer's projections and its rope table.

        Args:
            hidden_size: Width of the hidden state.
            num_heads: Query heads (64).
            qk_nope_head_dim: Content dims per head (64).
            qk_rope_head_dim: Rope dims per head (64).
            v_head_dim: Value dims per head (128).
            kv_lora_rank: Latent content width (512).
            scale: The softmax scale, ``qk_head_dim ** -0.5``.
            norm_eps: Epsilon of the latent RMSNorm.
            rope_inv_freq: Plain-YaRN inverse frequencies, ``[rope // 2]``.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.latent_dim = kv_lora_rank + qk_rope_head_dim
        self.scale = scale
        self.rope_inv_freq = rope_inv_freq
        self.wq = Linear(
            in_dim=hidden_size,
            out_dim=num_heads * self.qk_head_dim,
            bias=False,
        )
        self.wkv_a_with_mqa = Linear(
            in_dim=hidden_size, out_dim=self.latent_dim, bias=False
        )
        self.kv_a_norm = RMSNorm(kv_lora_rank, eps=norm_eps)
        self.wkv_b = Linear(
            in_dim=kv_lora_rank,
            out_dim=num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )
        self.wo = Linear(
            in_dim=num_heads * v_head_dim, out_dim=hidden_size, bias=False
        )
        mesh = self.wq.weight.mesh
        self.tensor_parallel = splits_over_tensor_parallel(mesh)
        if self.tensor_parallel:
            # The head axis is what splits, and it splits in three places at
            # once: ``wq``'s rows are ``heads * qk_head_dim``, ``wkv_b``'s are
            # ``heads * (nope + v)``, and ``wo`` consumes ``heads *
            # v_head_dim``, so all three take the one partition. The latent
            # projection and the latent norm stay whole -- the 576-wide latent
            # is not a head axis, and every rank must derive the same latent
            # key for its row -- which is the split the target's own sparse MLA
            # takes (:func:`tensor_parallel_sparse_mla`).
            self.wq.weight = self.wq.weight.to(NamedMapping(mesh, (TP, None)))
            self.wkv_b.weight = self.wkv_b.weight.to(
                NamedMapping(mesh, (TP, None))
            )
            self.wo.weight = self.wo.weight.to(NamedMapping(mesh, (None, TP)))

    @property
    def _kv_b_split(self) -> tuple[Tensor, Tensor]:
        """``wkv_b`` as ``(k_b [heads, nope, latent], v_b [heads, v, latent])``.

        The checkpoint stores the projection ``[heads * (nope + v), latent]``,
        head-major, so the nope slice of every head is its content half and
        the rest is its value half.
        """
        per_head = self.qk_nope_head_dim + self.v_head_dim
        kv_b = F.reshape(
            self.wkv_b.weight, [self.n_heads, per_head, self.kv_lora_rank]
        )
        return kv_b[:, : self.qk_nope_head_dim], kv_b[
            :, self.qk_nope_head_dim :
        ]

    def latent_keys(self, x: Tensor, cos_sin: Tensor) -> Tensor:
        """Returns this layer's ``[rows, 576]`` latent key for ``x``.

        Content is the normalized first half of ``wkv_a``'s projection, rope
        key the rotated second half. The drafter's context bypasses the
        layers' input norms -- the projection is already normalized -- which
        is what makes this a cache of latents rather than of hidden states.
        """
        kv = self.wkv_a_with_mqa(x)
        content = self.kv_a_norm(kv[:, : self.kv_lora_rank])
        rope = rotate_interleaved(kv[:, self.kv_lora_rank :], cos_sin)
        return F.concat([content, rope], axis=-1)

    def forward(
        self,
        x: Tensor,
        context_keys: Tensor,
        positions: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Attends the block rows of ``x`` over the window and themselves.

        Args:
            x: ``[rows, hidden]`` the block rows, already input-normed.
            context_keys: ``[n_ctx, 576]`` this layer's context latents, every
                row the caller holds; ``mask`` chooses which of them are read.
            positions: ``[rows]`` the block rows' rotated positions.
            mask: ``[rows, n_ctx + rows]`` which keys each block row may
                attend. Every block row sees every context row its own request
                holds in the window ``[anchor + 1 - window, anchor]`` and every
                block row of its own request, its own row included -- so the
                mask is not causal inside the block, and it is bounded on both
                ends against the context.

        Returns:
            ``[rows, hidden]`` the attention output, before the residual add.

        A masked key is dropped by an exactly-zero softmax probability, which
        does not clear a NaN in it: ``0 * NaN`` is ``NaN``. A caller whose key
        rows are only partly materialized (a paged read) must therefore leave
        the unmaterialized rows out of the tensor rather than zero- or
        NaN-filling them.
        """
        rows = x.shape[0]
        cos_sin = cos_sin_at(self.rope_inv_freq, positions, x.dtype)
        keys = F.concat([context_keys, self.latent_keys(x, cos_sin)], axis=0)
        n_keys = keys.shape[0]
        scores = F.matmul(
            F.cast(self._absorbed_query(x, cos_sin), DType.float32),
            F.cast(keys, DType.float32).T,
        )
        # The mask names the same key count as the concatenated keys: rebind so
        # the two symbolic widths are one dimension. It is handed over with a
        # size-1 head axis and the select broadcasts it, so each rank masks its
        # own heads out of one small tensor rather than a per-rank copy.
        keep = F.reshape(F.rebind(mask, [rows, n_keys]), [rows, 1, n_keys])
        masked = F.where(
            keep,
            scores * self.scale,
            negative_infinity(scores),
        )
        probabilities = F.cast(F.softmax(masked, axis=-1), keys.dtype)
        latent = F.matmul(probabilities, keys[:, : self.kv_lora_rank])
        return self._project_values(latent)

    def forward_paged(
        self,
        x: Tensor,
        kv: PagedCacheValues,
        reads: WindowReads,
        positions: Tensor,
        *,
        depth: int,
        window: int,
        layer_idx: int,
    ) -> Tensor:
        """Attends the block rows of ``x`` over the window the step read.

        The paged mirror of :meth:`forward`: the same absorbed query, the same
        float32 scores and softmax, the same value projection, and the same key
        set -- the window's rows plus the block's own. It differs in where the
        window comes from (this layer's plane of the drafter's leaf, copied by
        slot) and in that the keys are one rectangle per request instead of one
        pool shared by the batch, so the two attention matmuls are batched over
        the request axis. A context shorter than the window gathers its anchor
        row more than once to fill the rectangle, which is why ``keep`` masks
        those repeats out.

        The window's rows and the block's own rows are never concatenated:
        their *scores* are (2 x QK, summed into one softmax axis) and their
        value products are (2 x PV, summed), so the only tensor the two key
        sets share is the fp32 score rectangle. Concatenating them instead
        makes a [batch, window + depth, 576] key tensor and an fp32 copy of it
        on every layer of every step -- 150 MB and 300 MB apiece at a batch of
        64 with this window, against ~13 MB per rank for the scores (~100 MB
        before the head axis sharded).

        The score rectangle keeps the request, block-row and head axes apart
        (``[batch, depth, heads, keys]``) rather than folding the last two into
        one query axis. Under tensor parallelism each rank holds a slice of the
        head axis, and folding that slice into a ``depth * heads`` axis would
        put a shard behind a replicated dim, which no reshape can express: it
        would gather the queries back to every rank. The window enters the
        block-row axis as a broadcast instead -- one rectangle per request,
        shared by that request's rows.

        Args:
            x: ``[batch * depth, hidden]`` the block rows, already input-normed.
            kv: The drafter's paged leaf, holding every committed row.
            reads: This step's window addressing, from :func:`window_reads` --
                shared by every layer, since only the layer index changes.
            positions: ``[batch * depth]`` the block rows' rotated positions.
            depth: Rows per block.
            window: Slots in the window, which is the width of ``reads.slots``.
            layer_idx: Which of the drafter's layers reads here.

        Returns:
            ``[batch * depth, hidden]`` the attention output, before the
            residual add.
        """
        rows = x.shape[0]
        batch = reads.slots.shape[0]
        assert rows == batch * depth, (
            f"a block's rows are its batch times its depth ({batch} x"
            f" {depth}), not {rows}"
        )
        cos_sin = cos_sin_at(self.rope_inv_freq, positions, x.dtype)
        window_keys = gather_window_rows(
            kv, reads.slots, reads.row_offsets, layer_idx
        )
        own = F.reshape(
            self.latent_keys(x, cos_sin), [batch, depth, self.latent_dim]
        )
        query = F.reshape(
            self._absorbed_query(x, cos_sin),
            [batch, depth, self.n_heads, self.latent_dim],
        )
        # Two QK products, concatenated as scores. The operands stay in the
        # cache's dtype and the accumulator is float32, which is the same sum
        # the upcast multiplier would form (a bf16 product is exact in float32)
        # without materializing the upcast keys.
        scores = F.concat(
            [
                F.matmul(
                    query,
                    F.reshape(
                        F.permute(window_keys, [0, 2, 1]),
                        [batch, 1, self.latent_dim, window],
                    ),
                    out_dtype=DType.float32,
                ),
                F.matmul(
                    query,
                    F.reshape(
                        F.permute(own, [0, 2, 1]),
                        [batch, 1, self.latent_dim, depth],
                    ),
                    out_dtype=DType.float32,
                ),
            ],
            axis=-1,
        )
        n_keys = scores.shape[-1]
        # A block row is never masked, and every block row of a request sees all
        # of them; only the window's padding is dropped. The mask carries the
        # request axis alone and the select broadcasts it over the rows and
        # heads, so it stays one small tensor per step rather than one per head.
        visible = F.concat([reads.keep, reads.own_visible], axis=1)
        masked = F.where(
            F.reshape(visible, [batch, 1, 1, n_keys]),
            scores * self.scale,
            negative_infinity(scores),
        )
        probabilities = F.cast(F.softmax(masked, axis=-1), window_keys.dtype)
        # The value half is the same two-way split: each partial product keeps
        # its float32 accumulator and they are summed there, so the latent is
        # rounded to the activation dtype once rather than once per part.
        latent = F.matmul(
            probabilities[..., :window],
            F.reshape(
                window_keys[:, :, : self.kv_lora_rank],
                [batch, 1, window, self.kv_lora_rank],
            ),
            out_dtype=DType.float32,
        ) + F.matmul(
            probabilities[..., window:],
            F.reshape(
                own[:, :, : self.kv_lora_rank],
                [batch, 1, depth, self.kv_lora_rank],
            ),
            out_dtype=DType.float32,
        )
        return self._project_values(
            F.reshape(
                F.cast(latent, x.dtype),
                [rows, self.n_heads, self.kv_lora_rank],
            )
        )

    def _project_values(self, latent: Tensor) -> Tensor:
        """``[rows, heads, kv_lora_rank]`` attention output through ``wo``.

        The value half of an MLA layer is a per-head map from the latent to the
        v dims: ``value[d] = sum_c latent[c] * v_b[d, c]``. Both entries share
        this tail, so the two paths cannot drift in it.

        Under tensor parallelism the heads are this rank's, so ``wo`` -- whose
        columns are ``heads * v_head_dim``, one contiguous group per head --
        contracts a fraction of them: each rank sums its own slice and one
        all-reduce completes the projection.
        """
        rows = latent.shape[0]
        _, v_b = self._kv_b_split
        values = F.matmul(
            F.permute(latent, [1, 0, 2]), F.permute(v_b, [0, 2, 1])
        )
        flat = F.reshape(
            F.permute(values, [1, 0, 2]),
            [rows, self.n_heads * self.v_head_dim],
        )
        out = self.wo(flat)
        if self.tensor_parallel:
            out = F.allreduce_sum(out, mesh_axis=TP)
        return out

    def _absorbed_query(self, x: Tensor, cos_sin: Tensor) -> Tensor:
        """``[rows, heads, 576]``: the content half absorbed into the latent."""
        rows = x.shape[0]
        q = F.reshape(self.wq(x), [rows, self.n_heads, self.qk_head_dim])
        k_b, _ = self._kv_b_split
        absorbed = F.matmul(
            F.permute(q[..., : self.qk_nope_head_dim], [1, 0, 2]), k_b
        )
        return F.concat(
            [
                F.permute(absorbed, [1, 0, 2]),
                rotate_interleaved(q[..., self.qk_nope_head_dim :], cos_sin),
            ],
            axis=-1,
        )


class DSparkMLP(Module[..., Tensor]):
    """The drafter layer's SiLU-gated dense FFN."""

    def __init__(self, *, hidden_size: int, intermediate_size: int) -> None:
        """Builds the three projections.

        Args:
            hidden_size: Width of the hidden state.
            intermediate_size: Width of the gated inner layer.
        """
        super().__init__()
        self.w1 = Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = Linear(intermediate_size, hidden_size, bias=False)
        mesh = self.w1.weight.mesh
        self.tensor_parallel = splits_over_tensor_parallel(mesh)
        if self.tensor_parallel:
            # The target's own dense-FFN split: gate and up shard on their
            # output axis, down on its input axis. Column-sharding the first
            # two leaves the intermediate already split on the axis the third
            # contracts, which is what makes the down projection a per-rank
            # partial sum rather than a reshard of its own input.
            for gate_or_up in (self.w1, self.w3):
                gate_or_up.weight = gate_or_up.weight.to(
                    NamedMapping(mesh, (TP, None))
                )
            self.w2.weight = self.w2.weight.to(NamedMapping(mesh, (None, TP)))

    def forward(self, x: Tensor) -> Tensor:
        """Returns ``w2(silu(w1(x)) * w3(x))``, all-reduced."""
        out = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if self.tensor_parallel:
            # Each rank summed its own slice of the inner width, leaving a
            # partial sum the residual add below could not complete. The
            # placement picker would insert this reduce at that add anyway;
            # issuing it here keeps the placement and the trace deterministic.
            out = F.allreduce_sum(out, mesh_axis=TP)
        return out


class DSparkLayer(Module[..., Tensor]):
    """One pre-norm drafter layer: attention, norm, FFN, norm.

    The norms belong to the layer boundary, not to the layer body: the
    drafter pre-norms its input once with layer 0's ``attention_norm``, and
    every layer ends by normalizing its residual with the *next* layer's
    ``attention_norm`` -- ``final_norm`` after the last one. That "eagle"
    chain lives in ``DSparkDrafter``'s block entries
    (:meth:`~DSparkDrafter.forward_block` dense,
    :meth:`~DSparkDrafter.block_hidden_paged` paged), which is why this
    module exposes the two halves separately rather than as one forward.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        intermediate_size: int,
        scale: float,
        norm_eps: float,
        rope_inv_freq: NDArray[np.floating],
    ) -> None:
        """Builds the layer's attention, norms and FFN."""
        super().__init__()
        self.attention = DSparkAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora_rank,
            scale=scale,
            norm_eps=norm_eps,
            rope_inv_freq=rope_inv_freq,
        )
        self.attention_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.feed_forward = DSparkMLP(
            hidden_size=hidden_size, intermediate_size=intermediate_size
        )


class DSparkMarkovHead(Module[..., tuple[Tensor, Tensor]]):
    """The gated recurrent head that biases each block row's logits.

    Step ``i`` reads the state ``i`` carries, the Markov embedding of the
    token picked at step ``i - 1`` (the anchor at step 0) and block row ``i``'s
    hidden, and returns the state it advances to and the row's logit bias. The
    state resets to zeros per block: nothing about it persists across target
    steps.
    """

    def __init__(
        self, *, hidden_size: int, vocab_size: int, markov_rank: int
    ) -> None:
        """Builds the embedding, the joint projection and the output bias.

        Args:
            hidden_size: Width of the block rows.
            vocab_size: The target vocabulary the bias spans.
            markov_rank: The state's rank (512).
        """
        super().__init__()
        self.markov_rank = markov_rank
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, dim=markov_rank)
        self.output = Linear(markov_rank, vocab_size, bias=False)
        self.joint_proj = Linear(hidden_size + 2 * markov_rank, 3 * markov_rank)

    def forward(
        self, state: Tensor, previous: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        """One recurrent step.

        Args:
            state: ``[batch, rank]`` the state entering the step.
            previous: ``[batch]`` the token each row's step reads.
            hidden: ``[batch, hidden]`` the step's block row.

        Returns:
            The state the step advances to and the ``[batch, vocab]`` bias.
        """
        # An anchor outside the vocabulary is the reference's mask-token case:
        # it contributes no Markov embedding rather than reading a row the
        # table does not have.
        limit = device_scalar(self.vocab_size, previous)
        zero = device_scalar(0, previous)
        last = device_scalar(self.vocab_size - 1, previous)
        valid = (previous >= zero) & (previous < limit)
        clamp = F.where(previous > last, last, previous)
        clamp = F.where(clamp < zero, zero, clamp).cast(DType.int32)
        embedded = F.gather(self.embedding.weight, clamp, axis=0)
        embedded = F.where(valid.unsqueeze(1), embedded, F.zeros_like(embedded))
        # The order is the trained parameter: state, then the previous token's
        # embedding, then the block row.
        joint = self.joint_proj(F.concat([state, embedded, hidden], axis=-1))
        gate, candidate, emitted = F.split(
            joint, [self.markov_rank] * 3, axis=-1
        )
        gate = F.sigmoid(gate)
        state = gate * state + (1.0 - gate) * F.tanh(candidate)
        return state, self.output(F.tanh(emitted))


class DSparkBackbone(Module[..., Tensor]):
    """The tap projection, the drafter layers and the head's final norm.

    Named after the checkpoint's ``mtp_dspark.backbone.*`` block, so the
    weights load as stored.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        depth: int,
        tap_width: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        intermediate_size: int,
        scale: float,
        norm_eps: float,
        rope_inv_freq: NDArray[np.floating],
    ) -> None:
        """Builds the projection, the mask rows and the layer stack."""
        super().__init__()
        self.depth = depth
        self.target_projection = Linear(
            in_dim=tap_width, out_dim=hidden_size, bias=False
        )
        self.target_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.final_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.mask_embedding = Tensor.zeros([depth - 1, hidden_size])
        self.layers = ModuleList(
            [
                DSparkLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    qk_nope_head_dim=qk_nope_head_dim,
                    qk_rope_head_dim=qk_rope_head_dim,
                    v_head_dim=v_head_dim,
                    kv_lora_rank=kv_lora_rank,
                    intermediate_size=intermediate_size,
                    scale=scale,
                    norm_eps=norm_eps,
                    rope_inv_freq=rope_inv_freq,
                )
                for _ in range(num_layers)
            ]
        )
