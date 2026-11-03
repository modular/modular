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
"""DeepSeek-V4.1 DSpark heads and stage attention."""

from __future__ import annotations

from dataclasses import dataclass

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Embedding, Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded

from ...deepseekV41.layers.csa2 import INDEX_WIDTH
from ...deepseekV41.layers.dspark import (
    dspark_block_index_lists,
    dspark_window_slots,
)
from ..layer_resolver import LayerInfo
from ..model_config import DeepseekV41Config
from .attention import (
    _csa2_attend_op,
    _GroupedWoA,
    _leaf_slots_op,
    _norm_quantize_store_op,
    _store_window_op,
    position_rope_table,
    ragged_rows,
    window_store_slots,
)
from .block import _on_each_device
from .mxfp8_linear import (
    MXFP8Linear,
    Mxfp8Rows,
    _dense_op,
    block_diagonal_woa_supported,
    norm_quantize_store_supported,
)
from .norm import RMSNorm
from .rope import RopeParams, apply_rope_tail, rope_for_layer

__all__ = [
    "DSparkAttention",
    "DSparkConfidenceHead",
    "DSparkMarkovHead",
    "DSparkWindow",
]

_window_slots_op = F.functional(dspark_window_slots)
_block_index_lists_op = F.functional(dspark_block_index_lists)


class _DenseLinear(Module[[Tensor], Tensor]):
    weight: Tensor

    def __init__(self, out_dim: int, in_dim: int, dtype: DType) -> None:
        super().__init__()
        self.weight = Tensor.zeros([out_dim, in_dim], dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


# The Markov bias GEMM: the head's bf16 weights are widened in the graph
# and the whole bias is computed in float32 (``bias_float32``), half a
# bf16 ulp from the float64 oracle where a bf16 GEMM sits 2.5e-3 of max
# |bias| off -- a near-tie there can move a draft step's argmax across a
# gap the oracle puts at <= 1.1e-3: the DRAFTED token changes (never a
# verified output -- the target re-checks every draft).
class DSparkMarkovHead(Module[..., tuple[Tensor, Tensor]]):
    """The sequential bias DSpark adds between draft positions.

    Embeds the previously drafted token and projects the embedding to a bias
    over the draft vocabulary, which the caller adds to the next position's
    logits. Both weights stay replicated under tensor parallelism, keeping
    collectives out of the serial draft loop.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
    ) -> None:
        super().__init__()
        embed = Embedding(vocab_size, dim=markov_rank)
        embed.weight = Tensor.zeros(
            [vocab_size, markov_rank], dtype=DType.bfloat16
        )
        self.embed = embed
        self.head = _DenseLinear(draft_vocab_size, markov_rank, DType.bfloat16)

    def forward(self, prev_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Return the token embedding and its projected bias."""
        if prev_tokens.dtype != DType.int32:
            raise ValueError(
                "DSparkMarkovHead expects int32 token ids, got "
                f"{prev_tokens.dtype}"
            )
        markov_embed = self.embed(prev_tokens)
        return markov_embed, self.head(markov_embed)

    def bias_float32(self, prev_tokens: Tensor) -> Tensor:
        """Compute the draft bias in float32 to preserve near-tie argmaxes."""
        if prev_tokens.dtype != DType.int32:
            raise ValueError(
                "DSparkMarkovHead expects int32 token ids, got "
                f"{prev_tokens.dtype}"
            )
        return (
            self.embed(prev_tokens).cast(DType.float32)
            @ self.head.weight.cast(DType.float32).T
        )


class DSparkConfidenceHead(Module[[Tensor, Tensor], Tensor]):
    """Predicts one draft position's acceptance probability.

    A single float32 linear over the concatenation of a hidden state and
    that position's Markov embedding. The checkpoint stores the projection
    bf16; the loader widens it, like every other float32 weight.
    """

    def __init__(self, *, hidden_size: int, markov_rank: int) -> None:
        super().__init__()
        self.proj = _DenseLinear(1, hidden_size + markov_rank, DType.float32)

    def forward(self, hidden: Tensor, markov_embed: Tensor) -> Tensor:
        """Scores one position's acceptance probability.

        Args:
            hidden: ``[..., hidden_size]`` bf16 hidden state.
            markov_embed: ``[..., markov_rank]`` bf16, :meth:`DSparkMarkovHead`'s
                first return value.

        Returns:
            ``[...]`` float32 confidence score.

        Raises:
            ValueError: If either input is not bfloat16.
        """
        if hidden.dtype != DType.bfloat16:
            raise ValueError(
                f"DSparkConfidenceHead expects bfloat16 hidden, got "
                f"{hidden.dtype}"
            )
        if markov_embed.dtype != DType.bfloat16:
            raise ValueError(
                "DSparkConfidenceHead expects bfloat16 markov_embed, got "
                f"{markov_embed.dtype}"
            )
        combined = F.concat([hidden, markov_embed], axis=-1).cast(DType.float32)
        return F.squeeze(self.proj(combined), axis=-1)


@dataclass(frozen=True)
class DSparkWindow:
    """One stage's context window, per request, as the block attention names it.

    Request ``r``'s block starts at position ``L_r`` and sees context
    positions ``L_r - window .. L_r - 1``, those below zero absent.
    """

    slots: Tensor
    """``[b, window]`` int32 absolute window-leaf slots of those positions in
    the stage's layer, ``-1`` below position zero."""
    cache: PagedCacheValues
    """The sliding-window leaf the slots name."""


class DSparkAttention(Module[..., Tensor]):
    """One DSpark stage's attention over its context window and block.

    A block query attends over the stage's context records (projected from
    the combined hidden states and stored per verified token) plus the block's
    own records, projected the same way and stored at the block's positions,
    where the next forward's context store overwrites them. Stages are
    uncompressed sliding-window layers.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        info: LayerInfo,
        mesh: DeviceMesh,
        *,
        rope: RopeParams,
        max_position: int,
        data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if info.side != "drafter" or info.compress_ratio != 0:
            raise ValueError(
                f"layer {info.layer_id} is not a ratio-0 DSpark stage "
                f"(side={info.side!r}, "
                f"compress_ratio={info.compress_ratio})"
            )
        self.layer_id = info.layer_id
        heads, d = config.num_attention_heads, config.head_dim
        self.n_heads = heads
        self.head_dim = d
        self.window = config.sliding_window
        self.block_size = config.dspark_block_size
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.softmax_scale = d**-0.5
        self.rope_head_dim = rope.rope_head_dim
        self.max_position = max_position
        self.max_rows = config.max_batch_rows
        # Each block query names its window and its block, in whole tiles of
        # 64 list entries.
        self.list_entries = -(-(self.window + self.block_size) // 64) * 64
        if self.list_entries > INDEX_WIDTH:
            raise ValueError(
                f"a window of {self.window} and a block of "
                f"{self.block_size} do not fit a CSA2 list"
            )
        # Under data parallelism the stage weights replicate (TP1 per
        # rank, like the trunk's attention): each rank drafts its own
        # replica's rows, so a mesh-axis weight split would need an
        # all-reduce that sums across unrelated replicas.
        self._data_parallel = data_parallel
        self._tp = 1 if data_parallel else mesh.num_devices

        self.attn_sink = Tensor.zeros([heads], dtype=DType.float32)
        # ``wq_a`` and ``wkv`` read the same input, so the adapter stacks
        # them into one projection, as in the trunk's attention.
        self.q_lora_rank = config.q_lora_rank
        self.wqa_wkv = MXFP8Linear(config.hidden_size, config.q_lora_rank + d)
        self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = MXFP8Linear(config.q_lora_rank, heads * d)
        self.kv_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.wo_a = _GroupedWoA(
            self.o_groups * self.o_lora_rank,
            heads * d // self.o_groups,
            self.o_lora_rank,
        )
        self.wo_b = MXFP8Linear(
            self.o_groups * self.o_lora_rank, config.hidden_size
        )
        self.rope = rope_for_layer(info, rope)
        self._apply_tp(mesh)

    def _apply_tp(self, mesh: DeviceMesh) -> None:
        """Declares the reference's TP split on this stage's weights.

        ``wq_b`` rows and ``attn_sink`` by heads, ``wo_a`` by group, ``wo_b``
        column-parallel with the all-reduce after it; the latent side
        (``wqa_wkv``, the norms) stays whole, so every rank computes the
        whole context KV.

        Under data parallelism every weight replicates instead: the
        mesh's one axis is the replica axis, so there is no partial-sum
        partner for the all-reduce to complete.
        """
        if mesh.num_devices <= 1 or self._data_parallel:
            return
        by_rows = DeviceMapping(mesh, (Sharded(0),))
        self.wq_b.shard(0, mesh)
        self.wo_a.weight = self.wo_a.weight.to(by_rows)
        self.wo_a.scale = self.wo_a.scale.to(by_rows)
        self.attn_sink = self.attn_sink.to(by_rows)
        self.wo_b.shard(1, mesh)

    def _layer_slot(self) -> Tensor:
        """This stage's leaf index in the sliding-window cache."""
        return F.constant(self.layer_id, DType.uint32, device=CPU())

    def _kv_latent(self, x: Tensor) -> Tensor:
        """``wkv(x)``: the KV rows of the stacked projection's weights."""
        q, d = self.q_lora_rank, self.head_dim
        return _dense_op(
            x,
            self.wqa_wkv.weight[q : q + d, :],
            self.wqa_wkv.scale[q // 32 : (q + d) // 32, :],
        )

    def store_context(
        self,
        swa: PagedCacheValues,
        main_x: Tensor,
        row_offsets: Tensor,
        positions: Tensor,
    ) -> None:
        """Stores this stage's context KV for every token of a target forward.

        The trunk's own window store writes the rows: the normed,
        unrotated latent goes to ``mo.csa.window_kv_store``, which owns
        the tail rotation, the MXFP8 quantize and the paged insert into
        the sliding-window leaf's 528-byte records -- the same store, the
        same record bytes and the same slots the trunk layers use, so
        the drafter's window and the trunk's share one format.

        Args:
            swa: The sliding-window KV leaf, its ``cache_lengths`` already
                replaced with the payload's storage origins (the
                adapter's ``context_store_origins``).
            main_x: ``[T, hidden]`` combined hidden states of the forward's
                tokens.
            row_offsets: ``[batch + 1]`` uint32 ragged row offsets; token
                ``i`` of request ``r`` lands at position ``starts[r] + i``
                because the leaf's ``cache_lengths`` are the rows'
                storage origins -- the pre-forward lengths in the frame
                the forward ran.
            positions: ``[T]`` int32 absolute positions of those tokens.
        """
        total = main_x.shape[0]
        # The payload's positions are the storage-origin-relative rows
        # (the first kept row's position plus its local index), so the
        # trunk's own row builder reproduces them from the leaf's
        # replaced lengths.
        rows = ragged_rows(row_offsets, swa.cache_lengths, total)
        _store_window_op(
            swa.kv_blocks,
            self.kv_norm(self._kv_latent(main_x)),
            rows.positions,
            position_rope_table(
                self.rope.inv_freqs(),
                self.max_position,
                self.max_rows,
                rows.positions,
            ),
            window_store_slots(swa, rows, self._layer_slot()),
            block_stride=swa.page_stride,
        )

    def read_window(
        self, swa: PagedCacheValues, starts: Tensor
    ) -> DSparkWindow:
        """Names each request's context window in the sliding-window leaf.

        Positions ``starts[r] - window .. starts[r] - 1``. Records stored
        for tokens not yet accepted sit at or past ``starts`` and are
        overwritten by the next store before any window names them.

        Under data parallelism ``starts`` carries each replica's own
        requests and the leaf's records are per-device, so the slots build
        once per rank over that rank's rows and claim the rows' own
        placement.

        Args:
            swa: The sliding-window KV leaf.
            starts: ``[b]`` int32, each request's first block position,
                replicated over the mesh, or row-sharded per replica under
                data parallelism.
        """

        def _slots(leaf: PagedCacheValues, starts_d: Tensor) -> Tensor:
            return _window_slots_op(leaf, starts_d, self.window, self.layer_id)

        return DSparkWindow(
            slots=_on_each_device(
                starts.mesh, _slots, swa, starts, mapping=starts.mapping
            ),
            cache=swa,
        )

    def forward(
        self, h_normed: Tensor, window: DSparkWindow, positions: Tensor
    ) -> Tensor:
        """Attends each draft block over its context window and itself.

        The block's records are stored at its positions in this stage's layer
        of the window leaf, then every block query attends over the slots of
        its window and of its whole block.

        Args:
            h_normed: ``[b * block_size, hidden]`` bf16, the boundary's
                normalized collapse of the block rows, request-major.
            window: This stage's :class:`DSparkWindow`.
            positions: ``[b * block_size]`` int32, each block row's absolute
                position ``L_r .. L_r + block_size - 1``.

        Returns:
            ``[b * block_size, hidden]`` bf16, all-reduced across the mesh.
        """
        k = self.block_size
        swa = window.cache
        t = h_normed.shape[0]
        b = window.slots.shape[0]
        cos_sin = self.rope.cos_sin(positions)

        # The block is whole requests of ``k`` rows each.
        request = (
            F.range(
                0, t, 1, out_dim=t, dtype=DType.int32, device=positions.mapping
            )
            // k
        )
        block_slots = _leaf_slots_op(
            swa, request, positions, self._layer_slot()
        ).rebind_mapping(positions.mapping)
        rope_table = position_rope_table(
            self.rope.inv_freqs(),
            self.max_position,
            self.max_rows,
            positions,
        )
        projected = self.wqa_wkv(h_normed)
        if norm_quantize_store_supported(
            projected.mesh.devices[0], self.q_lora_rank, self.head_dim
        ):
            # One launch norms both latents, quantizes ``q`` and stores the
            # block's window records (the trunk layer's fused query).
            codes, scales = _norm_quantize_store_op(
                projected,
                self.q_norm.weight,
                self.kv_norm.weight,
                swa.kv_blocks,
                positions,
                rope_table,
                block_slots,
                swa.page_stride,
                self.q_norm.eps,
                self.q_lora_rank,
            )
            q = self.wq_b.forward_quantized_rope_tail(
                Mxfp8Rows(
                    codes.rebind_mapping(projected.mapping),
                    scales.rebind_mapping(projected.mapping),
                ),
                cos_sin,
                self.head_dim,
            )
        else:
            width = int(projected.shape[-1])
            q = self.wq_b.forward_rope_tail(
                self.q_norm(projected[:, : self.q_lora_rank]),
                cos_sin,
                self.head_dim,
            )
            _store_window_op(
                swa.kv_blocks,
                self.kv_norm(projected[:, self.q_lora_rank : width]),
                positions,
                rope_table,
                block_slots,
                block_stride=swa.page_stride,
            )
        q = F.reshape(q, [t, self.n_heads, self.head_dim])
        # ``t`` is the trace's ragged dim; ``b * k`` is its factored form.
        block_slots = F.reshape(F.rebind(block_slots, [b * k]), [b, k])
        idx = _block_index_lists_op(
            window.slots, block_slots, INDEX_WIDTH
        ).rebind_mapping(positions.mapping)
        out = _csa2_attend_op(
            q,
            F.rebind(idx, [t, INDEX_WIDTH]),
            self.attn_sink,
            swa.kv_blocks,
            None,
            scale=self.softmax_scale,
            window_stride=swa.page_stride,
            window_entries=self.list_entries,
        )
        # The rule-less op leaves its output Unknown-placed; every rank
        # computed its own heads' rows, which belong on the queries' mapping.
        out = out.rebind_mapping(q.mapping)
        # ``wo_a`` takes the block's output un-rotated and rotates it back
        # in its quantize where the block-diagonal launch applies.
        rope = None
        if self._folds_inverse_rope(out):
            rope = (cos_sin, self.head_dim)
        else:
            out = apply_rope_tail(
                out, cos_sin, self.rope_head_dim, inverse=True
            )
        flat = F.reshape(out, [t, self.n_heads * self.head_dim])
        result = self.wo_b(self.wo_a(flat, rope))
        # Column-parallel partial sums: one all-reduce completes it. Under
        # data parallelism the weights replicate and the result carries the
        # rows each rank owns, so no reduction runs anywhere.
        if result.is_distributed and not self._data_parallel:
            result = F.allreduce_sum(result)
        return result

    def _folds_inverse_rope(self, out: Tensor) -> bool:
        """Whether ``wo_a`` can take ``out`` un-rotated and rotate it back.

        The fold rides the block-diagonal launch, which a shard running a
        single group never takes.
        """
        return self.o_groups // self._tp >= 2 and block_diagonal_woa_supported(
            out.mesh.devices[0],
            self.n_heads * self.head_dim // self.o_groups,
            self.o_lora_rank,
        )
