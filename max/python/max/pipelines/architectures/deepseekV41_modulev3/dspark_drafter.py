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
"""The DSpark drafter (ModuleV3): a short stage stack that drafts token blocks.

The drafter reads the trunk's target taps, projects them to one context
stream (:meth:`DSparkDrafter.combine`), runs its attention-plus-MoE stages
over a block of draft positions (:meth:`DSparkDrafter.forward_block`), and
turns the result into draft tokens (:meth:`DSparkDrafter.logits`, then
:meth:`DSparkDrafter.draft`). It shares the target's token embedding and LM
head -- both arrive as call arguments, never as drafter parameters -- and a
low-rank Markov head adds a bigram bias to each draft position's logits.

Each stage is the trunk's deferred mHC schedule over the drafter's own
sublayers: the same two boundaries per body (:func:`mhc_fused_post_pre`
and the site's RMSNorm, the trunk layer body's own chain), with the
window attention and the drafter MoE in between. The drafter's KV
rides the trunk's sliding-window leaf -- the planner already reserves the
stage slots -- so the drafter only projects and stores records; it owns no
cache of its own.

Sources: ``dspark_drafter.py`` of the V2 port (structure 84-128, combine
138-148, context/store/window 150-198, forward_block 223-267, logits 314-340,
draft 342-379, block_ids 382-403) and the torch oracle ``RefDrafter``
(``dspark_v41_reference.py:442-533``).
"""

from __future__ import annotations

from collections.abc import Callable

from max import functional as F
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.sequential import ModuleList
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh, Sharded

from .layer_resolver import LayerInfo, resolve
from .layers.block import _on_each_device
from .layers.dspark import (
    DSparkAttention,
    DSparkConfidenceHead,
    DSparkMarkovHead,
    DSparkWindow,
)
from .layers.dspark_moe import DSparkMoE
from .layers.mhc import (
    mhc_final_collapse,
    mhc_fused_post_pre,
    mhc_identity_writeback,
    mhc_post,
)
from .layers.mxfp8_linear import MXFP8Linear
from .layers.norm import RMSNorm
from .layers.rope import RopeParams
from .model_config import DeepseekV41Config

__all__ = ["DSparkDrafter", "DSparkStage", "drafter_layer_infos"]


def drafter_layer_infos(config: DeepseekV41Config) -> list[LayerInfo]:
    """Returns the :class:`LayerInfo` of each drafter stage.

    The stage rows come from ``compress_ratios``'s trailing entries, the
    stage count from ``n_mtp_layers``/``num_nextn_predict_layers``, and the
    sliding-window leaf's width from ``dspark_target_layer_ids`` -- three
    checkpoint fields that must name the same three stages, or the drafter
    would read slots the leaf never reserved.

    Raises:
        ValueError: If the checkpoint declares no draft stages, or the
            fields disagree.
    """
    rows = [row for row in resolve(config) if row.side == "drafter"]
    if len(rows) != config.num_dspark_layers:
        raise ValueError(
            f"config.compress_ratios resolves {len(rows)} DSpark stages, "
            f"but num_dspark_layers is {config.num_dspark_layers}"
        )
    if len(config.dspark_target_layer_ids) != len(rows):
        raise ValueError(
            f"{len(config.dspark_target_layer_ids)} target taps size the "
            f"sliding-window leaf, but compress_ratios resolves "
            f"{len(rows)} DSpark stages -- the checkpoint's fields disagree"
        )
    return rows


def _greedy(logits: Tensor, step: int) -> Tensor:
    """Argmax over the vocabulary; ``F.argmax`` keeps the reduced axis."""
    del step
    return F.squeeze(F.argmax(logits, axis=-1), axis=-1)


class DSparkStage(Module[..., tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    """One drafter stage: window attention and the drafter MoE.

    Each sublayer sits between one fused mHC boundary, the trunk layer body's
    schedule over the drafter's sublayers.

    The stage starts from the previous stage's unwritten feed-forward output
    (``y`` with its ``post`` and row-major ``comb``) and ends the same way,
    so the caller materializes only ahead of the final collapse.
    """

    hc_attn_fn: Tensor
    hc_attn_base: Tensor
    hc_attn_scale: Tensor
    hc_ffn_fn: Tensor
    hc_ffn_base: Tensor
    hc_ffn_scale: Tensor

    def __init__(
        self,
        config: DeepseekV41Config,
        info: LayerInfo,
        mesh: DeviceMesh,
        *,
        rope: RopeParams,
        data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = info.layer_id
        self.hc_mult = config.hc_mult
        h = config.hidden_size
        self.attn_norm = RMSNorm(h)
        self.ffn_norm = RMSNorm(h)
        mix = (2 + config.hc_mult) * config.hc_mult
        for role in ("attn", "ffn"):
            setattr(
                self,
                f"hc_{role}_fn",
                Tensor.zeros(
                    [mix, config.hc_mult * h],
                    dtype=DType.float32,
                    device=mesh,
                ),
            )
            setattr(
                self,
                f"hc_{role}_base",
                Tensor.zeros([mix], dtype=DType.float32, device=mesh),
            )
            setattr(
                self,
                f"hc_{role}_scale",
                Tensor.zeros([3], dtype=DType.float32, device=mesh),
            )
        self.attn = DSparkAttention(
            config,
            info,
            mesh,
            rope=rope,
            max_position=config.max_seq_len,
            data_parallel=data_parallel,
        )
        self.ffn = DSparkMoE(config, mesh, data_parallel=data_parallel)

    def forward(
        self,
        y: Tensor,
        stream: Tensor,
        post: Tensor,
        comb: Tensor,
        pre: Tensor,
        window: DSparkWindow,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One stage on the deferred stream, returning its own deferred tail.

        Args:
            y: The previous stage's feed-forward output, unwritten.
            stream: The stream *before* that write-back, ``[t, hc, hidden]``.
            post: The previous stage's FFN ``post``, float32 ``[t, hc]``.
            comb: The previous stage's FFN ``comb``, row-major float32
                ``[t, hc * hc]``.
            pre: The carried collapse weights (the one-hot mix for the first
                stage), float32 ``[t, hc]``.
            window: This stage's :class:`DSparkWindow`.
            positions: ``[t]`` int32, each block row's absolute position.

        Returns:
            ``(stream, pre, y, post, comb)``: the stream before this stage's
            feed-forward write-back, this stage's FFN ``pre`` (the next
            stage's carried ``pre``), and the unwritten feed-forward output
            with its gates.
        """
        stream, post, comb, pre, x = mhc_fused_post_pre(
            stream,
            y,
            post,
            comb,
            pre,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.attn_norm,
        )
        y = self.attn(x, window, positions)
        stream, post, comb, pre, x = mhc_fused_post_pre(
            stream,
            y,
            post,
            comb,
            pre,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.ffn_norm,
        )
        y = self.ffn(x)
        return stream, pre, y, post, comb


class DSparkDrafter(Module[..., Tensor]):
    """DSpark draft stages plus their input projection and heads.

    A draft step stores the context KV of newly verified tokens
    (:meth:`store_context`), reads each stage's context window
    (:meth:`windows`), runs the draft blocks (:meth:`forward_block`), and
    turns the result into draft tokens (:meth:`logits`, then :meth:`draft`).
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        mesh: DeviceMesh,
        *,
        rope: RopeParams,
        load_confidence_head: bool = False,
        data_parallel_degree: int = 1,
    ) -> None:
        super().__init__()
        if not config.dspark_target_layer_ids:
            raise ValueError("DSpark needs dspark_target_layer_ids")
        self.config = config
        self.block_size = config.dspark_block_size
        self.mesh = mesh
        # TP1 per rank under data parallelism: the stage weights replicate
        # and every rank drafts its own replica's rows.
        self.data_parallel_degree = data_parallel_degree
        hidden = config.hidden_size
        n_taps = len(config.dspark_target_layer_ids)

        self.main_proj = MXFP8Linear(hidden * n_taps, hidden)
        self.main_norm = RMSNorm(hidden)
        self.layer_infos = drafter_layer_infos(config)
        self.stages = ModuleList(
            [
                DSparkStage(
                    config,
                    row,
                    mesh,
                    rope=rope,
                    data_parallel=data_parallel_degree > 1,
                )
                for row in self.layer_infos
            ]
        )
        self.norm = RMSNorm(hidden)
        self.markov_head = DSparkMarkovHead(
            vocab_size=config.vocab_size,
            draft_vocab_size=config.vocab_size,
            markov_rank=config.dspark_markov_rank,
        )
        # Mounted on request: the loader drops the checkpoint's
        # ``confidence_head.*`` tensors by default (the reference never
        # runs adaptive verification), and the serving flag flips both
        # this mount and that drop together.
        if load_confidence_head:
            self.confidence_head = DSparkConfidenceHead(
                hidden_size=hidden,
                markov_rank=config.dspark_markov_rank,
            )

    def combine(self, taps: Tensor) -> Tensor:
        """Projects the trunk taps to the hidden size and normalizes them.

        Args:
            taps: ``[T, hidden * n_taps]``, the trunk hidden states the
                drafter reads (one per target layer), concatenated.

        Returns:
            ``[T, hidden]``.
        """
        return self.main_norm(self.main_proj(taps))

    def store_context(
        self,
        swa: PagedCacheValues,
        taps: Tensor,
        row_offsets: Tensor,
        positions: Tensor,
    ) -> None:
        """Writes every stage's context KV for a target forward's tokens.

        Each stage writes to its own slot of the sliding-window leaf.

        Args:
            swa: The sliding-window KV leaf.
            taps: ``[T, hidden * n_taps]`` trunk taps of the forward's
                tokens.
            row_offsets: ``[batch + 1]`` uint32 ragged row offsets of those
                tokens; the leaf's ``cache_lengths`` are the rows'
                storage origins -- the pre-forward lengths in the frame
                the forward ran, so token ``i`` of request ``r`` lands at
                position ``starts[r] + i`` (``starts`` is the first
                retained row's position when the verify was narrowed).
            positions: ``[T]`` int32 absolute positions of those tokens.
        """
        if self.data_parallel_degree > 1:
            # Each replica's rows are its own, so the store runs once per
            # rank over that rank's shards: a mesh-level row axis is a
            # per-shard dim the ragged-row builders cannot lower. ``self``
            # rides as a pytree argument so each rank sees its weight shards.
            _on_each_device(
                self.mesh,
                DSparkDrafter._store_context_one,
                self,
                swa,
                taps,
                row_offsets,
                positions,
            )
        else:
            self._store_context_one(swa, taps, row_offsets, positions)

    def _store_context_one(
        self,
        swa: PagedCacheValues,
        taps: Tensor,
        row_offsets: Tensor,
        positions: Tensor,
    ) -> None:
        """One device's context store, over that device's own rows."""
        main_x = self.combine(taps)
        for stage in self.stages:
            stage.attn.store_context(swa, main_x, row_offsets, positions)

    def windows(
        self, swa: PagedCacheValues, starts: Tensor
    ) -> list[DSparkWindow]:
        """Reads every stage's :class:`DSparkWindow` from the KV cache.

        Args:
            swa: The sliding-window KV leaf.
            starts: ``[b]`` int32 start position of each draft block.
        """
        return [stage.attn.read_window(swa, starts) for stage in self.stages]

    def forward_block(
        self,
        block_ids: Tensor,
        positions: Tensor,
        windows: list[DSparkWindow],
        embed: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Runs ``b`` draft blocks through the stages.

        Args:
            block_ids: ``[b * block_size]`` int32, each request's anchor
                then ``block_size - 1`` noise tokens, request-major.
            positions: ``[b * block_size]`` int32, each block row's absolute
                position ``L_r .. L_r + block_size - 1``.
            windows: One :class:`DSparkWindow` per stage, in stage order.
            embed: The target's token embedding. Under data parallelism
                this is the proposer's DP embed, which gathers the global
                block, embeds it once and re-slices the hidden per rank.

        Returns:
            The ``[b * block_size, hidden]`` hidden state, with the
            hyper-connection copies merged into one stream, before the
            final norm.

        Raises:
            ValueError: If ``windows`` does not have one entry per stage.
        """
        if len(windows) != len(self.stages):
            raise ValueError(
                f"expected {len(self.stages)} stage windows, got {len(windows)}"
            )
        h = embed(block_ids)
        if self.data_parallel_degree > 1:
            # The stage stack runs once per rank over that rank's rows:
            # every shape read inside is the rank's own, never the global
            # expression a rule-based op would have to reshard. The
            # vocab-sharded embedding already ran above, over the full
            # replicated batch (the trunk's own rule). The stages ride as
            # a pytree argument so each rank sees its own weight shards.
            def _body(
                stages: ModuleList[DSparkStage],
                pos: Tensor,
                hidden: Tensor,
                swa: PagedCacheValues,
                *slots: Tensor,
            ) -> Tensor:
                rebuilt = [DSparkWindow(slots=s, cache=swa) for s in slots]
                return self._forward_stages(stages, hidden, pos, rebuilt)

            return _on_each_device(
                self.mesh,
                _body,
                self.stages,
                positions,
                h,
                windows[0].cache,
                *[w.slots for w in windows],
                mapping=h.mapping,
            )
        return self._forward_stages(self.stages, h, positions, windows)

    def _forward_stages(
        self,
        stages: ModuleList[DSparkStage],
        h: Tensor,
        positions: Tensor,
        windows: list[DSparkWindow],
    ) -> Tensor:
        """The stage stack over one device's already-embedded rows."""
        hc = self.config.hc_mult
        stream = F.stack([h] * hc, axis=1)
        pre = F.concat(
            [
                F.full(
                    [h.shape[0], 1], 1.0, dtype=DType.float32, device=h.mapping
                ),
                F.full(
                    [h.shape[0], hc - 1],
                    0.0,
                    dtype=DType.float32,
                    device=h.mapping,
                ),
            ],
            axis=1,
        )
        y, post, comb = mhc_identity_writeback(stream)
        for stage, window in zip(stages, windows, strict=True):
            stream, pre, y, post, comb = stage(
                y, stream, post, comb, pre, window, positions
            )
        # Materialize the last stage's feed-forward write-back, then collapse
        # with that stage's FFN ``pre`` -- the trunk's final-collapse
        # schedule (no ``hc_head``).
        stream = mhc_post(
            y,
            stream,
            post,
            comb.reshape([stream.shape[0], hc, hc]),
            hc,
        )
        return mhc_final_collapse(stream, pre)

    def logits(self, hidden: Tensor, head_weight: Tensor) -> Tensor:
        """Computes the draft logits before the Markov bias.

        The head weight is the target's (shared, not a drafter parameter):
        bf16 ``[vocab, hidden]``, or this rank's shard under tensor
        parallelism.

        Args:
            hidden: :meth:`forward_block`'s output.
            head_weight: The target's LM head weight.

        Returns:
            ``[b * block_size, vocab]`` float32, request-major. The GEMM
            runs in bfloat16 and widens after, so the draft logits round
            the same way as the target's bfloat16 LM head; the vocab-axis
            allgather splits the same element-wise dot products across
            ranks, changing no element's rounding.
        """
        normed = self.norm(hidden)
        out = normed @ head_weight.T
        if out.is_distributed and any(
            isinstance(p, Sharded) and p.axis == 0
            for p in head_weight.placements
        ):
            out = F.allgather(out, tensor_axis=-1).local_shards[0]
        return out.cast(DType.float32)

    def draft(
        self,
        logits: Tensor,
        anchor_ids: Tensor,
        *,
        hidden: Tensor | None = None,
        sampler: Callable[[Tensor, int], Tensor] | None = None,
        confidence_floor: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Draws ``block_size`` draft tokens per request, one at a time.

        Step ``i``'s logits get the bigram bias of the token drawn at step
        ``i - 1`` (the anchor at step 0); the loop unrolls over the static
        ``block_size``.

        With ``confidence_floor`` set, each step also scores the position's
        acceptance probability through the mounted confidence head, and a
        request whose score falls below the floor emits
        ``dspark_noise_token_id`` for that position and every remaining
        one instead of sampling. The driver already treats noise-filled
        tails as no-ops (all-noise rows are dropped, partial tails fail
        token comparison), so the committed stream is unaffected.

        Args:
            logits: ``[b * block_size, vocab]``, :meth:`logits`' output.
            anchor_ids: ``[b]`` int32, each block's first token.
            hidden: ``[b * block_size, hidden]`` :meth:`forward_block`'s
                output; required with ``confidence_floor``.
            sampler: Draws step ``i``'s ``[b]`` tokens from its ``[b,
                vocab]`` biased logits; greedy argmax when ``None``.
            confidence_floor: The adaptive stop's acceptance floor.

        Returns:
            ``([b, block_size]`` int32 draft ids, ``[b, block_size, vocab]``
            float32 biased logits).

        Raises:
            ValueError: If ``confidence_floor`` is set but the confidence
                head is not mounted, ``hidden`` is missing, or the config
                declares no noise token to fill stopped positions with.
        """
        sample = _greedy if sampler is None else sampler
        block_size = self.block_size
        b = anchor_ids.shape[0]
        vocab = logits.shape[-1]
        logits = F.reshape(
            F.rebind(logits, [b * block_size, vocab]), [b, block_size, vocab]
        )
        current = anchor_ids.cast(DType.int32)

        if confidence_floor is not None:
            head = getattr(self, "confidence_head", None)
            if head is None:
                raise ValueError(
                    "confidence_floor needs the confidence head mounted "
                    "(DSparkDrafter(load_confidence_head=True))"
                )
            if hidden is None:
                raise ValueError(
                    "confidence_floor needs forward_block's hidden states"
                )
            if self.config.dspark_noise_token_id is None:
                raise ValueError(
                    "confidence_floor needs dspark_noise_token_id to fill "
                    "stopped positions"
                )
            assert isinstance(head, DSparkConfidenceHead)
            confidence: DSparkConfidenceHead = head
            hidden_steps = F.reshape(
                F.rebind(hidden, [b * block_size, self.config.hidden_size]),
                [b, block_size, self.config.hidden_size],
            )
            noise = F.constant(
                self.config.dspark_noise_token_id,
                DType.int32,
                device=logits.device,
            )
            stopped: Tensor | None = None

        drafted: list[Tensor] = []
        biased: list[Tensor] = []
        for i in range(block_size):
            step = logits[:, i] + self.markov_head.bias_float32(current)
            biased.append(step)
            if confidence_floor is not None:
                markov_embed, _ = self.markov_head(current)
                score = confidence(hidden_steps[:, i], markov_embed)
                fire = score < confidence_floor
                stopped = (
                    fire if stopped is None else F.logical_or(stopped, fire)
                )
            current = sample(step, i).cast(DType.int32)
            if confidence_floor is not None and stopped is not None:
                current = F.where(stopped, noise, current)
            drafted.append(current)
        return F.stack(drafted, axis=1), F.stack(biased, axis=1)

    def draft_vocab_parallel(
        self, hidden: Tensor, head_weight: Tensor, anchor_ids: Tensor
    ) -> Tensor:
        """Greedy-drafts over a vocab-sharded head without joining the vocab.

        :meth:`logits` joins the head's vocabulary shards on rank 0, and the
        data-parallel caller then broadcasts the whole float32 ``[b *
        block_size, vocab]`` logits back out so every rank can run
        :meth:`draft` over the whole vocabulary. Here each rank keeps its
        own vocabulary shard of every row: it adds the Markov bias over
        that shard alone and reduces each row to one ``(max, index)`` pair,
        and one allgather of those pairs per position hands every rank the
        global argmax. The shards are vocabulary-ordered and the reduction
        takes the lowest index among the tied maxima -- the greedy argmax
        kernel's own tie-break -- so equal per-element logits and biases
        draw the same tokens as :meth:`draft`.

        Args:
            hidden: :meth:`forward_block`'s output, the whole batch on every
                rank.
            head_weight: The target's LM head weight, row-sharded over the
                vocabulary.
            anchor_ids: ``[b]`` anchors, replicated on the mesh.

        Returns:
            ``[b, block_size]`` int32 draft ids on the mesh's first device.
        """
        k = self.block_size
        logits = (self.norm(hidden) @ head_weight.T).local_shards
        embeds = self.markov_head.embed.weight.local_shards
        markov = self.markov_head.head.weight.local_shards
        anchors = anchor_ids.local_shards
        widths = [int(shard.shape[-1]) for shard in logits]
        starts = [sum(widths[:rank]) for rank in range(len(widths))]
        steps = []
        biases = []
        for rank, shard in enumerate(logits):
            b = anchors[rank].shape[0]
            steps.append(
                F.reshape(
                    F.rebind(shard.cast(DType.float32), [b * k, widths[rank]]),
                    [b, k, widths[rank]],
                )
            )
            biases.append(
                markov[rank][starts[rank] : starts[rank] + widths[rank]].cast(
                    DType.float32
                )
            )
        current = [anchor.cast(DType.int32) for anchor in anchors]
        pairs_mapping = DeviceMapping(self.mesh, (Sharded(0),))
        drafted: list[Tensor] = []
        for i in range(k):
            pairs = []
            for rank, step_logits in enumerate(steps):
                step = (
                    step_logits[:, i]
                    + F.gather(embeds[rank], current[rank], axis=0).cast(
                        DType.float32
                    )
                    @ biases[rank].T
                )
                # Vocabulary ids stay below 2**24, so they ride the float32
                # pair exactly and one collective carries both halves.
                index = (F.argmax(step, axis=-1) + starts[rank]).cast(
                    DType.float32
                )
                pairs.append(
                    F.unsqueeze(
                        F.concat([F.max(step, axis=-1), index], axis=-1), 0
                    )
                )
            gathered = F.allgather(
                Tensor.from_shard_values(pairs, pairs_mapping), tensor_axis=0
            )
            current = [
                _lowest_argmax(shard, self.config.vocab_size)
                for shard in gathered.local_shards
            ]
            drafted.append(current[0])
        return F.stack(drafted, axis=1)


def _lowest_argmax(pairs: Tensor, vocab_size: int) -> Tensor:
    """The global argmax from every vocabulary shard's ``(max, index)`` pair.

    Args:
        pairs: ``[shards, b, 2]`` float32, each shard's row maxima and their
            global indices.
        vocab_size: An index no shard can return, for the shards that lose.

    Returns:
        ``[b]`` int32, the lowest index among the shards holding the row's
        maximum.
    """
    values = pairs[:, :, 0]
    indices = pairs[:, :, 1]
    best = F.max(values, axis=0)
    candidates = F.where(F.equal(values, best), indices, float(vocab_size))
    return F.squeeze(F.min(candidates, axis=0), 0).cast(DType.int32)


def block_ids_for(
    anchor_ids: Tensor, block_size: int, noise_token_id: int
) -> Tensor:
    """Builds the draft block token ids from each request's anchor token.

    Args:
        anchor_ids: ``[b]``, the first token of each request's block.
        block_size: Tokens per draft block.
        noise_token_id: The placeholder token filling the rest of a block.

    Returns:
        ``[b * block_size]`` int32: each anchor followed by ``block_size -
        1`` copies of ``noise_token_id``, request-major.
    """
    batch = anchor_ids.shape[0]
    anchors = anchor_ids.cast(DType.int32).unsqueeze(-1)
    noise = F.broadcast_to(
        F.constant(noise_token_id, DType.int32, device=anchor_ids.device),
        [batch, block_size - 1],
    )
    return F.reshape(F.concat([anchors, noise], axis=-1), [batch * block_size])
