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
"""Model-agnostic driver for block (DFlash / DSpark) speculative decoding.

A block draft spends its whole ``K`` token budget in one parallel forward over
the accepted token followed by ``K - 1`` mask tokens, and reads every proposal
out of that single pass. That makes it the structural complement of the
sequential drafts in :mod:`.driver`: no per-step loop, so no carried hidden
state, no per-step cache advance, no token shift and no stack at the end.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, Literal, Protocol, TypeVar

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    DimLike,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.kernels import topk_fused_sampling_with_dist
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import (
    AcceptanceSampler,
    _draft_step_seed,
)
from typing_extensions import override

from .config import MAGIC_DRAFT_TOKEN_ID, SpeculativeConfig
from .ragged_token_merger import (
    RaggedTokenMerger,
    _shape_to_scalar,
    compute_host_merged_offsets,
)
from .spec_input_types import (
    SpecDecodeGraphSignature,
    SpecDecodeInputTypeSpec,
)
from .spec_target import SpecDecodeTarget
from .spec_width_policy import declares_skippable_draft
from .unified_graph_ops import apply_overlap_bitmask, broadcast_per_device

__all__ = [
    "Accepted",
    "ArgmaxDraftSampler",
    "BlockBatch",
    "BlockCaches",
    "BlockDriver",
    "BlockProposer",
    "DraftSampler",
    "SampledDraftSampler",
    "block_dispatch_metadata",
    "block_kv_with_dispatch",
    "local_row_offsets",
]


def block_dispatch_metadata(meta: TensorValue | None, k: int) -> TensorValue:
    """Rebuilds the MHA dispatch metadata at the draft block's query width.

    The 4-int CPU buffer is ``[batch_size, q_max_seq_len, num_partitions,
    max_cache_valid_length]``. Neither manager-supplied buffer fits the block:
    the leaf's own metadata carries the *verify* query width, which equals the
    block only on a decode batch and is far larger on a prefill batch, where
    the oversized query bound can drive the block's first attention layer to
    NaN. ``num_partitions`` is zeroed so the kernel recomputes the split-K
    count for the drafter's own head geometry instead of reusing the target's.

    Args:
        meta: The leaf's verify-width dispatch metadata buffer.
        k: The draft block width (anchor slot plus drafted tokens).

    Returns:
        The rebuilt dispatch metadata buffer.
    """
    assert meta is not None
    cpu = DeviceRef.CPU()
    return ops.concat(
        [
            meta[0:1],
            ops.constant(k, DType.int64, device=cpu).reshape((1,)),
            ops.constant(0, DType.int64, device=cpu).reshape((1,)),
            meta[3:4],
        ],
        axis=0,
    )


def block_kv_with_dispatch(
    block_kv: list[PagedCacheValues], k: int
) -> list[PagedCacheValues]:
    """The block caches with the dispatch buffer rebuilt at width ``k``."""
    return [
        replace(
            kv,
            attention_dispatch_metadata=block_dispatch_metadata(
                kv.attention_dispatch_metadata, k
            ),
            max_prompt_length=ops.constant(
                k, DType.uint32, device=DeviceRef.CPU()
            ).broadcast_to([1]),
        )
        for kv in block_kv
    ]


def local_row_offsets(
    offsets: TensorValue, splits: TensorValue, replica: int, dim_name: str
) -> TensorValue:
    """One replica's slice of a ragged offset vector, rebased to zero.

    A ragged offset vector has one entry more than it has rows, so a replica
    owning rows ``[start, end)`` needs entries ``[start, end]``; and the
    tensors it indexes hold only its own rows, so the slice is shifted to
    start at zero. Every data-parallel slice in a block graph is this.
    """
    start = splits[replica]
    local = ops.slice_tensor(
        offsets, [(slice(start, splits[replica + 1] + 1), dim_name)]
    )
    return local - offsets[start]


_TargetHiddenT = TypeVar("_TargetHiddenT", contravariant=True)
"""The target's captured-hidden payload, opaque to the driver.

Handed from :meth:`SpecDecodeTarget.verify` straight to
:meth:`BlockProposer.materialize`. Contravariant because it is only ever
consumed here -- nothing on this side hands it back out.
"""

_BlockHiddenT = TypeVar("_BlockHiddenT")
"""The draft's block hidden states, opaque to the driver.

Handed from :meth:`BlockProposer.forward_block` straight to
:meth:`BlockProposer.head`.
"""


@dataclass(frozen=True)
class BlockBatch:
    """One block spec-decode iteration's graph inputs, merged.

    Mirrors :class:`~.driver.SequentialBatch` for the block shape. The optional
    fields are the ones a ``distributed=False`` signature does not declare;
    the single-device DFlash graphs leave them unset.
    """

    tokens: TensorValue
    input_row_offsets: TensorValue
    draft_tokens: TensorValue
    kv_collections: list[PagedCacheValues]
    draft_kv_collections: list[PagedCacheValues]
    passthrough_kv: Mapping[str, list[PagedCacheValues]]
    """Target cache leaves past the primary one, keyed by the model's name.

    The primary leaf anchors the draft: its pre-iteration ``cache_lengths`` is
    where the draft materializes the target's context KV."""
    return_n_logits: TensorValue
    devices: Sequence[DeviceRef]
    signal_buffers: list[BufferValue]
    ep_inputs: list[Value[Any]] | None

    merged_tokens: TensorValue
    merged_offsets: TensorValue
    merged_offsets_per_dev: list[TensorValue]

    host_merged_offsets: TensorValue | None
    """CPU mirror of :attr:`merged_offsets`, letting a sharded target size its
    collectives without waiting on the device."""
    data_parallel_splits: TensorValue | None
    """Per-replica batch boundaries, ``None`` on a single-replica graph."""
    data_parallel_degree: int
    batch_context_lengths: list[TensorValue]
    """Per-device cache-length tensors, for a target with a sparse-attention
    budget to size."""

    vision_embeddings: list[TensorValue]
    """Per-device merged vision embeddings, empty for a text-only target.

    A vision target scatters these into the merged sequence before running its
    stack."""
    vision_scatter_indices: list[TensorValue]
    """Per-device merge positions for :attr:`vision_embeddings`."""

    extra: Mapping[str, Any]
    """Graph inputs the driver carries but never reads, keyed by the model.

    A model whose signature declares inputs outside the canonical set reaches
    them from its adapters through here, rather than the driver growing a
    field per model for values none of its phases understand."""

    num_accepted: TensorValue | None
    """How many draft tokens each request accepted, ``None`` before the accept.

    Set for the phases that run after the accept, so a target carrying state
    no length pointer can rewind -- a linear-attention recurrence, say -- can
    roll that state onto the accepted prefix without the driver knowing what
    the state is."""

    draft_slot_ids: TensorValue | None = None
    """Which block rows this step drafts, ``None`` on a graph that always
    drafts. Its extent is the row count: ``batch_size * K`` to draft, zero to
    skip. See :attr:`SpecDecodeInputTypeSpec.include_skippable_draft`."""

    draft_block_offsets: TensorValue | None = None
    """Runtime replacement for the block's fixed-``K``-stride row offsets,
    ``None`` on a graph that always drafts."""

    block_size: int = 0
    """``K``, so a phase can turn :attr:`draft_slot_ids`' extent into a
    sequence count. Zero only on a batch built before the driver knew it."""

    @property
    def drafts_all_rows(self) -> bool:
        """Whether every row in the batch drafts, as it always did."""
        return self.draft_slot_ids is None

    @property
    def num_draft_seqs(self) -> Dim:
        """How many sequences this step drafts for: ``batch_size``, or zero.

        Symbolic, so the graph carries one row count through the block rather
        than branching on which of the two it is.
        """
        if self.draft_slot_ids is None:
            return Dim("batch_size")
        assert self.block_size > 0, (
            "a skippable-draft batch must carry its block size"
        )
        return self.draft_slot_ids.shape[0] // self.block_size

    @property
    def devices_per_replica(self) -> int:
        """How many devices share one data-parallel replica's rows."""
        return self.n_devs // self.data_parallel_degree

    @property
    def replica_of(self) -> list[int]:
        """Which replica owns each device's rows, by device index."""
        per_replica = self.devices_per_replica
        return [i // per_replica for i in range(self.n_devs)]

    @property
    def n_devs(self) -> int:
        """Number of devices the target and draft are sharded across."""
        return len(self.devices)

    @property
    def device0(self) -> DeviceRef:
        """The device that owns the batch-wide (non-sharded) tensors."""
        return self.devices[0]

    @property
    def num_draft_tokens(self) -> Dim:
        """How many tokens the previous iteration proposed."""
        return self.draft_tokens.shape[1]


@dataclass(frozen=True)
class Accepted:
    """What the accept phase produced, corrected.

    :attr:`num_accepted` is already zeroed for rows that carried no real
    proposal, so every phase downstream reads one number instead of
    re-deriving the correction -- which is what the hand-written modules did,
    twice each, once for the block position and once for the returned metric.
    """

    num_accepted: TensorValue
    next_tokens: TensorValue
    """The token this iteration commits, one per row."""
    commit_lengths: TensorValue
    """How many tokens this iteration commits: the prompt for a prefill row,
    the accepted count plus the bonus token for a decode row."""
    is_prefill: TensorValue
    """Per-row flag: this iteration carried no proposals to verify."""


@dataclass(frozen=True)
class BlockCaches:
    """The two positions a block draft reads its own KV cache at.

    ``ctx`` sits at the pre-iteration cache length, where the draft
    materializes the target's context KV; ``block`` sits past the tokens this
    iteration commits, where the block forward writes. Both are the same
    per-device caches, differing only in ``cache_lengths``.
    """

    ctx: list[PagedCacheValues]
    block: list[PagedCacheValues]


class BlockProposer(Protocol[_TargetHiddenT, _BlockHiddenT]):
    """A draft that emits all ``K - 1`` proposals from one parallel forward."""

    block_size: int
    """``K``: the anchor slot plus the ``K - 1`` tokens drafted after it."""
    mask_token_id: int
    """Fills the block's tail, one slot per token to be drafted."""
    samples_from_anchor: bool
    """Whether the anchor slot's own logits predict a draft token.

    The anchor carries the committed token. A DFlash draft drops its position
    and proposes ``K - 1``; the dense DSpark drafter reads all ``K``. The
    difference is the acceptance sampler's step count as well as the head's
    output width, which is why it is declared rather than inferred."""
    supports_zero_draft_rows: bool
    """Whether this draft runs on the runtime row count the driver hands it
    rather than on ``batch_size``. That count is zero on a step whose drafts
    nothing would verify. A draft without it drafts every row even where the
    schedule skips.

    TODO(SERVOPT-1610): Every draft is meant to support this. The field goes
    away once each one is written against the draft row count."""

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: _TargetHiddenT,
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        """Writes the target's context KV into the draft's cache."""
        ...

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        """Embeds the flattened block token ids, per device."""
        ...

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[PagedCacheValues],
    ) -> _BlockHiddenT:
        """Runs the draft over the whole block; returns its hidden states."""
        ...

    def head(
        self,
        batch: BlockBatch,
        block_hs: _BlockHiddenT,
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Turns the block's hidden states into the next step's proposals.

        ``[batch.num_draft_seqs, K - 1]``, or ``[batch.num_draft_seqs,
        num_speculative_tokens]`` when the driver verifies fewer proposals than
        the block holds. ``batch.num_draft_seqs`` is ``batch_size`` for every
        proposer without :attr:`supports_zero_draft_rows`. The driver pads a
        skipping step's empty result back out to the fixed output shape.

        Owns the head's scoring: which ``lm_head``, whether the anchor slot is
        sliced off before or after the projection, logit softcapping. The
        tokens themselves come from ``sampler``, which the driver picks for
        the configured ``draft_proposal``.
        """
        ...


class DraftSampler(Protocol):
    """Picks a block draft's tokens from the logits its head scores.

    The driver hands one to :meth:`BlockProposer.head`, so a head scores its
    logits once and serves both ``draft_proposal`` modes. A head whose
    positions never read each other's tokens calls :meth:`sample_all`; a head
    whose logits at each position depend on the tokens picked before it calls
    :meth:`sample_next` once per position, in order.
    """

    def sample_all(self, logits: TensorValue) -> TensorValue:
        """Picks every position's token at once.

        Args:
            logits: ``[batch_size, n, vocab_size]`` target-vocab logits.

        Returns:
            ``[batch_size, n]`` token ids.
        """
        ...

    def sample_next(
        self,
        logits: TensorValue,
        position: int,
        token_ids: TensorValue | None = None,
    ) -> TensorValue:
        """Picks one position's token for every row.

        Args:
            logits: ``[batch_size, m]`` logits for this position. ``m`` is
                whatever space the head scores -- the target vocabulary, a
                pruned draft vocabulary, a candidate list -- and the head maps
                the returned index back to a token itself.
            position: The position in the block, from 0.
            token_ids: ``[batch_size, m]`` target-vocab id of each column, when
                the head scores less than the whole target vocabulary.

        Returns:
            The ``[batch_size]`` column index picked.
        """
        ...


class ArgmaxDraftSampler(DraftSampler):
    """Takes each position's argmax, for ``draft_proposal="argmax"``."""

    @override
    def sample_all(self, logits: TensorValue) -> TensorValue:
        return ops.squeeze(ops.argmax(logits, axis=-1), axis=-1)

    @override
    def sample_next(
        self,
        logits: TensorValue,
        position: int,
        token_ids: TensorValue | None = None,
    ) -> TensorValue:
        del position, token_ids
        return ops.squeeze(ops.argmax(logits, axis=-1), axis=-1)


class SampledDraftSampler(DraftSampler):
    """Draws each position at its request's temperature, top-k and top-p.

    Serves ``draft_proposal="sampled"``, keeping the distribution each draw
    came from, in target-vocab positions, for the verifier. The driver builds
    one per iteration from the per-row sampling parameters. Position ``i`` is
    keyed like sequential draft step ``i``, so no two positions of a request
    share a draw.
    """

    def __init__(
        self,
        *,
        seed: TensorValue,
        temperature: TensorValue,
        top_k: TensorValue,
        top_p: TensorValue,
        vocab_size: int,
        rows: DimLike = "batch_size",
    ) -> None:
        self._seed = seed
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._vocab_size = vocab_size
        # ``batch_size``, or the drafting sequences of a step that can skip.
        # The per-row parameters carry the same count.
        self._rows = rows
        self._all_dists: TensorValue | None = None
        self._next_dists: list[TensorValue] = []

    @override
    def sample_all(self, logits: TensorValue) -> TensorValue:
        assert self._all_dists is None and not self._next_dists
        n = int(logits.shape[1])
        vocab = logits.shape[2]

        def per_position(param: TensorValue) -> TensorValue:
            return ops.broadcast_to(
                ops.unsqueeze(param, axis=1), [self._rows, n]
            ).reshape((-1,))

        seeds = ops.stack(
            [_draft_step_seed(self._seed, i) for i in range(n)], axis=1
        ).reshape((-1,))
        tokens, dist = topk_fused_sampling_with_dist(
            logits.rebind([self._rows, n, vocab]).reshape((-1, vocab)),
            top_k=per_position(self._top_k),
            temperature=per_position(self._temperature),
            top_p=per_position(self._top_p),
            seed=seeds,
        )
        self._all_dists = dist.reshape((self._rows, n, vocab))
        return tokens.reshape((self._rows, n))

    @override
    def sample_next(
        self,
        logits: TensorValue,
        position: int,
        token_ids: TensorValue | None = None,
    ) -> TensorValue:
        assert self._all_dists is None
        assert position == len(self._next_dists), (
            "positions must be drawn in order"
        )
        index, dist = topk_fused_sampling_with_dist(
            logits.rebind([self._rows, logits.shape[1]]),
            top_k=self._top_k,
            temperature=self._temperature,
            top_p=self._top_p,
            seed=_draft_step_seed(self._seed, position),
        )
        if token_ids is not None:
            dist = _scatter_to_vocab(dist, token_ids, self._vocab_size)
        self._next_dists.append(dist)
        return index

    def distributions(self) -> TensorValue:
        """The ``[rows, n, vocab_size]`` distributions drawn from."""
        if self._all_dists is not None:
            return self._all_dists
        return ops.stack(self._next_dists, axis=1)


def _scatter_to_vocab(
    dist: TensorValue, token_ids: TensorValue, vocab_size: int
) -> TensorValue:
    """Spreads a distribution over a subset of the vocabulary onto all of it.

    The verifier compares the draft's ``q`` with the target's ``p`` token by
    token, so a head that sampled over a candidate list or a pruned vocabulary
    hands back ``q`` in target-vocab positions, zero wherever it put no mass.

    Args:
        dist: ``[batch_size, m]`` probabilities.
        token_ids: ``[batch_size, m]`` distinct target-vocab id of each column.
        vocab_size: The target vocabulary size.

    Returns:
        ``[batch_size, vocab_size]`` float32 probabilities.
    """
    device = dist.device
    batch = dist.shape[0]
    m = dist.shape[1]
    rows = ops.broadcast_to(
        ops.unsqueeze(
            ops.range(0, batch, 1, batch, dtype=DType.int64, device=device),
            axis=1,
        ),
        [batch, m],
    )
    indices = ops.stack([rows, token_ids.cast(DType.int64)], axis=-1)
    zeros = ops.broadcast_to(
        ops.constant(0.0, DType.float32, device=device), [batch, vocab_size]
    )
    return ops.scatter_nd(zeros, dist.cast(DType.float32), indices)


class BlockDriver(
    SpecDecodeGraphSignature, Module, Generic[_TargetHiddenT, _BlockHiddenT]
):
    """Merge -> verify -> mask -> accept -> materialize -> block -> head."""

    def __init__(
        self,
        target: SpecDecodeTarget[BlockBatch, _TargetHiddenT],
        proposer: BlockProposer[_TargetHiddenT, _BlockHiddenT],
        *,
        target_model: Module,
        draft_model: Module,
        input_spec: SpecDecodeInputTypeSpec,
        speculative_config: SpeculativeConfig,
        enable_structured_output: bool = False,
        relaxed_acceptance: bool = False,
        ctx_at_draft_cache_length: bool = False,
        num_speculative_tokens: int | None = None,
        use_greedy_acceptance: bool = False,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self._target = target
        self._proposer = proposer
        self._input_spec = replace(
            input_spec,
            enable_structured_output=enable_structured_output,
            include_skippable_draft=declares_skippable_draft(
                speculative_config
            ),
        )
        self.devices = self._input_spec.devices
        self.data_parallel_degree = self._input_spec.data_parallel_degree
        # TODO(SERVOPT-1610): The runtime offsets arrive on one device with no
        # host mirror, and data parallelism would have to re-split them per
        # replica, so a multi-device block draft drafts every row until the
        # block driver handles both.
        self._skips_draft_rows = (
            proposer.supports_zero_draft_rows
            and len(self.devices) == 1
            and self.data_parallel_degree == 1
        )
        # Where the draft materializes the target's context KV; the two caches
        # normally advance together.
        self._ctx_at_draft_cache_length = ctx_at_draft_cache_length
        self.enable_structured_output = enable_structured_output
        self.block_size = proposer.block_size
        # The block always runs at its trained width; a step may verify fewer
        # of its proposals than it holds.
        max_drafts = self.block_size - (
            0 if proposer.samples_from_anchor else 1
        )
        if num_speculative_tokens is None:
            num_speculative_tokens = max_drafts
        elif not 1 <= num_speculative_tokens <= max_drafts:
            raise ValueError(
                f"A block of {self.block_size} holds 1 to {max_drafts}"
                " proposals; got"
                f" num_speculative_tokens={num_speculative_tokens}."
            )
        self.num_speculative_tokens = num_speculative_tokens

        draft_proposal: Literal["argmax", "sampled"] = (
            speculative_config.draft_proposal
        )
        self._vocab_size = vocab_size
        self._sampled = draft_proposal == "sampled"
        if self._sampled:
            if vocab_size is None:
                raise ValueError(
                    "vocab_size is required when draft_proposal='sampled':"
                    " draft_probs_full's trailing dim has to be static"
                )
            if speculative_config.synthetic_acceptance_rate is not None:
                raise ValueError(
                    "draft_proposal='sampled' is incompatible with "
                    "synthetic_acceptance_rate"
                )
            if use_greedy_acceptance:
                raise ValueError(
                    "draft_proposal='sampled' requires stochastic acceptance:"
                    " greedy acceptance ignores the draft's distributions"
                )
            self._input_spec = replace(
                self._input_spec,
                enable_sampled_draft_proposal=True,
                vocab_size=vocab_size,
            )

        relaxed_topk: int | None = None
        relaxed_delta: float | None = None
        if (
            relaxed_acceptance
            and speculative_config.use_relaxed_acceptance_for_thinking
        ):
            relaxed_topk = speculative_config.relaxed_topk
            relaxed_delta = speculative_config.relaxed_delta
        if use_greedy_acceptance and relaxed_topk is not None:
            raise ValueError(
                "Greedy acceptance has no relaxed rule; it would silently"
                " verify strictly."
            )
        if (
            use_greedy_acceptance
            and speculative_config.synthetic_acceptance_rate is not None
        ):
            raise ValueError(
                "use_greedy_acceptance is incompatible with"
                " synthetic_acceptance_rate"
            )

        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=(
                speculative_config.synthetic_acceptance_rate
            ),
            num_draft_steps=self.num_speculative_tokens,
            use_stochastic=not use_greedy_acceptance,
            relaxed_topk=relaxed_topk,
            relaxed_delta=relaxed_delta,
            draft_proposal=draft_proposal,
            vocab_size=vocab_size,
        )
        # Registered under the hand-written modules' names, so state_dict keys
        # and the weights registry are unchanged.
        self.target = target_model
        self.merger = RaggedTokenMerger(self.devices[0])
        self.draft = draft_model

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        draft_tokens: TensorValue,
        kv_collections: list[PagedCacheValues],
        draft_kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        seed: TensorValue,
        temperature: TensorValue,
        top_k: TensorValue,
        max_k: TensorValue,
        top_p: TensorValue,
        min_top_p: TensorValue,
        signal_buffers: list[BufferValue] | None = None,
        passthrough_kv: Mapping[str, list[PagedCacheValues]] | None = None,
        in_thinking_phase: TensorValue | None = None,
        ep_inputs: list[Value[Any]] | None = None,
        host_input_row_offsets: TensorValue | None = None,
        data_parallel_splits: TensorValue | None = None,
        batch_context_lengths: list[TensorValue] | None = None,
        vision_embeddings: list[TensorValue] | None = None,
        vision_scatter_indices: list[TensorValue] | None = None,
        pinned_bitmask: TensorValue | None = None,
        wait_payload: BufferValue | None = None,
        device_bitmask_scratch: BufferValue | None = None,
        draft_slot_ids: TensorValue | None = None,
        draft_block_offsets: TensorValue | None = None,
        extra: Mapping[str, Any] | None = None,
        draft_probs_full: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        """Runs one block spec-decode iteration: verify K-1, propose K-1.

        Args:
            tokens: 1-D ragged prompt token IDs ``[total_seq_len]``.
            input_row_offsets: ``[batch + 1]`` exclusive prefix sum of the
                per-request sequence lengths.
            draft_tokens: ``[batch, K - 1]`` proposals from the previous
                iteration, or ``[batch, 0]`` on a prefill.
            kv_collections: Per-device target caches for the primary leaf.
            draft_kv_collections: Per-device draft caches.
            return_n_logits: How many tokens of logits the target returns.
            seed: Per-row RNG seed for the acceptance sampler.
            temperature: Per-row sampling temperature.
            top_k: Per-row top-k cutoff.
            max_k: The batch-wide maximum of ``top_k``, or ``-1`` if any
                row has no top-k limit, on CPU.
            top_p: Per-row nucleus cutoff.
            min_top_p: The batch-wide minimum of ``top_p``, on CPU.
            signal_buffers: One buffer per device; empty when not distributed.
            passthrough_kv: Target cache leaves past the primary one.
            in_thinking_phase: Per-row flag enabling relaxed acceptance.
            ep_inputs: Expert-parallel collective inputs, or None.
            host_input_row_offsets: CPU mirror of ``input_row_offsets``; None
                unless the signature is distributed, as with the two below.
            data_parallel_splits: Per-replica batch boundaries, on CPU.
            batch_context_lengths: Per-device cache-length tensors.
            vision_embeddings: Per-device merged vision embeddings; only a
                vision target reads them.
            vision_scatter_indices: Merge positions for the above.
            pinned_bitmask: Structured-output bitmask staged on the host.
            wait_payload: Host-side gate for the bitmask transfer.
            device_bitmask_scratch: Device buffer the bitmask lands in.
            draft_slot_ids: Which block rows to draft, as the tensor's
                extent. An empty one skips the block entirely. Declared only
                under ``include_skippable_draft``, and taken from
                :meth:`decode_inputs` when not passed.
            draft_block_offsets: Runtime row offsets for the block forward,
                paired with ``draft_slot_ids``.
            extra: This model's own graph inputs, reaching its adapters
                through :attr:`BlockBatch.extra` untouched.
            draft_probs_full: ``[batch, num_speculative_tokens, vocab_size]``
                distributions the previous iteration's draft drew its
                proposals from. Required iff ``draft_proposal="sampled"``.

        Returns:
            ``(num_accepted, next_tokens, next_draft_tokens)``, plus
            ``next_draft_probs_full`` under ``draft_proposal="sampled"``.
        """
        if (draft_probs_full is not None) != self._sampled:
            raise ValueError(
                "draft_probs_full is required iff the driver was built with"
                " draft_proposal='sampled'"
            )
        draft_slot_ids, draft_block_offsets = self._draft_rows(
            draft_slot_ids, draft_block_offsets
        )
        if not self._skips_draft_rows:
            # The pipeline binds the row inputs whenever the schedule can
            # skip, whatever the draft. This graph cannot run it on fewer
            # rows, so it leaves them unread and drafts every row.
            draft_slot_ids = draft_block_offsets = None
        signals = signal_buffers or []
        merged_tokens, merged_offsets = self.merger(
            tokens, input_row_offsets, draft_tokens
        )
        # Clean symbolic dims so the target's attention reshapes simplify.
        merged_tokens = merged_tokens.rebind(["merged_seq_len"])
        merged_offsets = merged_offsets.rebind(["input_row_offsets_len"])

        batch = BlockBatch(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            draft_tokens=draft_tokens,
            kv_collections=kv_collections,
            draft_kv_collections=draft_kv_collections,
            passthrough_kv=passthrough_kv or {},
            return_n_logits=return_n_logits,
            devices=self.devices,
            signal_buffers=signals,
            ep_inputs=ep_inputs,
            merged_tokens=merged_tokens,
            merged_offsets=merged_offsets,
            merged_offsets_per_dev=broadcast_per_device(
                merged_offsets, signals, len(self.devices)
            ),
            host_merged_offsets=(
                compute_host_merged_offsets(
                    host_input_row_offsets, draft_tokens
                )
                if host_input_row_offsets is not None
                else None
            ),
            data_parallel_splits=data_parallel_splits,
            data_parallel_degree=self.data_parallel_degree,
            batch_context_lengths=batch_context_lengths or [],
            vision_embeddings=vision_embeddings or [],
            vision_scatter_indices=vision_scatter_indices or [],
            extra=extra or {},
            num_accepted=None,
            draft_slot_ids=draft_slot_ids,
            draft_block_offsets=draft_block_offsets,
            block_size=self.block_size,
        )

        pre_cache_lengths = self._pre_cache_lengths(batch)

        verified = self._target.verify(batch)
        target_logits, target_hidden = verified.logits, verified.hidden

        effective_bitmasks = apply_overlap_bitmask(
            pinned_bitmask,
            wait_payload,
            device_bitmask_scratch,
            num_steps=batch.num_draft_tokens,
            device=batch.device0,
        )

        accepted = self._accept(
            batch,
            target_logits,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            max_k=max_k,
            top_p=top_p,
            min_top_p=min_top_p,
            in_thinking_phase=in_thinking_phase,
            token_bitmasks=effective_bitmasks,
            draft_probs_full=draft_probs_full,
        )

        # Every phase below runs after the accept, so each sees the count
        # rather than re-deriving it.
        batch = replace(batch, num_accepted=accepted.num_accepted)

        caches = self._block_caches(
            batch, pre_cache_lengths, accepted.commit_lengths
        )
        self._proposer.materialize(batch, target_hidden, caches.ctx)

        block_ids, block_offsets = self._build_block(batch, accepted)
        embeds = self._proposer.embed_block(batch, block_ids)
        block_hs = self._proposer.forward_block(
            batch, embeds, block_offsets, caches.block
        )
        if not self._sampled:
            next_draft_tokens = self._proposer.head(
                batch, block_hs, accepted, ArgmaxDraftSampler()
            )
            return (
                accepted.num_accepted,
                accepted.next_tokens,
                self._pad_skipped_rows(batch, next_draft_tokens),
            )

        assert self._vocab_size is not None
        rows = batch.num_draft_seqs
        if not batch.drafts_all_rows:
            # A skippable step drafts for all of its sequences or none, so
            # the drafting rows are a prefix and each row's sampling
            # parameters shrink to it.
            seq_ids = ops.range(
                0, rows, 1, rows, dtype=DType.int64, device=batch.device0
            )
            seed, temperature, top_k, top_p = (
                ops.gather(param, seq_ids, axis=0)
                for param in (seed, temperature, top_k, top_p)
            )
        sampler = SampledDraftSampler(
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            vocab_size=self._vocab_size,
            rows=rows,
        )
        next_draft_tokens = self._proposer.head(
            batch, block_hs, accepted, sampler
        )
        n = self.num_speculative_tokens
        return (
            accepted.num_accepted,
            accepted.next_tokens,
            self._pad_skipped_rows(batch, next_draft_tokens.rebind([rows, n])),
            # A skipped row proposes the invalid-draft sentinel, so it carries
            # no distribution: the verdict reads ``q == 0`` as none and never
            # divides by it.
            self._pad_skipped_rows(
                batch,
                sampler.distributions().rebind([rows, n, self._vocab_size]),
                fill=0,
            ),
        )

    def _accept(
        self,
        batch: BlockBatch,
        target_logits: TensorValue,
        *,
        seed: TensorValue,
        temperature: TensorValue,
        top_k: TensorValue,
        max_k: TensorValue,
        top_p: TensorValue,
        min_top_p: TensorValue,
        in_thinking_phase: TensorValue | None,
        token_bitmasks: TensorValue | None,
        draft_probs_full: TensorValue | None,
    ) -> Accepted:
        """Verifies the proposals, then corrects the rows that had none.

        Two kinds of row carry no real proposal and must report zero accepted
        however the sampler scored them: a prefill iteration, where
        ``draft_tokens`` is ``[batch, 0]``, and a decode row padded entirely
        with :data:`MAGIC_DRAFT_TOKEN_ID`. Reporting the sampler's number
        instead would inflate the acceptance metric and, worse, place the
        block forward past the tokens the iteration actually commits.
        """
        device = batch.device0
        num_accepted, recovered, bonus = self.acceptance_sampler(
            batch.draft_tokens,
            target_logits,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            max_k=max_k,
            top_p=top_p,
            min_top_p=min_top_p,
            in_thinking_phase=in_thinking_phase,
            token_bitmasks=token_bitmasks,
            draft_probs_full=draft_probs_full,
        )

        num_steps_u32 = _shape_to_scalar(
            batch.num_draft_tokens, device, dtype=DType.uint32
        )
        is_prefill = (num_steps_u32 == 0).broadcast_to(["batch_size"])

        num_magic_tokens = ops.squeeze(
            ops.sum(
                (batch.draft_tokens == MAGIC_DRAFT_TOKEN_ID)
                .cast(DType.int32)
                .rebind(["batch_size", "num_steps"]),
                axis=-1,
            ),
            axis=-1,
        )
        num_steps_i32 = _shape_to_scalar(
            batch.num_draft_tokens, device, dtype=DType.int32
        )
        is_dummy_draft = num_magic_tokens == num_steps_i32.broadcast_to(
            ["batch_size"]
        )
        no_proposals = is_prefill | is_dummy_draft
        num_accepted = ops.where(
            no_proposals,
            ops.constant(0, num_accepted.dtype, device=device).broadcast_to(
                ["batch_size"]
            ),
            num_accepted,
        )

        prompt_lens = (
            batch.input_row_offsets[1:] - batch.input_row_offsets[:-1]
        ).rebind(["batch_size"])
        # A dummy-draft row in a K>0 batch is either a decode row with no real
        # drafts (both branches equal) or a prefill row in a mixed batch, which
        # commits its whole chunk. The nested ``ops.where`` keeps
        # ``is_dummy_draft`` -- an empty-axis reduction at K == 0 -- off the
        # prefill path.
        decode_commit = (num_accepted + 1).cast(DType.uint32)
        commit_lengths = ops.where(
            is_prefill,
            prompt_lens,
            ops.where(is_dummy_draft, prompt_lens, decode_commit),
        )

        # The committed token is ``recovered`` at the accepted count, or
        # ``bonus`` when every proposal was accepted. A prefill row reads 0.
        target_tokens = ops.concat([recovered, bonus], axis=1)
        gather_idx = ops.where(
            is_prefill,
            ops.constant(0, DType.int64, device=device).broadcast_to(
                ["batch_size"]
            ),
            num_accepted.cast(DType.int64),
        )
        next_tokens = ops.gather_nd(
            target_tokens, ops.unsqueeze(gather_idx, axis=-1), batch_dims=1
        )

        return Accepted(
            num_accepted=num_accepted,
            next_tokens=next_tokens,
            commit_lengths=commit_lengths,
            is_prefill=is_prefill,
        )

    def _pre_cache_lengths(self, batch: BlockBatch) -> list[TensorValue]:
        """Where the draft materializes the target's context KV.

        Read before the verify for the reading, though these are graph inputs
        and so hold the pre-iteration lengths whenever they are read.
        """
        source = (
            batch.draft_kv_collections
            if self._ctx_at_draft_cache_length
            else batch.kv_collections
        )
        if self.data_parallel_degree > 1:
            # A device's cache lengths cover its replica's rows, so the global
            # name would be a false assertion.
            return [kv.cache_lengths for kv in source]
        return [ops.rebind(kv.cache_lengths, ["batch_size"]) for kv in source]

    def _commit_lengths_per_device(
        self, batch: BlockBatch, commit_lengths: TensorValue
    ) -> list[TensorValue]:
        """The commit counts each device needs, for its own rows.

        The accept runs on device 0, so under tensor parallelism the counts
        still have to reach the others; under data parallelism each device
        also keeps only the rows its replica owns.
        """
        per_dev = broadcast_per_device(
            commit_lengths, batch.signal_buffers, batch.n_devs
        )
        if batch.data_parallel_degree == 1:
            return per_dev

        splits = batch.data_parallel_splits
        assert splits is not None, (
            "a data-parallel block graph must pass data_parallel_splits"
        )
        local: list[TensorValue] = []
        for i, replica in enumerate(batch.replica_of):
            rows = ops.slice_tensor(
                per_dev[i],
                [
                    (
                        slice(splits[replica], splits[replica + 1]),
                        f"block_commit_split_{i}",
                    )
                ],
            )
            local.append(ops.rebind(rows, [f"replica_{replica}_batch_size"]))
        return local

    def _block_caches(
        self,
        batch: BlockBatch,
        pre_cache_lengths: list[TensorValue],
        commit_lengths: TensorValue,
    ) -> BlockCaches:
        """The draft cache at the context position and at the block position."""
        commit_per_dev = self._commit_lengths_per_device(batch, commit_lengths)
        ctx = [
            replace(kv, cache_lengths=pre)
            for kv, pre in zip(
                batch.draft_kv_collections, pre_cache_lengths, strict=True
            )
        ]
        block = [
            replace(kv, cache_lengths=pre + commit)
            for kv, pre, commit in zip(
                batch.draft_kv_collections,
                pre_cache_lengths,
                commit_per_dev,
                strict=True,
            )
        ]
        return BlockCaches(ctx=ctx, block=block)

    def _build_block(
        self, batch: BlockBatch, accepted: Accepted
    ) -> tuple[TensorValue, list[TensorValue]]:
        """The block's flattened token ids and its per-device row offsets.

        The block is the committed token followed by ``K - 1`` mask slots, one
        per token to draft, laid out contiguously per request -- so its offsets
        are a fixed ``K`` stride rather than a prefix sum.
        """
        k = self.block_size
        device = batch.device0
        mask_tail = ops.constant(
            self._proposer.mask_token_id, DType.int64, device=device
        ).broadcast_to(["batch_size", k - 1])
        block_ids = ops.concat(
            [ops.unsqueeze(accepted.next_tokens, axis=1), mask_tail], axis=1
        )

        block_ids_flat = block_ids.reshape((-1,))
        if not batch.drafts_all_rows:
            assert batch.draft_slot_ids is not None
            assert batch.draft_block_offsets is not None
            # The selection's extent is the row count: ``arange(batch * K)``
            # to draft, empty to skip. Taking the rows through a gather
            # collapses the block, the draft forward and the draft lm_head to
            # zero rows with no branch in the graph, and therefore none inside
            # a device-graph capture either. The offsets go all-zero, which
            # reads as ``batch_size`` sequences of zero query rows and so
            # leaves cache_lengths and the lookup table aligned 1:1 with the
            # ragged batch.
            return (
                ops.gather(block_ids_flat, batch.draft_slot_ids, axis=0),
                [batch.draft_block_offsets],
            )

        offsets = [
            ops.range(
                start=0,
                stop=batch.input_row_offsets.shape[0],
                out_dim="input_row_offsets_len",
                device=dev,
                dtype=DType.uint32,
            )
            * k
            for dev in self.devices
        ]
        if batch.data_parallel_degree > 1:
            splits = batch.data_parallel_splits
            assert splits is not None
            offsets = [
                local_row_offsets(
                    offsets[i], splits, replica, f"block_offset_split_{i}"
                )
                for i, replica in enumerate(batch.replica_of)
            ]
        return block_ids_flat, offsets

    def _pad_skipped_rows(
        self,
        batch: BlockBatch,
        draft_values: TensorValue,
        fill: int = MAGIC_DRAFT_TOKEN_ID,
    ) -> TensorValue:
        """Restores a draft output's ``batch_size`` leading dim after a skip.

        The head returns one row per *drafting* sequence, which is zero of
        them on a skipping step, but the graph's draft outputs are fixed at
        ``batch_size`` rows, because device-graph capture records the output
        shapes and replays into them. Proposals are filled with
        :data:`MAGIC_DRAFT_TOKEN_ID`, which the next step's accept already
        recognizes as "this row carried no proposal" and scores zero.
        ``broadcast_to`` of a constant hoists out of the step, so the fill
        costs nothing per iteration.
        """
        if batch.drafts_all_rows:
            return draft_values
        # The head may return fewer than ``K - 1`` columns when the driver
        # verifies fewer proposals than the block holds.
        trailing = list(draft_values.shape[1:])
        skipped = ops.constant(
            fill, draft_values.dtype, device=batch.device0
        ).broadcast_to([Dim("batch_size") - batch.num_draft_seqs, *trailing])
        return ops.concat([draft_values, skipped], axis=0).rebind(
            ["batch_size", *trailing]
        )

    @override
    @property
    def input_spec(self) -> SpecDecodeInputTypeSpec:
        return self._input_spec

    @override
    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return self._target.ep_input_types()
