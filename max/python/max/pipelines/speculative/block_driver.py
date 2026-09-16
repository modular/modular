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
from typing import Any, Generic, Protocol, TypeVar

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import AcceptanceSampler
from typing_extensions import override

from .config import MAGIC_DRAFT_TOKEN_ID, SpeculativeConfig
from .ragged_token_merger import RaggedTokenMerger, _shape_to_scalar
from .spec_input_types import (
    SpecDecodeGraphSignature,
    SpecDecodeInputTypeSpec,
)
from .spec_target import SpecDecodeTarget
from .unified_graph_ops import apply_overlap_bitmask

__all__ = [
    "Accepted",
    "BlockBatch",
    "BlockCaches",
    "BlockDriver",
    "BlockProposer",
]

_TargetHiddenT = TypeVar("_TargetHiddenT", contravariant=True)
"""The target's captured-hidden payload, opaque to the driver.

Handed from :meth:`SpecDecodeTarget.verify` straight to
:meth:`BlockProposer.materialize`. Contravariant because it is only ever
consumed here -- nothing on this side hands it back out.
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


class BlockProposer(Protocol[_TargetHiddenT]):
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
    ) -> TensorValue:
        """Runs the draft over the whole block; returns its hidden states."""
        ...

    def head(
        self, batch: BlockBatch, block_hs: TensorValue, accepted: Accepted
    ) -> TensorValue:
        """Turns the block's hidden states into ``[batch_size, K - 1]`` tokens.

        Owns the head entirely: which ``lm_head``, whether the anchor slot is
        sliced off before or after the projection, logit softcapping, and any
        model-specific draft sampling.
        """
        ...


class BlockDriver(SpecDecodeGraphSignature, Module, Generic[_TargetHiddenT]):
    """Merge -> verify -> mask -> accept -> materialize -> block -> head."""

    def __init__(
        self,
        target: SpecDecodeTarget[BlockBatch, _TargetHiddenT],
        proposer: BlockProposer[_TargetHiddenT],
        *,
        target_model: Module,
        draft_model: Module,
        input_spec: SpecDecodeInputTypeSpec,
        speculative_config: SpeculativeConfig,
        enable_structured_output: bool = False,
        relaxed_acceptance: bool = False,
    ) -> None:
        super().__init__()
        self._target = target
        self._proposer = proposer
        self._input_spec = replace(
            input_spec, enable_structured_output=enable_structured_output
        )
        self.devices = self._input_spec.devices
        self.enable_structured_output = enable_structured_output
        self.block_size = proposer.block_size
        # A block draft's budget is its width, less the anchor slot unless the
        # anchor predicts -- never the config's step count directly.
        self.num_speculative_tokens = self.block_size - (
            0 if proposer.samples_from_anchor else 1
        )

        relaxed_topk: int | None = None
        relaxed_delta: float | None = None
        if (
            relaxed_acceptance
            and speculative_config.use_relaxed_acceptance_for_thinking
        ):
            relaxed_topk = speculative_config.relaxed_topk
            relaxed_delta = speculative_config.relaxed_delta

        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=(
                speculative_config.synthetic_acceptance_rate
            ),
            num_draft_steps=self.num_speculative_tokens,
            use_stochastic=True,
            relaxed_topk=relaxed_topk,
            relaxed_delta=relaxed_delta,
        )
        # Registered under the hand-written modules' names, so state_dict keys
        # and the weights registry are unchanged.
        self.target = target_model
        self.merger = RaggedTokenMerger(self.devices[0])
        self.draft = draft_model

    def _per_dev(
        self, value: TensorValue, signal_buffers: list[BufferValue]
    ) -> list[TensorValue]:
        """One entry per device, broadcasting only when there is more than one.

        A ``distributed=False`` graph declares no signal buffers, so it cannot
        broadcast; at one device the list is the value itself.
        """
        if len(self.devices) == 1:
            return [value]
        return ops.distributed_broadcast(value, signal_buffers)

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
        pinned_bitmask: TensorValue | None = None,
        wait_payload: BufferValue | None = None,
        device_bitmask_scratch: BufferValue | None = None,
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
            max_k: The batch-wide maximum of ``top_k``, on CPU.
            top_p: Per-row nucleus cutoff.
            min_top_p: The batch-wide minimum of ``top_p``, on CPU.
            signal_buffers: One buffer per device; empty when not distributed.
            passthrough_kv: Target cache leaves past the primary one.
            in_thinking_phase: Per-row flag enabling relaxed acceptance.
            ep_inputs: Expert-parallel collective inputs, or None.
            pinned_bitmask: Structured-output bitmask staged on the host.
            wait_payload: Host-side gate for the bitmask transfer.
            device_bitmask_scratch: Device buffer the bitmask lands in.

        Returns:
            ``(num_accepted, next_tokens, next_draft_tokens)``.
        """
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
            merged_offsets_per_dev=self._per_dev(merged_offsets, signals),
        )

        # The draft materializes the target's context KV at the position the
        # target's primary leaf held before this iteration.
        pre_cache_lengths = [
            ops.rebind(kv.cache_lengths, ["batch_size"])
            for kv in kv_collections
        ]

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
        )

        caches = self._block_caches(
            batch, pre_cache_lengths, accepted.commit_lengths
        )
        self._proposer.materialize(batch, target_hidden, caches.ctx)

        block_ids, block_offsets = self._build_block(batch, accepted)
        embeds = self._proposer.embed_block(batch, block_ids)
        block_hs = self._proposer.forward_block(
            batch, embeds, block_offsets, caches.block
        )
        next_draft_tokens = self._proposer.head(batch, block_hs, accepted)

        return (accepted.num_accepted, accepted.next_tokens, next_draft_tokens)

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
            seed=seed[0],
            temperature=temperature,
            top_k=top_k,
            max_k=max_k,
            top_p=top_p,
            min_top_p=min_top_p,
            in_thinking_phase=in_thinking_phase,
            token_bitmasks=token_bitmasks,
        )

        num_steps_u32 = _shape_to_scalar(
            batch.num_draft_tokens, device, dtype=DType.uint32
        )
        zero_u32 = ops.constant(0, DType.uint32, device=device)
        is_prefill = (num_steps_u32 == zero_u32).broadcast_to(["batch_size"])

        magic_token = ops.constant(
            MAGIC_DRAFT_TOKEN_ID, DType.int64, device=device
        )
        num_magic_tokens = ops.squeeze(
            ops.sum(
                (batch.draft_tokens == magic_token)
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

    def _block_caches(
        self,
        batch: BlockBatch,
        pre_cache_lengths: list[TensorValue],
        commit_lengths: TensorValue,
    ) -> BlockCaches:
        """The draft cache at the context position and at the block position."""
        ctx = [
            replace(kv, cache_lengths=pre)
            for kv, pre in zip(
                batch.draft_kv_collections, pre_cache_lengths, strict=True
            )
        ]
        block = [
            replace(kv, cache_lengths=pre + commit_lengths)
            for kv, pre in zip(
                batch.draft_kv_collections, pre_cache_lengths, strict=True
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

        offsets = [
            ops.range(
                start=0,
                stop=batch.input_row_offsets.shape[0],
                out_dim="input_row_offsets_len",
                device=dev,
                dtype=DType.uint32,
            )
            * ops.constant(k, DType.uint32, device=dev)
            for dev in self.devices
        ]
        return block_ids.reshape((-1,)), offsets

    @override
    @property
    def input_spec(self) -> SpecDecodeInputTypeSpec:
        return self._input_spec

    @override
    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return self._target.ep_input_types()
