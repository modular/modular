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
"""Model-agnostic driver for sequential (eagle / MTP) speculative decoding.

Every sequential unified spec-decode module in the tree runs the same seven
phases -- merge, verify, mask, accept, shift, propose, pack -- and differs only
in how it calls its target, how it calls its draft, and what per-step cache
bookkeeping the draft needs. :class:`SequentialDriver` owns the phases; a model
contributes a :class:`SpecDecodeTarget` and a :class:`SequentialProposer`.

The two adapters are deliberately not ``Module`` subclasses. The driver
registers the target and draft modules under the same attribute names the
hand-written modules used, so weight loading, ``state_dict`` keys and the
weights registry are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Generic, Literal, Protocol, TypeVar

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    DimLike,
    Graph,
    ProfileScopeColor,
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
    _reshape_target_logits,
)
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.kv_cache.paged_kv_cache.increment_cache_lengths import (
    increment_cache_lengths_from_counts,
)
from typing_extensions import override

from .config import MAGIC_DRAFT_TOKEN_ID, SpeculativeConfig
from .ragged_token_merger import RaggedTokenMerger, _shape_to_scalar
from .spec_input_types import (
    SpecDecodeGraphSignature,
    SpecDecodeInputTypeSpec,
)
from .spec_target import SpecDecodeTarget
from .unified_graph_ops import (
    accept_and_pick_next_tokens,
    apply_overlap_bitmask,
    gather_accepted_hidden_states,
    merge_tokens_and_host_offsets,
    shift_corrected_tokens,
)

__all__ = [
    "CarryDimNames",
    "DecodeKVSwap",
    "DraftCache",
    "DraftStepInput",
    "Proposed",
    "ReuseSpec",
    "SequentialBatch",
    "SequentialDriver",
    "SequentialProposer",
]


@dataclass(frozen=True)
class SequentialBatch:
    """One spec-decode iteration's graph inputs, merged and broadcast.

    Built by the driver in phase 1 so that the target adapter, the proposer and
    the per-step loop all read the same merged offsets rather than each
    recomputing the broadcast.
    """

    tokens: TensorValue
    input_row_offsets: TensorValue
    draft_tokens: TensorValue
    signal_buffers: list[BufferValue]
    kv_collections: list[PagedCacheValues]
    draft_kv_collections: list[PagedCacheValues]
    passthrough_kv: Mapping[str, list[PagedCacheValues]]
    """Cache leaves beyond the primary pair, keyed by the model's name for them.

    The driver drives exactly one target leaf and one draft leaf: it advances
    the draft leaf's cache lengths and applies the declared decode swaps to it.
    A model whose attention is split across paired caches names the leaf the
    driver should drive as the primary one and reaches the rest through here,
    listing in :attr:`SequentialProposer.passthrough_decode_swaps` any that
    need the same ``q = 1`` retarget."""
    draft_cache_lengths: list[TensorValue]
    """Per-device draft cache lengths for this step.

    Always equal to each :attr:`draft_kv_collections` entry's
    ``cache_lengths`` -- except for a draft that reads a cache it does not own
    (:attr:`SequentialProposer.draft_cache` is ``TARGET``), where the count is
    a RoPE position rather than a write pointer and the collections keep the
    lengths they came in with."""
    return_n_logits: TensorValue
    host_input_row_offsets: TensorValue
    data_parallel_splits: TensorValue
    batch_context_lengths: list[TensorValue]
    ep_inputs: list[Value[Any]] | None
    devices: Sequence[DeviceRef]
    data_parallel_degree: int

    vision_embeddings: list[TensorValue]
    """Per-device merged vision embeddings, empty for a text-only target.

    A vision target scatters these into the merged sequence before running
    its stack, so they reach the target adapter rather than the driver's
    phases -- which never read them."""
    vision_scatter_indices: list[TensorValue]
    """Per-device merge positions for :attr:`vision_embeddings`."""

    merged_tokens: TensorValue
    merged_offsets: TensorValue
    host_merged_offsets: TensorValue
    merged_offsets_per_dev: list[TensorValue]
    """The verify window's offsets, on every device. Stable across the loop."""

    query_offsets_per_dev: list[TensorValue]
    """Offsets over *this* draft call's query, on every device.

    The merged offsets for step 0, which runs over the whole corrected
    sequence; a one-token-per-request ramp for steps 1..K-1. A draft that
    cross-attends into the target's cache needs both these and the stable
    :attr:`merged_offsets_per_dev`, which is why they are separate fields."""
    host_query_offsets: TensorValue
    """CPU mirror of :attr:`query_offsets_per_dev`."""

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
        """How many tokens the previous iteration proposed, ``K``."""
        return self.draft_tokens.shape[1]


_TargetHiddenT = TypeVar("_TargetHiddenT", contravariant=True)
"""The target's hidden-state payload.

Opaque to the driver, which passes it from :meth:`SpecDecodeTarget.verify`
straight to :meth:`SequentialProposer.prefill`. Contravariant because it is
only ever consumed here -- nothing on this side hands it back out.
"""


@dataclass(frozen=True)
class DraftStepInput:
    """Loop-carried state between draft steps.

    Every draft in the tree threads the previous step's token and its
    per-device hidden state.
    """

    tokens: TensorValue
    hidden: list[TensorValue]
    reuse: list[TensorValue] = field(default_factory=list)
    """Step 0's reused result, empty for a draft that declares no
    :attr:`SequentialProposer.reuse`."""


@dataclass(frozen=True)
class Proposed:
    """One draft invocation's contribution.

    Carries the step's logits alongside the hidden state the next step
    consumes.
    """

    logits: TensorValue
    hidden: list[TensorValue]
    reuse: list[TensorValue] = field(default_factory=list)
    """Per-device work step 0 did that every later step reuses unchanged.

    Read from the prefill only. A later step returns nothing here, because
    reusing step 0's result is the point."""


class DecodeKVSwap(Enum):
    """A single per-step edit to the draft's ``PagedCacheValues``.

    The draft's step-0 pass runs over the whole corrected sequence, so its
    query length is whatever the merged batch holds; steps 1..K-1 run one
    token per request, ``q = 1``. ``PagedCacheValues`` carries a dispatch
    buffer for each, the ``q = 1`` one under a ``draft_`` prefix -- which
    names the shorter query length, not the draft model; the target's cache
    leaf carries a ``draft_`` variant too. Selecting between them is a
    declaration rather than an inline ``replace`` per model.
    """

    MAX_PROMPT_LENGTH_ONE = "max_prompt_length_one"
    DRAFT_ATTENTION_DISPATCH_METADATA = "draft_attention_dispatch_metadata"
    DRAFT_MLA_NUM_PARTITIONS = "draft_mla_num_partitions"


@dataclass(frozen=True)
class ReuseSpec:
    """A per-step result the draft carries beside its hidden state.

    Declaring it is what lets the driver gather and rebind the reused result
    exactly as it does the hidden carry, rather than the model threading a
    second tensor through a loop of its own. Both halves arrive together, so
    a dim without a name to gather under is unrepresentable.
    """

    dim: DimLike
    """Trailing dim of :attr:`Proposed.reuse`."""
    split_prefix: str
    """Names the gather that carries it across steps."""

    def __post_init__(self) -> None:
        if not self.split_prefix:
            raise ValueError("ReuseSpec needs a non-empty split_prefix")


@dataclass(frozen=True)
class CarryDimNames:
    """How the driver names the carried hidden state's batch dim per step."""

    prefix: str
    """Names the carry dims ``{prefix}{step}_batch``."""
    per_device: bool = False
    """Whether the name also carries the device index.

    Naming one dim for every device asserts the per-device carries have equal
    batch, which holds under pure TP and not under DP. A draft that can run
    under DP must name them apart."""


class DraftCache(Enum):
    """Which cache the draft writes, and so what the driver advances."""

    OWN = "own"
    """The draft has a cache of its own.

    The driver advances its lengths by the accepted count before the loop and
    by one per step, and substitutes them into the collections."""
    TARGET = "target"
    """The draft only reads the target's cache.

    It writes no KV slot, so its position stays fixed across the steps and the
    per-step advance is skipped. The accepted-count advance still runs,
    reaching the draft as :attr:`SequentialBatch.draft_cache_lengths` rather
    than being substituted into the collections, which would lengthen the
    window it attends over."""


class SequentialProposer(Protocol[_TargetHiddenT]):
    """A draft that emits one token per step, K steps deep."""

    hidden_dim: DimLike
    """Trailing dim of the carried hidden state, used when rebinding it."""
    reuse: ReuseSpec | None
    """The result carried beside the hidden state, or ``None`` for a draft
    with none."""
    decode_swaps: tuple[DecodeKVSwap, ...]
    """Which draft-cache fields to retarget to ``q = 1`` before the loop."""
    passthrough_decode_swaps: tuple[str, ...]
    """Which :attr:`SequentialBatch.passthrough_kv` leaves swap too.

    The primary draft leaf always takes :attr:`decode_swaps`. A paired cache
    is only sometimes symmetric."""
    draft_cache: DraftCache
    """Which cache the draft writes, and so what the driver advances."""
    split_prefix: str
    """Names the accepted-position gather and each step's draft subgraph."""
    carry_dim_names: CarryDimNames
    """How each step's carry dims are named."""
    step_hidden_mode: ReturnHiddenStates
    """``LAST_PER_DEVICE`` returns post-allgather full-batch tensors and so
    needs a DP slice; ``ALL`` returns per-device ones and must not be sliced.
    The driver applies that rule once."""
    uses_thinking_phase: bool
    """Whether the draft's acceptance test reads ``in_thinking_phase``."""

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: _TargetHiddenT,
    ) -> Proposed:
        """Runs draft step 0 over the whole target-corrected sequence."""
        ...

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        """Runs draft step ``index`` over one token per batch element."""
        ...


class SequentialDriver(
    SpecDecodeGraphSignature, Module, Generic[_TargetHiddenT]
):
    """Drives the sequential speculative-decoding loop.

    Each iteration runs merge, verify, mask, accept, shift, propose,
    and pack.
    """

    def __init__(
        self,
        target: SpecDecodeTarget[SequentialBatch, _TargetHiddenT],
        proposer: SequentialProposer[_TargetHiddenT],
        *,
        target_model: Module,
        draft_model: Module,
        input_spec: SpecDecodeInputTypeSpec,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
        use_greedy_acceptance: bool = False,
        per_row_acceptance_seed: bool = False,
        draft_proposal: Literal["argmax", "sampled"] = "argmax",
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self._target = target
        self._proposer = proposer
        self._draft_proposal = draft_proposal
        self._vocab_size = vocab_size
        self._input_spec = replace(
            input_spec,
            include_in_thinking_phase=proposer.uses_thinking_phase,
            enable_structured_output=enable_structured_output,
        )
        if draft_proposal == "sampled":
            # Declares the draft_probs_full input, which only this mode reads.
            self._input_spec = replace(
                self._input_spec,
                enable_sampled_draft_proposal=True,
                vocab_size=vocab_size,
            )
        # Both come off the spec, so signature and loop share one device set.
        self.devices = self._input_spec.devices
        self.data_parallel_degree = self._input_spec.data_parallel_degree
        self.enable_structured_output = enable_structured_output
        self._per_row_acceptance_seed = per_row_acceptance_seed

        # The one contradiction the declarations cannot rule out, being between
        # a proposer's flag and how the driver was built.
        if (
            self.data_parallel_degree > 1
            and not proposer.carry_dim_names.per_device
        ):
            raise ValueError(
                f"{type(proposer).__name__} names one carry dim for every"
                " device, which asserts the replicas hold equal batch, but the"
                f" driver was built for data_parallel_degree="
                f"{self.data_parallel_degree}. Set"
                " CarryDimNames(per_device=True)."
            )
        self.num_draft_steps = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            and speculative_config.num_speculative_tokens is not None
            else 1
        )

        if use_greedy_acceptance and speculative_config is not None:
            if speculative_config.use_relaxed_acceptance_for_thinking:
                raise ValueError(
                    "use_greedy_acceptance is incompatible with "
                    "use_relaxed_acceptance_for_thinking"
                )
            if speculative_config.synthetic_acceptance_rate is not None:
                raise ValueError(
                    "use_greedy_acceptance is incompatible with "
                    "synthetic_acceptance_rate"
                )

        if draft_proposal == "sampled":
            if vocab_size is None:
                raise ValueError(
                    "vocab_size is required when draft_proposal='sampled':"
                    " draft_probs_full's trailing dim has to be static"
                )
            if use_greedy_acceptance:
                raise ValueError(
                    "draft_proposal='sampled' is incompatible with "
                    "use_greedy_acceptance"
                )
            if per_row_acceptance_seed:
                # The sampled verdict's residual draws come off one batch-level
                # RNG stream, so it has no per-row seed path -- see
                # stochastic_acceptance_sampler.
                raise ValueError(
                    "draft_proposal='sampled' is incompatible with "
                    "per_row_acceptance_seed"
                )
            if speculative_config is not None:
                if speculative_config.synthetic_acceptance_rate is not None:
                    raise ValueError(
                        "draft_proposal='sampled' is incompatible with "
                        "synthetic_acceptance_rate"
                    )
                if speculative_config.use_relaxed_acceptance_for_thinking:
                    raise ValueError(
                        "draft_proposal='sampled' is incompatible with "
                        "use_relaxed_acceptance_for_thinking"
                    )

        # Relaxed acceptance is the in-thinking-phase feature; a proposer that
        # does not bind that flag cannot use it.
        relaxed_topk: int | None = None
        relaxed_delta: float | None = None
        if (
            proposer.uses_thinking_phase
            and speculative_config is not None
            and speculative_config.use_relaxed_acceptance_for_thinking
        ):
            relaxed_topk = speculative_config.relaxed_topk
            relaxed_delta = speculative_config.relaxed_delta

        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=(
                speculative_config.synthetic_acceptance_rate
                if speculative_config
                else None
            ),
            num_draft_steps=self.num_draft_steps,
            use_stochastic=not use_greedy_acceptance,
            relaxed_topk=relaxed_topk,
            relaxed_delta=relaxed_delta,
            draft_proposal=draft_proposal,
            vocab_size=vocab_size,
        )
        # Registered under the names the hand-written modules used, so the
        # state_dict keys and the weights registry are unchanged.
        self.target = target_model
        self.merger = RaggedTokenMerger(self.devices[0])
        self.draft = draft_model

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        draft_tokens: TensorValue,
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        seed: TensorValue,
        temperature: TensorValue,
        top_k: TensorValue,
        max_k: TensorValue,
        top_p: TensorValue,
        min_top_p: TensorValue,
        in_thinking_phase: TensorValue | None = None,
        ep_inputs: list[Value[Any]] | None = None,
        draft_kv_collections: list[PagedCacheValues] | None = None,
        passthrough_kv: Mapping[str, list[PagedCacheValues]] | None = None,
        vision_embeddings: list[TensorValue] | None = None,
        vision_scatter_indices: list[TensorValue] | None = None,
        pinned_bitmask: TensorValue | None = None,
        wait_payload: BufferValue | None = None,
        device_bitmask_scratch: BufferValue | None = None,
        draft_probs_full: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        """Runs one spec-decode iteration: verify K drafts, propose K more.

        Args:
            tokens: 1-D ragged prompt token IDs ``[total_seq_len]``; segment
                boundaries come from ``input_row_offsets``.
            input_row_offsets: ``[batch + 1]`` exclusive prefix sum of the
                per-request sequence lengths, on device 0.
            draft_tokens: ``[batch, K]`` proposals from the previous iteration.
            signal_buffers: One buffer per device for collective signaling.
            kv_collections: Per-device target caches.
            return_n_logits: How many tokens of logits the target returns.
            host_input_row_offsets: CPU mirror of ``input_row_offsets``, used
                to compute the merged offsets without a device sync.
            data_parallel_splits: Per-replica batch boundaries, on CPU.
            batch_context_lengths: Per-device cache-length tensors.
            seed: Per-row RNG seed for the acceptance sampler.
            temperature: Per-row sampling temperature.
            top_k: Per-row top-k cutoff.
            max_k: The batch-wide maximum of ``top_k``, on CPU.
            top_p: Per-row nucleus cutoff.
            min_top_p: The batch-wide minimum of ``top_p``, on CPU.
            in_thinking_phase: Per-row flag enabling relaxed acceptance;
                ignored by proposers that do not declare it.
            ep_inputs: Expert-parallel collective inputs, or None.
            draft_kv_collections: Per-device draft caches.
            passthrough_kv: Cache leaves beyond the primary pair, keyed by name.
            vision_embeddings: Per-device merged vision embeddings; only a
                vision target reads them.
            vision_scatter_indices: Merge positions for the above.
            pinned_bitmask: Structured-output bitmask staged on the host.
            wait_payload: Host-side gate for the bitmask transfer.
            device_bitmask_scratch: Device buffer the bitmask lands in.
            draft_probs_full: ``[batch, K, vocab_size]`` distributions the
                previous iteration's draft drew its proposals from. Required
                iff the driver was built with ``draft_proposal="sampled"``,
                which is also the only mode that returns a fourth output.

        Returns:
            ``(num_accepted, next_tokens, next_draft_tokens)``, plus
            ``next_draft_probs_full`` under ``draft_proposal="sampled"``.
        """
        if (draft_probs_full is None) != (self._draft_proposal == "argmax"):
            raise ValueError(
                "draft_probs_full is required iff the driver was built with"
                f" draft_proposal='sampled' (got draft_proposal="
                f"'{self._draft_proposal}', draft_probs_full="
                f"{'a tensor' if draft_probs_full is not None else 'None'})"
            )
        with Graph.current.profile_scope(
            "target_forward", color=ProfileScopeColor.ORANGE
        ):
            merged_tokens, merged_offsets, host_merged_offsets = (
                merge_tokens_and_host_offsets(
                    self.merger,
                    tokens,
                    input_row_offsets,
                    draft_tokens,
                    host_input_row_offsets,
                )
            )

            # Broadcast merged_offsets once and reuse the per-device list for
            # target, draft step 0 and the accept-position gather, rather than
            # each running the broadcast itself.
            merged_offsets_per_dev = ops.distributed_broadcast(
                merged_offsets, signal_buffers
            )

            assert draft_kv_collections is not None
            batch = SequentialBatch(
                tokens=tokens,
                input_row_offsets=input_row_offsets,
                draft_tokens=draft_tokens,
                signal_buffers=signal_buffers,
                kv_collections=kv_collections,
                draft_kv_collections=draft_kv_collections,
                passthrough_kv=passthrough_kv or {},
                draft_cache_lengths=[
                    kv.cache_lengths for kv in draft_kv_collections
                ],
                return_n_logits=return_n_logits,
                host_input_row_offsets=host_input_row_offsets,
                data_parallel_splits=data_parallel_splits,
                batch_context_lengths=batch_context_lengths,
                ep_inputs=ep_inputs,
                devices=self.devices,
                data_parallel_degree=self.data_parallel_degree,
                merged_tokens=merged_tokens,
                merged_offsets=merged_offsets,
                host_merged_offsets=host_merged_offsets,
                merged_offsets_per_dev=merged_offsets_per_dev,
                query_offsets_per_dev=merged_offsets_per_dev,
                host_query_offsets=host_merged_offsets,
                vision_embeddings=vision_embeddings or [],
                vision_scatter_indices=vision_scatter_indices or [],
            )

            verified = self._target.verify(batch)

        with Graph.current.profile_scope(
            "verify_and_sample", color=ProfileScopeColor.ORANGE
        ):
            effective_bitmasks = apply_overlap_bitmask(
                pinned_bitmask,
                wait_payload,
                device_bitmask_scratch,
                num_steps=batch.num_draft_tokens,
                device=batch.device0,
            )

            # ``seed`` is the ``[batch_size]`` uint64 buffer feeding
            # ``topk_fused_sampling`` per row. Under argmax proposals the
            # verdict *is* that draw, so the seed picks the committed token: a
            # rank-0 seed keys row ``b`` position ``p`` off ``seed[0] + (b *
            # positions + p) * gamma``, the full tensor keys each row off its
            # own. Either way the committed marginal is the truncated target
            # distribution, so this buys reproducibility, not accuracy.
            num_accepted, recovered, bonus, next_tokens = (
                accept_and_pick_next_tokens(
                    self.acceptance_sampler,
                    draft_tokens,
                    verified.logits,
                    seed=seed if self._per_row_acceptance_seed else seed[0],
                    temperature=temperature,
                    top_k=top_k,
                    max_k=max_k,
                    top_p=top_p,
                    min_top_p=min_top_p,
                    in_thinking_phase=in_thinking_phase,
                    token_bitmasks=effective_bitmasks,
                    draft_probs_full=draft_probs_full,
                )
            )

            num_accepted, next_tokens = self._discard_absent_proposals(
                batch, num_accepted, recovered, bonus, next_tokens
            )

        with Graph.current.profile_scope(
            "draft_forward", color=ProfileScopeColor.ORANGE
        ):
            shifted_corrected = shift_corrected_tokens(
                self.merger, tokens, input_row_offsets, recovered, bonus
            )

            all_draft_tokens, all_draft_dists = self._propose(
                batch,
                shifted_corrected,
                verified.hidden,
                num_accepted,
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if len(all_draft_tokens) > 1:
                new_token = ops.stack(all_draft_tokens, axis=-1)
            else:
                new_token = ops.unsqueeze(all_draft_tokens[0], -1)

        if all_draft_dists is not None:
            return (
                num_accepted,
                next_tokens,
                new_token,
                ops.stack(all_draft_dists, axis=1),
            )

        return (num_accepted, next_tokens, new_token)

    def _discard_absent_proposals(
        self,
        batch: SequentialBatch,
        num_accepted: TensorValue,
        recovered: TensorValue,
        bonus: TensorValue,
        next_tokens: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Zeroes the accepted count on rows that carried no real proposal.

        Two rows reach the sampler with nothing to verify: a prefill row, whose
        ``draft_tokens`` is ``[batch, 0]``, and a decode row padded entirely
        with :data:`MAGIC_DRAFT_TOKEN_ID` -- which is what a prefill row riding
        a mixed prefill+decode batch looks like. Reporting the sampler's number
        would inflate the acceptance metric and commit past the tokens the
        iteration actually produced, so the count is forced to zero and the
        committed token re-picked at that index.
        """
        device = batch.device0
        num_steps = batch.num_draft_tokens
        zero_u32 = ops.constant(0, DType.uint32, device=device)
        is_prefill = (
            _shape_to_scalar(num_steps, device, dtype=DType.uint32) == zero_u32
        ).broadcast_to(["batch_size"])

        # The extra MAGIC column makes the reduction well-defined at K == 0.
        magic_token = ops.constant(
            MAGIC_DRAFT_TOKEN_ID, DType.int64, device=device
        )
        padded_drafts = ops.concat(
            [batch.draft_tokens, magic_token.broadcast_to(["batch_size", 1])],
            axis=1,
        )
        num_magic_tokens = ops.squeeze(
            ops.sum((padded_drafts == magic_token).cast(DType.int32), axis=-1),
            axis=-1,
        )
        num_steps_plus_one = _shape_to_scalar(
            num_steps, device, dtype=DType.int32
        ) + ops.constant(1, DType.int32, device=device)
        is_dummy_draft = num_magic_tokens == num_steps_plus_one.broadcast_to(
            ["batch_size"]
        )

        zero_accepted = ops.constant(
            0, num_accepted.dtype, device=device
        ).broadcast_to(["batch_size"])
        num_accepted = ops.where(
            is_prefill | is_dummy_draft, zero_accepted, num_accepted
        )
        next_tokens = ops.gather_nd(
            ops.concat([recovered, bonus], axis=1),
            ops.unsqueeze(num_accepted, axis=-1),
            batch_dims=1,
        )
        return num_accepted, next_tokens

    def _propose(
        self,
        batch: SequentialBatch,
        shifted_corrected: TensorValue,
        target_hidden: _TargetHiddenT,
        num_accepted: TensorValue,
        *,
        seed: TensorValue,
        temperature: TensorValue,
        top_k: TensorValue,
        top_p: TensorValue,
    ) -> tuple[list[TensorValue], list[TensorValue] | None]:
        """Runs the draft ``num_draft_steps`` times, one token per step.

        Step 0 runs over the whole target-corrected sequence and gathers the
        hidden state at each request's accepted position; steps 1..K-1 run one
        token per request and carry that hidden state forward.

        Under ``draft_proposal="sampled"`` each step draws its token from the
        truncated draft distribution instead of taking its argmax, and reports
        the distribution it drew from so the acceptance test can run the
        ``min(1, p/q)`` ratio against the real ``q``.

        Args:
            batch: The merged iteration inputs every phase reads.
            shifted_corrected: Target-corrected tokens from the shift phase.
            target_hidden: Per-device hidden states the target produced.
            num_accepted: Accepted draft-token count per batch element.
            seed: Per-row RNG seed, keyed per step by ``_draft_step_seed``.
            temperature: Per-row temperature the draw is taken at.
            top_k: Per-row top-k cutoff the draw is truncated to.
            top_p: Per-row nucleus cutoff the draw is truncated to.

        Returns:
            One ``[batch_size]`` token tensor per draft step, in step order,
            and the matching ``[batch_size, vocab_size]`` distributions under
            ``draft_proposal="sampled"`` (``None`` otherwise).
        """
        sampled = self._draft_proposal == "sampled"
        with Graph.current.profile_scope("draft_step_0"):
            prefill = self._proposer.prefill(
                batch, shifted_corrected, target_hidden
            )

            draft_logits_3d = _reshape_target_logits(prefill.logits)
            gather_idx = ops.unsqueeze(num_accepted, axis=-1)
            all_draft_dists: list[TensorValue] | None = None
            if sampled:
                assert self._vocab_size is not None
                # Gather the accepted row before sampling, so only [batch,
                # vocab] reaches the kernel rather than all batch * (K+1) rows.
                # Token and distribution come out of one call, so they cannot
                # disagree.
                accepted_logits = ops.gather_nd(
                    draft_logits_3d, gather_idx, batch_dims=1
                ).rebind(["batch_size", self._vocab_size])
                next_draft_tokens, next_draft_dist = (
                    topk_fused_sampling_with_dist(
                        accepted_logits,
                        top_k=top_k,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed,
                    )
                )
                next_draft_tokens = next_draft_tokens.reshape([-1])
                all_draft_dists = [
                    ops.rebind(
                        next_draft_dist, ["batch_size", self._vocab_size]
                    )
                ]
            else:
                draft_argmax = ops.squeeze(
                    ops.argmax(draft_logits_3d, axis=-1), axis=-1
                )
                next_draft_tokens = ops.gather_nd(
                    draft_argmax, gather_idx, batch_dims=1
                ).reshape([-1])

            carry_hidden = gather_accepted_hidden_states(
                prefill.hidden,
                merged_offsets=batch.merged_offsets,
                merged_offsets_per_dev=batch.merged_offsets_per_dev,
                num_accepted=num_accepted,
                num_draft_tokens=batch.num_draft_tokens,
                data_parallel_degree=self.data_parallel_degree,
                data_parallel_splits=batch.data_parallel_splits,
                signal_buffers=batch.signal_buffers,
                device=batch.device0,
                split_prefix=self._proposer.split_prefix,
            )

            # Gathered at the same accepted positions as the hidden carry.
            carry_reuse: list[TensorValue] = []
            if (reuse_spec := self._proposer.reuse) is not None:
                carry_reuse = gather_accepted_hidden_states(
                    prefill.reuse,
                    merged_offsets=batch.merged_offsets,
                    merged_offsets_per_dev=batch.merged_offsets_per_dev,
                    num_accepted=num_accepted,
                    num_draft_tokens=batch.num_draft_tokens,
                    data_parallel_degree=self.data_parallel_degree,
                    data_parallel_splits=batch.data_parallel_splits,
                    signal_buffers=batch.signal_buffers,
                    device=batch.device0,
                    split_prefix=reuse_spec.split_prefix,
                )

        input_lengths = ops.rebind(
            (batch.input_row_offsets[1:] - batch.input_row_offsets[:-1]).cast(
                DType.int64
            ),
            ["batch_size"],
        )
        accepted_lengths = (
            input_lengths + num_accepted.cast(DType.int64)
        ).rebind(["batch_size"])

        use_comm = len(self.devices) > 1
        cache_lengths_per_dev = increment_cache_lengths_from_counts(
            accepted_lengths,
            batch.data_parallel_splits,
            [kv.cache_lengths for kv in batch.draft_kv_collections],
            batch.signal_buffers if use_comm else None,
        )

        draft_return_n_logits = ops.constant(
            1, DType.int64, DeviceRef.CPU()
        ).broadcast_to([1])

        decode_offsets = ops.range(
            start=0,
            stop=batch.input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=batch.device0,
            dtype=DType.uint32,
        )
        # Broadcast once so the draft can skip its own broadcast for every
        # step of the multi-step loop.
        decode_offsets_per_dev = ops.distributed_broadcast(
            decode_offsets, batch.signal_buffers
        )
        host_decode_offsets = ops.range(
            start=0,
            stop=batch.input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=DeviceRef.CPU(),
            dtype=DType.uint32,
        )

        decode_kv = self._apply_decode_swaps(batch.draft_kv_collections)
        unknown = set(self._proposer.passthrough_decode_swaps) - set(
            batch.passthrough_kv
        )
        if unknown:
            raise ValueError(
                f"passthrough_decode_swaps names {sorted(unknown)}, which "
                "the model did not pass in passthrough_kv "
                f"(got {sorted(batch.passthrough_kv)})"
            )
        decode_passthrough_kv = {
            name: (
                self._apply_decode_swaps(leaf)
                if name in self._proposer.passthrough_decode_swaps
                else leaf
            )
            for name, leaf in batch.passthrough_kv.items()
        }

        next_draft_tokens = next_draft_tokens.rebind(["batch_size"])
        all_draft_tokens = [next_draft_tokens]

        draft_input = DraftStepInput(
            tokens=next_draft_tokens, hidden=carry_hidden, reuse=carry_reuse
        )

        step_batch = replace(
            batch,
            return_n_logits=draft_return_n_logits,
            passthrough_kv=decode_passthrough_kv,
            query_offsets_per_dev=decode_offsets_per_dev,
            host_query_offsets=host_decode_offsets,
        )

        batch_context_lengths = batch.batch_context_lengths
        for index in range(1, self.num_draft_steps):
            with Graph.current.profile_scope(f"draft_step_{index}"):
                draft_input = self._rebind_draft_input(draft_input, index)
                step_kv: list[PagedCacheValues] = (
                    [
                        replace(kv, cache_lengths=cl)
                        for kv, cl in zip(
                            decode_kv, cache_lengths_per_dev, strict=True
                        )
                    ]
                    if self._proposer.draft_cache is DraftCache.OWN
                    else decode_kv
                )
                step_batch = replace(
                    step_batch,
                    draft_kv_collections=step_kv,
                    draft_cache_lengths=cache_lengths_per_dev,
                    batch_context_lengths=batch_context_lengths,
                )

                proposed = self._proposer.step(step_batch, draft_input, index)

                if sampled:
                    assert self._vocab_size is not None
                    assert all_draft_dists is not None
                    # A step's logits carry a different symbolic row identity
                    # than "batch_size"; rebind to line up with the per-row
                    # sampling params.
                    step_tokens, step_dist = topk_fused_sampling_with_dist(
                        proposed.logits.rebind(
                            ["batch_size", self._vocab_size]
                        ),
                        top_k=top_k,
                        temperature=temperature,
                        top_p=top_p,
                        seed=_draft_step_seed(seed, index),
                    )
                    step_token_ids = step_tokens.reshape([-1])
                    all_draft_dists.append(
                        ops.rebind(step_dist, ["batch_size", self._vocab_size])
                    )
                else:
                    step_token_ids = ops.argmax(
                        proposed.logits, axis=-1
                    ).reshape([-1])

                # Name the row dim before the tokens are reused: the next step
                # embeds them and concatenates against the hidden carry, one
                # row per request.
                next_draft_tokens = ops.rebind(step_token_ids, ["batch_size"])
                all_draft_tokens.append(next_draft_tokens)
                draft_input = DraftStepInput(
                    tokens=next_draft_tokens,
                    hidden=self._slice_step_hidden(
                        proposed.hidden, index + 1, batch.data_parallel_splits
                    ),
                    reuse=draft_input.reuse,
                )

                if self._proposer.draft_cache is DraftCache.OWN:
                    cache_lengths_per_dev = [
                        cl + 1 for cl in cache_lengths_per_dev
                    ]
                batch_context_lengths = [
                    bcl + 1 for bcl in batch_context_lengths
                ]
        return all_draft_tokens, all_draft_dists

    def _apply_decode_swaps(
        self, draft_kv_collections: list[PagedCacheValues]
    ) -> list[PagedCacheValues]:
        """Retarget the draft caches to a ``q = 1`` dispatch for the loop."""
        swaps = self._proposer.decode_swaps
        if not swaps:
            return list(draft_kv_collections)

        one = ops.constant(1, DType.uint32, DeviceRef.CPU()).broadcast_to([1])

        def swapped(kv: PagedCacheValues) -> PagedCacheValues:
            change: dict[str, Any] = {}
            if DecodeKVSwap.MAX_PROMPT_LENGTH_ONE in swaps:
                change["max_prompt_length"] = one
            if DecodeKVSwap.DRAFT_ATTENTION_DISPATCH_METADATA in swaps:
                change["attention_dispatch_metadata"] = (
                    kv.draft_attention_dispatch_metadata
                )
            if DecodeKVSwap.DRAFT_MLA_NUM_PARTITIONS in swaps:
                change["mla_num_partitions"] = kv.draft_mla_num_partitions
            return replace(kv, **change)

        return [swapped(kv) for kv in draft_kv_collections]

    def _rebind_draft_input(
        self, draft_input: DraftStepInput, index: int
    ) -> DraftStepInput:
        """Name each field's per-device batch dim for this step.

        Per-device shapes differ across DP replicas, so the dim name has to
        carry the device index; the draft rebinds it once inside ``__call__``.
        """
        hidden_dim = self._proposer.hidden_dim
        reuse = self._proposer.reuse
        return DraftStepInput(
            tokens=draft_input.tokens,
            hidden=[
                draft_input.hidden[i].rebind(
                    [self._carry_dim(index, i), hidden_dim]
                )
                for i in range(len(self.devices))
            ],
            # Shares the hidden carry's batch dim: one row per request, taken
            # at the same accepted position.
            reuse=[
                draft_input.reuse[i].rebind(
                    [self._carry_dim(index, i), reuse.dim]
                )
                for i in range(len(self.devices))
            ]
            if reuse is not None
            else [],
        )

    def _carry_dim(self, index: int, device: int) -> str:
        """Name the carry's batch dim for one step on one device."""
        names = self._proposer.carry_dim_names
        if not names.per_device:
            return f"{names.prefix}{index}_batch"
        return f"{names.prefix}{index}_batch_dev_{device}"

    def _slice_step_hidden(
        self,
        hidden: list[TensorValue],
        next_index: int,
        splits: TensorValue,
    ) -> list[TensorValue]:
        """Take each device's own rows out of a step's hidden output.

        ``LAST_PER_DEVICE`` returns post-allgather full-batch tensors, so under
        DP each device must slice its own shard back out. ``ALL`` already
        returns per-device tensors and must not be sliced. Deciding this from
        the declared mode is what keeps the rule and the slice from drifting
        apart -- they sit 40 lines apart in the hand-written modules.

        Under mixed TP+DP the split index is the *replica*, not the device:
        ``tp_degree`` devices share one replica's rows. The hand-written MTP
        loop indexed by device, which disagrees with
        :func:`gather_accepted_hidden_states` feeding the same loop.
        """
        needs_slice = (
            self._proposer.step_hidden_mode
            == ReturnHiddenStates.LAST_PER_DEVICE
            and self.data_parallel_degree > 1
        )
        if not needs_slice:
            # TP / single-device: each device already holds a full replica.
            return list(hidden)

        tp_degree = len(self.devices) // self.data_parallel_degree
        return [
            ops.slice_tensor(
                hidden[i],
                [
                    (
                        slice(
                            splits[i // tp_degree],
                            splits[i // tp_degree + 1],
                        ),
                        self._carry_dim(next_index, i),
                    ),
                ],
            )
            for i in range(len(self.devices))
        ]

    @override
    @property
    def input_spec(self) -> SpecDecodeInputTypeSpec:
        return self._input_spec

    @override
    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return self._target.ep_input_types()
