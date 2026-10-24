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
"""The sequential proposer of DeepSeek-family drafts.

A DeepSeek-family draft -- the MTP NextN head a DeepseekV3 checkpoint ships,
or an MLA or MHA Eagle3 head -- takes one positional call, which
:class:`MLAProposer` drives. The pairings differ only in the class
attributes at its top, so each is a subclass setting them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from max.dtype import DType
from max.graph import DimLike, TensorValue, Value, ops
from max.nn.kv_cache import PagedCacheValues
from max.nn.transformer import DecoderOutputs, ReturnHiddenStates, ReturnLogits

from .driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
)

__all__ = [
    "Eagle3MHAProposer",
    "Eagle3MLAProposer",
    "MLADraft",
    "MLAProposer",
    "MTPProposer",
    "TargetHiddenT",
]

TargetHiddenT = TypeVar("TargetHiddenT")
"""The hidden payload a target hands its draft; opaque to the proposer."""


class DraftConfig(Protocol):
    """The only config field the proposer reads off its draft.

    An MHA draft over an MLA target carries its own config type rather than
    a :class:`DeepseekV3Config`, so this names the field instead of the class.
    """

    @property
    def hidden_size(self) -> int: ...


@runtime_checkable
class MLADraft(Protocol):
    """The draft-side surface :class:`DeepseekV3MLAProposer` drives."""

    return_hidden_states: ReturnHiddenStates
    return_logits: ReturnLogits

    @property
    def config(self) -> DraftConfig:
        """The draft's config."""
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> DecoderOutputs:
        """Runs the draft on one step's inputs."""
        ...


class MLAProposer(Generic[TargetHiddenT]):
    """Shared body of the DeepSeek-family sequential drafts.

    Their drafts take one positional call, the MTP NextN head's and the MLA
    and MHA Eagle3 heads' alike. Generic in the target's hidden payload,
    which the draft takes as an opaque argument: a flat per-device list for
    an MTP head, one list of captured layers per device for an Eagle3 head.
    """

    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
        DecodeKVSwap.DRAFT_ATTENTION_DISPATCH_METADATA,
        DecodeKVSwap.DRAFT_MLA_NUM_PARTITIONS,
    )
    passthrough_decode_swaps: tuple[str, ...] = ()
    draft_cache: DraftCache = DraftCache.OWN
    reuse: ReuseSpec | None = None
    split_prefix: str
    carry_dim_names: CarryDimNames
    step_hidden_mode: ReturnHiddenStates
    uses_thinking_phase: bool = True
    supports_zero_draft_rows: bool = False
    draft_takes_ep_inputs: bool = False
    """Whether the draft's ``__call__`` accepts the target's EP inputs."""
    takes_distributed_inputs = True

    def __init__(self, draft: MLADraft) -> None:
        self.draft = draft
        self.hidden_dim: DimLike = draft.config.hidden_size

    def _ep(
        self, batch: SequentialBatch
    ) -> tuple[list[Value[Any]] | None, ...]:
        return (batch.ep_inputs,) if self.draft_takes_ep_inputs else ()

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: TargetHiddenT,
    ) -> Proposed:
        """Runs the first draft step over the target's hidden states."""
        # Step 0 always uses ALL hidden states (for per-batch-element gather
        # at accepted positions) + VARIABLE logits (for draft argmax).
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        outputs = self.draft(
            tokens,
            target_hidden,
            batch.signal_buffers,  # reuse target signal buffers for draft
            batch.draft_kv_collections,
            batch.return_n_logits,
            batch.query_offsets_per_dev,
            batch.dist.host_query_offsets,
            batch.dist.data_parallel_splits,
            batch.batch_context_lengths,
            *self._ep(batch),
        )
        self.draft.return_hidden_states = self.step_hidden_mode
        self.draft.return_logits = ReturnLogits.LAST_TOKEN
        return Proposed(
            logits=outputs.logits["logits"],
            hidden=list(outputs.hidden_states.tensors())[: batch.n_devs],
        )

    def step(
        self, batch: SequentialBatch, draft_input: DraftStepInput, index: int
    ) -> Proposed:
        """Runs one later draft step on the previous step's output."""
        outputs = self.draft(
            draft_input.tokens,
            draft_input.hidden,
            batch.signal_buffers,
            batch.draft_kv_collections,
            batch.return_n_logits,
            batch.query_offsets_per_dev,
            batch.dist.host_query_offsets,
            batch.dist.data_parallel_splits,
            batch.batch_context_lengths,
            *self._ep(batch),
            split_prefix=f"{self.split_prefix}_draft_step{index}",
        )
        return Proposed(
            logits=outputs.logits["next_token_logits"],
            hidden=list(outputs.hidden_states.tensors())[: batch.n_devs],
        )


class MTPProposer(MLAProposer[list[TensorValue]]):
    """The NextN head a DeepseekV3 checkpoint ships, one token per step."""

    split_prefix = "mtp"
    carry_dim_names = CarryDimNames(prefix="mtp_step", per_device=True)
    # Steps 1..K use LAST_PER_DEVICE. The underlying LAST path's internal
    # allgather acts as a collective fence between successive draft
    # invocations -- without it, the draft's EP/MoE dispatch leaves pending
    # async work that surfaces `mgp.sync` on the 2nd decoder call during CUDA
    # graph capture. LAST_PER_DEVICE returns the post-allgather full-batch
    # tensors, so the driver slices each replica's rows back out under DP.
    step_hidden_mode = ReturnHiddenStates.LAST_PER_DEVICE
    supports_zero_draft_rows = True
    draft_takes_ep_inputs = True


class Eagle3MLAProposer(MLAProposer[list[list[TensorValue]]]):
    """An MLA EAGLE3 head, one token per step.

    The head fuses the target layers it names, which reach it as one list
    of captures per device.
    """

    split_prefix = "eagle3"
    carry_dim_names = CarryDimNames(prefix="draft_step", per_device=True)
    # Steps 1..K run in decode mode (one token per batch element), where
    # ALL-hs == LAST-hs. ALL returns per-device hidden states directly,
    # avoiding the LAST path's allgather, and needs no per-replica slice.
    step_hidden_mode = ReturnHiddenStates.ALL


class Eagle3MHAProposer(MLAProposer[list[list[TensorValue]]]):
    """A Llama-style MHA EAGLE3 head over an MLA target, one token per step.

    An MHA head has no MLA partition count to retarget, and its step-0 pass
    needs a wider dispatch metadata than the cache manager resolved.
    """

    split_prefix = "eagle3_mha"
    carry_dim_names = CarryDimNames(prefix="draft_mha_step", per_device=True)
    step_hidden_mode = ReturnHiddenStates.ALL
    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
        DecodeKVSwap.DRAFT_ATTENTION_DISPATCH_METADATA,
    )

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: list[list[TensorValue]],
    ) -> Proposed:
        """Runs the first draft step with step-0 dispatch metadata."""
        return super().prefill(
            replace(
                batch,
                draft_kv_collections=[
                    _step0_kv(kv) for kv in batch.draft_kv_collections
                ],
            ),
            tokens,
            target_hidden,
        )


def _step0_kv(kv: PagedCacheValues) -> PagedCacheValues:
    """Returns ``kv`` with its dispatch metadata sized for step 0.

    Step 0 runs each row's drafts and its next token.
    """
    decode_md = kv.draft_attention_dispatch_metadata
    assert decode_md is not None
    step0_max_prompt = kv.max_prompt_length.cast(DType.int64).reshape([1]) + 1
    step0_md = ops.concat(
        [decode_md[0:1], step0_max_prompt, decode_md[2:4]],
        axis=0,
    )
    return replace(kv, attention_dispatch_metadata=step0_md)
