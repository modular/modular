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
"""The sequential proposers of a draft that keeps no cache of its own.

Gemma 4's assistant has no K or V projections: each of its layers attends
with its queries alone into the target's own sliding-window or
full-attention cache, so the driver hands it the target's leaves. A ModuleV3
assistant runs through :class:`CrossAttentionModuleV3MTPProposer`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, DimLike, TensorValue, ops
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.legacy.transformer import (
    DecoderOutputs,
    ReturnHiddenStates,
    ReturnLogits,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, DeviceMesh

from .driver import (
    CarryDimNames,
    DecodeKVSwap,
    DraftCache,
    DraftStepInput,
    Proposed,
    ReuseSpec,
    SequentialBatch,
    TensorAdapter,
)

__all__ = [
    "CrossAttentionDraft",
    "CrossAttentionMTPProposer",
    "CrossAttentionModuleV3Draft",
    "CrossAttentionModuleV3MTPProposer",
]

FULL_KV = "full_attention"
"""The target's full-attention leaf, which rides in ``passthrough_kv`` beside
the sliding-window leaf the driver dispatches over."""


@runtime_checkable
class CrossAttentionDraft(Protocol):
    """The draft-side surface :class:`CrossAttentionMTPProposer` drives."""

    return_hidden_states: ReturnHiddenStates
    return_logits: ReturnLogits

    def __call__(self, **kwargs: Any) -> DecoderOutputs:
        """Runs the draft on one step's inputs."""
        ...


def _host_q_max_seq_len(host_offsets: TensorValue) -> TensorValue:
    """Longest per-request query in this call, for the attention kernel."""
    lengths = host_offsets[1:] - host_offsets[:-1]
    return ops.max(lengths, axis=0).cast(DType.uint32).broadcast_to([1])


class CrossAttentionMTPProposer:
    """Drafts one token a step, attending into the target's caches."""

    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
    )
    passthrough_decode_swaps: tuple[str, ...] = (FULL_KV,)
    draft_cache = DraftCache.TARGET
    reuse: ReuseSpec | None = None
    split_prefix = "mtp"
    carry_dim_names = CarryDimNames(prefix="mtp_step")
    step_hidden_mode = ReturnHiddenStates.LAST_PER_DEVICE
    uses_thinking_phase = True
    supports_zero_draft_rows = True
    takes_distributed_inputs = True

    def __init__(self, draft: CrossAttentionDraft, hidden_dim: DimLike) -> None:
        self.draft = draft
        self.hidden_dim = hidden_dim

    def prefill(
        self,
        batch: SequentialBatch,
        tokens: TensorValue,
        target_hidden: list[TensorValue],
    ) -> Proposed:
        """Runs the first draft step over the target's hidden states."""
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        outputs = self.draft(
            tokens=tokens,
            hidden_states=target_hidden,
            signal_buffers=batch.signal_buffers,
            target_sliding_kv=batch.draft_kv_collections,
            target_global_kv=batch.passthrough_kv[FULL_KV],
            return_n_logits=batch.return_n_logits,
            input_row_offsets=batch.query_offsets_per_dev,
            kv_input_row_offsets=batch.merged_offsets_per_dev,
            q_max_seq_len=_host_q_max_seq_len(batch.dist.host_query_offsets),
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
        del index
        outputs = self.draft(
            tokens=draft_input.tokens,
            hidden_states=draft_input.hidden,
            signal_buffers=batch.signal_buffers,
            target_sliding_kv=batch.draft_kv_collections,
            target_global_kv=batch.passthrough_kv[FULL_KV],
            return_n_logits=batch.return_n_logits,
            input_row_offsets=batch.query_offsets_per_dev,
            # The cache still spans the merged verify window, so these stay
            # the merged offsets.
            kv_input_row_offsets=batch.merged_offsets_per_dev,
            q_max_seq_len=ops.constant(
                1, DType.uint32, DeviceRef.CPU()
            ).broadcast_to([1]),
            rope_cache_lengths=batch.draft_cache_lengths,
        )
        return Proposed(
            logits=outputs.logits["next_token_logits"],
            hidden=list(outputs.hidden_states.tensors())[: batch.n_devs],
        )


@runtime_checkable
class CrossAttentionModuleV3Draft(Protocol):
    """The ModuleV3 draft :class:`CrossAttentionModuleV3MTPProposer` drives."""

    mesh: DeviceMesh

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Tensor, ...]:
        """Runs the draft on one step's inputs."""
        ...


def _host_q_max_seq_len_tensor(host_offsets: Tensor) -> Tensor:
    """Longest per-request query in this call, for the attention kernel."""
    lengths = host_offsets[1:] - host_offsets[:-1]
    return F.max(lengths, axis=0).cast(DType.uint32).broadcast_to([1])


class CrossAttentionModuleV3MTPProposer(TensorAdapter):
    """Drafts one token a step on ModuleV3, attending into the target's caches.

    The ModuleV3 counterpart of :class:`CrossAttentionMTPProposer`: the same
    swaps, carries and outputs, but its draft takes ``Tensor`` values and its
    modes as arguments rather than as attributes it holds.
    """

    decode_swaps: tuple[DecodeKVSwap, ...] = (
        DecodeKVSwap.MAX_PROMPT_LENGTH_ONE,
    )
    passthrough_decode_swaps: tuple[str, ...] = (FULL_KV,)
    draft_cache = DraftCache.TARGET
    reuse: ReuseSpec | None = None
    split_prefix = "mtp"
    carry_dim_names = CarryDimNames(prefix="mtp_step")
    step_hidden_mode = ReturnHiddenStates.LAST_PER_DEVICE
    uses_thinking_phase = True
    supports_zero_draft_rows = True
    takes_distributed_inputs = True

    def __init__(
        self, draft: CrossAttentionModuleV3Draft, hidden_dim: DimLike
    ) -> None:
        self.draft = draft
        self.hidden_dim = hidden_dim

    def _target_kv(
        self, batch: SequentialBatch[Tensor, Tensor, Tensor]
    ) -> tuple[PagedCacheValues, PagedCacheValues]:
        """Returns the target's sliding-window and full-attention caches."""
        replicated = DeviceMapping.replicated(self.draft.mesh)
        return (
            PagedCacheValues.from_upstream(
                batch.draft_kv_collections, replicated
            ),
            PagedCacheValues.from_upstream(
                batch.passthrough_kv[FULL_KV], replicated
            ),
        )

    def prefill(
        self,
        batch: SequentialBatch[Tensor, Tensor, Tensor],
        tokens: Tensor,
        target_hidden: list[Tensor],
    ) -> Proposed[Tensor]:
        """Runs the first draft step over the target's hidden states."""
        outputs = self.draft(
            tokens,
            target_hidden,
            *self._target_kv(batch),
            batch.return_n_logits,
            batch.query_offsets_per_dev,
            batch.merged_offsets_per_dev,
            _host_q_max_seq_len_tensor(batch.dist.host_query_offsets),
            None,
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.ALL,
        )
        # (last_logits, logits, offsets, hs_0..hs_{n-1}).
        return Proposed(
            logits=outputs[1].local_shards[0],
            hidden=list(outputs[3 : 3 + len(self.draft.mesh.devices)]),
        )

    def step(
        self,
        batch: SequentialBatch[Tensor, Tensor, Tensor],
        draft_input: DraftStepInput[Tensor],
        index: int,
    ) -> Proposed[Tensor]:
        """Runs one later draft step on the previous step's output."""
        del index
        outputs = self.draft(
            draft_input.tokens,
            draft_input.hidden,
            *self._target_kv(batch),
            batch.return_n_logits,
            batch.query_offsets_per_dev,
            # The cache still spans the merged verify window, so these stay
            # the merged offsets.
            batch.merged_offsets_per_dev,
            F.constant(1, DType.uint32, device=CPU()).broadcast_to([1]),
            batch.draft_cache_lengths,
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=self.step_hidden_mode,
        )
        # (last_logits, hs_0..hs_{n-1}).
        return Proposed(
            logits=outputs[0].local_shards[0],
            hidden=list(outputs[1 : 1 + len(self.draft.mesh.devices)]),
        )
