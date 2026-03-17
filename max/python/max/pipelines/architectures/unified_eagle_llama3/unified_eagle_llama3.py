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
"""Unified EAGLE nn.Module: merge + target forward + rejection + shift + draft."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.kv_cache.paged_kv_cache.increment_cache_lengths import (
    ragged_increment_cache_lengths,
)
from max.nn import PagedCacheValues
from max.nn.kernels import eagle_prefill_shift_tokens

# TODO: rename the kernel at the source
from max.nn.kernels import extract_accepted_hs as eagle_extract_accepted
from max.nn.kv_cache import AttentionDispatchMetadata
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import greedy_acceptance_sampler
from max.pipelines.lib.speculative_decoding.ragged_token_merger import (
    RaggedTokenMerger,
)

from ..eagle_llama3.eagle_llama3 import EagleLlama3
from ..llama3.llama3 import Llama3
from .model_config import UnifiedEagleLlama3Config


@dataclass
class UnifiedEagleLlama3Values:
    tokens: TensorValue
    input_row_offsets: TensorValue
    draft_tokens: TensorValue
    return_n_logits: TensorValue
    kv_collection: PagedCacheValues
    draft_kv_blocks: BufferValue
    draft_cache_lengths: TensorValue


class UnifiedEagleLlama3(Module):
    """Fused nn.Module: merge + target forward + greedy rejection + shift + draft."""

    def __init__(self, config: UnifiedEagleLlama3Config) -> None:
        super().__init__()

        self.config = config
        self.num_devices = 1

        # TODO: support distributed llama3 model
        if len(config.target.devices) != 1:
            raise ValueError("UnifiedEagleLlama3 only supports a single device")

        self.target = Llama3(config.target)
        self.draft = EagleLlama3(config.draft)
        self.merger = RaggedTokenMerger(config.target.devices[0])

    def _unflatten_graph_inputs(
        self,
        inputs: Sequence[Value[Any]],
    ) -> UnifiedEagleLlama3Values:
        (
            tokens,
            input_row_offsets,
            draft_tokens,
            return_n_logits,
            # target model kvcache inputs
            target_kv_blocks,
            cache_lengths,
            lookup_table,
            max_lengths,
            dispatch_metadata,
            # draft model kvcache inputs
            draft_kv_blocks,
            draft_cache_lengths,
        ) = inputs

        target_kv_collection = PagedCacheValues(
            kv_blocks=target_kv_blocks.buffer,
            cache_lengths=cache_lengths.tensor,
            lookup_table=lookup_table.tensor,
            max_lengths=max_lengths.tensor,
            dispatch_metadata=AttentionDispatchMetadata(
                dispatch_metadata.tensor
            ),
        )

        return UnifiedEagleLlama3Values(
            tokens=tokens.tensor,
            input_row_offsets=input_row_offsets.tensor,
            draft_tokens=draft_tokens.tensor,
            return_n_logits=return_n_logits.tensor,
            kv_collection=target_kv_collection,
            draft_kv_blocks=draft_kv_blocks.buffer,
            draft_cache_lengths=draft_cache_lengths.tensor,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        device_ref = self.config.target.devices[0]

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        draft_tokens_type = TensorType(
            DType.int64, ["batch_size", "num_steps"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        target_kv_inputs = self.config.target.kv_params.get_symbolic_inputs()
        assert len(target_kv_inputs) == 1
        target_kv_flat = list(target_kv_inputs[0])

        draft_kv_inputs = self.config.draft.kv_params.get_symbolic_inputs()
        assert len(draft_kv_inputs) == 1
        draft_kv_blocks = draft_kv_inputs[0].kv_blocks
        draft_cache_lengths = draft_kv_inputs[0].cache_lengths

        return (
            tokens_type,
            input_row_offsets_type,
            draft_tokens_type,
            return_n_logits_type,
            *target_kv_flat,
            draft_kv_blocks,
            draft_cache_lengths,
        )

    def __call__(
        self,
        inputs: UnifiedEagleLlama3Values,
    ) -> tuple[TensorValue, ...]:
        draft_kv_collection = PagedCacheValues(
            kv_blocks=inputs.draft_kv_blocks,
            cache_lengths=inputs.draft_cache_lengths,
            lookup_table=inputs.kv_collection.lookup_table,
            max_lengths=inputs.kv_collection.max_lengths,
            dispatch_metadata=inputs.kv_collection.dispatch_metadata,
        )

        merged_tokens, merged_offsets = self.merger(
            inputs.tokens, inputs.input_row_offsets, inputs.draft_tokens
        )
        # Rebind to clean symbolic dims so downstream reshapes in
        # Llama3's attention layers can simplify element counts.
        merged_tokens = merged_tokens.rebind(["merged_seq_len"])
        merged_offsets = merged_offsets.rebind(["merged_offsets_len"])

        target_outputs = self.target(
            merged_tokens,
            inputs.kv_collection,
            inputs.return_n_logits,
            merged_offsets,
        )
        last_logits = target_outputs[0]
        logits = target_outputs[1]
        logit_offsets = target_outputs[2]
        hidden_states = target_outputs[3]

        first_rejected, recovered, bonus = greedy_acceptance_sampler(
            inputs.draft_tokens, logits
        )

        num_draft_sentinel = ops.shape_to_tensor(
            [inputs.draft_tokens.shape[1]]
        ).cast(DType.int64)

        shifted_tokens = eagle_prefill_shift_tokens(
            inputs.tokens,
            inputs.input_row_offsets,
            bonus.reshape((-1,)),
            num_draft_sentinel,
        )
        accepted_hs, accepted_offsets = eagle_extract_accepted(
            hidden_states,
            merged_offsets,
            first_rejected,
            num_draft_sentinel,
        )

        # Build corrected merged tokens using target predictions instead
        # of original draft tokens. During decode, rejected drafts are
        # replaced with the target's recovered tokens; accepted drafts
        # stay the same (target argmax == draft token for greedy).
        # During prefill (K=0), recovered is empty so this is just inputs.tokens.
        corrected_merged, corrected_offsets = self.merger(
            inputs.tokens, inputs.input_row_offsets, recovered
        )
        corrected_merged = corrected_merged.rebind(["corrected_seq_len"])
        corrected_offsets = corrected_offsets.rebind(["corrected_offsets_len"])

        zero_sentinel = ops.constant(
            0, DType.int64, DeviceRef.CPU()
        ).broadcast_to([1])
        shifted_corrected = eagle_prefill_shift_tokens(
            corrected_merged,
            corrected_offsets,
            bonus.reshape((-1,)),
            zero_sentinel,
        )
        shifted_corrected_2d = ops.unsqueeze(shifted_corrected, -1)
        draft_input_tokens_2d, _ = eagle_extract_accepted(
            shifted_corrected_2d,
            corrected_offsets,
            first_rejected,
            num_draft_sentinel,
            zero_fill_rejected=True,
        )
        draft_input_tokens = draft_input_tokens_2d.reshape([-1])

        # Rebind to common dims so the draft model's concat(embed, hs) works.
        draft_input_tokens = draft_input_tokens.rebind(["draft_seq_len"])
        accepted_hs = accepted_hs.rebind(
            ["draft_seq_len", accepted_hs.shape[1]]
        )
        accepted_offsets = accepted_offsets.rebind(["draft_offsets_len"])

        draft_return_n_logits = ops.constant(
            1, DType.int64, DeviceRef.CPU()
        ).broadcast_to([1])

        draft_outputs = self.draft(
            draft_input_tokens,
            draft_kv_collection,
            draft_return_n_logits,
            accepted_offsets,
            accepted_hs,
        )
        draft_logits = draft_outputs[0]
        draft_hs = draft_outputs[1]

        new_token = ops.argmax(draft_logits, axis=-1).reshape([-1])

        _ = self._increment_draft_cache(accepted_offsets, draft_kv_collection)

        return (
            last_logits,
            logits,
            logit_offsets,
            hidden_states,
            first_rejected,
            recovered,
            bonus,
            shifted_tokens,
            new_token,
            draft_hs,
        )

    def _increment_draft_cache(
        self,
        input_row_offsets: TensorValue,
        draft_kv_collection: PagedCacheValues,
    ) -> PagedCacheValues:
        """Increment draft KV cache lengths after a draft forward step.

        Simplified for single GPU: dp=1, no signal buffers.
        """
        # Single GPU: dp=1, data_parallel_splits = [0, batch_size]
        batch_size = ops.shape_to_tensor([input_row_offsets.shape[0]]).cast(
            DType.int64
        ) - ops.constant(1, DType.int64, DeviceRef.CPU()).broadcast_to([1])
        data_parallel_splits = ops.concat(
            [
                ops.constant(0, DType.int64, DeviceRef.CPU()).broadcast_to([1]),
                batch_size,
            ],
            axis=0,
        )

        updated_lengths = ragged_increment_cache_lengths(
            input_row_offsets,
            data_parallel_splits,
            [draft_kv_collection.cache_lengths],
            signal_buffers=None,
        )

        return PagedCacheValues(
            kv_blocks=draft_kv_collection.kv_blocks,
            cache_lengths=updated_lengths[0],
            lookup_table=draft_kv_collection.lookup_table,
            max_lengths=draft_kv_collection.max_lengths[1:, :],
        )
