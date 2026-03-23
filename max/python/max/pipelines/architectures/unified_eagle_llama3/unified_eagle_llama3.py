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
from max.nn import PagedCacheValues

# TODO: rename the kernel at the source
from max.nn.kernels import (
    compute_mha_decode_num_partitions,
    eagle_prefill_shift_tokens,
)
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

        return (
            tokens_type,
            input_row_offsets_type,
            draft_tokens_type,
            return_n_logits_type,
            *target_kv_flat,
            draft_kv_blocks,
        )

    def __call__(
        self,
        inputs: UnifiedEagleLlama3Values,
    ) -> tuple[TensorValue, ...]:
        # Notation:
        #   B   = batch_size
        #   S   = total_seq_len   (ragged sum of prompt lengths)
        #   K   = num_draft_tokens_to_verify (K is either 0 or num_speculative_tokens)
        #   V   = vocab_size
        #   H   = hidden_size
        #
        # inputs.tokens           : [S]
        # inputs.input_row_offsets: [B+1]
        # inputs.draft_tokens     : [B, K]
        # inputs.return_n_logits  : [1] (CPU)
        tokens = inputs.tokens
        input_row_offsets = inputs.input_row_offsets
        draft_tokens = inputs.draft_tokens
        return_n_logits = inputs.return_n_logits
        kv_collection = inputs.kv_collection
        draft_kv_blocks = inputs.draft_kv_blocks

        device = tokens.device

        # merged_tokens : [S+B*K]
        # merged_offsets: [B+1]
        merged_tokens, merged_offsets = self.merger(
            tokens, input_row_offsets, draft_tokens
        )
        # Rebind to clean symbolic dims so downstream reshapes in
        # Llama3's attention layers can simplify element counts.
        merged_tokens = merged_tokens.rebind(["merged_seq_len"])
        merged_offsets = merged_offsets.rebind(["input_row_offsets_len"])

        # --- Target step ---
        target_outputs = self.target(
            merged_tokens,
            kv_collection,
            return_n_logits,
            merged_offsets,
        )
        # last_logits  : [B, V]
        # logits       : [B*(K+1), V] (K+1 logits per request)
        # logit_offsets: [B+1]
        # hidden_states: [S+B*K, H] (all-token hs)
        last_logits = target_outputs[0]
        logits = target_outputs[1]
        logit_offsets = target_outputs[2]
        hidden_states = target_outputs[3]

        hidden_dim = hidden_states.shape[1]

        # first_rejected: [B]     (index of first rejected step, 0..K)
        # recovered     : [B, K]  (target argmax at each draft position)
        # bonus         : [B, 1]  (target argmax at the +1 position)
        first_rejected, recovered, bonus = greedy_acceptance_sampler(
            inputs.draft_tokens, logits
        )

        # num_draft_sentinel: [1]
        # TODO: make this a graph input so it is compatible with cuda graphs
        num_draft_sentinel_cpu = ops.shape_to_tensor(
            [inputs.draft_tokens.shape[1]]
        ).cast(DType.int64)
        num_draft_sentinel_gpu = num_draft_sentinel_cpu.to(device)

        # K=0           : shift tokens left by 1 per request, append bonus
        # K>0           : passthrough copy.
        # shifted_tokens: [S]
        shifted_tokens = eagle_prefill_shift_tokens(
            inputs.tokens,  # [S]
            inputs.input_row_offsets,  # [B+1]
            bonus.reshape((-1,)),  # [B]
            num_draft_sentinel_gpu,  # [1]
        )

        # K=0              : passthrough (keeps all hidden states)
        # K>0              : keeps positions [0..first_rejected_idx] per request
        # accepted_hs      : [S+B*K, H]
        # accepted_offsets : [B+1]
        accepted_hs, accepted_offsets = eagle_extract_accepted(
            hidden_states,  # [S+B*K, H]
            merged_offsets,  # [B+1]
            first_rejected,  # [B]
            # TODO: make eagle_extract_accepted take in a gpu tensor so that
            # we always end up with the same kernels for cuda graph compatibility
            num_draft_sentinel_cpu,  # [1]
        )

        # Build corrected merged tokens using target predictions instead
        # of original draft tokens. During decode, rejected drafts are
        # replaced with the target's recovered tokens; accepted drafts
        # stay the same (target argmax == draft token for greedy).
        # During prefill (K=0), recovered is empty so this is just inputs.tokens.
        # corrected_merged:  [S+B*K]
        # corrected_offsets: [B+1]
        corrected_merged, corrected_offsets = self.merger(
            inputs.tokens, inputs.input_row_offsets, recovered
        )
        corrected_merged = corrected_merged.rebind(["merged_seq_len"])
        corrected_offsets = corrected_offsets.rebind(["input_row_offsets_len"])

        # Forces K=0 so the kernel always shifts left by 1 and appends bonus.
        zero_sentinel_gpu = ops.constant(0, DType.int64, device).broadcast_to(
            [1]
        )
        # shifted_corrected: [S+B*K]
        shifted_corrected = eagle_prefill_shift_tokens(
            corrected_merged,
            corrected_offsets,
            bonus.reshape((-1,)),
            zero_sentinel_gpu,
        )

        shifted_corrected_2d = ops.unsqueeze(shifted_corrected, -1)
        # draft_input_tokens_2d: [S+B*K, 1]
        draft_input_tokens_2d, _ = eagle_extract_accepted(
            shifted_corrected_2d,
            corrected_offsets,
            first_rejected,
            num_draft_sentinel_cpu,
            zero_fill_rejected=True,
        )
        # draft_input_tokens: [S+B*K]
        draft_input_tokens = draft_input_tokens_2d.reshape([-1])

        # Rebind to common dims so the draft model's concat(embed, hs) works.
        draft_input_tokens = draft_input_tokens.rebind(["merged_seq_len"])
        accepted_hs = accepted_hs.rebind(
            ["merged_seq_len", accepted_hs.shape[1]]
        )
        accepted_offsets = accepted_offsets.rebind(["input_row_offsets_len"])

        # draft_kv_collection is same as the target's kv_collection other than
        # the kv_blocks.
        draft_kv_collection = PagedCacheValues(
            kv_blocks=draft_kv_blocks,
            cache_lengths=kv_collection.cache_lengths,
            lookup_table=kv_collection.lookup_table,
            max_lengths=kv_collection.max_lengths,
            dispatch_metadata=kv_collection.dispatch_metadata,
        )
        # draft_return_n_logits: [1] (CPU)
        draft_return_n_logits = ops.constant(
            1, DType.int64, DeviceRef.CPU()
        ).broadcast_to([1])

        # --- Draft step 0 ---
        # The number of tokens is always [S+B*K] even if some draft tokens are
        # rejected. The suffix tokens are 0s and suffix hs are undefined.
        draft_outputs = self.draft(
            draft_input_tokens,  # [S+B*K]
            draft_kv_collection,
            draft_return_n_logits,
            accepted_offsets,  # [B+1]
            accepted_hs,  # [S+B*K, H]
        )
        # new_tokens: [B, V]
        logits = draft_outputs[0]
        # hs: [B, H]
        draft_hs = draft_outputs[1]

        # Sample the first draft token
        new_tokens = ops.argmax(logits, axis=-1).reshape([-1])

        # Compute the new kv cache collection
        prev_cache_lengths = ops.rebind(
            draft_kv_collection.cache_lengths, ["batch_size"]
        )
        input_lengths = input_row_offsets[1:] - input_row_offsets[:-1]
        cache_lengths = (
            prev_cache_lengths
            + ops.rebind(input_lengths, ["batch_size"])
            - num_draft_sentinel_gpu.broadcast_to(["batch_size"]).cast(
                DType.uint32
            )
            + first_rejected.cast(DType.uint32)
        )

        # Prepare the new input_row_offsets (all reqs have 1 token)
        input_row_offsets = ops.range(
            start=0,
            stop=input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=device,
            dtype=DType.uint32,
        )

        one = ops.constant(1, DType.uint32, DeviceRef.CPU()).broadcast_to([1])

        # Set up the max cache length for the next step.
        # Assume that all tokens are accepted in this calculation.
        # Confusingly max_cache_length != max(cache_lengths). Instead max_cache_length
        # is more like max_total_seq_len including cached and input tokens.
        max_cache_length = (
            draft_kv_collection.max_lengths[0, 1]
            .cast(DType.uint32)
            .broadcast_to([1])
        )
        max_cache_length = max_cache_length + 1

        # Extract values from the original dispatch_metadata for reuse.
        orig_metadata = draft_kv_collection.dispatch_metadata
        assert orig_metadata is not None
        orig_batch_size = orig_metadata.tensor[0]

        n_kv_heads = self.config.draft.kv_params.n_kv_heads

        # --- Draft steps 1..N-1 ---
        all_draft_tokens = [new_tokens]
        for _ in range(1, self.config.num_draft_steps):
            new_tokens = new_tokens.rebind(["batch_size"])
            draft_hs = draft_hs.rebind(["batch_size", hidden_dim])

            max_lengths = ops.concat([one, max_cache_length], axis=-1)

            max_cache_length_i64 = max_cache_length.cast(DType.int64)
            num_partitions = compute_mha_decode_num_partitions(
                orig_batch_size,
                max_cache_length_i64,
                n_kv_heads,
                device,
            )
            metadata_tensor = ops.concat(
                [
                    orig_batch_size.reshape([1]),
                    ops.constant(1, DType.int64, DeviceRef.CPU()).reshape([1]),
                    num_partitions,
                    max_cache_length_i64,
                ],
                axis=0,
            )
            dispatch_metadata = AttentionDispatchMetadata(metadata_tensor)

            kv_collection = PagedCacheValues(
                kv_blocks=draft_kv_blocks,
                cache_lengths=cache_lengths,
                lookup_table=draft_kv_collection.lookup_table,
                max_lengths=max_lengths.broadcast_to([1, 2]),
                dispatch_metadata=dispatch_metadata,
            )

            draft_outputs = self.draft(
                new_tokens,
                kv_collection,
                draft_return_n_logits,
                input_row_offsets,
                draft_hs,
            )
            logits = draft_outputs[0]
            draft_hs = draft_outputs[1]

            new_tokens = ops.argmax(logits, axis=-1).reshape([-1])

            # Store the new tokens for this step
            all_draft_tokens.append(new_tokens)

            # Increment cache length for the next step
            cache_lengths = cache_lengths + 1
            max_cache_length = max_cache_length + 1

        # draft_tokens_stacked: [B, num_draft_steps]
        if len(all_draft_tokens) > 1:
            draft_tokens_stacked = ops.stack(all_draft_tokens, axis=-1)
        else:
            draft_tokens_stacked = ops.unsqueeze(all_draft_tokens[0], -1)

        return (
            last_logits,  # [B, V]
            logits,  # [B*(K+1), V]
            logit_offsets,  # [B+1]
            hidden_states,  # [S+B*K, H]
            first_rejected,  # [B]
            recovered,  # [B, K]
            bonus,  # [B, 1]
            shifted_tokens,  # [S]
            draft_tokens_stacked,  # [B, num_draft_steps]
            draft_hs,  # [B, H]
        )
