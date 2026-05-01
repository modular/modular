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
from dataclasses import dataclass, replace
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
from max.nn import ReturnHiddenStates, ReturnLogits

# TODO: rename the kernel at the source
from max.nn.kernels import eagle_prefill_shift_tokens
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import (
    AcceptanceSampler,
    _reshape_target_logits,
)
from max.pipelines.lib.speculative_decoding.ragged_token_merger import (
    RaggedTokenMerger,
    shape_to_scalar,
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
    seed: TensorValue
    temperature: TensorValue
    top_k: TensorValue
    max_k: TensorValue
    top_p: TensorValue
    min_top_p: TensorValue
    token_bitmasks: TensorValue | None = None
    """Grammar constraint bitmask for structured output.

    Shape: [batch_size, num_speculative_tokens + 1, vocab_size].
    Position i contains valid token mask given FSM state after consuming
    draft[0:i-1]. Position num_speculative_tokens is for the bonus token.
    None when structured output is not enabled (graph compiled without bitmask).
    """


class UnifiedEagleLlama3(Module):
    """Fused nn.Module: merge + target forward + greedy rejection + shift + draft."""

    def __init__(self, config: UnifiedEagleLlama3Config) -> None:
        super().__init__()

        self.config = config
        self.num_draft_steps = config.speculative_config.num_speculative_tokens
        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=config.speculative_config.synthetic_acceptance_rate,
            num_draft_steps=self.num_draft_steps,
            use_stochastic=True,
        )
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
        # Use an iterator to consume inputs sequentially, avoiding a hardcoded
        # count that would break if the kv cache structure changes.
        it = iter(inputs)

        tokens = next(it)
        input_row_offsets = next(it)
        return_n_logits = next(it)
        # target model kvcache inputs
        target_kv_blocks = next(it)
        cache_lengths = next(it)
        lookup_table = next(it)
        max_lengths = next(it)
        dispatch_metadata = next(it)
        draft_dispatch_metadata = next(it)
        # draft model inputs
        draft_tokens = next(it)
        # draft kvcache
        draft_kv_blocks = next(it)
        # stochastic acceptance seed (scalar int64 on CPU)
        seed = next(it)
        # sampling params for stochastic acceptance
        temperature = next(it)
        top_k = next(it)
        max_k = next(it)
        top_p = next(it)
        min_top_p = next(it)
        # Optional structured output bitmask (appended when enabled)
        token_bitmask = (
            next(it) if self.config.enable_structured_output else None
        )

        target_kv_collection = PagedCacheValues(
            kv_blocks=target_kv_blocks.buffer,
            cache_lengths=cache_lengths.tensor,
            lookup_table=lookup_table.tensor,
            max_lengths=max_lengths.tensor,
            attention_dispatch_metadata=dispatch_metadata.tensor,
            draft_attention_dispatch_metadata=draft_dispatch_metadata.tensor,
        )

        return UnifiedEagleLlama3Values(
            tokens=tokens.tensor,
            input_row_offsets=input_row_offsets.tensor,
            draft_tokens=draft_tokens.tensor,
            return_n_logits=return_n_logits.tensor,
            kv_collection=target_kv_collection,
            draft_kv_blocks=draft_kv_blocks.buffer,
            seed=seed.tensor,
            temperature=temperature.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
            token_bitmasks=token_bitmask.tensor if token_bitmask else None,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Input types for the unified graph.

        Order: tokens, input_row_offsets, return_n_logits, target_kv_cache,
               draft_tokens, draft_kv_blocks, seed, temperature, top_k,
               max_k, top_p, min_top_p[, token_bitmasks].

        The token_bitmasks input is only included when structured output is
        enabled via config.enable_structured_output.
        """
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
        assert len(target_kv_inputs.inputs) == 1
        target_kv_flat = list(target_kv_inputs.inputs[0].flatten())

        draft_kv_inputs = self.config.draft.kv_params.get_symbolic_inputs()
        assert len(draft_kv_inputs.inputs) == 1
        draft_kv_blocks = draft_kv_inputs.inputs[0].kv_blocks

        temperature_type = TensorType(
            DType.float32, shape=["batch_size"], device=device_ref
        )
        top_k_type = TensorType(
            DType.int64, shape=["batch_size"], device=device_ref
        )
        max_k_type = TensorType(DType.int64, shape=[], device=DeviceRef.CPU())
        top_p_type = TensorType(
            DType.float32, shape=["batch_size"], device=device_ref
        )
        min_top_p_type = TensorType(
            DType.float32, shape=[], device=DeviceRef.CPU()
        )

        # Mandatory inputs (must match order in _unflatten_graph_inputs)
        result: tuple[TensorType | BufferType, ...] = (
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            *target_kv_flat,
            draft_tokens_type,
            draft_kv_blocks,
            ops.random.SeedType,
            temperature_type,
            top_k_type,
            max_k_type,
            top_p_type,
            min_top_p_type,
        )

        # Optional bitmask input for structured output
        if self.config.enable_structured_output:
            # num_bitmask_positions = num_speculative_tokens + 1
            # Position i contains valid tokens given FSM state after draft[0:i-1]
            # Position num_speculative_tokens is for the bonus token
            token_bitmasks_type = TensorType(
                DType.bool,
                shape=["batch_size", "num_bitmask_positions", "vocab_size"],
                device=device_ref,
            )
            result = result + (token_bitmasks_type,)

        return result

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
        # logits       : [B*(K+1), V] (K+1 logits per request)
        # hidden_states: [S+B*K, H]. ``extract_hs`` flattens per-device
        # hs into positional tuple elements; single-device here so the
        # hs is at index 3.
        logits = target_outputs[1]
        hidden_states = target_outputs[3]

        hidden_dim = hidden_states.shape[1]

        # num_accepted_draft_tokens: [B]     (index of first rejected step, 0..K)
        # recovered                : [B, K]  (target argmax at each draft position)
        # bonus                    : [B, 1]  (target argmax at the +1 position)
        num_accepted_draft_tokens, recovered, bonus = self.acceptance_sampler(
            draft_tokens,
            logits,
            seed=inputs.seed,
            temperature=inputs.temperature,
            top_k=inputs.top_k,
            max_k=inputs.max_k,
            top_p=inputs.top_p,
            min_top_p=inputs.min_top_p,
            token_bitmasks=inputs.token_bitmasks,
        )

        # target_tokens: [B, K+1]
        target_tokens = ops.concat([recovered, bonus], axis=1)
        next_tokens = ops.gather_nd(
            target_tokens,
            ops.unsqueeze(num_accepted_draft_tokens, axis=-1),
            batch_dims=1,
        )

        num_draft_sentinel_gpu = shape_to_scalar(draft_tokens.shape[1], device)

        # Build corrected merged tokens: replace draft tokens with target
        # argmax (recovered). For accepted positions draft == target argmax,
        # so only rejected positions actually change.
        corrected_merged, corrected_offsets = self.merger(
            tokens, input_row_offsets, recovered
        )
        corrected_merged = corrected_merged.rebind(["merged_seq_len"])
        corrected_offsets = corrected_offsets.rebind(["input_row_offsets_len"])

        # shifted_corrected: [S+B*K]
        shifted_corrected = eagle_prefill_shift_tokens(
            corrected_merged,
            corrected_offsets,
            bonus.reshape((-1,)),
        )

        # draft_kv_collection is same as the target's kv_collection other than
        # the kv_blocks.
        draft_kv_collection = PagedCacheValues(
            kv_blocks=draft_kv_blocks,
            cache_lengths=kv_collection.cache_lengths,
            lookup_table=kv_collection.lookup_table,
            max_lengths=kv_collection.max_lengths,
            attention_dispatch_metadata=kv_collection.attention_dispatch_metadata,
            draft_attention_dispatch_metadata=kv_collection.draft_attention_dispatch_metadata,
        )

        # --- Draft step 0 ---
        # Hack the return_hidden_states, return_logits and reset it to match
        # the target model.
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        draft_outputs = self.draft(
            shifted_corrected,  # [S+B*K]
            draft_kv_collection,
            return_n_logits,
            merged_offsets,  # [B+1]
            hidden_states,  # [S+B*K, H]
        )
        self.draft.return_hidden_states = ReturnHiddenStates.LAST
        self.draft.return_logits = ReturnLogits.LAST_TOKEN

        # logits       : [B*(K+1), V] (K+1 logits per request)
        # hidden_states: [S+B*K, H] (single-device, single TensorValue).
        logits = draft_outputs[1]
        hs = draft_outputs[3]

        last_idx = merged_offsets[1:] - 1
        last_accepted_idx = (
            ops.rebind(last_idx, ["batch_size"])
            - num_draft_sentinel_gpu.broadcast_to(["batch_size"])
            + num_accepted_draft_tokens
        )
        draft_hs = ops.gather(hs, last_accepted_idx, axis=0)

        # Sample the first draft token
        logits = _reshape_target_logits(logits)
        # tokens: [B, K+1]
        tokens = ops.squeeze(ops.argmax(logits, axis=-1), axis=-1)

        next_draft_tokens = ops.gather_nd(
            tokens,
            ops.unsqueeze(num_accepted_draft_tokens, axis=-1),
            batch_dims=1,
        )

        # Compute the new kv cache collection
        prev_cache_lengths = ops.rebind(
            draft_kv_collection.cache_lengths, ["batch_size"]
        )
        input_lengths = input_row_offsets[1:] - input_row_offsets[:-1]
        cache_lengths = (
            prev_cache_lengths
            + ops.rebind(input_lengths, ["batch_size"])
            + num_accepted_draft_tokens.cast(DType.uint32)
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
        orig_max_cache_length = (
            draft_kv_collection.max_lengths[0, 1]
            .cast(DType.uint32)
            .broadcast_to([1])
        )
        max_cache_length = orig_max_cache_length + 1

        # draft_return_n_logits: [1] (CPU)
        draft_return_n_logits = ops.constant(
            1, DType.int64, DeviceRef.CPU()
        ).broadcast_to([1])

        max_lengths = ops.concat(
            [one, draft_kv_collection.max_lengths[0, 1].broadcast_to([1])],
            axis=-1,
        ).reshape([1, 2])

        draft_kv_collection = replace(
            draft_kv_collection,
            max_lengths=max_lengths,
            attention_dispatch_metadata=draft_kv_collection.draft_attention_dispatch_metadata,
        )

        # --- Draft steps 1..N-1 ---
        all_draft_tokens = [next_draft_tokens]
        for _ in range(1, self.num_draft_steps):
            next_draft_tokens = next_draft_tokens.rebind(["batch_size"])
            draft_hs = draft_hs.rebind(["batch_size", hidden_dim])

            draft_kv_collection = replace(
                draft_kv_collection, cache_lengths=cache_lengths
            )

            draft_outputs = self.draft(
                next_draft_tokens,
                draft_kv_collection,
                draft_return_n_logits,
                input_row_offsets,
                draft_hs,
            )
            logits = draft_outputs[0]
            draft_hs = draft_outputs[1]

            next_draft_tokens = ops.argmax(logits, axis=-1).reshape([-1])

            # Store the new tokens for this step
            all_draft_tokens.append(
                ops.rebind(next_draft_tokens, ["batch_size"])
            )

            # Increment cache length for the next step
            cache_lengths = cache_lengths + 1
            max_cache_length = max_cache_length + 1

        # next_draft_tokens: [B, num_draft_steps]
        if len(all_draft_tokens) > 1:
            next_draft_tokens = ops.stack(all_draft_tokens, axis=-1)
        else:
            next_draft_tokens = ops.unsqueeze(all_draft_tokens[0], -1)

        return (
            num_accepted_draft_tokens,  # [B]
            next_tokens,  # [B]
            next_draft_tokens,  # [B, num_draft_steps]
        )
