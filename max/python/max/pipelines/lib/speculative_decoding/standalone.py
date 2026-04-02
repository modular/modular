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
"""Speculative decoding pipelines with factory function and implementations."""

from __future__ import annotations

import logging
from typing import final

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.interfaces import RequestID, TextGenerationInputs, TextGenerationOutput
from max.pipelines.core import TextContext
from max.pipelines.lib.interfaces import ModelInputs, PipelineModel
from max.profiler import traced

from ..sampling import SamplerInputs, apply_logits_processors
from .base import SpeculativeDecodingPipelineBase
from .utils import (
    ModelInputsWithTokensAndOffsets,
    build_response,
    compute_max_num_draft_steps,
    update_contexts_and_compute_metrics_standalone,
)

logger = logging.getLogger("max.pipelines")


@final
class StandaloneSpeculativeDecodingPipeline(SpeculativeDecodingPipelineBase):
    """Standalone speculative decoding where draft model runs independently.

    In this approach, the draft model generates tokens without any information
    from the target model, then the target model verifies these tokens.
    """

    @traced
    def prepare_batch(
        self,
        model: PipelineModel[TextContext],
        batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        return_n_logits: int,
        is_draft: bool = False,
        draft_inputs: ModelInputs | None = None,
        merged_draft_tokens: Buffer | None = None,
        merged_draft_offsets: Buffer | None = None,
    ) -> tuple[ModelInputs, int]:
        """Prepares batch inputs and KV cache for draft or target model."""
        kv_manager = self._target_kv_manager

        if is_draft:
            num_steps = compute_max_num_draft_steps(
                replica_batches,
                desired_num_draft_steps=self._num_draft_steps,
                max_seq_len=self._max_seq_len,
                is_draft=True,
            )
        else:
            num_steps = 1

        kv_cache_inputs = kv_manager.runtime_inputs([batch], num_steps)
        if is_draft:
            # Huge hack alert!
            # Swap out the target kv cache buffers for the draft kv cache buffers
            for replica_input, draft_blocks in zip(
                kv_cache_inputs.inputs, self._draft_kv_buffers, strict=True
            ):
                replica_input.blocks = draft_blocks
            return (
                model.prepare_initial_token_inputs(
                    replica_batches=replica_batches,
                    kv_cache_inputs=kv_cache_inputs,
                    return_n_logits=return_n_logits,
                ),
                num_steps,
            )
        else:
            assert merged_draft_tokens is not None
            assert merged_draft_offsets is not None
            assert draft_inputs is not None
            draft_inputs.update(
                tokens=merged_draft_tokens,
                input_row_offsets=merged_draft_offsets,
                signal_buffers=getattr(
                    self._target_model, "signal_buffers", []
                ),
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=Buffer.from_numpy(
                    np.array([return_n_logits], dtype=np.int64)
                ),
            )
            return (draft_inputs, num_steps)

    @traced
    def generate_draft_tokens(
        self,
        batch: list[TextContext],
        num_steps: int,
        model_inputs: ModelInputs,
    ) -> tuple[int, Buffer, Buffer, ModelInputs, Buffer | None]:
        """Generates draft tokens for the batch using the draft model."""
        # Create sampling parameters once for the entire batch
        sampler_inputs = SamplerInputs.create(batch, self.devices[0])

        # Generate tensor for generated tokens.
        generated_tokens = Buffer.zeros(
            (len(batch), 0), dtype=DType.int64, device=self.devices[0]
        )

        generated_logits = Buffer.zeros(
            (len(batch), 0), dtype=DType.float32, device=self.devices[0]
        )

        # Multi-step execution
        curr_step_inputs = model_inputs

        all_draft_logits = (
            Buffer.zeros(
                (num_steps, len(batch), self.vocab_size),
                dtype=DType.float32,
                device=self.devices[0],
            )
            if self._needs_all_draft_logits
            else None
        )

        for i in range(num_steps):
            # Execute the model and get next tokens.
            model_outputs = self._draft_model.execute(
                model_inputs=curr_step_inputs
            )

            if all_draft_logits is not None:
                all_draft_logits[i, :, :].inplace_copy_from(
                    model_outputs.logits
                )

            # Sample next_token
            new_tokens, new_generated_tokens, new_generated_logits = (
                self._sampler.sample_logits_with_prev(
                    logits=model_outputs.logits,
                    prev_tokens=generated_tokens,
                    prev_logits=generated_logits,
                    sampler_inputs=sampler_inputs,
                )
            )
            generated_tokens = new_generated_tokens
            generated_logits = new_generated_logits

            # Increment cache lengths.
            assert curr_step_inputs.kv_cache_inputs is not None
            curr_step_inputs.kv_cache_inputs = (
                self._increment_cache_lengths_processor.execute(
                    curr_step_inputs.kv_cache_inputs,
                    curr_step_inputs,
                )
            )

            # Prepare next token inputs.
            curr_step_inputs = self._draft_model.prepare_next_token_inputs(
                new_tokens, curr_step_inputs
            )

        return (
            num_steps,
            generated_tokens,
            generated_logits,
            model_inputs,
            all_draft_logits,
        )

    @traced
    def verify_draft_tokens_with_target_model(
        self,
        draft_inputs: ModelInputs,
        context_batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        num_draft_tokens_generated: int,
        draft_tokens: Buffer,
        draft_logits: Buffer,
        merged_draft_tokens: Buffer,
        merged_draft_offsets: Buffer,
        all_draft_logits: Buffer | None,
    ) -> tuple[Buffer, Buffer, Buffer | None]:
        """Verifies draft tokens against the target model and returns merged outputs."""
        # The kv cache manager for the target model uses these indices to set the lengths of the cache. We bump them manually here even though the tokens array has not been filled. They are reset when doing the final update of the contexts after both draft and target models have run.
        target_inputs, target_num_steps = self.prepare_batch(
            self._target_model,
            context_batch,
            replica_batches,
            draft_inputs=draft_inputs,
            return_n_logits=num_draft_tokens_generated + 1,
            is_draft=False,
            merged_draft_tokens=merged_draft_tokens,
            merged_draft_offsets=merged_draft_offsets,
        )
        assert target_num_steps == 1

        # Generate target tokens.
        target_outputs = self._target_model.execute(model_inputs=target_inputs)

        # Apply logits processors
        apply_logits_processors(
            context_batch=context_batch,
            batch_logits=target_outputs.logits,
            batch_logit_offsets=target_outputs.logit_offsets,
        )
        # Generate Final Samples
        assert target_outputs.logit_offsets is not None
        first_rejected_tokens, recovered_tokens, bonus_tokens = (
            self._rejection_runner.run(
                draft_tokens=draft_tokens,
                draft_logits=draft_logits,
                target_logits=target_outputs.logits,
                target_logit_offsets=target_outputs.logit_offsets,
                all_draft_logits=all_draft_logits,
                context_batch=context_batch,
            )
        )

        return first_rejected_tokens, recovered_tokens, bonus_tokens

    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        """Execute standalone speculative decoding.

        In standalone mode:
        1. Draft model generates tokens independently
        2. Target model verifies draft tokens
        3. Apply rejection sampling to accept/reject tokens
        """
        # Flatten batch and build replica batches for data parallelism
        context_batch = inputs.flat_batch
        replica_batches = inputs.batches

        # Allocate enough kv for 2 * num_draft_steps
        # This ensures we have enough to verify upwards of num_draft_steps tokens
        # and then generate num_draft_steps more tokens.
        # TODO: move this logic to the scheduler
        for context in context_batch:
            for replica_idx in range(len(replica_batches)):
                self._target_kv_manager.alloc(
                    context,
                    replica_idx=replica_idx,
                    num_steps=2 * self._num_draft_steps,
                )

        draft_inputs, draft_num_steps = self.prepare_batch(
            self._draft_model,
            context_batch,
            replica_batches,
            return_n_logits=1,
            is_draft=True,
        )
        (
            num_draft_tokens_generated,
            draft_tokens,
            draft_logits,
            model_inputs,
            all_draft_logits,
        ) = self.generate_draft_tokens(
            context_batch, draft_num_steps, draft_inputs
        )

        # Merge draft tokens with target tokens
        assert isinstance(model_inputs, ModelInputsWithTokensAndOffsets)
        merged_tokens, merged_offsets = self._ragged_token_merger.run(
            tokens=model_inputs.tokens,
            input_row_offsets=model_inputs.input_row_offsets,
            draft_tokens=draft_tokens,
        )

        assert isinstance(merged_tokens, Buffer)
        assert isinstance(merged_offsets, Buffer)
        # Verify draft tokens with target model
        first_rejected_tokens, recovered_tokens, bonus_tokens = (
            self.verify_draft_tokens_with_target_model(
                draft_inputs,
                context_batch,
                replica_batches,
                num_draft_tokens_generated,
                draft_tokens,
                draft_logits,
                merged_tokens,
                merged_offsets,
                all_draft_logits,
            )
        )

        draft_tokens_accepted, draft_tokens_generated = (
            update_contexts_and_compute_metrics_standalone(
                context_batch=context_batch,
                first_rejected_tokens=first_rejected_tokens.to_numpy(),
                recovered_tokens=recovered_tokens.to_numpy(),
                bonus_tokens=(
                    bonus_tokens.to_numpy()
                    if bonus_tokens is not None
                    else None
                ),
                draft_tokens=draft_tokens.to_numpy(),
                num_draft_tokens_generated=num_draft_tokens_generated,
            )
        )
        self.metrics.update(
            draft_tokens_accepted=draft_tokens_accepted,
            draft_tokens_generated=draft_tokens_generated,
        )

        res = build_response(
            context_batch=context_batch, max_seq_len=self._max_seq_len
        )

        # Maybe commit blocks into prefix cache
        self._target_kv_manager.step([context_batch])

        return res
