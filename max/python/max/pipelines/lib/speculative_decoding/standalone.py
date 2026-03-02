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
from collections.abc import Sequence
from typing import TYPE_CHECKING, final

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.interfaces import RequestID, TextGenerationInputs, TextGenerationOutput
from max.nn.kv_cache import KVCacheInputs, KVCacheInputsSequence
from max.pipelines.core import TextContext, reserve_token_space_for_batch
from max.pipelines.lib.interfaces import ModelInputs, PipelineModel
from max.profiler import traced

from ..sampling import apply_logits_processors
from .base import SpeculativeDecodingPipelineBase

if TYPE_CHECKING:
    pass

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
        num_steps: int,
        return_n_logits: int,
        is_draft: bool = False,
        draft_inputs: ModelInputs | None = None,
        merged_draft_tokens: Buffer | None = None,
        merged_draft_offsets: Buffer | None = None,
    ) -> tuple[ModelInputs, int]:
        """Prepares batch inputs and KV cache for draft or target model."""
        kv_manager = (
            self._draft_kv_manager if is_draft else self._target_kv_manager
        )

        # Claim cache rows
        # Build request_id -> replica_idx mapping from replica_batches
        request_to_replica: dict[RequestID, int] = {}
        for r_idx, replica_batch in enumerate(replica_batches):
            for ctx in replica_batch:
                request_to_replica[ctx.request_id] = r_idx

        for i, context in enumerate(batch):  # noqa: B007
            # Calculate num_steps.
            num_steps = self.calculate_num_steps(
                model, model.huggingface_config, num_steps, context, is_draft
            )
            replica_idx = request_to_replica.get(context.request_id, 0)
            if not kv_manager.contains(
                context.request_id, replica_idx=replica_idx
            ):
                kv_manager.claim(context.request_id, replica_idx=replica_idx)
            self._draft_replica_idx[context.request_id] = replica_idx

        for ctx in batch:
            r_idx = self._draft_replica_idx.get(ctx.request_id, 0)
            kv_manager.alloc(ctx, replica_idx=r_idx, num_steps=num_steps)
        kv_cache_inputs = kv_manager.runtime_inputs([batch], num_steps)
        if is_draft:
            return (
                model.prepare_initial_token_inputs(
                    replica_batches=replica_batches,
                    kv_cache_inputs=KVCacheInputsSequence(
                        kv_cache_inputs=kv_cache_inputs
                    ),
                    return_n_logits=return_n_logits,
                ),
                num_steps,
            )
        else:
            assert merged_draft_tokens is not None
            assert merged_draft_offsets is not None
            assert draft_inputs is not None
            kv_cache_updated_inputs: KVCacheInputs
            if isinstance(kv_cache_inputs, Sequence):
                kv_cache_updated_inputs = KVCacheInputsSequence(
                    kv_cache_inputs=kv_cache_inputs,
                )
            else:
                kv_cache_updated_inputs = kv_cache_inputs
            draft_inputs.update(
                tokens=merged_draft_tokens,
                input_row_offsets=merged_draft_offsets,
                signal_buffers=getattr(
                    self._target_model, "signal_buffers", []
                ),
                kv_cache_inputs=kv_cache_updated_inputs,
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
    ) -> tuple[int, Buffer, Buffer, ModelInputs, Buffer]:
        """Generates draft tokens for the batch using the draft model."""
        # Create sampling parameters once for the entire batch
        top_k, max_k, temperature, top_p, min_top_p, seed = (
            self._create_sampling_parameters(batch, self.draft_devices[0])
        )

        # Generate tensor for generated tokens.
        generated_tokens = Buffer.zeros(
            (len(batch), 0), dtype=DType.int64, device=self.draft_devices[0]
        )

        generated_logits = Buffer.zeros(
            (len(batch), 0), dtype=DType.float32, device=self.draft_devices[0]
        )

        # Multi-step execution
        curr_step_inputs = model_inputs

        # num_steps first so that slice indexing is contiguous
        all_draft_logits = Buffer.zeros(
            (num_steps, len(batch), self.vocab_size),
            dtype=DType.float32,
            device=self.draft_devices[0],
        )

        for i in range(num_steps):
            # Execute the model and get next tokens.
            model_outputs = self._draft_model.execute(
                model_inputs=curr_step_inputs
            )

            all_draft_logits[i, :, :].inplace_copy_from(model_outputs.logits)

            # Sample next_token
            new_tokens, new_generated_tokens, new_generated_logits = (
                self.sample_draft_logits(
                    model_outputs,
                    generated_tokens,
                    generated_logits,
                    top_k,
                    max_k,
                    temperature,
                    top_p,
                    min_top_p,
                    seed,
                )
            )
            generated_tokens = new_generated_tokens
            generated_logits = new_generated_logits

            # Increment cache lengths.
            assert isinstance(
                curr_step_inputs.kv_cache_inputs, KVCacheInputsSequence
            ), (
                "prepare_batch instantiates and passes this as a KVCacheInputsSequence"
            )
            assert isinstance(
                curr_step_inputs.kv_cache_inputs.kv_cache_inputs, list
            ), "increment_cache_lengths instantiates and passes this as a list"
            curr_step_inputs.kv_cache_inputs.kv_cache_inputs = (
                self._draft_kv_manager.increment_cache_lengths(
                    curr_step_inputs.kv_cache_inputs.kv_cache_inputs,
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
        all_draft_logits: Buffer,
    ) -> tuple[Buffer, Buffer, Buffer]:
        """Verifies draft tokens against the target model and returns merged outputs."""
        # # The kv cache manager for the target model uses these indices to set the lengths of the cache. We bump them manually here even though the tokens array has not been filled. They are reset when doing the final update of the contexts after both draft and target models have run.
        with reserve_token_space_for_batch(
            context_batch, num_draft_tokens_generated
        ):
            target_inputs, _target_num_steps = self.prepare_batch(
                self._target_model,
                context_batch,
                replica_batches,
                # num steps in this scenario is 1, as we are only
                # generating at one token beyond the draft tokens.
                num_steps=1,
                draft_inputs=draft_inputs,
                return_n_logits=num_draft_tokens_generated + 1,
                is_draft=False,
                merged_draft_tokens=merged_draft_tokens,
                merged_draft_offsets=merged_draft_offsets,
            )

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
            self._rejection_sampler(
                draft_tokens,
                draft_logits,
                target_outputs.logits,
                target_outputs.logit_offsets,
                all_draft_logits,
            )
        )
        assert isinstance(first_rejected_tokens, Buffer)
        assert isinstance(recovered_tokens, Buffer)
        assert isinstance(bonus_tokens, Buffer)

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

        draft_inputs, draft_num_steps = self.prepare_batch(
            self._draft_model,
            context_batch,
            replica_batches,
            self._num_draft_steps,
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
        merged_tokens, merged_offsets = self._ragged_token_merger(
            model_inputs.tokens,  # type: ignore
            model_inputs.input_row_offsets,  # type: ignore
            draft_tokens,
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

        self.update_contexts(
            context_batch=context_batch,
            first_rejected_tokens=first_rejected_tokens.to_numpy(),
            recovered_tokens=recovered_tokens.to_numpy(),
            bonus_tokens=bonus_tokens.to_numpy(),
            draft_tokens=draft_tokens.to_numpy(),
            num_draft_tokens_generated=num_draft_tokens_generated,
        )

        res = self.build_response(context_batch=context_batch)

        # Maybe commit blocks into prefix cache
        self._target_kv_manager.step([context_batch])
        self._draft_kv_manager.step([context_batch])

        return res
