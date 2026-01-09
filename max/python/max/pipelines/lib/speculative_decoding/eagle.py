# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""EAGLE speculative decoding pipeline with target-draft model interaction."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, final

import numpy as np
import numpy.typing as npt
from max.driver import Tensor
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightsAdapter, WeightsFormat
from max.interfaces import (
    PipelineTokenizer,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.kv_cache import NullKVCacheManager
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsSequence,
)
from max.pipelines.core import TextContext, reserve_token_space_for_batch
from max.pipelines.lib.interfaces import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
)
from max.profiler import traced

from ..sampling import token_sampler
from .accepted_hidden_states_extractor import (
    accepted_hidden_states_extractor,
    compute_extractor_inputs,
)
from .base import SpeculativeDecodingPipelineBase
from .hidden_states_filter import compute_filter_indices, filter_hidden_states

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger("max.pipelines")


@final
class EAGLESpeculativeDecodingPipeline(SpeculativeDecodingPipelineBase):
    """EAGLE speculative decoding with target-draft model interaction.

    In EAGLE approach:
    1. Target model generates one token and produces hidden states
    2. Draft model uses these hidden states to generate multiple tokens
    3. Target model verifies draft tokens using rejection sampling
    4. Weight sharing between models for embeddings and lm_head
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[TextContext]],
        eos_token_id: int,
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
        tokenizer: PipelineTokenizer[
            TextContext,
            npt.NDArray[np.integer[Any]],
            TextGenerationRequest,
        ],
        draft_pipeline_model: type[PipelineModel[TextContext]] | None = None,
        draft_weight_adapters: dict[WeightsFormat, WeightsAdapter]
        | None = None,
    ) -> None:
        super().__init__(
            pipeline_config,
            pipeline_model,
            eos_token_id,
            weight_adapters,
            tokenizer,
            draft_pipeline_model,
            draft_weight_adapters,
        )

        # TODO: We may need to support having different sampling parameters for the draft and target model
        self._target_sampler = self._target_session.load(
            token_sampler(
                self.pipeline_config.sampling_config,
                return_logits=True,
                device=DeviceRef.from_device(self.target_devices[0]),
            )
        )

        self._draft_kv_start_idx: dict[RequestID, int] = {}
        self._last_verified_token: dict[RequestID, int] = {}

        self._accepted_hidden_states_extractor = self._target_session.load(
            accepted_hidden_states_extractor(
                device=DeviceRef.from_device(self.target_devices[0])
            )
        )

        self._hidden_states_filter = self._target_session.load(
            filter_hidden_states(
                device=DeviceRef.from_device(self.target_devices[0])
            )
        )

    @traced
    def sample_target_token(
        self,
        target_outputs: ModelOutputs,
        context_batch: list[TextContext],
    ) -> Tensor:
        """Sample token from target model's logits.

        Args:
            target_outputs: Outputs from target model execution containing logits
            context_batch: List of context objects to update

        Returns:
            Tensor of sampled tokens with shape [batch_size, 1]
        """
        top_k, max_k, temperature, top_p, min_top_p, seed = (
            self._create_sampling_parameters(
                context_batch, self.target_devices[0]
            )
        )

        prev_tokens = Tensor.zeros(
            (len(context_batch), 0),
            dtype=DType.int64,
            device=self.target_devices[0],
        )
        prev_logits = Tensor.zeros(
            (len(context_batch), 0),
            dtype=DType.float32,
            device=self.target_devices[0],
        )

        graph_inputs = [
            target_outputs.logits,
            prev_tokens,
            top_k,
            max_k,
            temperature,
            top_p,
            min_top_p,
            seed,
            prev_logits,
        ]
        sampled_tokens, _, _ = self._target_sampler(*graph_inputs)[:3]
        assert isinstance(sampled_tokens, Tensor)

        return sampled_tokens

    def _prepare_common_setup(
        self,
        model: PipelineModel[TextContext],
        batch: list[TextContext],
        num_steps: int,
        is_draft: bool,
    ) -> int:
        """Common setup for both draft and target batch preparation.

        Args:
            model: The pipeline model to prepare batch for
            batch: List of text contexts to process
            num_steps: Number of steps to prepare for
            is_draft: Whether preparing for draft model

        Returns:
            The calculated num_steps for the batch
        """
        assert not isinstance(model.kv_manager, NullKVCacheManager)
        for context in batch:
            num_steps = self.calculate_num_steps(
                model, model.huggingface_config, num_steps, context, is_draft
            )
            if not model.kv_manager.contains(context.request_id):
                model.kv_manager.claim(context.request_id)
        return num_steps

    def _prepare_draft_batch(
        self,
        model: PipelineModel[TextContext],
        batch: list[TextContext],
        num_steps: int,
        return_n_logits: int,
        hidden_states: Tensor,
        needs_ce: bool,
    ) -> tuple[ModelInputs, int]:
        """Prepare batch for draft model execution.

        Handles the complex KV cache index management and hidden states
        for the draft model when using EAGLE speculative decoding.

        Args:
            model: The draft pipeline model
            batch: List of text contexts to process
            num_steps: Number of draft steps to prepare
            return_n_logits: Number of logits to return
            hidden_states: Hidden states from target model
            needs_ce: Whether this is the first iteration (needs context encoding)

        Returns:
            Tuple of (ModelInputs for draft model, num_steps)
        """
        start_indices = [context.tokens.processed_length for context in batch]

        # kv cache needs to fetch starting from 0
        for context in batch:
            if needs_ce:
                self._draft_kv_start_idx[context.request_id] = 0
                context.tokens.rewind_processing(
                    context.tokens.processed_length
                )
            else:
                delta = (
                    self._draft_kv_start_idx[context.request_id]
                    - context.tokens.processed_length
                )
                context.tokens.skip_processing(delta)

        for ctx in batch:
            model.kv_manager.alloc(ctx, num_steps=num_steps)
        kv_cache_inputs = model.kv_manager.get_runtime_inputs(batch, num_steps)

        for i, context in enumerate(batch):
            if needs_ce:
                # Skip the first token in CE
                context.tokens.skip_processing(1)
            else:
                delta = start_indices[i] - context.tokens.processed_length
                if delta > 0:
                    context.tokens.skip_processing(delta)
                else:
                    context.tokens.rewind_processing(-delta)

        base_inputs = model.prepare_initial_token_inputs(
            replica_batches=[batch],
            kv_cache_inputs=KVCacheInputsSequence(
                kv_cache_inputs=kv_cache_inputs
            ),
            return_n_logits=return_n_logits,
        )

        base_inputs.hidden_states = hidden_states

        for i, context in enumerate(batch):
            self._draft_kv_start_idx[context.request_id] += (
                context.tokens.active_length
            )
            delta = start_indices[i] - context.tokens.processed_length
            if delta > 0:
                context.tokens.skip_processing(delta)
            else:
                context.tokens.rewind_processing(-delta)
            context.apply_processing_offset(0)

        return (base_inputs, num_steps)

    def _prepare_initial_target_step(
        self,
        model: PipelineModel[TextContext],
        batch: list[TextContext],
        num_steps: int,
        return_n_logits: int,
    ) -> tuple[ModelInputs, int]:
        """Prepare batch for initial target model step.

        This is used for the first target model execution to generate
        the initial token and hidden states for EAGLE.

        Args:
            model: The target pipeline model
            batch: List of text contexts to process
            num_steps: Number of steps (will be overridden to 1)
            return_n_logits: Number of logits to return

        Returns:
            Tuple of (ModelInputs for target model, 1)
        """
        for ctx in batch:
            model.kv_manager.alloc(ctx, num_steps=num_steps)
        kv_cache_inputs = model.kv_manager.get_runtime_inputs(batch, num_steps)

        # Running 1 step of the target model to get initial token and hidden states for EAGLE
        inputs = model.prepare_initial_token_inputs(
            replica_batches=[batch],
            kv_cache_inputs=KVCacheInputsSequence(
                kv_cache_inputs=kv_cache_inputs
            ),
            return_n_logits=return_n_logits,
        )
        return (inputs, 1)

    def _prepare_verification_step(
        self,
        model: PipelineModel[TextContext],
        batch: list[TextContext],
        num_steps: int,
        return_n_logits: int,
        draft_inputs: ModelInputs,
        merged_tokens: Tensor | None,
        merged_offsets: Tensor | None,
    ) -> tuple[ModelInputs, int]:
        """Prepare batch for target model verification of draft tokens.

        Updates existing draft inputs with merged tokens and KV cache
        for the target model to verify the draft tokens.

        Args:
            model: The target pipeline model
            batch: List of text contexts to process
            num_steps: Number of steps to prepare
            return_n_logits: Number of logits to return
            draft_inputs: Existing model inputs to update
            merged_tokens: Merged draft and input tokens
            merged_offsets: Offsets for merged tokens

        Returns:
            Tuple of (Updated ModelInputs, num_steps)
        """
        for ctx in batch:
            model.kv_manager.alloc(ctx, num_steps=num_steps)
        kv_cache_inputs = model.kv_manager.get_runtime_inputs(batch, num_steps)

        kv_cache_updated_inputs: KVCacheInputs
        if isinstance(kv_cache_inputs, Sequence):
            kv_cache_updated_inputs = KVCacheInputsSequence(
                kv_cache_inputs=kv_cache_inputs,
            )
        else:
            kv_cache_updated_inputs = kv_cache_inputs

        draft_inputs.update(
            tokens=merged_tokens,
            input_row_offsets=merged_offsets,
            signal_buffers=getattr(self._target_model, "signal_buffers", []),
            kv_cache_inputs=kv_cache_updated_inputs,
            return_n_logits=Tensor.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
        )
        return (draft_inputs, num_steps)

    @traced
    def prepare_batch(
        self,
        model: PipelineModel[TextContext],
        batch: list[TextContext],
        num_steps: int,
        return_n_logits: int,
        is_draft: bool = False,
        draft_inputs: ModelInputs | None = None,
        draft_tokens: Tensor | None = None,
        merged_tokens: Tensor | None = None,
        merged_offsets: Tensor | None = None,
        hidden_states: Tensor | None = None,
        needs_ce: bool = False,
    ) -> tuple[ModelInputs, int]:
        """Prepare batch for model execution.

        Routes to appropriate preparation method based on execution mode:
        - Draft model: prepares with hidden states from target
        - Initial target step: generates first token and hidden states
        - Verification step: merges and verifies draft tokens
        """
        num_steps = self._prepare_common_setup(
            model, batch, num_steps, is_draft
        )

        if is_draft:
            assert hidden_states is not None
            return self._prepare_draft_batch(
                model,
                batch,
                num_steps,
                return_n_logits,
                hidden_states,
                needs_ce,
            )
        elif draft_inputs is None:
            return self._prepare_initial_target_step(
                model, batch, num_steps, return_n_logits
            )
        else:
            return self._prepare_verification_step(
                model,
                batch,
                num_steps,
                return_n_logits,
                draft_inputs,
                merged_tokens,
                merged_offsets,
            )

    @traced
    def generate_draft_tokens(
        self,
        batch: list[TextContext],
        num_steps: int,
        model_inputs: ModelInputs,
    ) -> tuple[int, Tensor, Tensor, Tensor, Tensor]:
        # Create sampling parameters once for the entire batch
        top_k, max_k, temperature, top_p, min_top_p, seed = (
            self._create_sampling_parameters(batch, self.draft_devices[0])
        )

        # Generate tensor for generated tokens.
        generated_tokens = Tensor.zeros(
            (len(batch), 0), dtype=DType.int64, device=self.draft_devices[0]
        )

        generated_logits = Tensor.zeros(
            (len(batch), 0), dtype=DType.float32, device=self.draft_devices[0]
        )

        # Multi-step execution
        curr_step_inputs = model_inputs

        # num_steps first so that slice indexing is contiguous
        all_draft_logits = Tensor.zeros(
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
                self._draft_model.kv_manager.increment_cache_lengths(
                    curr_step_inputs.kv_cache_inputs.kv_cache_inputs,
                    curr_step_inputs,
                )
            )

            # Prepare next token inputs.
            curr_step_inputs = self._draft_model.prepare_next_token_inputs(
                new_tokens, curr_step_inputs
            )
            # EAGLE specific
            curr_step_inputs.hidden_states = model_outputs.hidden_states

        assert model_outputs.hidden_states is not None
        return (
            num_steps,
            generated_tokens,
            generated_logits,
            all_draft_logits,
            model_outputs.hidden_states,
        )

    @traced
    def verify_draft_tokens_with_target_model(
        self,
        draft_inputs: ModelInputs,
        context_batch: list[TextContext],
        num_draft_tokens_generated: int,
        draft_tokens: Tensor,
        draft_logits: Tensor,
        all_draft_logits: Tensor,
        merged_tokens: Tensor | None = None,
        merged_offsets: Tensor | None = None,
    ) -> tuple[
        npt.NDArray[np.integer[Any]],
        npt.NDArray[np.integer[Any]],
        npt.NDArray[np.integer[Any]],
    ]:
        with reserve_token_space_for_batch(
            context_batch, num_draft_tokens_generated
        ):
            target_inputs, _ = self.prepare_batch(
                self._target_model,
                context_batch,
                # I believe, num steps in this scenario is 1, as we are only
                # generating one token beyond the draft tokens.
                num_steps=1,
                draft_inputs=draft_inputs,
                return_n_logits=num_draft_tokens_generated + 1,
                is_draft=False,
                draft_tokens=draft_tokens,
                merged_tokens=merged_tokens,
                merged_offsets=merged_offsets,
            )

        target_outputs = self._target_model.execute(model_inputs=target_inputs)

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
        assert isinstance(first_rejected_tokens, Tensor)
        assert isinstance(recovered_tokens, Tensor)
        assert isinstance(bonus_tokens, Tensor)

        first_rejected_tokens_np, recovered_tokens_np, bonus_tokens_np = (
            first_rejected_tokens.to_numpy(),
            recovered_tokens.to_numpy(),
            bonus_tokens.to_numpy(),
        )

        # We keep the hidden states for the target sampled token and the accepted draft tokens
        assert target_outputs.hidden_states is not None
        assert target_outputs.logit_offsets is not None

        # Compute inputs for the extractor graph
        total_range_np, output_offsets_np = compute_extractor_inputs(
            first_rejected_tokens_np
        )

        # Convert to GPU tensors
        total_range_tensor = Tensor.from_numpy(total_range_np).to(
            self.target_devices[0]
        )
        output_offsets_tensor = Tensor.from_numpy(output_offsets_np).to(
            self.target_devices[0]
        )

        # Extract accepted hidden states using the graph
        (accepted_hidden_states,) = self._accepted_hidden_states_extractor(
            target_outputs.hidden_states,
            target_outputs.logit_offsets,
            total_range_tensor,
            output_offsets_tensor,
        )
        assert isinstance(accepted_hidden_states, Tensor)
        self._draft_input_hidden_states = accepted_hidden_states

        return first_rejected_tokens_np, recovered_tokens_np, bonus_tokens_np

    def update_contexts(
        self,
        context_batch: list[TextContext],
        first_rejected_tokens: npt.NDArray[np.integer[Any]],
        recovered_tokens: npt.NDArray[np.integer[Any]],
        bonus_tokens: npt.NDArray[np.integer[Any]],
        draft_tokens: npt.NDArray[np.integer[Any]],
        num_draft_tokens_generated: int,
    ) -> None:
        """Update contexts after EAGLE verification.

        EAGLE-specific behavior:
        - Target token was already added via jump_ahead (start_idx not updated)
        - Draft indices were bumped for KV cache but tokens not written
        - After verification, we "commit" the target token and write accepted draft tokens
        """
        total_draft_generated = num_draft_tokens_generated * len(context_batch)
        total_draft_accepted = 0
        total_bonus_used = 0
        acceptance_lengths = []

        active_context_indices = []
        for idx, context in enumerate(context_batch):
            rejected_token_idx = int(first_rejected_tokens[idx].item())

            for token_idx in range(rejected_token_idx):
                if context.is_done:
                    break
                token = int(draft_tokens[idx, token_idx])
                context.update(token)

            # This is added because the draft needs to process the same tokens but with the hidden states received from the target model. This will also set the start index to the correct position for the kv cache
            context.apply_processing_offset(-rejected_token_idx)

            if not context.is_done:
                if rejected_token_idx < num_draft_tokens_generated:
                    token = int(recovered_tokens[idx, rejected_token_idx])
                else:
                    token = int(bonus_tokens[idx, 0])
                    total_bonus_used += 1
                context.update(token)
                if not context.is_done:
                    active_context_indices.append(idx)
                self._last_verified_token[context.request_id] = token

            total_draft_accepted += rejected_token_idx
            acceptance_lengths.append(rejected_token_idx)

        self._metrics.update(
            total_draft_generated,
            total_draft_accepted,
            total_bonus_used,
            acceptance_lengths,
        )

        # Filter hidden states to remove terminated sequences
        keep_indices_np, offsets = compute_filter_indices(
            first_rejected_tokens, active_context_indices
        )

        # Only filter if some sequences terminated
        if len(keep_indices_np) < int(offsets[-1]):
            keep_tensor = Tensor.from_numpy(keep_indices_np).to(
                self.target_devices[0]
            )
            (filtered_hidden_states,) = self._hidden_states_filter(
                self._draft_input_hidden_states, keep_tensor
            )
            assert isinstance(filtered_hidden_states, Tensor)
            self._draft_input_hidden_states = filtered_hidden_states

    def _target_extend(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> tuple[ModelOutputs, Tensor]:
        context_batch = list(inputs.batch.values())
        target_inputs, _ = self.prepare_batch(
            self._target_model,
            context_batch,
            num_steps=1,  # Only generate 1 token initially
            return_n_logits=1,
            is_draft=False,
            draft_inputs=None,
        )

        target_outputs = self._target_model.execute(model_inputs=target_inputs)

        assert isinstance(
            target_inputs.kv_cache_inputs, KVCacheInputsSequence
        ), (
            "prepare_batch instantiates and passes this as a KVCacheInputsSequence"
        )
        assert isinstance(
            target_inputs.kv_cache_inputs.kv_cache_inputs, list
        ), "increment_cache_lengths instantiates and passes this as a list"
        target_inputs.kv_cache_inputs.kv_cache_inputs = (
            self._target_model.kv_manager.increment_cache_lengths(
                target_inputs.kv_cache_inputs.kv_cache_inputs,
                target_inputs,
            )
        )

        target_sampled_tokens = self.sample_target_token(
            target_outputs, context_batch
        )
        target_sampled_tokens_np = target_sampled_tokens.to_numpy()

        for i, context in enumerate(context_batch):
            context.update(int(target_sampled_tokens_np[i].item()))

        return target_outputs, target_sampled_tokens

    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        """Execute EAGLE speculative decoding.

        EAGLE flow:
        1. Run target model first to generate 1 token and hidden states
        2. Run draft model with hidden states from target model
        3. Verify draft tokens with target model
        4. Update contexts and build response
        """
        context_batch = list(inputs.batch.values())

        needs_ce = context_batch[0].tokens.generated_length == 0
        if needs_ce:
            target_outputs, target_sampled_tokens = self._target_extend(inputs)
            assert target_outputs.hidden_states is not None
            self._draft_input_hidden_states = target_outputs.hidden_states

        draft_inputs, draft_num_steps = self.prepare_batch(
            self._draft_model,
            context_batch,
            self._num_draft_steps,
            return_n_logits=1,
            is_draft=True,
            hidden_states=self._draft_input_hidden_states,
            needs_ce=needs_ce,
        )

        (
            num_draft_tokens_generated,
            draft_tokens,
            draft_logits,
            all_draft_logits,
            self._draft_input_hidden_states,
        ) = self.generate_draft_tokens(
            context_batch, draft_num_steps, draft_inputs
        )

        if needs_ce:
            draft_input_tokens = target_sampled_tokens
        else:
            # Use the last verified tokens from the previous iteration
            last_tokens = np.array(
                [
                    self._last_verified_token[context.request_id]
                    for context in context_batch
                ],
                dtype=np.int64,
            )
            draft_input_tokens = Tensor.from_numpy(last_tokens).to(
                self.target_devices[0]
            )

        draft_input_offsets_np = np.cumsum(
            [0] + [1 for _ in context_batch],
            dtype=np.uint32,
        )
        draft_input_offsets = Tensor.from_numpy(draft_input_offsets_np).to(
            self.target_devices[0]
        )
        merged_tokens, merged_offsets = self._ragged_token_merger(
            draft_input_tokens,
            draft_input_offsets,
            draft_tokens,
        )
        assert isinstance(merged_tokens, Tensor)
        assert isinstance(merged_offsets, Tensor)

        first_rejected_tokens, recovered_tokens, bonus_tokens = (
            self.verify_draft_tokens_with_target_model(
                draft_inputs,
                context_batch,
                num_draft_tokens_generated,
                draft_tokens,
                draft_logits,
                all_draft_logits,
                merged_tokens,
                merged_offsets,
            )
        )

        self.update_contexts(
            context_batch=context_batch,
            first_rejected_tokens=first_rejected_tokens,
            recovered_tokens=recovered_tokens,
            bonus_tokens=bonus_tokens,
            draft_tokens=draft_tokens.to_numpy(),
            num_draft_tokens_generated=num_draft_tokens_generated,
        )

        res = self.build_response(context_batch=context_batch)

        self._target_model.kv_manager.step(context_batch)
        self._draft_model.kv_manager.step(context_batch)

        return res

    @traced
    def release(self, request_id: RequestID) -> None:
        """Release resources associated with this request ID.

        Args:
            request_id: The request ID to release resources for
        """
        super().release(request_id)
        self._draft_kv_start_idx.pop(request_id, None)
        self._last_verified_token.pop(request_id, None)
