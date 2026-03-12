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
"""EAGLE speculative decoding pipeline with target-draft model interaction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, final

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer
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
from max.pipelines.core import TextContext, reserve_token_space_for_batch
from max.pipelines.lib.interfaces import ModelInputs, PipelineModel
from max.profiler import traced
from transformers import AutoConfig

from ..sampling import PenaltyInputs, SamplerInputs
from .base import SpeculativeDecodingPipelineBase
from .eagle_hidden_state_graphs import build_gather_graph
from .utils import (
    ModelInputsWithTokensAndOffsets,
    build_response,
    compute_max_num_draft_steps,
    seek_processing_position,
    shift_draft_tokens,
    update_contexts_and_compute_metrics_eagle,
)

if TYPE_CHECKING:
    from ..config import PipelineConfig

logger = logging.getLogger("max.pipelines")


def _get_hidden_dim(hf_config: AutoConfig) -> int:
    if hasattr(hf_config, "hidden_size"):
        return hf_config.hidden_size
    elif hasattr(hf_config, "text_config") and hasattr(
        hf_config.text_config, "hidden_size"
    ):
        return hf_config.text_config.hidden_size
    else:
        raise ValueError(
            "Could not determine hidden_size from target model config"
        )


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

        # Gather graph for extracting hidden states corresponding to accepted tokens after verification
        device_refs = [DeviceRef.from_device(dev) for dev in self.devices]
        hf_config = self._target_model.huggingface_config
        hidden_dim = _get_hidden_dim(hf_config)
        # hidden_states is now a single tensor on device 0, so only build
        # the gather graph for one device.
        self._hs_gather_model = self._session.load(
            build_gather_graph(device_refs[:1], DType.bfloat16, hidden_dim)
        )

    def _prepare_draft_batch(
        self,
        inputs: TextGenerationInputs[TextContext],
        return_n_logits: int,
        hidden_states: Buffer,
        shift_next_tokens: npt.NDArray[np.int64] | None = None,
    ) -> tuple[ModelInputs, int]:
        """Prepare batch for draft model execution.

        Handles the complex KV cache index management and hidden states
        for the draft model when using EAGLE speculative decoding.

        Args:
            inputs: Inputs for the draft model
            return_n_logits: Number of logits to return
            hidden_states: Hidden states from target or draft model
            shift_next_tokens: Tokens to append after dropping the first token

        Returns:
            Tuple of (ModelInputs for draft model, num_steps)
        """
        num_steps = compute_max_num_draft_steps(
            inputs.batches,
            desired_num_draft_steps=self._num_draft_steps,
            max_seq_len=self._max_seq_len,
            is_draft=True,
        )

        context_batch = inputs.flat_batch
        saved_positions = [
            context.tokens.processed_length for context in context_batch
        ]
        # Check before accessing spec_decoding_state (which auto-creates it).
        new_request_flags = [
            context._spec_decoding_state is None for context in context_batch
        ]

        for context in context_batch:
            seek_processing_position(
                context, context.spec_decoding_state.draft_kv_start_idx
            )

        kv_cache_inputs = self._target_kv_manager.runtime_inputs(
            inputs.batches, num_steps
        )
        # Huge hack alert!
        # Swap out the target kv cache buffers for the draft kv cache buffers
        for replica_input, draft_blocks in zip(
            kv_cache_inputs.inputs, self._draft_kv_buffers, strict=True
        ):
            replica_input.blocks = draft_blocks

        for i, context in enumerate(context_batch):
            if new_request_flags[i]:
                if shift_next_tokens is None:
                    context.tokens.skip_processing(1)
            else:
                seek_processing_position(context, saved_positions[i])

        base_inputs = self._draft_model.prepare_initial_token_inputs(
            replica_batches=inputs.batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )

        if shift_next_tokens is not None:
            assert isinstance(base_inputs, ModelInputsWithTokensAndOffsets)
            tokens = base_inputs.tokens
            shifted = shift_draft_tokens(
                tokens.to_numpy(),
                context_batch,
                shift_next_tokens,
            )
            device = tokens.device
            base_inputs.tokens = Buffer.from_numpy(shifted).to(device)

        base_inputs.hidden_states = hidden_states

        # Compute per-device context lengths for DP mode
        if (
            hasattr(base_inputs, "batch_context_lengths")
            and base_inputs.batch_context_lengths
        ):
            page_size = self._draft_model.kv_cache_config.kv_cache_page_size

            def align_length(length: int) -> int:
                return (length + page_size - 1) // page_size * page_size

            for i, replica_batch in enumerate(inputs.batches):
                device_context_length = sum(
                    align_length(
                        ctx.spec_decoding_state.draft_kv_start_idx
                        + ctx.tokens.active_length
                    )
                    for ctx in replica_batch
                )

                base_inputs.batch_context_lengths[i][0] = device_context_length

            if len(inputs.batches) != len(self.devices):
                # We only support either DP=1 or DP=n_devices.
                assert len(inputs.batches) == 1
                # Duplicate the batch context lengths for each device.
                for dev_idx in range(1, len(base_inputs.batch_context_lengths)):
                    base_inputs.batch_context_lengths[dev_idx][0] = (
                        base_inputs.batch_context_lengths[0][0].item()
                    )

        for i, context in enumerate(context_batch):
            state = context.spec_decoding_state
            state.draft_kv_start_idx += context.tokens.active_length
            seek_processing_position(context, saved_positions[i])
            context.apply_processing_offset(0)

        return base_inputs, num_steps

    @traced
    def generate_draft_tokens(
        self,
        batch: list[TextContext],
        num_steps: int,
        model_inputs: ModelInputs,
    ) -> tuple[int, Buffer]:
        """Generates draft tokens for the batch using the draft model."""
        # Create sampling parameters once for the entire batch
        sampler_inputs = SamplerInputs.create(batch, self.devices[0])

        # Build penalty inputs once for the entire draft generation loop
        penalty_inputs: PenaltyInputs | None = None
        if self.pipeline_config.sampling.enable_penalties:
            penalty_inputs = PenaltyInputs.create(
                batch, self.devices[0], num_steps=num_steps
            )

        curr_step_inputs = model_inputs

        generated_tokens: list[Buffer] = []
        for _ in range(num_steps):
            model_outputs = self._draft_model.execute(
                model_inputs=curr_step_inputs
            )

            new_tokens = self._sampler.sample_logits(
                logits=model_outputs.logits,
                sampler_inputs=sampler_inputs,
                penalty_inputs=penalty_inputs,
            )
            generated_tokens.append(new_tokens)

            assert curr_step_inputs.kv_cache_inputs is not None
            curr_step_inputs.kv_cache_inputs = (
                self._target_kv_manager.increment_cache_lengths(
                    curr_step_inputs.kv_cache_inputs,
                    curr_step_inputs,
                )
            )

            curr_step_inputs = self._draft_model.prepare_next_token_inputs(
                new_tokens, curr_step_inputs
            )
            curr_step_inputs.hidden_states = model_outputs.hidden_states

        # Column stack the list of generated tokens per step
        # [(batch_size,), (batch_size,), ...] -> (batch_size, num_steps)
        generated_tokens_np = [token.to_numpy() for token in generated_tokens]
        generated_tokens_concat_np = np.column_stack(generated_tokens_np)
        generated_tokens_concat = Buffer.from_numpy(generated_tokens_concat_np)

        assert model_outputs.hidden_states is not None
        return num_steps, generated_tokens_concat

    @traced
    def _verify_draft_tokens_with_target_model(
        self,
        inputs: TextGenerationInputs[TextContext],
        num_draft_tokens_generated: int,
        draft_tokens: Buffer,
        merged_tokens: Buffer,
        merged_offsets: Buffer,
        host_merged_offsets: Buffer | None = None,
    ) -> tuple[
        npt.NDArray[np.integer[Any]],
        npt.NDArray[np.integer[Any]],
        npt.NDArray[np.integer[Any]] | None,
        Buffer,
    ]:
        """Verifies draft tokens against the target model.

        Returns:
            Tuple of (first_rejected_tokens, recovered_tokens, bonus_tokens,
            target_hidden_states, logit_offsets) where hidden states and
            logit offsets can be used for subsequent draft generation.
        """
        context_batch = inputs.flat_batch
        # KV alloc must happen inside reserve_token_space_for_batch so the
        # KV manager sees the expanded token count. prepare_initial_token_inputs
        # must happen outside because it accesses ctx.tokens.active which
        # would see a bumped range exceeding the underlying array capacity.
        with reserve_token_space_for_batch(
            context_batch, num_draft_tokens_generated
        ):
            kv_cache_inputs = self._target_kv_manager.runtime_inputs(
                inputs.batches, num_steps=1
            )

        target_inputs = self._target_model.prepare_initial_token_inputs(
            replica_batches=inputs.batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=num_draft_tokens_generated + 1,
        )

        assert isinstance(target_inputs, ModelInputsWithTokensAndOffsets)
        target_inputs.tokens = merged_tokens
        target_inputs.input_row_offsets = merged_offsets
        target_inputs.host_input_row_offsets = host_merged_offsets  # type: ignore[attr-defined]

        # Fix batch_context_lengths: prepare_initial_token_inputs computed
        # current_position outside the context manager (un-bumped).
        # Recompute with bumped positions so each request's length is
        # page-aligned correctly (adding raw tokens to an already-aligned
        # sum would produce invalid offsets for the MLA prefill kernel).
        if hasattr(target_inputs, "batch_context_lengths"):
            page_size = self._target_model.kv_cache_config.kv_cache_page_size

            def _align(length: int) -> int:
                return (length + page_size - 1) // page_size * page_size

            for i, replica_batch in enumerate(inputs.batches):
                target_inputs.batch_context_lengths[i][0] = sum(
                    _align(
                        ctx.tokens.current_position + num_draft_tokens_generated
                    )
                    for ctx in replica_batch
                )

        target_outputs = self._target_model.execute(model_inputs=target_inputs)

        assert target_outputs.logit_offsets is not None
        first_rejected_tokens, recovered_tokens, bonus_tokens = (
            self._rejection_runner.run(
                draft_tokens=draft_tokens,
                draft_logits=None,
                target_logits=target_outputs.logits,
                target_logit_offsets=target_outputs.logit_offsets,
                all_draft_logits=None,
                context_batch=context_batch,
            )
        )

        first_rejected_tokens_np = first_rejected_tokens.to_numpy()
        recovered_tokens_np = recovered_tokens.to_numpy()
        if bonus_tokens is not None:
            assert isinstance(bonus_tokens, Buffer)
            bonus_tokens_np: npt.NDArray[np.integer[Any]] | None = (
                bonus_tokens.to_numpy()
            )
        else:
            bonus_tokens_np = None

        assert target_outputs.hidden_states is not None

        hs = target_outputs.hidden_states
        assert isinstance(hs, Buffer)

        return (
            first_rejected_tokens_np,
            recovered_tokens_np,
            bonus_tokens_np,
            hs,
        )

    def _save_draft_tokens(
        self,
        context_batch: list[TextContext],
        draft_tokens: Buffer,
        num_draft_tokens: int,
    ) -> None:
        """Save draft tokens from this iteration for verification in the next."""
        draft_tokens_np = draft_tokens.to_numpy()
        for i, ctx in enumerate(context_batch):
            if not ctx.is_done:
                ctx.spec_decoding_state.saved_draft_tokens = draft_tokens_np[
                    i, :num_draft_tokens
                ].copy()

    def _load_saved_draft_tokens(
        self,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, int]:
        """Load saved draft tokens and reconstruct into a batch tensor."""
        max_num_tokens = max(
            len(ctx.spec_decoding_state.saved_draft_tokens)
            for ctx in context_batch
        )
        batch_tokens = np.zeros(
            (len(context_batch), max_num_tokens), dtype=np.int64
        )
        for i, ctx in enumerate(context_batch):
            tokens = ctx.spec_decoding_state.saved_draft_tokens
            batch_tokens[i, : len(tokens)] = tokens
        return (
            Buffer.from_numpy(batch_tokens).to(self.devices[0]),
            max_num_tokens,
        )

    def _extract_hs_for_draft(
        self,
        hidden_states: Buffer,
        logit_offsets: list[int],
        first_rejected: list[int],
    ) -> Buffer:
        """Gather accepted hidden states from verification output for draft input."""
        # Compute gather indices
        gather_indices: list[int] = []
        for start_row, num_rows in zip(
            logit_offsets[:-1], first_rejected, strict=True
        ):
            for r in range(num_rows + 1):
                gather_indices.append(start_row + r)
        if gather_indices:
            indices_np = np.array(gather_indices, dtype=np.int64)
        else:
            indices_np = np.array([], dtype=np.int64)

        indices_buf = Buffer.from_numpy(indices_np).to(hidden_states.device)
        (sliced_hs,) = self._hs_gather_model(hidden_states, indices_buf)
        return sliced_hs

    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        """Executes EAGLE speculative decoding.

        EAGLE verify-then-draft flow:

        1. Prefill: target forward + draft KV warmup + sample one draft token.
        2. Decode: verify saved drafts + draft new tokens using verification
           hidden states.
        """
        # TODO: The sampled draft token during prefill is only for having something to verify in the first call to decode
        context_batch = inputs.flat_batch

        need_penalties = any(
            context.sampling_params.needs_penalties for context in context_batch
        )
        if (
            need_penalties
            and not self.pipeline_config.sampling.enable_penalties
        ):
            logger.warning(
                "Penalties are provided in the request, but the model was not configured with enable_penalties=True, ignoring"
            )

        # Allocate enough kv for 2 * num_draft_steps
        # This ensures we have enough to verify upwards of num_draft_steps tokens
        # and then generate num_draft_steps more tokens.
        # TODO: move this logic to the scheduler
        for replica_idx, replica_batch in enumerate(inputs.batches):
            for context in replica_batch:
                self._target_kv_manager.alloc(
                    context,
                    replica_idx=replica_idx,
                    num_steps=2 * self._num_draft_steps,
                )

        # If any request is in prefill (generated_length == 0), route entire batch
        # through context encoding path
        has_prefill = any(
            ctx.tokens.generated_length == 0 for ctx in context_batch
        )

        if has_prefill:
            return self._execute_context_encoding(
                inputs=inputs,
            )
        else:
            return self._execute_token_generation(
                inputs=inputs,
            )

    def _execute_context_encoding(
        self,
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        kv_cache_inputs = self._target_kv_manager.runtime_inputs(
            inputs.batches, num_steps=1
        )

        target_ce_inputs = self._target_model.prepare_initial_token_inputs(
            replica_batches=inputs.batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=1,
        )

        target_outputs = self._target_model.execute(
            model_inputs=target_ce_inputs
        )

        context_batch = inputs.flat_batch
        penalty_inputs: PenaltyInputs | None = None
        if self.pipeline_config.sampling.enable_penalties:
            penalty_inputs = PenaltyInputs.create(
                context_batch, self.devices[0]
            )

        sampler_inputs = SamplerInputs.create(
            inputs.flat_batch, self.devices[0]
        )

        target_sampled_tokens = self._sampler.sample_logits(
            logits=target_outputs.logits,
            sampler_inputs=sampler_inputs,
            penalty_inputs=penalty_inputs,
        )
        target_sampled_tokens_np = target_sampled_tokens.to_numpy()

        assert target_outputs.hidden_states is not None
        hs = target_outputs.hidden_states
        assert isinstance(hs, Buffer)
        next_tokens_for_shift: npt.NDArray[np.int64] = (
            target_sampled_tokens_np.flatten().astype(np.int64)
        )

        draft_ce_inputs, _ = self._prepare_draft_batch(
            inputs=inputs,
            return_n_logits=1,
            hidden_states=hs,
            shift_next_tokens=next_tokens_for_shift,
        )
        draft_outputs = self._draft_model.execute(model_inputs=draft_ce_inputs)

        draft_logits_np = draft_outputs.logits.to_numpy()
        draft_sampled_tokens = draft_logits_np.argmax(axis=-1)

        for i, ctx in enumerate(context_batch):
            if not ctx.tokens.actively_chunked:
                state = ctx.spec_decoding_state
                state.saved_draft_tokens = np.array(
                    [int(draft_sampled_tokens[i])], dtype=np.int64
                )
            token = int(target_sampled_tokens_np[i].item())
            ctx.update(token)

        return build_response(
            context_batch=context_batch, max_seq_len=self._max_seq_len
        )

    def _execute_token_generation(
        self,
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        context_batch = inputs.flat_batch
        draft_tokens, num_draft_tokens_generated = (
            self._load_saved_draft_tokens(context_batch)
        )

        # Build merged tokens for verification: [last_verified, drafts...]
        last_tokens = np.array(
            [int(context.tokens[-1]) for context in context_batch],
            dtype=np.int64,
        )
        draft_input_tokens = Buffer.from_numpy(last_tokens).to(self.devices[0])
        draft_input_offsets_np = np.cumsum(
            [0] + [1 for _ in context_batch],
            dtype=np.uint32,
        )
        draft_input_offsets = Buffer.from_numpy(draft_input_offsets_np).to(
            self.devices[0]
        )
        merged_tokens, merged_offsets = self._ragged_token_merger.run(
            draft_input_tokens,
            draft_input_offsets,
            draft_tokens,
        )
        assert isinstance(merged_tokens, Buffer)
        assert isinstance(merged_offsets, Buffer)

        host_merged_offsets: Buffer | None = None
        if self._speculative_config.is_mtp():
            host_merged_offsets = merged_offsets.to(CPU())

        # 3. Verify saved draft tokens with target model
        (
            first_rejected_np,
            recovered_np,
            bonus_np,
            target_hs,
        ) = self._verify_draft_tokens_with_target_model(
            inputs=inputs,
            num_draft_tokens_generated=num_draft_tokens_generated,
            draft_tokens=draft_tokens,
            merged_tokens=merged_tokens,
            merged_offsets=merged_offsets,
            host_merged_offsets=host_merged_offsets,
        )

        metrics = update_contexts_and_compute_metrics_eagle(
            context_batch=context_batch,
            first_rejected_tokens=first_rejected_np,
            recovered_tokens=recovered_np,
            bonus_tokens=bonus_np,
            draft_tokens=draft_tokens.to_numpy(),
            num_draft_tokens_generated=num_draft_tokens_generated,
        )
        self.metrics.update(metrics)

        # We extract the hidden state for the tokens that were successfully verified
        # by the target model. This first draft step of the auto-regressive decode
        # loop will use these hidden states to extend the draft kv cache. Note that
        # we cannot just use the draft kv cache produced when speculating the draft
        # tokens since they are approximate!
        sliced_target_hs = self._extract_hs_for_draft(
            target_hs,
            merged_offsets.to_numpy().tolist(),
            first_rejected_np.tolist(),
        )
        draft_inputs, draft_num_steps = self._prepare_draft_batch(
            inputs=inputs,
            return_n_logits=1,
            hidden_states=sliced_target_hs,
        )

        new_num_draft_tokens, new_draft_tokens = self.generate_draft_tokens(
            context_batch, draft_num_steps, draft_inputs
        )

        self._save_draft_tokens(
            context_batch, new_draft_tokens, new_num_draft_tokens
        )

        res = build_response(
            context_batch=context_batch, max_seq_len=self._max_seq_len
        )

        self._target_kv_manager.step(inputs.batches)

        return res
