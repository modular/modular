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
from max.pipelines.lib.interfaces import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
)
from max.pipelines.lib.utils import compute_data_parallel_splits
from max.profiler import traced
from transformers import AutoConfig

from ..sampling import PenaltyInputs, SamplerInputs, token_sampler
from .base import SpeculativeDecodingPipelineBase, compute_max_num_draft_steps
from .eagle_hidden_state_graphs import build_gather_graph

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

        device_refs = [DeviceRef.from_device(dev) for dev in self.devices]
        self._target_sampler = self._session.load(
            token_sampler(
                self.pipeline_config.sampling,
                return_logits=True,
                device=device_refs[0],
            )
        )

        # Gather graph for extracting hidden states corresponding to accepted tokens after verification
        hf_config = self._target_model.huggingface_config
        hidden_dim = _get_hidden_dim(hf_config)
        self._hs_gather_model = self._session.load(
            build_gather_graph(device_refs, DType.bfloat16, hidden_dim)
        )

    @traced
    def sample_target_token(
        self,
        target_outputs: ModelOutputs,
        context_batch: list[TextContext],
    ) -> Buffer:
        """Sample token from target model's logits.

        Args:
            target_outputs: Outputs from target model execution containing logits
            context_batch: List of context objects to update

        Returns:
            Buffer of sampled tokens with shape [batch_size, 1]
        """
        sampler_inputs = SamplerInputs.create(context_batch, self.devices[0])

        prev_tokens = Buffer.zeros(
            (len(context_batch), 0),
            dtype=DType.int64,
            device=self.devices[0],
        )
        prev_logits = Buffer.zeros(
            (len(context_batch), 0),
            dtype=DType.float32,
            device=self.devices[0],
        )

        graph_inputs: list[Buffer] = [
            target_outputs.logits,
            prev_tokens,
            *sampler_inputs.as_list(),
            prev_logits,
        ]

        if self.pipeline_config.sampling.enable_penalties:
            penalty_inputs = PenaltyInputs.create(
                context_batch, self.devices[0]
            )
            graph_inputs.extend(penalty_inputs.as_list())

        sampled_tokens, _, _ = self._target_sampler(*graph_inputs)[:3]
        assert isinstance(sampled_tokens, Buffer)

        return sampled_tokens

    def _seek_processing_position(
        self,
        context: TextContext,
        target_position: int,
    ) -> None:
        delta = target_position - context.tokens.processed_length
        if delta > 0:
            context.tokens.skip_processing(delta)
        elif delta < 0:
            context.tokens.rewind_processing(-delta)

    def _shift_draft_tokens(
        self,
        base_inputs: ModelInputs,
        batch: list[TextContext],
        shift_next_tokens: npt.NDArray[np.int64],
    ) -> None:
        """Shift each context's tokens left by 1, appending the given token.

        For chunked prefill, the draft model needs to see the same prompt
        tokens but with positions shifted so the last token in each
        context's span is replaced by the target-sampled next token.
        """
        tokens_np = base_inputs.tokens.to_numpy()  # type: ignore[attr-defined]
        shifted = np.empty_like(tokens_np)
        offset = 0
        for i, ctx in enumerate(batch):
            n = ctx.tokens.active_length
            shifted[offset : offset + n - 1] = tokens_np[
                offset + 1 : offset + n
            ]
            shifted[offset + n - 1] = shift_next_tokens[i]
            offset += n
        base_inputs.tokens = Buffer.from_numpy(shifted).to(  # type: ignore[attr-defined]
            base_inputs.tokens.device  # type: ignore[attr-defined]
        )

    def _prepare_draft_batch(
        self,
        batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        return_n_logits: int,
        hidden_states: list[Buffer],
        shift_next_tokens: npt.NDArray[np.int64] | None = None,
    ) -> tuple[ModelInputs, int]:
        """Prepare batch for draft model execution.

        Handles the complex KV cache index management and hidden states
        for the draft model when using EAGLE speculative decoding.

        Args:
            batch: List of text contexts to process
            replica_batches: List of per-replica batches for data parallelism
            return_n_logits: Number of logits to return
            hidden_states: Hidden states from target model
            shift_next_tokens: Tokens to append after dropping the first token

        Returns:
            Tuple of (ModelInputs for draft model, num_steps)
        """
        num_steps = compute_max_num_draft_steps(
            replica_batches,
            desired_num_draft_steps=self._num_draft_steps,
            max_seq_len=self._max_seq_len,
            is_draft=True,
        )

        saved_positions = [context.tokens.processed_length for context in batch]
        # Check before accessing spec_decoding_state (which auto-creates it).
        new_request_flags = [
            context._spec_decoding_state is None for context in batch
        ]

        for context in batch:
            self._seek_processing_position(
                context, context.spec_decoding_state.draft_kv_start_idx
            )

        kv_cache_inputs = self._target_kv_manager.runtime_inputs(
            replica_batches, num_steps
        )
        # Huge hack alert!
        # Swap out the target kv cache buffers for the draft kv cache buffers
        for replica_input, draft_blocks in zip(
            kv_cache_inputs.inputs, self._draft_kv_buffers, strict=True
        ):
            replica_input.blocks = draft_blocks

        for i, context in enumerate(batch):
            if new_request_flags[i]:
                if shift_next_tokens is None:
                    context.tokens.skip_processing(1)
            else:
                self._seek_processing_position(context, saved_positions[i])

        base_inputs = self._draft_model.prepare_initial_token_inputs(
            replica_batches=replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )

        if shift_next_tokens is not None:
            self._shift_draft_tokens(base_inputs, batch, shift_next_tokens)

        if hidden_states is not None:
            base_inputs.hidden_states = (
                hidden_states[0] if len(hidden_states) == 1 else hidden_states
            )

        # Compute per-device context lengths for DP mode
        if (
            hasattr(base_inputs, "batch_context_lengths")
            and base_inputs.batch_context_lengths
        ):
            page_size = self._draft_model.kv_cache_config.kv_cache_page_size

            def align_length(length: int) -> int:
                return (length + page_size - 1) // page_size * page_size

            for i, replica_batch in enumerate(replica_batches):
                device_context_length = sum(
                    align_length(
                        ctx.spec_decoding_state.draft_kv_start_idx
                        + ctx.tokens.active_length
                    )
                    for ctx in replica_batch
                )

                base_inputs.batch_context_lengths[i][0] = device_context_length

            if len(replica_batches) != len(self.devices):
                # We only support either DP=1 or DP=n_devices.
                assert len(replica_batches) == 1
                # Duplicate the batch context lengths for each device.
                for dev_idx in range(1, len(base_inputs.batch_context_lengths)):
                    base_inputs.batch_context_lengths[dev_idx][0] = (
                        base_inputs.batch_context_lengths[0][0].item()
                    )

        for i, context in enumerate(batch):
            state = context.spec_decoding_state
            state.draft_kv_start_idx += context.tokens.active_length
            self._seek_processing_position(context, saved_positions[i])
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

        generated_tokens = Buffer.zeros(
            (len(batch), 0), dtype=DType.int64, device=self.devices[0]
        )

        generated_logits = Buffer.zeros(
            (len(batch), 0), dtype=DType.float32, device=self.devices[0]
        )

        curr_step_inputs = model_inputs

        for _ in range(num_steps):
            model_outputs = self._draft_model.execute(
                model_inputs=curr_step_inputs
            )

            new_tokens, new_generated_tokens, new_generated_logits = (
                self.sample_draft_logits(
                    model_outputs,
                    generated_tokens,
                    generated_logits,
                    sampler_inputs=sampler_inputs,
                    penalty_inputs=penalty_inputs,
                )
            )

            generated_tokens = new_generated_tokens
            generated_logits = new_generated_logits

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

        assert model_outputs.hidden_states is not None
        return num_steps, generated_tokens

    @traced
    def verify_draft_tokens_with_target_model(
        self,
        context_batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        num_draft_tokens_generated: int,
        draft_tokens: Buffer,
        merged_tokens: Buffer | None = None,
        merged_offsets: Buffer | None = None,
        host_merged_offsets: Buffer | None = None,
    ) -> tuple[
        npt.NDArray[np.integer[Any]],
        npt.NDArray[np.integer[Any]],
        npt.NDArray[np.integer[Any]] | None,
        list[Buffer],
        npt.NDArray[np.int64],
    ]:
        """Verifies draft tokens against the target model.

        Returns:
            Tuple of (first_rejected_tokens, recovered_tokens, bonus_tokens,
            target_hidden_states, logit_offsets) where hidden states and
            logit offsets can be used for subsequent draft generation.
        """
        # KV alloc must happen inside reserve_token_space_for_batch so the
        # KV manager sees the expanded token count. prepare_initial_token_inputs
        # must happen outside because it accesses ctx.tokens.active which
        # would see a bumped range exceeding the underlying array capacity.
        with reserve_token_space_for_batch(
            context_batch, num_draft_tokens_generated
        ):
            kv_cache_inputs = self._target_kv_manager.runtime_inputs(
                replica_batches, num_steps=1
            )

        target_inputs = self._target_model.prepare_initial_token_inputs(
            replica_batches=replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=num_draft_tokens_generated + 1,
        )

        target_inputs.tokens = merged_tokens  # type: ignore[attr-defined]
        target_inputs.input_row_offsets = merged_offsets  # type: ignore[attr-defined]
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

            for i, replica_batch in enumerate(replica_batches):
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
        logit_offsets_np = target_outputs.logit_offsets.to_numpy()

        hs = target_outputs.hidden_states
        target_hidden_states: list[Buffer] = (
            hs if isinstance(hs, list) else [hs]
        )

        return (
            first_rejected_tokens_np,
            recovered_tokens_np,
            bonus_tokens_np,
            target_hidden_states,
            logit_offsets_np,
        )

    def update_contexts(
        self,
        context_batch: list[TextContext],
        first_rejected_tokens: npt.NDArray[np.integer[Any]],
        recovered_tokens: npt.NDArray[np.integer[Any]],
        bonus_tokens: npt.NDArray[np.integer[Any]] | None,
        draft_tokens: npt.NDArray[np.integer[Any]],
        num_draft_tokens_generated: int,
        data_parallel_splits: npt.NDArray[np.int64] | None = None,
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

        for idx, context in enumerate(context_batch):
            rejected_token_idx = int(first_rejected_tokens[idx].item())

            for token_idx in range(rejected_token_idx):
                if context.is_done:
                    break
                token = int(draft_tokens[idx, token_idx])
                context.update(token)

            if not context.is_done:
                if rejected_token_idx < num_draft_tokens_generated:
                    # Draft token rejected - use recovered token from target
                    # Greedy sampler: recovered_tokens shape is [batch_size, 1]
                    # Residuals sampler: recovered_tokens shape is [batch_size, num_steps]
                    if bonus_tokens is None:
                        # Greedy sampler - only one recovered token per batch
                        token = int(recovered_tokens[idx, 0])
                    else:
                        # Residual sampler - tokens for all positions
                        token = int(recovered_tokens[idx, rejected_token_idx])
                    context.update(token)
                elif bonus_tokens is not None:
                    # All drafts accepted + bonus token available
                    token = int(bonus_tokens[idx, 0])
                    total_bonus_used += 1
                    context.update(token)

            # This is added because the draft needs to process the same tokens but with the hidden states received from the target model. This will also set the start index to the correct position for the kv cache
            context.apply_processing_offset(-rejected_token_idx)

            # Cap draft_kv_start_idx to processed_length so stale draft KV
            # entries (from rejected tokens) get overwritten on the next
            # iteration.
            state = context.spec_decoding_state
            state.draft_kv_start_idx = min(
                state.draft_kv_start_idx,
                context.tokens.processed_length,
            )

            total_draft_accepted += rejected_token_idx
            acceptance_lengths.append(rejected_token_idx)

        self._metrics.update(
            total_draft_generated,
            total_draft_accepted,
            total_bonus_used,
            acceptance_lengths,
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
        target_hidden_states: list[Buffer],
        logit_offsets_np: npt.NDArray[np.int64],
        first_rejected_np: npt.NDArray[np.integer[Any]],
        data_parallel_splits_np: npt.NDArray[np.int64],
    ) -> list[Buffer]:
        """Gather accepted hidden states from verification output for draft input."""
        model_args: list[Buffer] = []
        for dev_idx in range(len(self.devices)):
            start = int(data_parallel_splits_np[dev_idx])
            end = int(data_parallel_splits_np[dev_idx + 1])
            local_offset = int(logit_offsets_np[start])
            gather_indices: list[int] = []
            for i in range(start, end):
                num_rows = int(first_rejected_np[i]) + 1
                src_start = int(logit_offsets_np[i]) - local_offset
                for r in range(num_rows):
                    gather_indices.append(src_start + r)
            if gather_indices:
                indices_np = np.array(gather_indices, dtype=np.int64)
            else:
                indices_np = np.array([], dtype=np.int64)
            indices_buf = Buffer.from_numpy(indices_np).to(
                self.devices[dev_idx]
            )
            model_args.extend([target_hidden_states[dev_idx], indices_buf])

        outputs = self._hs_gather_model(*model_args)
        if isinstance(outputs, Buffer):
            return [outputs]
        result: list[Buffer] = []
        for out in outputs:
            assert isinstance(out, Buffer)
            result.append(out)
        return result

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
        replica_batches = inputs.batches
        data_parallel_splits_np = compute_data_parallel_splits(replica_batches)

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
        for context in context_batch:
            for replica_idx in range(len(replica_batches)):
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
                context_batch, replica_batches
            )
        else:
            return self._execute_token_generation(
                context_batch, replica_batches, data_parallel_splits_np
            )

    def _execute_context_encoding(
        self,
        context_batch: list[TextContext],
        replica_batches: list[list[TextContext]],
    ) -> dict[RequestID, TextGenerationOutput]:
        kv_cache_inputs = self._target_kv_manager.runtime_inputs(
            replica_batches, num_steps=1
        )

        inputs = self._target_model.prepare_initial_token_inputs(
            replica_batches=replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=1,
        )

        target_outputs = self._target_model.execute(model_inputs=inputs)

        target_sampled_tokens = self.sample_target_token(
            target_outputs, context_batch
        )
        target_sampled_tokens_np = target_sampled_tokens.to_numpy()

        assert target_outputs.hidden_states is not None
        hs = target_outputs.hidden_states
        ce_hs: list[Buffer] = hs if isinstance(hs, list) else [hs]
        next_tokens_for_shift: npt.NDArray[np.int64] = (
            target_sampled_tokens_np.flatten().astype(np.int64)
        )

        draft_ce_inputs, _ = self._prepare_draft_batch(
            context_batch,
            replica_batches,
            return_n_logits=1,
            hidden_states=ce_hs,
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

        return self.build_response(context_batch=context_batch)

    def _execute_token_generation(
        self,
        context_batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        data_parallel_splits_np: npt.NDArray[np.int64],
    ) -> dict[RequestID, TextGenerationOutput]:
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
        merged_tokens, merged_offsets = self._ragged_token_merger(
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
            logit_offsets_np,
        ) = self.verify_draft_tokens_with_target_model(
            context_batch,
            replica_batches,
            num_draft_tokens_generated,
            draft_tokens,
            merged_tokens=merged_tokens,
            merged_offsets=merged_offsets,
            host_merged_offsets=host_merged_offsets,
        )

        self.update_contexts(
            context_batch=context_batch,
            first_rejected_tokens=first_rejected_np,
            recovered_tokens=recovered_np,
            bonus_tokens=bonus_np,
            draft_tokens=draft_tokens.to_numpy(),
            num_draft_tokens_generated=num_draft_tokens_generated,
            data_parallel_splits=data_parallel_splits_np,
        )

        draft_hidden_states = self._extract_hs_for_draft(
            target_hs,
            logit_offsets_np,
            first_rejected_np,
            data_parallel_splits_np,
        )

        draft_inputs, draft_num_steps = self._prepare_draft_batch(
            context_batch,
            replica_batches,
            return_n_logits=1,
            hidden_states=draft_hidden_states,
        )

        new_num_draft_tokens, new_draft_tokens = self.generate_draft_tokens(
            context_batch, draft_num_steps, draft_inputs
        )

        self._save_draft_tokens(
            context_batch, new_draft_tokens, new_num_draft_tokens
        )

        res = self.build_response(context_batch=context_batch)

        self._target_kv_manager.step(replica_batches)

        return res
