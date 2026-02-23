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
from collections.abc import Sequence
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
from max.nn.legacy.kv_cache import KVCacheInputs, KVCacheInputsSequence
from max.pipelines.core import TextContext, reserve_token_space_for_batch
from max.pipelines.lib.interfaces import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
)
from max.pipelines.lib.utils import compute_data_parallel_splits
from max.profiler import traced

from ..sampling import token_sampler
from .base import SpeculativeDecodingPipelineBase
from .eagle_hidden_state_manager import EagleHiddenStateManager

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
                self.pipeline_config.sampling,
                return_logits=True,
                device=DeviceRef.from_device(self.target_devices[0]),
            )
        )

        self._draft_kv_start_idx: dict[RequestID, int] = {}
        self._last_verified_token: dict[RequestID, int] = {}

        # Initialize hidden state manager for per-request hidden state tracking
        hf_config = self._target_model.huggingface_config
        hidden_dim = getattr(hf_config, "hidden_size", None)
        if hidden_dim is None and hasattr(hf_config, "text_config"):
            hidden_dim = hf_config.text_config.hidden_size
        assert hidden_dim is not None, (
            "Could not determine hidden_size from target model config"
        )
        self._hidden_state_manager = EagleHiddenStateManager(
            hidden_dim=hidden_dim,
            dtype=DType.bfloat16,
            devices=self.target_devices,
            max_batch_size=self.pipeline_config.max_batch_size or 128,
            num_draft_steps=self._num_draft_steps,
            session=self._target_session,
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
        top_k, max_k, temperature, top_p, min_top_p, seed = (
            self._create_sampling_parameters(
                context_batch, self.target_devices[0]
            )
        )

        prev_tokens = Buffer.zeros(
            (len(context_batch), 0),
            dtype=DType.int64,
            device=self.target_devices[0],
        )
        prev_logits = Buffer.zeros(
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
        assert isinstance(sampled_tokens, Buffer)

        return sampled_tokens

    def _prepare_common_setup(
        self,
        model: PipelineModel[TextContext],
        batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        num_steps: int,
        is_draft: bool,
    ) -> int:
        """Common setup for both draft and target batch preparation.

        Args:
            model: The pipeline model to prepare batch for
            batch: List of text contexts to process
            replica_batches: Per-replica batches for data parallelism
            num_steps: Number of steps to prepare for
            is_draft: Whether preparing for draft model

        Returns:
            The calculated num_steps for the batch
        """
        # Build request_id -> replica_idx mapping from replica_batches
        request_to_replica: dict[RequestID, int] = {}
        for replica_idx, replica_batch in enumerate(replica_batches):
            for ctx in replica_batch:
                request_to_replica[ctx.request_id] = replica_idx

        for context in batch:
            num_steps = self.calculate_num_steps(
                model, model.huggingface_config, num_steps, context, is_draft
            )
            # For draft model: claim if not already claimed (target model is claimed by scheduler)
            # For target model: scheduler handles claiming, so skip here
            if is_draft:
                replica_idx = request_to_replica.get(context.request_id, 0)
                if not model.kv_manager.contains(
                    context.request_id, replica_idx=replica_idx
                ):
                    model.kv_manager.claim(
                        context.request_id, replica_idx=replica_idx
                    )
                self._draft_replica_idx[context.request_id] = replica_idx
            # For target model, scheduler handles claiming via batch_constructor
        return num_steps

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
        model: PipelineModel[TextContext],
        batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        num_steps: int,
        return_n_logits: int,
        hidden_states: list[Buffer],
        shift_next_tokens: npt.NDArray[np.int64] | None = None,
    ) -> tuple[ModelInputs, int]:
        """Prepare batch for draft model execution.

        Handles the complex KV cache index management and hidden states
        for the draft model when using EAGLE speculative decoding.

        Args:
            model: The draft pipeline model
            batch: List of text contexts to process
            replica_batches: List of per-replica batches for data parallelism
            num_steps: Number of draft steps to prepare
            return_n_logits: Number of logits to return
            hidden_states: Hidden states from target model
            shift_next_tokens: Tokens to append after dropping the first token

        Returns:
            Tuple of (ModelInputs for draft model, num_steps)
        """
        saved_positions = [context.tokens.processed_length for context in batch]

        # --- Seek to draft KV positions and allocate cache ---
        for context in batch:
            target = self._draft_kv_start_idx.get(context.request_id, 0)
            self._seek_processing_position(context, target)

        for replica_idx, replica_batch in enumerate(replica_batches):
            for ctx in replica_batch:
                model.kv_manager.alloc(
                    ctx, replica_idx=replica_idx, num_steps=num_steps
                )
        kv_cache_inputs = model.kv_manager.get_runtime_inputs(
            replica_batches, num_steps
        )

        # --- Seek to token input positions and build inputs ---
        for i, context in enumerate(batch):
            is_new = context.request_id not in self._draft_kv_start_idx
            self._draft_kv_start_idx.setdefault(context.request_id, 0)
            if is_new:
                if shift_next_tokens is None:
                    context.tokens.skip_processing(1)
            else:
                self._seek_processing_position(context, saved_positions[i])

        base_inputs = model.prepare_initial_token_inputs(
            replica_batches=replica_batches,
            kv_cache_inputs=KVCacheInputsSequence(
                kv_cache_inputs=kv_cache_inputs
            ),
            return_n_logits=return_n_logits,
        )

        # --- Apply token shifting and hidden states ---
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
            for i, replica_batch in enumerate(replica_batches):
                device_context_length = sum(
                    self._draft_kv_start_idx.get(ctx.request_id, 0)
                    + ctx.tokens.active_length
                    for ctx in replica_batch
                )
                base_inputs.batch_context_lengths[i] = Buffer.from_numpy(
                    np.array([device_context_length], dtype=np.int32)
                )

        # --- Update draft KV indices and restore positions ---
        for i, context in enumerate(batch):
            self._draft_kv_start_idx[context.request_id] = (
                self._draft_kv_start_idx.get(context.request_id, 0)
                + context.tokens.active_length
            )
            self._seek_processing_position(context, saved_positions[i])
            context.apply_processing_offset(0)

        return (base_inputs, num_steps)

    def _prepare_initial_target_step(
        self,
        model: PipelineModel[TextContext],
        batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        num_steps: int,
        return_n_logits: int,
    ) -> tuple[ModelInputs, int]:
        """Prepare batch for initial target model step.

        This is used for the first target model execution to generate
        the initial token and hidden states for EAGLE.

        Args:
            model: The target pipeline model
            batch: List of text contexts to process
            replica_batches: List of per-replica batches for data parallelism
            num_steps: Number of steps (will be overridden to 1)
            return_n_logits: Number of logits to return

        Returns:
            Tuple of (ModelInputs for target model, 1)
        """
        for replica_idx, replica_batch in enumerate(replica_batches):
            for ctx in replica_batch:
                model.kv_manager.alloc(
                    ctx, replica_idx=replica_idx, num_steps=num_steps
                )
        kv_cache_inputs = model.kv_manager.get_runtime_inputs(
            replica_batches, num_steps
        )

        inputs = model.prepare_initial_token_inputs(
            replica_batches=replica_batches,
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
        replica_batches: list[list[TextContext]],
        num_steps: int,
        return_n_logits: int,
        draft_inputs: ModelInputs,
        merged_tokens: Buffer | None,
        merged_offsets: Buffer | None,
        host_merged_offsets: Buffer | None = None,
        kv_cache_inputs: KVCacheInputs | Sequence[KVCacheInputs] | None = None,
    ) -> tuple[ModelInputs, int]:
        """Prepare batch for target model verification of draft tokens.

        Uses the target model's prepare_initial_token_inputs to construct
        proper inputs, then overrides tokens and offsets with merged values
        for verification.

        Args:
            model: The target pipeline model
            batch: List of text contexts to process
            replica_batches: List of per-replica batches for data parallelism
            num_steps: Number of steps to prepare
            return_n_logits: Number of logits to return
            draft_inputs: Draft model inputs (used for data_parallel_splits)
            merged_tokens: Merged draft and input tokens
            merged_offsets: Offsets for merged tokens
            host_merged_offsets: Host-side merged offsets for MTP
            kv_cache_inputs: Pre-computed KV cache inputs. When provided,
                skips KV alloc/get_runtime_inputs (used when those must run
                inside a different context than prepare_initial_token_inputs).

        Returns:
            Tuple of (ModelInputs for target model, num_steps)
        """
        if kv_cache_inputs is None:
            for replica_idx, replica_batch in enumerate(replica_batches):
                for ctx in replica_batch:
                    model.kv_manager.alloc(
                        ctx, replica_idx=replica_idx, num_steps=num_steps
                    )
            kv_cache_inputs = model.kv_manager.get_runtime_inputs(
                replica_batches, num_steps
            )

        kv_cache_updated_inputs: KVCacheInputs
        if isinstance(kv_cache_inputs, Sequence):
            kv_cache_updated_inputs = KVCacheInputsSequence(
                kv_cache_inputs=kv_cache_inputs,
            )
        else:
            assert isinstance(kv_cache_inputs, KVCacheInputs)
            kv_cache_updated_inputs = kv_cache_inputs

        inputs = model.prepare_initial_token_inputs(
            replica_batches=replica_batches,
            kv_cache_inputs=kv_cache_updated_inputs,
            return_n_logits=return_n_logits,
        )

        inputs.tokens = merged_tokens  # type: ignore[attr-defined]
        inputs.input_row_offsets = merged_offsets  # type: ignore[attr-defined]
        inputs.host_input_row_offsets = host_merged_offsets  # type: ignore[attr-defined]

        return (inputs, num_steps)

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
        draft_tokens: Buffer | None = None,
        merged_tokens: Buffer | None = None,
        merged_offsets: Buffer | None = None,
        hidden_states: list[Buffer] | None = None,
        host_merged_offsets: Buffer | None = None,
        kv_cache_inputs: KVCacheInputs | Sequence[KVCacheInputs] | None = None,
        shift_next_tokens: npt.NDArray[np.int64] | None = None,
    ) -> tuple[ModelInputs, int]:
        """Prepare batch for model execution.

        Routes to appropriate preparation method based on execution mode:
        - Draft model: prepares with hidden states from target
        - Initial target step: generates first token and hidden states
        - Verification step: merges and verifies draft tokens
        """
        num_steps = self._prepare_common_setup(
            model, batch, replica_batches, num_steps, is_draft
        )

        if is_draft:
            assert hidden_states is not None
            return self._prepare_draft_batch(
                model,
                batch,
                replica_batches,
                num_steps,
                return_n_logits,
                hidden_states,
                shift_next_tokens,
            )
        elif draft_inputs is None:
            return self._prepare_initial_target_step(
                model, batch, replica_batches, num_steps, return_n_logits
            )
        else:
            return self._prepare_verification_step(
                model,
                batch,
                replica_batches,
                num_steps,
                return_n_logits,
                draft_inputs,
                merged_tokens,
                merged_offsets,
                host_merged_offsets,
                kv_cache_inputs=kv_cache_inputs,
            )

    @traced
    def generate_draft_tokens(
        self,
        batch: list[TextContext],
        num_steps: int,
        model_inputs: ModelInputs,
    ) -> tuple[int, Buffer, Buffer, Buffer, Buffer | list[Buffer]]:
        """Generates draft tokens for the batch using the draft model."""
        # Create sampling parameters once for the entire batch
        top_k, max_k, temperature, top_p, min_top_p, seed = (
            self._create_sampling_parameters(batch, self.draft_devices[0])
        )

        generated_tokens = Buffer.zeros(
            (len(batch), 0), dtype=DType.int64, device=self.draft_devices[0]
        )

        generated_logits = Buffer.zeros(
            (len(batch), 0), dtype=DType.float32, device=self.draft_devices[0]
        )

        curr_step_inputs = model_inputs

        # num_steps first so that slice indexing is contiguous
        all_draft_logits = Buffer.zeros(
            (num_steps, len(batch), self.vocab_size),
            dtype=DType.float32,
            device=self.draft_devices[0],
        )

        for i in range(num_steps):
            model_outputs = self._draft_model.execute(
                model_inputs=curr_step_inputs
            )

            all_draft_logits[i, :, :].inplace_copy_from(model_outputs.logits)

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

            curr_step_inputs = self._draft_model.prepare_next_token_inputs(
                new_tokens, curr_step_inputs
            )
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
        replica_batches: list[list[TextContext]],
        num_draft_tokens_generated: int,
        draft_tokens: Buffer,
        draft_logits: Buffer,
        all_draft_logits: Buffer,
        data_parallel_splits_np: npt.NDArray[np.int64],
        merged_tokens: Buffer | None = None,
        merged_offsets: Buffer | None = None,
        host_merged_offsets: Buffer | None = None,
    ) -> tuple[
        npt.NDArray[np.integer[Any]],
        npt.NDArray[np.integer[Any]],
        npt.NDArray[np.integer[Any]] | None,
    ]:
        """Verifies draft tokens against the target model and returns accepted indices."""
        # KV alloc must happen inside reserve_token_space_for_batch so the
        # KV manager sees the expanded token count. prepare_initial_token_inputs
        # must happen outside because it accesses ctx.tokens.active which
        # would see a bumped range exceeding the underlying array capacity.
        with reserve_token_space_for_batch(
            context_batch, num_draft_tokens_generated
        ):
            for replica_idx, replica_batch in enumerate(replica_batches):
                for ctx in replica_batch:
                    self._target_model.kv_manager.alloc(
                        ctx, replica_idx=replica_idx, num_steps=1
                    )
            kv_cache_inputs = self._target_model.kv_manager.get_runtime_inputs(
                replica_batches, 1
            )

        target_inputs, _ = self.prepare_batch(
            self._target_model,
            context_batch,
            replica_batches,
            num_steps=1,
            draft_inputs=draft_inputs,
            return_n_logits=num_draft_tokens_generated + 1,
            is_draft=False,
            draft_tokens=draft_tokens,
            merged_tokens=merged_tokens,
            merged_offsets=merged_offsets,
            host_merged_offsets=host_merged_offsets,
            kv_cache_inputs=kv_cache_inputs,
        )

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
        assert target_outputs.logit_offsets is not None

        hs = target_outputs.hidden_states
        target_hidden_states: list[Buffer] = (
            hs if isinstance(hs, list) else [hs]
        )

        self._hidden_state_manager.save_extracted(
            context_batch,
            target_hidden_states,
            target_outputs.logit_offsets.to_numpy(),
            first_rejected_tokens_np,
            data_parallel_splits_np,
        )

        return first_rejected_tokens_np, recovered_tokens_np, bonus_tokens_np

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
                    self._last_verified_token[context.request_id] = token
                elif bonus_tokens is not None:
                    # All drafts accepted + bonus token available
                    token = int(bonus_tokens[idx, 0])
                    total_bonus_used += 1
                    context.update(token)
                    self._last_verified_token[context.request_id] = token
                else:
                    # All drafts accepted, no bonus (greedy MTP) - use last draft token
                    # as the last verified token for subsequent iterations
                    self._last_verified_token[context.request_id] = int(
                        draft_tokens[idx, num_draft_tokens_generated - 1]
                    )

            # This is added because the draft needs to process the same tokens but with the hidden states received from the target model. This will also set the start index to the correct position for the kv cache
            context.apply_processing_offset(-rejected_token_idx)

            # Cap draft_kv_start_idx to processed_length so stale draft KV
            # entries (from rejected tokens) get overwritten on the next
            # iteration.
            if context.request_id in self._draft_kv_start_idx:
                self._draft_kv_start_idx[context.request_id] = min(
                    self._draft_kv_start_idx[context.request_id],
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

    def _target_forward(
        self,
        context_batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        data_parallel_splits_np: npt.NDArray[np.int64],
    ) -> tuple[ModelOutputs, Buffer, npt.NDArray[np.integer]]:
        target_inputs, _ = self.prepare_batch(
            self._target_model,
            context_batch,
            num_steps=1,  # Only generate 1 token initially
            return_n_logits=1,
            is_draft=False,
            draft_inputs=None,
            replica_batches=replica_batches,
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

        return target_outputs, target_sampled_tokens, target_sampled_tokens_np

    def _target_extend(
        self,
        context_batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        data_parallel_splits_np: npt.NDArray[np.int64],
    ) -> tuple[ModelOutputs, Buffer]:
        target_outputs, target_sampled_tokens, target_sampled_tokens_np = (
            self._target_forward(
                context_batch, replica_batches, data_parallel_splits_np
            )
        )

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
        context_batch = inputs.flat_batch
        replica_batches = inputs.batches
        data_parallel_splits_np = compute_data_parallel_splits(replica_batches)

        # If any request is in prefill (generated_length == 0), route entire batch
        # through context encoding path
        has_prefill = any(
            ctx.tokens.generated_length == 0 for ctx in context_batch
        )

        if has_prefill:
            return self._execute_context_encoding(
                context_batch, replica_batches, data_parallel_splits_np
            )
        else:
            return self._execute_token_generation(
                context_batch, replica_batches, data_parallel_splits_np
            )

    def _execute_context_encoding(
        self,
        context_batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        data_parallel_splits_np: npt.NDArray[np.int64],
    ) -> dict[RequestID, TextGenerationOutput]:
        target_outputs, _target_sampled_tokens, target_sampled_tokens_np = (
            self._target_forward(
                context_batch, replica_batches, data_parallel_splits_np
            )
        )

        assert target_outputs.hidden_states is not None
        hs = target_outputs.hidden_states
        ce_hs: list[Buffer] = hs if isinstance(hs, list) else [hs]
        next_tokens_for_shift: npt.NDArray[np.int64] = (
            target_sampled_tokens_np.flatten().astype(np.int64)
        )

        draft_ce_inputs, _ = self.prepare_batch(
            self._draft_model,
            context_batch,
            replica_batches,
            self._num_draft_steps,
            return_n_logits=1,
            is_draft=True,
            hidden_states=ce_hs,
            shift_next_tokens=next_tokens_for_shift,
        )
        draft_outputs = self._draft_model.execute(model_inputs=draft_ce_inputs)

        assert isinstance(
            draft_ce_inputs.kv_cache_inputs, KVCacheInputsSequence
        )
        assert isinstance(draft_ce_inputs.kv_cache_inputs.kv_cache_inputs, list)
        draft_ce_inputs.kv_cache_inputs.kv_cache_inputs = (
            self._draft_model.kv_manager.increment_cache_lengths(
                draft_ce_inputs.kv_cache_inputs.kv_cache_inputs,
                draft_ce_inputs,
            )
        )

        assert draft_outputs.hidden_states is not None
        draft_hs_raw = draft_outputs.hidden_states
        draft_hs: list[Buffer] = (
            draft_hs_raw if isinstance(draft_hs_raw, list) else [draft_hs_raw]
        )
        self._hidden_state_manager.save_prefill_hidden_states(
            context_batch,
            draft_hs,
            data_parallel_splits_np,
        )

        for i, ctx in enumerate(context_batch):
            token = int(target_sampled_tokens_np[i].item())
            if not ctx.tokens.actively_chunked:
                self._last_verified_token[ctx.request_id] = token
            ctx.update(token)

        return self.build_response(context_batch=context_batch)

    def _execute_token_generation(
        self,
        context_batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        data_parallel_splits_np: npt.NDArray[np.int64],
    ) -> dict[RequestID, TextGenerationOutput]:
        draft_hidden_states = self._hidden_state_manager.get_draft_input(
            replica_batches
        )

        draft_inputs, draft_num_steps = self.prepare_batch(
            self._draft_model,
            context_batch,
            replica_batches,
            self._num_draft_steps,
            return_n_logits=1,
            is_draft=True,
            hidden_states=draft_hidden_states,
        )

        (
            num_draft_tokens_generated,
            draft_tokens,
            draft_logits,
            all_draft_logits,
            _,
        ) = self.generate_draft_tokens(
            context_batch, draft_num_steps, draft_inputs
        )

        last_tokens = np.array(
            [
                self._last_verified_token[context.request_id]
                for context in context_batch
            ],
            dtype=np.int64,
        )
        draft_input_tokens = Buffer.from_numpy(last_tokens).to(
            self.target_devices[0]
        )

        return self._verify_and_finalize(
            draft_inputs=draft_inputs,
            num_draft_tokens_generated=num_draft_tokens_generated,
            draft_tokens=draft_tokens,
            draft_logits=draft_logits,
            all_draft_logits=all_draft_logits,
            draft_input_tokens=draft_input_tokens,
            context_batch=context_batch,
            replica_batches=replica_batches,
            data_parallel_splits_np=data_parallel_splits_np,
        )

    def _verify_and_finalize(
        self,
        draft_inputs: ModelInputs,
        num_draft_tokens_generated: int,
        draft_tokens: Buffer,
        draft_logits: Buffer,
        all_draft_logits: Buffer,
        draft_input_tokens: Buffer,
        context_batch: list[TextContext],
        replica_batches: list[list[TextContext]],
        data_parallel_splits_np: npt.NDArray[np.int64],
    ) -> dict[RequestID, TextGenerationOutput]:
        draft_input_offsets_np = np.cumsum(
            [0] + [1 for _ in context_batch],
            dtype=np.uint32,
        )
        draft_input_offsets = Buffer.from_numpy(draft_input_offsets_np).to(
            self.target_devices[0]
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

        first_rejected_tokens, recovered_tokens, bonus_tokens = (
            self.verify_draft_tokens_with_target_model(
                draft_inputs,
                context_batch,
                replica_batches,
                num_draft_tokens_generated,
                draft_tokens,
                draft_logits,
                all_draft_logits,
                data_parallel_splits_np,
                merged_tokens,
                merged_offsets,
                host_merged_offsets,
            )
        )

        self.update_contexts(
            context_batch=context_batch,
            first_rejected_tokens=first_rejected_tokens,
            recovered_tokens=recovered_tokens,
            bonus_tokens=bonus_tokens,
            draft_tokens=draft_tokens.to_numpy(),
            num_draft_tokens_generated=num_draft_tokens_generated,
            data_parallel_splits=data_parallel_splits_np,
        )

        res = self.build_response(context_batch=context_batch)

        self._target_model.kv_manager.step(replica_batches)
        self._draft_model.kv_manager.step(replica_batches)

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
        self._hidden_state_manager.release(request_id)
