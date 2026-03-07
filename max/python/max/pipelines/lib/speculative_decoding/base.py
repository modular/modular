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
from abc import ABC
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, load_devices
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.interfaces import (
    GenerationStatus,
    PipelineTokenizer,
    RequestID,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.kv_cache import PagedKVCacheManager
from max.kv_cache.registry import load_multi_kv_managers
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.profiler import traced
from transformers import AutoConfig

from ..interfaces import ModelOutputs, PipelineModel, PipelineModelWithKVCache
from ..pipeline_variants.text_generation import (
    TextGenerationPipelineInterface,
    calculate_num_steps,
    get_eos_tokens,
    get_weight_paths,
)
from ..sampling import (
    PenaltyInputs,
    SamplerInputs,
    SamplingConfig,
    greedy_acceptance_sampler,
    rejection_sampler,
    rejection_sampler_with_residuals,
    token_sampler,
    typical_acceptance_sampler,
)
from ..utils import upper_bounded_default
from .ragged_token_merger import ragged_token_merger

if TYPE_CHECKING:
    from ..config import MAXModelConfig, PipelineConfig, SpeculativeConfig

logger = logging.getLogger("max.pipelines")


class SpeculativeDecodingMetrics:
    """Metrics tracker for speculative decoding performance."""

    def __init__(self) -> None:
        """Initialize metrics counters."""
        self.bonus_tokens_used = 0
        self.draft_tokens_accepted = 0
        self.draft_tokens_generated = 0
        self.total_acceptance_lengths = 0
        self.num_generations = 0

    def update(
        self,
        draft_tokens_generated: int,
        draft_tokens_accepted: int,
        bonus_tokens_used: int,
        acceptance_lengths: list[int],
    ) -> None:
        """Update metrics with results from a batch.

        Args:
            draft_tokens_generated: Total draft tokens generated in this batch
            draft_tokens_accepted: Total draft tokens accepted in this batch
            bonus_tokens_used: Number of bonus tokens used in this batch
            acceptance_lengths: List of acceptance lengths for each sequence in batch
        """
        self.draft_tokens_generated += draft_tokens_generated
        self.draft_tokens_accepted += draft_tokens_accepted
        self.bonus_tokens_used += bonus_tokens_used
        self.total_acceptance_lengths += sum(acceptance_lengths)
        self.num_generations += len(acceptance_lengths)

    def get_stats(self) -> dict[str, float]:
        """Get current statistics.

        Returns:
            Dictionary with acceptance rate and total counts
        """
        if self.draft_tokens_generated == 0:
            return {
                "acceptance_rate": 0.0,
                "bonus_tokens_used": 0,
                "draft_tokens_accepted": 0,
                "draft_tokens_generated": 0,
                "avg_acceptance_length": 0.0,
            }

        return {
            "acceptance_rate": self.draft_tokens_accepted
            / self.draft_tokens_generated,
            "bonus_tokens_used": self.bonus_tokens_used,
            "draft_tokens_accepted": self.draft_tokens_accepted,
            "draft_tokens_generated": self.draft_tokens_generated,
            "avg_acceptance_length": self.total_acceptance_lengths
            / self.num_generations
            if self.num_generations > 0
            else 0.0,
        }

    def __str__(self) -> str:
        """String representation of current metrics."""
        stats = self.get_stats()
        return (
            f"SpeculativeDecodingMetrics("
            f"acceptance_rate={stats['acceptance_rate']:.2%}, "
            f"avg_acceptance_length={stats['avg_acceptance_length']:.2f}, "
            f"bonus_tokens_used={stats['bonus_tokens_used']}, "
            f"draft_tokens_accepted={stats['draft_tokens_accepted']}/{stats['draft_tokens_generated']})"
        )


def hidden_states_return_config(
    pipeline_config: PipelineConfig, is_draft: bool = False
) -> ReturnHiddenStates:
    """Return the hidden states return config for the speculative config.

    For Eagle and DeepSeek MTP, we share the embedding and lm_head weights between the target and draft models and only take the last hidden state from the target model.

    """
    assert pipeline_config.speculative is not None
    is_eagle_or_mtp = (
        pipeline_config.speculative.is_eagle()
        or pipeline_config.speculative.is_mtp()
    )
    if is_eagle_or_mtp:
        if is_draft:
            return ReturnHiddenStates.LAST
        else:
            return ReturnHiddenStates.ALL_NORMALIZED

    else:
        return ReturnHiddenStates.NONE


def get_vocab_size(huggingface_config: AutoConfig) -> int:
    """Get the vocab size from the HuggingFace config."""
    if hasattr(huggingface_config, "vocab_size"):
        return huggingface_config.vocab_size
    elif hasattr(huggingface_config, "text_config") and hasattr(
        huggingface_config.text_config, "vocab_size"
    ):
        return huggingface_config.text_config.vocab_size
    else:
        raise ValueError(
            "MAXModelConfig's HuggingFace config must have a 'vocab_size' or 'text_config.vocab_size' param for Speculative Decoding"
        )


class RejectionRunner(Protocol):
    """Interface for rejection sampling runners."""

    def run(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer | None,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer | None,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, Buffer, Buffer | None]:
        """Run the rejection sampler."""
        ...


class _TypicalAcceptanceRunner(RejectionRunner):
    """Routes per-batch: temp=0 uses greedy (argmax), temp>0 uses stochastic."""

    def __init__(
        self,
        greedy_model: Any,
        stochastic_model: Any,
        target_device: Device,
    ) -> None:
        self._greedy = greedy_model
        self._stochastic = stochastic_model
        self._device = target_device

    def run(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer | None,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer | None,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, Buffer, Buffer]:
        temps = [ctx.sampling_params.temperature for ctx in context_batch]
        all_greedy = all(t == 0 for t in temps)
        if all_greedy:
            a, b, c = self._greedy(
                draft_tokens,
                target_logits,
                target_logit_offsets,
            )
        else:
            top_k_np = np.array(
                [ctx.sampling_params.top_k for ctx in context_batch],
                dtype=np.int64,
            )
            top_p_np = np.array(
                [ctx.sampling_params.top_p for ctx in context_batch],
                dtype=np.float32,
            )
            temps_np = np.array(temps, dtype=np.float32)
            np.clip(temps_np, a_min=1e-6, a_max=None, out=temps_np)
            a, b, c = self._stochastic(
                draft_tokens,
                target_logits,
                target_logit_offsets,
                Buffer.from_numpy(temps_np).to(self._device),
                Buffer.from_numpy(top_k_np).to(self._device),
                Buffer.from_numpy(np.array(np.max(top_k_np), dtype=np.int64)),
                Buffer.from_numpy(top_p_np).to(self._device),
                Buffer.from_numpy(np.array(np.min(top_p_np), dtype=np.float32)),
            )
        assert isinstance(a, Buffer)
        assert isinstance(b, Buffer)
        assert isinstance(c, Buffer)
        return a, b, c


class _GreedyRunner(RejectionRunner):
    """Always argmax acceptance. No draft logits needed."""

    def __init__(self, model: Model) -> None:
        self._model = model

    def run(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer | None,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer | None,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, Buffer, Buffer]:
        a, b, c = self._model(
            draft_tokens,
            target_logits,
            target_logit_offsets,
        )
        assert isinstance(a, Buffer)
        assert isinstance(b, Buffer)
        assert isinstance(c, Buffer)
        return a, b, c


class _LogitComparisonRunner(RejectionRunner):
    """draft_logit <= target_logit + eps. No bonus token (returns None)."""

    def __init__(self, model: Model) -> None:
        self._model = model

    def run(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer | None,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer | None,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, Buffer, None]:
        assert all_draft_logits is not None
        assert draft_logits is not None
        a, b = self._model(
            draft_tokens,
            draft_logits,
            target_logits,
            target_logit_offsets,
        )
        assert isinstance(a, Buffer)
        assert isinstance(b, Buffer)
        return a, b, None


class _ResidualRunner(RejectionRunner):
    """p_target/p_draft ratio acceptance. Needs all_draft_logits."""

    def __init__(self, model: Model) -> None:
        self._model = model

    def run(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer | None,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer | None,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, Buffer, Buffer]:
        assert draft_logits is not None
        assert all_draft_logits is not None
        a, b, c = self._model(
            draft_tokens,
            draft_logits,
            target_logits,
            target_logit_offsets,
            all_draft_logits,
        )
        assert isinstance(a, Buffer)
        assert isinstance(b, Buffer)
        assert isinstance(c, Buffer)
        return a, b, c


def _build_rejection_runner(
    strategy: str,
    session: InferenceSession,
    device_ref: DeviceRef,
) -> RejectionRunner:
    """Loads the compiled rejection sampler graph and returns a runner."""
    if strategy == "typical-acceptance":
        return _TypicalAcceptanceRunner(
            greedy_model=session.load(
                greedy_acceptance_sampler(device=device_ref)
            ),
            stochastic_model=session.load(
                typical_acceptance_sampler(device=device_ref)
            ),
            target_device=device_ref.to_device(),
        )
    if strategy == "greedy":
        return _GreedyRunner(
            session.load(greedy_acceptance_sampler(device=device_ref))
        )
    if strategy == "logit-comparison":
        return _LogitComparisonRunner(
            session.load(rejection_sampler(device=device_ref))
        )
    else:
        return _ResidualRunner(
            session.load(rejection_sampler_with_residuals(device=device_ref))
        )


def compute_max_num_draft_steps(
    replica_batches: list[list[TextContext]],
    desired_num_draft_steps: int,
    max_seq_len: int,
    is_draft: bool,
) -> int:
    """Compute the maximum number of draft steps that can be run for a batch."""
    max_seq_len = max_seq_len - 1 if is_draft else max_seq_len
    num_draft_steps = desired_num_draft_steps
    for replica_batch in replica_batches:
        for context in replica_batch:
            num_draft_steps = calculate_num_steps(
                context, num_draft_steps, max_seq_len
            )
    return num_draft_steps


class SpeculativeDecodingPipelineBase(
    TextGenerationPipelineInterface[TextContext],
    ABC,
):
    """Base class for speculative decoding pipelines with shared logic."""

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
        self._pipeline_config = pipeline_config
        self._tokenizer = tokenizer

        target_config: MAXModelConfig = self.pipeline_config.model
        if self.pipeline_config.draft_model is None:
            raise ValueError(
                "Draft model must be provided for speculative decoding"
            )
        draft_config: MAXModelConfig = self.pipeline_config.draft_model

        if target_config.device_specs != draft_config.device_specs:
            raise ValueError(
                "Target and draft model must have the same device specs"
            )
        target_hf_config = target_config.huggingface_config
        assert target_hf_config is not None
        draft_hf_config = draft_config.huggingface_config
        assert draft_hf_config is not None

        self._eos_token_id = get_eos_tokens(target_hf_config, eos_token_id)
        self.vocab_size = get_vocab_size(target_hf_config)
        if not self.pipeline_config.speculative:
            raise ValueError(
                "Speculative config must be provided for speculative decoding"
            )
        self._speculative_config: SpeculativeConfig = (
            self.pipeline_config.speculative
        )

        if not target_config.quantization_encoding:
            raise ValueError("quantization_encoding must not be None")

        # Use draft model's pipeline model and weight adapters if provided
        # Otherwise fall back to target model's (for backward compatibility)
        actual_draft_pipeline_model = (
            draft_pipeline_model
            if draft_pipeline_model is not None
            else pipeline_model
        )
        actual_draft_weight_adapters = (
            draft_weight_adapters
            if draft_weight_adapters is not None
            else weight_adapters
        )

        if not (
            issubclass(pipeline_model, PipelineModelWithKVCache)
            and issubclass(
                actual_draft_pipeline_model, PipelineModelWithKVCache
            )
        ):
            raise ValueError(
                f"Speculative decoding requires both the target and draft models to support KV cache, found {pipeline_model.__name__} and {actual_draft_pipeline_model.__name__}"
            )

        device_specs = target_config.device_specs
        self.devices = load_devices(device_specs)
        device_refs = [DeviceRef.from_device(dev) for dev in self.devices]
        self._session = InferenceSession(devices=[*self.devices])
        self.pipeline_config.configure_session(self._session)

        weight_paths = get_weight_paths(target_config)
        self._target_model = pipeline_model(
            pipeline_config=self.pipeline_config,
            session=self._session,
            devices=self.devices,
            kv_cache_config=target_config.kv_cache,
            weights=load_weights(weight_paths),
            adapter=weight_adapters.get(weights_format(weight_paths)),
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=hidden_states_return_config(
                self.pipeline_config, is_draft=False
            ),
        )

        from max.pipelines.architectures.deepseekV3_nextn.model import (  # type: ignore[import-not-found]
            maybe_build_deepseekv3_nextn_kwargs,
        )

        draft_kwargs = maybe_build_deepseekv3_nextn_kwargs(
            self._target_model,
            actual_draft_pipeline_model,
        )
        draft_weight_paths = get_weight_paths(draft_config)
        self._draft_model = actual_draft_pipeline_model(
            pipeline_config=self.pipeline_config,
            session=self._session,
            devices=self.devices,
            kv_cache_config=draft_config.kv_cache,
            weights=load_weights(draft_weight_paths),
            adapter=actual_draft_weight_adapters.get(
                weights_format(draft_weight_paths)
            ),
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=hidden_states_return_config(
                self.pipeline_config, is_draft=True
            ),
            **draft_kwargs,
        )

        # Load draft sampler
        draft_sampling_config: SamplingConfig = self.pipeline_config.sampling
        draft_sampling_config.enable_variable_logits = False
        self._draft_sampler = self._session.load(
            token_sampler(
                draft_sampling_config,
                return_logits=True,
                device=device_refs[0],
            )
        )

        strategy = self._speculative_config.rejection_sampling_strategy
        is_eagle_or_mtp = (
            self._speculative_config.is_eagle()
            or self._speculative_config.is_mtp()
        )
        if strategy is None:
            if is_eagle_or_mtp:
                strategy = "typical-acceptance"
            else:
                strategy = "residual"
        if strategy == "residual" and is_eagle_or_mtp:
            raise ValueError(
                "EAGLE/MTP speculative decoding does not support 'residual'"
                " rejection sampling strategy. Use 'greedy',"
                " 'typical-acceptance', or 'logit-comparison' instead."
            )
        logger.info(f"Using '{strategy}' rejection sampling strategy")
        self._rejection_runner: RejectionRunner = _build_rejection_runner(
            strategy, self._session, device_refs[0]
        )
        self._needs_all_draft_logits = strategy == "residual"

        # Initialize metrics tracker
        self._metrics = SpeculativeDecodingMetrics()

        # Track draft model replica assignments per request
        self._draft_replica_idx: dict[RequestID, int] = {}

        # Check that the max length for both models are the same
        draft_seq_len = self._draft_model.calculate_max_seq_len(
            self.pipeline_config, draft_hf_config
        )
        target_seq_len = self._target_model.calculate_max_seq_len(
            self.pipeline_config, target_hf_config
        )
        if draft_seq_len != target_seq_len:
            raise ValueError(
                f"draft maximum sequence length ({draft_seq_len}) must match target maximum sequence length."
            )
        self._max_seq_len = target_seq_len

        self._ragged_token_merger = self._session.load(
            ragged_token_merger(device=device_refs[0])
        )

        self._num_draft_steps = self._speculative_config.num_speculative_tokens

        target_cache_mem = target_config.kv_cache._available_cache_memory
        draft_cache_mem = draft_config.kv_cache._available_cache_memory
        # These should have been set during memory estimation.
        assert draft_cache_mem is not None
        assert target_cache_mem is not None
        cache_mem = target_cache_mem + draft_cache_mem

        target_cache_mem = (
            self._target_model.kv_cache_config._available_cache_memory
        )
        draft_cache_mem = (
            self._draft_model.kv_cache_config._available_cache_memory
        )
        # These should have been set during memory estimation.
        assert draft_cache_mem is not None
        assert target_cache_mem is not None
        cache_mem = target_cache_mem + draft_cache_mem

        target_kv_params = self._target_model.kv_params
        assert isinstance(target_kv_params, KVCacheParams)
        draft_kv_params = self._draft_model.kv_params
        assert isinstance(draft_kv_params, KVCacheParams)
        multi_kv_params = MultiKVCacheParams.from_params(
            target_kv_params, draft_kv_params
        )

        self._target_kv_manager, draft_kv_manager = load_multi_kv_managers(
            params=multi_kv_params,
            max_batch_size=pipeline_config.runtime.max_batch_size,
            max_seq_len=self._max_seq_len,
            session=self._session,
            available_cache_memory=cache_mem,
        )

        # We employ a crazy hack here where we completely disregard the draft
        # kv cache manager. We save the draft kv cache buffer and discard the rest
        # of the object. Now whenever we want to use the draft runtime inputs,
        # we use the target kv cache manager and secretly swap out the host kv
        # cache buffers for that of the draft model.
        # TODO: do this properly once the kv cache manager is refactored to support
        # spec decoding.
        draft_kv_inputs = draft_kv_manager.runtime_inputs(
            [[] * multi_kv_params.data_parallel_degree]
        )
        self._draft_kv_buffers = [
            replica_input.blocks for replica_input in draft_kv_inputs.inputs
        ]

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Returns the pipeline configuration."""
        return self._pipeline_config

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[
        TextContext,
        npt.NDArray[np.integer[Any]],
        TextGenerationRequest,
    ]:
        """Returns the tokenizer for this speculative pipeline."""
        return self._tokenizer

    @traced
    def sample_draft_logits(
        self,
        model_outputs: ModelOutputs,
        prev_tokens: Buffer,
        prev_logits: Buffer,
        sampler_inputs: SamplerInputs,
        penalty_inputs: PenaltyInputs | None = None,
    ) -> tuple[Buffer, Buffer, Buffer]:
        """Samples draft tokens from the draft model logits."""
        graph_inputs: list[Buffer] = [
            model_outputs.logits,
            prev_tokens,
            *sampler_inputs.as_list(),
            prev_logits,
        ]
        if penalty_inputs is not None:
            graph_inputs.extend(penalty_inputs.as_list())
        a, b, c = self._draft_sampler(*graph_inputs)[:3]
        assert isinstance(a, Buffer)
        assert isinstance(b, Buffer)
        assert isinstance(c, Buffer)
        return (a, b, c)

    @property
    def metrics(self) -> SpeculativeDecodingMetrics:
        """Get the current speculative decoding metrics.

        Returns:
            The SpeculativeDecodingMetrics instance with current statistics
        """
        return self._metrics

    def __del__(self) -> None:
        """Log metrics when the pipeline is destroyed."""
        if (
            hasattr(self, "_metrics")
            and self._metrics.draft_tokens_generated > 0
        ):
            logger.info(f"Speculative decoding metrics: {self._metrics}")

    def update_contexts(
        self,
        context_batch: list[TextContext],
        first_rejected_tokens: npt.NDArray[np.integer[Any]],
        recovered_tokens: npt.NDArray[np.integer[Any]],
        bonus_tokens: npt.NDArray[np.integer[Any]] | None,
        draft_tokens: npt.NDArray[np.integer[Any]],
        num_draft_tokens_generated: int,
    ) -> None:
        """Update contexts with the results of token generation.

        Args:
            context_batch: The list of context objects
            first_rejected_tokens: Array indicating the indices of first rejected tokens
            recovered_tokens: Array of recovered tokens from target model
            bonus_tokens: Array of bonus tokens from target model
            draft_tokens: Array of draft tokens
            num_draft_tokens_generated: Number of tokens generated by the draft model
        """
        total_draft_generated = num_draft_tokens_generated * len(context_batch)
        total_draft_accepted = 0
        total_bonus_used = 0
        acceptance_lengths = []

        for idx, rejected_token_idx in enumerate(first_rejected_tokens):
            context = context_batch[idx]
            rejected_token_idx = rejected_token_idx.item()

            for token_idx in range(rejected_token_idx):
                token = int(draft_tokens[idx, token_idx])
                context.update(token)

            if (
                rejected_token_idx == num_draft_tokens_generated
                and bonus_tokens is not None
            ):
                context.update(bonus_tokens[idx, 0].item())
                total_bonus_used += 1
            else:
                # For residual sampler, index by rejected position;
                # for greedy sampler (or no bonus), always index 0.
                recover_idx = (
                    rejected_token_idx if bonus_tokens is not None else 0
                )
                context.update(recovered_tokens[idx, recover_idx].item())

            total_draft_accepted += rejected_token_idx
            acceptance_lengths.append(rejected_token_idx)

            # When some or all draft tokens are rejected, we apply a token from
            # the residual distribution. The draft and target models have not
            # processed this token so the context goes back one step for both
            # of the models to process that token.
            # If all draft tokens are accepted, then the draft model has not
            # processed the bonus token. In this case only the draft needs to
            # go one step back. At the moment we do this for all cases.
            context.tokens.rewind_processing(1)

        # Update metrics
        self._metrics.update(
            total_draft_generated,
            total_draft_accepted,
            total_bonus_used,
            acceptance_lengths,
        )

    def build_response(
        self, context_batch: list[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        """Build response from updated contexts.

        Args:
            context_batch: The list of context objects

        Returns:
            Dictionary mapping request IDs to TextGenerationOutput objects
        """
        res: dict[RequestID, TextGenerationOutput] = {}

        for context in context_batch:
            # Identify the Max Length
            context_max_length = upper_bounded_default(
                upper_bound=self._max_seq_len, default=context.max_length
            )

            # Break early if beyond max length
            current_length = context.tokens.processed_length + 1
            if current_length >= context_max_length:
                context.status = GenerationStatus.MAXIMUM_LENGTH

            output = context.to_generation_output()
            if output.tokens:
                res[context.request_id] = output

        return res

    @traced
    def release(self, request_id: RequestID) -> None:
        """Release the state for the request."""
        pass

    @property
    def kv_manager(self) -> PagedKVCacheManager:
        """Returns the target model KV cache manager."""
        return self._target_kv_manager
