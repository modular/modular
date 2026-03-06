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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device, DLPackArray, load_devices
from max.engine import InferenceSession
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
from max.kv_cache import PagedKVCacheManager, load_kv_manager
from max.nn.kv_cache import KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.profiler import traced
from transformers import AutoConfig

from ..interfaces import ModelOutputs, PipelineModel, PipelineModelWithKVCache
from ..pipeline_variants.text_generation import (
    TextGenerationPipelineInterface,
    get_eos_tokens,
    get_weight_paths,
)
from ..sampling import (
    greedy_acceptance_sampler,
    rejection_sampler,
    rejection_sampler_with_residuals,
    token_sampler,
    typical_acceptance_sampler,
)
from ..sampling.sampling_logits_processor import (
    FrequencyData,
    _build_token_frequency_csr,
    _check_need_penalties,
)
from ..utils import upper_bounded_default
from .ragged_token_merger import ragged_token_merger

if TYPE_CHECKING:
    from ..config import PipelineConfig

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


class _TypicalAcceptanceRunner:
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

    def __call__(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer,
        context_batch: list[TextContext] | None,
    ) -> tuple[Buffer, Buffer, Buffer]:
        assert context_batch is not None
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


class _GreedyRunner:
    """Always argmax acceptance. No draft logits needed."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer,
        context_batch: list[TextContext] | None,
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


class _LogitComparisonRunner:
    """draft_logit <= target_logit + eps. No bonus token (returns None)."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer,
        context_batch: list[TextContext] | None,
    ) -> tuple[Buffer, Buffer, None]:
        a, b = self._model(
            draft_tokens,
            draft_logits,
            target_logits,
            target_logit_offsets,
        )
        assert isinstance(a, Buffer)
        assert isinstance(b, Buffer)
        return a, b, None


class _ResidualRunner:
    """p_target/p_draft ratio acceptance. Needs all_draft_logits."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer,
        context_batch: list[TextContext] | None,
    ) -> tuple[Buffer, Buffer, Buffer]:
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

        # Load target model
        self.target_devices = load_devices(
            self.pipeline_config.model.device_specs
        )
        target_config = self.pipeline_config.model.huggingface_config
        target_session = InferenceSession(devices=[*self.target_devices])
        self.pipeline_config.configure_session(target_session)
        target_config = AutoConfig.from_pretrained(
            self.pipeline_config.model.model_path,
            trust_remote_code=self.pipeline_config.model.trust_remote_code,
            revision=self.pipeline_config.model.huggingface_model_revision,
        )

        self._eos_token_id = get_eos_tokens(target_config, eos_token_id)

        weight_paths = get_weight_paths(self.pipeline_config.model)

        target_weights = load_weights(weight_paths)
        _target_weights_format = weights_format(weight_paths)

        if not self.pipeline_config.model.quantization_encoding:
            raise ValueError(
                f"quantization_encoding must be provided, {self.pipeline_config.model.quantization_encoding}"
            )

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

        self._target_model = pipeline_model(
            pipeline_config=self.pipeline_config,
            session=target_session,
            devices=self.target_devices,
            kv_cache_config=self.pipeline_config.model.kv_cache,
            weights=target_weights,
            adapter=weight_adapters.get(_target_weights_format),
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=hidden_states_return_config(
                self.pipeline_config, is_draft=False
            ),
        )

        # Validate that target model has HuggingFace config
        target_hf_config = self.pipeline_config.model.huggingface_config
        if target_hf_config is None:
            raise ValueError(
                f"Speculative decoding requires a HuggingFace config for the target model, "
                f"but could not load config for '{self.pipeline_config.model.model_path}'. "
                "Please ensure the target model is a standard Transformers model with a valid config.json."
            )

        # Calculate Max Length
        self._max_length = self._target_model.calculate_max_seq_len(
            self.pipeline_config,
            huggingface_config=target_hf_config,
        )

        # Load draft model
        assert self.pipeline_config.draft_model is not None
        self.draft_devices = load_devices(
            self.pipeline_config.draft_model.device_specs
        )
        draft_session = InferenceSession(devices=[*self.draft_devices])
        self.pipeline_config.configure_session(draft_session)

        if self.pipeline_config.draft_model is None:
            raise ValueError("Draft model is required for speculative decoding")
        draft_config = self.pipeline_config.draft_model.huggingface_config
        if draft_config is None:
            raise ValueError(
                f"Speculative decoding requires a HuggingFace config for the draft model, "
                f"but could not load config for '{self.pipeline_config.draft_model.model_path}'. "
                "Please ensure the draft model is a standard Transformers model with a valid config.json."
            )

        self.vocab_size = get_vocab_size(target_hf_config)

        # Retrieve Encoding, and Files for Draft Model
        if self.pipeline_config.draft_model is None:
            raise ValueError(
                "draft_model must be provided for speculative decoding"
            )

        # Use already-resolved weight paths from draft_model
        draft_weight_paths = get_weight_paths(self.pipeline_config.draft_model)

        draft_weights = load_weights(draft_weight_paths)
        _draft_weights_format = weights_format(draft_weight_paths)
        assert self.pipeline_config.speculative is not None
        self._speculative_config = self.pipeline_config.speculative

        shared_weights: dict[str, DLPackArray] | None = None
        shared_ep_comm_initializer = None
        if self.pipeline_config.speculative is not None and (
            self.pipeline_config.speculative.is_eagle()
            or self.pipeline_config.speculative.is_mtp()
        ):
            shared_weights, shared_ep_comm_initializer = (
                self._maybe_build_deepseekv3_nextn_shared_resources(
                    actual_draft_pipeline_model
                )
            )

        draft_model_kwargs: dict[str, Any] = {}
        if shared_weights and getattr(
            actual_draft_pipeline_model, "supports_shared_weights", False
        ):
            draft_model_kwargs["shared_weights"] = shared_weights
        elif shared_weights:
            logger.debug(
                "Draft model %s does not support shared weights; skipping",
                actual_draft_pipeline_model.__name__,
            )

        if shared_ep_comm_initializer is not None:
            draft_model_kwargs["shared_ep_comm_initializer"] = (
                shared_ep_comm_initializer
            )

        self._draft_model = actual_draft_pipeline_model(
            pipeline_config=self.pipeline_config,
            session=draft_session,
            devices=self.draft_devices,
            kv_cache_config=self.pipeline_config.draft_model.kv_cache,
            weights=draft_weights,
            adapter=actual_draft_weight_adapters.get(_draft_weights_format),
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=hidden_states_return_config(
                self.pipeline_config, is_draft=True
            ),
            **draft_model_kwargs,
        )

        # Load draft sampler
        draft_sampling_config = self.pipeline_config.sampling
        draft_sampling_config.enable_variable_logits = False
        self._draft_sampler = draft_session.load(
            token_sampler(
                draft_sampling_config,
                return_logits=True,
                device=DeviceRef.from_device(self.draft_devices[0]),
            )
        )

        strategy = self._speculative_config.rejection_sampling_strategy
        if strategy is None:
            if (
                self._speculative_config.is_eagle()
                or self._speculative_config.is_mtp()
            ):
                strategy = "typical-acceptance"
            else:
                strategy = "residual"
        logger.info(f"Using '{strategy}' rejection sampling strategy")
        target_device_ref = DeviceRef.from_device(self.target_devices[0])
        self._run_rejection_sampler = self._build_rejection_runner(
            strategy, target_session, target_device_ref
        )
        self._needs_all_draft_logits = strategy == "residual"

        # Initialize metrics tracker
        self._metrics = SpeculativeDecodingMetrics()

        # Track draft model replica assignments per request
        self._draft_replica_idx: dict[RequestID, int] = {}

        # Check that the max length for both models are the same
        draft_seq_len = self._draft_model.calculate_max_seq_len(
            self.pipeline_config, draft_config
        )
        target_seq_len = self._target_model.calculate_max_seq_len(
            self.pipeline_config, target_config
        )
        if draft_seq_len != target_seq_len:
            raise ValueError(
                f"draft maximum sequence length ({draft_seq_len}) must match target maximum sequence length."
            )

        self._ragged_token_merger = target_session.load(
            ragged_token_merger(
                device=DeviceRef.from_device(self.target_devices[0])
            )
        )

        self._draft_session = draft_session
        self._target_session = target_session

        self._num_draft_steps = (
            self.pipeline_config.speculative.num_speculative_tokens
        )

        target_kv_params = self._target_model.kv_params
        assert isinstance(target_kv_params, KVCacheParams)
        self._target_kv_manager: PagedKVCacheManager = load_kv_manager(
            params=target_kv_params,
            max_batch_size=pipeline_config.runtime.max_batch_size,
            max_seq_len=self._target_model.max_seq_len,
            session=self._target_session,
            available_cache_memory=self._target_model.kv_cache_config._available_cache_memory,
        )
        draft_kv_params = self._draft_model.kv_params
        assert isinstance(draft_kv_params, KVCacheParams)
        self._draft_kv_manager: PagedKVCacheManager = load_kv_manager(
            params=draft_kv_params,
            max_batch_size=pipeline_config.runtime.max_batch_size,
            max_seq_len=self._draft_model.max_seq_len,
            session=self._draft_session,
            available_cache_memory=self._draft_model.kv_cache_config._available_cache_memory,
        )

    def _maybe_build_deepseekv3_nextn_shared_resources(
        self,
        draft_model_cls: type[PipelineModel[TextContext]],
    ) -> tuple[dict[str, DLPackArray] | None, Any]:
        # Imported here to avoid circular imports
        from max.pipelines.architectures.deepseekV3.model import (  # type: ignore[import-not-found]
            DeepseekV3Model,
        )
        from max.pipelines.architectures.deepseekV3_nextn.model import (  # type: ignore[import-not-found]
            DeepseekV3NextNModel,
        )

        if not isinstance(self._target_model, DeepseekV3Model):
            return None, None
        if not issubclass(draft_model_cls, DeepseekV3NextNModel):
            return None, None

        # Share EP buffers between target and draft to avoid duplicating
        shared_ep_comm_initializer = self._target_model.ep_comm_initializer
        target_state_dict = getattr(self._target_model, "state_dict", None)
        if not isinstance(target_state_dict, dict):
            raise ValueError(
                "Target DeepseekV3 model has no state_dict; "
                "cannot share weights with NextN draft model."
            )

        required_prefixes = ("embed_tokens.", "lm_head.")
        shared_weights: dict[str, DLPackArray] = {}
        for name, value in target_state_dict.items():
            for prefix in required_prefixes:
                if name.startswith(prefix):
                    shared_weights[name] = value

        if len(shared_weights) != len(required_prefixes):
            raise ValueError(
                f"Missing weight prefixes {required_prefixes} in target DeepseekV3 "
                f"state_dict. Cannot share weights with NextN draft model."
            )

        logger.info(
            "Sharing DeepseekV3 embedding and head weights with NextN draft model."
        )
        return shared_weights, shared_ep_comm_initializer

    @traced
    def calculate_num_steps(
        self,
        model: PipelineModel[TextContext],
        huggingface_config: AutoConfig,
        num_steps: int,
        context: TextContext,
        is_draft: bool = False,
    ) -> int:
        """Computes the number of steps to run for the given context."""
        max_seq_len = model.calculate_max_seq_len(
            self.pipeline_config, huggingface_config=huggingface_config
        )
        if is_draft:
            max_seq_len -= 1
        num_available_steps = context.compute_num_available_steps(max_seq_len)

        if num_available_steps <= 0:
            raise ValueError(
                f"Request {context.request_id} length ({len(context.tokens)}) is larger than or equal to the configured max_length ({max_seq_len})"
            )

        return min(num_available_steps, num_steps)

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

    def _create_sampling_parameters(
        self,
        batch: list[TextContext],
        device: Device,
    ) -> tuple[Buffer, Buffer, Buffer, Buffer, Buffer, Buffer]:
        """Create sampling parameter tensors from context batch.

        Args:
            batch: List of context objects containing sampling parameters
            device: Device to place the tensors on

        Returns:
            Tuple of (top_k, max_k, temperature, top_p, min_top_p, seed) tensors
        """
        top_k_np = np.array(
            [context.sampling_params.top_k for context in batch], dtype=np.int64
        )
        top_k = Buffer.from_numpy(top_k_np).to(device)
        max_k_np = np.array(np.max(top_k_np), dtype=np.int64)
        max_k = Buffer.from_numpy(max_k_np)
        temperature_np = np.array(
            [context.sampling_params.temperature for context in batch],
            dtype=np.float32,
        )
        temperature = Buffer.from_numpy(temperature_np).to(device)
        top_p_np = np.array(
            [context.sampling_params.top_p for context in batch],
            dtype=np.float32,
        )
        top_p = Buffer.from_numpy(top_p_np).to(device)
        # min_top_p must be provided as a scalar CPU tensor
        min_top_p_np = np.array(np.min(top_p_np), dtype=np.float32)
        min_top_p = Buffer.from_numpy(min_top_p_np)
        seed_np = np.array(
            [context.sampling_params.seed for context in batch], dtype=np.uint64
        )
        seed = Buffer.from_numpy(seed_np).to(device)

        return (top_k, max_k, temperature, top_p, min_top_p, seed)

    def _create_penalty_inputs(
        self,
        batch: list[TextContext],
        device: Device,
        num_steps: int = 1,
    ) -> tuple[list[FrequencyData], Buffer, Buffer, Buffer] | None:
        """Create penalty input tensors from context batch.

        Args:
            batch: List of context objects containing sampling parameters.
            device: Device to place the tensors on.
            num_steps: Number of generation steps for frequency CSR padding.

        Returns:
            Tuple of (frequency_data, frequency_penalty, presence_penalty,
            repetition_penalty) or None if penalties are disabled and not
            needed.
        """
        if not self.pipeline_config.sampling.enable_penalties:
            _check_need_penalties(batch)
            return None

        frequency_data = [
            _build_token_frequency_csr(batch, num_steps, device),
            _build_token_frequency_csr(
                batch, num_steps, device, include_prompt=True
            ),
        ]

        frequency_penalty = Buffer.from_numpy(
            np.array(
                [
                    context.sampling_params.frequency_penalty
                    for context in batch
                ],
                dtype=np.float32,
            )
        ).to(device)
        presence_penalty = Buffer.from_numpy(
            np.array(
                [context.sampling_params.presence_penalty for context in batch],
                dtype=np.float32,
            )
        ).to(device)
        repetition_penalty = Buffer.from_numpy(
            np.array(
                [
                    context.sampling_params.repetition_penalty
                    for context in batch
                ],
                dtype=np.float32,
            )
        ).to(device)

        return (
            frequency_data,
            frequency_penalty,
            presence_penalty,
            repetition_penalty,
        )

    @traced
    def sample_draft_logits(
        self,
        model_outputs: ModelOutputs,
        prev_tokens: Buffer,
        prev_logits: Buffer,
        top_k: Buffer,
        max_k: Buffer,
        temperature: Buffer,
        top_p: Buffer,
        min_top_p: Buffer,
        seed: Buffer,
        frequency_data: list[FrequencyData] | None = None,
        frequency_penalty: Buffer | None = None,
        presence_penalty: Buffer | None = None,
        repetition_penalty: Buffer | None = None,
    ) -> tuple[Buffer, Buffer, Buffer]:
        """Samples draft tokens from the draft model logits."""
        graph_inputs: list[Buffer] = [
            model_outputs.logits,
            prev_tokens,
            top_k,
            max_k,
            temperature,
            top_p,
            min_top_p,
            seed,
            prev_logits,
        ]
        if frequency_data is not None:
            assert frequency_penalty is not None
            assert presence_penalty is not None
            assert repetition_penalty is not None
            for freq_data in frequency_data:
                graph_inputs.extend([freq_data.data, freq_data.offsets])
            graph_inputs.extend(
                [frequency_penalty, presence_penalty, repetition_penalty]
            )
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

    def _build_rejection_runner(
        self,
        strategy: str,
        target_session: InferenceSession,
        device_ref: DeviceRef,
    ) -> Callable[..., Any]:
        """Loads the compiled rejection sampler graph and returns a runner."""
        if strategy == "typical-acceptance":
            return _TypicalAcceptanceRunner(
                greedy_model=target_session.load(
                    greedy_acceptance_sampler(device=device_ref)
                ),
                stochastic_model=target_session.load(
                    typical_acceptance_sampler(device=device_ref)
                ),
                target_device=self.target_devices[0],
            )
        if strategy == "greedy":
            return _GreedyRunner(
                target_session.load(
                    greedy_acceptance_sampler(device=device_ref)
                )
            )
        if strategy == "logit-comparison":
            return _LogitComparisonRunner(
                target_session.load(rejection_sampler(device=device_ref))
            )
        return _ResidualRunner(
            target_session.load(
                rejection_sampler_with_residuals(device=device_ref)
            )
        )

    def _call_rejection_sampler(
        self,
        draft_tokens: Buffer,
        draft_logits: Buffer,
        target_logits: Buffer,
        target_logit_offsets: Buffer,
        all_draft_logits: Buffer | None,
        context_batch: list[TextContext] | None = None,
    ) -> tuple[Buffer, Buffer, Buffer | None]:
        """Calls the rejection sampler with the appropriate arguments.

        Args:
            draft_tokens: Draft token ids.
            draft_logits: Logits for sampled draft tokens.
            target_logits: Target model logits.
            target_logit_offsets: Offsets into target_logits per batch element.
            all_draft_logits: Full draft logits (used by residual sampler).
            context_batch: Batch contexts for per-request sampling params.

        Returns:
            A tuple of (first_rejected_tokens, recovered_tokens, bonus_tokens).
            bonus_tokens is None when using logit-comparison strategy.
        """
        return self._run_rejection_sampler(
            draft_tokens,
            draft_logits,
            target_logits,
            target_logit_offsets,
            all_draft_logits,
            context_batch,
        )

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
                upper_bound=self._max_length, default=context.max_length
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
        """Releases resources associated with this request ID.

        Args:
            request_id: Unique identifier for the finished request.

        Note: Target model KV cache is released by the scheduler via batch_constructor.
        This method only releases the draft model KV cache, which the scheduler
        doesn't know about.
        """
        # Release draft model KV cache (scheduler doesn't manage this).
        # The request may not have been claimed yet if it errored before
        # execute() ran the draft model, so check before releasing.
        replica_idx = self._draft_replica_idx.pop(request_id, 0)
        if self._draft_kv_manager.contains(request_id, replica_idx=replica_idx):
            self._draft_kv_manager.release(request_id, replica_idx=replica_idx)
        # Target model KV cache is released by scheduler via batch_constructor

    @property
    def kv_manager(self) -> PagedKVCacheManager:
        """Returns the target model KV cache manager."""
        return self._target_kv_manager
