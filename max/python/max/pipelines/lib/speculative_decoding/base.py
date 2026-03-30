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
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.driver import load_devices
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.interfaces import PipelineTokenizer, RequestID, TextGenerationRequest
from max.kv_cache import IncrementCacheLengthsProcessor, PagedKVCacheManager
from max.kv_cache.registry import load_multi_kv_managers
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib.sampling import (
    RejectionRunner,
    rejection_runner_registry,
)
from max.pipelines.lib.sampling.sampling import TokenSampler
from max.profiler import traced
from transformers import AutoConfig

from ..interfaces import PipelineModel, PipelineModelWithKVCache
from ..pipeline_variants.text_generation import (
    TextGenerationPipelineInterface,
    get_eos_tokens,
    get_weight_paths,
)
from ..sampling import SamplingConfig
from .ragged_token_merger import RaggedTokenMergerRunner
from .utils import SpeculativeDecodingMetrics

if TYPE_CHECKING:
    from ..config import MAXModelConfig, PipelineConfig, SpeculativeConfig

logger = logging.getLogger("max.pipelines")


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

        # Load sampler
        sampling_config: SamplingConfig = self.pipeline_config.sampling
        sampling_config.enable_variable_logits = False
        self._sampler = TokenSampler(
            session=self._session,
            sampling_config=sampling_config,
            device=device_refs[0],
            return_logits=True,
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
        rejection_runner_type = rejection_runner_registry(strategy)
        self._rejection_runner: RejectionRunner = rejection_runner_type(
            self._session, device_refs[0]
        )
        self._needs_all_draft_logits = strategy == "residual"

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

        self._ragged_token_merger = RaggedTokenMergerRunner(
            session=self._session, device_ref=device_refs[0]
        )

        self._num_draft_steps = self._speculative_config.num_speculative_tokens

        # Initialize metrics tracker
        self.metrics = SpeculativeDecodingMetrics.empty(
            num_speculative_tokens=self._num_draft_steps
        )

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

        # Initialize the ragged increment cache lengths model
        self._increment_cache_lengths_processor = (
            IncrementCacheLengthsProcessor(
                session=self._session, params=target_kv_params
            )
        )

        # We employ a crazy hack here where we completely disregard the draft
        # kv cache manager. We save the draft kv cache buffer and discard the rest
        # of the object. Now whenever we want to use the draft runtime inputs,
        # we use the target kv cache manager and secretly swap out the host kv
        # cache buffers for that of the draft model.
        # TODO: do this properly once the kv cache manager is refactored to support
        # spec decoding.
        draft_kv_inputs = draft_kv_manager.runtime_inputs(
            [[] for _ in range(multi_kv_params.data_parallel_degree)]
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
    def release(self, request_id: RequestID) -> None:
        """Release the state for the request."""
        pass

    @property
    def kv_manager(self) -> PagedKVCacheManager:
        """Returns the target model KV cache manager."""
        return self._target_kv_manager
