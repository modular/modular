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
"""Unified EAGLE pipeline: single fused graph for target + draft."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, final

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, DevicePinnedBuffer, load_devices
from max.engine import InferenceSession
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.interfaces import (
    PipelineTokenizer,
    RequestID,
    TextGenerationInputs,
    TextGenerationOutput,
    TextGenerationRequest,
)
from max.kv_cache import PagedKVCacheManager
from max.kv_cache.registry import load_multi_kv_managers
from max.nn import ReturnLogits
from max.nn.kv_cache import KVCacheInputs, KVCacheParams, MultiKVCacheParams
from max.pipelines.core import TextContext
from max.profiler import Tracer, traced

from ..interfaces import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    PipelineModelWithKVCache,
)
from ..pipeline_variants.text_generation import TextGenerationPipelineInterface
from ..pipeline_variants.utils import get_eos_tokens, get_weight_paths
from .utils import (
    Protocol,
    SpeculativeDecodingMetrics,
    build_response,
    runtime_checkable,
)

if TYPE_CHECKING:
    from ..config import MAXModelConfig, PipelineConfig

logger = logging.getLogger("max.pipelines")


@dataclass(kw_only=True)
class UnifiedEagleOutputs(ModelOutputs):
    """Outputs from a unified EAGLE graph execution."""

    num_accepted_draft_tokens: Buffer
    next_tokens: Buffer
    next_draft_tokens: Buffer

    # HACK: These are required to inherit from ModelOutputs but are unused
    # for UnifiedEagleOutputs!
    logits: Buffer | None = None  # type: ignore[assignment]
    next_token_logits: None = None
    logit_offsets: None = None
    hidden_states: None = None


@runtime_checkable
class UnifiedEagleModel(Protocol):
    """Protocol for models that support unified EAGLE execution."""

    _draft_kv_params: KVCacheParams

    def execute(self, model_inputs: ModelInputs) -> UnifiedEagleOutputs:
        """Executes the unified model and returns all speculative outputs."""
        ...

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
        draft_tokens: Buffer | None = None,
        draft_kv_cache_buffers: list[Buffer] | None = None,
        **kwargs: Any,
    ) -> ModelInputs:
        """Prepares model inputs for the unified graph."""
        ...


def _get_draft_kv_blocks(
    draft_kv_manager: PagedKVCacheManager,
    data_parallel_degree: int,
) -> list[Buffer]:
    """Extract persistent draft KV block buffers (one per device).

    cache_lengths are NOT saved here — they must be created fresh
    per-execute to match the runtime batch size.
    """
    draft_kv_inputs = draft_kv_manager.runtime_inputs(
        [[] for _ in range(data_parallel_degree)]
    )
    return [per_dev.blocks for per_dev in draft_kv_inputs.inputs]


@final
class UnifiedEAGLEPipeline(TextGenerationPipelineInterface[TextContext]):
    """Pipeline for unified EAGLE: single fused graph handles target + draft.

    Unlike EAGLESpeculativeDecodingPipeline which manages two separate models,
    this pipeline uses a single model that runs both target forward and draft
    generation in one compiled graph call. Rejection sampling also happens
    in-graph (greedy acceptance).

    Orchestration:
      Prefill: model(draft_tokens=[?,0]) -> commit bonus, save new_token.
      Decode:  model(draft_tokens=[?,K]) -> verify drafts, commit tokens,
               save new_token for next iteration.
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
        self._pipeline_config = pipeline_config
        self._tokenizer = tokenizer

        model_config: MAXModelConfig = pipeline_config.model
        hf_config = model_config.huggingface_config
        assert hf_config is not None

        self._eos_token_ids = get_eos_tokens(hf_config, eos_token_id)

        device_specs = model_config.device_specs
        self.devices = load_devices(device_specs)
        self._devices = self.devices  # Required by base interface.
        self._session = InferenceSession(devices=[*self.devices])
        pipeline_config.configure_session(self._session)

        if not issubclass(pipeline_model, PipelineModelWithKVCache):
            raise ValueError(
                f"Unified EAGLE requires a KV-cache model, got {pipeline_model.__name__}"
            )

        weight_paths = get_weight_paths(model_config)
        self._model = pipeline_model(
            pipeline_config=pipeline_config,
            session=self._session,
            devices=self.devices,
            kv_cache_config=model_config.kv_cache,
            weights=load_weights(weight_paths),
            adapter=weight_adapters.get(weights_format(weight_paths)),
            return_logits=ReturnLogits.VARIABLE,
        )
        self._pipeline_model = self._model  # Required by base interface.

        target_kv_params = self._model.kv_params
        assert isinstance(target_kv_params, KVCacheParams)
        draft_kv_params = self._model._draft_kv_params  # type: ignore[attr-defined]
        assert isinstance(draft_kv_params, KVCacheParams)

        multi_kv_params = MultiKVCacheParams.from_params(
            target_kv_params, draft_kv_params
        )

        target_cache_mem = model_config.kv_cache._available_cache_memory
        assert target_cache_mem is not None
        cache_mem = target_cache_mem

        self._max_seq_len = self._model.calculate_max_seq_len(
            pipeline_config, hf_config
        )

        target_kv_mgr, draft_kv_mgr = load_multi_kv_managers(
            params=multi_kv_params,
            max_batch_size=pipeline_config.runtime.max_batch_size,
            max_seq_len=self._max_seq_len,
            session=self._session,
            available_cache_memory=cache_mem,
        )
        self._target_kv_manager = target_kv_mgr

        self._draft_kv_blocks = _get_draft_kv_blocks(
            draft_kv_mgr, multi_kv_params.data_parallel_degree
        )
        n_devices = len(self.devices)
        assert len(self._draft_kv_blocks) == n_devices

        assert pipeline_config.speculative is not None
        self._num_speculative_tokens: int = (
            pipeline_config.speculative.num_speculative_tokens
        )

        self.metrics = SpeculativeDecodingMetrics.empty(
            num_speculative_tokens=self._num_speculative_tokens
        )

        logger.info(
            "Unified EAGLE pipeline: "
            f"devices={len(self.devices)}, "
            f"max_seq_len={self._max_seq_len}, "
            f"num_speculative_tokens={self._num_speculative_tokens}"
        )

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
        """Returns the tokenizer."""
        return self._tokenizer

    @property
    def kv_manager(self) -> PagedKVCacheManager:
        """Returns the target KV cache manager."""
        return self._target_kv_manager

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        """Executes unified EAGLE speculative decoding.

        Single graph call handles: merge, target forward, greedy rejection,
        shift, extract accepted HS, and draft forward.
        """
        assert isinstance(self._model, UnifiedEagleModel)

        context_batch = inputs.flat_batch
        verify_draft_tokens = all(
            ctx.spec_decoding_state.num_draft_tokens
            == self._num_speculative_tokens
            and ctx.tokens.generated_length > 1
            for ctx in context_batch
        )
        num_draft_tokens_to_verify = (
            self._num_speculative_tokens if verify_draft_tokens else 0
        )

        # Delete the saved draft tokens if we are not verifying them.
        if not verify_draft_tokens:
            for ctx in context_batch:
                if ctx.spec_decoding_state.num_draft_tokens:
                    ctx.spec_decoding_state.saved_draft_tokens = []

        # Load or create draft tokens.
        draft_tokens = np.zeros(
            (len(context_batch), num_draft_tokens_to_verify), dtype=np.int64
        )
        if num_draft_tokens_to_verify:
            for i, ctx in enumerate(context_batch):
                tokens = ctx.spec_decoding_state.saved_draft_tokens
                assert len(tokens) == num_draft_tokens_to_verify
                draft_tokens[i, :] = tokens

        kv_cache_inputs = self._target_kv_manager.runtime_inputs(
            inputs.batches,
            num_steps=1,
            num_speculative_steps=self._num_speculative_tokens,
        )

        return_n_logits = (
            self._num_speculative_tokens + 1 if verify_draft_tokens else 1
        )

        model_inputs = self._model.prepare_initial_token_inputs(
            replica_batches=inputs.batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            draft_tokens=Buffer.from_numpy(draft_tokens).to(self.devices[0]),
            draft_kv_cache_buffers=self._draft_kv_blocks,
        )

        # Single graph call.
        outputs = self._model.execute(model_inputs)
        assert isinstance(outputs, UnifiedEagleOutputs)

        # Do the copy to host for each model output using pinned memory.
        with Tracer("D2H generated_tokens"):
            device0 = self.devices[0]
            device0.synchronize()
            num_accepted_draft_tokens_device = outputs.num_accepted_draft_tokens
            generated_tokens_host = DevicePinnedBuffer(
                shape=num_accepted_draft_tokens_device.shape,
                dtype=num_accepted_draft_tokens_device.dtype,
                device=device0,
            )
            generated_tokens_host.inplace_copy_from(
                num_accepted_draft_tokens_device
            )

            next_tokens_device = outputs.next_tokens
            next_tokens_host = DevicePinnedBuffer(
                shape=next_tokens_device.shape,
                dtype=next_tokens_device.dtype,
                device=device0,
            )
            next_tokens_host.inplace_copy_from(next_tokens_device)

            next_draft_tokens_device = outputs.next_draft_tokens
            next_draft_tokens_host = DevicePinnedBuffer(
                shape=next_draft_tokens_device.shape,
                dtype=next_draft_tokens_device.dtype,
                device=device0,
            )
            next_draft_tokens_host.inplace_copy_from(next_draft_tokens_device)

            # Sync to ensure all prior pinned d2h transfers are complete.
            device0.synchronize()

            num_accepted_draft_tokens_np = generated_tokens_host.to_numpy()
            next_tokens_np = next_tokens_host.to_numpy()
            next_draft_tokens_np = next_draft_tokens_host.to_numpy()

        assert num_accepted_draft_tokens_np.shape == (len(context_batch),)
        assert next_tokens_np.shape == (len(context_batch),)
        assert next_draft_tokens_np.shape == (
            len(context_batch),
            self._num_speculative_tokens,
        )
        assert all(
            num_accept <= num_draft_tokens_to_verify
            for num_accept in num_accepted_draft_tokens_np
        )

        for batch_idx, ctx in enumerate(context_batch):
            for token_idx in range(num_accepted_draft_tokens_np[batch_idx]):
                if not ctx.is_done:
                    ctx.update(draft_tokens[batch_idx, token_idx])
            if not ctx.is_done:
                ctx.update(next_tokens_np[batch_idx])
                # Save the generated draft tokens for verification in next iteration.
                ctx.spec_decoding_state.saved_draft_tokens = (
                    next_draft_tokens_np[batch_idx].copy()
                )

        self.metrics.update(
            draft_tokens_accepted=num_accepted_draft_tokens_np.sum(),
            draft_tokens_generated=num_draft_tokens_to_verify
            * len(context_batch),
        )

        res = build_response(
            context_batch=context_batch, max_seq_len=self._max_seq_len
        )
        self._target_kv_manager.step(inputs.batches)

        return res

    def release(self, request_id: RequestID) -> None:
        """Releases resources for the given request."""
        pass
