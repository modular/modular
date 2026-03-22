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
from typing import TYPE_CHECKING, Any, Protocol, final, runtime_checkable

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, load_devices
from max.engine import InferenceSession
from max.graph import DeviceRef
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
from max.nn.comm import Signals
from max.nn.kv_cache import KVCacheInputs, KVCacheParams, MultiKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextContext
from max.profiler import traced

from ..interfaces import ModelInputs, PipelineModel, PipelineModelWithKVCache
from ..pipeline_variants.text_generation import TextGenerationPipelineInterface
from ..pipeline_variants.utils import get_weight_paths
from .utils import (
    SpeculativeDecodingMetrics,
    build_response,
    update_contexts_and_compute_metrics_eagle,
)

if TYPE_CHECKING:
    from ..config import MAXModelConfig, PipelineConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class UnifiedEagleOutputs:
    """Shared output type for all unified eagle models."""

    last_logits: Buffer
    logits: Buffer
    logit_offsets: Buffer
    hidden_states: Buffer
    first_rejected: Buffer
    recovered: Buffer
    bonus: Buffer
    shifted_tokens: Buffer
    new_token: Buffer
    draft_hs: Buffer


@runtime_checkable
class UnifiedEagleModel(Protocol):
    """Protocol for models that support unified EAGLE execution."""

    _draft_kv_params: KVCacheParams

    def execute_unified(self, model_inputs: ModelInputs) -> UnifiedEagleOutputs:
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
        device_specs = model_config.device_specs
        self.devices = load_devices(device_specs)
        self._devices = self.devices  # Required by base interface.
        session = InferenceSession(devices=[*self.devices])
        pipeline_config.configure_session(session)

        if not issubclass(pipeline_model, PipelineModelWithKVCache):
            raise ValueError(
                f"Unified EAGLE requires a KV-cache model, got {pipeline_model.__name__}"
            )

        weight_paths = get_weight_paths(model_config)

        self._model = pipeline_model(
            pipeline_config=pipeline_config,
            session=session,
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
            session=session,
            available_cache_memory=cache_mem,
        )
        self._target_kv_manager = target_kv_mgr

        self._draft_kv_blocks = _get_draft_kv_blocks(
            draft_kv_mgr, multi_kv_params.data_parallel_degree
        )
        n_devices = len(self.devices)
        assert len(self._draft_kv_blocks) == n_devices

        device_refs = [DeviceRef.from_device(dev) for dev in self.devices]
        if len(self.devices) > 1:
            self._draft_signal_buffers = Signals(device_refs).buffers()
        else:
            self._draft_signal_buffers = []

        self.metrics = SpeculativeDecodingMetrics.empty()

        assert pipeline_config.speculative is not None
        self._num_speculative_tokens: int = (
            pipeline_config.speculative.num_speculative_tokens
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
    # Draft token management
    # ------------------------------------------------------------------
    def _save_draft_tokens(
        self,
        context_batch: list[TextContext],
        new_tokens_np: npt.NDArray[np.int64],
    ) -> None:
        """Save draft tokens of shape [batch, num_draft_steps]."""
        for i, ctx in enumerate(context_batch):
            if not ctx.is_done:
                ctx.spec_decoding_state.saved_draft_tokens = new_tokens_np[
                    i
                ].copy()

    def _load_draft_tokens(
        self,
        context_batch: list[TextContext],
    ) -> npt.NDArray[np.int64]:
        """Load saved draft tokens into a [B, K] buffer."""
        batch_tokens = np.zeros(
            (len(context_batch), self._num_speculative_tokens), dtype=np.int64
        )
        for i, ctx in enumerate(context_batch):
            tokens = ctx.spec_decoding_state.saved_draft_tokens
            assert len(tokens) == self._num_speculative_tokens
            batch_tokens[i, :] = tokens
        return batch_tokens

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

        # Allocate KV pages (target + draft share page table).
        # Need space for: verify K drafts + generate K new drafts
        for replica_idx, replica_batch in enumerate(inputs.batches):
            for ctx in replica_batch:
                self._target_kv_manager.alloc(
                    ctx,
                    replica_idx=replica_idx,
                    num_steps=1,
                    num_speculative_steps=self._num_speculative_tokens,
                )

        context_batch = inputs.flat_batch
        verify_draft_tokens = all(
            ctx.spec_decoding_state.num_draft_tokens
            == self._num_speculative_tokens
            and ctx.tokens.generated_length > 1
            for ctx in context_batch
        )

        # Delete the saved draft tokens if we are not verifying them.
        if not verify_draft_tokens:
            for ctx in context_batch:
                if ctx.spec_decoding_state.num_draft_tokens:
                    ctx.spec_decoding_state.saved_draft_tokens = []

        # Load or create draft tokens.
        if verify_draft_tokens:
            draft_tokens = self._load_draft_tokens(context_batch)
        else:
            draft_tokens = np.zeros((len(context_batch), 0), dtype=np.int64)

        kv_cache_inputs = self._target_kv_manager.runtime_inputs(
            inputs.batches,
            num_steps=1,
            num_speculative_steps=self._num_speculative_tokens,
        )

        extra_kwargs: dict[str, Any] = {}
        if self._draft_signal_buffers:
            extra_kwargs["draft_signal_buffers"] = self._draft_signal_buffers

        return_n_logits = (
            self._num_speculative_tokens + 1 if verify_draft_tokens else 1
        )

        model_inputs = self._model.prepare_initial_token_inputs(
            replica_batches=inputs.batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
            draft_tokens=Buffer.from_numpy(draft_tokens).to(self.devices[0]),
            draft_kv_cache_buffers=self._draft_kv_blocks,
            **extra_kwargs,
        )

        outputs = self._model.execute_unified(model_inputs)

        first_rejected_np = outputs.first_rejected.to_numpy()
        recovered_np = outputs.recovered.to_numpy()
        bonus_np = outputs.bonus.to_numpy()
        new_token_np = outputs.new_token.to_numpy()

        if verify_draft_tokens:
            # Decode: verify drafts and commit accepted + recovered/bonus.
            assert all(
                num_accept <= self._num_speculative_tokens
                for num_accept in first_rejected_np
            )
            metrics = update_contexts_and_compute_metrics_eagle(
                context_batch=context_batch,
                first_rejected_tokens=first_rejected_np,
                recovered_tokens=recovered_np,
                bonus_tokens=bonus_np,
                draft_tokens=draft_tokens,
                num_draft_tokens_generated=self._num_speculative_tokens,
            )
            self.metrics.update(metrics)

            # TODO: delete this the call to apply_processing_offset in
            # update_contexts_and_compute_metrics_eagle so we don't need to do
            # this here!
            for ctx in context_batch:
                ctx.apply_processing_offset(0)
        else:
            # Commit bonus token which is the target sampled token.
            for i, ctx in enumerate(context_batch):
                if not ctx.is_done:
                    ctx.update(int(bonus_np[i, 0]))

        self._save_draft_tokens(context_batch, new_token_np)

        res = build_response(
            context_batch=context_batch, max_seq_len=self._max_seq_len
        )
        self._target_kv_manager.step(inputs.batches)

        return res

    def release(self, request_id: RequestID) -> None:
        """Releases resources for the given request."""
        pass
