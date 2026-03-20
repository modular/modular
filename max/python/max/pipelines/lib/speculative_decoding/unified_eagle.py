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
from max.pipelines.core import TextContext, reserve_token_space_for_batch
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
        if len(self._draft_kv_blocks) != n_devices:
            raise ValueError(
                f"Expected {n_devices} draft KV blocks, "
                f"got {len(self._draft_kv_blocks)}"
            )
        logger.info(
            f"Draft KV blocks: {len(self._draft_kv_blocks)} "
            f"({n_devices} devices)"
        )

        device_refs = [DeviceRef.from_device(dev) for dev in self.devices]
        if len(self.devices) > 1:
            self._draft_signal_buffers = Signals(device_refs).buffers()
        else:
            self._draft_signal_buffers = []

        self.metrics = SpeculativeDecodingMetrics.empty()

        logger.info(
            "Unified EAGLE pipeline active: "
            f"devices={len(self.devices)}, "
            f"draft_kv_blocks={len(self._draft_kv_blocks)}, "
            f"max_seq_len={self._max_seq_len}"
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
    # Draft KV cache helpers
    # ------------------------------------------------------------------
    def _build_draft_kv_buffers(
        self,
        kv_cache_inputs: KVCacheInputs,
        context_batch: list[TextContext],
    ) -> list[Buffer]:
        """Build draft_kv_cache_buffers with cache_lengths from draft_kv_start_idx.

        Interleaves persistent draft KV blocks with cache_lengths populated
        from each context's draft_kv_start_idx.

        For prefill (draft_kv_start_idx=0), cache_lengths are all zeros.
        For decode, cache_lengths reflect how much valid draft KV exists.

        Returns: [blocks_dev0, lengths_dev0, blocks_dev1, lengths_dev1, ...]
        """
        # Build the cache_lengths array from context state.
        # All devices share the same batch, so same cache_lengths values.
        draft_lengths_np = np.array(
            [
                ctx.spec_decoding_state.draft_kv_start_idx
                for ctx in context_batch
            ],
            dtype=np.uint32,
        )

        buffers: list[Buffer] = []
        for dev_idx, draft_blocks in enumerate(self._draft_kv_blocks):
            # Match target's cache_lengths device placement.
            target_device = kv_cache_inputs.inputs[dev_idx].cache_lengths.device
            draft_cache_lengths = Buffer.from_numpy(draft_lengths_np).to(
                target_device
            )
            buffers.append(draft_blocks)
            buffers.append(draft_cache_lengths)
        return buffers

    # ------------------------------------------------------------------
    # Draft token management
    # ------------------------------------------------------------------
    def _save_draft_token(
        self,
        context_batch: list[TextContext],
        new_token_np: npt.NDArray[np.int64],
    ) -> None:
        for i, ctx in enumerate(context_batch):
            if not ctx.is_done:
                ctx.spec_decoding_state.saved_draft_tokens = np.array(
                    [int(new_token_np[i])], dtype=np.int64
                )

    def _load_draft_tokens(
        self,
        context_batch: list[TextContext],
    ) -> tuple[Buffer, int]:
        """Load saved draft tokens into a [B, K] buffer."""
        max_k = max(
            len(ctx.spec_decoding_state.saved_draft_tokens)
            for ctx in context_batch
        )
        batch_tokens = np.zeros((len(context_batch), max_k), dtype=np.int64)
        for i, ctx in enumerate(context_batch):
            tokens = ctx.spec_decoding_state.saved_draft_tokens
            batch_tokens[i, : len(tokens)] = tokens
        return (
            Buffer.from_numpy(batch_tokens).to(self.devices[0]),
            max_k,
        )

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
        is_prefill = any(
            ctx.tokens.generated_length == 0 for ctx in context_batch
        )
        # Allocate KV pages (target + draft share page table).
        # Need space for: verify K drafts + generate K new drafts + 1 safety.
        num_draft_steps = max(
            1,
            max(
                len(ctx.spec_decoding_state.saved_draft_tokens)
                for ctx in context_batch
            )
            if not is_prefill
            else 1,
        )
        for replica_idx, replica_batch in enumerate(inputs.batches):
            for ctx in replica_batch:
                self._target_kv_manager.alloc(
                    ctx,
                    replica_idx=replica_idx,
                    num_steps=2 * num_draft_steps + 1,
                )
        # Load or create draft tokens.
        if is_prefill:
            draft_tokens_buf = Buffer.from_numpy(
                np.zeros((len(context_batch), 0), dtype=np.int64)
            ).to(self.devices[0])
            num_draft = 0
        else:
            draft_tokens_buf, num_draft = self._load_draft_tokens(context_batch)
        # KV cache inputs — bump token space for decode verification.
        if num_draft > 0:
            with reserve_token_space_for_batch(context_batch, num_draft):
                kv_cache_inputs = self._target_kv_manager.runtime_inputs(
                    inputs.batches, num_steps=1
                )
        else:
            kv_cache_inputs = self._target_kv_manager.runtime_inputs(
                inputs.batches, num_steps=1
            )

        draft_kv_cache_buffers = self._build_draft_kv_buffers(
            kv_cache_inputs, context_batch
        )
        extra_kwargs: dict[str, Any] = {}
        if self._draft_signal_buffers:
            extra_kwargs["draft_signal_buffers"] = self._draft_signal_buffers

        model_inputs = self._model.prepare_initial_token_inputs(
            replica_batches=inputs.batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=num_draft + 1,
            draft_tokens=draft_tokens_buf,
            draft_kv_cache_buffers=draft_kv_cache_buffers,
            **extra_kwargs,
        )

        outputs = self._model.execute_unified(model_inputs)

        first_rejected_np = outputs.first_rejected.to_numpy()
        recovered_np = outputs.recovered.to_numpy()
        bonus_np = outputs.bonus.to_numpy()
        new_token_np = outputs.new_token.to_numpy()

        if is_prefill:
            for ctx in context_batch:
                ctx.spec_decoding_state.draft_kv_start_idx = (
                    ctx.tokens.current_position
                )

            # Commit bonus token which is the target sampled token.
            for i, ctx in enumerate(context_batch):
                if not ctx.is_done:
                    ctx.update(int(bonus_np[i, 0]))

            for ctx in context_batch:
                remaining = ctx.tokens.active_length - 1
                if remaining > 0:
                    ctx.tokens.skip_processing(remaining)

        else:
            # Decode: verify drafts and commit accepted + recovered/bonus.
            draft_tokens_np = draft_tokens_buf.to_numpy()
            metrics = update_contexts_and_compute_metrics_eagle(
                context_batch=context_batch,
                first_rejected_tokens=first_rejected_np,
                recovered_tokens=recovered_np,
                bonus_tokens=bonus_np,
                draft_tokens=draft_tokens_np,
                num_draft_tokens_generated=num_draft,
            )
            self.metrics.update(metrics)

            for ctx in context_batch:
                ctx.apply_processing_offset(0)

            for i, ctx in enumerate(context_batch):
                fr = int(first_rejected_np[i])
                ctx.spec_decoding_state.draft_kv_start_idx += fr + 1

                ctx.spec_decoding_state.draft_kv_start_idx = min(
                    ctx.spec_decoding_state.draft_kv_start_idx,
                    ctx.tokens.processed_length,
                )

        self._save_draft_token(context_batch, new_token_np)

        res = build_response(
            context_batch=context_batch, max_seq_len=self._max_seq_len
        )
        if not is_prefill:
            self._target_kv_manager.step(inputs.batches)

        return res

    def release(self, request_id: RequestID) -> None:
        """Releases resources for the given request."""
        pass
