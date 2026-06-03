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
"""Unified DFlash Llama3 PipelineModel: target + draft in one graph."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from max.driver import Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph.weights import Weights, WeightsAdapter, load_weights
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    PipelineConfig,
    PipelineRuntimeConfig,
    UnifiedEagleOutputs,
)
from max.pipelines.lib._hf_config import PretrainedConfig
from max.pipelines.lib.interfaces import PipelineModelWithKVCache
from max.pipelines.lib.utils import parse_state_dict_from_weights

from ..llama3.model_config import Llama3Config
from ..llama3.weight_adapters import _convert_safetensor_with_model_config
from ..unified_eagle_llama3.weight_adapters import (
    convert_unified_safetensor_state_dict,
)
from .model_config import (
    UnifiedDflashLlama3Config,
    parse_dflash_draft_hf_config,
)
from .unified_dflash_llama3 import (
    UnifiedDflashLlama3 as UnifiedDflashLlama3Module,
)

logger = logging.getLogger("max.pipelines")


@dataclass
class UnifiedDflashLlama3Inputs(ModelInputs):
    """Inputs for the unified DFlash Llama3 graph.

    Carries the buffers consumed by a single execute of the unified graph:
    the merged tokens / ragged offsets, the draft tokens to verify
    (None on prefill), the persistent draft KV pool, and the sampling
    parameters used by the in-graph acceptance sampler.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    draft_tokens: Buffer | None = None
    draft_kv_blocks: list[Buffer] | None = None
    seed: Buffer | None = None
    temperature: Buffer | None = None
    top_k: Buffer | None = None
    max_k: Buffer | None = None
    top_p: Buffer | None = None
    min_top_p: Buffer | None = None
    # Set by ``OverlapTextGenerationPipeline`` and required by the
    # ``_UnifiedSpecDecodeInputs`` runtime-checkable Protocol; not consumed
    # by the DFlash graph today.
    in_thinking_phase: Buffer | None = None

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        buffers = (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *(self.kv_cache_inputs.flatten() if self.kv_cache_inputs else ()),
        )
        if self.draft_tokens is not None:
            buffers += (self.draft_tokens,)
        if self.draft_kv_blocks is not None:
            buffers += tuple(self.draft_kv_blocks)
        assert self.seed is not None
        buffers += (self.seed,)
        if self.draft_tokens is not None:
            assert self.temperature is not None
            assert self.top_k is not None
            assert self.max_k is not None
            assert self.top_p is not None
            assert self.min_top_p is not None
            buffers += (
                self.temperature,
                self.top_k,
                self.max_k,
                self.top_p,
                self.min_top_p,
            )
        return buffers


@dataclass
class PersistentInputBuffers:
    tokens: Buffer
    input_row_offsets: Buffer

    @classmethod
    def alloc(
        cls, max_batch_size: int, max_batch_input_tokens: int, device: Device
    ) -> PersistentInputBuffers:
        max_batch_input_tokens = max(max_batch_input_tokens, max_batch_size)
        tokens = Buffer(
            shape=(max_batch_input_tokens,), dtype=DType.int64, device=device
        )
        input_row_offsets = Buffer(
            shape=(max_batch_size + 1,), dtype=DType.uint32, device=device
        )
        return cls(tokens, input_row_offsets)


class UnifiedDflashLlama3Model(PipelineModelWithKVCache[TextContext]):
    """Unified DFlash Llama3: target + draft in one compiled graph."""

    model: Model

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
        )
        self.model = self.load_model(session)

        assert isinstance(pipeline_config.runtime, PipelineRuntimeConfig)
        assert pipeline_config.runtime.max_batch_size is not None
        self._persistent_input_buffers = PersistentInputBuffers.alloc(
            max_batch_size=pipeline_config.runtime.max_batch_size,
            max_batch_input_tokens=pipeline_config.runtime.max_batch_input_tokens,
            device=devices[0],
        )
        self._seed_counter = 0

    def _next_seed(self) -> Buffer:
        self._seed_counter += 1
        return Buffer.from_numpy(
            np.array([self._seed_counter], dtype=np.uint64)
        ).to(self.devices[0])

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: PretrainedConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Llama3Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def load_model(self, session: InferenceSession) -> Model:
        with CompilationTimer("unified_dflash_llama3_model") as timer:
            target_state_dict = parse_state_dict_from_weights(
                self.pipeline_config, self.weights, self.adapter
            )

            assert self.pipeline_config.draft_model is not None
            draft_model_config = self.pipeline_config.draft_model
            draft_weight_paths = draft_model_config.resolved_weight_paths()
            draft_weights = load_weights(draft_weight_paths)
            draft_hf_config = draft_model_config.huggingface_config
            assert draft_hf_config is not None

            draft_state_dict = _convert_safetensor_with_model_config(
                dict(draft_weights.items()),
                draft_hf_config,
                draft_model_config,
            )

            dflash_hf = parse_dflash_draft_hf_config(draft_hf_config)
            target_hf_config = self.huggingface_config
            assert target_hf_config is not None
            assert self.pipeline_config.speculative is not None

            target_config = Llama3Config.initialize(self.pipeline_config)
            target_config.finalize(
                huggingface_config=target_hf_config,
                state_dict=target_state_dict,
                return_logits=ReturnLogits.VARIABLE,
                return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
            )
            target_config.target_layer_ids = list(dflash_hf.target_layer_ids)

            draft_config = Llama3Config.initialize_from_config(
                self.pipeline_config, draft_hf_config, draft_model_config
            )
            # ``initialize_from_config`` defaults the draft to ``gpu:0``;
            # pin to the target's device(s) so the weights co-locate
            # whenever the target lives on a non-zero GPU.
            draft_config.devices = target_config.devices
            draft_config.kv_params = replace(
                draft_config.kv_params, devices=target_config.devices
            )
            draft_config.finalize(
                huggingface_config=draft_hf_config,
                state_dict=draft_state_dict,
                return_logits=ReturnLogits.LAST_TOKEN,
                return_hidden_states=ReturnHiddenStates.LAST,
            )

            unified_config = UnifiedDflashLlama3Config(
                target=target_config,
                draft=draft_config,
                speculative_config=self.pipeline_config.speculative,
                target_layer_ids=list(dflash_hf.target_layer_ids),
                mask_token_id=int(dflash_hf.mask_token_id),
                block_size=int(dflash_hf.block_size or 0),
            )
            unified_config.validate_dflash_fields()

            nn_model = UnifiedDflashLlama3Module(unified_config)

            # DFlash drafts have no embed_tokens / lm_head; alias the
            # target's BEFORE the state-dict walk.
            nn_model.draft.embed_tokens = nn_model.target.embed_tokens
            nn_model.draft.lm_head = nn_model.target.lm_head

            unified_state_dict = convert_unified_safetensor_state_dict(
                target_state_dict, draft_state_dict
            )

            # strict=False: shared embed_tokens / lm_head are aliased,
            # rotary cache keys from the adapter may be unused.
            nn_model.load_state_dict(
                unified_state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=False,
            )
            self.state_dict = nn_model.state_dict()

            assert isinstance(self.kv_params, KVCacheParams)
            self._draft_kv_params = replace(
                self.kv_params, num_layers=draft_config.num_hidden_layers
            )

            with Graph(
                "unified_dflash_llama3",
                input_types=nn_model.input_types(),
            ) as graph:
                inputs = nn_model._unflatten_graph_inputs(graph.inputs)
                outputs = nn_model(inputs)
                graph.output(*outputs)

            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)

        return model

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> UnifiedEagleOutputs:
        model_outputs = self.model.execute(*model_inputs.buffers)
        return UnifiedEagleOutputs(
            num_accepted_draft_tokens=model_outputs[0],
            next_tokens=model_outputs[1],
            next_draft_tokens=model_outputs[2],
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> UnifiedDflashLlama3Inputs:
        context_batch = [ctx for batch in replica_batches for ctx in batch]
        device0 = self.devices[0]
        buffer_type = Buffer if device0.is_host else DevicePinnedBuffer

        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)
        batch_size = len(context_batch)

        persistent_tokens = self._persistent_input_buffers.tokens
        persistent_tokens = persistent_tokens[:total_seq_len]
        persistent_input_row_offsets = (
            self._persistent_input_buffers.input_row_offsets
        )
        persistent_input_row_offsets = persistent_input_row_offsets[
            : batch_size + 1
        ]

        tokens_host = buffer_type(
            dtype=DType.int64,
            shape=(total_seq_len,),
            device=device0,
        )
        offsets_host = buffer_type(
            dtype=DType.uint32,
            shape=(batch_size + 1,),
            device=device0,
        )

        np.concatenate(
            [ctx.tokens.active for ctx in context_batch],
            out=tokens_host.to_numpy(),
        )
        persistent_tokens.inplace_copy_from(tokens_host)
        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=offsets_host.to_numpy(),
        )
        persistent_input_row_offsets.inplace_copy_from(offsets_host)

        return_n_logits_buf = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        return UnifiedDflashLlama3Inputs(
            tokens=persistent_tokens,
            input_row_offsets=persistent_input_row_offsets,
            return_n_logits=return_n_logits_buf,
            kv_cache_inputs=kv_cache_inputs,
            seed=self._next_seed(),
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: PretrainedConfig,
    ) -> int:
        return Llama3Config.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )
