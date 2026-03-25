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
"""Unified EAGLE Llama3 PipelineModel: target + draft in one graph."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from max.driver import Buffer, Device
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
)
from max.pipelines.lib.interfaces import PipelineModelWithKVCache
from max.pipelines.lib.pipeline_variants.utils import get_weight_paths
from max.pipelines.lib.registry import AutoConfig
from max.pipelines.lib.speculative_decoding.unified_eagle import (
    UnifiedEagleOutputs,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights

from ..llama3.model_config import Llama3Config
from ..llama3.weight_adapters import _convert_safetensor_with_model_config
from .model_config import UnifiedEagleLlama3Config
from .unified_eagle_llama3 import UnifiedEagleLlama3 as UnifiedEagleLlama3Module
from .weight_adapters import convert_unified_safetensor_state_dict

logger = logging.getLogger("max.pipelines")


@dataclass
class UnifiedEagleLlama3Inputs(ModelInputs):
    """Inputs for the unified EAGLE Llama3 model."""

    tokens: Buffer
    input_row_offsets: Buffer
    draft_tokens: Buffer
    return_n_logits: Buffer
    draft_kv_cache_buffers: list[Buffer] = field(default_factory=list)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return (
            self.tokens,
            self.input_row_offsets,
            self.draft_tokens,
            self.return_n_logits,
            *(self.kv_cache_inputs or ()),
            *self.draft_kv_cache_buffers,
        )


class UnifiedEagleLlama3Model(PipelineModelWithKVCache[TextContext]):
    """Unified EAGLE Llama3: target + draft in one compiled graph."""

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
            return_hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
        )
        self.model = self.load_model(session)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
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
        with CompilationTimer("unified_eagle_llama3_model") as timer:
            target_state_dict = parse_state_dict_from_weights(
                self.pipeline_config, self.weights, self.adapter
            )

            assert self.pipeline_config.draft_model is not None
            draft_model_config = self.pipeline_config.draft_model
            draft_weight_paths = get_weight_paths(draft_model_config)
            draft_weights = load_weights(draft_weight_paths)
            draft_hf_config = draft_model_config.huggingface_config
            assert draft_hf_config is not None

            draft_state_dict = _convert_safetensor_with_model_config(
                dict(draft_weights.items()),
                draft_hf_config,
                draft_model_config,
            )

            target_hf_config = self.huggingface_config
            assert target_hf_config is not None

            target_config = Llama3Config.initialize(self.pipeline_config)
            target_config.finalize(
                huggingface_config=target_hf_config,
                state_dict=target_state_dict,
                return_logits=ReturnLogits.VARIABLE,
                return_hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
            )

            draft_config = Llama3Config.initialize_from_config(
                self.pipeline_config, draft_hf_config, draft_model_config
            )
            draft_config.finalize(
                huggingface_config=draft_hf_config,
                state_dict=draft_state_dict,
                return_logits=ReturnLogits.LAST_TOKEN,
                return_hidden_states=ReturnHiddenStates.LAST,
            )

            assert self.pipeline_config.speculative is not None
            num_draft_steps = (
                self.pipeline_config.speculative.num_speculative_tokens
            )

            unified_config = UnifiedEagleLlama3Config(
                target=target_config,
                draft=draft_config,
                num_draft_steps=num_draft_steps,
            )

            nn_model = UnifiedEagleLlama3Module(unified_config)

            # Share embed_tokens and lm_head BEFORE loading so state_dict()
            # deduplicates them.
            nn_model.draft.embed_tokens = nn_model.target.embed_tokens
            nn_model.draft.lm_head = nn_model.target.lm_head

            # --- Merge and load weights at top level ---
            # Load with "target.*" and "draft.*" prefixed keys so the graph
            # sees unique weight names (both models have layers.0.*).
            unified_state_dict = convert_unified_safetensor_state_dict(
                target_state_dict, draft_state_dict
            )

            # strict=False: shared weights (embed_tokens, lm_head) are aliased
            # to target's and won't have draft.* copies. EAGLE also replaces
            # some norms with Identity. rope_freqs.weight is unused.
            nn_model.load_state_dict(
                unified_state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=False,
            )
            self.state_dict = nn_model.state_dict()

            assert isinstance(self.kv_params, KVCacheParams)
            draft_num_layers = draft_config.num_hidden_layers
            self._draft_kv_params = replace(
                self.kv_params, num_layers=draft_num_layers
            )

            with Graph(
                "unified_eagle_llama3",
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
        """Execute and return all graph outputs for speculative decoding."""
        assert isinstance(model_inputs, UnifiedEagleLlama3Inputs)
        model_outputs = self.model.execute(*model_inputs.buffers)

        return UnifiedEagleOutputs(
            num_accepted_draft_tokens=model_outputs[0],
            next_tokens=model_outputs[1],
            next_draft_tokens=model_outputs[2],
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
        draft_kv_cache_buffers: list[Buffer] | None = None,
        draft_tokens: Buffer | None = None,
        **kwargs,
    ) -> UnifiedEagleLlama3Inputs:
        if draft_kv_cache_buffers is None:
            raise ValueError("draft_kv_cache_buffers is required")
        if draft_tokens is None:
            raise ValueError("draft_tokens is required")
        context_batch = [ctx for batch in replica_batches for ctx in batch]
        device0 = self.devices[0]

        # Build tokens from active window (all unprocessed tokens).
        # During prefill: full prompt. During decode: includes tokens
        # not yet marked as processed, which get reprocessed to keep
        # the KV cache consistent.
        tokens_np = np.concatenate([ctx.tokens.active for ctx in context_batch])
        tokens_buf = Buffer.from_numpy(tokens_np).to(device0)

        offsets_np = np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
        )
        offsets_buf = Buffer.from_numpy(offsets_np).to(device0)

        return_n_logits_buf = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        return UnifiedEagleLlama3Inputs(
            tokens=tokens_buf,
            input_row_offsets=offsets_buf,
            draft_tokens=draft_tokens,
            return_n_logits=return_n_logits_buf,
            kv_cache_inputs=kv_cache_inputs,
            draft_kv_cache_buffers=draft_kv_cache_buffers,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> UnifiedEagleLlama3Inputs:
        assert isinstance(prev_model_inputs, UnifiedEagleLlama3Inputs)
        raise NotImplementedError(
            "Multistep execution is not supported for UnifiedEagleLlama3Model. "
            "The unified pipeline handles iteration internally."
        )

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return Llama3Config.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )
