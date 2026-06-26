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
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph.weights import Weights, WeightsAdapter, load_weights
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    PipelineConfig,
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lib._hf_config import PretrainedConfig
from max.pipelines.lib.interfaces import PipelineModelWithKVCache
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights

from ..llama3.model_config import Llama3Config
from ..llama3.weight_adapters import _convert_safetensor_with_model_config
from ..unified_eagle_llama3.weight_adapters import (
    convert_unified_safetensor_state_dict,
)
from .batch_processor import UnifiedDflashLlama3BatchProcessor
from .model_config import (
    UnifiedDflashLlama3Config,
    parse_dflash_draft_hf_config,
)
from .unified_dflash_llama3 import (
    UnifiedDflashLlama3 as UnifiedDflashLlama3Module,
)

logger = logging.getLogger("max.pipelines")


@dataclass
class UnifiedDflashLlama3Inputs(UnifiedSpecDecodeInputs):
    """Inputs for the unified DFlash Llama3 graph.

    The spec-decode fields and trailing buffer packing come from
    :class:`UnifiedSpecDecodeInputs`; ``tokens`` / ``input_row_offsets`` /
    ``return_n_logits`` plus the KV cache form this single-device graph's
    prefix. The DFlash graph does not bind ``in_thinking_phase``.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        buffers = (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *(self.kv_cache_inputs.flatten() if self.kv_cache_inputs else ()),
        )
        return buffers + self._spec_decode_tail_buffers(
            include_in_thinking_phase=False, supports_structured_output=False
        )


class UnifiedDflashLlama3Model(
    _UnifiedSpecDecodeModelMixin, PipelineModelWithKVCache[TextContext]
):
    """Unified DFlash Llama3: target + draft in one compiled graph."""

    model_config_cls: ClassVar[type[Any]] = UnifiedDflashLlama3Config
    batch_processor_cls: ClassVar[type[UnifiedDflashLlama3BatchProcessor]] = (
        UnifiedDflashLlama3BatchProcessor
    )

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
            draft_config.sliding_window = draft_model_config.sliding_window
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
            self.kv_params = MultiKVCacheParams.from_params(
                {"target": self.kv_params, "draft": self._draft_kv_params}
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
