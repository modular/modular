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
"""Unified DSpark Nemotron-H pipeline model: target and draft in one graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from max import tree
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device, default_dtype
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter, load_weights
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import (
    KVCacheConfig,
    PipelineConfig,
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lib._hf_config import PretrainedConfig
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.profiler import traced
from typing_extensions import override

from ..nemotron_h_modulev3.model import NemotronHModel
from ..nemotron_h_modulev3.nemotron_h import NemotronH
from .batch_processor import UnifiedDSparkNemotronHBatchProcessor
from .dspark_qwen3 import DSparkQwen3, parse_dspark_draft_config
from .model_config import (
    UnifiedDSparkNemotronHConfig,
    construct_draft_kv_params,
    construct_target_kv_params,
)
from .unified_dspark_nemotron_h import UnifiedDSparkNemotronHModule
from .weight_adapters import (
    convert_draft_state_dict,
    convert_unified_state_dict,
)

logger = logging.getLogger("max.pipelines")


@dataclass
class UnifiedDSparkNemotronHInputs(UnifiedSpecDecodeInputs):
    """Inputs for the unified DSpark Nemotron-H graph.

    The spec-decode fields and trailing buffer packing come from
    :class:`UnifiedSpecDecodeInputs`; ``tokens`` / ``input_row_offsets`` /
    ``return_n_logits`` plus the KV cache tree form the graph's prefix. The
    signal buffers of a multi-device mesh trail every declared input, because
    the graph is traced by ``Module.compile``.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    signal_buffers: list[Buffer]

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        buffers = (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs
                else ()
            ),
        )
        return (
            buffers
            + self._spec_decode_tail_buffers(include_in_thinking_phase=False)
            + tuple(self.signal_buffers)
        )


# ``model`` is the compiled engine model here, as the mixin declares it; the
# ModuleV3 base types it as the module's callable. The batch processor is this
# architecture's own, not the Llama one the base names.
class UnifiedDSparkNemotronHModel(  # type: ignore[misc]
    _UnifiedSpecDecodeModelMixin, NemotronHModel
):
    """Nemotron-H with the DSpark block draft, compiled as one graph."""

    model_config_cls: ClassVar[type[Any]] = UnifiedDSparkNemotronHConfig
    batch_processor_cls: ClassVar[
        type[UnifiedDSparkNemotronHBatchProcessor]
    ] = UnifiedDSparkNemotronHBatchProcessor  # type: ignore[assignment]

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        # The graph returns the verify logits and the tapped hidden states
        # whatever the caller asked of a plain target.
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )

    @classmethod
    @override
    def get_kv_params(
        cls,
        huggingface_config: PretrainedConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        assert pipeline_config.draft_model is not None
        draft_hf = pipeline_config.draft_model.huggingface_config
        assert draft_hf is not None
        draft = parse_dspark_draft_config(draft_hf)
        target = construct_target_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )
        return MultiKVCacheParams.from_params(
            {
                "target": target,
                "draft": construct_draft_kv_params(
                    pipeline_config,
                    draft,
                    devices,
                    page_size=target.page_size,
                ),
            }
        )

    @override
    def _create_model_config(  # type: ignore[override]
        self, state_dict: dict[str, Any]
    ) -> UnifiedDSparkNemotronHConfig:
        config = self.arch_config_as(UnifiedDSparkNemotronHConfig)
        config.target.quant_scheme.check_weights(state_dict.keys())
        return config

    def _load_draft_state_dict(
        self,
        config: UnifiedDSparkNemotronHConfig,
        target_state_dict: dict[str, Any],
    ) -> dict[str, Any]:
        draft_model = self.pipeline_config.draft_model
        assert draft_model is not None
        draft_weights = load_weights(draft_model.resolved_weight_paths())
        embedding = target_state_dict["backbone.embeddings.weight"]
        return convert_draft_state_dict(
            {name: weight.data() for name, weight in draft_weights.items()},
            config.draft,
            target_embedding=embedding,
        )

    # TODO(kathywu): Register this graph through `graph_specs` instead of
    # compiling it here.
    @traced
    @override
    def _load_decoder(self, session: InferenceSession) -> Any:
        del session
        target_state = self._load_state_dict()
        # Hold the checkpoint only as long as this method does.
        del self._checkpoint
        config = self._create_model_config(target_state)
        target_state = self._prepare_state_dict(target_state, config.target)
        draft_state = self._load_draft_state_dict(config, target_state)
        self._init_distributed_runtime(config)

        n_devices = len(self.devices)
        mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))
        assert self.pipeline_config.speculative is not None
        with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
            target = NemotronH(config.target)
            draft = DSparkQwen3(
                config.draft,
                config.draft_kv_params,
                num_devices=n_devices,
                max_seq_len=config.target.max_seq_len,
                device=self.devices[0],
            )
            module = UnifiedDSparkNemotronHModule(
                config,
                target,
                draft,
                kv_params=self.kv_params,
                speculative_config=self.pipeline_config.speculative,
                enable_structured_output=(
                    self.pipeline_config.needs_bitmask_constraints
                ),
            )
        compiled = module.compile(
            *module.input_types(),
            weights=convert_unified_state_dict(target_state, draft_state),
        )
        self.state_dict = dict(compiled.weights)
        self._compiled = compiled
        # Execution drives the engine model directly, appending this model's
        # own signal buffers (see UnifiedDSparkNemotronHInputs.buffers).
        return compiled.engine_model
