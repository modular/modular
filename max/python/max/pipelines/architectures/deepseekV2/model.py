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
"""Implements the DeepseekV2 nn.model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max import tree
from max.driver import Buffer, Device, DeviceSpec
from max.engine.api import InferenceSession, Model
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    BatchProcessor,
    CompilationTimer,
    GraphPipelineModelWithKVCache,
    GraphRole,
    GraphSpec,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    compile_and_init_graphs,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.profiler import traced
from typing_extensions import override

from .batch_processor import DeepseekV2BatchProcessor
from .deepseekV2 import DeepseekV2
from .distributed_deepseekV2 import DistributedDeepseekV2
from .model_config import DeepseekV2Config

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV2Inputs(ModelInputs):
    """A class representing inputs for the DeepseekV2 model.

    This class encapsulates the input tensors required for the DeepseekV2 model execution:
    - tokens: A tensor containing the input token IDs
    - input_row_offsets: A tensor containing the offsets for each row in the ragged input sequence
    - return_n_logits: A tensor containing the number of logits to return
    """

    tokens: Buffer
    input_row_offsets: Buffer
    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer = field(kw_only=True)


class DeepseekV2Model(
    LogProbabilitiesMixin,
    GraphPipelineModelWithKVCache[TextContext],
):
    model_config_cls: ClassVar[type[Any]] = DeepseekV2Config
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]]] = (
        DeepseekV2BatchProcessor
    )

    _models: dict[str, Model]
    _weights_dict: dict[str, Any]

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
        return_logits: ReturnLogits = ReturnLogits.ALL,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        if pipeline_config.model.device_specs[0] == DeviceSpec.cpu():
            raise ValueError("DeepseekV2 currently only supported on gpu.")

        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )

        self._models = self.load_graphs(session)

    @property
    def model(self) -> Model:
        """The compiled decoder graph."""
        return self._models[GraphRole.DECODER]

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        assert isinstance(model_inputs, DeepseekV2Inputs)

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *model_inputs.signal_buffers,
            *tree.leaves(curr_kv_cache_inputs),
        )
        if len(model_outputs) == 3:
            assert isinstance(model_outputs[0], Buffer)
            assert isinstance(model_outputs[1], Buffer)
            assert isinstance(model_outputs[2], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[1],
                logit_offsets=model_outputs[2],
            )
        else:
            assert isinstance(model_outputs[0], Buffer)
            return ModelOutputs(
                next_token_logits=model_outputs[0],
                logits=model_outputs[0],
            )

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights supported in DeepseekV2."
            )
        return super()._load_state_dict()

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        del state_dict
        model_config = DeepseekV2Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.max_batch_context_length = (
            self.planned_max_batch_total_tokens
            or model_config.max_batch_context_length
        )
        return model_config

    @override
    def graph_specs(self, model_config: Any) -> dict[str, GraphSpec]:
        """Returns the decoder graph, tensor parallel across several
        devices."""
        assert isinstance(model_config, DeepseekV2Config)
        nn_model: DeepseekV2 | DistributedDeepseekV2
        if len(self.devices) > 1:
            nn_model = DistributedDeepseekV2(model_config)
            nn_model.load_state_dict(
                self._weights_dict, weight_alignment=1, strict=False
            )
        else:
            nn_model = DeepseekV2(model_config)
            nn_model.load_state_dict(self._weights_dict, weight_alignment=1)
        return {
            GraphRole.DECODER: GraphSpec(
                nn_model,
                "deepseekV2",
                nn_model.input_types(),
                weights=nn_model.state_dict(),
            )
        }

    @override
    @traced
    def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
        """Compiles and initializes the graphs :meth:`graph_specs` returns.

        Sets up the distributed runtime with :meth:`_init_distributed_runtime`
        first, since the graphs it builds depend on it, and keeps any graph
        it registers in :attr:`compiled_modules` too.

        Args:
            session: The session that compiles and initializes the graphs.

        Returns:
            The initialized models, keyed by role.
        """
        self._weights_dict = self._load_state_dict()
        model_config = self._create_model_config(self._weights_dict)
        self._init_distributed_runtime(session, model_config)
        with CompilationTimer("model") as timer:
            specs = self.graph_specs(model_config)
            timer.mark_build_complete()
            # `_init_distributed_runtime` may have registered a graph already.
            self.compiled_modules = {
                **self.compiled_modules,
                **compile_and_init_graphs(session, specs),
            }
        models = {
            role: self.compiled_modules[role].engine_model for role in specs
        }
        self._weights_dict = {}
        self._maybe_release_host_weights(*models.values())
        self._bind_ep_comm_initializer()
        return models
