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
"""Inkling pipeline model: input preparation, cache plumbing, execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from max.driver import Device
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import WeightData, Weights, WeightsAdapter
from max.nn.kv_cache import MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
    SupportsSSMStateWarmup,
)
from max.pipelines.lib.graph_specs import (
    GraphRole,
    GraphSpec,
    compile_and_init_graphs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.profiler import traced
from typing_extensions import override

from .batch_processor import InklingBatchProcessor, InklingInputs
from .inkling import Inkling
from .layers.vision import InklingVisionModel
from .model_config import InklingConfig
from .processor import load_processor_config
from .weight_adapters import _TEXT, _VISION, VISION_PREFIX


class InklingModel(
    LogProbabilitiesMixin,
    PipelineModelWithKVCache[TextAndVisionContext, InklingConfig, Model | None],
    SupportsSSMStateWarmup,
):
    """Pipeline model for Inkling's decoder and its vision tower."""

    batch_processor_cls: ClassVar[type[InklingBatchProcessor]] = (
        InklingBatchProcessor
    )
    model_config_cls: ClassVar[type[Any]] = InklingConfig
    _compile_group: ClassVar[str] = "inkling"
    """The group the vision and language graphs compile into one artifact
    under."""

    model: Model
    _nn_model: Inkling
    _vision_weights_dict: dict[str, Any]
    _language_weights_dict: dict[str, Any]

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
        # The vision tower is only ever called through the batch processor,
        # which _wire_batch_processor hands it to.
        self.model = self.load_graphs(session)[GraphRole.DECODER]

    @override
    def _wire_batch_processor(
        self, model: Model | None, model_config: InklingConfig
    ) -> None:
        super()._wire_batch_processor(model, model_config)
        assert model is not None
        assert isinstance(self._batch_processor, InklingBatchProcessor)
        self._batch_processor.bind_vision_encoder(vision_model=model)

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        """Splits the checkpoint between the two graphs the towers compile to."""
        state_dict = super()._load_state_dict()
        self._vision_weights_dict = {
            name.removeprefix(VISION_PREFIX): data
            for name, data in state_dict.items()
            if name.startswith(VISION_PREFIX)
        }
        self._language_weights_dict = {
            name: data
            for name, data in state_dict.items()
            if not name.startswith(VISION_PREFIX)
        }
        return state_dict

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> InklingConfig:
        return _finalized_config(
            self.pipeline_config,
            self.huggingface_config,
            state_dict,
            max_seq_len=self.max_seq_len,
        )

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Inkling, InklingConfig]:
        """Builds the text model from the checkpoint ``context`` names.

        It merges in the image rows a caller's :meth:`build_vision_graph`
        encodes. A caller that sized the cache passes its KV params.
        """
        state_dict = _adapted(context, _TEXT)
        config = _finalized_config(
            context.pipeline_config,
            context.huggingface_config,
            state_dict,
            max_seq_len=context.max_seq_len,
        )
        if context.kv_params is not None:
            assert isinstance(context.kv_params, MultiKVCacheParams)
            config.kv_params = context.kv_params
        module = Inkling(
            config,
            return_logits=context.return_logits,
            return_hidden_states=context.return_hidden_states,
        )
        module.load_state_dict(state_dict, weight_alignment=1, strict=True)
        return module, config

    @classmethod
    def build_vision_graph(
        cls, context: ModelLoadContext
    ) -> tuple[Graph, dict[str, Any]]:
        """Builds the vision tower's graph and the weights it loads, for a
        caller that runs the text model itself."""
        config = InklingConfig.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        state_dict = {
            name.removeprefix(VISION_PREFIX): data
            for name, data in _adapted(context, _VISION).items()
        }
        vision_model = _vision_model(
            config, context.pipeline_config, state_dict
        )
        spec = GraphSpec(
            vision_model,
            "inkling_vision",
            vision_model.input_types(),
            weights=vision_model.state_dict(),
        )
        return spec.trace(), dict(spec.weights_registry())

    @override
    @traced
    def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
        """Compiles and initializes the graphs :meth:`graph_specs` returns,
        then hands the vision tower to the batch processor.

        Args:
            session: The session that compiles and initializes the graphs.

        Returns:
            The initialized models, keyed by role.
        """
        state_dict = self._load_state_dict()
        model_config = self._create_model_config(state_dict)
        with CompilationTimer("model") as timer:
            specs = self.graph_specs(model_config)
            timer.mark_build_complete()
            self.compiled_modules = compile_and_init_graphs(session, specs)
        models = {
            role: compiled.engine_model
            for role, compiled in self.compiled_modules.items()
        }
        self._maybe_release_host_weights(*models.values())
        self._wire_batch_processor(models[GraphRole.VISION], model_config)
        return models

    def graph_specs(self, model_config: InklingConfig) -> dict[str, GraphSpec]:
        """Returns the vision tower and the decoder, in one compile group."""
        return {
            GraphRole.VISION: self._vision_spec(
                model_config, self._vision_weights_dict
            ),
            GraphRole.DECODER: self._language_spec(
                model_config, self._language_weights_dict
            ),
        }

    def _vision_spec(
        self, model_config: InklingConfig, state_dict: dict[str, Any]
    ) -> GraphSpec:
        vision_model = _vision_model(
            model_config, self.pipeline_config, state_dict
        )
        return GraphSpec(
            vision_model,
            "inkling_vision",
            vision_model.input_types(),
            weights=vision_model.state_dict(),
            group=self._compile_group,
        )

    def _language_spec(
        self, model_config: InklingConfig, state_dict: dict[str, Any]
    ) -> GraphSpec:
        nn_model = Inkling(model_config, return_logits=self.return_logits)
        nn_model.load_state_dict(state_dict, weight_alignment=1)
        self._nn_model = nn_model
        return GraphSpec(
            nn_model,
            "inkling",
            nn_model.input_types(),
            weights=nn_model.state_dict(),
            group=self._compile_group,
        )

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, InklingInputs)
        model_outputs = list(self.model.execute(*model_inputs.buffers))

        assert self._batch_processor is not None
        return self._batch_processor.process_outputs(model_outputs)


def _adapted(context: ModelLoadContext, prefix: str) -> dict[str, WeightData]:
    """Adapts only the checkpoint's weights whose names start with
    ``prefix``."""
    assert context.adapter is not None
    return context.adapter(
        {
            name: weight
            for name, weight in context.weights.items()
            if name.startswith(prefix)
        },
        huggingface_config=context.huggingface_config,
        pipeline_config=context.pipeline_config,
    )


def _finalized_config(
    pipeline_config: PipelineConfig,
    huggingface_config: Any,
    state_dict: Mapping[str, WeightData],
    *,
    max_seq_len: int,
) -> InklingConfig:
    # Quantization is read off the loaded weights; the checkpoint config
    # may not declare it.
    config = InklingConfig.initialize(pipeline_config, max_seq_len=max_seq_len)
    config.finalize(huggingface_config, state_dict)
    return config


def _vision_model(
    config: InklingConfig,
    pipeline_config: PipelineConfig,
    state_dict: dict[str, Any],
) -> InklingVisionModel:
    image_processor = load_processor_config(
        pipeline_config.model.model_path,
        pipeline_config.model.huggingface_model_revision,
    )["image_processor"]
    # The tower is small and runs once per image, so it stays on one device.
    vision_model = InklingVisionModel(
        config.vision_config,
        config.dtype,
        config.devices[0],
        image_mean=image_processor["image_mean"],
        image_std=image_processor["image_std"],
    )
    vision_model.load_state_dict(state_dict, weight_alignment=1)
    return vision_model
