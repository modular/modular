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

from typing import Any, ClassVar

from max.driver import Buffer, Device, is_virtual_device_mode
from max.engine import InferenceSession, Model
from max.graph import Graph, ops
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    SupportsSSMStateWarmup,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.modeling.types import RequestID
from typing_extensions import override

from .batch_processor import InklingBatchProcessor, InklingInputs
from .inkling import Inkling
from .model_config import InklingConfig
from .state_cache import InklingConvStateCache


class InklingModel(
    LogProbabilitiesMixin,
    GraphPipelineModelWithKVCache[TextContext],
    SupportsSSMStateWarmup,
):
    """Pipeline model for Inkling's text decoder."""

    batch_processor_cls: ClassVar[type[InklingBatchProcessor]] = (
        InklingBatchProcessor
    )
    model_config_cls: ClassVar[type[Any]] = InklingConfig

    model: Model
    state_dict: dict[str, Any]
    _nn_model: Inkling

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
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
            return_hidden_states,
            max_batch_size=max_batch_size,
        )
        self._state_cache: InklingConvStateCache | None = None
        self.model = self.load_model(session)

    @override
    def _wire_batch_processor(
        self, model: Any = None, model_config: Any = None
    ) -> None:
        super()._wire_batch_processor(model, model_config)
        # Compile-only runs cannot allocate on a virtual device.
        if not is_virtual_device_mode():
            # The memory-plan-resolved batch size, not the runtime config's
            # (often None).
            max_batch_size = self.max_batch_size
            assert max_batch_size is not None
            self._state_cache = InklingConvStateCache(
                self._nn_model.conv_layout,
                max_slots=max_batch_size,
                devices=self.devices,
            )
        assert isinstance(self._batch_processor, InklingBatchProcessor)
        self._batch_processor.bind_runtime_state(self._state_cache)

    @property
    def emits_folded_sampled_tokens(self) -> bool:
        return self.pipeline_config.runtime.fold_sampler_into_graph

    @override
    def _create_model_config(self, state_dict: dict[str, Any]) -> InklingConfig:
        # Quantization is read off the loaded weights; the checkpoint config
        # may not declare it.
        model_config = InklingConfig.initialize(self.pipeline_config)
        model_config.finalize(self.huggingface_config, state_dict)
        return model_config

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: InklingConfig,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        nn_model = Inkling(model_config, return_logits=self.return_logits)
        nn_model.load_state_dict(state_dict, weight_alignment=1)
        self._nn_model = nn_model

        with Graph("inkling", input_types=nn_model.input_types()) as graph:
            outputs = nn_model(*nn_model.unpack_inputs(graph.inputs))
            if self.emits_folded_sampled_tokens:
                # argmax is a pure device op (no host readback), so folding it
                # into the captured graph is capture-safe.
                sampled_tokens = ops.argmax(outputs[0], axis=-1)
                graph.output(*outputs, sampled_tokens)
            else:
                graph.output(*outputs)
        return graph, nn_model.state_dict()

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, InklingInputs)
        model_outputs = list(self.model.execute(*model_inputs.buffers))

        sampled_tokens: Buffer | None = None
        if self.emits_folded_sampled_tokens:
            popped = model_outputs.pop()
            assert isinstance(popped, Buffer)
            sampled_tokens = popped

        assert self._batch_processor is not None
        outputs = self._batch_processor.process_outputs(model_outputs)
        outputs.sampled_tokens = sampled_tokens
        return outputs

    def release(self, request_id: RequestID) -> None:
        """Drops the request's convolution state, freeing its slot."""
        if self._state_cache is not None:
            self._state_cache.release(request_id)

    def release_warmup_state(self, request_ids: list[RequestID]) -> None:
        """Frees the slots a graph-capture warmup probe claimed.

        Without this the second probe finds no free slot and serving never
        starts; ``claim`` zeros a slot when a real request takes it.
        """
        for request_id in request_ids:
            self.release(request_id)
