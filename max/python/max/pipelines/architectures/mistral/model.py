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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import BufferType, Graph, TensorType
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.nn.layer import Module
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.utils import (
    parse_state_dict_from_weights,
    upper_bounded_default,
)
from max.profiler import traced
from transformers import AutoConfig

from .batch_processor import MistralBatchProcessor
from .distributed_mistral import DistributedMistral
from .mistral import Mistral
from .model_config import MistralConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class MistralInputs(ModelInputs):
    """A class representing inputs for the Mistral model.

    This class encapsulates the input tensors required for the Mistral model execution:
    - tokens: A tensor containing the input token IDs
    - input_row_offsets: A tensor containing the offsets for each row in the ragged input sequence
    - return_n_logits: A tensor containing the number of expected token logits.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""
    return_n_logits: Buffer


class MistralModel(PipelineModelWithKVCache[TextContext]):
    model_config_cls: ClassVar[type[Any]] = MistralConfig
    batch_processor_cls: ClassVar[type[MistralBatchProcessor]] = (
        MistralBatchProcessor
    )

    @classmethod
    def calculate_max_seq_len(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Bounds ``max_length`` by ``max_position_embeddings`` (config is permissive)."""
        try:
            return upper_bounded_default(
                upper_bound=huggingface_config.max_position_embeddings,
                default=pipeline_config.model.max_length,
            )
        except ValueError as e:
            raise ValueError(
                f"Unable to infer max_length for {cls.__qualname__}, "
                f"the provided max_length ({pipeline_config.model.max_length}) "
                f"exceeds the model's max_position_embeddings "
                f"({huggingface_config.max_position_embeddings})."
            ) from e

    model: Model
    """Compiled and initialized model ready for inference."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
        self.model = self.load_model(session)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Runs the graph."""
        assert isinstance(model_inputs, MistralInputs)

        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs.flatten(),
        )
        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)

    def graph_inputs(self) -> tuple[TensorType | BufferType, ...]:
        assert self.batch_processor is not None
        return tuple(
            self.batch_processor.get_symbolic_inputs(
                kv_params=self.kv_params,
                device_refs=self.device_refs,
            )
        )

    @traced
    def _build_graph(
        self, weights: Weights, adapter: WeightsAdapter | None = None
    ) -> Graph:
        text_config = getattr(
            self.huggingface_config, "text_config", self.huggingface_config
        )
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config,
            weights,
            adapter,
            hf_config=text_config,
        )

        model_config = MistralConfig.initialize_from_config(
            self.pipeline_config,
            text_config,
        )
        model_config.return_logits = self.return_logits

        # Get Graph Inputs
        graph_inputs = self.graph_inputs()

        # Build Graph
        nn_model: Module
        if len(self.devices) > 1:
            nn_model = DistributedMistral(model_config)
            nn_model.load_state_dict(
                state_dict,
                weight_alignment=1,
                strict=False,  # TODO(MODELS-551) vision tower weights not used
            )
            self.state_dict = nn_model.state_dict()

            with Graph("mistral", input_types=[*graph_inputs]) as graph:
                tokens, input_row_offsets, return_n_logits, *variadic_args = (
                    graph.inputs
                )

                # Multi-GPU passes a signal buffer per device: unmarshal these.
                signal_buffers = [
                    v.buffer for v in variadic_args[: len(self.devices)]
                ]

                # Unmarshal the remaining arguments, which are for KV cache.
                kv_caches_per_dev = self._unflatten_kv_inputs(
                    variadic_args[len(self.devices) :]
                )

                outputs = nn_model(
                    tokens.tensor,
                    signal_buffers,
                    kv_caches_per_dev,
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                )

                graph.output(*outputs)
                return graph

        else:
            nn_model = Mistral(model_config)
            nn_model.load_state_dict(
                state_dict,
                weight_alignment=1,
                strict=False,  # TODO(MODELS-551) vision tower weights not used
            )
            self.state_dict = nn_model.state_dict()

            with Graph("mistral", input_types=graph_inputs) as graph:
                tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                    graph.inputs
                )
                kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)
                outputs = nn_model(
                    tokens.tensor,
                    kv_collections[0],
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                )
                graph.output(*outputs)
                return graph

    @traced
    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        if self.pipeline_config.model.enable_echo:
            raise ValueError(
                "Mistral model does not currently implement enable echo."
            )

        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "only safetensors weights are currently supported in Mistral models."
            )

        with CompilationTimer("model") as timer:
            graph = self._build_graph(self.weights, self.adapter)
            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)

        return model
