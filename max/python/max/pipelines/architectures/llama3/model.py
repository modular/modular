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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.utils import (
    parse_state_dict_from_weights,
)
from max.profiler import traced

from .batch_processor import Llama3BatchProcessor
from .data_parallel_llama import create_graph as create_data_parallel_graph
from .distributed_llama import DistributedLlama3
from .llama3 import Llama3
from .model_config import Llama3Config

logger = logging.getLogger("max.pipelines")


@dataclass
class Llama3Inputs(ModelInputs):
    """A class representing inputs for the Llama3 model.

    This class encapsulates the input tensors required for the Llama3 model
    execution.
    """

    tokens: Buffer
    """Tensor containing the input token IDs."""

    input_row_offsets: Buffer
    """Tensor containing the offsets for each row in the ragged input
    sequence."""

    signal_buffers: list[Buffer]
    """Device buffers used for synchronization in communication collectives."""

    return_n_logits: Buffer

    data_parallel_splits: Buffer | Sequence[Sequence[int]] | None = None
    """Tensor containing the data parallel splits."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        if self.data_parallel_splits is not None:
            if isinstance(self.data_parallel_splits, Buffer):
                splits_tensor = self.data_parallel_splits
            else:
                splits_array = np.concatenate(
                    [
                        np.array(split, dtype=np.int64)
                        for split in self.data_parallel_splits
                    ]
                )
                splits_tensor = Buffer.from_numpy(splits_array).to(
                    self.tokens.device
                )
            return (
                self.tokens,
                self.input_row_offsets,
                self.return_n_logits,
                splits_tensor,
                *(
                    self.kv_cache_inputs.flatten()
                    if self.kv_cache_inputs is not None
                    else ()
                ),
            )

        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
        )


class LlamaModelBase(
    LogProbabilitiesMixin, PipelineModelWithKVCache[TextContext]
):
    """Base Llama pipeline model implementation."""

    model_config_cls: ClassVar[type[Any]] = Llama3Config
    batch_processor_cls: ClassVar[type[Llama3BatchProcessor]] = (
        Llama3BatchProcessor
    )

    model: Model
    """Compiled and initialized model ready for inference."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"]
    """Normalization layer."""

    attention_bias: bool = False
    """Whether to use attention bias."""

    state_dict: dict[str, Any]
    """Weights to load into the model."""

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
        """
        Args:
            pipeline_config: The configuration for this pipeline.
            session: The container for the runtime for this model.
        """
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
        self.model = self.load_model(session)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Llama3Inputs:
        """Delegates to the batch processor and narrows to ``Llama3Inputs``."""
        inputs = super().prepare_initial_token_inputs(
            replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )
        assert isinstance(inputs, Llama3Inputs)
        return inputs

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, Llama3Inputs)
        assert model_inputs.kv_cache_inputs is not None
        if self.pipeline_config.model.data_parallel_degree > 1:
            model_outputs = self.model.execute(*model_inputs.buffers)
        elif self._lora_manager:
            assert model_inputs.lora is not None
            model_outputs = self.model.execute(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
                model_inputs.return_n_logits,
                *model_inputs.lora.buffers(),
                *model_inputs.signal_buffers,
                *model_inputs.kv_cache_inputs.flatten(),
            )
        else:
            model_outputs = self.model.execute(*model_inputs.buffers)

        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        with CompilationTimer("model") as timer:
            graph = self._build_graph(self.weights, self.adapter)
            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)

        return model

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> Graph:
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, weights, adapter
        )
        model_config = Llama3Config.initialize(self.pipeline_config)
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )

        if model_config.data_parallel_degree > 1:
            graph, new_state_dict = create_data_parallel_graph(
                model_config, self.kv_params, state_dict
            )
            self.state_dict = new_state_dict
            return graph

        # Tensor Parallel case
        if len(self.devices) > 1:
            dist_model: DistributedLlama3 = DistributedLlama3(model_config)

            # Load weights.
            dist_model.load_state_dict(
                state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=False,  # TODO(MODELS-550) `rope_freqs.weight` not used
            )

            self.state_dict = dist_model.state_dict()

            with Graph(
                getattr(self.huggingface_config, "model_type", "llama3"),
                input_types=dist_model.input_types(self.kv_params),
            ) as graph:
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

                outputs = dist_model(
                    tokens.tensor,
                    signal_buffers,
                    kv_caches_per_dev,
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                )

                graph.output(*outputs)
                return graph

        # Single GPU case
        else:
            single_model: Llama3 = Llama3(model_config)

            if self._lora_manager:
                self._lora_manager.init_weights(single_model, state_dict)

            # Load weights.
            single_model.load_state_dict(
                state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
                strict=False,  # TODO(MODELS-550) `rope_freqs.weight` not used
            )
            self.state_dict = single_model.state_dict()

            with Graph(
                "llama3",
                input_types=single_model.input_types(
                    self.kv_params, self._lora_manager
                ),
            ) as graph:
                (
                    tokens,
                    input_row_offsets,
                    return_n_logits,
                    *rest,
                ) = graph.inputs
                if self._lora_manager:
                    rest = self._lora_manager.bind_graph_inputs(rest)
                kv_collections = self._unflatten_kv_inputs(rest)
                outputs = single_model(
                    tokens.tensor,
                    kv_collections[0],
                    return_n_logits.tensor,
                    input_row_offsets.tensor,
                )
                graph.output(*outputs)
                return graph


class Llama3Model(LlamaModelBase):
    """Llama 3 pipeline model implementation."""

    config_class: type[Llama3Config] = Llama3Config
    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    """Normalization layer."""

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
