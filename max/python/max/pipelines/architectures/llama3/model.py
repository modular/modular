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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.graph_specs import (
    GraphRole,
    GraphSpec,
    compile_and_init_graphs,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.profiler import traced

from .batch_processor import Llama3BatchProcessor
from .data_parallel_llama import create_model as create_data_parallel_model
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
                    tree.leaves(self.kv_cache_inputs)
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
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs is not None
                else ()
            ),
        )


def _load_weights(module: Module, state_dict: Mapping[str, Any]) -> None:
    module.load_state_dict(
        state_dict,
        override_quantization_encoding=True,
        weight_alignment=1,
        strict=False,  # TODO(MODELS-550) `rope_freqs.weight` not used
    )


class LlamaModelBase(
    LogProbabilitiesMixin,
    GraphPipelineModelWithKVCache[TextContext],
):
    """Base Llama pipeline model implementation."""

    model_config_cls: ClassVar[type[Any]] = Llama3Config
    batch_processor_cls: ClassVar[type[Llama3BatchProcessor]] = (
        Llama3BatchProcessor
    )

    model: Model
    """Compiled and initialized model ready for inference."""

    norm_method: Literal["rms_norm", "layer_norm"]
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
        *,
        memory_plan: MemoryPlan,
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
            adapter=adapter,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self.model = self.load_model(session)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
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
        model_outputs = self.model.execute(*model_inputs.buffers)

        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)

    @classmethod
    def _finalized_config(
        cls, context: ModelLoadContext, state_dict: dict[str, Any]
    ) -> Llama3Config:
        config = Llama3Config.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        config.finalize(
            huggingface_config=context.huggingface_config,
            state_dict=state_dict,
            norm_method=cls.norm_method,
            attention_bias=cls.attention_bias,
            return_logits=context.return_logits,
            return_hidden_states=context.return_hidden_states,
        )
        if context.target_layer_ids:
            config.target_layer_ids = list(context.target_layer_ids)
        return config

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        return self._finalized_config(
            ModelLoadContext.from_pipeline_model(self), state_dict
        )

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        """Loads the decoder graph :meth:`graph_specs` returns.

        A subclass that still overrides ``_build_graph_for_compile`` loads
        through that hook instead.

        Args:
            session: The session that compiles and initializes the graph.

        Returns:
            The initialized decoder.
        """
        # TODO(kathywu): Drop once every subclass implements graph_specs.
        if (
            type(self)._build_graph_for_compile
            is not GraphPipelineModelWithKVCache._build_graph_for_compile
        ):
            return super().load_model(session)
        return self.load_graphs(session)[GraphRole.DECODER]

    @traced
    def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
        """Compiles and initializes the graphs :meth:`graph_specs` returns.

        Sets up the distributed runtime with ``_init_distributed_runtime``
        first, since the graphs depend on it, and wires the batch processor
        with ``_wire_batch_processor`` after.

        Args:
            session: The session that compiles and initializes the graphs.

        Returns:
            The initialized models, keyed by role.
        """
        state_dict = self._load_state_dict()
        model_config = self._create_model_config(state_dict)
        self._init_distributed_runtime(session, model_config)
        with CompilationTimer("model") as timer:
            specs = self.graph_specs(model_config)
            timer.mark_build_complete()
            self.compiled_modules = compile_and_init_graphs(session, specs)
        models = {
            role: compiled.engine_model
            for role, compiled in self.compiled_modules.items()
        }
        self._maybe_release_host_weights(*models.values())
        self._wire_batch_processor(models[GraphRole.DECODER], model_config)
        return models

    def _load_state_dict(self) -> dict[str, Any]:
        self.state_dict = super()._load_state_dict()
        return self.state_dict

    def graph_specs(self, model_config: Any) -> dict[str, GraphSpec]:
        """Returns the decoder: data-parallel when the config asks for it,
        else tensor-parallel across several devices, else single-device."""
        assert isinstance(model_config, Llama3Config)
        if model_config.data_parallel_degree > 1:
            spec = self._data_parallel_spec(model_config)
        elif len(self.devices) > 1:
            spec = self._tensor_parallel_spec(model_config)
        else:
            spec = self._single_device_spec(model_config)
        return {GraphRole.DECODER: spec}

    def _data_parallel_spec(self, model_config: Llama3Config) -> GraphSpec:
        dp_model = create_data_parallel_model(model_config, self.state_dict)
        # The weights registry is read after tracing, as it always was.
        return GraphSpec(
            dp_model, "llama3", dp_model.input_types(self.kv_params)
        )

    def _tensor_parallel_spec(self, model_config: Llama3Config) -> GraphSpec:
        dist_model = DistributedLlama3(model_config)
        _load_weights(dist_model, self.state_dict)
        return GraphSpec(
            dist_model,
            getattr(self.huggingface_config, "model_type", "llama3"),
            self._bind_graph_inputs(dist_model.input_tree(self.kv_params)),
            weights=dist_model.state_dict(),
        )

    def _single_device_spec(self, model_config: Llama3Config) -> GraphSpec:
        single_model = Llama3(model_config)
        _load_weights(single_model, self.state_dict)
        return GraphSpec(
            single_model,
            "llama3",
            self._bind_graph_inputs(single_model.input_tree(self.kv_params)),
            weights=single_model.state_dict(),
        )


class Llama3Model(LlamaModelBase):
    """Llama 3 pipeline model implementation."""

    config_class: type[Llama3Config] = Llama3Config
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    """Normalization layer."""

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

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[Module, Llama3Config]:
        """Builds the Llama 3 module on one device or tensor-parallel.

        Defined here rather than on :class:`LlamaModelBase`, whose other
        subclasses build modules of their own.

        Raises:
            ValueError: If the config is data-parallel, whose replicas the
                data-parallel graph builds itself.
        """
        state_dict = parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
        config = cls._finalized_config(context, state_dict)
        if config.data_parallel_degree > 1:
            raise ValueError(
                "Llama 3 builds data-parallel replicas inside its graph;"
                " there is no single module to return."
            )
        module: Module = (
            DistributedLlama3(config)
            if len(config.devices) > 1
            else Llama3(config)
        )
        _load_weights(module, state_dict)
        return module, config
