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

"""Implements the DeepSeek-V4-Flash PipelineModel."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from max import tree
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph import Graph
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs, MultiKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.interfaces.batch_processor import (
    SingleReplicaRaggedBatchProcessor,
)
from max.pipelines.lib.memory_estimation import MemoryPlan

from .deepseekV4 import DeepseekV4
from .layers import DeepseekV4Cache
from .model_config import DeepseekV4Config

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV4Inputs(ModelInputs):
    """Inputs for one DeepSeek-V4 model execution."""

    tokens: Buffer
    """Input token IDs, ragged."""

    input_row_offsets: Buffer
    """Offsets of each sequence in the ragged ``tokens``."""

    return_n_logits: Buffer
    """Number of trailing logits to return."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *tree.leaves(self.kv_cache_inputs),
        )


class DeepseekV4BatchProcessor(
    SingleReplicaRaggedBatchProcessor[TextContext, DeepseekV4Inputs]
):
    """Ragged single-replica batching; the stock ragged KV input order."""

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        signal_buffers: list[Buffer],
    ) -> DeepseekV4Inputs:
        del signal_buffers
        return DeepseekV4Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
        )


class DeepseekV4Model(GraphPipelineModelWithKVCache[TextContext]):
    """A DeepSeek-V4-Flash pipeline model for text generation."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV4Config
    batch_processor_cls: ClassVar[type[DeepseekV4BatchProcessor]] = (
        DeepseekV4BatchProcessor
    )

    model: Model
    """The compiled and initialized MAX Engine model."""

    _strict_state_dict_loading = True

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
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        if len(devices) != 1:
            raise ValueError(
                "DeepSeek-V4 bringup is single-device only; got "
                f"{len(devices)} devices"
            )
        self.model = self.load_model(session)

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV4Config:
        model_config = self.arch_config_as(DeepseekV4Config)
        model_config.return_logits = self.return_logits
        model_config.return_hidden_states = self.return_hidden_states
        # ``norm.weight`` is the one norm every V4 checkpoint has, hash-routed
        # or not, so it is the safe probe for the norm storage dtype.
        model_config.norm_dtype = state_dict["norm.weight"].dtype
        # The adapter drops the ``mtp.*`` weights, so the stages are not built.
        model_config.dspark_stages = False
        return model_config

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: DeepseekV4Config,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        nn_model = DeepseekV4(model_config)
        nn_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        weights_registry = nn_model.state_dict(auto_initialize=False)

        assert self.batch_processor is not None
        input_types = self.batch_processor.get_symbolic_inputs(
            kv_params=self.kv_params, device_refs=self.device_refs
        )
        with Graph("deepseekV4", input_types=input_types) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic_args = (
                graph.inputs
            )
            assert isinstance(self.kv_params, MultiKVCacheParams)
            cache = DeepseekV4Cache.from_groups(
                model_config,
                self.kv_params.unflatten_basic_kv_tree(iter(variadic_args)),
            )
            outputs = nn_model.serve(
                tokens.tensor,
                input_row_offsets.tensor,
                return_n_logits.tensor,
                cache,
            )
            graph.output(*outputs)
        return graph, weights_registry

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(DeepseekV4Inputs, model_inputs)
        model_outputs = self.model.execute(*model_inputs.buffers)
        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)
