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

import numpy as np
import numpy.typing as npt
from max import tree
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan

from .deepseekV4 import DeepseekV4
from .model_config import DeepseekV4Config

logger = logging.getLogger("max.pipelines")


@dataclass
class DeepseekV4Inputs(ModelInputs):
    """Inputs for one DeepSeek-V4 model execution."""

    tokens: npt.NDArray[np.integer[Any]] | Buffer
    """Input token IDs, ragged."""

    input_row_offsets: npt.NDArray[np.integer[Any]] | Buffer
    """Offsets of each sequence in the ragged ``tokens``."""

    return_n_logits: Buffer
    """Number of trailing logits to return."""


class DeepseekV4Model(GraphPipelineModelWithKVCache[TextContext]):
    """A DeepSeek-V4-Flash pipeline model for text generation."""

    model_config_cls: ClassVar[type[Any]] = DeepseekV4Config

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
        max_batch_total_tokens = self.planned_max_batch_total_tokens
        assert max_batch_total_tokens is not None, "max_length must be set"
        model_config.max_batch_context_length = max_batch_total_tokens
        return model_config

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: DeepseekV4Config,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        nn_model = DeepseekV4(model_config)
        nn_model.load_state_dict(
            state_dict,
            weight_alignment=1,
            strict=self._strict_state_dict_loading,
        )
        weights_registry = nn_model.state_dict(auto_initialize=False)

        flattened_kv_types = self.kv_params.flattened_kv_inputs()

        with Graph(
            "deepseekV4",
            input_types=[
                tokens_type,
                return_n_logits_type,
                input_row_offsets_type,
                *flattened_kv_types,
            ],
        ) as graph:
            _tokens, return_n_logits, input_row_offsets, *variadic_args = (
                graph.inputs
            )
            # ``kv_params`` is a ``MultiKVCacheParams`` with an ``mla`` group
            # and an ``indexer`` group, so this unflattens into one list per
            # group, each holding one entry per device.
            attn_kv_collections, indexer_kv_collections = (
                self.kv_params.unflatten_basic_kv_tree(iter(variadic_args))
            )
            del attn_kv_collections, indexer_kv_collections
            del return_n_logits, input_row_offsets
            # The serving graph needs the ragged and cache plumbing that the
            # decode path is blocked on: V4 appends one compressed KV entry
            # per ``compress_ratio`` tokens and MAX's paged cache indexes
            # slots by token position, with no stride mode
            # (.agent/backlogs/192/ISSUES.md Issue 30). ``DeepseekV4.__call__``
            # is the prefill path and takes a padded ``[batch, seq]`` instead;
            # that is what the logit verification drives.
            raise NotImplementedError(
                "DeepSeek-V4 serving needs the compressed KV cache; prefill "
                "runs through DeepseekV4.__call__"
            )
        return graph, weights_registry

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(DeepseekV4Inputs, model_inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None

        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            model_inputs.input_row_offsets,
            *tree.leaves(curr_kv_cache_inputs),
        )
        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=cast(Buffer, model_outputs[1]),
                next_token_logits=cast(Buffer, model_outputs[0]),
                logit_offsets=cast(Buffer, model_outputs[2]),
            )
        return ModelOutputs(
            logits=cast(Buffer, model_outputs[0]),
            next_token_logits=cast(Buffer, model_outputs[0]),
        )
