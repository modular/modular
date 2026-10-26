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
"""The Nemotron-H pipeline model."""

from __future__ import annotations

from typing import Any, ClassVar

from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device, default_dtype
from max.nn.kv_cache import MultiKVCacheParams
from max.pipelines.context import TextContext
from max.pipelines.lib import ModuleV3PipelineModelWithKVCache
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lib.utils import parse_state_dict_from_weights

from ..llama3_modulev3.batch_processor import Llama3ModuleV3BatchProcessor
from .model_config import LayerKind, NemotronHConfig
from .nemotron_h import NemotronH
from .weight_adapters import (
    cast_router_gates,
    stack_bf16_experts,
    stack_nvfp4_experts,
)


class NemotronHModel(
    LogProbabilitiesMixin, ModuleV3PipelineModelWithKVCache[TextContext]
):
    """Nemotron-H with its Mamba state on the KV cache."""

    model_config_cls: ClassVar[type[Any]] = NemotronHConfig
    batch_processor_cls: ClassVar[type[Llama3ModuleV3BatchProcessor]] = (
        Llama3ModuleV3BatchProcessor
    )

    @classmethod
    def build_module(
        cls, context: ModelLoadContext
    ) -> tuple[NemotronH, NemotronHConfig]:
        """Builds Nemotron-H for the checkpoint ``context`` names.

        The module holds no weights: it compiles with those
        :meth:`checkpoint_weights` returns. A caller that sized the cache
        passes its KV params. Under speculative decoding each Mamba layer
        also returns the inputs of its state updates, which the verify's
        replay reads.
        """
        config = NemotronHConfig.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        if context.kv_params is not None:
            assert isinstance(context.kv_params, MultiKVCacheParams)
            config.kv_params = context.kv_params
        config.return_logits = context.return_logits
        config.return_hidden_states = context.return_hidden_states
        # A draft names the layers whose outputs it reads, and Nemotron-H
        # taps layer inputs: a layer's output is the next layer's input.
        config.target_layer_ids = tuple(i + 1 for i in context.target_layer_ids)
        config.capture_mamba_replay = (
            context.pipeline_config.speculative is not None
        )
        devices = [ref.to_device() for ref in config.devices]
        mesh = DeviceMesh(tuple(devices), (len(devices),), ("tp",))
        with F.lazy(), default_dtype(config.dtype), default_device(mesh):
            module = NemotronH(config)
        return module, config

    @classmethod
    def checkpoint_weights(
        cls, context: ModelLoadContext, config: NemotronHConfig
    ) -> dict[str, Any]:
        """Returns the weights the built module compiles with.

        ``Module.compile`` shards them inside the graph. Weights loaded into
        the module under ``F.lazy()`` would instead be lazy shards, which
        the session can't realize across devices.
        """
        state_dict = parse_state_dict_from_weights(
            context.pipeline_config, context.weights, context.adapter
        )
        config.quant_scheme.check_weights(state_dict.keys())
        return _prepare_state_dict(state_dict, config)

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> NemotronHConfig:
        config = self.arch_config_as(NemotronHConfig)
        config.return_logits = self.return_logits
        config.return_hidden_states = self.return_hidden_states
        config.quant_scheme.check_weights(state_dict.keys())
        return config

    def _prepare_state_dict(
        self, state_dict: dict[str, Any], model_config: NemotronHConfig
    ) -> dict[str, Any]:
        return _prepare_state_dict(state_dict, model_config)

    def _instantiate_module(self, model_config: NemotronHConfig) -> NemotronH:
        n_devices = len(self.devices)
        mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))
        with default_device(mesh):
            return NemotronH(model_config)


def _prepare_state_dict(
    state_dict: dict[str, Any], config: NemotronHConfig
) -> dict[str, Any]:
    """Stacks the experts and casts the router gates as the module reads them."""
    num_experts = config.num_experts
    w4a4_mixers = config.w4a4_mixers()
    state_dict = stack_nvfp4_experts(state_dict, num_experts, w4a4_mixers)
    state_dict = stack_bf16_experts(
        state_dict,
        num_experts,
        config.mixers(LayerKind.MOE) - w4a4_mixers,
    )
    return cast_router_gates(state_dict, config.router_gate_dtype)
