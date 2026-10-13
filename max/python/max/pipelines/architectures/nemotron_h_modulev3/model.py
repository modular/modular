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

from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device
from max.pipelines.context import TextContext
from max.pipelines.lib import ModuleV3PipelineModelWithKVCache
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin

from ..llama3_modulev3.batch_processor import Llama3ModuleV3BatchProcessor
from .model_config import LayerKind, NemotronHConfig
from .nemotron_h import NemotronH
from .weight_adapters import (
    permute_mamba_for_tp,
    prepare_nvfp4_linears,
    repeat_kv_heads_for_tp,
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

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> NemotronHConfig:
        config = self.arch_config_as(NemotronHConfig)
        config.return_logits = self.return_logits
        config.quant_scheme.check_weights(state_dict.keys())
        return config

    def _prepare_state_dict(
        self, state_dict: dict[str, Any], model_config: NemotronHConfig
    ) -> dict[str, Any]:
        num_experts = model_config.num_experts
        w4a4_mixers = model_config.w4a4_mixers()
        n = len(self.devices)
        state_dict = stack_nvfp4_experts(
            state_dict,
            num_experts,
            {m: model_config.shared_expert_slices(m) for m in w4a4_mixers},
            n,
        )
        state_dict = stack_bf16_experts(
            state_dict,
            num_experts,
            model_config.mixers(LayerKind.MOE) - w4a4_mixers,
            n,
        )
        state_dict = permute_mamba_for_tp(state_dict, model_config, n)
        state_dict = prepare_nvfp4_linears(state_dict, model_config, n)
        return repeat_kv_heads_for_tp(state_dict, model_config, n)

    def _instantiate_module(self, model_config: NemotronHConfig) -> NemotronH:
        n_devices = len(self.devices)
        mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))
        with default_device(mesh):
            return NemotronH(model_config)
