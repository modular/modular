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

import logging
import time
from typing import Any, ClassVar

from max.pipelines.context import TextContext
from max.pipelines.lib import ModuleV3PipelineModelWithKVCache
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin

from ..llama3_modulev3.batch_processor import Llama3ModuleV3BatchProcessor
from .model_config import NemotronHConfig
from .nemotron_h import NemotronH
from .weight_adapters import dequantize_to_bf16, stack_nvfp4_experts

logger = logging.getLogger("max.pipelines")


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
        modules = model_config.quant_scheme.quantized
        if modules:
            if mixers := model_config.w4a4_mixers():
                state_dict, modules = stack_nvfp4_experts(
                    state_dict, modules, mixers
                )
            start = time.perf_counter()
            state_dict = dequantize_to_bf16(state_dict, modules)
            logger.info(
                f"Nemotron-H: dequantized {len(modules)} modules to BF16 in"
                f" {time.perf_counter() - start:.1f}s"
            )
        return state_dict

    def _instantiate_module(self, model_config: NemotronHConfig) -> NemotronH:
        nn_model = NemotronH(model_config)
        nn_model.to(self.devices[0])
        return nn_model
