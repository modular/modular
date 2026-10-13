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
from typing import Any, ClassVar, Literal

from max.pipelines.context import TextContext
from max.pipelines.lib import ModuleV3PipelineModelWithKVCache
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from max.pipelines.lora import LoRATargetModule

from .batch_processor import Llama3ModuleV3BatchProcessor
from .llama3 import Llama3
from .lora import LLAMA3_LORA_TARGETS
from .model_config import Llama3Config

logger = logging.getLogger("max.pipelines")


class Llama3Model(
    LogProbabilitiesMixin,
    ModuleV3PipelineModelWithKVCache[TextContext],
):
    """Llama3 pipeline model using the ModuleV3 API."""

    model_config_cls: ClassVar[type[Any]] = Llama3Config
    batch_processor_cls: ClassVar[type[Llama3ModuleV3BatchProcessor]] = (
        Llama3ModuleV3BatchProcessor
    )

    config_class: type[Any] = Llama3Config
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    attention_bias: bool = False

    #: Serve LoRA via the ModuleV3 adapters-as-inputs path (LoRAManagerV3).
    lora_modulev3: ClassVar[bool] = True
    #: The projections ModuleV3 LoRA wraps: the fused qkv (one adapter per
    #: q/k/v) and o_proj.
    lora_targets: ClassVar[tuple[LoRATargetModule, ...]] = LLAMA3_LORA_TARGETS

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        model_config = self.config_class.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )
        return model_config

    def _instantiate_module(self, model_config: Any) -> Any:
        nn_model = Llama3(model_config, self.kv_params)
        nn_model.to(self.devices[0])
        return nn_model
