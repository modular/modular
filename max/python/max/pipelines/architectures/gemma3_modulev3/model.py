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
from typing import Any, ClassVar

from max import tree
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    ModelInputs,
    ModelOutputs,
    ModuleV3PipelineModelWithKVCache,
)
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin
from transformers import AutoConfig

from .batch_processor import Gemma3ModuleV3BatchProcessor
from .gemma3 import Gemma3
from .inputs import Gemma3Inputs
from .model_config import Gemma3Config

logger = logging.getLogger("max.pipelines")


class Gemma3Model(
    LogProbabilitiesMixin,
    ModuleV3PipelineModelWithKVCache[TextContext],
):
    """A Gemma3 pipeline model for text generation using the ModuleV3 API.

    This class integrates the Gemma3 architecture with the MAX Engine pipeline
    infrastructure using the V3 eager compilation API.
    """

    model_config_cls: ClassVar[type[Any]] = Gemma3Config
    batch_processor_cls: ClassVar[type[Gemma3ModuleV3BatchProcessor]] = (
        Gemma3ModuleV3BatchProcessor
    )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return Gemma3Config.get_num_layers(huggingface_config)

    @property
    def _is_multimodal(self) -> bool:
        return hasattr(self.huggingface_config, "text_config")

    def _hf_config_for_weights(self) -> AutoConfig | None:
        if self._is_multimodal:
            return self.huggingface_config.text_config
        return self.huggingface_config

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        text_config = (
            self.huggingface_config.text_config
            if self._is_multimodal
            else self.huggingface_config
        )
        model_config = Gemma3Config.initialize_from_config(
            self.pipeline_config, text_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=text_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )
        return model_config

    def _instantiate_module(self, model_config: Any) -> Any:
        n_devices = len(self.devices)
        mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))
        with default_device(mesh):
            return Gemma3(model_config, self.kv_params)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Executes the Gemma3 model with the prepared inputs."""
        assert isinstance(model_inputs, Gemma3Inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs
        assert curr_kv_cache_inputs is not None

        model_outputs = self.model(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            model_inputs.input_row_offsets,
            *tree.leaves(curr_kv_cache_inputs),
        )
        return self._to_model_outputs(model_outputs)
