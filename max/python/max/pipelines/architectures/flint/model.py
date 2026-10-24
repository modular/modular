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

"""Flint pipeline model for the ModuleV3 API."""

from __future__ import annotations

from typing import Any, ClassVar

from max.graph.weights import WeightData
from max.pipelines.context import TextContext
from max.pipelines.lib import ModuleV3PipelineModelWithKVCache
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin

from .batch_processor import FlintBatchProcessor
from .flint import Flint
from .layers.state_dict_cast import cast_state_dict
from .model_config import FlintConfig


class FlintModel(
    LogProbabilitiesMixin,
    ModuleV3PipelineModelWithKVCache[TextContext],
):
    """Builds and compiles the ModuleV3 network for Flint."""

    model_config_cls: ClassVar[type[Any]] = FlintConfig
    batch_processor_cls: ClassVar[type[FlintBatchProcessor]] = (
        FlintBatchProcessor
    )
    config_class: type[Any] = FlintConfig

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        model_config = self.config_class.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
            return_hidden_states=self.return_hidden_states,
        )
        return model_config

    def _instantiate_module(self, model_config: Any) -> Any:
        nn_model = Flint(model_config, self.kv_params)
        nn_model.to(self.devices[0])
        return nn_model

    def _prepare_state_dict(
        self, state_dict: dict[str, WeightData], model_config: Any
    ) -> dict[str, WeightData]:
        """Casts the floating-point weights to the module dtype.

        The ModuleV3 load path does not apply the default weights dtype
        cast, so the model applies it here.
        """
        return cast_state_dict(state_dict, model_config.dtype)
