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
"""The LFM3 pipeline model."""

from __future__ import annotations

from typing import Any, ClassVar

from max.pipelines.context import TextContext
from max.pipelines.lib import ModuleV3PipelineModelWithKVCache
from max.pipelines.lib.log_probabilities import LogProbabilitiesMixin

from ..llama3_modulev3.batch_processor import Llama3ModuleV3BatchProcessor
from .lfm3 import Lfm3
from .model_config import Lfm3Config


class Lfm3Model(
    LogProbabilitiesMixin, ModuleV3PipelineModelWithKVCache[TextContext]
):
    """LFM3 with its conv windows on the KV cache."""

    model_config_cls: ClassVar[type[Any]] = Lfm3Config
    batch_processor_cls: ClassVar[type[Llama3ModuleV3BatchProcessor]] = (
        Llama3ModuleV3BatchProcessor
    )

    def _create_model_config(self, state_dict: dict[str, Any]) -> Lfm3Config:
        config = Lfm3Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        config.return_logits = self.return_logits
        return config

    def _instantiate_module(self, model_config: Lfm3Config) -> Lfm3:
        module = Lfm3(model_config)
        module.to(self.devices[0])
        return module
