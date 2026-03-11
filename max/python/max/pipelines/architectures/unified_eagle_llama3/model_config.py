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
"""Config for EAGLE Llama3 draft models."""

from __future__ import annotations

from dataclasses import dataclass

from max.nn import ReturnHiddenStates
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.pipelines.lib.config import PipelineConfig
from typing_extensions import Self

from ..llama3.model_config import ArchConfigWithKVCache, Llama3Config


@dataclass(kw_only=True)
class UnifiedEagleLlama3Config(ArchConfigWithKVCache):
    target: Llama3Config
    draft: Llama3Config

    def __post_init__(self) -> None:
        self.target.return_hidden_states = ReturnHiddenStates.ALL_NORMALIZED
        self.draft.return_hidden_states = ReturnHiddenStates.LAST

        if len(self.target.devices) != len(self.draft.devices):
            raise ValueError(
                f"Target and draft must have the same number of devices. Found {len(self.target.devices)} and {len(self.draft.devices)} respectively."
            )

    def get_kv_params(self) -> KVCacheParamInterface:
        target_kv_params = self.target.get_kv_params()
        draft_kv_params = self.draft.get_kv_params()
        return MultiKVCacheParams.from_params(target_kv_params, draft_kv_params)

    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        assert pipeline_config.model.huggingface_config is not None
        assert pipeline_config.draft_model is not None
        assert pipeline_config.draft_model.huggingface_config is not None
        target_config = Llama3Config.initialize_from_config(
            pipeline_config, pipeline_config.model.huggingface_config
        )
        draft_config = Llama3Config.initialize_from_config(
            pipeline_config, pipeline_config.draft_model.huggingface_config
        )
        return cls(
            target=target_config,
            draft=draft_config,
        )

    def get_max_seq_len(self) -> int:
        return self.target.get_max_seq_len()
