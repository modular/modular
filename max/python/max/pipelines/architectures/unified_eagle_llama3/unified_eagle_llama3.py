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
"""Unified EAGLE Llama3: the shared sequential driver plus this pair's adapters."""

from __future__ import annotations

from max.graph import TensorValue
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.pipelines.speculative.driver import SequentialDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from typing_extensions import override

from ..eagle_llama3.eagle_llama3 import EagleLlama3
from ..llama3.llama3 import Llama3
from .model_config import UnifiedEagleLlama3Config
from .spec_adapters import EagleLlama3Proposer, Llama3Target

__all__ = ["UnifiedEagleLlama3"]


class UnifiedEagleLlama3(SequentialDriver[TensorValue]):
    """Fused Llama3 target + EAGLE draft on a single device."""

    target: Llama3
    draft: EagleLlama3

    def __init__(self, config: UnifiedEagleLlama3Config) -> None:
        # TODO: support distributed llama3 model
        if len(config.target.devices) != 1:
            raise ValueError("UnifiedEagleLlama3 only supports a single device")

        target = Llama3(config.target)
        draft = EagleLlama3(config.draft)
        super().__init__(
            Llama3Target(target),
            EagleLlama3Proposer(draft, config.draft.hidden_size),
            target_model=target,
            draft_model=draft,
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.target.devices,
                distributed=False,
            ),
            speculative_config=config.speculative_config,
            enable_structured_output=config.enable_structured_output,
        )
        self.config = config

    @override
    @property
    def signature_kv_params(self) -> KVCacheParamInterface:
        return MultiKVCacheParams.from_params(
            {
                "target": self.config.target.kv_params,
                "draft": self.config.draft.kv_params,
            }
        )
