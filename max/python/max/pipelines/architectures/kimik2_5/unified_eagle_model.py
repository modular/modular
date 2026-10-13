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
"""Eagle3 + Kimi K2.5: the MLA draft wired to the spec-decode driver."""

from __future__ import annotations

from max.graph import TensorValue
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..deepseekV3.model_config import DeepseekV3Config
from .eagle3_kimi_k25 import Eagle3KimiK25
from .spec_adapters import Eagle3KimiK25Proposer, KimiK25Target


class Eagle3KimiK25Unified(SequentialDriver[list[list[TensorValue]]]):
    """Eagle3 + Kimi K2.5: the MLA draft wired to the spec-decode driver.

    The target returns hidden states captured at three intermediate layers;
    the draft fuses them via ``fc`` and proposes the next token. The loop
    itself lives in :class:`SequentialDriver`.
    """

    target: DeepseekV3
    draft: Eagle3KimiK25

    def __init__(
        self,
        config: DeepseekV3Config,
        draft_config: DeepseekV3Config | None = None,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
        enable_vision: bool = False,
    ) -> None:
        assert draft_config is not None
        target = DeepseekV3(config)
        draft = Eagle3KimiK25(draft_config)
        super().__init__(
            KimiK25Target(target, enable_vision=enable_vision),
            Eagle3KimiK25Proposer(draft),
            target_model=target,
            draft_model=draft,
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.devices,
                distributed=True,
                data_parallel_degree=config.data_parallel_degree,
                # Only a Kimi-style target with a vision encoder declares
                # these.
                enable_vision=enable_vision,
                vision_hidden_size=config.hidden_size,
                enable_structured_output=enable_structured_output,
            ),
            speculative_config=speculative_config,
            enable_structured_output=enable_structured_output,
        )
        self.config = config
        self.enable_vision = enable_vision
