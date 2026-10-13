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
"""Gemma4 with MTP: the assistant draft wired to the spec-decode driver."""

from __future__ import annotations

from max.graph import TensorValue
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

from ..gemma4.gemma4 import Gemma4TextModel
from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..gemma4_assistant.gemma4_assistant import Gemma4Assistant
from ..gemma4_assistant.model_config import Gemma4AssistantConfig
from .spec_adapters import Gemma4MTPProposer, Gemma4Target


def gemma4_mtp_input_spec(
    config: Gemma4ForConditionalGenerationConfig,
    *,
    enable_structured_output: bool,
) -> SpecDecodeInputTypeSpec:
    """The signature this arch declares: distributed, with vision inputs.

    The per-row ``in_thinking_phase`` flag is not named here -- the driver adds
    it from :attr:`Gemma4MTPProposer.uses_thinking_phase`. Split out of the
    constructor so a test can assert the declaration without building the
    target and draft modules.
    """
    return SpecDecodeInputTypeSpec(
        devices=config.devices,
        distributed=True,
        data_parallel_degree=1,
        enable_vision=True,
        vision_hidden_size=config.text_config.hidden_size,
        enable_structured_output=enable_structured_output,
    )


class UnifiedMTPGemma4(SequentialDriver[list[TensorValue]]):
    """Gemma4 with MTP: the assistant draft wired to the spec-decode driver.

    The assistant cross-attends into the target's paired sliding/full caches
    rather than keeping one of its own; the loop itself lives in
    :class:`SequentialDriver`.
    """

    target: Gemma4TextModel
    draft: Gemma4Assistant

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        draft_config: Gemma4AssistantConfig,
        draft: Gemma4Assistant,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
        use_greedy_acceptance: bool = False,
    ) -> None:
        target = Gemma4TextModel(config)
        super().__init__(
            Gemma4Target(target),
            Gemma4MTPProposer(draft, draft_config.backbone_hidden_size),
            target_model=target,
            draft_model=draft,
            input_spec=gemma4_mtp_input_spec(
                config, enable_structured_output=enable_structured_output
            ),
            speculative_config=speculative_config,
            enable_structured_output=enable_structured_output,
            use_greedy_acceptance=use_greedy_acceptance,
        )
        self.config = config
        self._draft_config = draft_config
        self.use_greedy_acceptance = use_greedy_acceptance
