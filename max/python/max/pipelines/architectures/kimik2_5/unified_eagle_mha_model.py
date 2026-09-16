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
"""Eagle3 + Kimi K2.5: the MHA draft wired to the spec-decode driver."""

from __future__ import annotations

from max.graph import TensorValue
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..deepseekV3.model_config import DeepseekV3Config
from ..eagle_common.eagle_mha_draft import (
    Eagle3MHADraft,
    Eagle3MHADraftConfig,
)
from .spec_adapters import Eagle3MHAKimiK25Proposer, KimiK25Target


class Eagle3MHAKimiK25Unified(SequentialDriver[list[list[TensorValue]]]):
    """Eagle3 + Kimi K2.5 with an MHA draft, wired to the spec-decode driver.

    Differs from the MLA-draft variant only in the draft it builds and the
    three declarations on :class:`Eagle3MHAKimiK25Proposer`; the loop lives in
    :class:`SequentialDriver`.
    """

    target: DeepseekV3
    draft: Eagle3MHADraft

    def __init__(
        self,
        config: DeepseekV3Config,
        draft_config: Eagle3MHADraftConfig,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
        enable_vision: bool = False,
    ) -> None:
        target = DeepseekV3(config)
        draft = Eagle3MHADraft(draft_config)
        super().__init__(
            KimiK25Target(target, enable_vision=enable_vision),
            Eagle3MHAKimiK25Proposer(draft),
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
        self.draft_config = draft_config
        self.enable_vision = enable_vision

        aux_layer_ids = config.eagle_aux_hidden_state_layer_ids
        assert aux_layer_ids is not None
        assert len(aux_layer_ids) == draft_config.fc_input_multiplier, (
            f"the target captures {len(aux_layer_ids)} aux hidden states "
            f"but the draft's fc fuses {draft_config.fc_input_multiplier}"
        )
