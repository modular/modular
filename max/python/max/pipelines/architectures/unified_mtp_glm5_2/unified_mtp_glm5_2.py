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
"""GLM-5.2 (DeepSeek-V3.2 sparse) with MTP nn.Module.

The loop lives in :class:`SequentialDriver`; this class only names the target
and draft and hands over the adapter pair from :mod:`.spec_adapters`.
"""

from __future__ import annotations

from max.graph import TensorValue
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.driver import SequentialDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

from ..deepseekV3_2.deepseekV3_2 import DeepseekV3_2
from ..deepseekV3_2.model_config import DeepseekV3_2Config
from ..deepseekV3_2_nextn.deepseekV3_2_nextn import DeepseekV3_2NextN
from ..deepseekV3_2_nextn.model_config import DeepseekV3_2NextNConfig
from .spec_adapters import Glm5_2MTPProposer, Glm5_2Target


class UnifiedMTPGlm5_2(SequentialDriver[list[TensorValue]]):
    """Fused nn.Module: merge + V3.2 target + rejection sampling + sparse draft."""

    target: DeepseekV3_2
    draft: DeepseekV3_2NextN

    def __init__(
        self,
        config: DeepseekV3_2Config,
        draft_config: DeepseekV3_2NextNConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
    ) -> None:
        assert draft_config is not None
        self.sampled_draft_proposal = (
            speculative_config is not None
            and speculative_config.draft_proposal == "sampled"
        )
        target = DeepseekV3_2(config)
        target.emit_last_token_logits = False
        draft = DeepseekV3_2NextN(draft_config)
        super().__init__(
            Glm5_2Target(target),
            Glm5_2MTPProposer(draft),
            target_model=target,
            draft_model=draft,
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.devices,
                distributed=True,
                data_parallel_degree=config.data_parallel_degree,
            ),
            speculative_config=speculative_config,
            enable_structured_output=enable_structured_output,
            draft_proposal=(
                "sampled" if self.sampled_draft_proposal else "argmax"
            ),
            vocab_size=(
                config.vocab_size if self.sampled_draft_proposal else None
            ),
        )
        self.config = config
