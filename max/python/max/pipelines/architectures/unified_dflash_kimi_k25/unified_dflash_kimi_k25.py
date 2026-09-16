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
"""Unified DFlash Kimi K2.5: MLA target + DFlash MHA/GQA draft on the driver."""

from __future__ import annotations

from max.graph import TensorValue
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..dflash_kimi_k25 import DFlashKimiK25
from .model_config import UnifiedDflashKimiK25Config
from .spec_adapters import DFlashKimiK25Proposer, DFlashKimiK25Target

__all__ = ["UnifiedDflashKimiK25", "dflash_kimi_k25_input_spec"]


def dflash_kimi_k25_input_spec(
    config: UnifiedDflashKimiK25Config,
    *,
    enable_structured_output: bool = False,
) -> SpecDecodeInputTypeSpec:
    """The graph signature's shape: distributed, data-parallel, with vision.

    A function so the signature can be built from a config alone.
    """
    return SpecDecodeInputTypeSpec(
        devices=config.target.devices,
        distributed=True,
        data_parallel_degree=config.target.data_parallel_degree,
        enable_vision=True,
        vision_hidden_size=config.target.hidden_size,
        enable_structured_output=enable_structured_output,
    )


class UnifiedDflashKimiK25(BlockDriver[list[TensorValue], list[TensorValue]]):
    """Merge -> target (MLA) -> reject -> materialize -> draft block.

    The only block architecture that runs sharded and data-parallel, so the
    per-replica slicing the other four never needed lives partly in the
    driver's block phases and partly in :mod:`.spec_adapters`.
    """

    target: DeepseekV3
    draft: DFlashKimiK25

    def __init__(
        self,
        config: UnifiedDflashKimiK25Config,
        enable_structured_output: bool = False,
    ) -> None:
        block_size = config.resolve_block_size()
        target = DeepseekV3(config.target)
        draft = DFlashKimiK25(config.draft)
        super().__init__(
            DFlashKimiK25Target(target),
            DFlashKimiK25Proposer(
                draft,
                target,
                block_size=block_size,
                mask_token_id=int(config.mask_token_id),
                hidden_size=config.draft.hidden_size,
            ),
            target_model=target,
            draft_model=draft,
            input_spec=dflash_kimi_k25_input_spec(
                config, enable_structured_output=enable_structured_output
            ),
            speculative_config=config.speculative_config,
            enable_structured_output=enable_structured_output,
            ctx_at_draft_cache_length=True,
        )
        self.config = config
        self.target_layer_ids = list(config.target_layer_ids)
        self.mask_token_id = int(config.mask_token_id)
