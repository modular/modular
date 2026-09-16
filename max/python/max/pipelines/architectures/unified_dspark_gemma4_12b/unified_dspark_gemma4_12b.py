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
"""DSpark + Gemma4-12B: the dense block draft on the block driver."""

from __future__ import annotations

from max.graph import TensorValue
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from typing_extensions import override

from ..gemma4.block_spec_adapters import Gemma4BlockTarget
from ..gemma4.gemma4 import Gemma4TextModel
from .dspark_gemma4 import DSparkGemma4
from .model_config import UnifiedDSparkGemma4_12BConfig
from .spec_adapters import DSparkGemma4_12BProposer

__all__ = ["UnifiedDSparkGemma4_12B"]


class UnifiedDSparkGemma4_12B(BlockDriver[TensorValue]):
    """Spark + Gemma4-12B: the dense block draft on the block driver

    DSpark drafts at every block position: the anchor predicts draft 1,
    unlike DFlash, which drops slot 0. The phases live in
    :class:`BlockDriver`.
    """

    target: Gemma4TextModel
    draft: DSparkGemma4

    def __init__(
        self,
        config: UnifiedDSparkGemma4_12BConfig,
    ) -> None:
        target = Gemma4TextModel(config.target)
        draft = DSparkGemma4(
            config.draft,
            kv_params=config.draft_kv_params,
            devices=list(config.target.devices),
            dtype=config.target.unquantized_dtype,
        )
        block_size = config.resolve_block_size()
        super().__init__(
            Gemma4BlockTarget(
                target,
                hidden_size=config.target.text_config.hidden_size,
                dtype=config.target.unquantized_dtype,
            ),
            DSparkGemma4_12BProposer(
                draft,
                target,
                block_size=block_size,
                mask_token_id=int(config.mask_token_id),
                hidden_size=config.draft.hidden_size,
                final_logit_softcapping=config.draft.final_logit_softcapping,
            ),
            target_model=target,
            draft_model=draft,
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.target.devices,
                distributed=False,
                # Gemma4's embedding and lm_head are collective even at one
                # device.
                include_signal_buffers=True,
            ),
            speculative_config=config.speculative_config,
        )
        self.config = config
        self.target_layer_ids = list(config.target_layer_ids)
        self.mask_token_id = int(config.mask_token_id)

    @override
    @property
    def signature_kv_params(self) -> KVCacheParamInterface:
        return MultiKVCacheParams.from_params(
            {
                "target": self.config.target.kv_params,
                "draft": self.config.draft_kv_params,
            }
        )
