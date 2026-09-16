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
"""DFlash + Gemma4-31B: the block draft wired to the block driver."""

from __future__ import annotations

from max.graph import TensorValue
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from typing_extensions import override

from ..dflash_llama3 import DFlashLlama3
from ..gemma4.block_spec_adapters import SLIDING_KV, Gemma4BlockTarget
from ..gemma4.gemma4 import Gemma4TextModel
from .model_config import UnifiedDflashGemma4_31BConfig
from .spec_adapters import DFlashGemma4_31BProposer

__all__ = ["SLIDING_KV", "UnifiedDflashGemma4_31B"]


class UnifiedDflashGemma4_31B(BlockDriver[TensorValue]):
    """Flash + Gemma4-31B: the block draft wired to the block driver

    The block reuses the target's embedding and ``lm_head``; the phases
    live in :class:`BlockDriver`.
    """

    target: Gemma4TextModel
    draft: DFlashLlama3

    def __init__(
        self,
        config: UnifiedDflashGemma4_31BConfig,
        enable_structured_output: bool = False,
    ) -> None:
        # AcceptanceSampler dispatches to the synthetic path first, which
        # ignores token bitmasks -- grammar constraints would silently stop
        # being enforced while the serve layer believes they are.
        if (
            enable_structured_output
            and config.speculative_config.synthetic_acceptance_rate is not None
        ):
            raise ValueError(
                "synthetic_acceptance_rate is incompatible with structured"
                " output: the synthetic acceptance path ignores token"
                " bitmasks. This arch enables the bitmask path by default"
                " for tool-call grammars; for synthetic-acceptance"
                " benchmarking pass --tool-parser none and leave"
                " --enable-structured-output off."
            )
        target = Gemma4TextModel(config.target)
        draft = DFlashLlama3(
            config.draft,
            num_context_features=len(config.target_layer_ids),
            layer_types=config.layer_types or None,
        )
        block_size = config.effective_block_size
        super().__init__(
            Gemma4BlockTarget(
                target,
                hidden_size=config.target.text_config.hidden_size,
                dtype=config.target.unquantized_dtype,
            ),
            DFlashGemma4_31BProposer(
                draft,
                target,
                block_size=block_size,
                mask_token_id=int(config.mask_token_id),
                hidden_size=config.draft.hidden_size,
            ),
            target_model=target,
            draft_model=draft,
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.target.devices,
                distributed=False,
                # Gemma4's embedding and lm_head are collective even at one
                # device.
                include_signal_buffers=True,
                include_in_thinking_phase=True,
                enable_structured_output=enable_structured_output,
            ),
            speculative_config=config.speculative_config,
            enable_structured_output=enable_structured_output,
            relaxed_acceptance=True,
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
