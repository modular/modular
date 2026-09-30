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
"""MiMo-V2.6-Flash + DFlash: the fused speculative graph on the block driver."""

from __future__ import annotations

from dataclasses import dataclass

from max.graph import TensorValue
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import SpeculativeConfig
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from typing_extensions import override

from ..dflash_mimo_v2 import DFlashMiMoV2, DFlashMiMoV2Config
from ..mimo_v2.mimo_v2 import MiMoV2
from ..mimo_v2.model_config import MiMoV2Config
from .model_config import DRAFT
from .spec_adapters import DFlashMiMoV2Proposer, MiMoV2BlockTarget

__all__ = ["UnifiedDflashMiMoV2", "UnifiedDflashMiMoV2Spec"]


@dataclass(kw_only=True)
class UnifiedDflashMiMoV2Spec:
    """What the fused graph is built from."""

    target: MiMoV2Config
    draft: DFlashMiMoV2Config
    speculative_config: SpeculativeConfig
    num_speculative_tokens: int
    """How many of the block's proposals a step verifies."""
    sampleable_vocab_size: int
    """The tokenizer's size. ``lm_head`` rows from here up are padding with
    live weights, so the graph masks them out of every verify."""


class UnifiedDflashMiMoV2(
    BlockDriver[list[list[TensorValue]], list[TensorValue]]
):
    """Verify ``[anchor, d1..dK]`` on the target, write the drafter's context
    for the verified rows, then draft the next block.

    The block is the drafter's trained 8 wide whatever K is, so the next
    step's block always overwrites every rejected row's context.
    """

    target: MiMoV2
    draft: DFlashMiMoV2

    def __init__(
        self,
        spec: UnifiedDflashMiMoV2Spec,
        mask_embedding: WeightData,
        enable_structured_output: bool = False,
    ) -> None:
        """Builds the fused graph's module.

        Args:
            spec: The target, drafter and verify width.
            mask_embedding: The drafter's trained mask slot embedding, baked
                into the graph as a constant.
            enable_structured_output: Whether acceptance takes the grammar
                bitmask.
        """
        target_config = spec.target
        target_config.return_logits = ReturnLogits.VARIABLE
        target_config.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
        target_config.target_layer_ids = list(spec.draft.target_layer_ids)
        if spec.draft.target_layer_ids != sorted(spec.draft.target_layer_ids):
            raise ValueError(
                "MiMo-V2 DFlash: the target captures its taps in layer order;"
                f" got target_layer_ids {spec.draft.target_layer_ids}."
            )
        if not 0 < spec.sampleable_vocab_size <= target_config.vocab_size:
            raise ValueError(
                f"MiMo-V2 DFlash: sampleable_vocab_size"
                f" {spec.sampleable_vocab_size} is outside the"
                f" {target_config.vocab_size}-row lm_head."
            )
        target = MiMoV2(target_config)
        drafter = DFlashMiMoV2(spec.draft, mask_embedding=mask_embedding)
        super().__init__(
            MiMoV2BlockTarget(
                target,
                vocab_size=target_config.vocab_size,
                sampleable_vocab_size=spec.sampleable_vocab_size,
            ),
            DFlashMiMoV2Proposer(
                drafter,
                target,
                num_speculative_tokens=spec.num_speculative_tokens,
                sampleable_vocab_size=spec.sampleable_vocab_size,
            ),
            target_model=target,
            draft_model=drafter,
            input_spec=SpecDecodeInputTypeSpec(
                devices=target_config.devices,
                distributed=False,
                include_signal_buffers=True,
                enable_structured_output=enable_structured_output,
            ),
            speculative_config=spec.speculative_config,
            enable_structured_output=enable_structured_output,
            # The drafter's own lengths place its context, as they do in an
            # engine that sizes each group's claim separately.
            ctx_at_draft_cache_length=True,
            num_speculative_tokens=spec.num_speculative_tokens,
            use_greedy_acceptance=spec.speculative_config.use_greedy_acceptance,
        )
        self.spec = spec

    @override
    @property
    def signature_kv_params(self) -> KVCacheParamInterface:
        return MultiKVCacheParams.from_params(
            {
                "target": self.spec.target.kv_params,
                DRAFT: self.spec.draft.kv_params,
            }
        )
