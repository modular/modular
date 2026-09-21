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
"""Qwen3.5 fused with a DFlash2 block drafter, wired to the block driver.

The phases live in :class:`BlockDriver`; what is this architecture's own is
the pair of adapters in :mod:`.spec_adapters` and the state-pool tail
:meth:`UnifiedDflash2Qwen3_5.input_types` appends past the canonical
signature.

Acceptance is stochastic, never the greedy fast path: only the stochastic
sampler applies the grammar bitmask, and this checkpoint's ``lm_head`` carries
243 live padding rows past ``sampleable_vocab_size`` that nothing else
excludes.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferType, TensorType, TensorValue
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from typing_extensions import override

from ..dflash2_qwen3_5 import DFlash2Qwen3_5
from ..qwen3_5.qwen3_5 import Qwen3_5
from ..qwen3_5.state_cache import linear_state_regions
from .model_config import UnifiedDflash2Qwen3_5Config
from .spec_adapters import DFlash2Qwen3_5Proposer, Qwen3_5BlockTarget

__all__ = ["UnifiedDflash2Qwen3_5"]


class UnifiedDflash2Qwen3_5(BlockDriver[TensorValue, TensorValue]):
    """Merge, target verify, state rollback and block draft, on the driver."""

    target: Qwen3_5
    draft: DFlash2Qwen3_5

    def __init__(
        self,
        config: UnifiedDflash2Qwen3_5Config,
        enable_structured_output: bool = False,
    ) -> None:
        speculative_config = config.speculative_config
        if speculative_config.use_greedy_acceptance:
            raise ValueError(
                "DFlash2 on Qwen3.5 requires stochastic acceptance: the greedy"
                " path ignores token_bitmasks, and this checkpoint's lm_head"
                " padding rows are only excluded through the bitmask."
            )
        if speculative_config.synthetic_acceptance_rate is not None:
            raise ValueError(
                "synthetic acceptance would bypass the state rollback's"
                " accepted-length plan"
            )

        target = Qwen3_5(config.target)
        target.return_logits = ReturnLogits.VARIABLE
        target.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
        draft = DFlash2Qwen3_5(
            config.draft,
            num_context_features=len(config.target_layer_ids),
            block_size=config.block_size,
            conv_kernel_size=config.conv_kernel_size,
            conv_group_size=config.conv_group_size,
            selector_rank=config.selector_rank,
            selector_top_k=config.selector_top_k,
            layer_types=config.layer_types or None,
        )

        # The same geometry the cache declares, so a shadow row is shaped
        # like the live row it holds a copy of.
        num_linear_layers = len(target.linear_layer_indices)
        state_regions = linear_state_regions(
            num_linear_layers=num_linear_layers,
            key_head_dim=config.target.linear_key_head_dim,
            num_key_heads=config.target.linear_num_key_heads,
            value_head_dim=config.target.linear_value_head_dim,
            num_value_heads=config.target.linear_num_value_heads,
            conv_kernel_dim=config.target.linear_conv_kernel_dim,
            dtype=config.target.state_dtype,
            num_devices=len(config.target.devices),
        )

        target_adapter = Qwen3_5BlockTarget(target, state_regions)
        super().__init__(
            target_adapter,
            DFlash2Qwen3_5Proposer(
                draft,
                target_adapter,
                block_size=config.block_size,
                mask_token_id=config.mask_token_id,
                hidden_size=config.draft.hidden_size,
            ),
            target_model=target,
            draft_model=draft,
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.target.devices,
                distributed=True,
                data_parallel_degree=1,
                include_in_thinking_phase=True,
            ),
            speculative_config=speculative_config,
            enable_structured_output=enable_structured_output,
            relaxed_acceptance=True,
        )
        self.config = config
        self.target_layer_ids = list(config.target_layer_ids)
        self.mask_token_id = config.mask_token_id
        self.num_linear_layers = num_linear_layers
        self.state_regions = state_regions

    @override
    @property
    def has_trailing_inputs(self) -> bool:
        # The state-pool tail below.
        return True

    @override
    def input_types(
        self, kv_params: KVCacheParamInterface | None = None
    ) -> tuple[TensorType | BufferType, ...]:
        """Canonical spec-decode signature plus the Qwen state-pool tail.

        Byte-for-byte the Qwen3.5 MTP graph's signature: the tail is the live
        pools, then the rows addressing them, then the shadow pools, every
        block device-major. Only the draft KV leaf's shapes differ (five
        drafter layers of 8 x 128 rather than one target-shaped layer), so
        Mach's Qwen slot layout carries over unchanged.

        The shadow takes no rows; ``state_rollback.shadow_row_ids`` builds
        them in-graph.
        """
        devices = self.config.target.devices
        spec_types = super().input_types(kv_params)

        tail: list[TensorType | BufferType] = []
        for region in self.state_regions:
            tail.extend(
                BufferType(
                    region.dtype,
                    shape=[region.rows_dim, *region.row_shape],
                    device=device,
                )
                for device in devices
            )
        for region in self.state_regions:
            tail.extend(
                TensorType(
                    DType.uint32,
                    shape=[region.num_layers, "batch_size"],
                    device=device,
                )
                for device in devices
            )
        for region in self.state_regions:
            tail.extend(
                BufferType(
                    region.dtype,
                    shape=[f"shadow_{region.rows_dim}", *region.row_shape],
                    device=device,
                )
                for device in devices
            )

        return (*spec_types, *tail)
