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
"""DFlash + Llama3: the block draft wired to the block spec-decode driver."""

from __future__ import annotations

from max.graph import TensorValue
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.pipelines.speculative.block_driver import BlockDriver
from max.pipelines.speculative.spec_input_types import SpecDecodeInputTypeSpec
from typing_extensions import override

from ..dflash_llama3 import DFlashLlama3
from ..llama3.llama3 import Llama3
from .model_config import UnifiedDflashLlama3Config
from .spec_adapters import DFlashLlama3Proposer, DFlashLlama3Target


class UnifiedDflashLlama3(BlockDriver[TensorValue, TensorValue]):
    """DFlash + Llama3: merge, verify, materialize the context KV, block.

    Single device throughout -- Llama3 uses no collectives, so the graph
    declares neither signal buffers nor data-parallel splits. The phases
    themselves live in :class:`BlockDriver`.
    """

    target: Llama3
    draft: DFlashLlama3

    def __init__(self, config: UnifiedDflashLlama3Config) -> None:
        target = Llama3(config.target)
        draft = DFlashLlama3(
            config.draft,
            num_context_features=len(config.target_layer_ids),
        )
        block_size = config.resolve_block_size()
        super().__init__(
            DFlashLlama3Target(target),
            DFlashLlama3Proposer(
                draft,
                target,
                block_size=block_size,
                mask_token_id=int(config.mask_token_id),
                hidden_size=config.draft.hidden_size,
            ),
            target_model=target,
            draft_model=draft,
            input_spec=SpecDecodeInputTypeSpec(
                devices=config.target.devices, distributed=False
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
                "draft": self.config.draft.kv_params,
            }
        )
