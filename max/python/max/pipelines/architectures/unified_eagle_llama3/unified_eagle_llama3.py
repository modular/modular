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
"""EAGLE draft model extending Llama3 with hidden state fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from max.graph import BufferType, BufferValue, TensorType, TensorValue, Value
from max.nn import PagedCacheValues
from max.nn.kv_cache import AttentionDispatchMetadata
from max.nn.layer import Module

from ..eagle_llama3.eagle_llama3 import EagleLlama3
from ..llama3.llama3 import Llama3
from .model_config import UnifiedEagleLlama3Config


@dataclass
class UnifiedEagleLlama3Values:
    tokens: TensorValue
    input_row_offsets: TensorValue
    return_n_logits: TensorValue
    kv_collection: PagedCacheValues
    draft_kv_blocks: BufferValue


class UnifiedEagleLlama3(Module):
    """EAGLE draft model that extends Llama3 with hidden state fusion."""

    def __init__(self, config: UnifiedEagleLlama3Config) -> None:
        super().__init__()

        self.config = config
        self.num_devices = 1

        # TODO: support distributed llama3 model
        if len(config.target.devices) != 1:
            raise ValueError("UnifiedEagleLlama3 only supports a single device")

        self.target = Llama3(config.target)
        self.draft = EagleLlama3(config.draft)

    def _unflatten_graph_inputs(
        self,
        inputs: Sequence[Value[Any]],
    ) -> UnifiedEagleLlama3Values:
        (
            tokens,
            input_row_offsets,
            return_n_logits,
            # target model kvcache inputs
            target_kv_blocks,
            cache_lengths,
            lookup_table,
            max_lengths,
            dispatch_metadata,
            # draft model kvcache blocks
            draft_kv_blocks,
        ) = inputs

        target_kv_collection = PagedCacheValues(
            kv_blocks=target_kv_blocks.buffer,
            cache_lengths=cache_lengths.tensor,
            lookup_table=lookup_table.tensor,
            max_lengths=max_lengths.tensor,
            dispatch_metadata=AttentionDispatchMetadata(
                dispatch_metadata.tensor
            ),
        )

        return UnifiedEagleLlama3Values(
            tokens=tokens.tensor,
            input_row_offsets=input_row_offsets.tensor,
            return_n_logits=return_n_logits.tensor,
            kv_collection=target_kv_collection,
            draft_kv_blocks=draft_kv_blocks.buffer,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        target_inputs = self.target.input_types(
            kv_params=self.config.target.kv_params, lora_manager=None
        )

        draft_kv_inputs = self.config.draft.kv_params.get_symbolic_inputs()
        assert len(draft_kv_inputs) == 1
        draft_kv_blocks = draft_kv_inputs[0].kv_blocks

        return target_inputs + (draft_kv_blocks,)

    def __call__(
        self,
        inputs: UnifiedEagleLlama3Values,
    ) -> tuple[TensorValue, ...]:
        # TODO: Fix this! We are just running the target and draft models sequentially.
        draft_kv_collections = PagedCacheValues(
            kv_blocks=inputs.draft_kv_blocks,
            cache_lengths=inputs.kv_collection.cache_lengths,
            lookup_table=inputs.kv_collection.lookup_table,
            max_lengths=inputs.kv_collection.max_lengths,
            kv_scales=inputs.kv_collection.kv_scales,
            dispatch_metadata=inputs.kv_collection.dispatch_metadata,
        )
        (*_target_logits, target_hs) = self.target(
            inputs.tokens,
            inputs.kv_collection,
            inputs.return_n_logits,
            inputs.input_row_offsets,
        )
        # TODO: shift the tokens for the draft model.
        (draft_logits, _draft_hs) = self.draft(
            inputs.tokens,
            draft_kv_collections,
            inputs.return_n_logits,
            inputs.input_row_offsets,
            target_hs,
        )
        # TODO: we should not just be returning the draft logits
        return (draft_logits,)
