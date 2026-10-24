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
"""Llama 3's call, which Granite's targets take too."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from max.graph import TensorValue
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.driver import SequentialBatch
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import Verified

if TYPE_CHECKING:
    from max.pipelines.speculative.spec_adapters import VerifyFn


def verify(
    target: Module,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a single-device target called as Llama 3 is."""
    del target_config, draft_config
    return partial(_verify, target, hidden_states)


def _verify(
    target: Module,
    hidden_states: ReturnHiddenStates,
    batch: SequentialBatch | BlockBatch,
) -> Verified[TensorValue | list[TensorValue]]:
    outputs = target(
        batch.merged_tokens,
        batch.kv_collections[0],
        batch.return_n_logits,
        batch.merged_offsets,
    )
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        return Verified(
            logits=outputs.logits["logits"],
            hidden=list(outputs.hidden_states.tensors())[0],
        )
    captures = fuse_captured_hidden_states(
        captures_by_device(list(outputs.hidden_states.tensors()), 1)
    )
    # A block draft reads its captures per device, an EAGLE3 head one tensor.
    if isinstance(batch, BlockBatch):
        return Verified(logits=outputs.logits["logits"], hidden=captures)
    return Verified(logits=outputs.logits["logits"], hidden=captures[0])


ADAPTER = TargetAdapter(verify=verify)
