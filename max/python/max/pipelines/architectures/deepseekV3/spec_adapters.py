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
"""DeepseekV3's call, tensor- or data-parallel with expert-parallel MoE."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from max.graph import BufferType, TensorType, TensorValue
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates
from max.nn.transformer.transformer import captures_by_device
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.driver import SequentialBatch
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import DistributedVerify, Verified

if TYPE_CHECKING:
    from max.pipelines.speculative.spec_adapters import VerifyFn

PRIMARY_KV = ""


def ep_input_types(target: Any) -> Sequence[TensorType | BufferType]:
    """Returns the target's expert-parallel inputs, empty when EP is off."""
    if target.ep_manager is None:
        return ()
    return target.ep_manager.input_types()


def verify(
    target: Module,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a DeepseekV3 target, on every device.

    Raises:
        ValueError: If the target wasn't built to return its normalized
            hidden states or the layers an EAGLE3 draft names, which only
            the one-token-per-step drafts read.
    """
    del target_config, draft_config
    if hidden_states not in (
        ReturnHiddenStates.ALL_NORMALIZED,
        ReturnHiddenStates.SELECTED_LAYERS,
    ):
        raise ValueError(
            "A DeepseekV3 target verifies only for drafts that read its"
            " normalized hidden states or selected layers. Got"
            f" {hidden_states.value}."
        )
    return DistributedVerify(_verify, target, hidden_states)


def _verify(
    target: Module,
    hidden_states: ReturnHiddenStates,
    batch: SequentialBatch | BlockBatch,
) -> Verified[list[TensorValue] | list[list[TensorValue]]]:
    assert isinstance(batch, SequentialBatch)
    outputs = target(
        batch.merged_tokens,
        batch.signal_buffers,
        batch.kv_collections,
        batch.return_n_logits,
        batch.merged_offsets_per_dev,
        batch.dist.host_merged_offsets,
        batch.dist.data_parallel_splits,
        batch.batch_context_lengths,
        batch.ep_inputs,
    )
    if hidden_states == ReturnHiddenStates.SELECTED_LAYERS:
        return Verified(
            logits=outputs.logits["logits"],
            hidden=captures_by_device(
                list(outputs.hidden_states.tensors()), batch.n_devs
            ),
        )
    return Verified(
        logits=outputs.logits["logits"],
        hidden=list(outputs.hidden_states.tensors())[: batch.n_devs],
    )


ADAPTER = TargetAdapter(
    primary_kv=PRIMARY_KV,
    verify=verify,
    ep_input_types=ep_input_types,
)
