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
"""Kimi K2.5's call: DeepseekV3's, with the batch's images merged in first."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from max.graph import TensorValue, ops
from max.nn.kernels import scatter_nd_skip_oob_indices
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.driver import SequentialBatch
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import DistributedVerify, Verified

from ..deepseekV3.spec_adapters import ep_input_types

if TYPE_CHECKING:
    from max.pipelines.speculative.spec_adapters import VerifyFn

__all__ = ["ep_input_types"]

PRIMARY_KV = ""


def verify(
    target: Module,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a Kimi K2.5 language model, on every device.

    Prefill merges no draft tokens into a row, so the images' merge
    positions hold in the merged sequence; decode brings no new images. A
    block draft reads each device's captured layers fused into one tensor.

    Raises:
        ValueError: If the target wasn't built to return the layers its
            draft names.
    """
    del target_config, draft_config
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        raise ValueError(
            "A Kimi K2.5 target verifies only for drafts that read its"
            f" selected layers. Got {hidden_states.value}."
        )
    return DistributedVerify(_verify, target)


def _verify(
    target: Any, batch: SequentialBatch | BlockBatch
) -> Verified[list[list[TensorValue]] | list[TensorValue]]:
    if isinstance(batch, SequentialBatch):
        host_offsets = batch.dist.host_merged_offsets
        splits = batch.dist.data_parallel_splits
    else:
        # A distributed signature always declares both.
        assert batch.host_merged_offsets is not None
        assert batch.data_parallel_splits is not None
        host_offsets = batch.host_merged_offsets
        splits = batch.data_parallel_splits
    h = [
        scatter_nd_skip_oob_indices(
            input=h_dev,
            updates=ops.cast(images, h_dev.dtype),
            indices=ops.unsqueeze(positions, -1),
        )
        for h_dev, images, positions in zip(
            target.embed_tokens(batch.merged_tokens, batch.signal_buffers),
            batch.vision_embeddings,
            batch.vision_scatter_indices,
            strict=True,
        )
    ]
    outputs = target._process_hidden_states(
        h,
        batch.signal_buffers,
        batch.kv_collections,
        batch.return_n_logits,
        list(batch.merged_offsets_per_dev),
        host_offsets,
        splits,
        batch.batch_context_lengths,
        batch.ep_inputs,
    )
    captures = captures_by_device(
        list(outputs.hidden_states.tensors()), batch.n_devs
    )
    if isinstance(batch, BlockBatch):
        return Verified(
            logits=outputs.logits["logits"],
            hidden=fuse_captured_hidden_states(captures),
        )
    return Verified(logits=outputs.logits["logits"], hidden=captures)


ADAPTER = TargetAdapter(
    primary_kv=PRIMARY_KV, ep_input_types=ep_input_types, verify=verify
)
