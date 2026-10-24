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
"""Gemma 4's language-model call.

A block draft verifies over the full-attention cache on one device, text
only. The MTP assistant cross-attends into the target's sliding-window cache
on every device the target spans, and its pairing takes images.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.driver import SequentialBatch
from max.pipelines.speculative.spec_adapters.target_adapter import (
    STAGE_BOUNDARY,
    TargetAdapter,
)
from max.pipelines.speculative.spec_target import Verified

if TYPE_CHECKING:
    from max.pipelines.speculative.spec_adapters import VerifyFn

_SLIDING_KV = "sliding_attention"

_FULL_KV = "full_attention"

PRIMARY_KV = _FULL_KV


def verify(
    target: Module,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a Gemma 4 language model.

    Raises:
        ValueError: If the target wasn't built to return the selected
            layers a block draft reads, or the last normalized layer the
            MTP assistant reads.
    """
    del draft_config
    if hidden_states not in (
        ReturnHiddenStates.SELECTED_LAYERS,
        ReturnHiddenStates.ALL_NORMALIZED,
    ):
        raise ValueError(
            "A Gemma 4 target verifies only for drafts that read its selected"
            f" layers or its last normalized one. Got {hidden_states.value}."
        )
    return partial(
        _verify,
        target,
        target_config.text_config.hidden_size,
        target_config.unquantized_dtype,
    )


def _verify(
    target: Module,
    hidden_size: int,
    dtype: DType,
    batch: SequentialBatch | BlockBatch,
) -> Verified[list[TensorValue]]:
    # Gemma 4's embedding and head use collectives unconditionally, so the
    # entry point is the distributed one: every argument arrives as a
    # per-device list, beside the vision inputs a text-only forward still
    # declares.
    if isinstance(batch, SequentialBatch):
        # Images arrive only in prefill, which merges no drafts into a row.
        # A later pipeline stage starts from the previous stage's hidden
        # states over the merged window instead of embedding its tokens.
        outputs = target(
            batch.merged_tokens,
            batch.signal_buffers,
            batch.kv_collections,
            batch.passthrough_kv[_FULL_KV],
            batch.return_n_logits,
            batch.merged_offsets_per_dev,
            batch.vision_embeddings,
            batch.vision_scatter_indices,
            boundary_inputs=batch.extra.get(STAGE_BOUNDARY),
        )
        return Verified(
            logits=outputs.logits["logits"],
            hidden=list(outputs.hidden_states.tensors())[: batch.n_devs],
        )
    device = batch.device0
    # The vision merge scatters nothing on an empty index set, but the
    # signature still takes both operands.
    empty_embeds = ops.constant(
        np.zeros((0, hidden_size), dtype=np.float32),
        DType.float32,
        device=device,
    ).cast(dtype)
    empty_indices = ops.range(
        0, 0, 1, out_dim=0, dtype=DType.int32, device=device
    )
    outputs = target(
        batch.merged_tokens,
        batch.signal_buffers,
        batch.passthrough_kv[_SLIDING_KV],
        batch.kv_collections,
        batch.return_n_logits,
        batch.merged_offsets_per_dev,
        [empty_embeds],
        [empty_indices],
    )
    hidden = fuse_captured_hidden_states(
        captures_by_device(list(outputs.hidden_states.tensors()), 1)
    )
    return Verified(logits=outputs.logits["logits"], hidden=hidden)


_BASE = TargetAdapter(primary_kv=PRIMARY_KV, verify=verify)
# The assistant reads the final hidden states and the last sliding and full
# layers' caches, all of which the last pipeline stage holds.
ADAPTER = replace(
    _BASE,
    by_method={
        "mtp": replace(
            _BASE, primary_kv=_SLIDING_KV, speculates_on_last_stage=True
        )
    },
)
