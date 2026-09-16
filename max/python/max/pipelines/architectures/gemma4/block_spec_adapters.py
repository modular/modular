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
"""Block spec-decode adapters shared by Gemma4's DFlash and DSpark drafts.

Three unified archs -- DFlash-31B, DSpark-31B and DSpark-12B -- put a block
draft behind the same Gemma4 target. What they share is everything on the
target side: the paired ``{sliding, full}`` attention caches, the zero-row
vision inputs a text-only forward still has to declare, and the dispatch
buffer the block query width needs rebuilt. They differ only in which module
owns the block's embedding and head, which is why :class:`Gemma4BlockTarget`
lives here while each arch keeps its own proposer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType, TensorValue, ops
from max.nn.kv_cache import PagedCacheValues
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.spec_target import Verified

from .gemma4 import Gemma4TextModel

__all__ = [
    "SLIDING_KV",
    "Gemma4BlockTarget",
    "block_dispatch_metadata",
    "block_kv_with_dispatch",
]

SLIDING_KV = "sliding_attention"
"""Name the sliding leaf rides under in :attr:`BlockBatch.passthrough_kv`. """


def block_dispatch_metadata(meta: TensorValue | None, k: int) -> TensorValue:
    """Rebuilds the MHA dispatch metadata at the draft block's query width.

    The 4-int CPU buffer is ``[batch_size, q_max_seq_len, num_partitions,
    max_cache_valid_length]``. ``q_max_seq_len`` becomes the block width ``k``
    and ``num_partitions`` is zeroed so the decode kernel recomputes the
    split-K count for the draft's own head geometry instead of reusing the
    target's.

    Args:
        meta: The leaf's verify-width dispatch metadata buffer.
        k: The draft block width (anchor slot plus drafted tokens).

    Returns:
        The rebuilt dispatch metadata buffer.
    """
    assert meta is not None
    cpu = DeviceRef.CPU()
    return ops.concat(
        [
            meta[0:1],
            ops.constant(k, DType.int64, device=cpu).reshape((1,)),
            ops.constant(0, DType.int64, device=cpu).reshape((1,)),
            meta[3:4],
        ],
        axis=0,
    )


class Gemma4BlockTarget:
    """The Gemma4 target behind a block draft.

    Single device, but the entry point is the distributed one: Gemma4's
    embedding and ``lm_head`` use collectives unconditionally, so the graph
    declares signal buffers even at one device and every argument arrives as a
    one-element list.
    """

    def __init__(
        self,
        target: Gemma4TextModel,
        *,
        hidden_size: int,
        dtype: DType,
    ) -> None:
        self.target = target
        self.hidden_size = hidden_size
        self.dtype = dtype

    def _empty_vision_inputs(
        self, device: DeviceRef
    ) -> tuple[TensorValue, TensorValue]:
        """Zero-row image embeddings + scatter indices for a text-only forward.

        The vision merge scatter is a no-op on an empty index set, but the
        target's signature still requires both operands.
        """
        empty_embeds = ops.constant(
            np.zeros((0, self.hidden_size), dtype=np.float32),
            DType.float32,
            device=device,
        ).cast(self.dtype)
        empty_indices = ops.range(
            0, 0, 1, out_dim=0, dtype=DType.int32, device=device
        )
        return empty_embeds, empty_indices

    def verify(self, batch: BlockBatch) -> Verified[TensorValue]:
        empty_embeds, empty_indices = self._empty_vision_inputs(batch.device0)
        outputs = self.target(
            batch.merged_tokens,
            batch.signal_buffers,
            batch.passthrough_kv[SLIDING_KV],
            batch.kv_collections,
            batch.return_n_logits,
            batch.merged_offsets_per_dev,
            [empty_embeds],
            [empty_indices],
        )
        # Single device, so the capture layers fuse to one tensor.
        hidden = fuse_captured_hidden_states(
            captures_by_device(outputs[3:], 1)
        )[0]
        return Verified(logits=outputs[1], hidden=hidden)

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


def block_kv_with_dispatch(
    block_kv: list[PagedCacheValues], k: int
) -> list[PagedCacheValues]:
    """The block caches with the dispatch buffer rebuilt at width ``k``."""
    return [
        replace(
            kv,
            attention_dispatch_metadata=block_dispatch_metadata(
                kv.attention_dispatch_metadata, k
            ),
            max_prompt_length=ops.constant(
                k, DType.uint32, device=DeviceRef.CPU()
            ).broadcast_to([1]),
        )
        for kv in block_kv
    ]
