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

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    Type,
    Value,
    ops,
)
from max.graph import Module as GraphModule
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.speculative.block_driver import (
    BlockBatch,
    BlockDriver,
    block_dispatch_metadata,
    block_kv_with_dispatch,
)
from max.pipelines.speculative.spec_target import Verified

from .gemma4 import Gemma4TextModel

__all__ = [
    "SLIDING_KV",
    "Gemma4BlockDriver",
    "Gemma4BlockTarget",
    "block_dispatch_metadata",
    "block_kv_with_dispatch",
]

SLIDING_KV = "sliding_attention"
"""Name the sliding leaf rides under in :attr:`BlockBatch.passthrough_kv`. """


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


class Gemma4BlockDriver(BlockDriver[TensorValue, TensorValue]):
    """A block draft on the block driver, behind a Gemma4 target.

    Traces the unified spec-decode graph with the target's paired caches
    routed the same way for every block draft: the target verifies on its
    full-attention cache and carries the sliding cache through.
    """

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces one spec-decode step, whose inputs are :meth:`input_types`.

        Decodes the flat graph inputs by :meth:`decode_inputs`. Inputs the
        signature doesn't declare reach :meth:`__call__` as ``None``.

        Args:
            *input_types: The graph's input types, :meth:`input_types`.
            name: The name of the graph. Defaults to the class's qualified
                name.
            module: The MLIR module to add the graph to. Defaults to a new
                module.
            custom_extensions: Paths to custom kernel libraries the graph
                uses.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced :class:`~max.graph.Graph`.
        """

        def forward(*inputs: Value[Any]) -> tuple[TensorValue, ...]:
            graph_inputs = self.decode_inputs(inputs)
            return self(
                tokens=graph_inputs.tokens,
                input_row_offsets=graph_inputs.input_row_offsets,
                draft_tokens=graph_inputs.draft_tokens,
                signal_buffers=graph_inputs.signal_buffers,
                kv_collections=graph_inputs.kv("target", "full_attention"),
                passthrough_kv={
                    SLIDING_KV: graph_inputs.kv("target", SLIDING_KV)
                },
                draft_kv_collections=graph_inputs.kv("draft"),
                return_n_logits=graph_inputs.return_n_logits,
                seed=graph_inputs.seed,
                temperature=graph_inputs.temperature,
                top_k=graph_inputs.top_k,
                max_k=graph_inputs.max_k,
                top_p=graph_inputs.top_p,
                min_top_p=graph_inputs.min_top_p,
                draft_probs_full=graph_inputs.draft_probs_full,
                in_thinking_phase=graph_inputs.in_thinking_phase,
                pinned_bitmask=graph_inputs.pinned_bitmask,
                wait_payload=graph_inputs.wait_payload,
                device_bitmask_scratch=graph_inputs.device_bitmask_scratch,
            )

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )
