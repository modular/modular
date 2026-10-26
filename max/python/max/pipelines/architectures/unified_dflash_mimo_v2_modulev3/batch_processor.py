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
"""Inputs and batching for the ModuleV3 fused MiMo-V2 DFlash pipeline model.

The input ABI is the graph-API fused graph's, which Mach's block-spec
executor binds at every device count: a multi-device trace inserts the
mesh's signal buffers at the slot the graph-API signature declared, and a
single-device one declares its own there. Under data-parallel attention the
CPU batch-split pair joins the batch inputs ahead of that slot, and the
per-device batch context lengths and the EP tail follow the KV tree, as the
canonical distributed signature declares them. The generic
``SpecDecodePipelineModel`` batching packs a composed graph's signature;
this package builds its fused module directly, so it owns its packing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from max import tree
from max.driver import Buffer
from max.dtype import DType
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.context import TextContext
from max.pipelines.lib import UnifiedSpecDecodeInputs
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    UnifiedSpecDecodeBatchProcessor,
)
from max.pipelines.lib.utils import compute_data_parallel_splits
from typing_extensions import override

from .model_config import UnifiedDflashMiMoV2Config


@dataclass
class UnifiedDflashMiMoV2Inputs(UnifiedSpecDecodeInputs):
    """The fused graph's inputs: tokens, one row-offsets tensor, the logit
    count, a signal buffer per device, the ``{target, draft}`` KV tree, then
    the spec-decode tail.

    Under data-parallel attention the batch side also carries the CPU pair
    the forward splits each replica's batch by -- the host row offsets
    beside the offsets, the per-replica boundaries past the logit count --
    and the KV tree is followed by one context-length tensor per device and
    the EP input tail.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    signal_buffers: list[Buffer]
    host_input_row_offsets: Buffer | None = None
    """The row offsets on the host, beside ``input_row_offsets``."""
    data_parallel_splits: Buffer | None = None
    """Int64 ``[replicas + 1]`` cumulative request boundaries, past
    ``return_n_logits``: where each replica's requests start in the
    concatenated batch."""
    batch_context_lengths: list[Buffer] = field(default_factory=list)
    """One per-device ``[1]`` context length, past the KV leaves."""
    ep_inputs: tuple[Buffer, ...] = ()
    """The EP collective buffers, past the context lengths."""

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        # The pair arrives together or not at all: either one alone shifts
        # every slot past it, so the wiring error surfaces here rather
        # than in a downstream explosion -- the DSpark sibling's rule.
        assert (self.host_input_row_offsets is None) == (
            self.data_parallel_splits is None
        )
        host = (
            (self.host_input_row_offsets,)
            if self.host_input_row_offsets is not None
            else ()
        )
        splits = (
            (self.data_parallel_splits,)
            if self.data_parallel_splits is not None
            else ()
        )
        buffers = (
            self.tokens,
            self.input_row_offsets,
            *host,
            self.return_n_logits,
            *splits,
            *self.signal_buffers,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs
                else ()
            ),
            *self.batch_context_lengths,
            *self.ep_inputs,
        )
        return buffers + self._spec_decode_tail_buffers(
            include_in_thinking_phase=False
        )


class UnifiedDflashMiMoV2ModuleV3BatchProcessor(
    UnifiedSpecDecodeBatchProcessor[
        UnifiedDflashMiMoV2Inputs, UnifiedDflashMiMoV2Config
    ]
):
    """Ragged batching for the fused graph, with the per-device signal
    buffers its collectives need.

    Under data parallelism every replica's requests concatenate into one
    device-0 batch beside the CPU split boundaries and host row offsets the
    graph slices it by -- the runtime half of the DP inputs the graph's
    signature declares, as the ModuleV3 base model's batch processor packs
    them.
    """

    def __init__(
        self, config: UnifiedDflashMiMoV2Config, runtime: BatchProcessorRuntime
    ) -> None:
        super().__init__(config, runtime)
        self._ep_comm_initializer: EPCommInitializer | None = None
        # The canonical distributed signature declares one per-device
        # context length the engine binds; nothing in this graph reads
        # them, so preallocated zeros serve, as the M3 DSpark model's do.
        self._batch_context_lengths: list[Buffer] = []
        if runtime.data_parallel_degree > 1:
            self._batch_context_lengths = [
                Buffer.zeros(shape=[1], dtype=DType.int32)
                for _ in runtime.devices
            ]

    def bind_ep_comm_initializer(
        self, initializer: EPCommInitializer | None
    ) -> None:
        """Wires the EP buffers the model's load created."""
        self._ep_comm_initializer = initializer

    def _ep_inputs(self) -> tuple[Buffer, ...]:
        if self._ep_comm_initializer is None:
            return ()
        return tuple(self._ep_comm_initializer.model_inputs())

    @override
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> UnifiedDflashMiMoV2Inputs:
        """Packs the ragged batch, with the DP split tensors the forward
        reads.

        Under data parallelism every replica's requests concatenate into
        one device-0 batch, beside the CPU split boundaries and host row
        offsets the graph slices it by.
        """
        dp_degree = self.runtime.data_parallel_degree
        if dp_degree > 1 and len(replica_batches) != dp_degree:
            raise ValueError(
                f"MiMo-V2 DFlash data parallelism expects {dp_degree} "
                f"replica batches, got {len(replica_batches)}."
            )

        inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        if dp_degree > 1:
            # An empty replica contributes a zero-width boundary; the
            # in-graph split still hands it a 0-token shard.
            inputs.data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
            # The host mirror beside the device offsets, cumsum over the
            # same request lengths the staged offsets carry.
            inputs.host_input_row_offsets = Buffer.from_numpy(
                np.cumsum(
                    [0]
                    + [
                        ctx.tokens.active_length
                        for batch in replica_batches
                        for ctx in batch
                    ],
                    dtype=np.uint32,
                )
            )
        return inputs

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        seed: Buffer,
        structured_output: bool,
    ) -> UnifiedDflashMiMoV2Inputs:
        return UnifiedDflashMiMoV2Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            signal_buffers=list(self.runtime.signal_buffers),
            batch_context_lengths=list(self._batch_context_lengths),
            ep_inputs=self._ep_inputs(),
            kv_cache_inputs=kv_cache_inputs,
            seed=seed,
            structured_output=structured_output,
        )
