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
"""Input batching for the unified DSpark DeepSeek-V4.1-Flash pipeline model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from max import tree
from max.driver import CPU, Buffer
from max.dtype import DType
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.context import TextContext
from max.pipelines.lib import UnifiedSpecDecodeInputs
from max.pipelines.lib.interfaces.batch_processor import (
    UnifiedSpecDecodeBatchProcessor,
)
from max.pipelines.lib.utils import compute_data_parallel_splits
from typing_extensions import override

from ..deepseekV41.model import engram_lookback_ids_for_contexts
from ..deepseekV41.model_config import DeepseekV41Config

__all__ = [
    "MERGED_ROWS_DIM",
    "UnifiedDSparkDeepseekV41BatchProcessor",
    "UnifiedDSparkDeepseekV41Inputs",
]

MERGED_ROWS_DIM = "merged_seq_len"
"""Name of the graph dimension counting prompt tokens plus all draft tokens.

The graph's last input has this length, which sizes the combined
prompt-and-draft token tensors inside the graph.
"""


@dataclass
class UnifiedDSparkDeepseekV41Inputs(UnifiedSpecDecodeInputs):
    """Graph inputs for one step of target-plus-drafter decoding.

    :attr:`buffers` lists them in graph-input order: the ragged tokens; the
    per-device image span rows and their token rows (set by the pipeline's
    vision encoder cache, zero rows on a step without images); the row
    offsets (and, under data parallelism, their host copy) and logit count;
    under data parallelism the replica splits; the signal buffers (only with
    more than one device); the KV cache buffers; under data parallelism each
    device's context length and the expert-parallel buffers; the
    speculative-decoding inputs; the Engram
    lookback window and pinned Engram tables (only when the model has
    Engram); and last, a placeholder buffer whose length sets
    :data:`MERGED_ROWS_DIM`.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    signal_buffers: list[Buffer]
    host_input_row_offsets: Buffer | None = None
    """CPU copy of :attr:`input_row_offsets`; only under data parallelism."""
    data_parallel_splits: Buffer | None = None
    """Int64 ``[replicas + 1]`` request boundaries of each replica, on CPU.

    Only under data parallelism, where the batch is replica-major.
    """
    batch_context_lengths: list[Buffer] = field(default_factory=list)
    """Per device, its replica's summed context length, int32 ``[1]`` on CPU.

    Only under data parallelism.
    """
    ep_buffers: Sequence[Buffer] = ()
    """The routed experts' expert-parallel communication buffers; only
    under data parallelism."""
    engram_lookback_ids: Buffer | None = None
    """``[batch, max_ngram_size - 1]`` int64 token ids before each chunk.

    Each request's raw token ids just before its chunk, on device 0.
    """
    engram_host_buffers: Sequence[Buffer] = ()
    """The pinned host Engram tables, the graph's last inputs but one.

    Held here so :attr:`buffers` is the complete input list, which device
    graph capture needs.
    """

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        lookback = (
            (self.engram_lookback_ids,)
            if self.engram_lookback_ids is not None
            else ()
        )
        host_offsets: tuple[Buffer, ...] = ()
        splits: tuple[Buffer, ...] = ()
        if self.data_parallel_splits is not None:
            assert self.host_input_row_offsets is not None
            host_offsets = (self.host_input_row_offsets,)
            splits = (self.data_parallel_splits,)
        return (
            self.tokens,
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            self.input_row_offsets,
            *host_offsets,
            self.return_n_logits,
            *splits,
            *self.signal_buffers,
            *tree.leaves(self.kv_cache_inputs),
            *self.batch_context_lengths,
            *self.ep_buffers,
            *self._spec_decode_tail_buffers(include_in_thinking_phase=False),
            *lookback,
            *self.engram_host_buffers,
            self._merged_rows_marker(),
        )

    def _merged_rows_marker(self) -> Buffer:
        """Returns a host int8 placeholder sized to the merged token count.

        Its shape is ``[S + B * K]`` (the prompt tokens plus every draft
        token), which sets :data:`MERGED_ROWS_DIM`; its contents are never
        read.
        """
        assert self.draft_tokens is not None
        batch, width = self.draft_tokens.shape
        rows = self.tokens.shape[0] + batch * width
        return Buffer(shape=(rows,), dtype=DType.int8, device=CPU())


class UnifiedDSparkDeepseekV41BatchProcessor(
    UnifiedSpecDecodeBatchProcessor[
        UnifiedDSparkDeepseekV41Inputs, DeepseekV41Config
    ]
):
    """Ragged batching with the sampling seed and the Engram lookback window."""

    _ep_buffers: tuple[Buffer, ...] = ()

    def bind_ep_buffers(self, buffers: Sequence[Buffer]) -> None:
        """Binds the expert-parallel buffers created at model load."""
        self._ep_buffers = tuple(buffers)

    @override
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> UnifiedDSparkDeepseekV41Inputs:
        inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )
        # One batch per data-parallel replica.
        if len(replica_batches) > 1:
            contexts = [ctx for batch in replica_batches for ctx in batch]
            inputs.host_input_row_offsets = Buffer.from_numpy(
                np.cumsum(
                    [0] + [ctx.tokens.active_length for ctx in contexts],
                    dtype=np.uint32,
                )
            )
            inputs.data_parallel_splits = Buffer.from_numpy(
                compute_data_parallel_splits(replica_batches)
            )
            inputs.batch_context_lengths = [
                Buffer.from_numpy(
                    np.array(
                        [sum(ctx.tokens.current_position for ctx in batch)],
                        dtype=np.int32,
                    )
                )
                for batch in replica_batches
            ]
        if self.config.engram_layer_ids:
            # The verification pass's chunk starts at each request's
            # processed length, the same cache length the graph gets: the
            # drafts it verifies follow the chunk, so the lookback is
            # unchanged.
            contexts = [ctx for batch in replica_batches for ctx in batch]
            inputs.engram_lookback_ids = Buffer.from_numpy(
                engram_lookback_ids_for_contexts(
                    contexts, self.config.engram_max_ngram_size - 1
                )
            ).to(self.runtime.devices[0])
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
    ) -> UnifiedDSparkDeepseekV41Inputs:
        return UnifiedDSparkDeepseekV41Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            # Empty on one device, as the graph declares none there.
            signal_buffers=list(self.runtime.signal_buffers),
            kv_cache_inputs=kv_cache_inputs,
            seed=seed,
            structured_output=structured_output,
            ep_buffers=self._ep_buffers,
        )
