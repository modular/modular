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
"""The Engram lookback window's device history, as graph ops.

Under the overlap scheduler the host does not yet know a step's accepted
drafts when it builds the next step's window, so the window a request's
Engram hash reads comes from a history buffer the previous step recorded
in-graph. Both V4.1 trunks (V2's serving model and the ModuleV3 export)
read and record through these two helpers, so the semantics cannot fork.
The batch state that picks host window or history row per request -- the
persistent buffer itself, and each request's row in the previous batch --
is shared here too (:class:`EngramLookbackBatchState`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorValue, ops
from max.pipelines.context import TextContext
from max.pipelines.request import RequestID
from max.tensor import Tensor

# V2's serving model calls these with graph values; the ModuleV3 trunk
# holds the same values wrapped as Tensors. Both realize into the same ops.
_GraphOrTensor = TensorValue | Tensor


def read_engram_lookback(
    host_lookback: _GraphOrTensor, source: _GraphOrTensor, history: BufferValue
) -> _GraphOrTensor:
    """Returns the lookback window of each row, from the host or the history.

    Args:
        host_lookback: ``[batch, depth]`` window the host built from the
            request's ``processed_length``.
        source: ``[batch]`` int64, the row of ``history`` to read, or -1 for
            the host window.
        history: ``[rows, depth]`` windows the previous step recorded.
    """
    device = host_lookback.device
    zero = ops.constant(0, DType.int64, device=device)
    recorded = ops.gather(
        ops.buffer_load(history), ops.max(source, zero), axis=0
    )
    return ops.where(ops.unsqueeze(source >= zero, -1), recorded, host_lookback)


def record_engram_lookback(
    history: BufferValue,
    lookback: _GraphOrTensor,
    tokens: TensorValue,
    input_row_offsets: TensorValue,
    draft_tokens: TensorValue,
    num_accepted: TensorValue,
) -> None:
    """Stores each row's lookback window for the step after this one.

    The next step starts after the row's chunk and its accepted drafts, so
    its window ends with those tokens, then this step's window.

    Args:
        history: ``[rows, depth]`` buffer; row ``i`` receives batch row
            ``i``'s window.
        lookback: ``[batch, depth]`` this step's window; column ``j`` is the
            token ``j + 1`` positions before the chunk.
        tokens: ``[total_seq_len]`` ragged chunk tokens.
        input_row_offsets: ``[batch + 1]`` offsets of the chunks.
        draft_tokens: ``[batch, num_drafts]`` drafts this step verified.
        num_accepted: ``[batch]`` drafts accepted per row.
    """
    device = lookback.device
    batch, depth = lookback.shape[0], int(lookback.shape[1])
    i = ops.range(0, depth, 1, depth, dtype=DType.int64, device=device)
    starts = input_row_offsets[:-1].cast(DType.int64).rebind([batch])
    lengths = input_row_offsets[1:].cast(DType.int64).rebind([batch]) - starts
    # The last `depth` tokens up to the chunk's end, oldest first: chunk
    # position `lengths - depth + i`, or this step's window before it.
    position = ops.unsqueeze(lengths, -1) - depth + i
    zero = ops.constant(0, DType.int64, device=device)
    from_chunk = ops.gather(
        tokens, ops.unsqueeze(starts, -1) + ops.max(position, zero), axis=0
    )
    from_window = ops.gather_nd(
        lookback,
        ops.unsqueeze(ops.max(-1 - position, zero), -1),
        batch_dims=1,
    )
    ordered = ops.concat(
        [
            ops.where(position >= zero, from_chunk, from_window),
            draft_tokens.rebind([batch, draft_tokens.shape[1]]),
        ],
        axis=1,
    )
    # Newest first, ending with the last accepted draft.
    newest = ops.unsqueeze(
        num_accepted.cast(DType.int64).rebind([batch]), -1
    ) + (depth - 1 - i)
    window = ops.gather_nd(ordered, ops.unsqueeze(newest, -1), batch_dims=1)
    rows = ops.shape_to_tensor([batch]).reshape([])
    first = ops.constant(0, DType.int64, DeviceRef.CPU())
    ops.buffer_store_slice(
        history, window, [(slice(first, rows), batch), slice(None)]
    )


class EngramLookbackBatchState:
    """The lookback batch state both V4.1 trunks stage their steps from.

    Owns the persistent history buffer a step records the next step's
    windows in (allocated once, so every step's graph reads the same
    address), and the row each request sat in the previous batch at,
    which is where this batch's rows source their windows.
    """

    def __init__(self) -> None:
        self.history: Buffer | None = None
        self._prev_rows: dict[RequestID, int] = {}

    def ensure_history(
        self, *, capacity: int, batch: int, depth: int, device: Device
    ) -> Buffer:
        """Returns the persistent history buffer, allocated at first use.

        Rows follow the global batch, as the window does, so one buffer
        serves every data-parallel replica; a step records batch row
        ``i``'s window into history row ``i`` for the next step to read.

        Args:
            capacity: Rows to allocate: the widest batch the runtime
                admits.
            batch: Rows this batch stages; the buffer refuses a batch
                wider than what it holds.
            depth: The lookback's columns, ``engram_max_ngram_size - 1``.
            device: The device the graph reads and records the history on.

        Raises:
            ValueError: If ``batch`` exceeds the allocated rows.
        """
        history = self.history
        if history is None:
            history = Buffer.zeros(
                (capacity, depth), DType.int64, device=device
            )
            self.history = history
        if batch > history.shape[0]:
            raise ValueError(
                f"{batch} requests exceed the "
                f"{history.shape[0]}-row Engram lookback history"
            )
        return history

    def source(self, contexts: Sequence[TextContext]) -> np.ndarray:
        """Returns each row's source map: its history row, or -1.

        A request whose previous step is still in flight starts after
        that step's accepted drafts, which only the device knows, so the
        graph reads the window that step recorded; the predicate is the
        one the overlap pipeline realizes the cache lengths with. The
        previous-row map reseats at this batch's rows, so the batch after
        this one sources from it.

        Args:
            contexts: The batch's requests, in row order.

        Returns:
            ``[batch]`` int64 row of :attr:`history` to read each request's
            window from, or -1 for the host window.
        """
        source = np.array(
            [
                self._prev_rows.get(ctx.request_id, -1)
                if ctx.spec_decoding_state.maybe_accepted_draft_tokens
                and ctx.tokens.generated_length
                else -1
                for ctx in contexts
            ],
            np.int64,
        )
        self._prev_rows = {ctx.request_id: i for i, ctx in enumerate(contexts)}
        return source
