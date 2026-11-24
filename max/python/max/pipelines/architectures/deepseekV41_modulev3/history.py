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
"""Request-owned Engram lookback in the KV arena."""

from collections.abc import Sequence

from max import functional as F
from max.dtype import DType
from max.graph import BufferValue, TensorValue, ops
from max.nn._graph_ops.state_space import read_state_rows, write_state_rows
from max.nn.common_layers.kv_cache import cache_tensor
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    RecurrentLeafInputs,
    RecurrentStateInputsPerDevice,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping


def distributed_history(
    per_device: Sequence[
        RecurrentStateInputsPerDevice[TensorValue, BufferValue]
        | KVCacheInputsPerDevice[TensorValue, BufferValue]
    ],
    mapping: DeviceMapping,
) -> RecurrentLeafInputs[Tensor, Tensor]:
    """Keeps each replica's pool and live row indices together."""
    records: list[RecurrentStateInputsPerDevice[TensorValue, BufferValue]] = []
    for record in per_device:
        assert isinstance(record, RecurrentStateInputsPerDevice)
        records.append(record)
    return RecurrentLeafInputs(
        pool=cache_tensor([x.leaves[0].pool for x in records], mapping),
        live_row_ids=cache_tensor(
            [x.leaves[0].live_row_ids for x in records], mapping
        ),
    )


def _read(
    pool: BufferValue,
    row_ids: TensorValue,
    host: TensorValue,
    source: TensorValue,
) -> TensorValue:
    # Only the batch's live rows move: the pool views the whole slab. A pool
    # row may be wider than the window (see LOOKBACK_HISTORY_ROW).
    depth = int(host.shape[1])
    recorded = read_state_rows(pool, row_ids[0])[:, :depth].rebind(host.shape)
    return ops.where(
        ops.unsqueeze(source.rebind([host.shape[0]]) >= 0, -1), recorded, host
    )


def _mask(
    source: TensorValue,
    lookback: TensorValue,
    host_dead: TensorValue,
    image_token_id: int,
) -> TensorValue:
    return ops.where(
        ops.unsqueeze(source.rebind([lookback.shape[0]]) >= 0, -1),
        (lookback == image_token_id).cast(DType.uint8),
        host_dead.rebind(lookback.shape),
    )


history_mask = F.functional(_mask)


read_history = F.functional(_read)


def _record(
    pool: BufferValue,
    row_ids: TensorValue,
    lookback: TensorValue,
    tokens: TensorValue,
    offsets: TensorValue,
    drafts: TensorValue,
    accepted: TensorValue,
) -> None:
    batch, depth = lookback.shape[0], int(lookback.shape[1])
    i = ops.range(0, depth, 1, depth, dtype=DType.int64, device=lookback.device)
    starts = offsets[:-1].cast(DType.int64).rebind([batch])
    lengths = offsets[1:].cast(DType.int64).rebind([batch]) - starts
    position = ops.unsqueeze(lengths, -1) - depth + i
    zero = ops.constant(0, DType.int64, device=lookback.device)
    chunk = ops.gather(
        tokens, ops.unsqueeze(starts, -1) + ops.max(position, zero), axis=0
    )
    previous = ops.gather_nd(
        lookback, ops.unsqueeze(ops.max(-1 - position, zero), -1), batch_dims=1
    )
    ordered = ops.concat(
        [
            ops.where(position >= zero, chunk, previous),
            drafts.rebind([batch, drafts.shape[1]]),
        ],
        axis=1,
    )
    newest = (
        ops.unsqueeze(accepted.cast(DType.int64).rebind([batch]), -1)
        + depth
        - 1
        - i
    )
    window = ops.gather_nd(ordered, ops.unsqueeze(newest, -1), batch_dims=1)
    width = int(pool.shape[1])
    if width > depth:
        window = ops.concat(
            [
                window,
                ops.broadcast_to(
                    ops.constant(0, DType.int64, device=lookback.device),
                    [batch, width - depth],
                ),
            ],
            axis=1,
        )
    # The live indices follow request identity through reorder and
    # preemption, and only those rows of the slab are written.
    write_state_rows(pool, row_ids[0], window)


record_history = F.functional(_record)
