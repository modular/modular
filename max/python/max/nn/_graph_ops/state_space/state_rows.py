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
"""Moves a batch's rows between a recurrent state leaf and a tensor.

For state no kernel reads in place: both go through
``state_space_copy_rows`` and a batch-sized scratch buffer, so only the
batch's rows move. A cache leaf spans every row the cache holds, so loading it
whole to gather a batch's rows would copy the whole leaf every step, and
storing it whole would also rewrite rows other requests and cached prefixes
own.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    DimLike,
    TensorValue,
    ops,
)


def _batch_rows(batch: DimLike, device: DeviceRef) -> TensorValue:
    """Returns ``[1, batch]`` uint32 rows of a batch-ordered scratch buffer."""
    rows = ops.range(
        0,
        ops.shape_to_tensor([batch])[0],
        1,
        out_dim=batch,
        dtype=DType.int64,
        device=device,
    )
    return ops.reshape(rows.cast(DType.uint32), (1, batch))


def read_state_rows(pool: BufferValue, rows: TensorValue) -> TensorValue:
    """Returns ``pool[rows]``, copying only those rows out of the pool.

    Args:
        pool: A state leaf, ``[num_rows, ...]``.
        rows: ``[batch]`` uint32 rows of ``pool``. Its dim sizes the scratch
            buffer, so it must be a named dim rather than one derived from
            another.

    Returns:
        ``[batch, ...]`` rows of ``pool``.
    """
    batch = rows.shape[0]
    scratch = ops.buffer_create(
        BufferType(pool.dtype, [batch, *pool.shape[1:]], device=pool.device)
    )
    ops.inplace_custom(
        "state_space_copy_rows",
        pool.device,
        [
            scratch,
            pool,
            _batch_rows(batch, pool.device),
            ops.reshape(rows.cast(DType.uint32), (1, batch)),
        ],
    )
    return ops.buffer_load(scratch)


def write_state_rows(
    pool: BufferValue, rows: TensorValue, values: TensorValue
) -> None:
    """Writes ``values[b]`` to ``pool[rows[b]]`` in place.

    Args:
        pool: A state leaf, ``[num_rows, ...]``.
        rows: ``[batch]`` uint32 rows of ``pool``, over a named dim as in
            :func:`read_state_rows`.
        values: ``[batch, ...]`` rows at the pool's dtype.
    """
    batch = rows.shape[0]
    values = values.rebind([batch, *values.shape[1:]])
    scratch = ops.buffer_create(
        BufferType(values.dtype, values.shape, device=values.device)
    )
    ops.buffer_store(scratch, values)
    ops.inplace_custom(
        "state_space_copy_rows",
        pool.device,
        [
            pool,
            scratch,
            ops.reshape(rows.cast(DType.uint32), (1, batch)),
            _batch_rows(batch, pool.device),
        ],
    )
