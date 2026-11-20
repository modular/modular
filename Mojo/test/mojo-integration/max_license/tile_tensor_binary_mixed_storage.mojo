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

# The TensorOps elementwise binary ops accept an rhs with a different storage
# engine than the destination. Inside a GPU kernel this mixes a local
# `DefaultEngine`-backed tile with a global `DevicePointerEngine`-backed
# tensor in both directions: the rhs is loaded through its own engine and the
# destination is read and written through its own engine, including the
# rank-1 broadcast path. TileTensor's in-place operators gate on matching
# storage classes (see tile_tensor_binary_mixed_storage_error.mojo), so
# mixed-engine ops go through the engine-level TensorOps API.

# REQUIRES: NVIDIA-GPU || AMD-GPU
# RUN: %mojo %s | FileCheck %s

from layout import TileTensor, row_major
from layout.tensor_engine import (
    DevicePointerEngine,
    StaticOffsetEngine,
    TensorOps,
)

from max.gpu import global_idx
from max.gpu.host import DeviceContext

comptime _ROWS = 2
comptime _COLS = 4
comptime _NUM_ELEMENTS = _ROWS * _COLS

comptime _GlobalTile = TileTensor[
    .float32,
    type_of(row_major[_ROWS, _COLS]()),
    UnsafeAnyOrigin[mut=True],
    Engine=DevicePointerEngine[element_width=1],
]


def _add_into(
    dst: TileTensor, rhs: TileTensor[dst.dtype, ...]
) where dst.mut and conforms_to(dst.Engine, TensorOps):
    """Engine-level `TensorOps.add`, open to mixed engines.

    The rhs engine only needs `TensorEngine`, and its `OtherEngine`
    parameter is only inferable from a symbolic rhs storage, so the
    mixed-engine calls go through this generic helper rather than a direct
    call with concrete handles.
    """
    type_of(dst).Engine.iadd(
        (dst._unsafe_storage_cast[to_mut=True](), dst.layout),
        (rhs._storage, rhs.layout),
    )


def _mul_into(
    dst: TileTensor, rhs: TileTensor[dst.dtype, ...]
) where dst.mut and conforms_to(dst.Engine, TensorOps):
    """Engine-level `TensorOps.mul`, open to mixed engines."""
    type_of(dst).Engine.imul(
        (dst._unsafe_storage_cast[to_mut=True](), dst.layout),
        (rhs._storage, rhs.layout),
    )


def mixed_storage_binary_ops_kernel(a: _GlobalTile, result: _GlobalTile):
    """Accumulates the global tensor into a local tile and writes back.

    The first `add` and `mul` load the rhs through `DevicePointerEngine`
    into a `DefaultEngine` destination; the next `add` stores through
    `DevicePointerEngine` from a `DefaultEngine` rhs; the last `add` reads
    the rhs through `StaticOffsetEngine`, skipping a sentinel header.
    """
    if global_idx.x != 0:
        return

    var data = Array[Float32, _NUM_ELEMENTS](fill=0)
    var local = TileTensor(data, row_major[_ROWS, _COLS]())
    for i in range(_ROWS):
        for j in range(_COLS):
            local[i, j] = Float32(i * _COLS + j)

    _add_into(local, a)
    _mul_into(local, a)
    _add_into(result, local)

    var off_data = Array[Float32, 1 + _NUM_ELEMENTS](fill=-1000)
    for i in range(_NUM_ELEMENTS):
        off_data[1 + i] = Float32(i)
    var off_tile = TileTensor[
        .float32,
        type_of(row_major[_ROWS, _COLS]()),
        origin_of(off_data),
        Engine=StaticOffsetEngine[static_offset=1],
    ](off_data.unsafe_ptr(), row_major[_ROWS, _COLS]())
    _add_into(result, off_tile)


def mixed_storage_broadcast_kernel(result: _GlobalTile):
    """Broadcast-adds a local rank-1 tile into the global rank-2 tensor."""
    if global_idx.x != 0:
        return

    var data = Array[Float32, _ROWS](fill=0)
    var bias = TileTensor(data, row_major[_ROWS]())
    bias[0] = 1000.0
    bias[1] = 2000.0

    _add_into(result, bias)


def main() raises:
    with DeviceContext() as ctx:
        var a_buf = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
        var out_buf = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
        out_buf.enqueue_fill(Float32(0))

        var host_in = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
        for i in range(_NUM_ELEMENTS):
            host_in[i] = Float32(10 * (i + 1))
        ctx.enqueue_copy(a_buf, host_in)

        var a_tile = TileTensor(a_buf.device_ptr(), row_major[_ROWS, _COLS]())
        var out_tile = TileTensor(
            out_buf.device_ptr(), row_major[_ROWS, _COLS]()
        )

        ctx.enqueue_function[mixed_storage_binary_ops_kernel](
            a_tile.as_unsafe_any_origin(),
            out_tile.as_unsafe_any_origin(),
            grid_dim=1,
            block_dim=1,
        )
        ctx.enqueue_function[mixed_storage_broadcast_kernel](
            out_tile.as_unsafe_any_origin(),
            grid_dim=1,
            block_dim=1,
        )

        var host_out = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
        ctx.enqueue_copy(host_out, out_buf)
        ctx.synchronize()

        # local = arange(8); local += a -> [10, 21, ..., 87];
        # local *= a -> [100, 420, ..., 6960]; result += local; result +=
        # offset-viewed arange(8); then the broadcast kernel adds 1000 to
        # row 0 and 2000 to row 1.
        # CHECK: 1100.0
        # CHECK-NEXT: 1421.0
        # CHECK-NEXT: 1962.0
        # CHECK-NEXT: 2723.0
        # CHECK-NEXT: 4704.0
        # CHECK-NEXT: 5905.0
        # CHECK-NEXT: 7326.0
        # CHECK-NEXT: 8967.0
        for i in range(_NUM_ELEMENTS):
            print(host_out[i])
