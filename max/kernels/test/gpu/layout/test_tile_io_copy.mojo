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
"""Roundtrip tests for the `copy_*` free-function wrappers in tile_io.mojo.

These mirror the structure of `test_tile_io.mojo` but drive the data
movement through the module-level `copy_*` wrappers (which delegate to the
`TileCopier` structs) instead of constructing the copier structs directly.
Each kernel moves a source tile through one or more intermediate address
spaces and back out to a destination DRAM tile using only `TileTensor`; the
host verifies that the destination tile matches the source.

Coverage:

- GENERIC -> SHARED -> GENERIC via `copy_dram_to_sram` / `copy_sram_to_dram`.
- GENERIC -> LOCAL -> GENERIC via `copy_dram_to_local` / `copy_local_to_dram`.
- GENERIC -> SHARED -> LOCAL -> SHARED -> GENERIC via `copy_dram_to_sram`,
  `copy_sram_to_local`, `copy_local_to_shared`, `copy_sram_to_dram`
  (unswizzled).
- GENERIC -> LOCAL -> SHARED (swizzled) -> GENERIC (swizzled) via
  `copy_dram_to_local`, `copy_local_to_shared`, `copy_sram_to_dram`.
- GENERIC -> SHARED (cp.async) -> GENERIC via `copy_dram_to_sram_async` /
  `copy_sram_to_dram`.
- SHARED (swizzled) -> GENERIC via `copy_sram_to_dram` on vectorized tiles,
  where each thread owns a single vector.
- SHARED -> GENERIC via the `copy_sram_to_dram` overload that takes a worker
  index and bounds, with the copying threads starting past thread 0, only the
  leading rows and columns written, and a destination row stride narrower
  than the tile (so out-of-bounds columns alias the next row).
"""

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_all,
)

from layout import Coord, Idx, TileTensor, row_major
from layout.swizzle import Swizzle
from layout.tile_layout import Layout
from layout.tile_io import (
    copy_dram_to_local,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_shared,
    copy_sram_to_dram,
    copy_sram_to_local,
)
from layout.tile_tensor import stack_allocation

from std.testing import assert_equal


# 4x4 tile, distributed over 4 threads (2x2 thread layout -> each thread
# owns a 2x2 fragment).
comptime _N = 4
comptime _NUM_ELEMENTS = _N * _N
comptime _BLOCK_DIM = 4

# 8x16 tile of float32 vectorized by 4 and distributed over an 8x4 thread
# layout, so each thread owns exactly one vector.
comptime _V_ROWS = 8
comptime _V_COLS = 16
comptime _V_WIDTH = 4

# Threads [0, _W_OFFSET) sit out; the rest copy. The destination is a
# _V_ROWS x _V_COLS view with a row stride of _W_DST_STRIDE, and only its
# first _W_VALID_ROWS rows and _W_VALID_COLS columns may be written.
comptime _W_OFFSET = 32
comptime _W_VALID_ROWS = 5
comptime _W_VALID_COLS = 12
comptime _W_DST_STRIDE = 12
comptime _W_DST_SIZE = (_V_ROWS - 1) * _W_DST_STRIDE + _V_COLS


def dram_to_sram_to_dram_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Roundtrip a tile through shared memory using the free-function wrappers.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    copy_dram_to_sram[thread_layout](smem, src)
    barrier()
    copy_sram_to_dram[thread_layout](dst, smem)


def dram_to_local_to_dram_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Roundtrip a tile through registers using the free-function wrappers.

    Each thread holds its own 2x2 fragment in local memory.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var local = stack_allocation[dtype=DType.float32, address_space=.LOCAL](
        row_major[2, 2]()
    )

    copy_dram_to_local[thread_layout](local, src)
    copy_local_to_dram[thread_layout](dst, local)


def sram_local_sram_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Roundtrip a tile through shared -> local -> shared -> generic.

    Exercises `copy_sram_to_local` and `copy_local_to_shared` between two
    DRAM <-> SRAM legs.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())

    var smem_in = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )
    var smem_out = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )
    var local = stack_allocation[dtype=DType.float32, address_space=.LOCAL](
        row_major[2, 2]()
    )

    copy_dram_to_sram[thread_layout](smem_in, src)
    barrier()
    copy_sram_to_local[thread_layout](local, smem_in)
    copy_local_to_shared[thread_layout](smem_out, local)
    barrier()
    copy_sram_to_dram[thread_layout](dst, smem_out)


def swizzled_local_to_shared_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Swizzled roundtrip via the wrappers: GENERIC -> LOCAL -> SHARED
    (swizzled) -> GENERIC (swizzled).

    The SHARED -> GENERIC read must use the same swizzle as the LOCAL ->
    SHARED write to round-trip correctly.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])
    comptime swizzle = Swizzle(1, 0, 2)

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())

    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )
    var local = stack_allocation[dtype=DType.float32, address_space=.LOCAL](
        row_major[2, 2]()
    )

    copy_dram_to_local[thread_layout](local, src)
    copy_local_to_shared[thread_layout, swizzle=swizzle](smem, local)
    barrier()
    copy_sram_to_dram[thread_layout, swizzle=swizzle](dst, smem)


def swizzled_vectorized_sram_to_dram_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """SHARED (swizzled) -> GENERIC with vectorized tiles.

    Thread 0 fills shared memory in swizzled order, then every thread copies
    its one vector back out through `copy_sram_to_dram`.
    """
    comptime thread_layout = row_major(Idx[_V_ROWS], Idx[_V_COLS // _V_WIDTH])
    comptime swizzle = Swizzle(1, 2, 3)

    var dst = TileTensor(dst_ptr, row_major[_V_ROWS, _V_COLS]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_V_ROWS, _V_COLS]()
    )

    if thread_idx.x == 0:
        for i in range(_V_ROWS * _V_COLS):
            smem.ptr[swizzle(i)] = src_ptr[i]
    barrier()
    copy_sram_to_dram[thread_layout, swizzle=swizzle](
        dst.vectorize[1, _V_WIDTH](), smem.vectorize[1, _V_WIDTH]()
    )


def offset_worker_bounded_sram_to_dram_kernel[
    swizzle: Optional[Swizzle]
](
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """SHARED -> GENERIC with vectorized tiles, where the copying threads
    start at thread `_W_OFFSET` and only a `_W_VALID_ROWS` x `_W_VALID_COLS`
    corner of the destination is written.
    """
    comptime thread_layout = row_major(Idx[_V_ROWS], Idx[_V_COLS // _V_WIDTH])

    var dst = TileTensor(
        dst_ptr,
        Layout(
            Coord(Idx[_V_ROWS], Idx[_V_COLS]),
            Coord(Idx[_W_DST_STRIDE], Idx[1]),
        ),
    )
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_V_ROWS, _V_COLS]()
    )

    if thread_idx.x == 0:
        for i in range(_V_ROWS * _V_COLS):
            comptime if swizzle:
                smem.ptr[swizzle.value()(i)] = src_ptr[i]
            else:
                smem.ptr[i] = src_ptr[i]
    barrier()
    if thread_idx.x >= _W_OFFSET:
        copy_sram_to_dram[thread_layout, swizzle=swizzle](
            dst.vectorize[1, _V_WIDTH](),
            smem.vectorize[1, _V_WIDTH](),
            Int(thread_idx.x) - _W_OFFSET,
            _W_VALID_ROWS,
            _W_VALID_COLS,
        )


def async_dram_to_sram_to_dram_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Roundtrip through shared memory using `copy_dram_to_sram_async` for the
    DRAM->SMEM leg.

    The async copy must be committed and waited on before the destination
    tile can be read back; on AMD / Apple the underlying intrinsic falls back
    to synchronous loads, but the commit/wait calls remain valid no-ops.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    copy_dram_to_sram_async[thread_layout](smem, src)
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()
    copy_sram_to_dram[thread_layout](dst, smem)


def _run_roundtrip[
    kernel_fn: def(
        MutPointer[Float32, MutAnyOrigin],
        MutPointer[Float32, MutAnyOrigin],
    ) thin -> None,
    num_elements: Int = _NUM_ELEMENTS,
    block_dim: Int = _BLOCK_DIM,
](name: String, ctx: DeviceContext) raises:
    print("==", name)

    var src_host = ctx.enqueue_create_host_buffer[.float32](num_elements)
    for i in range(num_elements):
        src_host[i] = Float32(i + 1)

    var src_dev = ctx.enqueue_create_buffer[.float32](num_elements)
    var dst_dev = ctx.enqueue_create_buffer[.float32](num_elements)
    ctx.enqueue_copy(src_dev, src_host)

    ctx.enqueue_function[kernel_fn](
        src_dev, dst_dev, grid_dim=(1), block_dim=(block_dim)
    )

    var dst_host = ctx.enqueue_create_host_buffer[.float32](num_elements)
    ctx.enqueue_copy(dst_host, dst_dev)
    ctx.synchronize()

    for i in range(num_elements):
        assert_equal(dst_host[i], src_host[i])


def _run_offset_worker_bounded[
    swizzle: Optional[Swizzle]
](name: String, ctx: DeviceContext) raises:
    print("==", name)
    comptime num_elements = _V_ROWS * _V_COLS

    var src_host = ctx.enqueue_create_host_buffer[.float32](num_elements)
    var dst_host = ctx.enqueue_create_host_buffer[.float32](_W_DST_SIZE)
    for i in range(num_elements):
        src_host[i] = Float32(i + 1)
    for i in range(_W_DST_SIZE):
        dst_host[i] = -1.0

    var src_dev = ctx.enqueue_create_buffer[.float32](num_elements)
    var dst_dev = ctx.enqueue_create_buffer[.float32](_W_DST_SIZE)
    ctx.enqueue_copy(src_dev, src_host)
    ctx.enqueue_copy(dst_dev, dst_host)

    ctx.enqueue_function[offset_worker_bounded_sram_to_dram_kernel[swizzle]](
        src_dev,
        dst_dev,
        grid_dim=(1),
        block_dim=(_W_OFFSET + num_elements // _V_WIDTH),
    )

    ctx.enqueue_copy(dst_host, dst_dev)
    ctx.synchronize()

    for i in range(_W_DST_SIZE):
        var row, col = divmod(i, _W_DST_STRIDE)
        if row < _W_VALID_ROWS and col < _W_VALID_COLS:
            assert_equal(dst_host[i], src_host[row * _V_COLS + col])
        else:
            assert_equal(dst_host[i], -1.0)


def test_dram_to_sram_to_dram(ctx: DeviceContext) raises:
    _run_roundtrip[dram_to_sram_to_dram_kernel](
        "test_dram_to_sram_to_dram", ctx
    )


def test_dram_to_local_to_dram(ctx: DeviceContext) raises:
    _run_roundtrip[dram_to_local_to_dram_kernel](
        "test_dram_to_local_to_dram", ctx
    )


def test_sram_local_sram_roundtrip(ctx: DeviceContext) raises:
    _run_roundtrip[sram_local_sram_kernel](
        "test_sram_local_sram_roundtrip", ctx
    )


def test_swizzled_local_to_shared(ctx: DeviceContext) raises:
    _run_roundtrip[swizzled_local_to_shared_kernel](
        "test_swizzled_local_to_shared", ctx
    )


def test_swizzled_vectorized_sram_to_dram(ctx: DeviceContext) raises:
    _run_roundtrip[
        swizzled_vectorized_sram_to_dram_kernel,
        num_elements=_V_ROWS * _V_COLS,
        block_dim=_V_ROWS * _V_COLS // _V_WIDTH,
    ]("test_swizzled_vectorized_sram_to_dram", ctx)


def test_offset_worker_bounded_sram_to_dram(ctx: DeviceContext) raises:
    _run_offset_worker_bounded[Swizzle(1, 2, 3)](
        "test_offset_worker_bounded_sram_to_dram_swizzled", ctx
    )
    _run_offset_worker_bounded[None](
        "test_offset_worker_bounded_sram_to_dram_unswizzled", ctx
    )


def test_async_dram_to_sram_to_dram(ctx: DeviceContext) raises:
    _run_roundtrip[async_dram_to_sram_to_dram_kernel](
        "test_async_dram_to_sram_to_dram", ctx
    )


def main() raises:
    with DeviceContext() as ctx:
        test_dram_to_sram_to_dram(ctx)
        test_dram_to_local_to_dram(ctx)
        test_sram_local_sram_roundtrip(ctx)
        test_swizzled_local_to_shared(ctx)
        test_swizzled_vectorized_sram_to_dram(ctx)
        test_offset_worker_bounded_sram_to_dram(ctx)
        test_async_dram_to_sram_to_dram(ctx)
