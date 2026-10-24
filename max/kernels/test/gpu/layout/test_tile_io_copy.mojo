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
- LOCAL -> SHARED (swizzled) via `copy_local_to_shared` on vectorized tiles,
  where each thread owns a single vector.
"""

from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_all,
)

from layout import Coord, Idx, TileTensor, row_major
from layout.tile_io import ThreadScope
from layout.tile_layout import Layout as TileLayout
from layout.swizzle import Swizzle
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
            smem.reshape[_V_ROWS * _V_COLS]()[swizzle(i)] = src_ptr[i]
    barrier()
    copy_sram_to_dram[thread_layout, swizzle=swizzle](
        dst.vectorize[1, _V_WIDTH](), smem.vectorize[1, _V_WIDTH]()
    )


def swizzled_vectorized_local_to_shared_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """LOCAL -> SHARED (swizzled) with vectorized tiles.

    Each thread fills its one-vector local tile, copies it into shared memory
    through `copy_local_to_shared`, and thread 0 then reads shared memory back
    in swizzled order. Shared memory is sentinel-filled first, since it is not
    cleared between launches.
    """
    comptime thread_layout = row_major(Idx[_V_ROWS], Idx[_V_COLS // _V_WIDTH])
    comptime swizzle = Swizzle(1, 2, 3)
    comptime threads_per_row = _V_COLS // _V_WIDTH

    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_V_ROWS, _V_COLS]()
    )
    var local = stack_allocation[dtype=DType.float32, address_space=.LOCAL](
        row_major[1, _V_WIDTH]()
    )

    if thread_idx.x == 0:
        for row in range(_V_ROWS):
            for col in range(_V_COLS):
                smem[row, col] = -1.0
    barrier()

    var row = thread_idx.x // threads_per_row
    var col = (thread_idx.x % threads_per_row) * _V_WIDTH
    for j in range(_V_WIDTH):
        local[0, j] = src_ptr[row * _V_COLS + col + j]
    copy_local_to_shared[thread_layout, swizzle=swizzle](
        smem.vectorize[1, _V_WIDTH](), local.vectorize[1, _V_WIDTH]()
    )
    barrier()

    if thread_idx.x == 0:
        for i in range(_V_ROWS * _V_COLS):
            dst_ptr[i] = smem.reshape[_V_ROWS * _V_COLS]()[swizzle(i)]


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
    var dst_host = ctx.enqueue_create_host_buffer[.float32](num_elements)
    for i in range(num_elements):
        src_host[i] = Float32(i + 1)
        dst_host[i] = -1.0

    # Device memory is not cleared between tests, so start dst from a
    # sentinel rather than whatever the last test left there.
    var src_dev = ctx.enqueue_create_buffer[.float32](num_elements)
    var dst_dev = ctx.enqueue_create_buffer[.float32](num_elements)
    ctx.enqueue_copy(src_dev, src_host)
    ctx.enqueue_copy(dst_dev, dst_host)

    ctx.enqueue_function[kernel_fn](
        src_dev, dst_dev, grid_dim=(1), block_dim=(block_dim)
    )

    ctx.enqueue_copy(dst_host, dst_dev)
    ctx.synchronize()

    for i in range(num_elements):
        assert_equal(dst_host[i], src_host[i])


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


def test_swizzled_vectorized_local_to_shared(ctx: DeviceContext) raises:
    _run_roundtrip[
        swizzled_vectorized_local_to_shared_kernel,
        num_elements=_V_ROWS * _V_COLS,
        block_dim=_V_ROWS * _V_COLS // _V_WIDTH,
    ]("test_swizzled_vectorized_local_to_shared", ctx)


def test_async_dram_to_sram_to_dram(ctx: DeviceContext) raises:
    _run_roundtrip[async_dram_to_sram_to_dram_kernel](
        "test_async_dram_to_sram_to_dram", ctx
    )


def _bounded_local_to_dram_kernel[
    ROWS: Int,
    NUM_THREADS: Int = 32,
    SCOPE: ThreadScope = ThreadScope.BLOCK,
](
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
    after_ptr: MutPointer[Float32, MutAnyOrigin],
    valid0: Int32,
    valid1: Int32,
    row_stride: Int32,
):
    comptime thread_layout = row_major[8, 4]()
    var worker = thread_idx.x
    var warp = worker // 32
    var lane = worker % 32
    var owner = stack_allocation[.float32, address_space=.LOCAL, alignment=16](
        row_major[ROWS + 2]()
    )
    owner[0] = Float32(-7001)
    owner[ROWS + 1] = Float32(-7001)
    comptime for i in range(ROWS):
        owner[i + 1] = Float32(warp * 100000 + lane * 1000 + i)
    var local = owner.slice[1 : ROWS + 1]().reshape(row_major[ROWS // 4, 4]())
    var layout = TileLayout(
        Coord(Idx[ROWS], Idx[32]), Coord(row_stride, Idx[1])
    )
    var offset = 9 + (warp * (ROWS + 4) + 2) * Int(row_stride) + 32
    var dst = TileTensor[linear_idx_type=.int32](
        dst_ptr.unsafe_offset(offset), layout
    )
    var valid_rows = Int(valid0 if warp == 0 else valid1)
    copy_local_to_dram[
        thread_layout, num_threads=NUM_THREADS, thread_scope=SCOPE
    ](
        dst.vectorize[1, 2](),
        local.vectorize[1, 2]().transpose().as_imm(),
        valid_rows,
    )
    comptime for i in range(ROWS + 2):
        after_ptr[worker * (ROWS + 2) + i] = owner[i]


def _check_bounded_local_to_dram[
    ROWS: Int,
    NUM_THREADS: Int = 32,
    SCOPE: ThreadScope = ThreadScope.BLOCK,
](ctx: DeviceContext, valid0: Int, valid1: Int, stride: Int) raises:
    print(
        "== bounded_local_to_dram",
        ROWS,
        valid0,
        valid1,
        stride,
        NUM_THREADS,
        SCOPE,
    )
    comptime prefix = 9
    comptime suffix = 16
    var size = prefix + 2 * (ROWS + 4) * stride + suffix
    var host = ctx.enqueue_create_host_buffer[.float32](size)
    var device = ctx.enqueue_create_buffer[.float32](size)
    var after_host = ctx.enqueue_create_host_buffer[.float32](64 * (ROWS + 2))
    var after_device = ctx.enqueue_create_buffer[.float32](64 * (ROWS + 2))
    for i in range(size):
        host[i] = Float32(-9999)
    for i in range(64 * (ROWS + 2)):
        after_host[i] = Float32(-9999)
    assert_equal(Int(device.unsafe_ptr().unsafe_offset(prefix)) % 4, 0)
    for tile in range(2):
        var offset = prefix + (tile * (ROWS + 4) + 2) * stride + 32
        assert_equal(Int(device.unsafe_ptr().unsafe_offset(offset)) % 8, 4)
    ctx.enqueue_copy(device, host)
    ctx.enqueue_copy(after_device, after_host)
    ctx.enqueue_function[
        _bounded_local_to_dram_kernel[ROWS, NUM_THREADS, SCOPE]
    ](
        device,
        after_device,
        Int32(valid0),
        Int32(valid1),
        Int32(stride),
        grid_dim=1,
        block_dim=64,
    )
    ctx.enqueue_copy(host, device)
    ctx.enqueue_copy(after_host, after_device)
    ctx.synchronize()

    for i in range(size):
        var expected = Float32(-9999)
        var physical = i - prefix
        if physical >= 0 and physical < 2 * (ROWS + 4) * stride:
            var tile = physical // ((ROWS + 4) * stride)
            var row = (physical // stride) % (ROWS + 4) - 2
            var col = physical % stride - 32
            var bound = min(ROWS, max(0, valid0 if tile == 0 else valid1))
            var active: Bool
            comptime if NUM_THREADS > 32 and SCOPE == ThreadScope.BLOCK:
                active = tile == 0
            else:
                active = True
            if active and row >= 0 and row < bound and col >= 0 and col < 32:
                var lane = (row % 8) * 4 + (col % 8) // 2
                var mma: Int
                var fragment: Int
                comptime if ROWS == 16:
                    mma = col // 8
                    fragment = 2 * (row // 8) + col % 2
                else:
                    # Literal MMA row-group packing, not a distribution replay.
                    var group = row // 8
                    mma = 2 * (col // 8) + (1 if group >= 2 else 0)
                    fragment = (2 if group % 2 == 1 else 0) + col % 2
                expected = Float32(
                    tile * 100000 + lane * 1000 + 4 * mma + fragment
                )
        assert_equal(host[i], expected)
    for worker in range(64):
        for i in range(ROWS + 2):
            var expected = Float32(-7001)
            if i > 0 and i <= ROWS:
                expected = Float32(
                    (worker // 32) * 100000 + (worker % 32) * 1000 + i - 1
                )
            assert_equal(after_host[worker * (ROWS + 2) + i], expected)
    print(
        "PASS bounded_local_to_dram",
        ROWS,
        valid0,
        valid1,
        stride,
        NUM_THREADS,
        SCOPE,
    )


def test_bounded_local_to_dram(ctx: DeviceContext) raises:
    var bounds: List[Int] = [-1, 0, 1, 7, 8, 9, 16, 17, 31, 32, 33]
    for bound in bounds:
        _check_bounded_local_to_dram[32](
            ctx, bound, 32 - bound, 101 if bound % 2 != 0 else 104
        )
    _check_bounded_local_to_dram[32](ctx, 32, 32, 101)
    _check_bounded_local_to_dram[16](ctx, 16, 16, 101)
    _check_bounded_local_to_dram[16](ctx, 9, 7, 104)
    _check_bounded_local_to_dram[32, 64, ThreadScope.BLOCK](ctx, 32, 32, 101)
    _check_bounded_local_to_dram[32, 64, ThreadScope.WARP](ctx, 32, 32, 101)


def main() raises:
    with DeviceContext() as ctx:
        test_dram_to_sram_to_dram(ctx)
        test_dram_to_local_to_dram(ctx)
        test_sram_local_sram_roundtrip(ctx)
        test_swizzled_local_to_shared(ctx)
        test_swizzled_vectorized_sram_to_dram(ctx)
        test_swizzled_vectorized_local_to_shared(ctx)
        test_async_dram_to_sram_to_dram(ctx)
        comptime if ctx.default_device_info.warp_size == 32:
            test_bounded_local_to_dram(ctx)
