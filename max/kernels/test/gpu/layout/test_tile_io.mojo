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
"""Roundtrip tests for the `TileCopier` structs in tile_io.mojo.

Each kernel moves a source tile through one or more intermediate address
spaces (SHARED, LOCAL) and back out to a destination DRAM tile, using
only `TileTensor` and the concrete `TileCopier` structs (no
`LayoutTensor` / `ManagedLayoutTensor`). The host verifies that the
destination tile matches the source.

Coverage:

- GENERIC -> SHARED -> GENERIC (unswizzled)
- GENERIC -> LOCAL -> GENERIC (unswizzled)
- GENERIC -> SHARED -> LOCAL -> SHARED -> GENERIC (exercises
  SharedToLocal and LocalToShared unswizzled paths)
- GENERIC -> LOCAL -> SHARED -> GENERIC with swizzle (exercises the
  correctness-critical swizzled paths of LocalToShared and
  SharedToGeneric, which must use identical swizzled addresses for
  writes and reads to round-trip).
"""

from std.gpu import barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace

from layout import Idx, TileTensor, row_major
from layout.swizzle import Swizzle
from layout.tile_io import (
    GenericToLocalTileCopier,
    GenericToSharedTileCopier,
    LocalToGenericTileCopier,
    LocalToSharedTileCopier,
    SharedToGenericTileCopier,
    SharedToLocalTileCopier,
)
from layout.tile_tensor import stack_allocation

from std.testing import assert_equal


# 4x4 tile, distributed over 4 threads (2x2 thread layout -> each thread
# owns a 2x2 fragment).
comptime _N = 4
comptime _NUM_ELEMENTS = _N * _N
comptime _BLOCK_DIM = 4


def generic_to_shared_to_generic_kernel(
    src_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
):
    """Roundtrip a tile through shared memory: GENERIC -> SHARED -> GENERIC."""
    comptime thread_layout = row_major(Idx[2](), Idx[2]())

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[_N, _N]())

    GenericToSharedTileCopier[thread_layout]().copy(smem, src)
    barrier()
    SharedToGenericTileCopier[thread_layout]().copy(dst, smem)


def generic_to_local_to_generic_kernel(
    src_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
):
    """Roundtrip a tile through registers: GENERIC -> LOCAL -> GENERIC.

    Each thread holds its own 2x2 fragment in local memory.
    """
    comptime thread_layout = row_major(Idx[2](), Idx[2]())

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    # Per-thread fragment of a 4x4 tile under a 2x2 thread layout.
    var local = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.LOCAL
    ](row_major[2, 2]())

    GenericToLocalTileCopier[thread_layout]().copy(local, src)
    LocalToGenericTileCopier[thread_layout]().copy(dst, local)


def shared_local_shared_kernel(
    src_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
):
    """Roundtrip a tile through shared -> local -> shared -> generic.

    GENERIC -> SHARED loads src into `smem_in`.
    SHARED -> LOCAL distributes `smem_in` into per-thread fragments.
    LOCAL -> SHARED writes those fragments into a second `smem_out`.
    SHARED -> GENERIC reads `smem_out` back to `dst`.
    """
    comptime thread_layout = row_major(Idx[2](), Idx[2]())

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())

    var smem_in = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[_N, _N]())
    var smem_out = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[_N, _N]())
    var local = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.LOCAL
    ](row_major[2, 2]())

    GenericToSharedTileCopier[thread_layout]().copy(smem_in, src)
    barrier()
    SharedToLocalTileCopier[thread_layout]().copy(local, smem_in)
    LocalToSharedTileCopier[thread_layout]().copy(smem_out, local)
    barrier()
    SharedToGenericTileCopier[thread_layout]().copy(dst, smem_out)


def swizzled_local_to_shared_to_generic_kernel(
    src_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    dst_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
):
    """Swizzled roundtrip: GENERIC -> LOCAL -> SHARED (swizzled) ->
    GENERIC (swizzled).

    Exercises the correctness-critical swizzled paths in LocalToShared
    and SharedToGeneric. A `None` swizzle on the SHARED -> GENERIC read
    would produce garbage (that copier's docstring flags this), so this
    test is the guard that the two swizzled branches compute identical
    swizzled addresses per thread / per index.
    """
    comptime thread_layout = row_major(Idx[2](), Idx[2]())
    comptime swizzle = Swizzle(1, 0, 2)

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())

    var smem = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.SHARED
    ](row_major[_N, _N]())
    var local = stack_allocation[
        dtype=DType.float32, address_space=AddressSpace.LOCAL
    ](row_major[2, 2]())

    GenericToLocalTileCopier[thread_layout]().copy(local, src)
    LocalToSharedTileCopier[thread_layout, swizzle=swizzle]().copy(smem, local)
    barrier()
    SharedToGenericTileCopier[thread_layout, swizzle=swizzle]().copy(dst, smem)


def _run_roundtrip[
    kernel_fn: def(
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ) thin -> None,
](name: String, ctx: DeviceContext) raises:
    print("==", name)

    var src_host = ctx.enqueue_create_host_buffer[DType.float32](_NUM_ELEMENTS)
    for i in range(_NUM_ELEMENTS):
        src_host[i] = Float32(i + 1)

    var src_dev = ctx.enqueue_create_buffer[DType.float32](_NUM_ELEMENTS)
    var dst_dev = ctx.enqueue_create_buffer[DType.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(src_dev, src_host)

    ctx.enqueue_function_experimental[kernel_fn](
        src_dev, dst_dev, grid_dim=(1), block_dim=(_BLOCK_DIM)
    )

    var dst_host = ctx.enqueue_create_host_buffer[DType.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(dst_host, dst_dev)
    ctx.synchronize()

    for i in range(_NUM_ELEMENTS):
        assert_equal(dst_host[i], src_host[i])


def test_generic_to_shared_to_generic(ctx: DeviceContext) raises:
    _run_roundtrip[generic_to_shared_to_generic_kernel](
        "test_generic_to_shared_to_generic", ctx
    )


def test_generic_to_local_to_generic(ctx: DeviceContext) raises:
    _run_roundtrip[generic_to_local_to_generic_kernel](
        "test_generic_to_local_to_generic", ctx
    )


def test_shared_local_shared_roundtrip(ctx: DeviceContext) raises:
    _run_roundtrip[shared_local_shared_kernel](
        "test_shared_local_shared_roundtrip", ctx
    )


def test_swizzled_local_to_shared_to_generic(ctx: DeviceContext) raises:
    _run_roundtrip[swizzled_local_to_shared_to_generic_kernel](
        "test_swizzled_local_to_shared_to_generic", ctx
    )


def main() raises:
    with DeviceContext() as ctx:
        test_generic_to_shared_to_generic(ctx)
        test_generic_to_local_to_generic(ctx)
        test_shared_local_shared_roundtrip(ctx)
        test_swizzled_local_to_shared_to_generic(ctx)
