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
"""Tests for `TileTensor.copy_from_async`.

Each kernel drives the DRAM -> SMEM leg through `copy_from_async` directly
(rather than through the `tile_io` wrappers) and reads the tile back out with
the synchronous `copy_sram_to_dram`, so a roundtrip mismatch isolates the
async primitive.

Coverage: unswizzled, swizzled, vectorized (16-byte elements), and masked
copies under each `Fill` policy. On AMD and Apple GPUs the underlying
`async_copy` intrinsic falls back to synchronous loads and stores; the
commit/wait calls stay valid.
"""

from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import (
    Fill,
    async_copy_commit_group,
    async_copy_wait_all,
)
from max.gpu.sync import barrier

from layout import Idx, TileTensor, row_major
from layout.swizzle import Swizzle
from layout.tile_io import copy_sram_to_dram
from layout.tile_tensor import stack_allocation

from std.math import isnan, nan

from std.testing import assert_equal, assert_true


# 4x4 tile distributed over 4 threads: each thread owns a 2x2 fragment.
comptime _N = 4
comptime _NUM_ELEMENTS = _N * _N
comptime _BLOCK_DIM = 4

# Pre-written into the destination tile before a masked copy, so a fill policy
# that must not write shows up as the sentinel surviving.
comptime _SENTINEL = Float32(-7)


def async_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    var tid = thread_idx.x
    smem.distribute[thread_layout](tid).copy_from_async(
        src.distribute[thread_layout](tid)
    )
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()

    copy_sram_to_dram[thread_layout](dst, smem)


def async_swizzled_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """The SMEM -> DRAM read must undo the same swizzle the async write
    applied, so a roundtrip match proves the swizzled destination indices
    agree between the two paths."""
    comptime thread_layout = row_major(Idx[2], Idx[2])
    comptime swizzle = Swizzle(1, 0, 2)

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    var tid = thread_idx.x
    var smem_frag = smem.distribute[thread_layout](tid)
    smem_frag.copy_from_async[swizzle=swizzle](
        src.distribute[thread_layout](tid),
        base_offset=smem_frag._distance(smem),
    )
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()

    copy_sram_to_dram[thread_layout, swizzle=swizzle](dst, smem)


def async_vectorized_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Each thread copies one 16-byte (4 x float32) vector."""
    comptime thread_layout = row_major(Idx[4], Idx[1])

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    var tid = thread_idx.x
    smem.vectorize[1, 4]().distribute[thread_layout](tid).copy_from_async(
        src.vectorize[1, 4]().distribute[thread_layout](tid)
    )
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()

    copy_sram_to_dram[row_major(Idx[2], Idx[2])](dst, smem)


def _async_masked_kernel[
    fill: Fill
](
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """Bound the copy to the first two rows and let `fill` decide what happens
    to the rest.

    The destination is pre-written with `_SENTINEL`, so the rows past the bound
    distinguish the policies: `Fill.NONE` leaves the sentinel in place,
    `Fill.ZERO` overwrites it with zeros, and `Fill.NAN` with NaN.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])
    comptime valid_rows = 2

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    var tid = thread_idx.x
    var src_frag = src.distribute[thread_layout](tid)
    var smem_frag = smem.distribute[thread_layout](tid)

    # The barrier orders the sentinel stores ahead of the `cp.async` writes to
    # the same addresses; without it the copy could land first and the
    # `Fill.NONE` case would read back a sentinel it was supposed to keep.
    _ = smem_frag.fill(_SENTINEL)
    barrier()

    smem_frag.copy_from_async[is_masked=True, fill=fill](
        src_frag,
        src_idx_bound=Scalar[src_frag.linear_idx_type](
            valid_rows * _N - Int(src_frag._distance(src))
        ),
    )
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()

    copy_sram_to_dram[thread_layout](dst, smem)


def async_masked_no_fill_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    _async_masked_kernel[Fill.NONE](src_ptr, dst_ptr)


def async_masked_zero_fill_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    _async_masked_kernel[Fill.ZERO](src_ptr, dst_ptr)


def async_masked_nan_fill_kernel(
    src_ptr: MutPointer[Float32, MutAnyOrigin],
    dst_ptr: MutPointer[Float32, MutAnyOrigin],
):
    """`Fill.NAN` needs 16-byte elements, so each thread carries one
    4 x float32 vector rather than a 2x2 fragment."""
    comptime thread_layout = row_major(Idx[4], Idx[1])
    comptime valid_rows = 2

    var src = TileTensor(src_ptr, row_major[_N, _N]())
    var dst = TileTensor(dst_ptr, row_major[_N, _N]())
    var smem = stack_allocation[dtype=DType.float32, address_space=.SHARED](
        row_major[_N, _N]()
    )

    var tid = thread_idx.x
    var src_frag = src.vectorize[1, 4]().distribute[thread_layout](tid)
    var smem_frag = smem.vectorize[1, 4]().distribute[thread_layout](tid)

    _ = smem_frag.fill(_SENTINEL)
    barrier()

    smem_frag.copy_from_async[is_masked=True, fill=Fill.NAN](
        src_frag,
        src_idx_bound=Scalar[src_frag.linear_idx_type](
            valid_rows * _N - Int(src_frag._distance(src))
        ),
    )
    async_copy_commit_group()
    async_copy_wait_all()
    barrier()

    copy_sram_to_dram[row_major(Idx[2], Idx[2])](dst, smem)


def _run[
    kernel_fn: def(
        MutPointer[Float32, MutAnyOrigin],
        MutPointer[Float32, MutAnyOrigin],
    ) thin -> None,
](
    name: String,
    ctx: DeviceContext,
    valid_rows: Int = _N,
    masked_fill: Float32 = 0,
) raises:
    print("==", name)

    var src_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    for i in range(_NUM_ELEMENTS):
        src_host[i] = Float32(i + 1)

    var src_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    var dst_dev = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(src_dev, src_host)

    ctx.enqueue_function[kernel_fn](
        src_dev, dst_dev, grid_dim=(1), block_dim=(_BLOCK_DIM)
    )

    var dst_host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(dst_host, dst_dev)
    ctx.synchronize()

    for i in range(_NUM_ELEMENTS):
        var expected = src_host[i] if i < valid_rows * _N else masked_fill
        if isnan(expected):
            assert_true(isnan(dst_host[i]))
        else:
            assert_equal(dst_host[i], expected)


def main() raises:
    with DeviceContext() as ctx:
        _run[async_kernel]("test_async", ctx)
        _run[async_swizzled_kernel]("test_async_swizzled", ctx)
        _run[async_vectorized_kernel]("test_async_vectorized", ctx)
        _run[async_masked_no_fill_kernel](
            "test_async_masked_no_fill",
            ctx,
            valid_rows=2,
            masked_fill=_SENTINEL,
        )
        _run[async_masked_zero_fill_kernel](
            "test_async_masked_zero_fill", ctx, valid_rows=2, masked_fill=0
        )
        _run[async_masked_nan_fill_kernel](
            "test_async_masked_nan_fill",
            ctx,
            valid_rows=2,
            masked_fill=nan[DType.float32](),
        )
