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

from std.math import ceildiv
from std.bit import log2_floor
from std.sys import simd_width_of

from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu import block_idx, thread_idx
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_all,
)
from layout import *
from layout._fillers import arange
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.swizzle import make_ldmatrix_swizzle
from layout.layout_tensor import (
    copy_dram_to_sram,
)

from layout.tile_io import copy_dram_to_sram_async, copy_sram_to_dram

from std.utils import IndexList

# ----------------------------------------------------------------------
# dynamic async copy tests
# ----------------------------------------------------------------------


def print_tile_tensor(tensor: TileTensor):
    comptime assert tensor.flat_rank == 2
    for i in range(tensor.dim[0]()):
        for j in range(tensor.dim[1]()):
            print(tensor[i, j], end=" ")
        print()


def async_dynamic_copy_kernel[
    input_layout: Layout,
    output_layout: Layout,
    BM: Int,
    BN: Int,
    num_rows: Int,
](
    input: LayoutTensor[.float32, input_layout, MutAnyOrigin],
    output: LayoutTensor[.float32, output_layout, MutAnyOrigin],
):
    var masked_input = LayoutTensor[
        .float32,
        input_layout,
        MutAnyOrigin,
        masked=True,
    ](
        input.ptr,
        type_of(input.runtime_layout)(
            type_of(input.runtime_layout.shape)(num_rows, input.dim[1]()),
            input.runtime_layout.stride,
        ),
    )

    var input_tile = masked_input.tile[BM, BN](block_idx.x, block_idx.y)
    var output_tile = output.tile[BM, BN](block_idx.x, block_idx.y)

    var smem_tile = LayoutTensor[
        .float32,
        Layout(IntTuple(BM, BN)),
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()

    smem_tile.copy_from_async[is_masked=True](input_tile)
    async_copy_wait_all()

    output_tile.copy_from(smem_tile)


def test_dynamic_async_copy[
    M: Int, N: Int, BM: Int, BN: Int, num_rows: Int
](ctx: DeviceContext) raises:
    print("=== test_dynamic_async_copy")

    comptime input_layout = row_major((Int64(M), Int64(N)))
    comptime output_layout = row_major((Int64(num_rows), Int64(N)))
    var input = HostDeviceTileTensor[.float32](input_layout, ctx)
    arange(input.host_tensor())
    var output = HostDeviceTileTensor[.float32](output_layout, ctx)
    input.to_device()
    var input_view = (
        input.device_tensor().as_unsafe_any_origin().to_layout_tensor()
    )
    var output_view = (
        output.device_tensor().as_unsafe_any_origin().to_layout_tensor()
    )

    comptime kernel_type = async_dynamic_copy_kernel[
        input_view.layout,
        output_view.layout,
        BM,
        BN,
        num_rows,
    ]
    ctx.enqueue_function[kernel_type](
        input_view,
        output_view,
        grid_dim=(ceildiv(M, BM), ceildiv(M, BN)),
        block_dim=(1, 1),
    )
    output.to_host()
    print_tile_tensor(output.host_tensor())


def run_dynamic_async_copy_tests(ctx: DeviceContext) raises:
    # CHECK: === test_dynamic_async_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0 16.0 17.0
    # CHECK: 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0
    test_dynamic_async_copy[
        M=6,
        N=6,
        BM=2,
        BN=3,
        num_rows=5,
    ](ctx)


# ----------------------------------------------------------------------
# swizzle copy tests
# ----------------------------------------------------------------------


def swizzle_copy[
    layout: TensorLayout, BM: Int, BK: Int, num_threads: Int
](
    a: TileTensor[.float32, layout, ImmutAnyOrigin],
    b: TileTensor[.float32, layout, MutAnyOrigin],
):
    comptime assert a.flat_rank == b.flat_rank == 2
    comptime simd_size = simd_width_of[DType.float32]()
    var a_smem_tile = stack_allocation[
        .float32, address_space=.SHARED, alignment=16
    ](row_major[BM, BK]()).fill(0)
    comptime thread_layout = row_major[
        num_threads * simd_size // BK, BK // simd_size
    ]()
    comptime swizzle = make_ldmatrix_swizzle[
        .float32, BK, log2_floor(simd_size)
    ]()
    var valid_rows = max(0, min(BM, Int(a.dim[0]()) - block_idx.x * BM))
    copy_dram_to_sram_async[
        thread_layout=thread_layout, swizzle=swizzle, masked=True
    ](
        a_smem_tile.vectorize[1, simd_size](),
        a.tile[BM, BK](block_idx.x, 0).vectorize[1, simd_size](),
        valid_rows,
    )
    async_copy_wait_all()
    barrier()

    # Read without unswizzling so the oracle checks the physical permutation.
    var b_gmem_frag = (
        b.tile[BM, BK](block_idx.x, 0)
        .vectorize[1, simd_size]()
        .distribute[thread_layout](thread_idx.x)
    )
    var a_smem_frag = a_smem_tile.vectorize[1, simd_size]().distribute[
        thread_layout
    ](thread_idx.x)
    b_gmem_frag.copy_from(a_smem_frag)


def test_swizzle_copy[
    M: Int, K: Int, BM: Int, BK: Int, num_threads: Int, skew_M: Int = 0
](ctx: DeviceContext) raises:
    print("=== test_swizzle_copy")
    var a_tensor = HostDeviceTileTensor[.float32](
        row_major((M - skew_M, Idx[K])), ctx
    )
    var b_tensor = HostDeviceTileTensor[.float32](row_major((M, Idx[K])), ctx)
    arange(a_tensor.host_tensor())
    a_tensor.to_device()

    comptime copy = swizzle_copy[
        type_of(row_major((M, Idx[K]))), BM, BK, num_threads
    ]
    ctx.enqueue_function[copy](
        a_tensor.device_tensor().as_imm().as_unsafe_any_origin(),
        b_tensor.device_tensor().as_unsafe_any_origin(),
        grid_dim=(ceildiv(M, BM), 1, 1),
        block_dim=(num_threads, 1, 1),
    )
    b_tensor.to_host()
    print_tile_tensor(b_tensor.host_tensor())


def run_swizzle_copy_tests(ctx: DeviceContext) raises:
    # CHECK: === test_swizzle_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 36.0 37.0 38.0 39.0 32.0 33.0 34.0 35.0 44.0 45.0 46.0 47.0 40.0 41.0 42.0 43.0
    # CHECK: 52.0 53.0 54.0 55.0 48.0 49.0 50.0 51.0 60.0 61.0 62.0 63.0 56.0 57.0 58.0 59.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0
    # CHECK: 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_swizzle_copy[
        M=8,
        K=16,
        BM=8,
        BK=16,
        num_threads=32,
        skew_M=2,
    ](ctx)


# ----------------------------------------------------------------------
# masked async copy tests
# ----------------------------------------------------------------------


@inline(.always)
def masked_async_copy_kernel[
    M: Int, N: Int, num_rows: Int
](input: TileTensor[.float32, type_of(row_major[M, N]()), MutAnyOrigin],):
    comptime thread_layout = row_major[4, 2]()
    var smem_tile = stack_allocation[
        .float32, address_space=.SHARED, alignment=16
    ](row_major[M, N]()).fill(-1.0)
    copy_dram_to_sram_async[thread_layout=thread_layout, masked=True](
        smem_tile.vectorize[1, 4](),
        input.as_imm().vectorize[1, 4](),
        num_rows,
    )
    async_copy_commit_group()
    async_copy_wait_all()
    copy_sram_to_dram[thread_layout=thread_layout](
        input.vectorize[1, 4](), smem_tile.vectorize[1, 4]()
    )


def test_masked_async_copy[
    layout: MixedLayout, M: Int, N: Int, skew_rows: Int
](ctx: DeviceContext) raises:
    print("=== test_masked_async_copy")
    var input = HostDeviceTileTensor[.float32](layout, ctx)
    arange(input.host_tensor())
    input.to_device()
    var input_tensor = input.device_tensor().reshape(row_major[M, N]())
    comptime kernel_type = masked_async_copy_kernel[M, N, M - skew_rows]
    ctx.enqueue_function[kernel_type](
        input_tensor.as_unsafe_any_origin(), grid_dim=(1,), block_dim=(8,)
    )
    input.to_host()
    print_tile_tensor(input.host_tensor())


def run_masked_async_copy_tests(ctx: DeviceContext) raises:
    # CHECK: === test_masked_async_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_masked_async_copy[
        row_major[8, 8](),
        M=8,
        N=8,
        skew_rows=1,
    ](ctx)

    # CHECK: === test_masked_async_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_masked_async_copy[
        row_major((8, 8)),
        M=8,
        N=8,
        skew_rows=1,
    ](ctx)


# ----------------------------------------------------------------------
# masked copy tests
# ----------------------------------------------------------------------


@inline(.always)
def masked_copy_kernel[
    layout: TensorLayout, num_rows: Int
](input: TileTensor[.float32, layout, MutAnyOrigin]):
    comptime thread_layout = Layout.row_major(4, 2)

    var input_legacy = input.to_layout_tensor()
    var masked_input = LayoutTensor[
        .float32,
        input_legacy.layout,
        MutAnyOrigin,
        masked=True,
    ](
        input_legacy.ptr,
        type_of(input_legacy.runtime_layout)(
            type_of(input_legacy.runtime_layout.shape)(
                num_rows, Int(input.dim[1]())
            ),
            input_legacy.runtime_layout.stride,
        ),
    )

    var smem_tile = stack_allocation[.float32, address_space=.SHARED](
        row_major[input.static_shape[0], input.static_shape[1]]()
    ).fill(0)

    copy_dram_to_sram[thread_layout=thread_layout](
        smem_tile.to_layout_tensor().vectorize[1, 4](),
        masked_input.vectorize[1, 4](),
    )

    barrier()

    copy_sram_to_dram[thread_layout=row_major[4, 2]()](
        input.vectorize[1, 4](),
        smem_tile.vectorize[1, 4](),
    )


def test_masked_copy[
    layout: MixedLayout, M: Int, N: Int, skew_rows: Int
](ctx: DeviceContext) raises:
    print("=== test_masked_copy")
    var input = HostDeviceTileTensor[.float32](layout, ctx)
    arange(input.host_tensor())
    input.to_device()
    var input_tensor = input.device_tensor().reshape(row_major[M, N]())
    comptime kernel_type = masked_copy_kernel[
        input_tensor.LayoutType, M - skew_rows
    ]
    ctx.enqueue_function[kernel_type](
        input_tensor.as_unsafe_any_origin(), grid_dim=(1,), block_dim=(8,)
    )
    input.to_host()
    print_tile_tensor(input.host_tensor())


def run_masked_copy_tests(ctx: DeviceContext) raises:
    # CHECK: === test_masked_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_masked_copy[
        row_major[8, 8](),
        M=8,
        N=8,
        skew_rows=1,
    ](ctx)

    # CHECK: === test_masked_copy
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    test_masked_copy[
        row_major((8, 8)),
        M=8,
        N=8,
        skew_rows=1,
    ](ctx)


def main() raises:
    with DeviceContext() as ctx:
        run_dynamic_async_copy_tests(ctx)
        run_swizzle_copy_tests(ctx)
        run_masked_async_copy_tests(ctx)
        run_masked_copy_tests(ctx)
