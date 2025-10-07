# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


from gpu import barrier
from gpu.host import DeviceContext, get_gpu_target
from gpu.host.compile import _compile_code
from gpu.id import thread_idx
from gpu.memory import AddressSpace, CacheOperation
from layout import *
from layout._fillers import arange
from layout._utils import load_to_simd
from layout.layout_tensor import (
    LayoutTensor,
    copy_dram_to_local,
    copy_dram_to_sram,
)
from sys import simd_width_of
from utils import IndexList
from benchmark import keep


fn copy_dram_to_sram_buffer_load_kernel[
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    thread_layout: Layout,
](input_ptr: UnsafePointer[Scalar[dtype]], m: Int,):
    alias layout = Layout.row_major(BM, BN)
    alias q_tile_type = LayoutTensor[
        dtype, layout, masked=True, address_space = AddressSpace.GLOBAL
    ]

    var runtime_layout = RuntimeLayout[
        layout,
        element_type = q_tile_type.layout_int_type,
        linear_idx_type = q_tile_type.linear_idx_type,
    ].row_major(IndexList[2, element_type = q_tile_type.layout_int_type](m, BN))

    var q_tile = q_tile_type(
        input_ptr.address_space_cast[AddressSpace.GLOBAL](),
        runtime_layout,
    )
    alias layout_bmn = Layout.row_major(BM, BN)
    var smem = LayoutTensor[
        dtype, layout_bmn, MutableAnyOrigin, address_space = AddressSpace.SHARED
    ].stack_allocation()
    if thread_idx.x == 0:
        _ = smem.fill(-1)
    barrier()

    var q_gmem_iter = q_tile.tiled_iterator[BM, BK, axis=1](0, 0)
    var smem_iter = smem.tiled_iterator[BM, BK, axis=1](0, 0)

    @parameter
    for i in range(BN // BK):
        var smem_tile = smem_iter.next_unsafe(i)[]
        copy_dram_to_sram[thread_layout=thread_layout](
            smem_tile.vectorize[1, 4](),
            q_gmem_iter,
            q_tile.runtime_layout.size(),
        )
        q_gmem_iter._incr()

    barrier()

    if thread_idx.x == 0:
        print("=== test_copy_dram_to_sram_buffer_load_tests")
        print(smem)


fn run_copy_dram_to_sram_buffer_load_tests(ctx: DeviceContext) raises:
    # CHECK: === test_copy_dram_to_sram_buffer_load_tests
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

    alias thread_layout = Layout.row_major(4, 2)
    alias layout = Layout.row_major(4, 16)
    var stack = InlineArray[BFloat16, layout.size()](uninitialized=True)
    var input_tensor = LayoutTensor[DType.bfloat16, layout](stack)
    arange(input_tensor)
    var device_tensor = ctx.enqueue_create_buffer[DType.bfloat16](
        input_tensor.layout.size()
    )
    ctx.enqueue_copy(device_tensor, input_tensor.ptr)
    alias kernel = copy_dram_to_sram_buffer_load_kernel[
        DType.bfloat16, 4, 16, 8, thread_layout
    ]
    ctx.enqueue_function_checked[kernel, kernel](
        device_tensor,
        3,
        grid_dim=1,
        block_dim=(thread_layout.size()),
    )
    ctx.synchronize()
    _ = device_tensor^


fn copy_dram_to_local_buffer_load_kernel[
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    thread_layout: Layout,
](input_ptr: UnsafePointer[Scalar[dtype]], m: Int,):
    alias layout = Layout.row_major(BM, BN)
    alias q_tile_type = LayoutTensor[
        dtype, layout, masked=True, address_space = AddressSpace.GLOBAL
    ]

    var runtime_layout = RuntimeLayout[
        layout,
        element_type = q_tile_type.layout_int_type,
        linear_idx_type = q_tile_type.linear_idx_type,
    ].row_major(IndexList[2, element_type = q_tile_type.layout_int_type](m, BN))

    var q_tile = q_tile_type(
        input_ptr.address_space_cast[AddressSpace.GLOBAL](),
        runtime_layout,
    )

    barrier()

    var q_gmem_iter = q_tile.tiled_iterator[BM, BK, axis=1](0, 0)

    alias a_reg_layout = Layout.row_major(
        (BM * BN) // thread_layout.size() // 2, 2
    )
    var a_reg_tile = LayoutTensor[
        dtype,
        a_reg_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    @parameter
    for i in range(BN // BK):
        copy_dram_to_local[src_thread_layout=thread_layout](
            a_reg_tile.tile[a_reg_tile.shape[0]() // (BN // BK), 2](
                i, 0
            ).vectorize[1, 2](),
            q_gmem_iter[].vectorize[1, 2](),
            q_tile,
        )
        q_gmem_iter._incr()

    barrier()
    if thread_idx.x == 0:
        print("=== test_copy_dram_to_local_buffer_load_tests")
    print("tid =", thread_idx.x, "reg =", load_to_simd(a_reg_tile))


fn run_copy_dram_to_local_buffer_load_tests(ctx: DeviceContext) raises:
    # CHECK: === test_copy_dram_to_local_buffer_load_tests
    # CHECK: tid = 0 reg = [0.0, 1.0, 8.0, 9.0]
    # CHECK: tid = 1 reg = [2.0, 3.0, 10.0, 11.0]
    # CHECK: tid = 2 reg = [4.0, 5.0, 12.0, 13.0]
    # CHECK: tid = 3 reg = [6.0, 7.0, 14.0, 15.0]
    # CHECK: tid = 4 reg = [16.0, 17.0, 24.0, 25.0]
    # CHECK: tid = 5 reg = [18.0, 19.0, 26.0, 27.0]
    # CHECK: tid = 6 reg = [20.0, 21.0, 28.0, 29.0]
    # CHECK: tid = 7 reg = [22.0, 23.0, 30.0, 31.0]
    # CHECK: tid = 8 reg = [32.0, 33.0, 40.0, 41.0]
    # CHECK: tid = 9 reg = [34.0, 35.0, 42.0, 43.0]
    # CHECK: tid = 10 reg = [36.0, 37.0, 44.0, 45.0]
    # CHECK: tid = 11 reg = [38.0, 39.0, 46.0, 47.0]
    # CHECK: tid = 12 reg = [0.0, 0.0, 0.0, 0.0]
    # CHECK: tid = 13 reg = [0.0, 0.0, 0.0, 0.0]
    # CHECK: tid = 14 reg = [0.0, 0.0, 0.0, 0.0]
    # CHECK: tid = 15 reg = [0.0, 0.0, 0.0, 0.0]
    alias thread_layout = Layout.row_major(4, 4)
    alias input_layout = Layout.row_major(4, 16)
    var input_stack = InlineArray[BFloat16, input_layout.size()](
        uninitialized=True
    )
    var input_tensor = LayoutTensor[DType.bfloat16, input_layout](input_stack)
    arange(input_tensor)
    var device_tensor = ctx.enqueue_create_buffer[DType.bfloat16](
        input_tensor.layout.size()
    )
    ctx.enqueue_copy(device_tensor, input_tensor.ptr)
    alias kernel = copy_dram_to_local_buffer_load_kernel[
        DType.bfloat16, 4, 16, 8, thread_layout
    ]
    ctx.enqueue_function_checked[kernel, kernel](
        device_tensor,
        3,
        grid_dim=1,
        block_dim=(thread_layout.size()),
    )
    ctx.synchronize()
    _ = device_tensor^


fn test_codegen_copy_dram_to_local(ctx: DeviceContext) raises:
    fn kernel[cache_policy: CacheOperation](ptr: UnsafePointer[BFloat16]):
        alias simd_width = simd_width_of[DType.bfloat16]()
        var global_tensor = LayoutTensor[
            DType.bfloat16,
            Layout.row_major(16, 128),
        ](ptr)
        var local_tensor = LayoutTensor[
            DType.bfloat16,
            Layout.row_major(16, 8),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        alias thread_layout = Layout.row_major(16, 16)
        copy_dram_to_local[
            src_thread_layout=thread_layout,
            cache_policy=cache_policy,
        ](
            local_tensor.vectorize[1, simd_width](),
            global_tensor.vectorize[1, simd_width](),
            global_tensor,
        )
        keep(local_tensor)

    alias MI355X_TARGET = get_gpu_target["mi355x"]()
    # CHECK: === test_codegen_copy_dram_to_local_streaming_workgroup
    # CHECK: buffer_load_dwordx4 v[0:3], v0, s[0:3], 0 offen sc0 nt
    print("=== test_codegen_copy_dram_to_local_streaming_workgroup")
    print(
        _compile_code[
            kernel[CacheOperation.STREAMING | CacheOperation.WORKGROUP],
            target=MI355X_TARGET,
        ]()
    )


def main():
    with DeviceContext() as ctx:
        run_copy_dram_to_sram_buffer_load_tests(ctx)
        run_copy_dram_to_local_buffer_load_tests(ctx)
        test_codegen_copy_dram_to_local(ctx)
