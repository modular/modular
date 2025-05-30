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

from builtin.io import _printf
from gpu import WARP_SIZE, barrier, lane_id
from gpu.host import DeviceContext
from gpu.id import thread_idx
from gpu.memory import _GPUAddressSpace as AddressSpace
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor, load_to_simd
from layout.layout_tensor import copy_dram_to_sram
from layout.tensor_core import TensorCore

from utils.index import Index, IndexList


fn mma_load_and_multiply[
    dst_dtype: DType,
    dtype: DType,
    lhs_layout: Layout,
    rhs_layout: Layout,
    inst_shape: IndexList[3],
    transpose_b: Bool = False,
](
    lhs: LayoutTensor[dtype, lhs_layout, MutableAnyOrigin],
    rhs: LayoutTensor[dtype, rhs_layout, MutableAnyOrigin],
):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, transpose_b]()
    var a_reg_tile = mma.load_a(lhs)
    var a_frags = load_to_simd(a_reg_tile).cast[DType.float64]()
    var b_reg_tile = mma.load_b(rhs)
    var b_frags = load_to_simd(b_reg_tile).cast[DType.float64]()

    var c_reg_tile = mma.c_reg_tile_type.stack_allocation().fill(1.0)
    var d_reg_tile = mma.mma_op(a_reg_tile, b_reg_tile, c_reg_tile)
    var d_frags = load_to_simd(d_reg_tile).cast[DType.float64]()

    @parameter
    # NVIDIA
    if a_frags.size == 8 and b_frags.size == 4:
        _printf[
            "thread %u a_vals=[%g %g %g %g %g %g %g %g], b_vals=[%g %g %g %g],"
            " d_vals=[%g %g %g %g]\n"
        ](
            thread_idx.x,
            a_frags[0],
            a_frags[1],
            a_frags[2],
            a_frags[3],
            a_frags[4],
            a_frags[5],
            a_frags[6],
            a_frags[7],
            b_frags[0],
            b_frags[1],
            b_frags[2],
            b_frags[3],
            d_frags[0],
            d_frags[1],
            d_frags[2],
            d_frags[3],
        )
    elif a_frags.size == 4 and b_frags.size == 2:
        _printf[
            "thread %u a_vals=[%g %g %g %g], b_vals=[%g %g], d_vals=[%g %g %g"
            " %g]\n"
        ](
            thread_idx.x,
            a_frags[0],
            a_frags[1],
            a_frags[2],
            a_frags[3],
            b_frags[0],
            b_frags[1],
            d_frags[0],
            d_frags[1],
            d_frags[2],
            d_frags[3],
        )
    elif a_frags.size == 2 and b_frags.size == 1:
        _printf[
            "thread %u a_vals=[%g %g], b_vals=[%g], d_vals=[%g %g %g %g]\n"
        ](
            thread_idx.x,
            a_frags[0],
            a_frags[1],
            b_frags[0],
            d_frags[0],
            d_frags[1],
            d_frags[2],
            d_frags[3],
        )
    # AMD-MI300
    elif a_frags.size == 4 and b_frags.size == 4:
        _printf[
            "thread %u a_vals=[%g %g %g %g], b_vals=[%g %g %g %g], d_vals=[%g"
            " %g %g %g]\n"
        ](
            thread_idx.x,
            a_frags[0],
            a_frags[1],
            a_frags[2],
            a_frags[3],
            b_frags[0],
            b_frags[1],
            b_frags[2],
            b_frags[3],
            d_frags[0],
            d_frags[1],
            d_frags[2],
            d_frags[3],
        )
    elif a_frags.size == 1 and b_frags.size == 1:
        _printf["thread %u a_vals=[%g], b_vals=[%g], d_vals=[%g %g %g %g]\n"](
            thread_idx.x,
            a_frags[0],
            b_frags[0],
            d_frags[0],
            d_frags[1],
            d_frags[2],
            d_frags[3],
        )

    _ = c_reg_tile


fn mma_write_operand_kernel[
    dst_dtype: DType,
    dtype: DType,
    layout: Layout,
    inst_shape: IndexList[3],
](out: LayoutTensor[dst_dtype, layout, MutableAnyOrigin]):
    var mma = TensorCore[dst_dtype, dtype, inst_shape]()
    var thread_reg_tile = mma.c_reg_tile_type.stack_allocation()
    var thread_reg_tile_v = thread_reg_tile.vectorize[1, mma.c_reg_type.size]()
    thread_reg_tile_v[0, 0] = rebind[__type_of(thread_reg_tile_v[0, 0])](
        mma.c_reg_type(thread_idx.x)
    )
    mma.store_d(out, thread_reg_tile)


def test_load_and_mma_and_multiply_operands[
    dst_dtype: DType,
    dtype: DType,
    shape: IndexList[3],
    transpose_b: Bool = False,
](ctx: DeviceContext):
    alias M = shape[0]
    alias N = shape[1]
    alias K = shape[2]

    var lhs = ManagedLayoutTensor[dtype, Layout.row_major(M, K)](ctx)
    arange(lhs.tensor())
    alias rhs_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var rhs = ManagedLayoutTensor[dtype, rhs_layout](ctx)
    arange(rhs.tensor())
    alias mma_load_and_print_kernel_fn = mma_load_and_multiply[
        dst_dtype, dtype, lhs.layout, rhs.layout, shape, transpose_b
    ]

    ctx.enqueue_function[mma_load_and_print_kernel_fn](
        lhs.device_tensor(),
        rhs.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(WARP_SIZE),
    )
    ctx.synchronize()

    _ = lhs^
    _ = rhs^


def test_write_res_operand[
    dst_dtype: DType, dtype: DType, shape: IndexList[3]
](ctx: DeviceContext):
    alias M = shape[0]
    alias N = shape[1]
    alias K = shape[2]

    var dst = ManagedLayoutTensor[dst_dtype, Layout.row_major(M, N)](ctx)
    _ = dst.tensor().fill(0)
    alias mma_load_and_print_kernel_fn = mma_write_operand_kernel[
        dst_dtype, dtype, dst.layout, shape
    ]
    ctx.enqueue_function[mma_load_and_print_kernel_fn](
        dst.device_tensor(), grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )
    ctx.synchronize()

    print(dst.tensor())

    _ = dst^


fn mma_load_and_print_operands_kernel_ldmatrix[
    dst_dtype: DType,
    dtype: DType,
    lhs_layout: Layout,
    rhs_layout: Layout,
    inst_shape: IndexList[3],
    transpose_b: Bool = False,
](
    lhs: LayoutTensor[dtype, lhs_layout, MutableAnyOrigin],
    rhs: LayoutTensor[dtype, rhs_layout, MutableAnyOrigin],
):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, transpose_b]()
    var a_smem = LayoutTensor[
        dtype,
        lhs.layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var b_smem = LayoutTensor[
        dtype,
        rhs.layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    alias thread_layout = Layout.row_major(WARP_SIZE // 4, 4)
    copy_dram_to_sram[thread_layout=thread_layout](a_smem, lhs)
    copy_dram_to_sram[thread_layout=thread_layout](b_smem, rhs)
    barrier()

    alias a_simd_width = mma.a_reg_type.size
    alias b_simd_width = mma.b_reg_type.size
    var a_reg_tile = (
        LayoutTensor[
            dtype,
            Layout.row_major(1, a_simd_width),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .vectorize[1, a_simd_width]()
    )

    var b_reg_tile = (
        LayoutTensor[
            dtype,
            Layout.row_major(1, b_simd_width),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .vectorize[1, b_simd_width]()
    )

    mma.load_a(a_smem, a_reg_tile)
    mma.load_b(b_smem, b_reg_tile)

    var a_frags = a_reg_tile[0, 0].cast[DType.float64]()
    var b_frags = b_reg_tile[0, 0].cast[DType.float64]()

    @parameter
    # NVIDIA
    if a_frags.size == 4 and b_frags.size == 2:
        _printf["thread %u a_vals=[%g %g %g %g], b_vals=[%g %g]\n"](
            thread_idx.x,
            a_frags[0],
            a_frags[1],
            a_frags[2],
            a_frags[3],
            b_frags[0],
            b_frags[1],
        )
    elif a_frags.size == 8 and b_frags.size == 4:
        _printf[
            "thread %u a_vals=[%g %g %g %g %g %g %g %g], b_vals=[%g %g %g %g]\n"
        ](
            thread_idx.x,
            a_frags[0],
            a_frags[1],
            a_frags[2],
            a_frags[3],
            a_frags[4],
            a_frags[5],
            a_frags[6],
            a_frags[7],
            b_frags[0],
            b_frags[1],
            b_frags[2],
            b_frags[3],
        )
    # AMD-MI300
    elif a_frags.size == 4 and b_frags.size == 4:
        _printf["thread %u a_vals=[%g %g %g %g], b_vals=[%g %g %g %g]\n"](
            thread_idx.x,
            a_frags[0],
            a_frags[1],
            a_frags[2],
            a_frags[3],
            b_frags[0],
            b_frags[1],
            b_frags[2],
            b_frags[3],
        )
    elif a_frags.size == 1 and b_frags.size == 1:
        _printf["thread %u a_vals=[%g], b_vals=[%g]\n"](
            thread_idx.x,
            a_frags[0],
            b_frags[0],
        )


def test_load_operands_ldmatrix[
    dst_dtype: DType,
    dtype: DType,
    shape: IndexList[3],
    transpose_b: Bool = False,
](ctx: DeviceContext):
    alias M = shape[0]
    alias N = shape[1]
    alias K = shape[2]

    var lhs = ManagedLayoutTensor[dtype, Layout.row_major(M, K)](ctx)
    arange(lhs.tensor())
    var rhs = ManagedLayoutTensor[dtype, Layout.row_major(K, N)](ctx)
    arange(rhs.tensor())

    alias mma_load_and_print_kernel_fn = mma_load_and_print_operands_kernel_ldmatrix[
        dst_dtype, dtype, lhs.layout, rhs.layout, shape, transpose_b
    ]
    ctx.enqueue_function[mma_load_and_print_kernel_fn](
        lhs.device_tensor(),
        rhs.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(WARP_SIZE),
    )
    ctx.synchronize()
    _ = lhs^
    _ = rhs^
