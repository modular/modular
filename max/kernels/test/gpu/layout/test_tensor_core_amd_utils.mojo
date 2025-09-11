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

from gpu import WARP_SIZE, lane_id
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import TensorCore
from utils.index import Index, IndexList
from gpu.host.info import MI300X

alias fp8_dtype = DType.float8_e4m3fnuz if DeviceContext.default_device_info <= MI300X else DType.float8_e4m3fn
alias bf8_dtype = DType.float8_e5m2fnuz if DeviceContext.default_device_info <= MI300X else DType.float8_e5m2


fn test_load_a[
    dst_dtype: DType,
    dtype: DType,
    layout: Layout,
    inst_shape: IndexList[3],
](
    a: LayoutTensor[dtype, layout, MutableAnyOrigin],
    a_lane: LayoutTensor[dtype, Layout(WARP_SIZE), MutableAnyOrigin],
):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, False]()
    var a_reg_tile = mma.load_a(a)
    # only storing 0th element for result
    a_lane[lane_id()] = a_reg_tile[0, 0]


fn test_load_b[
    dst_dtype: DType,
    dtype: DType,
    layout: Layout,
    inst_shape: IndexList[3],
    transpose_b: Bool,
](
    b: LayoutTensor[dtype, layout, MutableAnyOrigin],
    b_lane: LayoutTensor[dtype, Layout(WARP_SIZE), MutableAnyOrigin],
):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, transpose_b]()
    var b_reg_tile = mma.load_b(b)
    # only storing 0th element for result
    b_lane[lane_id()] = b_reg_tile[0, 0]


fn test_load_c[
    dst_dtype: DType,
    dtype: DType,
    layout: Layout,
    c_lane_layout: Layout,
    inst_shape: IndexList[3],
](
    c: LayoutTensor[dst_dtype, layout, MutableAnyOrigin],
    c_lane: LayoutTensor[dst_dtype, c_lane_layout, MutableAnyOrigin],
):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, False]()
    var c_reg_tile = mma.load_c(c)
    for i in range(4):
        c_lane[lane_id(), i] = c_reg_tile[0, i]


fn test_store_d[
    dst_dtype: DType,
    dtype: DType,
    layout: Layout,
    inst_shape: IndexList[3],
](d: LayoutTensor[dst_dtype, layout, MutableAnyOrigin]):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, False]()
    var src = __type_of(mma).c_reg_tile_type.stack_allocation().fill(lane_id())
    mma.store_d(d, src)


fn test_mma_op[
    dst_dtype: DType,
    dtype: DType,
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
    inst_shape: IndexList[3],
    transpose_b: Bool,
](
    a: LayoutTensor[dtype, layout_a, MutableAnyOrigin],
    b: LayoutTensor[dtype, layout_b, MutableAnyOrigin],
    c: LayoutTensor[dst_dtype, layout_c, MutableAnyOrigin],
    d: LayoutTensor[dst_dtype, layout_c, MutableAnyOrigin],
):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, transpose_b]()
    alias k_group_size = a.layout.shape[1].value() // inst_shape[2]
    var a_reg = mma.load_a(a)
    var b_reg = mma.load_b(b)
    var d_reg = mma.load_c(c)

    @parameter
    for k in range(k_group_size):
        var a_reg_k = a_reg.tile[1, a_reg.layout.size() // k_group_size](0, k)
        var b_reg_k = b_reg.tile[b_reg.layout.size() // k_group_size, 1](k, 0)
        d_reg = mma.mma_op(a_reg_k, b_reg_k, d_reg)

    mma.store_d(d, d_reg)


fn _arange(tensor: LayoutTensor[mut=True, **_]):
    # use custom arange and the current arange does not work with fp8
    @parameter
    if tensor.dtype in (DType.bfloat16, DType.float16, DType.float32):
        arange(tensor)
    elif tensor.dtype in (fp8_dtype, bf8_dtype):
        # scale with 0.1 to avoid overflow
        for i in range(tensor.shape[0]()):

            @parameter
            for j in range(tensor.shape[1]()):
                tensor[i, j] = Scalar[tensor.dtype](Float32(0.1 * i + 0.2 * j))
    else:
        constrained[False, "Unsupported dtype"]()


def test_load_and_mma_and_multiply_operands[
    dst_dtype: DType,
    dtype: DType,
    shape: IndexList[3],
    transpose_b: Bool,
    k_group_size: Int = 1,
](ctx: DeviceContext):
    alias M = shape[0]
    alias N = shape[1]
    alias K = shape[2] * k_group_size

    var a_host_ptr = UnsafePointer[Scalar[dtype]].alloc(M * K)
    var b_host_ptr = UnsafePointer[Scalar[dtype]].alloc(K * N)
    var c_host_ptr = UnsafePointer[Scalar[dst_dtype]].alloc(M * N)
    var d_host_ptr = UnsafePointer[Scalar[dst_dtype]].alloc(M * N)
    var d_ref_ptr = UnsafePointer[Scalar[dst_dtype]].alloc(M * N)

    var a_lane_host_ptr = UnsafePointer[Scalar[dtype]].alloc(WARP_SIZE)
    var b_lane_host_ptr = UnsafePointer[Scalar[dtype]].alloc(WARP_SIZE)
    var c_lane_host_ptr = UnsafePointer[Scalar[dst_dtype]].alloc(WARP_SIZE * 4)

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[dst_dtype](M * N)
    var d_device = ctx.enqueue_create_buffer[dst_dtype](M * N)

    var d_device_mma = ctx.enqueue_create_buffer[dst_dtype](M * N)

    var a_lane_device = ctx.enqueue_create_buffer[dtype](WARP_SIZE)
    var b_lane_device = ctx.enqueue_create_buffer[dtype](WARP_SIZE)
    var c_lane_device = ctx.enqueue_create_buffer[dst_dtype](WARP_SIZE * 4)

    var a_host = tb[dtype]().row_major[M, K]().view(a_host_ptr)
    var a_dev = tb[dtype]().row_major[M, K]().view(a_device.unsafe_ptr())

    alias B_row = N if transpose_b else K
    alias B_col = K if transpose_b else N

    var b_host = tb[dtype]().row_major[B_row, B_col]().view(b_host_ptr)
    var b_dev = (
        tb[dtype]().row_major[B_row, B_col]().view(b_device.unsafe_ptr())
    )

    var c_host = tb[dst_dtype]().row_major[M, N]().view(c_host_ptr).fill(0)
    var c_dev = tb[dst_dtype]().row_major[M, N]().view(c_device.unsafe_ptr())

    var d_host = tb[dst_dtype]().row_major[M, N]().view(d_host_ptr).fill(0)

    var d_dev = tb[dst_dtype]().row_major[M, N]().view(d_device.unsafe_ptr())
    var d_dev_mma = (
        tb[dst_dtype]().row_major[M, N]().view(d_device_mma.unsafe_ptr())
    )

    var a_lane_host = tb[dtype]().layout[WARP_SIZE]().view(a_lane_host_ptr)
    var a_lane_dev = (
        tb[dtype]().layout[WARP_SIZE]().view(a_lane_device.unsafe_ptr())
    )
    var b_lane_host = tb[dtype]().layout[WARP_SIZE]().view(b_lane_host_ptr)
    var b_lane_dev = (
        tb[dtype]().layout[WARP_SIZE]().view(b_lane_device.unsafe_ptr())
    )

    var c_lane_host = (
        tb[dst_dtype]().row_major[WARP_SIZE, 4]().view(c_lane_host_ptr)
    )
    var c_lane_dev = (
        tb[dst_dtype]()
        .row_major[WARP_SIZE, 4]()
        .view(c_lane_device.unsafe_ptr())
    )

    _arange(a_host)
    _arange(b_host)
    _arange(c_host)
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host_ptr)

    alias kernel_load_a = test_load_a[dst_dtype, dtype, a_dev.layout, shape]
    alias kernel_load_b = test_load_b[
        dst_dtype, dtype, b_dev.layout, shape, transpose_b
    ]
    alias kernel_load_c = test_load_c[
        dst_dtype, dtype, c_dev.layout, c_lane_dev.layout, shape
    ]
    alias kernel_store_d = test_store_d[dst_dtype, dtype, c_dev.layout, shape]

    ctx.enqueue_function_checked[kernel_load_a, kernel_load_a](
        a_dev, a_lane_dev, grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )

    ctx.enqueue_function_checked[kernel_load_b, kernel_load_b](
        b_dev, b_lane_dev, grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )

    ctx.enqueue_function_checked[kernel_load_c, kernel_load_c](
        c_dev, c_lane_dev, grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )

    ctx.enqueue_function_checked[kernel_store_d, kernel_store_d](
        d_dev, grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )

    alias kernel = test_mma_op[
        dst_dtype,
        dtype,
        a_dev.layout,
        b_dev.layout,
        c_dev.layout,
        shape,
        transpose_b,
    ]

    ctx.enqueue_function_checked[kernel, kernel](
        a_dev,
        b_dev,
        c_dev,
        d_dev_mma,
        grid_dim=(1, 1),
        block_dim=(WARP_SIZE),
    )

    ctx.enqueue_copy(a_lane_host_ptr, a_lane_device)
    ctx.enqueue_copy(b_lane_host_ptr, b_lane_device)
    ctx.enqueue_copy(c_lane_host_ptr, c_lane_device)
    ctx.enqueue_copy(d_host_ptr, d_device)
    ctx.synchronize()

    print("== test_load_a")
    print(a_lane_host)

    print("== test_load_b")
    print(b_lane_host)

    print("== test_load_c")
    print(c_lane_host)

    print("== test_load_d")
    print(d_host)

    ctx.enqueue_copy(d_host_ptr, d_device_mma)
    ctx.synchronize()

    print("== test_mma")
    print(d_host)
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = d_device^
    _ = a_lane_device^
    _ = b_lane_device^
    _ = c_lane_device^
    _ = d_device_mma^

    _ = a_host_ptr
    _ = b_host_ptr
    _ = c_host_ptr
    _ = d_host_ptr
    _ = a_lane_host_ptr
    _ = b_lane_host_ptr
    _ = c_lane_host_ptr
    _ = d_ref_ptr
