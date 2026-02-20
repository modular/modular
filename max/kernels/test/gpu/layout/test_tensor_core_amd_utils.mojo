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

from gpu import WARP_SIZE, lane_id
from gpu.host import DeviceContext
from gpu.host.info import MI300X
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout.tensor_core import TensorCore

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from utils.index import Index, IndexList

comptime fp8_dtype = (
    DType.float8_e4m3fnuz if DeviceContext.default_device_info.compute
    <= MI300X.compute else DType.float8_e4m3fn
)
comptime bf8_dtype = (
    DType.float8_e5m2fnuz if DeviceContext.default_device_info.compute
    <= MI300X.compute else DType.float8_e5m2
)


fn test_load_a[
    dst_dtype: DType,
    dtype: DType,
    layout: Layout,
    inst_shape: IndexList[3],
](
    a: LayoutTensor[dtype, layout, MutAnyOrigin],
    a_lane: LayoutTensor[dtype, Layout(WARP_SIZE), MutAnyOrigin],
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
    b: LayoutTensor[dtype, layout, MutAnyOrigin],
    b_lane: LayoutTensor[dtype, Layout(WARP_SIZE), MutAnyOrigin],
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
    c: LayoutTensor[dst_dtype, layout, MutAnyOrigin],
    c_lane: LayoutTensor[dst_dtype, c_lane_layout, MutAnyOrigin],
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
](d: LayoutTensor[dst_dtype, layout, MutAnyOrigin]):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, False]()
    var src = (
        type_of(mma)
        .c_reg_tile_type.stack_allocation()
        .fill(Scalar[dst_dtype](lane_id()))
    )
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
    a: LayoutTensor[dtype, layout_a, MutAnyOrigin],
    b: LayoutTensor[dtype, layout_b, MutAnyOrigin],
    c: LayoutTensor[dst_dtype, layout_c, MutAnyOrigin],
    d: LayoutTensor[dst_dtype, layout_c, MutAnyOrigin],
):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, transpose_b]()
    comptime k_group_size = a.layout.shape[1].value() // inst_shape[2]
    var a_reg = mma.load_a(a)
    var b_reg = mma.load_b(b)
    var d_reg = mma.load_c(c)

    comptime for k in range(k_group_size):
        var a_reg_k = a_reg.tile[1, a_reg.layout.size() // k_group_size](0, k)
        var b_reg_k = b_reg.tile[b_reg.layout.size() // k_group_size, 1](k, 0)
        d_reg = mma.mma_op(a_reg_k, b_reg_k, d_reg)

    mma.store_d(d, d_reg)


fn _arange(tensor: LayoutTensor[mut=True, ...]):
    # use custom arange and the current arange does not work with fp8
    comptime if tensor.dtype in (DType.bfloat16, DType.float16, DType.float32):
        arange(tensor)
    elif tensor.dtype in (fp8_dtype, bf8_dtype):
        # scale with 0.1 to avoid overflow
        for i in range(tensor.shape[0]()):
            comptime for j in range(tensor.shape[1]()):
                tensor[i, j] = Scalar[tensor.dtype](
                    Float32(0.1 * Float64(i) + 0.2 * Float64(j))
                )
    else:
        constrained[False, "Unsupported dtype"]()


def test_load_and_mma_and_multiply_operands[
    dst_dtype: DType,
    dtype: DType,
    shape: IndexList[3],
    transpose_b: Bool,
    k_group_size: Int = 1,
](ctx: DeviceContext):
    comptime M = shape[0]
    comptime N = shape[1]
    comptime K = shape[2] * k_group_size

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

    comptime layout_mk = Layout.row_major(M, K)
    comptime layout_mn = Layout.row_major(M, N)
    var a_host = LayoutTensor[dtype, layout_mk](a_host_ptr)
    var a_dev = LayoutTensor[dtype, layout_mk](a_device)

    comptime B_row = N if transpose_b else K
    comptime B_col = K if transpose_b else N

    comptime layout_b = Layout.row_major(B_row, B_col)

    var b_host = LayoutTensor[dtype, layout_b](b_host_ptr)
    var b_dev = LayoutTensor[dtype, layout_b](b_device)

    var c_host = LayoutTensor[dst_dtype, layout_mn](c_host_ptr).fill(0)
    var c_dev = LayoutTensor[dst_dtype, layout_mn](c_device)

    var d_host = LayoutTensor[dst_dtype, layout_mn](d_host_ptr).fill(0)

    var d_dev = LayoutTensor[dst_dtype, layout_mn](d_device)
    var d_dev_mma = LayoutTensor[dst_dtype, layout_mn](d_device_mma)

    comptime layout_warp = Layout(WARP_SIZE)
    comptime layout_warp4 = Layout.row_major(WARP_SIZE, 4)

    var a_lane_host = LayoutTensor[dtype, layout_warp](a_lane_host_ptr)
    var a_lane_dev = LayoutTensor[dtype, layout_warp](a_lane_device)
    var b_lane_host = LayoutTensor[dtype, layout_warp](b_lane_host_ptr)
    var b_lane_dev = LayoutTensor[dtype, layout_warp](b_lane_device)

    var c_lane_host = LayoutTensor[dst_dtype, layout_warp4](c_lane_host_ptr)
    var c_lane_dev = LayoutTensor[dst_dtype, layout_warp4](c_lane_device)

    _arange(a_host)
    _arange(b_host)
    _arange(c_host)
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host_ptr)

    comptime kernel_load_a = test_load_a[dst_dtype, dtype, a_dev.layout, shape]
    comptime kernel_load_b = test_load_b[
        dst_dtype, dtype, b_dev.layout, shape, transpose_b
    ]
    comptime kernel_load_c = test_load_c[
        dst_dtype, dtype, c_dev.layout, c_lane_dev.layout, shape
    ]
    comptime kernel_store_d = test_store_d[
        dst_dtype, dtype, c_dev.layout, shape
    ]

    ctx.enqueue_function[kernel_load_a, kernel_load_a](
        a_dev, a_lane_dev, grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )

    ctx.enqueue_function[kernel_load_b, kernel_load_b](
        b_dev, b_lane_dev, grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )

    ctx.enqueue_function[kernel_load_c, kernel_load_c](
        c_dev, c_lane_dev, grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )

    ctx.enqueue_function[kernel_store_d, kernel_store_d](
        d_dev, grid_dim=(1, 1), block_dim=(WARP_SIZE)
    )

    comptime kernel = test_mma_op[
        dst_dtype,
        dtype,
        a_dev.layout,
        b_dev.layout,
        c_dev.layout,
        shape,
        transpose_b,
    ]

    ctx.enqueue_function[kernel, kernel](
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
