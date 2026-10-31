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

from max.gpu import WARP_SIZE, lane_id
from max.gpu.host import DeviceContext
from max.gpu.host.info import MI300X
from layout import (
    Idx,
    TensorLayout,
    TileTensor,
    row_major,
    stack_allocation,
)
from layout._fillers import arange
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.tensor_core import TensorCore

from std.utils.coord import DynamicCoord

comptime fp8_dtype = (
    DType.float8_e4m3fnuz if DeviceContext.default_device_info.compute
    <= MI300X.compute else DType.float8_e4m3fn
)
comptime bf8_dtype = (
    DType.float8_e5m2fnuz if DeviceContext.default_device_info.compute
    <= MI300X.compute else DType.float8_e5m2
)


def test_load_a[
    dst_dtype: DType,
    dtype: DType,
    layout: TensorLayout,
    inst_shape: DynamicCoord[.int, 3],
    a_origin: MutOrigin,
    a_lane_origin: MutOrigin,
](
    a: TileTensor[dtype, layout, a_origin],
    a_lane: TileTensor[dtype, type_of(row_major[WARP_SIZE]()), a_lane_origin],
):
    comptime assert type_of(a).LayoutType.all_dims_known
    var mma = TensorCore[dst_dtype, dtype, inst_shape, False]()
    var a_reg_tile = mma.load_a(a)
    comptime assert a_reg_tile.rank == a_reg_tile.flat_rank == 2
    comptime assert a_lane.rank == a_lane.flat_rank == 1
    a_lane[lane_id()] = a_reg_tile[0, 0][0]


def test_load_b[
    dst_dtype: DType,
    dtype: DType,
    layout: TensorLayout,
    inst_shape: DynamicCoord[.int, 3],
    transpose_b: Bool,
    b_origin: MutOrigin,
    b_lane_origin: MutOrigin,
](
    b: TileTensor[dtype, layout, b_origin],
    b_lane: TileTensor[dtype, type_of(row_major[WARP_SIZE]()), b_lane_origin],
):
    comptime assert type_of(b).LayoutType.all_dims_known
    var mma = TensorCore[dst_dtype, dtype, inst_shape, transpose_b]()
    var b_reg_tile = mma.load_b(b)
    comptime assert b_reg_tile.rank == b_reg_tile.flat_rank == 2
    comptime assert b_lane.rank == b_lane.flat_rank == 1
    b_lane[lane_id()] = b_reg_tile[0, 0][0]


def test_load_c[
    dst_dtype: DType,
    dtype: DType,
    layout: TensorLayout,
    c_lane_layout: TensorLayout,
    inst_shape: DynamicCoord[.int, 3],
    c_origin: MutOrigin,
    c_lane_origin: MutOrigin,
](
    c: TileTensor[dst_dtype, layout, c_origin],
    c_lane: TileTensor[dst_dtype, c_lane_layout, c_lane_origin],
):
    comptime assert type_of(c).LayoutType.all_dims_known
    var mma = TensorCore[dst_dtype, dtype, inst_shape, False]()
    var c_reg_tile = mma.load_c(c)
    comptime assert c_reg_tile.rank == c_reg_tile.flat_rank == 2
    comptime assert c_lane.rank == c_lane.flat_rank == 2
    for i in range(4):
        c_lane[lane_id(), i] = c_reg_tile[0, i][0]


def test_store_d[
    dst_dtype: DType,
    dtype: DType,
    layout: TensorLayout,
    inst_shape: DynamicCoord[.int, 3],
    d_origin: MutOrigin,
](d: TileTensor[dst_dtype, layout, d_origin]):
    comptime assert type_of(d).LayoutType.all_dims_known
    var mma = TensorCore[dst_dtype, dtype, inst_shape, False]()
    var src = stack_allocation[dst_dtype, address_space=.LOCAL](
        mma.c_fragment_layout
    ).fill(Scalar[dst_dtype](lane_id()))
    mma.store_d(d, src)


def test_mma_op[
    dst_dtype: DType,
    dtype: DType,
    layout_a: TensorLayout,
    layout_b: TensorLayout,
    layout_c: TensorLayout,
    inst_shape: DynamicCoord[.int, 3],
    transpose_b: Bool,
    a_origin: MutOrigin,
    b_origin: MutOrigin,
    c_origin: MutOrigin,
    d_origin: MutOrigin,
](
    a: TileTensor[dtype, layout_a, a_origin],
    b: TileTensor[dtype, layout_b, b_origin],
    c: TileTensor[dst_dtype, layout_c, c_origin],
    d: TileTensor[dst_dtype, layout_c, d_origin],
):
    var mma = TensorCore[dst_dtype, dtype, inst_shape, transpose_b]()
    comptime assert type_of(a).LayoutType.all_dims_known
    comptime assert type_of(b).LayoutType.all_dims_known
    comptime assert type_of(c).LayoutType.all_dims_known
    comptime assert type_of(d).LayoutType.all_dims_known
    comptime assert a.rank == a.flat_rank == 2
    comptime k_group_size = type_of(a).static_shape[1] // inst_shape[2]
    var a_reg = mma.load_a(a)
    var b_reg = mma.load_b(b)
    var d_reg = mma.load_c(c)

    comptime for k in range(k_group_size):
        var a_reg_k = a_reg.tile[1, mma.a_reg_type.length](0, k)
        var b_reg_k = b_reg.tile[mma.b_reg_type.length, 1](k, 0)
        d_reg = mma.mma_op(a_reg_k, b_reg_k, d_reg)

    mma.store_d(d, d_reg)


def _arange(tensor: TileTensor[mut=True, ...]):
    comptime assert tensor.rank == tensor.flat_rank == 2
    comptime assert type_of(tensor).LayoutType.shape_known
    # The generic arange does not support FP8.
    comptime if tensor.dtype in (DType.bfloat16, DType.float16, DType.float32):
        arange(tensor)
    elif tensor.dtype in (fp8_dtype, bf8_dtype):
        # Keep FP8 inputs in range.
        for i in range(Int(tensor.dim[0]())):
            comptime for j in range(type_of(tensor).static_shape[1]):
                tensor[i, j] = Scalar[tensor.dtype](
                    Float32(0.1 * Float64(i) + 0.2 * Float64(j))
                )
    else:
        comptime assert False, "Unsupported dtype"


def _print_rows(tensor: TileTensor[...]):
    comptime assert tensor.rank == tensor.flat_rank
    comptime assert tensor.rank in (1, 2)
    comptime if tensor.rank == 1:
        for i in range(Int(tensor.dim[0]())):
            print(tensor[i], end=" ")
        print("")
    else:
        for i in range(Int(tensor.dim[0]())):
            for j in range(Int(tensor.dim[1]())):
                print(tensor[i, j], end=" ")
            print("")


def test_load_and_mma_and_multiply_operands[
    dst_dtype: DType,
    dtype: DType,
    shape: DynamicCoord[.int, 3],
    transpose_b: Bool,
    k_group_size: Int = 1,
](ctx: DeviceContext) raises:
    comptime warp_size = ctx.default_device_info.warp_size
    comptime M = shape[0]
    comptime N = shape[1]
    comptime K = shape[2] * k_group_size

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[dst_dtype](M * N)
    var d_device = ctx.enqueue_create_buffer[dst_dtype](M * N)

    var d_device_mma = ctx.enqueue_create_buffer[dst_dtype](M * N)

    var a_lane_device = ctx.enqueue_create_buffer[dtype](warp_size)
    var b_lane_device = ctx.enqueue_create_buffer[dtype](warp_size)
    var c_lane_device = ctx.enqueue_create_buffer[dst_dtype](warp_size * 4)

    comptime layout_mk = row_major[M, K]()
    comptime layout_mn = row_major[M, N]()
    var a_host_managed = HostDeviceTileTensor[dtype](layout_mk)
    var a_host = a_host_managed.host_tensor()
    var a_dev = TileTensor(a_device, row_major(Idx[M], Idx[K]))

    comptime B_row = N if transpose_b else K
    comptime B_col = K if transpose_b else N

    comptime layout_b = row_major[B_row, B_col]()

    var b_host_managed = HostDeviceTileTensor[dtype](layout_b)
    var b_host = b_host_managed.host_tensor()
    var b_dev = TileTensor(b_device, row_major(Idx[B_row], Idx[B_col]))

    var c_host_managed = HostDeviceTileTensor[dst_dtype](layout_mn)
    var c_host = c_host_managed.host_tensor().fill(0)
    var c_dev = TileTensor(c_device, row_major(Idx[M], Idx[N]))

    var d_host_managed = HostDeviceTileTensor[dst_dtype](layout_mn)
    var d_host = d_host_managed.host_tensor().fill(0)

    var d_dev = TileTensor(d_device, row_major(Idx[M], Idx[N]))
    var d_dev_mma = TileTensor(d_device_mma, row_major(Idx[M], Idx[N]))

    comptime layout_warp = row_major[warp_size]()
    comptime layout_warp4 = row_major[warp_size, 4]()

    var a_lane_host_managed = HostDeviceTileTensor[dtype](layout_warp)
    var a_lane_host = a_lane_host_managed.host_tensor()
    var a_lane_dev = TileTensor(a_lane_device, row_major(Idx[warp_size]))
    var b_lane_host_managed = HostDeviceTileTensor[dtype](layout_warp)
    var b_lane_host = b_lane_host_managed.host_tensor()
    var b_lane_dev = TileTensor(b_lane_device, row_major(Idx[warp_size]))

    var c_lane_host_managed = HostDeviceTileTensor[dst_dtype](layout_warp4)
    var c_lane_host = c_lane_host_managed.host_tensor()
    var c_lane_dev = TileTensor(
        c_lane_device, row_major(Idx[warp_size], Idx[4])
    )

    _arange(a_host)
    _arange(b_host)
    _arange(c_host)
    ctx.enqueue_copy(a_device, a_host.ptr)
    ctx.enqueue_copy(b_device, b_host.ptr)
    ctx.enqueue_copy(c_device, c_host.ptr)

    comptime kernel_load_a = test_load_a[
        dst_dtype,
        dtype,
        type_of(a_dev).LayoutType,
        shape,
        a_origin=a_dev.origin,
        a_lane_origin=a_lane_dev.origin,
    ]
    comptime kernel_load_b = test_load_b[
        dst_dtype,
        dtype,
        type_of(b_dev).LayoutType,
        shape,
        transpose_b,
        b_origin=b_dev.origin,
        b_lane_origin=b_lane_dev.origin,
    ]
    comptime kernel_load_c = test_load_c[
        dst_dtype,
        dtype,
        type_of(c_dev).LayoutType,
        type_of(c_lane_dev).LayoutType,
        shape,
        c_origin=c_dev.origin,
        c_lane_origin=c_lane_dev.origin,
    ]
    comptime kernel_store_d = test_store_d[
        dst_dtype,
        dtype,
        type_of(d_dev).LayoutType,
        shape,
        d_origin=d_dev.origin,
    ]

    ctx.enqueue_function[kernel_load_a](
        a_dev,
        a_lane_dev,
        grid_dim=(1, 1),
        block_dim=(warp_size),
    )

    ctx.enqueue_function[kernel_load_b](
        b_dev,
        b_lane_dev,
        grid_dim=(1, 1),
        block_dim=(warp_size),
    )

    ctx.enqueue_function[kernel_load_c](
        c_dev,
        c_lane_dev,
        grid_dim=(1, 1),
        block_dim=(warp_size),
    )

    ctx.enqueue_function[kernel_store_d](
        d_dev, grid_dim=(1, 1), block_dim=(warp_size)
    )

    comptime kernel = test_mma_op[
        dst_dtype,
        dtype,
        type_of(a_dev).LayoutType,
        type_of(b_dev).LayoutType,
        type_of(c_dev).LayoutType,
        shape,
        transpose_b,
        a_origin=a_dev.origin,
        b_origin=b_dev.origin,
        c_origin=c_dev.origin,
        d_origin=d_dev_mma.origin,
    ]

    ctx.enqueue_function[kernel](
        a_dev,
        b_dev,
        c_dev,
        d_dev_mma,
        grid_dim=(1, 1),
        block_dim=(warp_size),
    )

    ctx.enqueue_copy(a_lane_host.ptr, a_lane_device)
    ctx.enqueue_copy(b_lane_host.ptr, b_lane_device)
    ctx.enqueue_copy(c_lane_host.ptr, c_lane_device)
    ctx.enqueue_copy(d_host.ptr, d_device)
    ctx.synchronize()

    print("== test_load_a")
    _print_rows(a_lane_host.as_imm())

    print("== test_load_b")
    _print_rows(b_lane_host.as_imm())

    print("== test_load_c")
    _print_rows(c_lane_host.as_imm())

    print("== test_load_d")
    _print_rows(d_host.as_imm())

    ctx.enqueue_copy(d_host.ptr, d_device_mma)
    ctx.synchronize()

    print("== test_mma")
    _print_rows(d_host.as_imm())
