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

from std.math import ceildiv, isclose, isnan, nan
from std.os import abort
from std.random import random_float64

from max.gpu import WARP_SIZE, global_idx, lane_id
from max.gpu.host import DeviceContext
from max.gpu.host.info import H100, MI300X
from layout import *
from layout._utils import ManagedLayoutTensor
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.tensor_core import *
from std.testing import *


def mma_layout_tc[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    layout_c: TensorLayout,
    layout_a: Layout,
    layout_b: Layout,
](
    mat_c: TileTensor[out_type, layout_c, MutAnyOrigin],
    mat_a: LayoutTensor[in_type, layout_a, MutAnyOrigin],
    mat_b: LayoutTensor[in_type, layout_b, MutAnyOrigin],
):
    var tc = TensorCore[out_type, in_type, shape]()
    var a = tc.load_a(mat_a)
    var b = tc.load_b(mat_b)
    var c = tc.load_c(mat_c)
    var d = tc.mma_op(a, b, c)
    tc.store_d(mat_c, d)


def matmul_naive[
    out_type: DType,
    in_type: DType,
    layout_c: TensorLayout,
    layout_a: Layout,
    layout_b: Layout,
](
    mat_c: TileTensor[out_type, layout_c, MutAnyOrigin],
    mat_a: LayoutTensor[in_type, layout_a, MutAnyOrigin],
    mat_b: LayoutTensor[in_type, layout_b, MutAnyOrigin],
):
    comptime assert mat_c.rank == mat_c.flat_rank == 2
    var x = global_idx.x
    var y = global_idx.y

    if x >= Int(mat_c.dim[0]()) or y >= Int(mat_c.dim[1]()):
        return

    var accum = mat_c[x, y]
    for i in range(mat_a.shape[1]()):
        accum += (
            mat_a[x, i][0].cast[out_type]() * mat_b[i, y][0].cast[out_type]()
        )
    mat_c[x, y] = accum


def test_layout_mma[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    M: Int,
    N: Int,
    K: Int,
](
    ctx: DeviceContext,
    rtol: Float64 = 1e-05,
    rng_width: Float64 = 10.0,
    debug: Bool = False,
) raises:
    print("== run layout mma => ", String(out_type), String(in_type), M, N, K)

    comptime layout_a = Layout(IntTuple(M, K), IntTuple(K, 1))
    comptime layout_b = Layout(IntTuple(K, N), IntTuple(N, 1))
    comptime layout_c = row_major[M, N]()

    var mat_a = ManagedLayoutTensor[in_type, layout_a](ctx)
    var mat_b = ManagedLayoutTensor[in_type, layout_b](ctx)
    var mat_c = HostDeviceTileTensor[out_type](layout_c, ctx)
    var mat_a_n = ManagedLayoutTensor[in_type, layout_a](ctx)
    var mat_b_n = ManagedLayoutTensor[in_type, layout_b](ctx)
    var mat_c_n = HostDeviceTileTensor[out_type](layout_c, ctx)

    var rand_min = -1 * rng_width
    var rand_max = rng_width
    var mat_a_tensor = mat_a.tensor()
    var mat_b_tensor = mat_b.tensor()
    var mat_c_tensor = mat_c.host_tensor()

    var mat_a_n_tensor = mat_a_n.tensor()
    var mat_b_n_tensor = mat_b_n.tensor()
    var mat_c_n_tensor = mat_c_n.host_tensor()

    for i in range(M):
        for j in range(K):
            var val = random_float64(rand_min, rand_max).cast[.float32]()
            mat_a_tensor[i, j] = val.cast[in_type]()
            mat_a_n_tensor[i, j] = mat_a_tensor[i, j]
    for i in range(K):
        for j in range(N):
            var val = random_float64(rand_min, rand_max).cast[.float32]()
            mat_b_tensor[i, j] = val.cast[in_type]()
            mat_b_n_tensor[i, j] = mat_b_tensor[i, j]
    for i in range(M):
        for j in range(N):
            var val = Float32((i * N + j) % 13 - 6)
            mat_c_tensor[i, j] = val.cast[out_type]()
            mat_c_n_tensor[i, j] = mat_c_tensor[i, j]

    mat_c.to_device()
    mat_c_n.to_device()

    comptime kernel = mma_layout_tc[
        out_type, in_type, shape, type_of(layout_c), layout_a, layout_b
    ]
    ctx.enqueue_function[kernel](
        mat_c.device_tensor().as_unsafe_any_origin(),
        mat_a.device_tensor(),
        mat_b.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(WARP_SIZE),
    )

    ctx.synchronize()

    comptime warps_per_block = 16
    comptime naive_func = matmul_naive[
        out_type, in_type, type_of(layout_c), layout_a, layout_b
    ]
    ctx.enqueue_function[naive_func](
        mat_c_n.device_tensor().as_unsafe_any_origin(),
        mat_a_n.device_tensor(),
        mat_b_n.device_tensor(),
        grid_dim=(ceildiv(M, warps_per_block), ceildiv(N, warps_per_block)),
        block_dim=(warps_per_block, warps_per_block),
    )

    ctx.synchronize()

    mat_c.to_host()
    mat_c_n.to_host()

    for i in range(M):
        for j in range(N):
            var out_val = mat_c.host_tensor()[i, j]
            var out_ref = mat_c_n.host_tensor()[i, j]
            if debug:
                if not isclose(out_val, out_ref, rtol=rtol):
                    print(i, out_val, out_ref)
            assert_true(isclose(out_val, out_ref, rtol=rtol))


def c_fragment_kernel[
    dtype: DType,
    shape: IndexList[3],
    MatrixLayout: TensorLayout,
    FragmentLayout: TensorLayout,
](
    c: TileTensor[dtype, MatrixLayout, ImmutAnyOrigin],
    d: TileTensor[dtype, MatrixLayout, MutAnyOrigin],
    loaded: TileTensor[dtype, FragmentLayout, MutAnyOrigin],
):
    comptime assert loaded.rank == loaded.flat_rank == 2
    var tc = TensorCore[
        dtype, DType.float64 if dtype == .float64 else DType.float16, shape
    ]()
    var fragment = tc.load_c(c)
    comptime registers = shape[0] * shape[1] // WARP_SIZE
    comptime for register in range(registers):
        loaded[lane_id(), register] = fragment[0, register][0]
        fragment[0, register] = Scalar[dtype](lane_id() * registers + register)
    tc.store_d(d, fragment)


def test_c_fragment_layout[
    dtype: DType,
    shape: IndexList[3],
    row_stride: Int,
    col_stride: Int,
](ctx: DeviceContext) raises:
    comptime M = shape[0]
    comptime N = shape[1]
    comptime registers = M * N // WARP_SIZE
    comptime guard = 8
    comptime storage_size = 2 * guard + (M - 1) * row_stride + (
        N - 1
    ) * col_stride + 1
    comptime storage_layout = row_major[storage_size]()
    comptime fragment_layout = row_major[WARP_SIZE, registers]()
    # Runtime strides exercise non-contiguous columns as well as padded rows.
    var matrix_layout = MixedLayout(
        Coord(Idx[M], Idx[N]), Coord(Int(row_stride), Int(col_stride))
    )
    var c = HostDeviceTileTensor[dtype](storage_layout, ctx)
    var d = HostDeviceTileTensor[dtype](storage_layout, ctx)
    var loaded = HostDeviceTileTensor[dtype](fragment_layout, ctx)
    var expected = HostDeviceTileTensor[dtype](storage_layout)
    var c_host = c.host_tensor().fill(nan[dtype]())
    var expected_host = expected.host_tensor().fill(nan[dtype]())
    _ = d.host_tensor().fill(nan[dtype]())
    _ = loaded.host_tensor().fill(nan[dtype]())
    for row in range(M):
        for col in range(N):
            c_host[guard + row * row_stride + col * col_stride] = Scalar[dtype](
                row * N + col + 1
            )
    c.to_device()
    d.to_device()
    loaded.to_device()
    var c_view = TileTensor(
        c.device_tensor().unsafe_ptr().unsafe_offset(guard), matrix_layout
    )
    var d_view = TileTensor(
        d.device_tensor().unsafe_ptr().unsafe_offset(guard), matrix_layout
    )
    comptime kernel = c_fragment_kernel[
        dtype, shape, type_of(matrix_layout), type_of(fragment_layout)
    ]
    ctx.enqueue_function[kernel](
        c_view.as_imm().as_unsafe_any_origin(),
        d_view.as_unsafe_any_origin(),
        loaded.device_tensor().as_unsafe_any_origin(),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )
    ctx.synchronize()
    loaded.to_host()
    d.to_host()
    var fragments = loaded.host_tensor()
    for row in range(M):
        for col in range(N):
            var lane: Int
            var register: Int
            comptime if ctx.target.is_nvidia_gpu():
                lane = (row % 8) * 4 + col // 2
                register = (row // 8) * 2 + col % 2
            else:
                lane = ((row // 4) % (WARP_SIZE // N)) * N + col
                register = (row // (4 * (WARP_SIZE // N))) * 4 + row % 4
            assert_equal(
                fragments[lane, register], Scalar[dtype](row * N + col + 1)
            )
            expected_host[guard + row * row_stride + col * col_stride] = Scalar[
                dtype
            ](lane * registers + register)
    var d_host = d.host_tensor()
    for i in range(storage_size):
        if isnan(expected_host[i]):
            assert_true(isnan(d_host[i]))
        else:
            assert_equal(d_host[i], expected_host[i])


def test_c_fragments[
    dtype: DType, shape: IndexList[3]
](ctx: DeviceContext) raises:
    test_c_fragment_layout[dtype, shape, shape[1], 1](ctx)
    test_c_fragment_layout[dtype, shape, shape[1] + 3, 1](ctx)
    test_c_fragment_layout[dtype, shape, 1, shape[0] + 3](ctx)


def input_fragment_kernel[
    dtype: DType,
    shape: IndexList[3],
    InputLayout: TensorLayout,
    FragmentLayout: TensorLayout,
](
    source: TileTensor[dtype, InputLayout, ImmutAnyOrigin],
    loaded: TileTensor[dtype, FragmentLayout, MutAnyOrigin],
):
    comptime assert loaded.rank == loaded.flat_rank == 3
    comptime assert shape[0] == shape[1]
    var tc = TensorCore[DType.float32, dtype, shape]()
    var tc_transpose = TensorCore[DType.float32, dtype, shape, True]()
    var a = tc.load_a(source)
    var b = tc.load_b(source.transpose())
    var b_transpose = tc_transpose.load_b(source)
    comptime for register in range(loaded.static_shape[1]):
        loaded[lane_id(), register, 0] = a[0, register][0]
        loaded[lane_id(), register, 1] = b[register, 0][0]
        loaded[lane_id(), register, 2] = b_transpose[register, 0][0]


def test_input_fragment_layout[
    dtype: DType,
    shape: IndexList[3],
    k_groups: Int,
](ctx: DeviceContext, column_major: Bool) raises:
    comptime rows = shape[0]
    comptime cols = shape[2] * k_groups
    comptime registers = rows * cols // WARP_SIZE
    var row_stride = 1 if column_major else cols + 3
    var col_stride = rows + 3 if column_major else 1
    comptime guard = 8
    comptime storage_size = 2 * guard + max(
        rows * (cols + 3), cols * (rows + 3)
    )
    comptime storage_layout = row_major[storage_size]()
    comptime fragment_layout = row_major[WARP_SIZE, registers, 3]()
    var matrix_layout = MixedLayout(
        Coord(Idx[rows], Idx[cols]), Coord(row_stride, col_stride)
    )
    var source = HostDeviceTileTensor[dtype](storage_layout, ctx)
    var loaded = HostDeviceTileTensor[dtype](fragment_layout, ctx)
    var source_host = source.host_tensor().fill(nan[dtype]())
    var source_view = TileTensor(
        source.device_tensor().unsafe_ptr().unsafe_offset(guard), matrix_layout
    )
    comptime kernel = input_fragment_kernel[
        dtype, shape, type_of(matrix_layout), type_of(fragment_layout)
    ]
    # Base-eight digits keep every coordinate distinguishable even in FP8.
    comptime tag_count = 4 if dtype.is_float8() else 2
    for tag in range(tag_count):
        var axis = tag // 2 if dtype.is_float8() else tag
        for row in range(rows):
            for col in range(cols):
                var coordinate = row if axis == 0 else col
                comptime if dtype.is_float8():
                    coordinate = (
                        coordinate % 8 if tag % 2 == 0 else coordinate // 8
                    )
                source_host[
                    guard + row * row_stride + col * col_stride
                ] = Scalar[dtype](coordinate)
        source.to_device()
        _ = loaded.host_tensor().fill(nan[dtype]())
        loaded.to_device()
        ctx.enqueue_function[kernel](
            source_view.as_imm().as_unsafe_any_origin(),
            loaded.device_tensor().as_unsafe_any_origin(),
            grid_dim=1,
            block_dim=WARP_SIZE,
        )
        ctx.synchronize()
        loaded.to_host()
        var fragments = loaded.host_tensor()
        for row in range(rows):
            for col in range(cols):
                var lane = (col // registers) * rows + row
                var register = col % registers
                for operand in range(3):
                    assert_equal(
                        Float32(fragments[lane, register, operand]),
                        Float32(
                            source_host[
                                guard + row * row_stride + col * col_stride
                            ]
                        ),
                    )


def test_input_fragments[
    dtype: DType, shape: IndexList[3], k_groups: Int = 1
](ctx: DeviceContext) raises:
    test_input_fragment_layout[dtype, shape, k_groups](ctx, False)
    test_input_fragment_layout[dtype, shape, k_groups](ctx, True)


def main() raises:
    with DeviceContext() as ctx:
        comptime if ctx.target.is_nvidia_gpu():
            comptime shape_884 = IndexList[3](8, 8, 4)
            comptime shape_1684 = IndexList[3](16, 8, 4)
            comptime shape_1688 = IndexList[3](16, 8, 8)
            comptime shape_16816 = IndexList[3](16, 8, 16)

            test_c_fragments[.float64, shape_884](ctx)
            test_c_fragments[.float64, shape_16816](ctx)
            test_c_fragments[.float32, shape_16816](ctx)

            test_layout_mma[.float64, DType.float64, shape_884, 8, 8, 4](
                ctx, rtol=1e-01
            )

            # FP64 M=16 MMA requires SM90; the M=8 case also supports SM80.
            comptime if ctx.default_device_info.compute >= H100.compute:
                test_layout_mma[.float64, DType.float64, shape_1684, 16, 8, 4](
                    ctx, rtol=1e-01
                )

                test_layout_mma[.float64, DType.float64, shape_1688, 16, 8, 8](
                    ctx, rtol=1e-01
                )

                test_layout_mma[
                    DType.float64, DType.float64, shape_16816, 16, 8, 16
                ](ctx, rtol=1e-01)

            test_layout_mma[.float32, DType.float32, shape_1684, 16, 8, 4](
                ctx, rtol=1e-01
            )
            test_layout_mma[.float32, DType.float32, shape_1688, 16, 8, 8](
                ctx, rtol=1e-01
            )
            test_layout_mma[
                DType.float32, DType.bfloat16, shape_1688, 16, 8, 8
            ](ctx, rtol=1e-01)
            test_layout_mma[.float32, DType.float16, shape_1688, 16, 8, 8](
                ctx, rtol=1e-01
            )
        elif ctx.target.is_amd_gpu():
            comptime shape_161616 = IndexList[3](16, 16, 16)
            comptime shape_16164 = IndexList[3](16, 16, 4)

            test_input_fragments[.float32, shape_16164](ctx)
            test_input_fragments[.float32, shape_16164, 4](ctx)
            test_input_fragments[.float16, shape_161616](ctx)
            test_input_fragments[.float16, shape_161616, 2](ctx)
            test_input_fragments[.bfloat16, shape_32x32x8](ctx)
            test_input_fragments[.bfloat16, shape_32x32x8, 2](ctx)
            test_input_fragments[.bfloat16, shape_32x32x16](ctx)
            comptime fp8 = DType.float8_e4m3fnuz if ctx.default_device_info.compute <= MI300X.compute else DType.float8_e4m3fn
            comptime bf8 = DType.float8_e5m2fnuz if ctx.default_device_info.compute <= MI300X.compute else DType.float8_e5m2
            test_input_fragments[fp8, shape_16x16x32](ctx)
            test_input_fragments[fp8, shape_16x16x32, 2](ctx)
            test_input_fragments[bf8, shape_16x16x32](ctx)
            test_input_fragments[bf8, shape_16x16x32, 2](ctx)

            test_c_fragments[.float32, shape_161616](ctx)
            test_c_fragments[.float32, shape_32x32x8](ctx)

            test_layout_mma[
                DType.float32, DType.float16, shape_161616, 16, 16, 16
            ](ctx, rtol=1e-01)
            test_layout_mma[
                DType.float32, DType.bfloat16, shape_161616, 16, 16, 16
            ](ctx, rtol=1e-01)
            test_layout_mma[
                DType.float32, DType.float32, shape_16164, 16, 16, 4
            ](ctx, rtol=1e-01)
        else:
            abort("Unknown GPU Accelerator.")
