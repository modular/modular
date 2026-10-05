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
from layout import TileTensor, coord, row_major
from layout.tile_layout import Layout
from layout._host_device_tile_tensor import HostDeviceTileTensor
from layout.math import max, sum
from max.gpu.host import DeviceContext
from std.testing import assert_equal


def mixed_dtype_reductions(
    src: TileTensor[.float16, type_of(row_major[12]()), ImmutAnyOrigin],
    dst: TileTensor[.float32, type_of(row_major[16]()), MutAnyOrigin],
):
    var inp = TileTensor(src.unsafe_ptr(), Layout(coord[2, 3], coord[1, 4]))
    var summed = TileTensor(dst.unsafe_ptr(), Layout(coord[3], coord[2]))
    var maxima = TileTensor(dst.unsafe_ptr() + 6, Layout(coord[3], coord[2]))
    sum[0](inp, summed)
    max[0](inp, maxima)
    var rows_sum = sum[1](inp)
    var rows_max = max[1](inp)
    comptime for i in range(2):
        dst[12 + i] = rows_sum[i].cast[.float32]()
        dst[14 + i] = rows_max[i].cast[.float32]()


def vector_reductions(
    src: TileTensor[.float32, type_of(row_major[25]()), ImmutAnyOrigin],
    dst: TileTensor[.float32, type_of(row_major[76]()), MutAnyOrigin],
):
    # Scalar-aligned SIMD cells need not have vector-aligned base addresses.
    var inp = TileTensor(src.unsafe_ptr() + 1, row_major[2, 12]()).vectorize[
        1, 4
    ]()
    var rows_sum = (
        TileTensor(dst.unsafe_ptr(), row_major[2, 9]())
        .vectorize[1, 4]()
        .slice[:, 0]()
    )
    var rows_max = (
        TileTensor(dst.unsafe_ptr() + 18, row_major[2, 9]())
        .vectorize[1, 4]()
        .slice[:, 0]()
    )
    sum[1](inp, rows_sum)
    max[1](inp, rows_max)
    var allocated_cols_sum = sum[0](inp)
    var allocated_cols_max = max[0](inp)
    var allocated_rows_sum = sum[1](inp)
    var allocated_rows_max = max[1](inp)
    comptime for j in range(3):
        comptime for lane in range(4):
            dst[36 + 4 * j + lane] = allocated_cols_sum[j][lane]
            dst[48 + 4 * j + lane] = allocated_cols_max[j][lane]
    comptime for i in range(2):
        comptime for lane in range(4):
            dst[60 + 4 * i + lane] = allocated_rows_sum[i][lane]
            dst[68 + 4 * i + lane] = allocated_rows_max[i][lane]


def main() raises:
    with DeviceContext() as ctx:
        var mixed_src = HostDeviceTileTensor[.float16](row_major[12](), ctx)
        var mixed_dst = HostDeviceTileTensor[.float32](row_major[16](), ctx)
        _ = mixed_src.host_tensor().fill(-99)
        _ = mixed_dst.host_tensor().fill(-77)
        for i in range(2):
            for j in range(3):
                mixed_src.host_tensor()[i + 4 * j] = Float16(-10 + 3 * i + j)
        mixed_src.to_device()
        mixed_dst.to_device()
        ctx.enqueue_function[mixed_dtype_reductions](
            mixed_src.device_tensor().as_imm().as_unsafe_any_origin(),
            mixed_dst.device_tensor().as_unsafe_any_origin(),
            grid_dim=1,
            block_dim=1,
        )
        mixed_dst.to_host()
        for j in range(3):
            assert_equal(mixed_dst.host_tensor()[2 * j], Float32(-17 + 2 * j))
            assert_equal(mixed_dst.host_tensor()[6 + 2 * j], Float32(-7 + j))
            assert_equal(mixed_dst.host_tensor()[2 * j + 1], Float32(-77))
            assert_equal(mixed_dst.host_tensor()[7 + 2 * j], Float32(-77))
        for i in range(2):
            assert_equal(mixed_dst.host_tensor()[12 + i], Float32(-27 + 9 * i))
            assert_equal(mixed_dst.host_tensor()[14 + i], Float32(-8 + 3 * i))

        var vec_src = HostDeviceTileTensor[.float32](row_major[25](), ctx)
        var vec_dst = HostDeviceTileTensor[.float32](row_major[76](), ctx)
        vec_src.host_tensor()[0] = -99
        for i in range(24):
            vec_src.host_tensor()[i + 1] = Float32(i - 30)
        _ = vec_dst.host_tensor().fill(-77)
        vec_src.to_device()
        vec_dst.to_device()
        ctx.enqueue_function[vector_reductions](
            vec_src.device_tensor().as_imm().as_unsafe_any_origin(),
            vec_dst.device_tensor().as_unsafe_any_origin(),
            grid_dim=1,
            block_dim=1,
        )
        vec_dst.to_host()
        for i in range(2):
            for lane in range(4):
                var expected_sum = Float32(36 * i + 3 * lane - 78)
                var expected_max = Float32(12 * i + lane - 22)
                assert_equal(vec_dst.host_tensor()[9 * i + lane], expected_sum)
                assert_equal(
                    vec_dst.host_tensor()[18 + 9 * i + lane], expected_max
                )
                assert_equal(
                    vec_dst.host_tensor()[60 + 4 * i + lane], expected_sum
                )
                assert_equal(
                    vec_dst.host_tensor()[68 + 4 * i + lane], expected_max
                )
            for lane in range(4, 9):
                assert_equal(vec_dst.host_tensor()[9 * i + lane], Float32(-77))
                assert_equal(
                    vec_dst.host_tensor()[18 + 9 * i + lane], Float32(-77)
                )
        for j in range(3):
            for lane in range(4):
                assert_equal(
                    vec_dst.host_tensor()[36 + 4 * j + lane],
                    Float32(8 * j + 2 * lane - 48),
                )
                assert_equal(
                    vec_dst.host_tensor()[48 + 4 * j + lane],
                    Float32(4 * j + lane - 18),
                )
