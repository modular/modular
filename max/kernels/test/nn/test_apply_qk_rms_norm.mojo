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

from std.math import divmod, isnan, nan, rsqrt
from std.sys.info import simd_width_of
from std.testing import assert_equal, assert_true

from layout import Coord, TileTensor, row_major
from layout.tile_layout import Layout
from nn.normalization import apply_qk_rms_norm_cpu


def run_apply_qk_rms_norm_cpu[
    in_dtype: DType, out_dtype: DType
](
    cols: Int,
    input_stride: Int = 1,
    gamma_stride: Int = 1,
    output_stride: Int = 1,
    strided_k: Bool = False,
) raises:
    comptime rows = 3
    comptime epsilon = Float32(1e-6)
    var k_cols = cols + 1
    var q_input_stride = 1 if strided_k else input_stride
    var k_input_stride = input_stride if strided_k else 1
    var q_gamma_stride = 1 if strided_k else gamma_stride
    var k_gamma_stride = gamma_stride if strided_k else 1
    var q_output_stride = 1 if strided_k else output_stride
    var k_output_stride = output_stride if strided_k else 1
    # Odd row strides exercise unaligned vector loads and stores.
    var q_input_row_stride = cols * q_input_stride + 1
    var k_input_row_stride = k_cols * k_input_stride + 1
    var q_output_row_stride = cols * q_output_stride + 1
    var k_output_row_stride = k_cols * k_output_stride + 1
    var q_data = List[Scalar[in_dtype]](
        length=rows * q_input_row_stride, fill=nan[in_dtype]()
    )
    var k_data = List[Scalar[in_dtype]](
        length=rows * k_input_row_stride, fill=nan[in_dtype]()
    )
    var gamma_q_data = List[Float32](
        length=cols * q_gamma_stride, fill=nan[.float32]()
    )
    var gamma_k_data = List[Float32](
        length=k_cols * k_gamma_stride, fill=nan[.float32]()
    )
    var q_out_data = List[Scalar[out_dtype]](
        length=rows * q_output_row_stride + 1, fill=nan[out_dtype]()
    )
    var k_out_data = List[Scalar[out_dtype]](
        length=rows * k_output_row_stride + 1, fill=nan[out_dtype]()
    )
    var stats_data = List[Float32](length=rows * 2, fill=0)
    for r in range(rows):
        stats_data[r * 2] = Float32(r + 1) * 0.25
        stats_data[r * 2 + 1] = Float32(r + 2) * 0.5
        for c in range(cols):
            q_data[r * q_input_row_stride + c * q_input_stride] = (
                Float32((r * cols + c) % 19 - 9) * 0.125
            ).cast[in_dtype]()
        for c in range(k_cols):
            k_data[r * k_input_row_stride + c * k_input_stride] = (
                Float32((r * k_cols + c) % 17 - 8) * 0.25
            ).cast[in_dtype]()
    for c in range(cols):
        gamma_q_data[c * q_gamma_stride] = Float32(c % 7 - 3) * 0.25
    for c in range(k_cols):
        gamma_k_data[c * k_gamma_stride] = Float32(c % 5 - 2) * 0.5

    var q = TileTensor(
        q_data,
        Layout(Coord(rows, cols), Coord(q_input_row_stride, q_input_stride)),
    )
    var k = TileTensor(
        k_data,
        Layout(Coord(rows, k_cols), Coord(k_input_row_stride, k_input_stride)),
    )
    var q_out = TileTensor(
        q_out_data,
        Layout(Coord(rows, cols), Coord(q_output_row_stride, q_output_stride)),
    )
    var k_out = TileTensor(
        k_out_data,
        Layout(
            Coord(rows, k_cols), Coord(k_output_row_stride, k_output_stride)
        ),
    )
    var gamma_q = TileTensor(
        gamma_q_data, Layout(Coord(cols), Coord(q_gamma_stride))
    )
    var gamma_k = TileTensor(
        gamma_k_data, Layout(Coord(k_cols), Coord(k_gamma_stride))
    )
    var stats = TileTensor(stats_data, row_major[rows, 2]())

    apply_qk_rms_norm_cpu(
        q_out,
        k_out,
        gamma_q.as_imm(),
        gamma_k.as_imm(),
        stats.as_imm(),
        q.as_imm(),
        k.as_imm(),
        epsilon,
        rows,
        cols,
        k_cols,
    )

    for r in range(rows):
        var rs_q = rsqrt(stats_data[r * 2] + epsilon)
        var rs_k = rsqrt(stats_data[r * 2 + 1] + epsilon)
        for c in range(cols):
            var offset = r * q_input_row_stride + c * q_input_stride
            var expected_q = (
                (q_data[offset].cast[.float32]() * rs_q)
                * gamma_q_data[c * q_gamma_stride]
            ).cast[out_dtype]()
            assert_equal(q_out[r, c], expected_q)
        for c in range(k_cols):
            var offset = r * k_input_row_stride + c * k_input_stride
            var expected_k = (
                (k_data[offset].cast[.float32]() * rs_k)
                * gamma_k_data[c * k_gamma_stride]
            ).cast[out_dtype]()
            assert_equal(k_out[r, c], expected_k)

    check_padding(q_out_data, rows, cols, q_output_row_stride, q_output_stride)
    check_padding(
        k_out_data, rows, k_cols, k_output_row_stride, k_output_stride
    )


def check_padding[
    dtype: DType
](
    data: List[Scalar[dtype]],
    rows: Int,
    cols: Int,
    row_stride: Int,
    stride: Int,
) raises:
    for i in range(len(data)):
        var r, offset = divmod(i, row_stride)
        if r >= rows or offset % stride != 0 or offset // stride >= cols:
            assert_true(isnan(data[i].cast[.float32]()))


def run_widths[in_dtype: DType, out_dtype: DType]() raises:
    comptime width = simd_width_of[DType.float32]()
    for cols in [0, 1, width - 1, width, width + 1, 2 * width + 1]:
        run_apply_qk_rms_norm_cpu[in_dtype, out_dtype](cols)
        for strided_k in [False, True]:
            run_apply_qk_rms_norm_cpu[in_dtype, out_dtype](
                cols, input_stride=2, strided_k=strided_k
            )
            run_apply_qk_rms_norm_cpu[in_dtype, out_dtype](
                cols, gamma_stride=2, strided_k=strided_k
            )
            run_apply_qk_rms_norm_cpu[in_dtype, out_dtype](
                cols, output_stride=2, strided_k=strided_k
            )


def main() raises:
    run_widths[.float32, .float32]()
    run_widths[.bfloat16, .bfloat16]()
    run_widths[.bfloat16, .float32]()
    run_widths[.float32, .bfloat16]()
