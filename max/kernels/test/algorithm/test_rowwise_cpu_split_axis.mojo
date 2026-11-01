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
"""Normalize-shaped rowwise kernels on CPU rows long enough to reach the
split-axis tiers (few rows, `axis_size >= 262144`). One row takes the
pointwise split-axis tier; two rows stay one worker per row."""

from std.math import exp, log, sqrt

from builtin_kernels.rowwise_normalization import rms_norm
from builtin_kernels.rowwise_softmax import softmax
from layout import Coord, TileTensor, row_major
from std.testing import assert_almost_equal, TestSuite
from std.utils.index import Index

comptime dtype = DType.float32
comptime cols_even = 524288
# Not a multiple of the split count or the SIMD width, so the last split's
# stripe is short and ends in a scalar tail.
comptime cols_uneven = 262147


def _input(rows: Int, cols: Int) -> List[Scalar[dtype]]:
    var x = List[Scalar[dtype]](length=rows * cols, fill=0)
    for i in range(rows * cols):
        x[i] = Scalar[dtype]((i * 7919) % 101) * 0.02 - 1.0
    return x^


def _sample_cols(cols: Int) -> List[Int]:
    var out = List[Int]()
    for c in range(0, cols, 4099):
        out.append(c)
    out.append(cols - 1)
    return out^


def _check_rms_norm[cols: Int](rows: Int) raises:
    var x = _input(rows, cols)
    var out = List[Scalar[dtype]](length=rows * cols, fill=0)
    var gamma_h = List[Scalar[dtype]](length=cols, fill=1)
    var gamma = TileTensor[mut=False](gamma_h.unsafe_ptr(), row_major(cols))
    var x_ptr = x.unsafe_ptr()
    var out_ptr = out.unsafe_ptr()
    comptime eps = Float32(1e-5)

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](c: Coord) {var x_ptr} -> SIMD[dtype, width]:
        var i = Int(c[0]) * cols + Int(c[1])
        return x_ptr.unsafe_offset(i).unsafe_load[width=width]()

    @inline(.always)
    def output_fn[
        width: SIMDLength, alignment: Int
    ](c: Coord, v: SIMD[dtype, width]) {var out_ptr}:
        var i = Int(c[0]) * cols + Int(c[1])
        out_ptr.unsafe_offset(i).unsafe_store[width=width](v)

    rms_norm[dtype, 2, target="cpu", epsilon=eps, weight_offset=Float32(0)](
        input_fn,
        output_fn,
        Coord(Index(rows, cols)),
        cols,
        gamma,
    )

    for r in range(rows):
        var ss = Float64(0)
        for c in range(cols):
            ss += Float64(x[r * cols + c]) ** 2
        var inv = 1.0 / sqrt(ss / Float64(cols) + Float64(eps))
        for c in _sample_cols(cols):
            assert_almost_equal(
                Float64(out[r * cols + c]),
                Float64(x[r * cols + c]) * inv,
                atol=1e-3,
                msg=String("rms_norm row ", r, " col ", c),
            )
    _ = gamma_h^


def _check_softmax[cols: Int, logsoftmax: Bool = False](rows: Int) raises:
    var x = _input(rows, cols)
    var out = List[Scalar[dtype]](length=rows * cols, fill=0)
    var out_t = TileTensor(out.unsafe_ptr(), row_major(rows, cols))
    var x_ptr = x.unsafe_ptr()

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](c: Coord) {var x_ptr} -> SIMD[dtype, width]:
        var i = Int(c[0]) * cols + Int(c[1])
        return x_ptr.unsafe_offset(i).unsafe_load[width=width]()

    softmax[dtype, 2, target="cpu", logsoftmax=logsoftmax](
        input_fn, Coord(Index(rows, cols)), cols, out_t, 1
    )

    for r in range(rows):
        var m = Float64(x[r * cols])
        for c in range(cols):
            m = max(m, Float64(x[r * cols + c]))
        var s = Float64(0)
        for c in range(cols):
            s += exp(Float64(x[r * cols + c]) - m)
        for c in _sample_cols(cols):
            var shifted = Float64(x[r * cols + c]) - m
            comptime if logsoftmax:
                assert_almost_equal(
                    Float64(out[r * cols + c]),
                    shifted - log(s),
                    atol=1e-3,
                    msg=String("logsoftmax row ", r, " col ", c),
                )
            else:
                assert_almost_equal(
                    Float64(out[r * cols + c]),
                    exp(shifted) / s,
                    atol=1e-9,
                    rtol=1e-3,
                    msg=String("softmax row ", r, " col ", c),
                )


def test_rms_norm_one_row() raises:
    _check_rms_norm[cols_even](1)


def test_rms_norm_one_row_uneven() raises:
    _check_rms_norm[cols_uneven](1)


def test_rms_norm_two_rows() raises:
    _check_rms_norm[cols_even](2)


def test_softmax_one_row() raises:
    _check_softmax[cols_even](1)


def test_softmax_one_row_uneven() raises:
    _check_softmax[cols_uneven](1)


def test_softmax_two_rows() raises:
    _check_softmax[cols_even](2)


def test_logsoftmax_one_row() raises:
    _check_softmax[cols_even, logsoftmax=True](1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
