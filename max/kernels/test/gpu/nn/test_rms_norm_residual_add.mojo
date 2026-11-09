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
"""GPU test for the rowwise fused residual-add RMSNorm."""

from std.math import sqrt
from std.random import randn, seed
from std.sys import align_of
from std.utils.coord import CoordLike
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from builtin_kernels.rowwise_normalization import rms_norm_residual_add
from layout import ComptimeInt, Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceContext

comptime epsilon = Float32(1e-6)


def _run[
    dtype: DType,
    cols: Int,
    weight_offset: Float32,
    multiply_before_cast: Bool = True,
    exact_rounding: Bool = False,
    cancellation: Bool = False,
](ctx: DeviceContext, rows: Int, axis_size: Some[CoordLike]) raises:
    var n = rows * cols
    var x_h = ctx.enqueue_create_host_buffer[dtype](n)
    var r_h = ctx.enqueue_create_host_buffer[dtype](n)
    var g_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var tmp = ctx.enqueue_create_host_buffer[DType.float32](n)
    seed(5)
    randn(tmp.unsafe_ptr(), n)
    for i in range(n):
        x_h[i] = tmp[i].cast[dtype]()
    randn(tmp.unsafe_ptr(), n)
    for i in range(n):
        r_h[i] = (tmp[i] * 3.0).cast[dtype]()
    for i in range(cols):
        g_h[i] = (Float32(i % 13) * 0.1 + 0.5).cast[dtype]()

    comptime if exact_rounding:
        for i in range(n):
            x_h[i] = Scalar[dtype](i % 4 + 1)
            r_h[i] = 0
    comptime if cancellation:
        for i in range(n):
            x_h[i] = 1
            r_h[i] = 0
        for i in range(cols):
            g_h[i] = Scalar[dtype](-0.1)

    var x_d = ctx.enqueue_create_buffer[dtype](n)
    var r_d = ctx.enqueue_create_buffer[dtype](n)
    var g_d = ctx.enqueue_create_buffer[dtype](cols)
    var out_d = ctx.enqueue_create_buffer[dtype](n)
    var res_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(r_d, r_h)
    ctx.enqueue_copy(g_d, g_h)

    var shape = row_major(rows, Idx[cols])
    var x = TileTensor(x_d, shape)
    var r = TileTensor(r_d, shape)
    var out = TileTensor(out_d, shape)
    var res = TileTensor(res_d, shape)
    var gamma = TileTensor(g_d, row_major[cols]()).as_imm()

    @inline(.always)
    def in0[width: Int, alignment: Int](c: Coord) {var x} -> SIMD[dtype, width]:
        return x.load[width=width, alignment=alignment * align_of[dtype]()](c)

    @inline(.always)
    def in1[width: Int, alignment: Int](c: Coord) {var r} -> SIMD[dtype, width]:
        return r.load[width=width, alignment=alignment * align_of[dtype]()](c)

    @inline(.always)
    def out_fn[
        width: SIMDLength, alignment: Int
    ](c: Coord, v: SIMD[dtype, width]) {var out}:
        out.store[alignment=alignment * align_of[dtype]()](c, v)

    @inline(.always)
    def res_fn[
        width: SIMDLength, alignment: Int
    ](c: Coord, v: SIMD[dtype, width]) {var res}:
        res.store[alignment=alignment * align_of[dtype]()](c, v)

    rms_norm_residual_add[
        dtype,
        2,
        epsilon=epsilon,
        weight_offset=weight_offset,
        multiply_before_cast=multiply_before_cast,
    ](in0, in1, out_fn, res_fn, Coord(rows, Idx[cols]), axis_size, gamma, ctx)

    var out_h = ctx.enqueue_create_host_buffer[dtype](n)
    var res_h = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.enqueue_copy(out_h, out_d)
    ctx.enqueue_copy(res_h, res_d)
    ctx.synchronize()

    var sum_row = List[Scalar[dtype]](length=cols, fill=0)
    var different_rounding = 0
    var offset = weight_offset.cast[dtype]()
    for row in range(rows):
        var ssq = Float32(0)
        for c in range(cols):
            sum_row[c] = x_h[row * cols + c] + r_h[row * cols + c]
            var v = sum_row[c].cast[DType.float32]()
            ssq += v * v
        var inv_rms = 1.0 / sqrt(ssq / Float32(cols) + epsilon)
        for c in range(cols):
            assert_equal(res_h[row * cols + c], sum_row[c])
            var normalized = sum_row[c].cast[DType.float32]() * inv_rms
            var before = (
                normalized
                * (g_h[c].cast[DType.float32]() + offset.cast[DType.float32]())
            ).cast[dtype]()
            var after = normalized.cast[dtype]() * (g_h[c] + offset)
            var expected = before if multiply_before_cast else after
            different_rounding += Int(before != after)
            var got = out_h[row * cols + c]
            comptime if exact_rounding or cancellation:
                assert_equal(got, expected)
            else:
                assert_almost_equal(
                    got.cast[DType.float32](),
                    expected.cast[DType.float32](),
                    rtol=0.01,
                    atol=1e-3,
                )
    comptime if exact_rounding:
        assert_true(different_rounding > 0)


def test_rms_norm_residual_add_bf16() raises:
    with DeviceContext() as ctx:
        _run[DType.bfloat16, 6144, Float32(0)](ctx, 1, ComptimeInt[6144]())
        _run[DType.bfloat16, 6144, Float32(0)](ctx, 64, Int(6144))
        _run[DType.bfloat16, 6144, Float32(0)](ctx, 2048, ComptimeInt[6144]())
        _run[DType.bfloat16, 6144, Float32(1)](ctx, 2500, ComptimeInt[6144]())
        _run[DType.bfloat16, 4096, Float32(0)](ctx, 3001, Int(4096))
        _run[DType.bfloat16, 64, Float32(1), multiply_before_cast=False](
            ctx, 2049, ComptimeInt[64]()
        )


def test_rounding_and_fractional_offset() raises:
    with DeviceContext() as ctx:
        comptime for before in [False, True]:
            _run[
                DType.bfloat16,
                64,
                Float32(0),
                before,
                exact_rounding=True,
            ](ctx, 8, ComptimeInt[64]())
            _run[
                DType.bfloat16,
                64,
                Float32(0),
                before,
                exact_rounding=True,
            ](ctx, 8, Int(64))
            _run[
                DType.bfloat16,
                64,
                Float32(0.1),
                before,
                cancellation=True,
            ](ctx, 8, ComptimeInt[64]())
            _run[
                DType.bfloat16,
                64,
                Float32(0.1),
                before,
                cancellation=True,
            ](ctx, 8, Int(64))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
