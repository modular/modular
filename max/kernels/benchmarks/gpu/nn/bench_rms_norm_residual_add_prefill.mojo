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
"""Fused residual-add RMSNorm at prefill sizes: block-per-row kernel vs the
rowwise kernel, with an output comparison.

Prints microseconds per launch and the
number of output elements that differ between the two kernels.
"""

from std.random import randn, seed
from std.sys import align_of, get_defined_int
from std.utils.coord import CoordLike

from builtin_kernels.rowwise_normalization import rms_norm_residual_add
from layout import ComptimeInt, Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceContext
from state_space.rms_norm_fused_residual import rms_norm_fused_residual

comptime dtype = DType.bfloat16
comptime COLS = get_defined_int["COLS", 6144]()
comptime ITERS = get_defined_int["ITERS", 50]()
comptime EPS = Float32(1e-6)


def run[ROWS: Int](ctx: DeviceContext, axis_size: Some[CoordLike]) raises:
    var n = ROWS * COLS
    var x_h = ctx.enqueue_create_host_buffer[dtype](n)
    var r_h = ctx.enqueue_create_host_buffer[dtype](n)
    var g_h = ctx.enqueue_create_host_buffer[dtype](COLS)
    var tmp = ctx.enqueue_create_host_buffer[DType.float32](n)
    seed(3)
    randn(tmp.unsafe_ptr(), n)
    for i in range(n):
        x_h[i] = tmp[i].cast[dtype]()
    randn(tmp.unsafe_ptr(), n)
    for i in range(n):
        r_h[i] = (tmp[i] * 4.0).cast[dtype]()
    for i in range(COLS):
        g_h[i] = (Float32(i % 17) * 0.1 + 0.5).cast[dtype]()

    var x_d = ctx.enqueue_create_buffer[dtype](n)
    var r_d = ctx.enqueue_create_buffer[dtype](n)
    var g_d = ctx.enqueue_create_buffer[dtype](COLS)
    var out_a = ctx.enqueue_create_buffer[dtype](n)
    var res_a = ctx.enqueue_create_buffer[dtype](n)
    var out_b = ctx.enqueue_create_buffer[dtype](n)
    var res_b = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(r_d, r_h)
    ctx.enqueue_copy(g_d, g_h)

    var shape = row_major(ROWS, Idx[COLS])
    var x = TileTensor(x_d, shape)
    var r = TileTensor(r_d, shape)
    var oa = TileTensor(out_a, shape)
    var ra = TileTensor(res_a, shape)
    var ob = TileTensor(out_b, shape)
    var rb = TileTensor(res_b, shape)
    var gamma = TileTensor(g_d, row_major[COLS]()).as_imm()

    @inline(.always)
    def in0[width: Int](c: Coord) {var x} -> SIMD[dtype, width]:
        return x.load[width=width](c)

    @inline(.always)
    def in1[width: Int](c: Coord) {var r} -> SIMD[dtype, width]:
        return r.load[width=width](c)

    @inline(.always)
    def in0a[
        width: Int, alignment: Int
    ](c: Coord) {var x} -> SIMD[dtype, width]:
        return x.load[width=width, alignment=alignment * align_of[dtype]()](c)

    @inline(.always)
    def in1a[
        width: Int, alignment: Int
    ](c: Coord) {var r} -> SIMD[dtype, width]:
        return r.load[width=width, alignment=alignment * align_of[dtype]()](c)

    @inline(.always)
    def out_a_fn[
        width: SIMDLength, alignment: Int
    ](c: Coord, v: SIMD[dtype, width]) {var oa}:
        oa.store[alignment=alignment * align_of[dtype]()](c, v)

    @inline(.always)
    def res_a_fn[
        width: SIMDLength, alignment: Int
    ](c: Coord, v: SIMD[dtype, width]) {var ra}:
        ra.store[alignment=alignment * align_of[dtype]()](c, v)

    @inline(.always)
    def out_b_fn[
        width: SIMDLength, alignment: Int
    ](c: Coord, v: SIMD[dtype, width]) {var ob}:
        ob.store[alignment=alignment * align_of[dtype]()](c, v)

    @inline(.always)
    def res_b_fn[
        width: SIMDLength, alignment: Int
    ](c: Coord, v: SIMD[dtype, width]) {var rb}:
        rb.store[alignment=alignment * align_of[dtype]()](c, v)

    def old_kernel() raises {imm}:
        rms_norm_fused_residual(
            in0,
            in1,
            out_a_fn,
            res_a_fn,
            Coord(ROWS, Idx[COLS]),
            gamma,
            EPS,
            Scalar[dtype](0),
            ctx,
        )

    def new_kernel() raises {imm}:
        rms_norm_residual_add[
            dtype,
            2,
            epsilon=EPS,
            weight_offset=Float32(0),
        ](
            in0a,
            in1a,
            out_b_fn,
            res_b_fn,
            Coord(ROWS, Idx[COLS]),
            axis_size,
            gamma,
            ctx,
        )

    old_kernel()
    new_kernel()
    ctx.synchronize()
    var t_old = (
        Float64(ctx.execution_time(old_kernel, ITERS)) / 1e3 / Float64(ITERS)
    )
    var t_new = (
        Float64(ctx.execution_time(new_kernel, ITERS)) / 1e3 / Float64(ITERS)
    )

    var oa_h = ctx.enqueue_create_host_buffer[dtype](n)
    var ob_h = ctx.enqueue_create_host_buffer[dtype](n)
    var ra_h = ctx.enqueue_create_host_buffer[dtype](n)
    var rb_h = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.enqueue_copy(oa_h, out_a)
    ctx.enqueue_copy(ob_h, out_b)
    ctx.enqueue_copy(ra_h, res_a)
    ctx.enqueue_copy(rb_h, res_b)
    ctx.synchronize()
    var diff_out = 0
    var diff_res = 0
    for i in range(n):
        if oa_h[i] != ob_h[i]:
            diff_out += 1
        if ra_h[i] != rb_h[i]:
            diff_res += 1
    var bytes = Float64(4 * n * 2)
    print(
        "rows=" + String(ROWS),
        "cols=" + String(COLS),
        "static_width=" + String(type_of(axis_size).is_static_value),
        "old_us=" + String(t_old),
        "new_us=" + String(t_new),
        "old_GBps=" + String(bytes / t_old / 1e3),
        "new_GBps=" + String(bytes / t_new / 1e3),
        "diff_out=" + String(diff_out),
        "diff_residual=" + String(diff_res),
    )


def main() raises:
    with DeviceContext() as ctx:
        comptime for rows in [16384, 15000, 8192, 2048, 512, 64, 8]:
            run[rows](ctx, ComptimeInt[COLS]())
            run[rows](ctx, Int(COLS))
