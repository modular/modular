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

"""Bodies that call the public `rowwise.reduce` directly, in its stateless and
state-carrying forms, with coords made dynamic by the body. The CPU and GPU
tests run these on their own context and shapes."""

from algorithm import rowwise
from builtin_kernels.monoids import ReduceSum
from layout import Coord
from max.gpu.host import DeviceContext
from std.testing import assert_equal

comptime dtype = DType.float32


def check_sums(ctx: DeviceContext, rows: Int, cols: Int) raises:
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[dtype, 1], type_of(ctx).target, 64, dtype
    ]()
    var n = rows * cols
    var x_h = ctx.enqueue_create_host_buffer[dtype](n)
    var out_h = ctx.enqueue_create_host_buffer[dtype](rows)
    ctx.synchronize()
    for i in range(n):
        x_h[i] = Scalar[dtype](i % 7)
    var x_d = ctx.enqueue_create_buffer[dtype](n)
    var out_d = ctx.enqueue_create_buffer[dtype](rows)
    ctx.enqueue_copy(x_d, x_h)
    # A CPU launch runs on this thread, outside ctx's queue.
    ctx.synchronize()
    var x_ptr = x_d.unsafe_ptr()
    var out_ptr = out_d.unsafe_ptr()

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var cols, var x_ptr, var out_ptr
    }:
        comptime W = params.simd_width
        var coords = row_coords.make_dynamic[DType.int]()
        var state = ReduceSum[dtype, W]()

        @inline(.always)
        def tile_fn[
            ws: Int
        ](mut state: ReduceSum[dtype, W], idx: type_of(coords)) {
            var x_ptr, var cols
        }:
            var i = Int(idx[0].value()) * cols + Int(idx[1].value())
            state.accumulate[dtype, ws](
                x_ptr.unsafe_offset(i).unsafe_load[width=ws]()
            )

        rowwise.reduce(coords, cols, ctx_p, state, tile_fn)
        rowwise.pjoin(state, ctx_p)

        var row = Int(coords[0].value())
        var total = state.acc[0]

        @inline(.always)
        def emit() {var row, var total, var out_ptr}:
            out_ptr.unsafe_offset(row).unsafe_store(total)

        rowwise.once(emit, ctx_p)

    rowwise.launch[axis=1, simd_width=simd_width, num_phases=1](
        body, Coord(rows, cols), ctx
    )
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()
    for r in range(rows):
        var expected = Scalar[dtype](0)
        for c in range(cols):
            expected += Scalar[dtype]((r * cols + c) % 7)
        assert_equal(out_h[r], expected)
    _ = x_d^
    _ = out_d^


def check_doubled(ctx: DeviceContext, rows: Int, cols: Int) raises:
    comptime simd_width = rowwise.pick_simd_width[
        ReduceSum[dtype, 1], type_of(ctx).target, 64, dtype
    ]()
    var n = rows * cols
    var x_h = ctx.enqueue_create_host_buffer[dtype](n)
    var out_h = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.synchronize()
    for i in range(n):
        x_h[i] = Scalar[dtype](i % 7)
    var x_d = ctx.enqueue_create_buffer[dtype](n)
    var out_d = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(x_d, x_h)
    # A CPU launch runs on this thread, outside ctx's queue.
    ctx.synchronize()
    var x_ptr = x_d.unsafe_ptr()
    var out_ptr = out_d.unsafe_ptr()

    @inline(.always)
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut ctx_p: rowwise.Context[params]) {
        var cols, var x_ptr, var out_ptr
    }:
        var coords = row_coords.make_dynamic[DType.int]()

        @inline(.always)
        def tile_fn[
            ws: Int
        ](idx: type_of(coords)) {var x_ptr, var out_ptr, var cols}:
            var i = Int(idx[0].value()) * cols + Int(idx[1].value())
            out_ptr.unsafe_offset(i).unsafe_store[width=ws](
                x_ptr.unsafe_offset(i).unsafe_load[width=ws]() * 2
            )

        rowwise.reduce(coords, cols, ctx_p, tile_fn)

    rowwise.launch[axis=1, simd_width=simd_width, num_phases=1](
        body, Coord(rows, cols), ctx
    )
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()
    for i in range(n):
        assert_equal(out_h[i], Scalar[dtype](i % 7) * 2)
    _ = x_d^
    _ = out_d^
