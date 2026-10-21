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
"""Row-wise softmax and logsoftmax on the GPU tiers that write one element at
a time: the per-element split-K tier (few long rows) and ragged widths on the
block tier. A store whose alignment is under-claimed splits a bf16 element into
two byte stores, which some NVIDIA drivers corrupt."""

from std.math import exp, log
from std.random import rand, seed
from std.sys import align_of

from builtin_kernels.rowwise_softmax import softmax
from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from std.testing import assert_almost_equal


def _check[
    dtype: DType, logsoftmax: Bool = False
](ctx: DeviceContext, rows: Int, cols: Int) raises:
    var n = rows * cols
    var data_h = ctx.enqueue_create_host_buffer[dtype](n)
    var res = ctx.enqueue_create_host_buffer[dtype](n)
    rand[dtype](data_h.as_span())
    var data_d = ctx.enqueue_create_buffer[dtype](n)
    var out_d = ctx.enqueue_create_buffer[dtype](n)
    var data_buf = TileTensor(data_d, row_major((rows, cols)))
    var out_t = TileTensor(out_d, row_major((rows, cols)))
    ctx.enqueue_copy(data_d, data_h)

    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        # `alignment` counts elements; `load` takes bytes.
        return data_buf.load[
            width=width, alignment=alignment * align_of[dtype]()
        ](coords)

    softmax[dtype, 2, target="gpu", logsoftmax=logsoftmax](
        input_fn, Coord(rows, cols), cols, out_t, 1, ctx
    )
    ctx.enqueue_copy(res, out_d)
    ctx.synchronize()

    comptime rtol = 1e-4 if dtype == DType.float32 else 1e-2
    for r in range(rows):
        var m = Float64(data_h[r * cols])
        for c in range(cols):
            m = max(m, Float64(data_h[r * cols + c]))
        var s = Float64(0)
        for c in range(cols):
            s += exp(Float64(data_h[r * cols + c]) - m)
        for c in range(cols):
            var i = r * cols + c
            var shifted = Float64(data_h[i]) - m
            comptime if logsoftmax:
                assert_almost_equal(
                    Float64(res[i]),
                    shifted - log(s),
                    atol=2e-2,
                    rtol=rtol,
                    msg=String("logsoftmax row ", r, " col ", c),
                )
            else:
                assert_almost_equal(
                    Float64(res[i]),
                    exp(shifted) / s,
                    atol=1e-9,
                    rtol=rtol,
                    msg=String("softmax row ", r, " col ", c),
                )
    _ = data_d
    _ = out_d


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        # Per-element split-K tier: at most 16 rows of at least 224 KB each.
        _check[DType.bfloat16](ctx, 1, 262144)
        _check[DType.bfloat16](ctx, 16, 131072)
        _check[DType.bfloat16](ctx, 3, 200003)
        _check[DType.bfloat16, logsoftmax=True](ctx, 2, 200003)
        _check[DType.float32](ctx, 3, 100003)
        # Block tier with a ragged width: every element is written alone.
        _check[DType.bfloat16](ctx, 16, 65537)
        _check[DType.bfloat16, logsoftmax=True](ctx, 64, 4099)
        # Block tier with an even width: vector writes.
        _check[DType.bfloat16](ctx, 32, 4096)
        _check[DType.float32](ctx, 32, 4096)
