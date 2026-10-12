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

from std.math import isnan
from std.random import random_float64, seed

from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.cumsum import cumsum
from std.testing import assert_almost_equal, assert_equal, assert_true
from std.utils import IndexList
from std.utils.numerics import max_finite, nan

# Elements after the output that the kernel must not touch.
comptime _GUARD = 64


def _check[
    dtype: DType,
    axis: Int,
    exclusive: Bool,
    reverse: Bool,
](
    ctx: DeviceContext,
    shape: IndexList,
    integral_values: Bool,
    zero_input: Bool,
) raises:
    var layout = row_major(Coord(shape))
    var n = shape.flattened_length()

    var input_host = ctx.enqueue_create_host_buffer[dtype](n)
    for i in range(n):
        # Small integers keep every partial sum exact, so the GPU result must
        # match the CPU result bit for bit whatever order it adds in.
        var v = random_float64(-4, 4)
        if integral_values:
            v = v.__round__()
        input_host[i] = Scalar[dtype](0) if zero_input else v.cast[dtype]()

    var input_dev = ctx.enqueue_create_buffer[dtype](n)
    # Poison the output and the guard band after it with a value the kernel
    # cannot produce, so a skipped or stray write cannot pass. Integers have
    # no such value in general, so the zero-input runs back this up: every
    # correct output there is zero.
    var poison: Scalar[dtype]
    comptime if dtype.is_floating_point():
        poison = nan[dtype]()
    else:
        poison = max_finite[dtype]()
    var output_dev = ctx.enqueue_create_buffer[dtype](n + _GUARD)
    output_dev.enqueue_fill(poison)
    ctx.enqueue_copy(input_dev, input_host)

    cumsum[dtype, exclusive, reverse, axis=axis, target="gpu"](
        TileTensor(output_dev, layout), TileTensor(input_dev, layout), ctx
    )

    var output_host = ctx.enqueue_create_host_buffer[dtype](n + _GUARD)
    ctx.enqueue_copy(output_host, output_dev)
    var expected_host = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.synchronize()

    cumsum[dtype, exclusive, reverse, axis=axis](
        TileTensor(expected_host, layout), TileTensor(input_host, layout)
    )

    for i in range(n, n + _GUARD):
        comptime if dtype.is_floating_point():
            assert_true(isnan(output_host[i]), msg="guard band written")
        else:
            assert_equal(output_host[i], poison, msg="guard band written")

    for i in range(n):
        var msg = String(
            t"{dtype} shape={shape} axis={axis} exclusive={exclusive}"
            t" reverse={reverse} at flat index {i}"
        )
        comptime if dtype.is_integral():
            assert_equal(output_host[i], expected_host[i], msg=msg)
        else:
            if integral_values:
                assert_equal(output_host[i], expected_host[i], msg=msg)
            else:
                assert_almost_equal(
                    output_host[i].cast[.float64](),
                    expected_host[i].cast[.float64](),
                    atol=1e-3,
                    rtol=1e-4,
                    msg=msg,
                )


def _check_modes[
    dtype: DType, axis: Int
](
    ctx: DeviceContext,
    shape: IndexList,
    integral_values: Bool = True,
    zero_input: Bool = False,
) raises:
    _check[dtype, axis, False, False](ctx, shape, integral_values, zero_input)
    _check[dtype, axis, True, False](ctx, shape, integral_values, zero_input)
    _check[dtype, axis, False, True](ctx, shape, integral_values, zero_input)
    _check[dtype, axis, True, True](ctx, shape, integral_values, zero_input)


def _check_shapes[dtype: DType](ctx: DeviceContext, zero_input: Bool) raises:
    # Short rows that one thread per row scans whole.
    _check_modes[dtype, 0](ctx, IndexList[1](13), zero_input=zero_input)
    _check_modes[dtype, 1](ctx, IndexList[2](37, 63), zero_input=zero_input)
    # Row scan: short rows too few to fill the GPU one thread each, one tile,
    # a partial tile, several tiles with a ragged tail.
    _check_modes[dtype, 0](ctx, IndexList[1](100), zero_input=zero_input)
    _check_modes[dtype, 1](ctx, IndexList[2](37, 255), zero_input=zero_input)
    _check_modes[dtype, 0](ctx, IndexList[1](256), zero_input=zero_input)
    _check_modes[dtype, -1](ctx, IndexList[2](5, 1023), zero_input=zero_input)
    _check_modes[dtype, 1](ctx, IndexList[2](3, 5000), zero_input=zero_input)
    # Strided: the scanned axis is not innermost.
    _check_modes[dtype, 0](ctx, IndexList[2](300, 65), zero_input=zero_input)
    _check_modes[dtype, 1](ctx, IndexList[3](3, 301, 7), zero_input=zero_input)
    _check_modes[dtype, 0](ctx, IndexList[3](9, 4, 5), zero_input=zero_input)
    # Empty.
    _check_modes[dtype, 1](ctx, IndexList[2](0, 300), zero_input=zero_input)
    # Long axes with too few lines to fill the GPU are split into chunks:
    # contiguous rows, then strided lines. Lengths are prime-ish so the last
    # chunk is ragged.
    _check_modes[dtype, 1](
        ctx, IndexList[2](1, 3_000_017), zero_input=zero_input
    )
    _check_modes[dtype, -1](
        ctx, IndexList[2](2, 1_048_583), zero_input=zero_input
    )
    _check_modes[dtype, 1](
        ctx, IndexList[3](3, 200_003, 2), zero_input=zero_input
    )
    # Split lines whose per-chunk carries are themselves scanned by the
    # row-scan kernel, with fewer and with `_CUMSUM_MIN_ROW_SCAN_LEN` chunks.
    _check_modes[dtype, 1](ctx, IndexList[3](8, 3000, 8), zero_input=zero_input)
    _check_modes[dtype, 1](ctx, IndexList[3](8, 4096, 8), zero_input=zero_input)
    _check_modes[dtype, 0](ctx, IndexList[2](262_147, 3), zero_input=zero_input)


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        _check_shapes[.float32](ctx, False)
        _check_shapes[.bfloat16](ctx, False)
        _check_shapes[.int32](ctx, False)
        _check_shapes[.int64](ctx, False)
        _check_shapes[.int8](ctx, False)
        _check_shapes[.uint16](ctx, False)
        _check_shapes[.float16](ctx, False)
        comptime if not ctx.target.is_apple_gpu():
            # Metal does not support DType.float64
            _check_shapes[.float64](ctx, False)
        # With an all-zero input every correct output is zero, so a skipped
        # write fails even for integer types that have no unreachable poison.
        _check_shapes[.float32](ctx, True)
        _check_shapes[.int32](ctx, True)
        _check_shapes[.int8](ctx, True)
        # Fractional values: float32 accumulates in float32 on GPU and in
        # float64 on CPU, so allow rounding differences.
        _check_modes[.float32, 1](
            ctx, IndexList[2](4, 262_144), integral_values=False
        )
        _check_modes[.float32, 0](
            ctx, IndexList[2](2000, 33), integral_values=False
        )
        _check_modes[.float32, 1](
            ctx, IndexList[2](1, 1_048_583), integral_values=False
        )
        _check_modes[.float32, 0](
            ctx, IndexList[2](262_147, 3), integral_values=False
        )
