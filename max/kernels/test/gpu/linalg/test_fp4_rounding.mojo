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
"""Checks portable FP4 rounding against the scalar threshold ladder on GPU."""

from std.math import ceildiv, inf, nan
from std.memory import bitcast
from std.sys import align_of
from std.testing import assert_equal
from std.utils import StaticTuple

from layout import TensorEngine, TensorLayout, TileTensor, row_major
from linalg.fp4_utils import cast_fp_to_fp4e2m1
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from max.gpu.host import DeviceContext


comptime BLOCK_SIZE = 64


def _scalar_round[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    var magnitude = abs(x)
    var rounded: Scalar[dtype]
    if magnitude <= 0.25:
        rounded = 0.0
    elif magnitude < 0.75:
        rounded = 0.5
    elif magnitude <= 1.25:
        rounded = 1.0
    elif magnitude < 1.75:
        rounded = 1.5
    elif magnitude <= 2.5:
        rounded = 2.0
    elif magnitude < 3.5:
        rounded = 3.0
    elif magnitude <= 5.0:
        rounded = 4.0
    else:
        rounded = 6.0
    var sign = Scalar[dtype](-1.0) if x < 0 else Scalar[dtype](1.0)
    return rounded * sign


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def _round_kernel[
    dtype: DType,
    input_layout: TensorLayout,
    output_layout: TensorLayout,
    input_engine: TensorEngine,
    output_engine: TensorEngine,
    width: Int,
](
    input: TileTensor[dtype, input_layout, ImmutAnyOrigin, Engine=input_engine],
    output: TileTensor[
        dtype, output_layout, MutAnyOrigin, Engine=output_engine
    ],
    count: Int32,
):
    comptime assert input.flat_rank == 1 and output.flat_rank == 1
    var size = Int(count)
    var offset = global_idx.x * width
    if offset + width <= size:
        var values = input.load[width=width, alignment=align_of[dtype]()](
            (offset,)
        )
        output.store[alignment=align_of[dtype]()](
            (offset,), cast_fp_to_fp4e2m1(values)
        )
    else:
        for i in range(offset, min(offset + width, size)):
            output.store((i,), cast_fp_to_fp4e2m1(input[i]))


def _run[dtype: DType, width: Int](ctx: DeviceContext) raises:
    comptime bits_dtype = DType.uint32 if dtype == .float32 else DType.uint16
    var values = List[Scalar[dtype]]()
    var midpoints: List[Scalar[dtype]] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    for midpoint in midpoints:
        # Stepping each dtype's bits avoids rounding both neighbors back to
        # the midpoint when float32 neighbors are cast to f16 or bf16.
        var bits = bitcast[bits_dtype](midpoint)
        var below = bitcast[dtype](bits - 1)
        var above = bitcast[dtype](bits + 1)
        values.extend([below, midpoint, above, -below, -midpoint, -above])
    values.extend(
        [
            Scalar[dtype](0.0),
            Scalar[dtype](-0.0),
            inf[dtype](),
            -inf[dtype](),
            nan[dtype](),
            -nan[dtype](),
            Scalar[dtype](0.125),
            Scalar[dtype](-0.125),
            Scalar[dtype](8.0),
            Scalar[dtype](-8.0),
            Scalar[dtype](0.1),
            Scalar[dtype](-0.1),
            Scalar[dtype](2.0),
        ]
    )

    var count = len(values)
    var input_h = ctx.enqueue_create_host_buffer[dtype](count)
    for i in range(count):
        input_h[i] = values[i]
    var input_d = ctx.enqueue_create_buffer[dtype](count)
    var output_d = ctx.enqueue_create_buffer[dtype](count + 8)
    ctx.enqueue_copy(input_d, input_h)
    output_d.enqueue_fill(inf[dtype]())

    var input_t = TileTensor(input_d, row_major(Int32(count)))
    var output_t = TileTensor(output_d, row_major(Int32(count + 8)))
    comptime kernel = _round_kernel[
        dtype,
        type_of(input_t).LayoutType,
        type_of(output_t).LayoutType,
        type_of(input_t).Engine,
        type_of(output_t).Engine,
        width,
    ]
    ctx.enqueue_function[kernel](
        input_t.as_imm().as_unsafe_any_origin(),
        output_t.as_unsafe_any_origin(),
        Int32(count),
        grid_dim=ceildiv(ceildiv(count, width), BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    var output_h = ctx.enqueue_create_host_buffer[dtype](count + 8)
    ctx.enqueue_copy(output_h, output_d)
    ctx.synchronize()

    for i in range(count):
        assert_equal(
            bitcast[bits_dtype](output_h[i]),
            bitcast[bits_dtype](_scalar_round(values[i])),
            msg=String("dtype=", dtype, " width=", width, " index=", i),
        )
    for i in range(count, count + 8):
        assert_equal(output_h[i], inf[dtype]())


def _run_dtype[dtype: DType](ctx: DeviceContext) raises:
    comptime for log_width in range(4):
        _run[dtype, 1 << log_width](ctx)


def main() raises:
    var ctx = DeviceContext()
    _run_dtype[.float32](ctx)
    _run_dtype[.float16](ctx)
    _run_dtype[.bfloat16](ctx)
