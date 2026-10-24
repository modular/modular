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
from std.memory import bitcast
from std.sys.info import align_of
from std.testing import assert_equal, assert_true
from std.utils import StaticTuple
from std.utils.index import IndexList
from std.utils.numerics import inf, max_finite, min_or_neg_inf, nan, neg_inf

from layout import TileTensor, row_major
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, WARP_SIZE, global_idx
from max.gpu.host import DeviceContext
from nn.attention.mha_utils import _kernel_mask

comptime NUM_CASES = 16


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(WARP_SIZE))
)
def mask_kernel[
    dtype: DType,
    width: Int,
    output_origin: MutOrigin,
    input_origin: ImmOrigin,
    cases_origin: ImmOrigin,
](
    output: TileTensor[
        dtype, type_of(row_major[NUM_CASES, width + 1]()), output_origin
    ],
    input: TileTensor[
        dtype, type_of(row_major[NUM_CASES, width]()), input_origin
    ],
    cases: TileTensor[.int64, type_of(row_major[NUM_CASES, 4]()), cases_origin],
):
    comptime assert output.flat_rank == input.flat_rank == cases.flat_rank == 2
    var case_idx = global_idx.x
    if case_idx >= NUM_CASES:
        return
    var coord = IndexList[2](Int(cases[case_idx, 0]), Int(cases[case_idx, 1]))
    var bound = IndexList[2](Int(cases[case_idx, 2]), Int(cases[case_idx, 3]))
    var values = input.load[width=width]((case_idx, 0))
    output.store[alignment=align_of[Scalar[dtype]]()](
        (case_idx, 0), _kernel_mask(coord, bound, values)
    )


def check_mask[dtype: DType, width: Int](ctx: DeviceContext) raises:
    var cases = [
        [0, 0, 1, width],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 1, width + 1],
        [0, width - 1, 1, width],
        [0, width, 1, width],
        [1, 0, 1, width],
        [2, 0, 1, width],
        [-1, 0, 0, width],
        [-1, 0, -1, width],
        [-2, 0, -1, width],
        [0, -1, 1, 2],
        [0, 4_294_967_294, 1, 3],
        [0, 4_294_967_296, 1, 3],
        [0, 0, 1, -1],
        [0, -1, 1, -1],
    ]
    assert_equal(len(cases), NUM_CASES)
    var input_host = ctx.enqueue_create_host_buffer[dtype](NUM_CASES * width)
    var output_host = ctx.enqueue_create_host_buffer[dtype](
        NUM_CASES * (width + 1)
    )
    var cases_host = ctx.enqueue_create_host_buffer[.int64](NUM_CASES * 4)
    for r in range(NUM_CASES):
        for i in range(4):
            cases_host[r * 4 + i] = Int64(cases[r][i])
        for lane in range(width):
            var value = Scalar[dtype](r * 16 + lane + 1)
            comptime if dtype.is_floating_point():
                if r % 2 == 0:
                    if lane % 4 == 0:
                        value = nan[dtype]()
                    elif lane % 4 == 1:
                        value = inf[dtype]()
                    elif lane % 4 == 2:
                        value = neg_inf[dtype]()
                    else:
                        value = Scalar[dtype](-0.0)
            input_host[r * width + lane] = value
    for i in range(len(output_host)):
        comptime if dtype.is_floating_point():
            output_host[i] = nan[dtype]()
        else:
            output_host[i] = max_finite[dtype]()

    var input_dev = ctx.enqueue_create_buffer[dtype](len(input_host))
    var output_dev = ctx.enqueue_create_buffer[dtype](len(output_host))
    var cases_dev = ctx.enqueue_create_buffer[.int64](len(cases_host))
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(output_dev, output_host)
    ctx.enqueue_copy(cases_dev, cases_host)
    var input = TileTensor(input_dev, row_major[NUM_CASES, width]()).as_imm()
    var output = TileTensor(output_dev, row_major[NUM_CASES, width + 1]())
    var coordinates = TileTensor(cases_dev, row_major[NUM_CASES, 4]()).as_imm()
    comptime kernel = mask_kernel[
        dtype, width, output.origin, input.origin, coordinates.origin
    ]
    ctx.enqueue_function[kernel](
        output,
        input,
        coordinates,
        grid_dim=1,
        block_dim=ctx.default_device_info.warp_size,
    )
    ctx.enqueue_copy(output_host, output_dev)
    ctx.synchronize()

    for r in range(NUM_CASES):
        for lane in range(width):
            var expected = input_host[r * width + lane] if (
                cases[r][0] < cases[r][2]
                and UInt32(cases[r][1]) + UInt32(lane) < UInt32(cases[r][3])
            ) else min_or_neg_inf[dtype]()
            var actual = output_host[r * (width + 1) + lane]
            comptime if dtype.is_floating_point():
                if isnan(expected):
                    assert_true(isnan(actual))
                else:
                    assert_equal(
                        bitcast[DType.uint32](actual.cast[.float32]()),
                        bitcast[DType.uint32](expected.cast[.float32]()),
                    )
            else:
                assert_equal(actual, expected)
        var guard = output_host[r * (width + 1) + width]
        comptime if dtype.is_floating_point():
            assert_true(isnan(guard))
        else:
            assert_equal(guard, max_finite[dtype]())


def main() raises:
    with DeviceContext() as ctx:
        comptime for log_width in range(4):
            check_mask[.float32, 1 << log_width](ctx)
            check_mask[.float16, 1 << log_width](ctx)
            check_mask[.bfloat16, 1 << log_width](ctx)
            check_mask[.int32, 1 << log_width](ctx)
