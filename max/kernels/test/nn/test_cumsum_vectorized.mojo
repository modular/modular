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

from std.sys.info import simd_width_of
from std.testing import assert_equal
from std.utils.numerics import get_accum_type

from layout import TileTensor, row_major
from nn.cumsum import cumsum


def run_cumsum[
    dtype: DType, exclusive: Bool, reverse: Bool
](trailing: Int, depth: Int = 5) raises:
    comptime leading = 2
    comptime accum_type = DType.float64 if dtype == .float32 else get_accum_type[
        dtype
    ]()
    var size = leading * depth * trailing
    var input_data = List[Scalar[dtype]](length=size, fill=0)
    var output_data = List[Scalar[dtype]](length=size, fill=0)
    for i in range(size):
        input_data[i] = Scalar[dtype](i % 19 - 9)
    # Cancellation distinguishes the original sequential scan from a reassociated one.
    comptime if dtype.is_floating_point():
        comptime magnitude = 9_007_199_254_740_992 if accum_type == .float64 else 16_777_216
        if depth >= 3:
            for n in range(leading):
                for c in range(trailing):
                    var start = n * depth * trailing + c
                    input_data[start] = Scalar[dtype](magnitude)
                    input_data[start + trailing] = Scalar[dtype](0.125)
                    input_data[start + 2 * trailing] = Scalar[dtype](-magnitude)

    var input = TileTensor(input_data, row_major((leading, depth, trailing)))
    var output = TileTensor(output_data, row_major((leading, depth, trailing)))
    cumsum[dtype, exclusive, reverse, axis=-2](output, input)

    for n in range(leading):
        for c in range(trailing):
            var accumulator = Scalar[accum_type](0)
            for d in range(depth):
                var scan_d = depth - 1 - d if reverse else d
                var offset = n * depth * trailing + scan_d * trailing + c
                comptime if exclusive:
                    assert_equal(output_data[offset], accumulator.cast[dtype]())
                    accumulator += input_data[offset].cast[accum_type]()
                else:
                    accumulator += input_data[offset].cast[accum_type]()
                    assert_equal(output_data[offset], accumulator.cast[dtype]())


def run_widths[dtype: DType]() raises:
    comptime accum_type = DType.float64 if dtype == .float32 else get_accum_type[
        dtype
    ]()
    comptime width = simd_width_of[accum_type]()
    for trailing in [0, 1, width - 1, width, width + 1, 2 * width + 1]:
        run_cumsum[dtype, False, False](trailing)
        run_cumsum[dtype, False, True](trailing)
        run_cumsum[dtype, True, False](trailing)
        run_cumsum[dtype, True, True](trailing)
    run_cumsum[dtype, False, False](width + 1, depth=0)
    run_cumsum[dtype, False, True](width + 1, depth=0)
    run_cumsum[dtype, True, False](width + 1, depth=0)
    run_cumsum[dtype, True, True](width + 1, depth=0)


def main() raises:
    run_widths[.float32]()
    run_widths[.float64]()
    run_widths[.bfloat16]()
    run_widths[.int32]()
