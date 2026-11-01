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
"""Direct GPU tests for the tile kernel: ranks 1-4, dtypes, empty output."""

from std.math import nan

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.tile import tile
from std.testing import assert_equal, TestSuite


def test_rank2_basic() raises:
    """[2,3] tiled by [2,3] -> [4,9], cross-axis repeats."""
    comptime dtype = DType.float32
    with DeviceContext() as ctx:
        var input_host_buf = ctx.enqueue_create_host_buffer[dtype](6)
        for i in range(6):
            input_host_buf[i] = Float32(i)

        var input_dev_buf = ctx.enqueue_create_buffer[dtype](6)
        var output_dev_buf = ctx.enqueue_create_buffer[dtype](36)
        ctx.enqueue_copy(input_dev_buf, input_host_buf)
        var input = TileTensor(input_dev_buf, row_major[2, 3]())
        var output = TileTensor(output_dev_buf, row_major[4, 9]())
        var repeats_stack = Array[Scalar[DType.int64], 2](fill={})
        repeats_stack[0] = 2
        repeats_stack[1] = 3
        var repeats = TileTensor(repeats_stack, row_major[2]())

        tile[target="gpu"](
            input.make_dynamic[.int64](),
            repeats.make_dynamic[.int64](),
            output.make_dynamic[.int64](),
            ctx,
        )

        var output_host_buf = ctx.enqueue_create_host_buffer[dtype](36)
        ctx.enqueue_copy(output_host_buf, output_dev_buf)
        ctx.synchronize()
        for row in range(4):
            for col in range(9):
                assert_equal(
                    output_host_buf[row * 9 + col],
                    input_host_buf[(row % 2) * 3 + (col % 3)],
                )


def test_rank1_int32() raises:
    """[5] tiled by [3] -> [15], int32."""
    comptime dtype = DType.int32
    with DeviceContext() as ctx:
        var input_host_buf = ctx.enqueue_create_host_buffer[dtype](5)
        for i in range(5):
            input_host_buf[i] = Int32(i * 7)

        var input_dev_buf = ctx.enqueue_create_buffer[dtype](5)
        var output_dev_buf = ctx.enqueue_create_buffer[dtype](15)
        ctx.enqueue_copy(input_dev_buf, input_host_buf)
        var input = TileTensor(input_dev_buf, row_major[5]())
        var output = TileTensor(output_dev_buf, row_major[15]())
        var repeats_stack = Array[Scalar[DType.int64], 1](fill={})
        repeats_stack[0] = 3
        var repeats = TileTensor(repeats_stack, row_major[1]())

        tile[target="gpu"](
            input.make_dynamic[.int64](),
            repeats.make_dynamic[.int64](),
            output.make_dynamic[.int64](),
            ctx,
        )

        var output_host_buf = ctx.enqueue_create_host_buffer[dtype](15)
        ctx.enqueue_copy(output_host_buf, output_dev_buf)
        ctx.synchronize()
        for i in range(15):
            assert_equal(output_host_buf[i], input_host_buf[i % 5])


def test_rank3_and_rank4() raises:
    """[2,2,3]x[1,2,1] -> [2,4,3] and [2,1,2,1]x[1,2,1,2] -> [2,2,2,2]."""
    comptime dtype = DType.uint8
    with DeviceContext() as ctx:
        var input3_host = ctx.enqueue_create_host_buffer[dtype](12)
        for i in range(12):
            input3_host[i] = UInt8(i)
        var input3_dev = ctx.enqueue_create_buffer[dtype](12)
        var output3_dev = ctx.enqueue_create_buffer[dtype](24)
        ctx.enqueue_copy(input3_dev, input3_host)
        var input3 = TileTensor(input3_dev, row_major[2, 2, 3]())
        var output3 = TileTensor(output3_dev, row_major[2, 4, 3]())
        var repeats3_stack = Array[Scalar[DType.int64], 3](fill={})
        repeats3_stack[0] = 1
        repeats3_stack[1] = 2
        repeats3_stack[2] = 1
        var repeats3 = TileTensor(repeats3_stack, row_major[3]())
        tile[target="gpu"](
            input3.make_dynamic[.int64](),
            repeats3.make_dynamic[.int64](),
            output3.make_dynamic[.int64](),
            ctx,
        )
        var output3_host = ctx.enqueue_create_host_buffer[dtype](24)
        ctx.enqueue_copy(output3_host, output3_dev)
        ctx.synchronize()
        for d in range(2):
            for r in range(4):
                for c in range(3):
                    assert_equal(
                        output3_host[d * 12 + r * 3 + c],
                        input3_host[d * 6 + (r % 2) * 3 + c],
                    )

        var input4_host = ctx.enqueue_create_host_buffer[dtype](4)
        for i in range(4):
            input4_host[i] = UInt8(i + 1)
        var input4_dev = ctx.enqueue_create_buffer[dtype](4)
        var output4_dev = ctx.enqueue_create_buffer[dtype](16)
        ctx.enqueue_copy(input4_dev, input4_host)
        var input4 = TileTensor(input4_dev, row_major[2, 1, 2, 1]())
        var output4 = TileTensor(output4_dev, row_major[2, 2, 2, 2]())
        var repeats4_stack = Array[Scalar[DType.int64], 4](fill={})
        repeats4_stack[0] = 1
        repeats4_stack[1] = 2
        repeats4_stack[2] = 1
        repeats4_stack[3] = 2
        var repeats4 = TileTensor(repeats4_stack, row_major[4]())
        tile[target="gpu"](
            input4.make_dynamic[.int64](),
            repeats4.make_dynamic[.int64](),
            output4.make_dynamic[.int64](),
            ctx,
        )
        var output4_host = ctx.enqueue_create_host_buffer[dtype](16)
        ctx.enqueue_copy(output4_host, output4_dev)
        ctx.synchronize()
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        assert_equal(
                            output4_host[a * 8 + b * 4 + c * 2 + d],
                            input4_host[a * 2 + c],
                        )


def test_identity_repeats() raises:
    """All-ones repeats is a copy."""
    comptime dtype = DType.float32
    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[dtype](6)
        for i in range(6):
            input_host[i] = Float32(i) + 0.5
        var input_dev = ctx.enqueue_create_buffer[dtype](6)
        var output_dev = ctx.enqueue_create_buffer[dtype](6)
        ctx.enqueue_copy(input_dev, input_host)
        var input = TileTensor(input_dev, row_major[2, 3]())
        var output = TileTensor(output_dev, row_major[2, 3]())
        var repeats_stack = Array[Scalar[DType.int64], 2](fill={})
        repeats_stack[0] = 1
        repeats_stack[1] = 1
        var repeats = TileTensor(repeats_stack, row_major[2]())
        tile[target="gpu"](
            input.make_dynamic[.int64](),
            repeats.make_dynamic[.int64](),
            output.make_dynamic[.int64](),
            ctx,
        )
        var output_host = ctx.enqueue_create_host_buffer[dtype](6)
        ctx.enqueue_copy(output_host, output_dev)
        ctx.synchronize()
        for i in range(6):
            assert_equal(output_host[i], input_host[i])


def test_float_bit_patterns() raises:
    """NaN, -0.0, +0.0 tile exactly."""
    comptime dtype = DType.float32
    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[dtype](3)
        input_host[0] = Float32(0.0)
        input_host[1] = Float32(-0.0)
        input_host[2] = nan[.float32]()

        var input_dev = ctx.enqueue_create_buffer[dtype](3)
        var output_dev = ctx.enqueue_create_buffer[dtype](6)
        ctx.enqueue_copy(input_dev, input_host)
        var input = TileTensor(input_dev, row_major[3]())
        var output = TileTensor(output_dev, row_major[6]())
        var repeats_stack = Array[Scalar[DType.int64], 1](fill={})
        repeats_stack[0] = 2
        var repeats = TileTensor(repeats_stack, row_major[1]())
        tile[target="gpu"](
            input.make_dynamic[.int64](),
            repeats.make_dynamic[.int64](),
            output.make_dynamic[.int64](),
            ctx,
        )
        var output_host = ctx.enqueue_create_host_buffer[dtype](6)
        ctx.enqueue_copy(output_host, output_dev)
        ctx.synchronize()
        for i in range(6):
            var got = output_host[i]
            var want = input_host[i % 3]
            if i % 3 == 2:
                assert_equal(got != got, True)
            else:
                # -0.0 vs +0.0 compare equal but differ in sign; check
                # sign through division: -0.0 -> -inf.
                assert_equal(1.0 / got, 1.0 / want)


def test_empty_output() raises:
    """Zero repeats yield an empty output; the launch must short-circuit."""
    comptime dtype = DType.float32
    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[dtype](6)
        for i in range(6):
            input_host[i] = Float32(i)
        var input_dev = ctx.enqueue_create_buffer[dtype](6)
        var output_dev = ctx.enqueue_create_buffer[dtype](1)
        ctx.enqueue_copy(input_dev, input_host)
        var input = TileTensor(input_dev, row_major[2, 3]())
        # Zero repeats on axis 0: output [0, 6]. One-byte allocation backs
        # the empty view.
        var output = TileTensor(output_dev, row_major[0, 6]())
        var repeats_stack = Array[Scalar[DType.int64], 2](fill={})
        repeats_stack[0] = 0
        repeats_stack[1] = 2
        var repeats = TileTensor(repeats_stack, row_major[2]())
        tile[target="gpu"](
            input.make_dynamic[.int64](),
            repeats.make_dynamic[.int64](),
            output.make_dynamic[.int64](),
            ctx,
        )
        ctx.synchronize()
        assert_equal(output.num_elements(), 0)
        print("empty output: no crash, zero elements")


def main() raises:
    var suite = TestSuite()
    suite.test[test_rank2_basic]()
    suite.test[test_rank1_int32]()
    suite.test[test_rank3_and_rank4]()
    suite.test[test_identity_repeats]()
    suite.test[test_float_bit_patterns]()
    suite.test[test_empty_output]()
    suite^.run()
