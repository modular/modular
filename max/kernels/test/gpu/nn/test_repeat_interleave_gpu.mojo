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

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.repeat_interleave import _repeat_interleave_gpu
from std.testing import assert_equal, TestSuite


def test_repeat_interleave_gpu_rank1() raises:
    """Scalar repeats expand each element into consecutive output slots.

    [1, 2, 3] with repeats=2 gives [1, 1, 2, 2, 3, 3]: every input
    element maps to `repeats` consecutive output positions, so the
    source index decodes as output_index // repeats.
    """
    comptime input_layout = row_major[3]()
    comptime output_layout = row_major[6]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](3)
        input_host[0] = Float32(1.0)
        input_host[1] = Float32(2.0)
        input_host[2] = Float32(3.0)
        # The repeats buffer mirrors the graph path: a size-1 int64
        # buffer on the input's device.
        var repeats_host = ctx.enqueue_create_host_buffer[.int64](1)
        repeats_host[0] = 2

        var input_device = ctx.enqueue_create_buffer[.float32](3)
        var output_device = ctx.enqueue_create_buffer[.float32](6)
        var repeats_device = ctx.enqueue_create_buffer[.int64](1)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(repeats_device, repeats_host)

        _repeat_interleave_gpu[target="gpu"](
            TileTensor(input_device, input_layout),
            TileTensor(repeats_device, row_major(1)),
            0,
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](6)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var values = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        for i in range(6):
            assert_equal(output_host[i], Float32(values[i]))


def test_repeat_interleave_gpu_rank2_axis1() raises:
    """Rank-2 input repeats along the last axis; other axes untouched.

    [[1, 2], [3, 4]] with repeats=3, axis=1 gives
    [[1,1,1,2,2,2],[3,3,3,4,4,4]].
    """
    comptime input_layout = row_major[2, 2]()
    comptime output_layout = row_major[2, 6]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](4)
        input_host[0] = Float32(1.0)
        input_host[1] = Float32(2.0)
        input_host[2] = Float32(3.0)
        input_host[3] = Float32(4.0)
        var repeats_host = ctx.enqueue_create_host_buffer[.int64](1)
        repeats_host[0] = 3

        var input_device = ctx.enqueue_create_buffer[.float32](4)
        var output_device = ctx.enqueue_create_buffer[.float32](12)
        var repeats_device = ctx.enqueue_create_buffer[.int64](1)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(repeats_device, repeats_host)

        _repeat_interleave_gpu[target="gpu"](
            TileTensor(input_device, input_layout),
            TileTensor(repeats_device, row_major(1)),
            1,
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](12)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var values = [
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            3.0,
            3.0,
            3.0,
            4.0,
            4.0,
            4.0,
        ]
        for i in range(12):
            assert_equal(output_host[i], Float32(values[i]))


def test_repeat_interleave_gpu_tensor_repeats() raises:
    """Per-element repeats build the offset map on the device.

    [10, 20, 30] with repeats [2, 0, 3] gives [10, 10, 30, 30, 30]:
    a zero repeat drops the middle element, and the map kernel's
    exclusive prefix places each element's span without reading any
    repeats value on the host.
    """
    comptime input_layout = row_major[3]()
    comptime output_layout = row_major[5]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](3)
        input_host[0] = Float32(10.0)
        input_host[1] = Float32(20.0)
        input_host[2] = Float32(30.0)
        var repeats_host = ctx.enqueue_create_host_buffer[.int64](3)
        repeats_host[0] = 2
        repeats_host[1] = 0
        repeats_host[2] = 3

        var input_device = ctx.enqueue_create_buffer[.float32](3)
        var output_device = ctx.enqueue_create_buffer[.float32](5)
        var repeats_device = ctx.enqueue_create_buffer[.int64](3)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(repeats_device, repeats_host)

        _repeat_interleave_gpu[target="gpu"](
            TileTensor(input_device, input_layout),
            TileTensor(repeats_device, row_major(3)),
            0,
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](5)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var values = [10.0, 10.0, 30.0, 30.0, 30.0]
        for i in range(5):
            assert_equal(output_host[i], Float32(values[i]))


def test_repeat_interleave_gpu_tensor_repeats_axis1() raises:
    """Per-element repeats along the last axis of a rank-2 input.

    [[1, 2], [3, 4]] with repeats [1, 2], axis=1 gives
    [[1, 2, 2], [3, 4, 4]].
    """
    comptime input_layout = row_major[2, 2]()
    comptime output_layout = row_major[2, 3]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](4)
        input_host[0] = Float32(1.0)
        input_host[1] = Float32(2.0)
        input_host[2] = Float32(3.0)
        input_host[3] = Float32(4.0)
        var repeats_host = ctx.enqueue_create_host_buffer[.int64](2)
        repeats_host[0] = 1
        repeats_host[1] = 2

        var input_device = ctx.enqueue_create_buffer[.float32](4)
        var output_device = ctx.enqueue_create_buffer[.float32](6)
        var repeats_device = ctx.enqueue_create_buffer[.int64](2)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(repeats_device, repeats_host)

        _repeat_interleave_gpu[target="gpu"](
            TileTensor(input_device, input_layout),
            TileTensor(repeats_device, row_major(2)),
            1,
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](6)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var values = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0]
        for i in range(6):
            assert_equal(output_host[i], Float32(values[i]))


@inline(.always)
def _expected_total(numel: Int) -> Int:
    var total = 0
    for i in range(numel):
        total += i % 3
    return total


def test_repeat_interleave_gpu_large_axis_tensor_repeats() raises:
    """Per-element repeats over an axis larger than one block.

    700 elements with repeats cycling 0,1,2: the map kernel's chunked
    scan must keep the running base exact across the 256-lane chunks.
    The expected output is [0, 1, 1, 2, 2, 2, ...]: element i repeats
    (i % 3) times.
    """
    comptime numel = 700
    comptime expected_total = _expected_total(numel)
    comptime input_layout = row_major[numel]()
    comptime output_layout = row_major[expected_total]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.int32](numel)
        var repeats_host = ctx.enqueue_create_host_buffer[.int64](numel)
        for i in range(numel):
            input_host[i] = Int32(i)
            repeats_host[i] = Int64(i % 3)

        var input_device = ctx.enqueue_create_buffer[.int32](numel)
        var output_device = ctx.enqueue_create_buffer[.int32](expected_total)
        var repeats_device = ctx.enqueue_create_buffer[.int64](numel)
        ctx.enqueue_copy(input_device, input_host)
        ctx.enqueue_copy(repeats_device, repeats_host)

        _repeat_interleave_gpu[target="gpu"](
            TileTensor(input_device, input_layout),
            TileTensor(repeats_device, row_major(numel)),
            0,
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.int32](expected_total)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var out_i = 0
        for i in range(numel):
            for _ in range(i % 3):
                assert_equal(Int(output_host[out_i]), i)
                out_i += 1
        assert_equal(out_i, expected_total)


def main() raises:
    var suite = TestSuite()
    suite.test[test_repeat_interleave_gpu_rank1]()
    suite.test[test_repeat_interleave_gpu_rank2_axis1]()
    suite.test[test_repeat_interleave_gpu_tensor_repeats]()
    suite.test[test_repeat_interleave_gpu_tensor_repeats_axis1]()
    suite.test[test_repeat_interleave_gpu_large_axis_tensor_repeats]()
    suite^.run()
