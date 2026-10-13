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
from nn.resize import (
    CoordinateTransformationMode,
    RoundMode,
    resize_nearest_neighbor,
)
from std.testing import assert_equal, TestSuite


def test_resize_nearest_gpu_upsample() raises:
    """Upsamples 2x2 to 4x6 with half_pixel/HalfDown on the GPU.

    Mirrors the CPU test_upsample_sizes_nearest_1 fixture: each input
    element expands into a 2x3 block of output elements.
    """
    comptime input_layout = row_major[1, 1, 2, 2]()
    comptime output_layout = row_major[1, 1, 4, 6]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](4)
        input_host[0] = Float32(1.0)
        input_host[1] = Float32(2.0)
        input_host[2] = Float32(3.0)
        input_host[3] = Float32(4.0)

        var input_device = ctx.enqueue_create_buffer[.float32](4)
        var output_device = ctx.enqueue_create_buffer[.float32](24)
        ctx.enqueue_copy(input_device, input_host)

        resize_nearest_neighbor[
            coordinate_transformation_mode=CoordinateTransformationMode.HalfPixel,
            round_mode=RoundMode.HalfDown,
            target="gpu",
        ](
            TileTensor(input_device, input_layout),
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](24)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        # Each input element repeats 2x3: 1,1,1,2,2,2 per row pair.
        var expected = [
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
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
            3.0,
            3.0,
            3.0,
            4.0,
            4.0,
            4.0,
        ]
        for i in range(24):
            assert_equal(output_host[i], Float32(expected[i]))


def test_resize_nearest_gpu_downsample() raises:
    """Downsamples 4x4 to 2x2 with asymmetric/Floor on the GPU.

    Asymmetric maps out coord / scale, so the 2x2 output picks input
    rows/cols 0 and 2.
    """
    comptime input_layout = row_major[1, 1, 4, 4]()
    comptime output_layout = row_major[1, 1, 2, 2]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](16)
        for i in range(16):
            input_host[i] = Float32(i)

        var input_device = ctx.enqueue_create_buffer[.float32](16)
        var output_device = ctx.enqueue_create_buffer[.float32](4)
        ctx.enqueue_copy(input_device, input_host)

        resize_nearest_neighbor[
            coordinate_transformation_mode=CoordinateTransformationMode.Asymmetric,
            round_mode=RoundMode.Floor,
            target="gpu",
        ](
            TileTensor(input_device, input_layout),
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](4)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        # Row/col 0 and 2 of a 4x4 arange: values 0, 2, 8, 10.
        var expected = [0.0, 2.0, 8.0, 10.0]
        for i in range(4):
            assert_equal(output_host[i], Float32(expected[i]))


def main() raises:
    var suite = TestSuite()
    suite.test[test_resize_nearest_gpu_upsample]()
    suite.test[test_resize_nearest_gpu_downsample]()
    suite^.run()
