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
    resize_linear_gpu,
)
from std.testing import assert_almost_equal, assert_equal, TestSuite


def test_resize_linear_gpu_upsample() raises:
    """Upsamples 2x2 to 4x4 with half_pixel on the GPU.

    Mirrors the CPU test_upsample_sizes_linear fixture: torch reference
    `interpolate(x, (4, 4), mode="bilinear")`. Two axes resize, so the
    kernel runs two passes through a scratch buffer.
    """
    comptime input_layout = row_major[1, 1, 2, 2]()
    comptime output_layout = row_major[1, 1, 4, 4]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](4)
        input_host[0] = Float32(1.0)
        input_host[1] = Float32(2.0)
        input_host[2] = Float32(3.0)
        input_host[3] = Float32(4.0)

        var input_device = ctx.enqueue_create_buffer[.float32](4)
        var output_device = ctx.enqueue_create_buffer[.float32](16)
        ctx.enqueue_copy(input_device, input_host)

        resize_linear_gpu[
            coordinate_transformation_mode=CoordinateTransformationMode.HalfPixel,
            antialias=False,
        ](
            TileTensor(input_device, input_layout),
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](16)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        # torch.nn.functional.interpolate(torch.Tensor([1,2],[3,4]), (4,4))
        var expected = [
            1.0,
            1.25,
            1.75,
            2.0,
            1.5,
            1.75,
            2.25,
            2.5,
            2.5,
            2.75,
            3.25,
            3.5,
            3.0,
            3.25,
            3.75,
            4.0,
        ]
        for i in range(16):
            assert_almost_equal(
                output_host[i], Float32(expected[i]), atol=1e-5, rtol=1e-4
            )


def test_resize_linear_gpu_upsample_float16() raises:
    """Upsamples the 2x2->4x4 fixture at float16.

    The kernel accumulates in float32 and narrows on store, so these
    small-integer fixtures store exactly; a per-tap float16 rounding would
    shift the 0.25-tap values.
    """
    comptime input_layout = row_major[1, 1, 2, 2]()
    comptime output_layout = row_major[1, 1, 4, 4]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float16](4)
        input_host[0] = Float16(1.0)
        input_host[1] = Float16(2.0)
        input_host[2] = Float16(3.0)
        input_host[3] = Float16(4.0)

        var input_device = ctx.enqueue_create_buffer[.float16](4)
        var output_device = ctx.enqueue_create_buffer[.float16](16)
        ctx.enqueue_copy(input_device, input_host)

        resize_linear_gpu[
            coordinate_transformation_mode=CoordinateTransformationMode.HalfPixel,
            antialias=False,
        ](
            TileTensor(input_device, input_layout),
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float16](16)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var expected = [
            1.0,
            1.25,
            1.75,
            2.0,
            1.5,
            1.75,
            2.25,
            2.5,
            2.5,
            2.75,
            3.25,
            3.5,
            3.0,
            3.25,
            3.75,
            4.0,
        ]
        for i in range(16):
            assert_equal(output_host[i], Float16(expected[i]))


def test_resize_linear_gpu_downsample_antialias() raises:
    """Downsamples 4x4 to 2x2 with half_pixel and antialias on the GPU.

    Mirrors the CPU test_downsample_sizes_linear_antialias fixture:
    torch reference `interpolate(x, (2, 2), mode="bilinear",
    antialias=True)`.
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

        resize_linear_gpu[
            coordinate_transformation_mode=CoordinateTransformationMode.HalfPixel,
            antialias=True,
        ](
            TileTensor(input_device, input_layout),
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](4)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var expected = [3.57143, 5.14286, 9.85714, 11.42857]
        for i in range(4):
            assert_almost_equal(
                output_host[i], Float32(expected[i]), atol=1e-4, rtol=1e-4
            )


def test_resize_linear_gpu_two_axis_pass() raises:
    """Resizes two axes at different scales: 2x2 to 4x6.

    The height and width passes run through one scratch buffer in turn;
    a wrong ping-pong order scrambles both axes.
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

        resize_linear_gpu[
            coordinate_transformation_mode=CoordinateTransformationMode.HalfPixel,
            antialias=False,
        ](
            TileTensor(input_device, input_layout),
            TileTensor(output_device, output_layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](24)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        var expected = [
            1.0,
            1.0,
            1.33333,
            1.66667,
            2.0,
            2.0,
            1.5,
            1.5,
            1.83333,
            2.16667,
            2.5,
            2.5,
            2.5,
            2.5,
            2.83333,
            3.16667,
            3.5,
            3.5,
            3.0,
            3.0,
            3.33333,
            3.66667,
            4.0,
            4.0,
        ]
        for i in range(24):
            assert_almost_equal(
                output_host[i], Float32(expected[i]), atol=1e-4, rtol=1e-4
            )


def test_resize_linear_gpu_same_shape_copies() raises:
    """An unchanged shape takes the elementwise copy path, which must write."""

    comptime layout = row_major[1, 2, 3, 4]()

    with DeviceContext() as ctx:
        var input_host = ctx.enqueue_create_host_buffer[.float32](24)
        for i in range(24):
            input_host[i] = Float32(i)

        var input_device = ctx.enqueue_create_buffer[.float32](24)
        var output_device = ctx.enqueue_create_buffer[.float32](24)
        ctx.enqueue_copy(input_device, input_host)

        resize_linear_gpu[
            coordinate_transformation_mode=CoordinateTransformationMode.HalfPixel,
            antialias=False,
        ](
            TileTensor(input_device, layout),
            TileTensor(output_device, layout),
            ctx,
        )

        var output_host = ctx.enqueue_create_host_buffer[.float32](24)
        ctx.enqueue_copy(output_host, output_device)
        ctx.synchronize()

        for i in range(24):
            assert_equal(output_host[i], Float32(i))


def main() raises:
    var suite = TestSuite()
    suite.test[test_resize_linear_gpu_upsample]()
    suite.test[test_resize_linear_gpu_upsample_float16]()
    suite.test[test_resize_linear_gpu_downsample_antialias]()
    suite.test[test_resize_linear_gpu_two_axis_pass]()
    suite.test[test_resize_linear_gpu_same_shape_copies]()
    suite^.run()
