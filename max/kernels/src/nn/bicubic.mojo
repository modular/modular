# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""This module provides CPU and GPU implementations for bicubic interpolation.

Bicubic interpolation is a 2D extension of cubic interpolation for resampling
digital images. It uses the weighted average of the 4x4 neighborhood of pixels
around the target location to compute the interpolated value.
"""
from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from math import floor, clamp
from memory import UnsafePointer
from sys import exit
from sys.info import has_accelerator
from buffer.dimlist import DimList
from buffer import NDBuffer


fn cubic_kernel(x: Float32) -> Float32:
    """Cubic interpolation kernel with a=-0.5.

    Args:
        x: Distance from the center point.

    Returns:
        Weight contribution based on the distance.
    """
    var a: Float32 = -0.5
    var abs_x = abs(x)
    var abs_x_squared = abs_x * abs_x
    var abs_x_cubed = abs_x_squared * abs_x

    if abs_x <= 1:
        return (a + 2) * abs_x_cubed - (a + 3) * abs_x_squared + 1
    elif abs_x < 2:
        return a * abs_x_cubed - 5 * a * abs_x_squared + 8 * a * abs_x - 4 * a
    else:
        return 0


fn cpu_bicubic_kernel[
    input_dim: DimList,
    output_dim: DimList,
    type: DType,
](
    input_host: NDBuffer[type, 4, StaticConstantOrigin, input_dim],
    output_host: NDBuffer[type, 4, MutableAnyOrigin, output_dim],
) -> NDBuffer[type, 4, MutableAnyOrigin, output_dim]:
    """Perform bicubic interpolation on an NDBuffer of form NCHW.

    Args:
        input_host: Input tensor of shape [B, C, H, W].
        output_host: Output tensor with desired dimensions.

    Returns:
        Interpolated tensor.
    """
    # get dimensions
    var batch_size = input_host.dynamic_shape[0]
    var channels = input_host.dynamic_shape[1]
    var in_height = input_host.dynamic_shape[2]
    var in_width = input_host.dynamic_shape[3]
    var out_height = output_host.dynamic_shape[2]
    var out_width = output_host.dynamic_shape[3]

    var scale_h = Float32(in_height) / Float32(out_height)
    var scale_w = Float32(in_width) / Float32(out_width)

    # each output pixel
    for b in range(batch_size):
        for c in range(channels):
            for y_out in range(out_height):
                for x_out in range(out_width):
                    # get the mapping of where the output pixel lies on input (so if 1d, and scaling from 1-5 to 1-10, x=7 output is mapped to x=3.5 input)
                    var in_y = (Float32(y_out) + 0.5) * scale_h - 0.5
                    var in_x = (Float32(x_out) + 0.5) * scale_w - 0.5

                    # get the pixel righhtttt above and to left of the output pixel
                    var in_y_floor = Int(floor(in_y))
                    var in_x_floor = Int(floor(in_x))

                    # (how far away from the pixel above and to the left is the output pixel?)
                    var dy = in_y - Float32(in_y_floor)
                    var dx = in_x - Float32(in_x_floor)

                    # i want to look at surrounding 4x4 pixels, assign weights to them (closer pixels have more weight) and get final value, for each channel
                    var sum_value: Float32 = 0.0
                    var sum_weights: Float32 = 0.0

                    # get the 4x4 surrounding pixels, and assign weights to them
                    @parameter
                    for i in range(4):

                        @parameter
                        for j in range(4):
                            var y_pos = in_y_floor + i - 1
                            var x_pos = in_x_floor + j - 1

                            # don't be <0 or >frame bounds
                            y_pos = clamp(y_pos, 0, in_height - 1)
                            x_pos = clamp(x_pos, 0, in_width - 1)

                            # SOURCE: https://en.wikipedia.org/wiki/Bicubic_interpolation
                            # NOTE:
                            # This implementation uses the convolution-based bicubic interpolation method (Keys' kernel with a = -0.5),
                            # which is described in the "Bicubic convolution algorithm" section of the Wikipedia article.
                            # It does NOT implement the full spline-based interpolation described earlier in the article,
                            # which requires solving for 16 coefficients using function values and their partial derivatives (f, fx, fy, fxy).
                            # The convolution approach is what most image processing libraries (e.g., OpenCV, PIL, PyTorch) actually use.
                            var weight_y = cubic_kernel(Float32(i) - 1.0 - dy)
                            var weight_x = cubic_kernel(Float32(j) - 1.0 - dx)
                            var weight: Float32 = weight_y * weight_x

                            # now that i have the weight y and x of said pixel, i multiply it by its weight and add it to the sum
                            var pixel_value: Float32 = Float32(
                                input_host[b, c, y_pos, x_pos]
                            )
                            sum_value = sum_value + (pixel_value * weight)
                            sum_weights = sum_weights + weight

                    # normalize if needed
                    if sum_weights > 0:
                        sum_value = sum_value / sum_weights

                    # store the result in the output tensor
                    output_host[b, c, y_out, x_out] = SIMD[type, 1](sum_value)

    return output_host


fn gpu_bicubic_kernel[
    input_dim: DimList,
    output_dim: DimList,
    type: DType,
](
    input: NDBuffer[type, 4, MutableAnyOrigin, input_dim],
    output: NDBuffer[type, 4, MutableAnyOrigin, output_dim],
):
    """Perform bicubic interpolation using GPU.

    Args:
        input: Input tensor of shape [B, C, H, W] on the device.
        output: Output tensor with desired dimensions on the device.
    """
    var b = block_idx.x
    var c = block_idx.y
    var y_out = thread_idx.x
    var x_out = thread_idx.y

    var in_height = input.dynamic_shape[2]
    var in_width = input.dynamic_shape[3]
    var out_height = output.dynamic_shape[2]
    var out_width = output.dynamic_shape[3]

    var scale_h = Float32(in_height) / Float32(out_height)
    var scale_w = Float32(in_width) / Float32(out_width)

    var in_y = (Float32(y_out) + 0.5) * scale_h - 0.5
    var in_x = (Float32(x_out) + 0.5) * scale_w - 0.5

    # get the pixel righhtttt above and to left of the output pixel
    var in_y_floor = Int(floor(in_y))
    var in_x_floor = Int(floor(in_x))

    # (how far away from the pixel above and to the left is the output pixel?)
    var dy = in_y - in_y_floor
    var dx = in_x - in_x_floor

    # i want to look at surrounding 4x4 pixels, assign weights to them (closer pixels have more weight) and get final value, for each channel
    var sum_value: Float32 = 0.0
    var sum_weights: Float32 = 0.0

    # get the 4x4 surrounding pixels, and assign weights to them
    @parameter
    for i in range(4):

        @parameter
        for j in range(4):
            var y_pos = in_y_floor + i - 1
            var x_pos = in_x_floor + j - 1

            # don't be <0 or >frame bounds
            y_pos = clamp(y_pos, 0, in_height - 1)
            x_pos = clamp(x_pos, 0, in_width - 1)

            # get the weight of the surrounding pixel, lifted off wikipedia calc 2 vibes
            var weight_y = cubic_kernel(Float32(i) - 1.0 - dy)
            var weight_x = cubic_kernel(Float32(j) - 1.0 - dx)
            var weight: Float32 = weight_y * weight_x

            # now that i have the weight y and x of said pixel, i multiply it by its weight and add it to the sum
            var pixel_value: Float32 = Float32(input[b, c, y_pos, x_pos])
            sum_value = sum_value + (pixel_value * weight)
            sum_weights = sum_weights + weight

    # normalize if needed
    if sum_weights > 0:
        sum_value = sum_value / sum_weights
    # store the result in the output tensor
    output[b, c, y_out, x_out] = SIMD[type, 1](sum_value)
