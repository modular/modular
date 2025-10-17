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
from buffer import DimList
from complex import ComplexFloat32
from gpu.host import DeviceContext
from gpu.host.info import Vendor
from internal_utils import DeviceNDBuffer, HostNDBuffer
from layout import Layout, LayoutTensor, RuntimeLayout
from math import sqrt
from nn.irfft import irfft
from testing import assert_almost_equal

from utils.index import IndexList

alias dtype = DType.float32


fn test_irfft_basic[
    batch_size: Int,
    input_size: Int,  # Size of complex input (number of complex values)
    output_size: Int,  # Size of real output
    dtype: DType = DType.float32,
](ctx: DeviceContext) raises:
    """
    Basic IRFFT test.

    The input is complex data stored as interleaved Float32 values:
    [real0, imag0, real1, imag1, ...]

    The output is real Float32 values.
    """
    print(
        "== test_irfft_basic: batch_size=",
        batch_size,
        ", input_size=",
        input_size,
        ", output_size=",
        output_size,
    )

    # Input shape: [batch_size, input_size*2] because complex values are stored
    # as interleaved float32 (real, imag, real, imag, ...)
    alias input_shape = DimList(batch_size, input_size * 2)
    alias output_shape = DimList(batch_size, output_size)

    # Create host buffers
    var input_host = HostNDBuffer[dtype, 2, input_shape](input_shape)
    var output_host = HostNDBuffer[dtype, 2, output_shape](output_shape)

    # Initialize input with a simple test pattern
    # Set DC component (first complex value) to a known value
    # All other frequencies to zero
    for b in range(batch_size):
        # DC component: real=1.0, imag=0.0
        input_host.tensor[b, 0] = 1.0  # real part
        input_host.tensor[b, 1] = 0.0  # imaginary part

        # Set all other frequencies to zero
        for i in range(1, input_size):
            input_host.tensor[b, 2 * i] = 0.0  # real part
            input_host.tensor[b, 2 * i + 1] = 0.0  # imaginary part

    # Create device buffers
    var input_dev = DeviceNDBuffer[dtype, 2, input_shape](input_shape, ctx=ctx)
    var output_dev = DeviceNDBuffer[dtype, 2, output_shape](
        output_shape, ctx=ctx
    )

    # Copy input to device
    ctx.enqueue_copy(input_dev.buffer, input_host.tensor.data)

    # Create LayoutTensors for the irfft call
    alias layout_2d = Layout.row_major[2]()
    alias alignment = 1

    # Execute IRFFT
    irfft[dtype, dtype, alignment](
        LayoutTensor[dtype, layout_2d, alignment=alignment](
            input_dev.buffer,
            RuntimeLayout[layout_2d].row_major(
                IndexList[2](batch_size, input_size * 2)
            ),
        ),
        LayoutTensor[mut=True, dtype, layout_2d, alignment=alignment](
            output_dev.buffer,
            RuntimeLayout[layout_2d].row_major(
                IndexList[2](batch_size, output_size)
            ),
        ),
        output_size,
        128,  # buffer_size_mb
        ctx,
    )

    # Copy result back to host
    ctx.enqueue_copy(output_host.tensor.data, output_dev.buffer)

    # Verify results
    # For a DC-only signal (frequency = 0), the IRFFT should produce
    # a constant value in all output samples.
    # The expected value depends on normalization, but all samples should be equal
    var first_value = output_host.tensor[0, 0]
    print("First output value:", first_value)

    for b in range(batch_size):
        for i in range(output_size):
            # All output values should be approximately equal for DC-only input
            assert_almost_equal(
                output_host.tensor[b, i],
                first_value,
                rtol=0.01,
                msg="Output values should be constant for DC-only input",
            )

    print("Succeed")

    # Clean up
    _ = input_host
    _ = output_host
    _ = input_dev^
    _ = output_dev^


def main():
    with DeviceContext() as ctx:
        # Check if we're running on an NVIDIA GPU
        if ctx.default_device_info.vendor != Vendor.NVIDIA_GPU:
            print("Skipping cuFFT tests - not running on NVIDIA GPU")
            return

        # Basic tests with different sizes
        test_irfft_basic[batch_size=1, input_size=32, output_size=62](ctx=ctx)

        test_irfft_basic[batch_size=2, input_size=64, output_size=126](ctx=ctx)

        test_irfft_basic[batch_size=4, input_size=128, output_size=254](ctx=ctx)

    # Test with multiple device contexts consecutively
    print("\n== Testing with multiple device contexts ==")

    # First context - default device (GPU 0)
    print("Creating first device context (default device)...")
    with DeviceContext() as ctx1:
        if ctx1.default_device_info.vendor != Vendor.NVIDIA_GPU:
            print("Skipping cuFFT tests - not running on NVIDIA GPU")
            return

        test_irfft_basic[batch_size=1, input_size=32, output_size=62](ctx=ctx1)

    if DeviceContext.number_of_devices() >= 2:
        # Second context - device 1
        print("Creating second device context (device 1)...")
        with DeviceContext(device_id=1) as ctx2:
            if ctx2.default_device_info.vendor != Vendor.NVIDIA_GPU:
                print(
                    "Skipping cuFFT tests on device 1 - not running on NVIDIA"
                    " GPU"
                )
                return

            test_irfft_basic[batch_size=1, input_size=32, output_size=62](
                ctx=ctx2
            )

        print("Multiple device context test completed successfully!")
