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
"""Inverse real FFT kernel using cuFFT."""

from buffer.buffer import NDBuffer
from complex import ComplexFloat32
from gpu._cufft.cufft import (
    cufftDestroy,
    cufftExecC2R,
    cufftHandle,
    cufftPlan1d,
    cufftSetStream,
)
from gpu._cufft.types import Status, Type
from gpu._cufft.utils import check_error
from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import CUDA
from memory import UnsafePointer


fn irfft[
    input_rank: Int,
    input_type: DType,
    output_type: DType,
](
    input: NDBuffer[input_type, input_rank],
    output: NDBuffer[mut=True, output_type, input_rank],
    n: Int,
    ctx: DeviceContext,
) raises:
    """Compute the inverse real FFT of the input tensor.

    Currently, only applies it to the last dimension.

    Args:
        input: Complex input tensor (NDBuffer).
        output: Real output tensor (NDBuffer).
        n: Output signal size (if <= 0, computed as 2*(input.size(axis) - 1)).
        ctx: Device context.
    """
    constrained[
        input_type is DType.float32, "Only Float32 is supported for IRFFT"
    ]()
    constrained[
        output_type is DType.float32, "Only Float32 is supported for IRFFT"
    ]()
    axis = input_rank - 1

    # Get input and output dimensions
    input_shape = input.get_shape()
    # Signal size is set to half the size of the last dimension of the input
    # tensor, because the input tensor is an interleaved complex value.
    input_size = input_shape[axis] // 2
    output_size = n if n > 0 else 2 * (input_size - 1)

    # Verify output dimensions
    output_shape = output.get_shape()
    if output_shape[axis] != output_size:
        raise Error(
            "Output shape mismatch: got "
            + String(output_shape[axis])
            + " expected "
            + String(output_size)
        )

    # Calculate batch size.
    var batch_size = 1
    for i in range(input_rank - 1):
        batch_size *= input_shape[i]

    # Create cuFFT plan
    var plan: cufftHandle = 0
    var plan_ptr = UnsafePointer(to=plan)

    var plan_status = cufftPlan1d(
        plan_ptr,
        output_size,
        Type.CUFFT_C2R,
        batch_size,
    )
    check_error(plan_status)

    # Set up cuda stream.
    var cuda_stream = CUDA(ctx.stream())
    check_error(cufftSetStream(plan, cuda_stream))

    try:
        var input_ptr = input.data.bitcast[ComplexFloat32]()
        var output_ptr = output.data.bitcast[Float32]()
        var exec_status = cufftExecC2R(plan, input_ptr, output_ptr)
        if exec_status != Status.CUFFT_SUCCESS:  # CUFFT_SUCCESS is 0
            raise Error(
                "cufftExecC2R failed with status " + String(exec_status)
            )

    finally:
        check_error(cufftDestroy(plan))
