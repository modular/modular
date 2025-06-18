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

from sys.ffi import _get_global_or_null, external_call
from sys.intrinsics import _unsafe_aliasing_address_to_pointer


# This should eventually be moved to ffi.mojo with a more general global cache method
# cache key is a string and cache value is a pointer.
@always_inline
fn global_cache_lookup(key: String) -> OpaquePointer:
    return external_call["KGEN_CompilerRT_GetGlobalOrNull", OpaquePointer](
        key.unsafe_ptr(), key.byte_length()
    )


@always_inline
fn global_cache_insert(key: String, value: OpaquePointer):
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(key),
        value,
    )


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

    # We want to cache the cuFFT plan to avoid calling high overhead cuda
    # calls each time the plane is created and destroyed
    var cached_plan_key = String(output_size) + "," + String(batch_size)
    plan = cufftHandle(Int(global_cache_lookup(cached_plan_key)))

    if not plan:
        var plan_status = cufftPlan1d(
            plan_ptr,
            output_size,
            Type.CUFFT_C2R,
            batch_size,
        )
        check_error(plan_status)
        global_cache_insert(
            cached_plan_key,
            # we are bitcasting the integer plan to a void * to cache it,
            # because that's what KGEN_CompilerRT_InsertGlobal expects.
            _unsafe_aliasing_address_to_pointer[DType.index](Int(plan)).bitcast[
                NoneType
            ](),
        )

    # Set up cuda stream.
    # Notice that we do not want to have this part of the cache
    # The stream is set everytime the call is executed and we get the
    # stream from the context we are executing within
    var cuda_stream = CUDA(ctx.stream())
    check_error(cufftSetStream(plan, cuda_stream))

    var input_ptr = input.data.bitcast[ComplexFloat32]()
    var output_ptr = output.data.bitcast[Float32]()
    var exec_status = cufftExecC2R(plan, input_ptr, output_ptr)
    if exec_status != Status.CUFFT_SUCCESS:  # CUFFT_SUCCESS is 0
        raise Error("cufftExecC2R failed with status " + String(exec_status))
