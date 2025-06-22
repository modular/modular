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
    cufftCreate,
    cufftEstimate1d,
    cufftExecC2R,
    cufftGetSize1d,
    cufftHandle,
    cufftMakePlan1d,
    cufftSetAutoAllocation,
    cufftSetStream,
    cufftSetWorkArea,
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


fn _get_fft_plan(output_size: Int, batch_size: Int) raises -> cufftHandle:
    var cached_plan_key = String("CUFFT_PLAN_", output_size, ",", batch_size)

    if lookup := global_cache_lookup(cached_plan_key):
        # We found the plan in the cache, so just return it
        return cufftHandle(Int(lookup))

    var plan = cufftHandle(0)
    var mem_size: Int = 0
    check_error(cufftCreate(UnsafePointer(to=plan)))
    check_error(cufftSetAutoAllocation(plan, 0))
    check_error(
        cufftMakePlan1d(
            plan,
            output_size,
            Type.CUFFT_C2R,
            batch_size,
            UnsafePointer(to=mem_size),
        )
    )

    # We want to cache the cuFFT plan to avoid calling high overhead cuda
    # calls each time the plane is created and destroyed
    global_cache_insert(
        cached_plan_key,
        # we are bitcasting the integer plan to a void * to cache it,
        # because that's what KGEN_CompilerRT_InsertGlobal expects.
        _unsafe_aliasing_address_to_pointer[DType.index](Int(plan)).bitcast[
            NoneType
        ](),
    )

    return plan


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

    alias MAX_FFT_WORKSPACE_SIZE = 512 * 1024 * 1024  # 512 MB

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

    var work_size: Int = 0
    check_error(
        cufftEstimate1d(
            output_size, Type.CUFFT_C2R, batch_size, UnsafePointer(to=work_size)
        )
    )

    if work_size < MAX_FFT_WORKSPACE_SIZE:
        # Create a single cuFFT plan if the workspace size is less than 512 MB
        var plan = _get_fft_plan(output_size, batch_size)

        # Get the precise size of the plan, and allocate the workspace
        check_error(
            cufftGetSize1d(
                plan,
                output_size,
                Type.CUFFT_C2R,
                batch_size,
                UnsafePointer(to=work_size),
            )
        )
        var work_space = ctx.enqueue_create_buffer[DType.uint8](work_size)
        check_error(
            cufftSetWorkArea(plan, work_space.unsafe_ptr().bitcast[NoneType]())
        )

        # Set up cuda stream.
        # Notice that we do not want to have this part of the cache
        # The stream is set everytime the call is executed and we get the
        # stream from the context we are executing within
        var cuda_stream = CUDA(ctx.stream())
        check_error(cufftSetStream(plan, cuda_stream))

        var input_ptr = input.data.bitcast[ComplexFloat32]()
        var output_ptr = output.data.bitcast[Float32]()
        check_error(cufftExecC2R(plan, input_ptr, output_ptr))

        _ = work_space^
    else:
        # If the workspace size is too large, we need to run multiple steps
        # try to find the largest batch size that fits in the workspace
        var reduced_batch_size = batch_size

        while reduced_batch_size > 0:
            reduced_batch_size //= 2
            check_error(
                cufftEstimate1d(
                    output_size,
                    Type.CUFFT_C2R,
                    reduced_batch_size,
                    UnsafePointer(to=work_size),
                )
            )

            if work_size < MAX_FFT_WORKSPACE_SIZE:
                break

        if reduced_batch_size == 0:
            raise Error("FFT Output signal size is too large")

        # Create cuFFT plan
        var plan = _get_fft_plan(output_size, reduced_batch_size)

        # Get the precise size of the plan, and allocate the workspace
        check_error(
            cufftGetSize1d(
                plan,
                output_size,
                Type.CUFFT_C2R,
                reduced_batch_size,
                UnsafePointer(to=work_size),
            )
        )
        var work_space = ctx.enqueue_create_buffer[DType.uint8](work_size)
        check_error(
            cufftSetWorkArea(plan, work_space.unsafe_ptr().bitcast[NoneType]())
        )

        # Set up cuda stream.
        var cuda_stream = CUDA(ctx.stream())
        check_error(cufftSetStream(plan, cuda_stream))

        var input_ptr = input.data
        var output_ptr = output.data

        while batch_size >= reduced_batch_size:
            # Execute the cuFFT plan for the current batch size
            check_error(
                cufftExecC2R(
                    plan,
                    input_ptr.bitcast[ComplexFloat32](),
                    output_ptr.bitcast[Float32](),
                )
            )

            # Update the pointers for the next batch
            batch_size -= reduced_batch_size
            input_ptr += reduced_batch_size * input_shape[axis]
            output_ptr += reduced_batch_size * output_shape[axis]

        if batch_size > 0:
            # Create a new cuFFT plan for the remaining batch size
            # we reuse the allocated workspace, as it is already large enough
            plan = _get_fft_plan(output_size, batch_size)
            check_error(
                cufftSetWorkArea(
                    plan, work_space.unsafe_ptr().bitcast[NoneType]()
                )
            )
            check_error(cufftSetStream(plan, cuda_stream))

            check_error(
                cufftExecC2R(
                    plan,
                    input_ptr.bitcast[ComplexFloat32](),
                    output_ptr.bitcast[Float32](),
                )
            )

        _ = work_space^
