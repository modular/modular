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

from complex import ComplexFloat32
from gpu._cufft.cufft import (
    cufftCreate,
    cufftEstimate1d,
    cufftExecC2R,
    cufftGetSize,
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
from layout import LayoutTensor, Layout

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


fn _get_fft_workarea(
    buffer_size: Int, ctx: DeviceContext
) raises -> OpaquePointer:
    var fft_buffer_key = String("CUFFT_BUFFER_PTR_", buffer_size)

    if lookup := global_cache_lookup(fft_buffer_key):
        # we found the allocated device buffer
        return lookup

    # manually allocate the memory on the device, and cache the pointer
    var work_space = ctx.enqueue_create_buffer[DType.uint8](buffer_size)
    var device_ptr = work_space.take_ptr()

    global_cache_insert(
        fft_buffer_key,
        # bitcast the device pointer to a void * to cache it
        device_ptr.bitcast[NoneType](),
    )

    return device_ptr.bitcast[NoneType]()


fn _get_fft_plan[
    create_if_not_found: Bool = True
](
    output_size: Int,
    batch_size: Int,
    workspace_size: Int,
    ctx: DeviceContext,
) raises -> cufftHandle:
    var cached_plan_key = String("CUFFT_PLAN_", output_size, ",", batch_size)

    if lookup := global_cache_lookup(cached_plan_key):
        # We found the plan in the cache, so just return it
        return cufftHandle(Int(lookup))

    @parameter
    if not create_if_not_found:
        # a valid cufft handle is always non-zero
        return cufftHandle(0)

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
    var work_size: Int = 0
    # Get the precise size of the plan, assert that it is less than the allocated size
    check_error(cufftGetSize(plan, UnsafePointer(to=work_size)))
    work_space_ptr = _get_fft_workarea(workspace_size, ctx)

    if work_size > workspace_size:
        raise Error(
            "Need "
            + String(work_size // 1024 // 1024)
            + " MB of buffer allocated for cuFFT."
        )

    check_error(cufftSetWorkArea(plan, work_space_ptr))

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
    input_type: DType,
    output_type: DType,
    alignment: Int,
](
    input: LayoutTensor[
        input_type,
        alignment=alignment,
        address_space = AddressSpace.GENERIC, **_,
    ],
    output: LayoutTensor[
        mut=True,
        output_type,
        alignment=alignment,
        address_space = AddressSpace.GENERIC, **_,
    ],
    n: Int,
    buffer_size_mb: Int,
    ctx: DeviceContext,
) raises:
    """Compute the inverse real FFT of the input tensor.

    Currently, only applies it to the last dimension.

    Args:
        input: Complex input tensor (NDBuffer).
        output: Real output tensor (NDBuffer).
        n: Output signal size (if <= 0, computed as 2*(input.size(axis) - 1)).
        buffer_size_mb: Esimated buffer size in MB.
        ctx: Device context.
    """
    constrained[
        input.rank == output.rank, "Input and output must have the same rank"
    ]()
    constrained[
        input_type is DType.float32, "Only Float32 is supported for IRFFT"
    ]()
    constrained[
        output_type is DType.float32, "Only Float32 is supported for IRFFT"
    ]()
    # we allocate 64 MB more than the buffer size because the estimation might
    # not be exact.
    EST_WORKSPACE_SIZE = buffer_size_mb * 1024 * 1024
    ALLOCATED_WORKSPACE_SIZE = (buffer_size_mb + 64) * 1024 * 1024

    axis = input.rank - 1
    cuda_stream = CUDA(ctx.stream())

    # Get input and output dimensions
    input_shape = input.runtime_layout.shape.value
    # Signal size is set to half the size of the last dimension of the input
    # tensor, because the input tensor is an interleaved complex value.
    input_size = input_shape[axis] // 2
    output_size = n if n > 0 else 2 * (input_size - 1)

    # Verify output dimensions
    output_shape = output.runtime_layout.shape.value
    if output_shape[axis] != output_size:
        raise Error(
            "Output shape mismatch: got "
            + String(output_shape[axis])
            + " expected "
            + String(output_size)
        )

    # Calculate batch size.
    var batch_size = 1
    for i in range(input.rank - 1):
        batch_size *= input_shape[i]

    # skip size estimations if the plan is already cached, as
    # the function call is expensive
    if plan := _get_fft_plan[create_if_not_found=False](
        output_size, batch_size, ALLOCATED_WORKSPACE_SIZE, ctx
    ):
        check_error(cufftSetStream(plan, cuda_stream))
        var input_ptr = input.ptr.bitcast[ComplexFloat32]()
        var output_ptr = output.ptr.bitcast[Float32]()
        check_error(cufftExecC2R(plan, input_ptr, output_ptr))

        return

    var work_size: Int = 0
    check_error(
        cufftEstimate1d(
            output_size, Type.CUFFT_C2R, batch_size, UnsafePointer(to=work_size)
        )
    )

    if work_size < EST_WORKSPACE_SIZE:
        # Create a single cuFFT plan if the workspace size is less than
        # the given buffer size.
        var plan = _get_fft_plan(
            output_size, batch_size, ALLOCATED_WORKSPACE_SIZE, ctx
        )

        # Set up cuda stream.
        # Notice that we do not want to have this part of the cache
        # The stream is set everytime the call is executed and we get the
        # stream from the context we are executing within
        check_error(cufftSetStream(plan, cuda_stream))

        var input_ptr = input.ptr.bitcast[ComplexFloat32]()
        var output_ptr = output.ptr.bitcast[Float32]()
        check_error(cufftExecC2R(plan, input_ptr, output_ptr))

    else:
        # If the workspace size is too large, we need to run multiple steps
        # try to find the largest batch size that fits in the workspace
        var reduced_batch_size = batch_size

        while reduced_batch_size > 0:
            reduced_batch_size //= 2
            try:
                check_error(
                    cufftEstimate1d(
                        output_size,
                        Type.CUFFT_C2R,
                        reduced_batch_size,
                        UnsafePointer(to=work_size),
                    )
                )
                if work_size < EST_WORKSPACE_SIZE:
                    break
            except e:
                # Try the next work_size
                pass

        if reduced_batch_size == 0:
            raise Error(
                "FFT output signal size is too large, try to increase the"
                " buffer size."
            )

        # Create cuFFT plan
        var plan = _get_fft_plan(
            output_size, reduced_batch_size, ALLOCATED_WORKSPACE_SIZE, ctx
        )

        # Set up cuda stream.
        check_error(cufftSetStream(plan, cuda_stream))

        var input_ptr = input.ptr
        var output_ptr = output.ptr

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
            plan = _get_fft_plan(
                output_size, batch_size, ALLOCATED_WORKSPACE_SIZE, ctx
            )
            check_error(cufftSetStream(plan, cuda_stream))

            check_error(
                cufftExecC2R(
                    plan,
                    input_ptr.bitcast[ComplexFloat32](),
                    output_ptr.bitcast[Float32](),
                )
            )
