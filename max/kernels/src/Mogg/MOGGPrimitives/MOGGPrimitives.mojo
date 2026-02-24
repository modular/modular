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

from math import fma
from ffi import external_call
from sys import size_of, align_of

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from compiler_internal import StaticTensorSpec
from collections import InlineArray
from gpu.host import DeviceBuffer
from gpu.host.info import is_cpu, is_gpu
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from layout._coord import Coord, Idx
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from memory import memcpy

from nn.concat import concat
from register import register_internal
from runtime.asyncrt import DeviceContextPtr
from tensor import (
    DynamicTensor,
    InputTensor,
    IOSpec,
    ManagedTensorSlice,
)
from tensor.io_spec import IO
from tensor.managed_tensor_slice import get_kernel_simd_width
from weights_registry import WeightsRegistry

from utils import Index, IndexList, StaticTuple

from .MOGGIntList import IntList

# ===-----------------------------------------------------------------------===#
# Helper Structures
# ===-----------------------------------------------------------------------===#


fn bytecount_with_dtype[dtype: DType](shape: IndexList) -> Int:
    return shape.flattened_length() * size_of[dtype]()


# TODO: This struct should be deleted. Mojo and C++ should always communicate
# with pointers. If the Mojo wants to do something with this object, we should
# just create a C++ function for it. For the time being, this is safe because of
# the `constrained` and `static_assert` we added to ensure the type has the
# right byte size.
struct StateContext(TrivialRegisterPassable):
    """Defines a StateContext structure which holds a ptr to context and has accessors that go to external calls
    This is currently meant as a mojo-side container for GML::StateContext."""

    var num_slots: Int
    var ctx_ptr: OpaquePointer[MutAnyOrigin]

    @always_inline
    fn __init__(out self, num_slots: Int, ctx_ptr: OpaquePointer[MutAnyOrigin]):
        self.num_slots = num_slots
        self.ctx_ptr = ctx_ptr

        comptime assert size_of[StateContext]() == 16, (
            "Expecting StateContext to be 16 bytes wide, to match the C++"
            " equivalent"
        )

    @always_inline
    fn __getitem__(self, index: Int) -> OpaquePointer[MutAnyOrigin]:
        debug_assert(0 <= index < self.num_slots, "index must be within bounds")
        return external_call[
            "MGP_RT_GetContextPayloadPtr",
            OpaquePointer[MutAnyOrigin],
        ](index, self.ctx_ptr)


fn pack_string_res(
    str_ptr: UnsafePointer[Byte, ImmutAnyOrigin], str_len: Int
) raises -> String:
    var span = Span(ptr=str_ptr, length=str_len)
    # We can not free the resource ptr embedded in MEF, create a copy
    return String(StringSlice(from_utf8=span))


# ===-----------------------------------------------------------------------===#
# Async Packing/Unpacking functions
# ===-----------------------------------------------------------------------===#


@register_internal("builtin.create_error_async_values_and_destruct_error")
@no_inline
fn create_error_async_values_and_destruct_error(
    async_ptr: UnsafePointer[OpaquePointer[MutAnyOrigin], MutAnyOrigin],
    async_len: Int,
    var err: Error,
):
    """Indicates to the C++ runtime that the kernel has failed."""
    var error_message = String(err)
    external_call["KGEN_CompilerRT_AsyncRT_CreateAsyncs_Error", NoneType](
        async_ptr,
        async_len,
        error_message.as_c_string_slice().unsafe_ptr(),
        error_message.byte_length(),
    )


@register_internal("builtin.create_index_async")
@no_inline
fn create_index_async(value: Int, async_ptr: OpaquePointer[MutAnyOrigin]):
    external_call["MGP_RT_CreateAsync_ssizet", NoneType](value, async_ptr)


@register_internal("builtin.create_si64_async")
@no_inline
@export
fn create_si64_async(value: Int64, async_ptr: OpaquePointer[MutAnyOrigin]):
    external_call["MGP_RT_CreateAsync_int64t", NoneType](value, async_ptr)


@register_internal("builtin.create_chain_async")
@no_inline
fn create_chain_async(async_ptr: OpaquePointer[MutAnyOrigin]):
    external_call["MGP_RT_CreateAsync_chain", NoneType](async_ptr)


@register_internal("builtin.create_bool_async")
@register_internal("builtin.create_i1_async")
@no_inline
fn create_i1_async(
    value: Bool,
    async_ptr: OpaquePointer[MutAnyOrigin],
):
    external_call["MGP_RT_CreateAsync_bool", NoneType](value, async_ptr)


@register_internal("builtin.create_buffer_ref_async")
@no_inline
fn create_buffer_ref_async(
    buffer: NDBuffer[DType.int8, 1, MutAnyOrigin],
    async_ptr: OpaquePointer[MutAnyOrigin],
    call_ctx: DeviceContextPtr,
):
    external_call["MGP_RT_CreateAsyncDeviceBufferRef", NoneType](
        buffer.data, len(buffer), async_ptr, call_ctx._handle
    )


@register_internal("builtin.create_non_tracked_buffer_ref_async")
@no_inline
fn create_non_tracked_buffer_ref_async(
    buffer: NDBuffer[DType.int8, 1, MutAnyOrigin],
    async_ptr: OpaquePointer[MutAnyOrigin],
):
    external_call["MGP_RT_CreateAsyncNonTrackedBufferRef", NoneType](
        buffer.data, len(buffer), async_ptr
    )


@register_internal("builtin.create_non_tracked_tensor_async")
@no_inline
fn create_non_tracked_tensor_async[
    tensor_rank: Int,
    buffer_rank: Int,
    dtype: DType,
](
    buffer: NDBuffer[dtype, buffer_rank, MutAnyOrigin],
    async_ptr: OpaquePointer[MutAnyOrigin],
):
    comptime assert tensor_rank == buffer_rank or (
        tensor_rank == 0 and buffer_rank == 1
    )
    external_call["MGP_RT_CreateAsyncNonTrackedTensor", NoneType](
        buffer.data,
        bytecount_with_dtype[dtype](buffer.dynamic_shape),
        tensor_rank,
        UnsafePointer(to=buffer.dynamic_shape.data),
        dtype,
        async_ptr,
    )


@register_internal("builtin.create_buffer_ref_with_borrow_async")
@no_inline
fn create_buffer_ref_with_borrow_async[
    borrowee_type: Int,
](
    buffer: NDBuffer[DType.int8, 1, MutAnyOrigin],
    async_to_borrow: OpaquePointer[MutAnyOrigin],
    output_async: OpaquePointer[MutAnyOrigin],
):
    external_call["MGP_RT_CreateAsyncBufferWithBorrow", NoneType](
        buffer.data,
        len(buffer),
        async_to_borrow,
        borrowee_type,
        output_async,
    )


@register_internal("builtin.create_tensor_spec_async")
@no_inline
fn create_tensor_spec_async[
    spec_rank: Int
](spec: IndexList[spec_rank], async_ptr: OpaquePointer[MutAnyOrigin],):
    # Mojo impl is bitwise compatible with cpp variant, can construct TensorSpec in mojo
    # and pass it back to C++ -- However, this is an issue for the heap allocated dims.
    # For the benefit of simplicity, allocate the shapes and ptrs and free explicitly after
    var storage = InlineArray[Int, spec_rank](uninitialized=True)

    comptime for i in range(spec_rank):
        storage[i] = spec[i]

    external_call["MGP_RT_CreateAsyncTensorShape", NoneType](
        storage.unsafe_ptr(), spec_rank, async_ptr
    )


@register_internal("builtin.create_tensor_with_borrow_async")
@no_inline
fn create_tensor_async[
    tensor_rank: Int,
    buffer_rank: Int,
    dtype: DType,
    borrowee_type: Int,
](
    buffer: NDBuffer[dtype, buffer_rank, MutAnyOrigin],
    async_to_borrow: OpaquePointer[MutAnyOrigin],
    output_async: OpaquePointer[MutAnyOrigin],
):
    # Tensor and the underlying buffer must have the same rank, unless it is a
    # scalar tensor stored with a NDBuffer<[1]>
    comptime assert tensor_rank == buffer_rank or (
        tensor_rank == 0 and buffer_rank == 1
    )
    external_call["MGP_RT_CreateAsyncTensorWithBorrow", NoneType](
        buffer.data,
        bytecount_with_dtype[dtype](buffer.dynamic_shape),
        tensor_rank,
        UnsafePointer(to=buffer.dynamic_shape.data),
        dtype,
        async_to_borrow,
        borrowee_type,
        output_async,
    )
    pass


@export
fn empty_destructor(ptr: UnsafePointer[UInt8, MutExternalOrigin]):
    pass


@register_internal("builtin.create_mojo_value_async")
@no_inline
fn create_mojo_value_async(
    val_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    async_ptr: OpaquePointer[MutAnyOrigin],
    size: Int,
    align: Int,
    destructor_fn: fn(UnsafePointer[UInt8, MutExternalOrigin]) -> None,
    move_fn: fn(
        UnsafePointer[UInt8, MutAnyOrigin], UnsafePointer[UInt8, MutAnyOrigin]
    ) -> None,
):
    # Check if we have a nullptr, if so, don't use a destructor.
    if not val_ptr:
        external_call["MGP_RT_CreateOwnedAsyncMojoValue", NoneType](
            val_ptr,
            empty_destructor,
            async_ptr,
        )
        return
    var dst_ptr = external_call[
        "MGP_RT_MojoValueAllocateBuffer",
        UnsafePointer[UInt8, MutExternalOrigin],
    ](size, align)
    move_fn(val_ptr, dst_ptr)

    external_call["MGP_RT_CreateOwnedAsyncMojoValue", NoneType](
        dst_ptr,
        destructor_fn,
        async_ptr,
    )


@register_internal("builtin.create_python_mojo_value_async")
@no_inline
fn create_python_mojo_value_async(
    val_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    async_ptr: OpaquePointer[MutAnyOrigin],
    size: Int,
    align: Int,
    destructor_fn: fn(UnsafePointer[UInt8, MutExternalOrigin]) -> None,
    move_fn: fn(
        UnsafePointer[UInt8, MutAnyOrigin], UnsafePointer[UInt8, MutAnyOrigin]
    ) -> None,
):
    var dst_ptr = external_call[
        "MGP_RT_MojoValueAllocateBuffer",
        UnsafePointer[UInt8, MutExternalOrigin],
    ](size, align)
    move_fn(val_ptr, dst_ptr)

    external_call["MGP_RT_CreateOwnedAsyncPythonMojoValue", NoneType](
        dst_ptr,
        destructor_fn,
        async_ptr,
    )


@register_internal("builtin.transfer_async")
@no_inline
fn transfer_async(
    async_src: OpaquePointer[MutAnyOrigin],
    async_dst: OpaquePointer[MutAnyOrigin],
):
    external_call[
        "MGP_RT_TransferAsyncRef",
        NoneType,
    ](async_src, async_dst)


@register_internal("builtin.unpack_async")
@no_inline
fn unpack_async(
    async_ptr: OpaquePointer[MutAnyOrigin],
) -> OpaquePointer[MutAnyOrigin]:
    return external_call[
        "MGP_RT_GetValueFromAsync",
        OpaquePointer[MutAnyOrigin],
    ](async_ptr)


@register_internal("builtin.unpack_device_ctx")
@no_inline
fn unpack_device_ctx(
    async_ptr: OpaquePointer[MutAnyOrigin],
) -> DeviceContextPtr:
    var ptr = external_call[
        "MGP_RT_UnpackDeviceContext",
        OpaquePointer[MutAnyOrigin],
    ](async_ptr)

    return DeviceContextPtr(ptr.unsafe_origin_cast[MutExternalOrigin]())


@register_internal("builtin.unpack_buffer_ref")
@no_inline
fn unpack_buffer_ref(
    async_ptr: OpaquePointer[MutAnyOrigin],
) -> NDBuffer[DType.int8, 1, MutAnyOrigin]:
    var size: UInt64 = 0
    var data_ptr = external_call[
        "MGP_RT_GetDataFromBuffer",
        OpaquePointer[MutAnyOrigin],
    ](async_ptr, UnsafePointer(to=size))
    var shape = IndexList[1](Int(size))
    return NDBuffer[DType.int8, 1](data_ptr.bitcast[Int8](), shape)


@register_internal("builtin.unpack_tensor")
@no_inline
fn unpack_tensor[
    buffer_rank: Int,
    tensor_rank: Int,
    dtype: DType,
](tensor_async_ptr: OpaquePointer[MutAnyOrigin]) -> NDBuffer[
    dtype, buffer_rank, MutAnyOrigin
]:
    # Tensor and the underlying buffer must have the same rank, unless it is a
    # scalar tensor stored with a NDBuffer<[1]>
    comptime assert tensor_rank == buffer_rank or (
        tensor_rank == 0 and buffer_rank == 1
    )
    var shapes = IndexList[buffer_rank]()
    var buffer_ptr = external_call[
        "MGP_RT_GetShapeAndDataFromTensor",
        OpaquePointer[MutAnyOrigin],
    ](
        UnsafePointer(to=shapes.data),
        tensor_async_ptr,
    )

    comptime if tensor_rank == 0:
        shapes[0] = 1

    return NDBuffer[dtype, buffer_rank](
        buffer_ptr.bitcast[Scalar[dtype]](), shapes
    )


@register_internal("builtin.unpack_tensor_spec")
@no_inline
fn unpack_tensor_spec[
    spec_rank: Int
](async_ptr: OpaquePointer[MutAnyOrigin]) -> IndexList[spec_rank]:
    var storage = InlineArray[Int, spec_rank](uninitialized=True)
    external_call[
        "MGP_RT_GetTensorShapeFromAsync",
        NoneType,
    ](storage.unsafe_ptr(), spec_rank, async_ptr)
    var shape = IndexList[spec_rank]()

    comptime for i in range(spec_rank):
        shape[i] = storage[i]

    return shape


@register_internal("builtin.unpack_context")
@no_inline
fn unpack_context(
    async_ptr: OpaquePointer[MutAnyOrigin],
) -> StateContext:
    # We want to construct this because we want all payloads to be implemented
    var num_slots: UInt64 = 0
    var ctx_ptr: OpaquePointer[MutAnyOrigin] = external_call[
        "MGP_RT_GetContextAndSizeFromAsync",
        OpaquePointer[MutAnyOrigin],
    ](UnsafePointer(to=num_slots), async_ptr)
    return StateContext(Int(num_slots), ctx_ptr)


@register_internal("builtin.get_buffer_data")
@always_inline
fn get_buffer_data(
    buffer: NDBuffer[DType.int8, 1, MutAnyOrigin]
) -> UnsafePointer[Int8, MutAnyOrigin]:
    return buffer.data


# ===-----------------------------------------------------------------------===#
# MGP Tensor Primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.tensor.create")
@no_inline
fn mgp_tensor_create[
    spec_rank: Int,
    buffer_rank: Int,
    dtype: DType,
](
    buffer: NDBuffer[DType.int8, 1, MutAnyOrigin],
    spec: IndexList[spec_rank],
) -> NDBuffer[dtype, buffer_rank, MutAnyOrigin]:
    comptime if spec_rank == 0:
        # We promote scalar tensor to tensor<[1]>
        comptime assert buffer_rank == 1
        return NDBuffer[dtype, buffer_rank](
            buffer.data.bitcast[Scalar[dtype]](),
            rebind[IndexList[buffer_rank]](IndexList[1](1)),
        )
    else:
        comptime assert spec_rank == buffer_rank
        return NDBuffer[dtype, buffer_rank](
            buffer.data.bitcast[Scalar[dtype]](),
            rebind[IndexList[buffer_rank]](spec),
        )


@register_internal("mgp.tensor.extract.tensor_spec")
@no_inline
fn mgp_tensor_extract_tensor_spec[
    tensor_rank: Int,
    buffer_rank: Int,
    dtype: DType,
](buffer: NDBuffer[dtype, buffer_rank, ImmutAnyOrigin]) -> IndexList[
    tensor_rank
]:
    comptime if tensor_rank == 0:
        comptime assert buffer_rank == 1
        return rebind[IndexList[tensor_rank]](IndexList[0]())
    else:
        comptime assert buffer_rank == tensor_rank
        return rebind[IndexList[tensor_rank]](
            buffer.dynamic_shape.canonicalize()
        )


@register_internal("mgp.tensor.extract.buffer")
@no_inline
fn mgp_tensor_extract_buffer[
    buffer_rank: Int,
    dtype: DType,
](buffer: NDBuffer[dtype, buffer_rank, MutAnyOrigin]) -> NDBuffer[
    DType.int8, 1, MutAnyOrigin
]:
    # Unwrap the tensor into a size-less buffer pointer.
    return NDBuffer[DType.int8, 1](
        buffer.data.bitcast[Int8](), buffer.bytecount()
    )


# ===-----------------------------------------------------------------------===#
# MGP Buffer Primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.buffer.alloc")
@no_inline
fn mgp_buffer_alloc(
    byte_size: Int, dev_context: DeviceContextPtr
) raises -> NDBuffer[DType.int8, 1, MutAnyOrigin]:
    # Default to alignment of 0 which means kPreferredMemoryAlignment if cRawAlign is kUnknownSize (SizeUtils.h).
    # alias alignment = 0 if bRawAlign == UInt64.MAX else Int(bRawAlign)

    # This primitive has a byte-size input, so always assume a byte format
    var shape = IndexList[1](byte_size)
    var buf = dev_context[].enqueue_create_buffer[DType.int8](byte_size)
    return NDBuffer[DType.int8, 1](buf^.take_ptr(), shape)


@register_internal("mgp.buffer.constant")
@export
fn mgp_buffer_constant(
    resource_ptr: OpaquePointer[MutAnyOrigin],
    resource_bytecount: Int,
) -> NDBuffer[DType.int8, 1, MutAnyOrigin]:
    # Should we keep the alignment? It seems that the static alignment is
    # dropped in the kernels anyway.
    return NDBuffer[DType.int8, 1](
        resource_ptr.bitcast[Int8](), resource_bytecount
    )


@register_internal("mgp.buffer.constant.external")
fn mgp_buffer_constant_external(
    weights: UnsafePointer[WeightsRegistry, MutAnyOrigin],
    name_ptr: UnsafePointer[Byte, ImmutAnyOrigin],
    name_len: UInt,
    size: UInt64,
    align: UInt64,
) raises -> NDBuffer[DType.int8, 1, MutAnyOrigin]:
    debug_assert(align > 0, "align must be a positive integer value")

    if not weights:
        raise Error(
            "received null weights registry in mgp.buffer.constant.external"
        )

    var weight_ptr = weights[][pack_string_res(name_ptr, Int(name_len))]
    if (UInt64(Int(weight_ptr)) % align) != 0:
        raise Error(
            "invalid alignment for address ",
            weight_ptr,
            " and align ",
            align,
        )

    return NDBuffer[DType.int8, 1](weight_ptr.bitcast[Int8](), DimList(size))


@no_inline
fn fill_buffer[
    dtype: DType
](buf: NDBuffer[DType.int8, 1, MutAnyOrigin], vals: VariadicList[Int]):
    var ptr = buf.data.bitcast[Scalar[dtype]]()
    var offset: Int = 0
    for val in vals:
        ptr.store(offset, Scalar[dtype](val))
        offset += 1


@register_internal("mgp.buffer.set_with_index")
@no_inline
fn mgp_buffer_set_with_index[
    bDevice: StaticString
](buffer: NDBuffer[DType.int8, 1, MutAnyOrigin], *vals: Int) raises:
    debug_assert(
        is_cpu[bDevice](), "set_with_index can only work on cpu buffers"
    )
    var bufSize = buffer.num_elements()
    var numArgs = len(vals)
    debug_assert(
        bufSize % numArgs == 0,
        "buffer size not divisible by number of index args",
    )

    var elSize = bufSize // numArgs
    if elSize == 4:
        fill_buffer[DType.int32](buffer, vals)
    elif elSize == 8:
        fill_buffer[DType.int64](buffer, vals)
    else:
        raise Error("unsupported element size")


@register_internal("mgp.buffer.to_bool")
@no_inline
fn mgp_buffer_to_bool[
    bDevice: StaticString
](buffer: NDBuffer[DType.int8, 1, ImmutAnyOrigin]) -> Bool:
    debug_assert(is_cpu[bDevice](), "to_bool can only work on cpu buffers")
    var bufSize = buffer.num_elements()
    debug_assert(
        bufSize == 1,
        "buffer size must be a size of 1",
    )
    return buffer[0] != 0


@register_internal("mgp.buffer.to_index")
@no_inline
fn mgp_buffer_to_index(
    buffer: NDBuffer[DType.int8, 1, ImmutAnyOrigin]
) raises -> Int:
    var bufSize = buffer.num_elements()
    if bufSize == 4:
        return Int(buffer.data.bitcast[Int32]()[0])
    if bufSize == 8:
        return Int(buffer.data.bitcast[Int64]()[0])

    raise Error(
        "mgp.buffer.to_index must be called on either a 4- or 8-byte buffer"
    )


@register_internal("mgp.buffer.slice")
@no_inline
fn mgp_buffer_slice(
    buffer: NDBuffer[DType.int8, 1, MutAnyOrigin], offset: Int, size: Int
) -> NDBuffer[DType.int8, 1, MutAnyOrigin]:
    return NDBuffer[DType.int8, 1](buffer.data + offset, Index(size))


@register_internal("mgp.buffer.concat")
@no_inline
fn mgp_buffer_concat[
    bDevice: StaticString
](
    output: NDBuffer[DType.int8, 1, MutAnyOrigin],
    inputs: StaticTuple[NDBuffer[DType.int8, 1, MutAnyOrigin], ...],
    call_ctx: DeviceContextPtr,
) raises:
    var output_lt = TileTensor(
        output.data,
        row_major(Coord(Idx(len(output)))),
    )
    var input_tensors = StaticTuple[_, inputs.size](
        TileTensor(inputs[0])
        .make_dynamic[DType.int64]()
        .as_any_origin()
        .as_immut()
    )
    for i in range(1, len(inputs)):
        input_tensors[i] = (
            TileTensor(inputs[i])
            .make_dynamic[DType.int64]()
            .as_any_origin()
            .as_immut()
        )
    if len(output) < 4096:
        concat[DType.int8, True, bDevice, None](
            output_lt, 0, input_tensors, context=call_ctx
        )
    else:
        concat[DType.int8, False, bDevice, None](
            output_lt, 0, input_tensors, context=call_ctx
        )


@register_internal("mgp.buffer.device_to_host")
@no_inline
fn mgp_buffer_device_to_host[
    cOtherDevice: StaticString,
    dHostDevice: StaticString,
](
    dev_buf: NDBuffer[DType.int8, 1, MutAnyOrigin],
    host_buf: NDBuffer[DType.int8, 1, MutAnyOrigin],
    dev_ctx: DeviceContextPtr,
) raises:
    comptime if is_cpu[dHostDevice]() and is_gpu[cOtherDevice]():
        dev_ctx[].enqueue_copy[DType.int8](
            host_buf.data,
            DeviceBuffer[DType.int8](
                dev_ctx[],
                dev_buf.data,
                dev_buf.size(),
                owning=False,
            ),
        )
    else:
        raise Error("mgp.buffer.device_to_host must be scheduled on gpu device")


@register_internal("mgp.buffer.device_to_device")
@no_inline
fn mgp_buffer_device_to_device[
    cSrcDevice: StaticString,
    dDstDevice: StaticString,
](
    src_buf: NDBuffer[DType.int8, 1, MutAnyOrigin],
    dst_buf: NDBuffer[DType.int8, 1, MutAnyOrigin],
    src_dev_ctx: DeviceContextPtr,
    dst_dev_ctx: DeviceContextPtr,
) raises:
    comptime if is_gpu[cSrcDevice]() and is_gpu[dDstDevice]():
        dst_dev_ctx[].enqueue_copy[DType.int8](
            DeviceBuffer[DType.int8](
                dst_dev_ctx[],
                dst_buf.data,
                dst_buf.size(),
                owning=False,
            ),
            DeviceBuffer[DType.int8](
                src_dev_ctx[],
                src_buf.data,
                src_buf.size(),
                owning=False,
            ),
        )
    elif is_cpu[cSrcDevice]() and is_cpu[dDstDevice]():
        memcpy(dest=dst_buf.data, src=src_buf.data, count=src_buf.size())
    else:
        raise Error(
            "mgp.buffer.device_to_device can be scheduled between same device"
            " dtypes (cpu-cpu) or (gpu-gpu)"
        )


@register_internal("mgp.buffer.host_to_device")
@no_inline
fn mgp_buffer_host_to_device[
    cHostDevice: StaticString,
    dOtherDevice: StaticString,
](
    host_buf: NDBuffer[DType.int8, 1, MutAnyOrigin],
    dev_buf: NDBuffer[DType.int8, 1, MutAnyOrigin],
    dev_ctx: DeviceContextPtr,
) raises:
    comptime if is_gpu[dOtherDevice]() and is_cpu[cHostDevice]():
        dev_ctx[].enqueue_copy[DType.int8](
            DeviceBuffer[DType.int8](
                dev_ctx[],
                dev_buf.data,
                dev_buf.size(),
                owning=False,
            ),
            host_buf.data,
        )
    else:
        raise Error("mgp.buffer.host_to_device must be scheduled on gpu device")


@register_internal("mgp.buffer.get_cached")
@no_inline
fn mgp_buffer_get_cached(
    ctx: StateContext,
    buffer_slot: UInt,
    storage_ref_addr: UnsafePointer[OpaquePointer[MutAnyOrigin], MutAnyOrigin],
) raises -> NDBuffer[DType.int8, 1, MutAnyOrigin]:
    var buffer_size: UInt64 = 0
    var buffer_data: OpaquePointer[MutAnyOrigin] = external_call[
        "MGP_RT_GetCachedBuffer", OpaquePointer[MutAnyOrigin]
    ](
        Int(buffer_slot),
        ctx.ctx_ptr,
        UnsafePointer(to=buffer_size),
        storage_ref_addr,
    )

    return NDBuffer[DType.int8, 1](
        buffer_data.bitcast[Int8](), Index(buffer_size)
    )


@register_internal("mgp.buffer.remove_cached")
@no_inline
fn mgp_buffer_remove_cached(ctx: StateContext, buffer_slot: UInt64):
    external_call["MGP_RT_RemoveCachedBuffer", NoneType](
        Int(buffer_slot), ctx.ctx_ptr
    )


@register_internal("mgp.buffer.get_size")
@no_inline
fn mgp_buffer_get_size(buf: NDBuffer[DType.int8, 1, ImmutAnyOrigin]) -> Int:
    return buf.num_elements()


@register_internal("destruct_async_refs")
@no_inline
fn destruct_async_refs(
    storage_ref_addr: UnsafePointer[OpaquePointer[MutAnyOrigin], MutAnyOrigin],
    size: Int,
    direct_ref: Bool,
):
    external_call["MGP_RT_DestructAsyncRefs", NoneType](
        size, storage_ref_addr, direct_ref
    )


# ===-----------------------------------------------------------------------===#
# MGP Tensor Spec Primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.tensor_spec.create")
@no_inline
fn mgp_tensor_spec_create[
    aRawDims: DimList,
    aRawDimsRank: Int,
](*runtimeDims: Int) -> IndexList[aRawDimsRank]:
    var static_shape = IntList[aRawDims]()
    var shape = IndexList[aRawDimsRank]()
    var runtimeIndex = 0
    # Update Shape with runtime elements.
    for i in range(aRawDimsRank):
        if static_shape[i] > -1:
            shape[i] = static_shape[i]
        else:
            shape[i] = runtimeDims[runtimeIndex]
            runtimeIndex = runtimeIndex + 1
    return shape


@register_internal("mgp.tensor_spec.get_dim")
@no_inline
fn mgp_tensor_spec_get_dim[
    spec_rank: Int, axis: UInt64
](spec: IndexList[spec_rank]) -> Int:
    comptime assert axis < UInt64(
        spec_rank
    ), "axis for get_dim must be less than rank of TensorSpec"
    return spec[Int(axis)]


# ===-----------------------------------------------------------------------===#
# MGP Device Context Primitives
# ===-----------------------------------------------------------------------===#


@export
fn mgp_device_context_destroy(dev_ctx: DeviceContextPtr):
    # DeviceContext is refcounted, we don't need to explicitly destroy it
    pass


@register_internal("mgp.sync")
@no_inline
fn mgp_sync(ctx: StateContext, dev_ctx: DeviceContextPtr) raises:
    dev_ctx[].synchronize()


@register_internal("mgp.debug.print")
@no_inline
fn mgp_debug_print[
    aDebugString: StaticString,
    bLabel: StaticString,
](ctx: StateContext,) raises:
    var prefix = String()
    if bLabel:
        prefix = "[" + bLabel + "] "
    print(prefix + aDebugString)


@register_internal("mgp.debug.tensor.print")
@no_inline
fn mgp_debug_tensor_print[
    spec_rank: Int,
    dtype: DType,
](
    buffer: NDBuffer[DType.int8, 1, ImmutAnyOrigin],
    shape: IndexList[spec_rank],
    label_ptr: UnsafePointer[Byte, ImmutAnyOrigin],
    label_len: Int,
) raises:
    external_call["MGP_RT_DebugTensorPrint", NoneType](
        label_ptr,
        UInt(label_len),
        dtype,
        UnsafePointer(to=shape.data),
        spec_rank,
        buffer.data,
        len(buffer),
    )


# ===-----------------------------------------------------------------------===#
# Mojo generation and general type lookups
# ===-----------------------------------------------------------------------===#


@register_internal("float4_e2m1fn")
fn DTypeFloat4E2M1TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.float4_e2m1fn._mlir_value


@register_internal("float8_e8m0fnu")
fn DTypeFloat8E8M0FnuTypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.float8_e8m0fnu._mlir_value


@register_internal("float8_e5m2")
fn DTypeFloat8E5M2TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.float8_e5m2._mlir_value


@register_internal("float8_e5m2fnuz")
fn DTypeFloat8E5M2FnuzTypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.float8_e5m2fnuz._mlir_value


@register_internal("float8_e3m4")
fn DTypeFloat8E3M4TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.float8_e3m4._mlir_value


@register_internal("float8_e4m3fn")
fn DTypeFloat8E4M3FnTypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.float8_e4m3fn._mlir_value


@register_internal("float8_e4m3fnuz")
fn DTypeFloat8E4M3FnuzTypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.float8_e4m3fnuz._mlir_value


@register_internal("bfloat16")
fn DTypeBFloat16TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.bfloat16._mlir_value


@register_internal("float16")
fn DTypeFloat16TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.float16._mlir_value


@register_internal("float32")
fn DTypeFloat32TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.float32._mlir_value


@register_internal("float64")
fn DTypeFloat64TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.float64._mlir_value


@register_internal("int8")
fn DTypeInt8TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.int8._mlir_value


@register_internal("int16")
fn DTypeInt16TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.int16._mlir_value


@register_internal("int32")
fn DTypeInt32TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.int32._mlir_value


@register_internal("uint32")
fn DTypeUInt32TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.uint32._mlir_value


@register_internal("uint64")
fn DTypeUInt64TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.uint64._mlir_value


@register_internal("int64")
fn DTypeInt64TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.int64._mlir_value


@register_internal("uint8")
fn DTypeUInt8TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.uint8._mlir_value


@register_internal("uint16")
fn DTypeUInt16TypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.uint16._mlir_value


@register_internal("bool")
fn DTypeBoolTypeDef(ty: DType._mlir_type) -> DType._mlir_type:
    return DType.bool._mlir_value


@register_internal("index")
fn IndexTypeDef(ty: Int) -> Int:
    return ty


@register_internal("deviceContext")
fn DeviceContextDef(ty: DeviceContextPtr):
    pass


@register_internal("simd")
fn SimdTypeDef[
    dtype: DType, width: Int
](ty: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return ty


@register_internal("indices")
fn TensorIndicesTypeDef[rank: Int](ty: IndexList[rank]) -> IndexList[rank]:
    return ty


@register_internal("dim_type")
fn DimTypeDef(ty: Dim) -> Dim:
    return ty


@register_internal("managed_tensor_slice")
fn ManagedTensorSliceDef[
    mut: Bool,
    input: IO,
    dtype: DType,
    rank: Int,
    //,
    io_spec: IOSpec[mut, input],
    static_spec: StaticTensorSpec[dtype, rank],
](
    ty: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
) -> ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]:
    return ty


# ===-----------------------------------------------------------------------===#
# Hooks to help build static shapes.
# ===-----------------------------------------------------------------------===#


@register_internal("create_unknown_dim")
fn create_unknown_dim() -> Dim:
    return Dim()


@register_internal("create_known_dim")
fn create_known_dim[known_val: Int]() -> Dim:
    return Dim(known_val)


@register_internal("reshape_contiguous_managed_tensor_slice")
@always_inline
fn reshape_contiguous_buffer[
    dtype: DType, old_rank: Int, new_rank: Int, mut: Bool, input: IO
](
    buffer: ManagedTensorSlice[
        io_spec = IOSpec[mut, input](),
        static_spec = StaticTensorSpec[dtype, old_rank].create_unknown(),
    ],
    shape: IndexList[new_rank],
) -> DynamicTensor[dtype, new_rank]:
    return DynamicTensor[dtype, new_rank](buffer._ptr, shape)


# ===----------------------------------------------------------------------===#
# Additional expected primitives
# ===-----------------------------------------------------------------------===#


@register_internal("get_simd_width_for_dtypes")
@always_inline
fn get_simd_width_for_dtypes[
    dtypes: StaticTuple[DType], target: StaticString
]() -> Int:
    comptime assert dtypes.size > 0

    var width = get_kernel_simd_width[dtypes[0], target]()

    comptime for i in range(dtypes.size - 1):
        width = max(get_kernel_simd_width[dtypes[i + 1], target](), width)

    return width


@register_internal("get_address_space")
fn get_address_space() -> AddressSpace:
    return AddressSpace.GENERIC


# Build the StaticTensorSpec parameter for the DPS kernels
@register_internal("build_static_tensor_specs")
fn build_static_tensor_specs[
    dtype: DType,
    rank: Int,
](
    shape: DimList,
    strides: DimList,
    alignment: Int,
    address_space: AddressSpace,
    exclusive: Bool,
) -> StaticTensorSpec[dtype, rank]:
    comptime SpecType = StaticTensorSpec[dtype, rank]

    return SpecType(
        shape, strides, alignment, address_space, exclusive, None, None, None
    )


# Build the tuple of StaticTensorSpecs for DPS kernels
@register_internal("build_static_tensor_specs_tuple")
fn build_static_tensor_specs_tuple[
    dtype: DType,
    rank: Int,
    size: Int,
](
    array_of_specs: VariadicList[StaticTensorSpec[dtype, rank]],
    out result: StaticTuple[StaticTensorSpec[dtype, rank], size],
):
    return {array_of_specs}


# TODO: this should take IOSpec as a param -- will require graph compiler changes
# Used by the graph compiler to construct tensors from MGP repr. of tensor
@register_internal("to_managed_tensor_slice")
@always_inline
fn to_managed_tensor_slice[
    dtype: DType, rank: Int, mut: Bool, input: IO
](
    data: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    shape: UnsafePointer[Int, ImmutAnyOrigin],
) -> ManagedTensorSlice[
    io_spec = IOSpec[mut, input](),
    static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
]:
    var shape_ptr = shape
    var shape_tuple = IndexList[rank]()

    var stride_tuple = IndexList[rank]()
    var stride: Int = 1

    comptime for i in reversed(range(rank)):
        # Start from the back so we can accumulate the strides.
        shape_tuple[i] = shape_ptr[i]
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    return {data, shape_tuple, stride_tuple}


# Extract a scalar from a managed tensor slice.
@always_inline
fn _get_scalar_from_managed_tensor_slice[
    dtype: DType,
](tensor: ManagedTensorSlice[dtype=dtype]) -> Scalar[dtype]:
    # Assumes that tensor is on the host!
    # This is used instead of [0] since __getitem__ for `ManagedTesnorSlice`
    # does not work with `register_internal` out of the box.
    return tensor.load[width=1](IndexList[1](0))


@register_internal("get_scalar_from_managed_tensor_slice")
@always_inline
fn get_scalar_from_managed_tensor_slice[
    dtype: DType, mut: Bool, input: IO
](
    tensor: ManagedTensorSlice[
        io_spec = IOSpec[mut, input](),
        static_spec = StaticTensorSpec[dtype, 1].create_unknown(),
    ]
) -> Scalar[dtype]:
    return _get_scalar_from_managed_tensor_slice(tensor)


@register_internal("get_int_from_shape")
@always_inline
fn get_int_from_shape[
    param_index: Int, rank: Int
](shape: IndexList[rank]) -> Int:
    return shape[param_index]


@register_internal("rebuild_static_tensor_specs_with_output_compute_lambda")
@no_inline
fn rebuild_static_tensor_specs_with_output_compute_lambda[
    func_type: __TypeOfAllTypes,
    //,
    dtype: DType,
    rank: Int,
](
    spec: StaticTensorSpec[dtype, rank],
    out_compute_lambda: func_type,
) -> StaticTensorSpec[dtype, rank]:
    return StaticTensorSpec[dtype, rank](
        shape=spec.shape,
        strides=spec.strides,
        alignment=spec.alignment,
        address_space=spec.address_space,
        exclusive=spec.exclusive,
        in_lambda=None,
        out_lambda=None,
        out_compute_lambda=rebind[spec.out_compute_lambda_t](
            out_compute_lambda
        ),
    )


@always_inline
fn _to_managed_tensor_slice_index_list_shape[
    dtype: DType, rank: Int, mut: Bool, input: IO
](
    data: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    shape_tuple: IndexList[rank],
) -> ManagedTensorSlice[
    io_spec = IOSpec[mut, input](),
    static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
]:
    var stride_tuple = IndexList[rank]()
    var stride: Int = 1

    comptime for i in reversed(range(rank)):
        # Start from the back so we can accumulate the strides.
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    return {data, shape_tuple, stride_tuple}


# Helper method used by compiler to reconcile MGP list with dtype Mojo expects.
@register_internal("to_managed_tensor_slice_list")
@always_inline
fn to_managed_tensor_slice_list[
    dtype: DType, rank: Int, mut: Bool, input: IO
](
    raw_list_ptr: OpaquePointer[MutAnyOrigin],
    out out_list: List[
        ManagedTensorSlice[
            io_spec = IOSpec[mut, input](),
            static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
        ]
    ],
):
    var num_elements = external_call["MGP_RT_ListSize", Int64](
        raw_list_ptr
    ).__int__()

    var data_ptrs = List[OpaquePointer[MutAnyOrigin]](capacity=num_elements)
    var dim_values = List[Int64](capacity=num_elements * rank)

    # Collect the data pointers and dimensions of each element from the list.
    external_call["MGP_RT_ListPopulate", NoneType](
        raw_list_ptr, data_ptrs.unsafe_ptr(), dim_values.unsafe_ptr()
    )

    # TODO: revisit the use of unknown here
    # Create output list
    out_list = type_of(out_list)(capacity=num_elements)

    # Convert individual elements of the input list into NDBuffer, and
    # accumulate the results to output list.
    for i in range(num_elements):
        var data = data_ptrs[i].bitcast[Scalar[dtype]]()

        var dims = IndexList[rank]()

        comptime for dim in range(rank):
            dims[dim] = dim_values[dim + i * rank].__int__()

        var buffer = _to_managed_tensor_slice_index_list_shape[
            dtype, rank, mut, input
        ](data, dims)
        out_list.append(buffer)


# ===-----------------------------------------------------------------------===#
# Opaque Test Primitives
# ===-----------------------------------------------------------------------===#


struct MyInt(Movable):
    var val: Int

    fn __init__(out self, val: Int):
        self.val = val

    fn __init__(out self, *, deinit take: MyInt):
        print("MyInt.__moveinit__", take.val)
        self.val = take.val

    fn __del__(deinit self):
        print("MyInt.__del__", self.val)


@register_internal("testfuse.my_int.from_index")
@no_inline
fn test_my_int_from_index(x: Int) -> MyInt:
    return MyInt(x)


@register_internal("testfuse.my_int.square")
@no_inline
fn test_my_int_square(x: MyInt) -> MyInt:
    return MyInt(x.val * x.val)


@register_internal("testfuse.my_int.to_index")
@no_inline
fn test_my_int_to_index(x: MyInt) -> Int:
    return x.val


struct MyIntReg(TrivialRegisterPassable):
    var val: Int

    fn __init__(out self, val: Int):
        self.val = val


@register_internal("testfuse.my_int_reg.square")
@no_inline
fn test_my_int_reg_square(x: MyIntReg) -> MyIntReg:
    return MyIntReg(x.val * x.val)


struct MyIntReg2(ImplicitlyCopyable, RegisterPassable):
    var val: Int

    fn __init__(out self, val: Int):
        self.val = val

    fn __del__(deinit self):
        print("MyIntReg2.__del__", self.val)


@register_internal("testfuse.my_int_reg2.from_index")
@no_inline
fn test_my_int_reg2_from_index(x: Int) -> MyIntReg2:
    return MyIntReg2(x)


@register_internal("testfuse.my_int_reg2.square")
@no_inline
fn test_my_int_reg2_square(x: MyIntReg2) -> MyIntReg2:
    return MyIntReg2(x.val * x.val)


@register_internal("testfuse.my_int_reg2.to_index")
@no_inline
fn test_my_int_reg2_to_index(x: MyIntReg2) -> Int:
    return x.val


# ===-----------------------------------------------------------------------===#
# Mojo generation hooks
# ===-----------------------------------------------------------------------===#

# ===-----------------------------------------------------------------------===#
# Mojo-C++ interop aliases
# ===-----------------------------------------------------------------------===#

# The purpose of these aliases is to make it easier to visually parse the
# interop. There is only one rule: Do not use types, always use OpaquePointer.
# This saves us from having to statically assert that a certain type has a
# specific byte size.

# AnyAsyncValueRef is a C++ struct. The runtime passes a reference to it.
# Therefore, we alias it to OpaquePointer which will have the same bitwidth as
# C++'s pointers.
comptime AnyAsyncValueRefPtr = OpaquePointer[MutAnyOrigin]

# TensorBufferRef is a C++ struct. Primitives should always manipulate a
# reference to it. Therefore, it is modeled here as an OpaquePointer.
comptime TensorBufferRefPtr = OpaquePointer[MutAnyOrigin]

# StateContext is a C++ struct. Primitives should always manipulate a reference
# to it. Therefore, it is modeled here as an OpaquePointer.
comptime StateContextRef = OpaquePointer[MutAnyOrigin]


# ===-----------------------------------------------------------------------===#
# MOGG primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mogg.as_scalar")
@always_inline
fn mogg_as_scalar(tensor: ManagedTensorSlice) -> Scalar[tensor.dtype]:
    return _get_scalar_from_managed_tensor_slice(tensor)


@register_internal("mogg.async.__del__")
@no_inline
fn mogg_async_del(
    async_ptr: UnsafePointer[AnyAsyncValueRefPtr, MutAnyOrigin], size: Int
):
    """
    Decrement the AnyAsyncValueRef. Typically called at the end of a kernel for
    all input and output operands.
    """
    external_call["MGP_RT_DestructAsyncRefs", NoneType](size, async_ptr, False)


@register_internal("mogg.async.unpack")
@no_inline
fn mogg_async_unpack[T: __TypeOfAllTypes](async_ptr: AnyAsyncValueRefPtr) -> T:
    """
    Returns the value stored in the AnyAsyncValueRef.
    """
    var ptr = external_call[
        "MGP_RT_GetValueFromAsync", OpaquePointer[MutAnyOrigin]
    ](async_ptr).bitcast[T]()

    return UnsafePointer[T, MutAnyOrigin].__getitem__(ptr, 0)


struct MoggAsyncPackHelper:
    """
    Helper struct for packing various data types into an asynchronous context
    for MOGG operations. Provides constructor overloads for different supported
    types.
    """

    fn __init__(out self, data: Int, async_ptr: AnyAsyncValueRefPtr):
        """
        Packs an integer value into the asynchronous context.
        Calls create_index_async to handle the packing.
        """
        create_index_async(data, async_ptr)

    fn __init__(out self, data: Int64, async_ptr: AnyAsyncValueRefPtr):
        """
        Packs a 64-bit integer value into the asynchronous context.
        Calls create_si64_async to handle the packing.
        """
        create_si64_async(data, async_ptr)

    fn __init__(out self, data: Bool, async_ptr: AnyAsyncValueRefPtr):
        """
        Packs a boolean value into the asynchronous context.
        Calls create_i1_async to handle the packing.
        """
        create_i1_async(data, async_ptr)

    fn __init__[
        spec_rank: Int
    ](out self, data: IndexList[spec_rank], async_ptr: AnyAsyncValueRefPtr):
        """
        Packs an IndexList of specified rank into the asynchronous context.
        Calls create_tensor_spec_async to handle the packing.
        """
        create_tensor_spec_async(data, async_ptr)

    fn __init__(
        out self,
        data: NDBuffer[DType.int8, 1, MutAnyOrigin],
        device_ctx_ptr: DeviceContextPtr,
        async_ptr: AnyAsyncValueRefPtr,
    ):
        """
        Packs a buffer reference instance (modeled by NDBuffer[DType.int8, 1, MutAnyOrigin] for now) into the asynchronous context. Calls create_buffer_ref_async to handle the packing.
        """
        create_buffer_ref_async(data, async_ptr, device_ctx_ptr)

    fn __init__(
        out self,
        var data: Some[Movable & ImplicitlyDestructible],
        async_ptr: AnyAsyncValueRefPtr,
    ):
        """
        Packs a generic Movable value into the asynchronous context.
        Used for opaque types like SIMDPair.
        """
        comptime Type = type_of(data)

        # MGP_RT_CreateOwnedAsyncMojoValue expects a type erased destructor
        @always_inline("nodebug")
        fn erased_destructor(ptr: UnsafePointer[UInt8, MutExternalOrigin]):
            ptr.bitcast[Type]().destroy_pointee()

        var dst_ptr = external_call[
            "MGP_RT_MojoValueAllocateBuffer",
            UnsafePointer[UInt8, MutExternalOrigin],
        ](size_of[Type](), align_of[Type]())

        dst_ptr.bitcast[Type]().init_pointee_move(data^)

        external_call["MGP_RT_CreateOwnedAsyncMojoValue", NoneType](
            dst_ptr,
            erased_destructor,
            async_ptr,
        )


@register_internal("mogg.async.pack")
@no_inline
fn mogg_async_pack(pack_helper: MoggAsyncPackHelper):
    """
    Packs asynchronous data using the provided MoggAsyncPackHelper.

    This function serves as an entry point for packing data into an asynchronous
    reference. The actual packing logic is handled by the MoggAsyncPackHelper struct,
    which provides specialized constructors for different data types. This function
    itself is a no-op and exists to satisfy the internal registration mechanism.
    """
    return


@no_inline
fn mogg_async_pack_borrow[
    buffer_rank: Int,
    dtype: DType,
    //,
    spec_rank: Int,
    is_tensor: Bool,
](
    borrower: AnyAsyncValueRefPtr,
    buffer: NDBuffer[dtype, buffer_rank, MutAnyOrigin],
    mem: TensorBufferRefPtr,
):
    """
    Borrows an async value. This differs from `mogg.async.pack` which assigns a
    value to the given async value in that it's a simple refcount increment.
    """

    comptime if is_tensor:
        external_call["MGP_RT_TensorBorrowV2", NoneType](
            borrower,
            buffer.data,
            bytecount_with_dtype[dtype](buffer.dynamic_shape),
            spec_rank,
            UnsafePointer(to=buffer.dynamic_shape.data),
            dtype,
            mem,
        )
    else:
        external_call["MGP_RT_BufferBorrowV2", NoneType](
            borrower, buffer.data, len(buffer), mem
        )


@no_inline
fn mogg_async_pack_borrow[
    spec_rank: Int,  # unused
    is_tensor: Bool,  # unused
](
    borrower: AnyAsyncValueRefPtr,
    buffer: TensorBufferRefPtr,
    mem: TensorBufferRefPtr,
):
    """
    Borrows an async value. This differs from `mogg.async.pack` which assigns a
    value to the given async value in that it's a simple refcount increment.
    """
    external_call["MGP_RT_BufferBorrowForTensorRef", NoneType](
        borrower, buffer, mem
    )


@register_internal("mogg.tensor.__init__")
@always_inline
fn mogg_tensor_init[
    dtype: DType,
    rank: Int,
    mut: Bool,
    input: IO,
    static_shape: DimList,
    static_stride: DimList,
    alignment: Int,
    exclusive: Bool,
](
    ptr: OpaquePointer[MutAnyOrigin], shape: IndexList[rank]
) -> ManagedTensorSlice[
    io_spec = IOSpec[mut, input](),
    static_spec = StaticTensorSpec[dtype, rank](
        static_shape,
        static_stride,
        alignment,
        AddressSpace.GENERIC,
        exclusive,
        None,
        None,
        None,
    ),
]:
    """
    Helper for constructing a ManagedTensorSlice.
    """
    return {ptr.bitcast[Scalar[dtype]](), shape}


@register_internal("mogg.async.ready")
@no_inline
fn mogg_async_ready(async_ptr: AnyAsyncValueRefPtr):
    """
    Marks the chain as ready.
    """
    external_call["MGP_RT_CreateAsync_chain", NoneType](async_ptr)


@register_internal("mogg.async.error")
@no_inline
fn mogg_async_error(async_ptr: AnyAsyncValueRefPtr, err: Error):
    """Indicates to the C++ runtime that the kernel has failed."""
    var error_message = String(err)
    external_call["MGP_RT_AsyncRT_CreateAsync_Error", NoneType](
        async_ptr,
        error_message.as_c_string_slice().unsafe_ptr(),
        error_message.byte_length(),
    )


@register_internal("tmp.reshape_contiguous_managed_tensor_slice")
@always_inline
fn tmp_reshape_contiguous_buffer[
    static_shape: DimList, static_stride: DimList, new_rank: Int
](
    buffer: ManagedTensorSlice,
    shape: IndexList[new_rank],
) -> ManagedTensorSlice[
    io_spec = buffer.io_spec,
    static_spec = StaticTensorSpec[buffer.dtype, new_rank](
        static_shape,
        static_stride,
        1,
        AddressSpace.GENERIC,
        True,
        None,
        None,
        None,
    ),
]:
    """
    Constructs a new ManagedTensorSlice with a new shape and static spec.
    """
    return {buffer._ptr, shape}


# ===-----------------------------------------------------------------------===#
# MGP primitives
# ===-----------------------------------------------------------------------===#


@register_internal("tmp.mgp.buffer.get_cached")
@no_inline
fn tmp_mgp_buffer_get_cached(
    ctx: StateContextRef,
    buffer_slot: Int,
) -> Tuple[NDBuffer[DType.int8, 1, MutAnyOrigin], TensorBufferRefPtr]:
    """
    Get a reference to the cached tensor.
    """
    var buffer_size: UInt64 = 0
    var buffer_data = OpaquePointer[MutAnyOrigin]()

    var buffer_ref = external_call[
        "TMP_MGP_RT_GetCachedBuffer", TensorBufferRefPtr
    ](
        buffer_slot,
        ctx,
        UnsafePointer(to=buffer_size),
        UnsafePointer(to=buffer_data),
    )

    var buffer = NDBuffer[DType.int8, 1](
        buffer_data.bitcast[Int8](), Index(buffer_size)
    )
    var res = Tuple[NDBuffer[DType.int8, 1, MutAnyOrigin], TensorBufferRefPtr](
        buffer, buffer_ref
    )

    return res


@register_internal("tmp.mgp.buffer.remove_cached")
@no_inline
fn tmp_mgp_buffer_remove_cached(ctx: StateContextRef, buffer_slot: UInt64):
    external_call["TMP_MGP_RT_RemoveCachedBuffer", NoneType](buffer_slot, ctx)


@register_internal("mgp.assert")
@no_inline
fn mgp_assert(
    cond: Bool, msg_ptr: UnsafePointer[Byte, ImmutAnyOrigin], msg_len: Int
) raises:
    """
    Raises an error when the input condition is not true.
    """
    if not cond:
        raise Error(pack_string_res(msg_ptr, msg_len))


# ===----------------------------------------------------------------------===#
# Affine view kernels
# ===----------------------------------------------------------------------===#


@register_internal("split_dim_indices")
@always_inline
fn split_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], new_shape_dim: Int) -> IndexList[rank + 1]:
    var out = IndexList[rank + 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [40, 30, 2] and we reshape it to [5, 8, 30, 2].
    # If we are accessing the index [21, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [2, 5, 16, 1].
    # Or [21 // 8, 21 % 8, ...old dims...].
    # In this case, the axis = 0 and the new_shape_dim = 8.

    comptime for i in range(rank + 1):
        comptime if i == axis:
            out[i] = indices[axis] // new_shape_dim
        elif i == axis + 1:
            out[i] = indices[axis] % new_shape_dim
        elif i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]

    return out


@register_internal("merge_dim_indices")
@always_inline
fn merge_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], old_shape_dim: Int) -> IndexList[rank - 1]:
    var out = IndexList[rank - 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [5, 8, 30, 2] and we reshape it to [40, 30, 2].
    # If we are accessing the index [2, 5, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [21, 16, 1].
    # Or [2 * 8 + 5, 16, 1].
    # In this case, the axis = 0 and the old_shape_dim = 8.

    comptime for i in range(rank - 1):
        comptime if i == axis:
            out[i] = fma(indices[i], old_shape_dim, indices[i + 1])
        elif i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i + 1]

    return out


@register_internal("insert_index")
@always_inline
fn insert_index[
    rank: Int, axis: Int, value: Int
](indices: IndexList[rank]) -> IndexList[rank + 1]:
    var out = IndexList[rank + 1]()

    comptime for i in range(rank + 1):
        comptime if i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]
        else:
            out[i] = value

    return out


fn all_zeros(indices: IndexList) -> Bool:
    comptime for i in range(indices.size):
        if indices[i] != 0:
            return False
    return True


fn get_buffer_mem_storage_handle(
    buffer: OpaquePointer[MutAnyOrigin],
    type: Int,
    memStorageHandle: OpaquePointer[MutAnyOrigin],
):
    external_call["MGP_RT_GetBufferMemStorageHandle", NoneType](
        buffer, type, memStorageHandle
    )


@register_internal("pop.select")
@always_inline
fn select[T: __TypeOfAllTypes](cond: Bool, true_case: T, false_case: T) -> T:
    if cond:
        return true_case

    return false_case


@register_internal("pop.simd.select")
@always_inline
fn simd_select[
    T: __TypeOfAllTypes
](cond: Bool, true_case: T, false_case: T) -> T:
    return select(cond, true_case, false_case)
