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

from sys import external_call

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from compiler_internal import StaticTensorSpec
from gpu.host import DeviceBuffer
from gpu.host.info import is_cpu, is_gpu
from math import fma
from memory import memcpy
from nn.concat import concat
from register import register_internal
from runtime.asyncrt import DeviceContextPtr
from tensor_internal.managed_tensor_slice import get_kernel_simd_width
from tensor_internal.io_spec import IO
from tensor_internal import (
    DynamicTensor,
    InputTensor,
    IOSpec,
    ManagedTensorSlice,
)
from weights_registry import WeightsRegistry

from utils import Index, IndexList, StaticTuple

from .MOGGIntList import IntList

# ===-----------------------------------------------------------------------===#
# Helper Structures
# ===-----------------------------------------------------------------------===#


fn bytecount_with_dtype(shape: IndexList, dtype: DType) -> Int:
    return shape.flattened_length() * dtype.size_of()


@register_passable("trivial")
struct StateContext:
    """Defines a StateContext structure which holds a ptr to context and has accessors that go to external calls
    This is currently meant as a mojo-side container for GML::StateContext."""

    var num_slots: Int
    var ctx_ptr: OpaquePointer

    @always_inline
    fn __init__(out self, num_slots: Int, ctx_ptr: OpaquePointer):
        self.num_slots = num_slots
        self.ctx_ptr = ctx_ptr

    @always_inline
    fn __getitem__(self, index: Int) -> OpaquePointer:
        debug_assert(0 <= index < self.num_slots, "index must be within bounds")
        return external_call[
            "KGEN_CompilerRT_GetContextPayloadPtr",
            OpaquePointer,
        ](index, self.ctx_ptr)


fn pack_string_res(
    str_ptr: UnsafePointer[Byte], str_len: UInt
) raises -> String:
    var span = Span[Byte, ImmutableAnyOrigin](
        ptr=str_ptr,
        length=str_len,
    )
    # We can not free the resource ptr embedded in MEF, create a copy
    return String(StringSlice(from_utf8=span))


# ===-----------------------------------------------------------------------===#
# Async Packing/Unpacking functions
# ===-----------------------------------------------------------------------===#


@register_internal("builtin.create_error_async_values_and_destruct_error")
@no_inline
fn create_error_async_values_and_destruct_error(
    async_ptr: UnsafePointer[OpaquePointer],
    async_len: Int,
    var err: Error,
):
    """Indicates to the C++ runtime that the kernel has failed."""
    external_call["KGEN_CompilerRT_AsyncRT_CreateAsyncs_Error", NoneType](
        async_ptr,
        async_len,
        err.unsafe_cstr_ptr(),
        err.byte_length(),
    )


@register_internal("builtin.create_index_async")
@no_inline
fn create_index_async(value: Int, async_ptr: OpaquePointer):
    external_call["KGEN_CompilerRT_CreateAsync_ssizet", NoneType](
        value, async_ptr
    )


@register_internal("builtin.create_si64_async")
@no_inline
@export
fn create_si64_async(value: Scalar[DType.int64], async_ptr: OpaquePointer):
    external_call["KGEN_CompilerRT_CreateAsync_int64t", NoneType](
        value, async_ptr
    )


@register_internal("builtin.create_chain_async")
@no_inline
fn create_chain_async(async_ptr: OpaquePointer):
    external_call["KGEN_CompilerRT_CreateAsync_chain", NoneType](async_ptr)


@register_internal("builtin.create_bool_async")
@register_internal("builtin.create_i1_async")
@no_inline
fn create_i1_async(
    value: Bool,
    async_ptr: OpaquePointer,
):
    external_call["KGEN_CompilerRT_CreateAsync_bool", NoneType](
        value, async_ptr
    )


@register_internal("builtin.create_buffer_ref_async")
@no_inline
fn create_buffer_ref_async(
    buffer: NDBuffer[DType.int8, 1, MutableAnyOrigin],
    async_ptr: OpaquePointer,
    call_ctx: DeviceContextPtr,
):
    external_call["KGEN_CompilerRT_CreateAsyncDeviceBufferRef", NoneType](
        buffer.data, len(buffer), async_ptr, call_ctx._handle
    )


@register_internal("builtin.create_non_tracked_buffer_ref_async")
@no_inline
fn create_non_tracked_buffer_ref_async(
    buffer: NDBuffer[DType.int8, 1, MutableAnyOrigin],
    async_ptr: OpaquePointer,
):
    external_call["KGEN_CompilerRT_CreateAsyncNonTrackedBufferRef", NoneType](
        buffer.data, len(buffer), async_ptr
    )


@register_internal("builtin.create_non_tracked_tensor_async")
@no_inline
fn create_non_tracked_tensor_async[
    tensor_rank: Int,
    buffer_rank: Int,
    dtype: DType,
](
    buffer: NDBuffer[dtype, buffer_rank, MutableAnyOrigin],
    async_ptr: OpaquePointer,
):
    constrained[
        tensor_rank == buffer_rank or (tensor_rank == 0 and buffer_rank == 1)
    ]()
    external_call["KGEN_CompilerRT_CreateAsyncNonTrackedTensor", NoneType](
        buffer.data,
        bytecount_with_dtype(buffer.dynamic_shape, dtype),
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
    buffer: NDBuffer[DType.int8, 1, MutableAnyOrigin],
    async_to_borrow: OpaquePointer,
    output_async: OpaquePointer,
):
    external_call["KGEN_CompilerRT_CreateAsyncBufferWithBorrow", NoneType](
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
](spec: IndexList[spec_rank], async_ptr: OpaquePointer,):
    # Mojo impl is bitwise compatible with cpp variant, can construct TensorSpec in mojo
    # and pass it back to C++ -- However, this is an issue for the heap allocated dims.
    # For the benefit of simplicity, allocate the shapes and ptrs and free explicitly after
    var shape_ptr = UnsafePointer[Int].alloc(spec_rank)

    @parameter
    for i in range(spec_rank):
        shape_ptr[i] = spec[i]

    external_call["KGEN_CompilerRT_CreateAsyncTensorShape", NoneType](
        shape_ptr, spec_rank, async_ptr
    )
    shape_ptr.free()


@register_internal("builtin.create_tensor_with_borrow_async")
@no_inline
fn create_tensor_async[
    tensor_rank: Int,
    buffer_rank: Int,
    dtype: DType,
    borrowee_type: Int,
](
    buffer: NDBuffer[dtype, buffer_rank, MutableAnyOrigin],
    async_to_borrow: OpaquePointer,
    output_async: OpaquePointer,
):
    # Tensor and the underlying buffer must have the same rank, unless it is a
    # scalar tensor stored with a NDBuffer<[1]>
    constrained[
        tensor_rank == buffer_rank or (tensor_rank == 0 and buffer_rank == 1)
    ]()
    external_call["KGEN_CompilerRT_CreateAsyncTensorWithBorrow", NoneType](
        buffer.data,
        bytecount_with_dtype(buffer.dynamic_shape, dtype),
        tensor_rank,
        UnsafePointer(to=buffer.dynamic_shape.data),
        dtype,
        async_to_borrow,
        borrowee_type,
        output_async,
    )
    pass


@export
fn empty_destructor(ptr: UnsafePointer[UInt8]):
    pass


@register_internal("builtin.create_mojo_value_async")
@no_inline
fn create_mojo_value_async(
    val_ptr: UnsafePointer[UInt8],
    async_ptr: OpaquePointer,
    size: Int,
    align: Int,
    destructor_fn: fn (UnsafePointer[UInt8]) -> None,
    move_fn: fn (UnsafePointer[UInt8], UnsafePointer[UInt8]) -> None,
):
    # Check if we have a nullptr, if so, don't use a destructor.
    if not val_ptr:
        external_call["KGEN_CompilerRT_CreateOwnedAsyncMojoValue", NoneType](
            val_ptr,
            empty_destructor,
            async_ptr,
        )
        return
    var dst_ptr = external_call[
        "KGEN_CompilerRT_MojoValueAllocateBuffer", UnsafePointer[UInt8]
    ](size, align)
    move_fn(val_ptr, dst_ptr)

    external_call["KGEN_CompilerRT_CreateOwnedAsyncMojoValue", NoneType](
        dst_ptr,
        destructor_fn,
        async_ptr,
    )


@register_internal("builtin.create_python_mojo_value_async")
@no_inline
fn create_python_mojo_value_async(
    val_ptr: UnsafePointer[UInt8],
    async_ptr: OpaquePointer,
    size: Int,
    align: Int,
    destructor_fn: fn (UnsafePointer[UInt8]) -> None,
    move_fn: fn (UnsafePointer[UInt8], UnsafePointer[UInt8]) -> None,
):
    var dst_ptr = external_call[
        "KGEN_CompilerRT_MojoValueAllocateBuffer", UnsafePointer[UInt8]
    ](size, align)
    move_fn(val_ptr, dst_ptr)

    external_call["KGEN_CompilerRT_CreateOwnedAsyncPythonMojoValue", NoneType](
        dst_ptr,
        destructor_fn,
        async_ptr,
    )


@register_internal("builtin.transfer_async")
@no_inline
fn transfer_async(
    async_src: OpaquePointer,
    async_dst: OpaquePointer,
):
    external_call[
        "KGEN_CompilerRT_TransferAsyncRef",
        NoneType,
    ](async_src, async_dst)


@register_internal("builtin.unpack_async")
@no_inline
fn unpack_async(
    async_ptr: OpaquePointer,
) -> OpaquePointer:
    return external_call[
        "KGEN_CompilerRT_GetValueFromAsync",
        OpaquePointer,
    ](async_ptr)


@register_internal("builtin.unpack_device_ctx")
@no_inline
fn unpack_device_ctx(
    async_ptr: OpaquePointer,
) -> DeviceContextPtr:
    var ptr = external_call[
        "KGEN_CompilerRT_UnpackDeviceContext",
        OpaquePointer,
    ](async_ptr)

    return DeviceContextPtr(ptr)


@register_internal("builtin.unpack_buffer_ref")
@no_inline
fn unpack_buffer_ref(
    async_ptr: OpaquePointer,
) -> NDBuffer[DType.uint8, 1, MutableAnyOrigin]:
    var size: UInt64 = 0
    var data_ptr = external_call[
        "KGEN_CompilerRT_GetDataFromBuffer",
        OpaquePointer,
    ](async_ptr, UnsafePointer(to=size))
    var shape = IndexList[1](Int(size))
    return NDBuffer[DType.uint8, 1](data_ptr.bitcast[UInt8](), shape)


@register_internal("builtin.unpack_tensor")
@no_inline
fn unpack_tensor[
    buffer_rank: Int,
    tensor_rank: Int,
    dtype: DType,
](tensor_async_ptr: OpaquePointer) -> NDBuffer[
    dtype, buffer_rank, MutableAnyOrigin
]:
    # Tensor and the underlying buffer must have the same rank, unless it is a
    # scalar tensor stored with a NDBuffer<[1]>
    constrained[
        tensor_rank == buffer_rank or (tensor_rank == 0 and buffer_rank == 1)
    ]()
    var shapes = IndexList[buffer_rank]()
    var buffer_ptr = external_call[
        "KGEN_CompilerRT_GetShapeAndDataFromTensor",
        OpaquePointer,
    ](
        UnsafePointer(to=shapes.data),
        tensor_async_ptr,
    )

    @parameter
    if tensor_rank == 0:
        shapes[0] = 1

    return NDBuffer[dtype, buffer_rank](
        buffer_ptr.bitcast[Scalar[dtype]](), shapes
    )


@register_internal("builtin.unpack_tensor_spec")
@no_inline
fn unpack_tensor_spec[
    spec_rank: Int
](async_ptr: OpaquePointer) -> IndexList[spec_rank]:
    var shape_ptr = UnsafePointer[Int].alloc(spec_rank)
    external_call[
        "KGEN_CompilerRT_GetTensorShapeFromAsync",
        NoneType,
    ](shape_ptr, spec_rank, async_ptr)
    var shape = IndexList[spec_rank]()

    @parameter
    for i in range(spec_rank):
        shape[i] = Int(shape_ptr[i])

    shape_ptr.free()
    return shape


@register_internal("builtin.unpack_context")
@no_inline
fn unpack_context(
    async_ptr: OpaquePointer,
) -> StateContext:
    # We want to construct this because we want all payloads to be implemented
    var num_slots: UInt64 = 0
    var ctx_ptr: OpaquePointer = external_call[
        "KGEN_CompilerRT_GetContextAndSizeFromAsync",
        OpaquePointer,
    ](UnsafePointer(to=num_slots), async_ptr)
    return StateContext(Int(num_slots), ctx_ptr)


@register_internal("builtin.get_buffer_data")
@always_inline
fn get_buffer_data(
    buffer: NDBuffer[DType.uint8, 1, MutableAnyOrigin]
) -> UnsafePointer[UInt8]:
    return buffer.data


# ===-----------------------------------------------------------------------===#
# Mojo generation hooks
# ===-----------------------------------------------------------------------===#


@register_internal("mogg.async.__del__")
@no_inline
fn mogg_async_del(async_ptr: OpaquePointer):
    var ptr = UnsafePointer(to=async_ptr)
    external_call["KGEN_CompilerRT_DestructAsyncRefs", NoneType](1, ptr, False)


@register_internal("mogg.async.unpack")
@no_inline
fn mogg_async_unpack[T: AnyTrivialRegType](async_ptr: OpaquePointer) -> T:
    return external_call["KGEN_CompilerRT_GetValueFromAsync", OpaquePointer](
        async_ptr
    ).bitcast[T]()[0]


@register_internal("mogg.tensor.__init__")
@no_inline
fn mogg_tensor_init[
    dtype: DType,
    rank: Int,
    mut: Bool,
    input: IO,
    static_shape: DimList,
    static_stride: DimList,
    alignment: Int,
](ptr: OpaquePointer, shape: IndexList[rank]) -> ManagedTensorSlice[
    io_spec = IOSpec[mut, input](),
    static_spec = StaticTensorSpec[dtype, rank](
        static_shape,
        static_stride,
        alignment,
        AddressSpace.GENERIC,
        True,
        None,
        None,
        None,
    ),
]:
    alias static_spec = StaticTensorSpec[dtype, rank](
        static_shape,
        static_stride,
        alignment,
        AddressSpace.GENERIC,
        True,
        None,
        None,
        None,
    )
    return ManagedTensorSlice[
        io_spec = IOSpec[mut, input](),
        static_spec=static_spec,
    ](ptr.bitcast[Scalar[dtype]](), shape)


@register_internal("mogg.async.ready")
@no_inline
fn mogg_async_ready(async_ptr: OpaquePointer):
    external_call["KGEN_CompilerRT_CreateAsync_chain", NoneType](async_ptr)


# ===-----------------------------------------------------------------------===#
# MGP Common Primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.assert")
@no_inline
fn mgp_assert(cond: Bool, msg_ptr: UnsafePointer[Byte], msg_len: UInt) raises:
    if not cond:
        raise Error(pack_string_res(msg_ptr, msg_len))


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
    buffer: NDBuffer[DType.uint8, 1, MutableAnyOrigin],
    spec: IndexList[spec_rank],
) -> NDBuffer[dtype, buffer_rank, MutableAnyOrigin]:
    @parameter
    if spec_rank == 0:
        # We promote scalar tensor to tensor<[1]>
        constrained[buffer_rank == 1]()
        return NDBuffer[dtype, buffer_rank](
            buffer.data.bitcast[Scalar[dtype]](),
            rebind[IndexList[buffer_rank]](IndexList[1](1)),
        )
    else:
        constrained[spec_rank == buffer_rank]()
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
](buffer: NDBuffer[dtype, buffer_rank, MutableAnyOrigin]) -> IndexList[
    tensor_rank
]:
    @parameter
    if tensor_rank == 0:
        constrained[buffer_rank == 1]()
        return rebind[IndexList[tensor_rank]](IndexList[0]())
    else:
        constrained[buffer_rank == tensor_rank]()
        return rebind[IndexList[tensor_rank]](
            buffer.dynamic_shape.canonicalize()
        )


@register_internal("mgp.tensor.extract.buffer")
@no_inline
fn mgp_tensor_extract_buffer[
    tensor_rank: Int,
    buffer_rank: Int,
    dtype: DType,
](buffer: NDBuffer[dtype, buffer_rank, MutableAnyOrigin]) -> NDBuffer[
    DType.uint8, 1, MutableAnyOrigin
]:
    # Unwrap the tensor into a size-less buffer pointer.
    return NDBuffer[DType.uint8, 1](
        buffer.data.bitcast[UInt8](), buffer.bytecount()
    )


# ===-----------------------------------------------------------------------===#
# MGP Buffer Primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.buffer.alloc")
@no_inline
fn mgp_buffer_alloc(
    byte_size: Int, dev_context: DeviceContextPtr
) raises -> NDBuffer[DType.int8, 1, MutableAnyOrigin]:
    # Default to alignment of 0 which means kPreferredMemoryAlignment if cRawAlign is kUnknownSize (SizeUtils.h).
    # alias alignment = 0 if bRawAlign == UInt64.MAX else Int(bRawAlign)

    # This primitive has a byte-size input, so always assume a byte format
    var shape = IndexList[1](byte_size)
    var buf = dev_context[].enqueue_create_buffer[DType.int8](byte_size)
    return NDBuffer[DType.int8, 1](buf^.take_ptr(), shape)


@register_internal("mgp.buffer.constant")
@export
fn mgp_buffer_constant(
    resource_ptr: OpaquePointer,
    resource_bytecount: Int,
) -> NDBuffer[DType.int8, 1, MutableAnyOrigin]:
    # Should we keep the alignment? It seems that the static alignment is
    # dropped in the kernels anyway.
    return NDBuffer[DType.int8, 1](
        resource_ptr.bitcast[Int8](), resource_bytecount
    )


@register_internal("mgp.buffer.constant.external")
fn mgp_buffer_constant_external(
    weights: UnsafePointer[WeightsRegistry],
    name_ptr: UnsafePointer[Byte],
    name_len: UInt,
    size: UInt64,
    align: UInt64,
) raises -> NDBuffer[DType.int8, 1, MutableAnyOrigin]:
    debug_assert(align > 0, "align must be a positive integer value")

    if not weights:
        raise Error(
            "received null weights registry in mgp.buffer.constant.external"
        )

    var weight_ptr = weights[][pack_string_res(name_ptr, name_len)]
    if (Int(weight_ptr) % align) != 0:
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
](buf: NDBuffer[DType.uint8, 1, MutableAnyOrigin], vals: VariadicList[Int]):
    var ptr = buf.data.bitcast[Scalar[dtype]]()
    var offset: Int = 0
    for val in vals:
        ptr.store(offset, val)
        offset += 1


@register_internal("mgp.buffer.set_with_index")
@no_inline
fn mgp_buffer_set_with_index[
    bDevice: StaticString
](buffer: NDBuffer[DType.uint8, 1, MutableAnyOrigin], *vals: Int) raises:
    debug_assert(
        is_cpu[bDevice](), "set_with_index can only work on cpu buffers"
    )
    var bufSize = buffer.num_elements()
    var numArgs = len(vals)
    debug_assert(
        bufSize % numArgs == 0,
        "buffer size not divisible by number of index args",
    )

    var elSize = bufSize / numArgs
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
](buffer: NDBuffer[DType.uint8, 1, MutableAnyOrigin]) -> Bool:
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
    buffer: NDBuffer[DType.uint8, 1, MutableAnyOrigin]
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
    buffer: NDBuffer[DType.uint8, 1, MutableAnyOrigin], offset: Int, size: Int
) -> NDBuffer[DType.uint8, 1, MutableAnyOrigin]:
    return NDBuffer[DType.uint8, 1](buffer.data.offset(offset), Index(size))


@register_internal("mgp.buffer.concat")
@no_inline
fn mgp_buffer_concat[
    bDevice: StaticString
](
    output: NDBuffer[DType.uint8, 1, MutableAnyOrigin],
    inputs: StaticTuple[NDBuffer[DType.uint8, 1, MutableAnyOrigin], *_],
    call_ctx: DeviceContextPtr,
) raises:
    if len(output) < 4096:
        concat[1, DType.uint8, True, bDevice, None](
            output, 0, inputs, context=call_ctx
        )
    else:
        concat[1, DType.uint8, False, bDevice, None](
            output, 0, inputs, context=call_ctx
        )


@register_internal("mgp.buffer.device_to_host")
@no_inline
fn mgp_buffer_device_to_host[
    cOtherDevice: StaticString,
    dHostDevice: StaticString,
](
    dev_buf: NDBuffer[DType.uint8, 1, MutableAnyOrigin],
    host_buf: NDBuffer[DType.uint8, 1, MutableAnyOrigin],
    dev_ctx: DeviceContextPtr,
) raises:
    @parameter
    if is_cpu[dHostDevice]() and is_gpu[cOtherDevice]():
        dev_ctx[].enqueue_copy[DType.uint8](
            host_buf.data,
            DeviceBuffer[DType.uint8](
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
    src_buf: NDBuffer[DType.uint8, 1, MutableAnyOrigin],
    dst_buf: NDBuffer[DType.uint8, 1, MutableAnyOrigin],
    src_dev_ctx: DeviceContextPtr,
    dst_dev_ctx: DeviceContextPtr,
) raises:
    @parameter
    if is_gpu[cSrcDevice]() and is_gpu[dDstDevice]():
        dst_dev_ctx[].enqueue_copy[DType.uint8](
            DeviceBuffer[DType.uint8](
                dst_dev_ctx[],
                dst_buf.data,
                dst_buf.size(),
                owning=False,
            ),
            DeviceBuffer[DType.uint8](
                src_dev_ctx[],
                src_buf.data,
                src_buf.size(),
                owning=False,
            ),
        )
    elif is_cpu[cSrcDevice]() and is_cpu[dDstDevice]():
        memcpy(dst_buf.data, src_buf.data, src_buf.size())
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
    host_buf: NDBuffer[DType.uint8, 1, MutableAnyOrigin],
    dev_buf: NDBuffer[DType.uint8, 1, MutableAnyOrigin],
    dev_ctx: DeviceContextPtr,
) raises:
    @parameter
    if is_gpu[dOtherDevice]() and is_cpu[cHostDevice]():
        dev_ctx[].enqueue_copy[DType.uint8](
            DeviceBuffer[DType.uint8](
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
    storage_ref_addr: UnsafePointer[OpaquePointer],
    buffer_slot: UInt64,
) raises -> NDBuffer[DType.uint8, 1, MutableAnyOrigin]:
    var buffer_size: UInt64 = 0
    var buffer_data: OpaquePointer = external_call[
        "MGP_RT_GetCachedBuffer", OpaquePointer
    ](
        Int(buffer_slot),
        ctx.ctx_ptr,
        UnsafePointer(to=buffer_size),
        storage_ref_addr,
    )

    return NDBuffer[DType.uint8, 1](
        buffer_data.bitcast[UInt8](), Index(buffer_size)
    )


@register_internal("mgp.buffer.remove_cached")
@no_inline
fn mgp_buffer_remove_cached(ctx: StateContext, buffer_slot: UInt64):
    external_call["MGP_RT_RemoveCachedBuffer", NoneType](
        Int(buffer_slot), ctx.ctx_ptr
    )


@register_internal("mgp.buffer.get_size")
@no_inline
fn mgp_buffer_get_size(buf: NDBuffer[DType.uint8, 1, MutableAnyOrigin]) -> Int:
    return buf.num_elements()


@register_internal("destruct_async_refs")
@no_inline
fn destruct_async_refs(
    storage_ref_addr: UnsafePointer[OpaquePointer],
    size: Int,
    direct_ref: Bool,
):
    external_call["KGEN_CompilerRT_DestructAsyncRefs", NoneType](
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


@register_internal("mgp.tensor_spec.equal.static")
@no_inline
fn mgp_tensor_spec_equal_static[
    spec_rank: Int, *rawDims: Dim
](spec: IndexList[spec_rank]) -> Bool:
    var dims: VariadicList[Dim] = rawDims
    var numDims = len(dims)
    if spec_rank != numDims:
        return False
    for i in range(numDims):
        var dim = dims[i]
        var expectedDim = spec[i]
        if dim and dim != -1 and dim != expectedDim:
            return False

    return True


@register_internal("mgp.tensor_spec.get_dim")
@no_inline
fn mgp_tensor_spec_get_dim[
    spec_rank: Int, axis: UInt64
](spec: IndexList[spec_rank]) -> Int:
    constrained[
        axis < spec_rank,
        "axis for get_dim must be less than rank of TensorSpec",
    ]()
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
    buffer: NDBuffer[DType.uint8, 1, MutableAnyOrigin],
    shape: IndexList[spec_rank],
    label_ptr: UnsafePointer[Byte],
    label_len: UInt,
) raises:
    external_call["KGEN_CompilerRT_DebugTensorPrint", NoneType](
        label_ptr,
        label_len,
        dtype,
        UnsafePointer(to=shape.data),
        spec_rank,
        buffer.data,
        len(buffer),
    )


# ===-----------------------------------------------------------------------===#
# Mojo generation and general type lookups
# ===-----------------------------------------------------------------------===#


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
    rank: Int, //,
    io_spec: IOSpec[mut, input],
    static_spec: StaticTensorSpec[dtype, rank],
](
    ty: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
) -> ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]:
    return ty


@register_internal("list_of_tensor")
fn ListOfTensorDef[
    dtype: DType,
    rank: Int,
](
    ty: List[
        InputTensor[
            static_spec = StaticTensorSpec[dtype, rank].create_unknown()
        ]
    ]
) -> __type_of(ty):
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
    constrained[dtypes.size > 0]()

    var width = get_kernel_simd_width[dtypes[0], target]()

    @parameter
    for i in range(dtypes.size - 1):
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
    alias SpecType = StaticTensorSpec[dtype, rank]

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
    return __type_of(result)(array_of_specs)


# TODO: this should take IOSpec as a param -- will require graph compiler changes
# Used by the graph compiler to construct tensors from MGP repr. of tensor
@register_internal("to_managed_tensor_slice")
@always_inline
fn to_managed_tensor_slice[
    dtype: DType, rank: Int, mut: Bool, input: IO
](
    data: UnsafePointer[Scalar[dtype]],
    shape: UnsafePointer[Int],
) -> ManagedTensorSlice[
    io_spec = IOSpec[mut, input](),
    static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
]:
    var shape_ptr = shape
    var shape_tuple = IndexList[rank]()

    var stride_tuple = IndexList[rank]()
    var stride: Int = 1

    @parameter
    for i in reversed(range(rank)):
        # Start from the back so we can accumulate the strides.
        shape_tuple[i] = shape_ptr[i]
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    return ManagedTensorSlice[
        io_spec = IOSpec[mut, input](),
        static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
    ](data, shape_tuple, stride_tuple)


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
    func_type: AnyTrivialRegType, //,
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
    data: UnsafePointer[Scalar[dtype]],
    shape_tuple: IndexList[rank],
) -> ManagedTensorSlice[
    io_spec = IOSpec[mut, input](),
    static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
]:
    var stride_tuple = IndexList[rank]()
    var stride: Int = 1

    @parameter
    for i in reversed(range(rank)):
        # Start from the back so we can accumulate the strides.
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    return ManagedTensorSlice[
        io_spec = IOSpec[mut, input](),
        static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
    ](data, shape_tuple, stride_tuple)


# Helper method used by compiler to reconcile MGP list with dtype Mojo expects.
@register_internal("to_managed_tensor_slice_list")
@always_inline
fn to_managed_tensor_slice_list[
    dtype: DType, rank: Int, mut: Bool, input: IO
](
    raw_list_ptr: OpaquePointer,
) -> List[
    ManagedTensorSlice[
        io_spec = IOSpec[mut, input](),
        static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
    ]
]:
    var num_elements = external_call["MGP_RT_ListSize", Int64](
        raw_list_ptr
    ).__int__()

    var data_ptrs = List[OpaquePointer](capacity=num_elements)
    var dim_values = List[Int64](capacity=num_elements * rank)

    # Collect the data pointers and dimensions of each element from the list.
    external_call["MGP_RT_ListPopulate", NoneType](
        raw_list_ptr, data_ptrs.unsafe_ptr(), dim_values.unsafe_ptr()
    )

    # TODO: revisit the use of unknown here
    # Create output list
    var out_list = List[
        ManagedTensorSlice[
            io_spec = IOSpec[mut, input](),
            static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
        ]
    ](capacity=num_elements)

    # Convert individual elements of the input list into NDBuffer, and
    # accumulate the results to output list.
    for i in range(num_elements):
        var data = data_ptrs[i].bitcast[Scalar[dtype]]()

        var dims = IndexList[rank]()

        @parameter
        for dim in range(rank):
            dims[dim] = dim_values[dim + i * rank].__int__()

        var buffer = _to_managed_tensor_slice_index_list_shape[
            dtype, rank, mut, input
        ](data, dims)
        out_list.append(buffer)

    return out_list^


# ===----------------------------------------------------------------------===#
# Affine view kernel implementations
# ===----------------------------------------------------------------------===#


@register_internal("split_dim_indices")
@always_inline
fn split_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], new_shape_dim: Int64) -> IndexList[rank + 1]:
    var out = IndexList[rank + 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [40, 30, 2] and we reshape it to [5, 8, 30, 2].
    # If we are accessing the index [21, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [2, 5, 16, 1].
    # Or [21 // 8, 21 % 8, ...old dims...].
    # In this case, the axis = 0 and the new_shape_dim = 8.

    @parameter
    for i in range(rank + 1):

        @parameter
        if i == axis:
            out[i] = indices[axis] // Int(new_shape_dim)
        elif i == axis + 1:
            out[i] = indices[axis] % Int(new_shape_dim)
        elif i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]

    return out


@register_internal("merge_dim_indices")
@always_inline
fn merge_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], old_shape_dim: Int64) -> IndexList[rank - 1]:
    var out = IndexList[rank - 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [5, 8, 30, 2] and we reshape it to [40, 30, 2].
    # If we are accessing the index [2, 5, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [21, 16, 1].
    # Or [2 * 8 + 5, 16, 1].
    # In this case, the axis = 0 and the old_shape_dim = 8.

    @parameter
    for i in range(rank - 1):

        @parameter
        if i == axis:
            out[i] = fma(indices[i], Int(old_shape_dim), indices[i + 1])
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

    @parameter
    for i in range(rank + 1):

        @parameter
        if i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]
        else:
            out[i] = value

    return out


# ===-----------------------------------------------------------------------===#
# Opaque Test Primitives
# ===-----------------------------------------------------------------------===#


struct MyInt(Movable):
    var val: Int

    fn __init__(out self, val: Int):
        self.val = val

    fn __moveinit__(out self, deinit other: MyInt):
        print("MyInt.__moveinit__", other.val)
        self.val = other.val

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


@register_passable("trivial")
struct MyIntReg(Copyable, Movable):
    var val: Int

    fn __init__(out self, val: Int):
        self.val = val


@register_internal("testfuse.my_int_reg.square")
@no_inline
fn test_my_int_reg_square(x: MyIntReg) -> MyIntReg:
    return MyIntReg(x.val * x.val)


@register_passable
struct MyIntReg2(Copyable, Movable):
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
