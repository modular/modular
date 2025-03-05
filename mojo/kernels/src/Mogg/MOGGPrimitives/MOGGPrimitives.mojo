# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg, List
from collections.string import StringSlice
from math import ceildiv
from os import abort
from sys import alignof, external_call, sizeof

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host.info import is_cpu, is_gpu
from memory import UnsafePointer, memcpy, Span
from memory.memory import _malloc as _malloc_cpu
from nn.concat import concat
from register import *
from runtime.asyncrt import DeviceContextPtr
from weights_registry import WeightsRegistry

from utils import Index, IndexList, StaticTuple

from .MOGGIntList import IntList

# ===-----------------------------------------------------------------------===#
# Helper Structures
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct StaticTensorSpec[rank: Int]():
    """Defines a static-rank tensor spec which - has a static-rank shape
    in a IndexList and dtype. This is analagous to TensorSpec from
    tensor_spec, but this is fully static and will not have any allocations."""

    var shape: IndexList[rank]
    var type: DType

    @always_inline
    fn __init__(out self, shape: IndexList[rank, **_], type: DType):
        """Constructs a static tensor spec with a static rank shape and type.

        Args:
            shape: The shape of static rank we are creating tensor spec with.
            type: The DType we are creating tensor spec with.
        """
        self.shape = shape.canonicalize()
        self.type = type

    @always_inline
    fn __len__(self) -> Int:
        """Returns the size of the StaticTensorSpec.

        Returns:
            The flattened size of the shape.
        """
        return self.shape.flattened_length()

    @always_inline
    fn bytecount(self) -> Int:
        """Returns the byte size of the StaticTensorSpec.

        Returns:
            The byte size of the tensor-spec.
        """
        return len(self) * self.type.sizeof()

    @always_inline
    fn __eq__(self, rhs: StaticTensorSpec[rank]) -> Bool:
        """Compares this StaticTensorSpec to another StaticTensorSpec for equality.

        The StaticTensorSpec are equal if the shapes are equal and the types are
        also equal.

        Args:
            rhs: The other StaticTensorSpec.

        Returns:
            The comparison result.
        """
        return self.shape == rhs.shape and self.type == rhs.type


@register_passable("trivial")
struct StateContext:
    """Defines a StateContext structure which holds a ptr to context and has accessors that go to external calls
    This is currently meant as a mojo-side container for GML::StateContext."""

    var num_slots: Int
    var ctx_ptr: UnsafePointer[NoneType]

    @always_inline
    fn __init__(out self, num_slots: Int, ctx_ptr: UnsafePointer[NoneType]):
        self.num_slots = num_slots
        self.ctx_ptr = ctx_ptr

    @always_inline
    fn __getitem__(self, index: Int) -> UnsafePointer[NoneType]:
        debug_assert(0 <= index < self.num_slots, "index must be within bounds")
        return external_call[
            "KGEN_CompilerRT_GetContextPayloadPtr",
            UnsafePointer[NoneType],
        ](index, self.ctx_ptr)


fn pack_string_res(
    str_ptr: UnsafePointer[Byte], str_len: UInt
) raises -> String:
    var span = Span[Byte, ImmutableAnyOrigin](
        ptr=str_ptr,
        length=str_len,
    )
    # We can not free the resource ptr embedded in MEF, create a copy
    return StringSlice.from_utf8(span).__str__()


# ===-----------------------------------------------------------------------===#
# Async Packing/Unpacking functions
# ===-----------------------------------------------------------------------===#


@register_internal("builtin.create_error_async_values_and_destruct_error")
@no_inline
fn create_error_async_values_and_destruct_error(
    async_ptr: UnsafePointer[UnsafePointer[NoneType]],
    async_len: Int,
    owned err: Error,
):
    """Indicates to the C++ runtime that the kernel has failed."""
    var str = err.__str__()
    var strslice = str.as_string_slice()
    external_call["KGEN_CompilerRT_AsyncRT_CreateAsyncs_Error", NoneType](
        async_ptr,
        async_len,
        strslice.unsafe_ptr(),
        strslice.byte_length(),
    )


@register_internal("builtin.create_index_async")
@no_inline
fn create_index_async(value: Int, async_ptr: UnsafePointer[NoneType]):
    external_call["KGEN_CompilerRT_CreateAsync_ssizet", NoneType](
        value, async_ptr
    )


@register_internal("builtin.create_si64_async")
@no_inline
@export
fn create_si64_async(value: Int64, async_ptr: UnsafePointer[NoneType]):
    external_call["KGEN_CompilerRT_CreateAsync_int64t", NoneType](
        value, async_ptr
    )


@register_internal("builtin.create_chain_async")
@no_inline
fn create_chain_async(async_ptr: UnsafePointer[NoneType]):
    external_call["KGEN_CompilerRT_CreateAsync_chain", NoneType](async_ptr)


@register_internal("builtin.create_bool_async")
@register_internal("builtin.create_i1_async")
@no_inline
fn create_i1_async(
    value: Bool,
    async_ptr: UnsafePointer[NoneType],
):
    external_call["KGEN_CompilerRT_CreateAsync_bool", NoneType](
        value, async_ptr
    )


@register_internal("builtin.create_buffer_ref_async")
@no_inline
fn create_buffer_ref_async(
    buffer: NDBuffer[DType.int8, 1],
    async_ptr: UnsafePointer[NoneType],
    call_ctx: DeviceContextPtr,
):
    external_call["KGEN_CompilerRT_CreateAsyncDeviceBufferRef", NoneType](
        buffer.data, len(buffer), async_ptr, call_ctx.handle_
    )


@register_internal("builtin.create_non_tracked_buffer_ref_async")
@no_inline
fn create_non_tracked_buffer_ref_async(
    buffer: NDBuffer[DType.int8, 1], async_ptr: UnsafePointer[NoneType]
):
    external_call["KGEN_CompilerRT_CreateAsyncNonTrackedBufferRef", NoneType](
        buffer.data, len(buffer), async_ptr
    )


@register_internal("builtin.create_buffer_ref_with_borrow_async")
@no_inline
fn create_buffer_ref_with_borrow_async[
    borrowee_type: Int,
](
    buffer: NDBuffer[DType.int8, 1],
    async_to_borrow: UnsafePointer[NoneType],
    output_async: UnsafePointer[NoneType],
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
](spec: StaticTensorSpec[spec_rank], async_ptr: UnsafePointer[NoneType],):
    # Mojo impl is bitwise compatible with cpp variant, can construct TensorSpec in mojo
    # and pass it back to C++ -- However, this is an issue for the heap allocated dims.
    # For the benefit of simplicity, allocate the shapes and ptrs and free explicitly after
    var shape_ptr = UnsafePointer[Int].alloc(spec_rank)
    for i in range(spec_rank):
        shape_ptr[i] = spec.shape[i]
    external_call["KGEN_CompilerRT_CreateAsyncTensorSpec", NoneType](
        shape_ptr, spec_rank, spec.type, async_ptr
    )
    shape_ptr.free()


@register_internal("builtin.create_tensor_with_borrow_async")
@no_inline
fn create_tensor_async[
    tensor_rank: Int,
    buffer_rank: Int,
    type: DType,
    borrowee_type: Int,
](
    buffer: NDBuffer[type, buffer_rank],
    async_to_borrow: UnsafePointer[NoneType],
    output_async: UnsafePointer[NoneType],
):
    # Tensor and the underlying buffer must have the same rank, unless it is a
    # scalar tensor stored with a NDBuffer<[1]>
    constrained[
        tensor_rank == buffer_rank or (tensor_rank == 0 and buffer_rank == 1)
    ]()
    var spec = StaticTensorSpec(buffer.dynamic_shape, type)
    external_call["KGEN_CompilerRT_CreateAsyncTensorWithBorrow", NoneType](
        buffer.data,
        spec.bytecount(),
        tensor_rank,
        UnsafePointer.address_of(buffer.dynamic_shape.data.array),
        type,
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
    async_ptr: UnsafePointer[NoneType],
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
    async_ptr: UnsafePointer[NoneType],
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
    async_src: UnsafePointer[NoneType],
    async_dst: UnsafePointer[NoneType],
):
    external_call[
        "KGEN_CompilerRT_TransferAsyncRef",
        NoneType,
    ](async_src, async_dst)


@register_internal("builtin.unpack_async")
@no_inline
fn unpack_async(
    async_ptr: UnsafePointer[NoneType],
) -> UnsafePointer[NoneType]:
    return external_call[
        "KGEN_CompilerRT_GetValueFromAsync",
        UnsafePointer[NoneType],
    ](async_ptr)


@register_internal("builtin.unpack_device_ctx")
@no_inline
fn unpack_device_ctx(
    async_ptr: UnsafePointer[NoneType],
) -> DeviceContextPtr:
    var ptr = external_call[
        "KGEN_CompilerRT_UnpackDeviceContext",
        UnsafePointer[NoneType],
    ](async_ptr)

    return DeviceContextPtr(ptr)


@register_internal("builtin.unpack_buffer_ref")
@no_inline
fn unpack_buffer_ref(
    async_ptr: UnsafePointer[NoneType],
) -> NDBuffer[DType.uint8, 1]:
    var size: UInt64 = 0
    var data_ptr = external_call[
        "KGEN_CompilerRT_GetDataFromBuffer",
        UnsafePointer[NoneType],
    ](async_ptr, UnsafePointer.address_of(size))
    var shape = IndexList[1](Int(size))
    return NDBuffer[DType.uint8, 1](data_ptr.bitcast[UInt8](), shape)


@register_internal("builtin.unpack_tensor")
@no_inline
fn unpack_tensor[
    buffer_rank: Int,
    tensor_rank: Int,
    type: DType,
](tensor_async_ptr: UnsafePointer[NoneType]) -> NDBuffer[type, buffer_rank]:
    # Tensor and the underlying buffer must have the same rank, unless it is a
    # scalar tensor stored with a NDBuffer<[1]>
    constrained[
        tensor_rank == buffer_rank or (tensor_rank == 0 and buffer_rank == 1)
    ]()
    var shapes = IndexList[buffer_rank]()
    var buffer_ptr = external_call[
        "KGEN_CompilerRT_GetShapeAndDataFromTensor",
        UnsafePointer[NoneType],
    ](
        UnsafePointer.address_of(shapes.data.array),
        tensor_async_ptr,
    )

    @parameter
    if tensor_rank == 0:
        shapes[0] = 1

    return NDBuffer[type, buffer_rank](
        buffer_ptr.bitcast[Scalar[type]](), shapes
    )


@register_internal("builtin.unpack_tensor_spec")
@no_inline
fn unpack_tensor_spec[
    spec_rank: Int
](async_ptr: UnsafePointer[NoneType]) -> StaticTensorSpec[spec_rank]:
    var shape_ptr = UnsafePointer[Int].alloc(spec_rank)
    var raw_dtype = external_call[
        "KGEN_CompilerRT_GetTensorSpecFromAsync",
        UInt8,
    ](shape_ptr, spec_rank, async_ptr)
    var shape = IndexList[spec_rank]()
    for i in range(spec_rank):
        shape[i] = Int(shape_ptr[i])
    shape_ptr.free()
    return StaticTensorSpec[spec_rank](shape, DType._from_ui8(raw_dtype.value))


@register_internal("builtin.unpack_context")
@no_inline
fn unpack_context(
    async_ptr: UnsafePointer[NoneType],
) -> StateContext:
    # We want to construct this because we want all payloads to be implemented
    var num_slots: UInt64 = 0
    var ctx_ptr: UnsafePointer[NoneType] = external_call[
        "KGEN_CompilerRT_GetContextAndSizeFromAsync",
        UnsafePointer[NoneType],
    ](UnsafePointer.address_of(num_slots), async_ptr)
    return StateContext(Int(num_slots), ctx_ptr)


@register_internal("builtin.get_buffer_data")
@always_inline
fn get_buffer_data(buffer: NDBuffer[DType.uint8, 1]) -> UnsafePointer[UInt8]:
    return buffer.data


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
    type: DType,
](
    buffer: NDBuffer[DType.uint8, 1],
    spec: StaticTensorSpec[spec_rank],
) -> NDBuffer[type, buffer_rank]:
    debug_assert(type == spec.type)

    @parameter
    if spec_rank == 0:
        # We promote scalar tensor to tensor<[1]>
        constrained[buffer_rank == 1]()
        return NDBuffer[type, buffer_rank](
            buffer.data.bitcast[Scalar[type]](),
            rebind[IndexList[buffer_rank]](IndexList[1](1)),
        )
    else:
        constrained[spec_rank == buffer_rank]()
        return NDBuffer[type, buffer_rank](
            buffer.data.bitcast[Scalar[type]](),
            rebind[IndexList[buffer_rank]](spec.shape),
        )


@register_internal("mgp.tensor.extract.tensor_spec")
@no_inline
fn mgp_tensor_extract_tensor_spec[
    tensor_rank: Int,
    buffer_rank: Int,
    type: DType,
](
    buffer: NDBuffer[type, buffer_rank],
    out result: StaticTensorSpec[tensor_rank],
):
    @parameter
    if tensor_rank == 0:
        constrained[buffer_rank == 1]()
        return rebind[__type_of(result)](
            StaticTensorSpec[0](IndexList[0](), type)
        )
    else:
        constrained[buffer_rank == tensor_rank]()
        return rebind[__type_of(result)](
            StaticTensorSpec[buffer_rank](buffer.dynamic_shape, type)
        )


@register_internal("mgp.tensor.extract.buffer")
@no_inline
fn mgp_tensor_extract_buffer[
    tensor_rank: Int,
    buffer_rank: Int,
    type: DType,
](buffer: NDBuffer[type, buffer_rank]) -> NDBuffer[DType.uint8, 1]:
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
) raises -> NDBuffer[DType.int8, 1]:
    # Default to alignment of 0 which means kPreferredMemoryAlignment if cRawAlign is kUnknownSize (SizeUtils.h).
    # alias alignment = 0 if bRawAlign == UInt64.MAX else Int(bRawAlign)

    # This primitive has a byte-size input, so always assume a byte format
    var shape = IndexList[1](byte_size)
    var buf = dev_context[].enqueue_create_buffer[DType.int8](byte_size)
    return NDBuffer[DType.int8, 1](buf^.take_ptr(), shape)


@register_internal("mgp.buffer.constant")
@export
fn mgp_buffer_constant(
    resource_ptr: UnsafePointer[NoneType],
    resource_bytecount: Int,
    out result: NDBuffer[DType.int8, 1],
):
    # Should we keep the alignment? It seems that the static alignment is
    # dropped in the kernels anyway.
    return __type_of(result)(resource_ptr.bitcast[Int8](), resource_bytecount)


@register_internal("mgp.buffer.constant.external")
fn mgp_buffer_constant_external(
    weights: UnsafePointer[WeightsRegistry],
    name_ptr: UnsafePointer[Byte],
    name_len: UInt,
    size: UInt64,
    align: UInt64,
) raises -> NDBuffer[DType.int8, 1]:
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
    type: DType
](buf: NDBuffer[DType.uint8, 1], vals: VariadicList[Int]):
    var ptr = buf.data.bitcast[Scalar[type]]()
    var offset: Int = 0
    for val in vals:
        ptr.store(offset, val)
        offset += 1


@register_internal("mgp.buffer.set_with_index")
@no_inline
fn mgp_buffer_set_with_index[
    bDevice: StringLiteral
](buffer: NDBuffer[DType.uint8, 1], *vals: Int) raises:
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
    bDevice: StringLiteral
](buffer: NDBuffer[DType.uint8, 1]) -> Bool:
    debug_assert(is_cpu[bDevice](), "to_bool can only work on cpu buffers")
    var bufSize = buffer.num_elements()
    debug_assert(
        bufSize == 1,
        "buffer size must be a size of 1",
    )
    return buffer[0] != 0


@register_internal("mgp.buffer.to_index")
@no_inline
fn mgp_buffer_to_index(buffer: NDBuffer[DType.uint8, 1]) raises -> Int:
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
    buffer: NDBuffer[DType.uint8, 1], offset: Int, size: Int
) -> NDBuffer[DType.uint8, 1]:
    return NDBuffer[DType.uint8, 1](buffer.data.offset(offset), Index(size))


@register_internal("mgp.buffer.concat")
@no_inline
fn mgp_buffer_concat[
    bDevice: StringLiteral
](
    output: NDBuffer[DType.uint8, 1],
    inputs: StaticTuple[NDBuffer[DType.uint8, 1], *_],
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
    cOtherDevice: StringLiteral,
    dHostDevice: StringLiteral,
](
    dev_buf: NDBuffer[DType.uint8, 1],
    host_buf: NDBuffer[DType.uint8, 1],
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
    cSrcDevice: StringLiteral,
    dDstDevice: StringLiteral,
](
    src_buf: NDBuffer[DType.uint8, 1],
    dst_buf: NDBuffer[DType.uint8, 1],
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
            " types (cpu-cpu) or (gpu-gpu)"
        )


@register_internal("mgp.buffer.host_to_device")
@no_inline
fn mgp_buffer_host_to_device[
    cHostDevice: StringLiteral,
    dOtherDevice: StringLiteral,
](
    host_buf: NDBuffer[DType.uint8, 1],
    dev_buf: NDBuffer[DType.uint8, 1],
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
    storage_ref_addr: UnsafePointer[UnsafePointer[NoneType]],
    buffer_slot: UInt64,
) raises -> NDBuffer[DType.uint8, 1]:
    var buffer_size: UInt64 = 0
    var buffer_data: UnsafePointer[NoneType] = external_call[
        "MGP_RT_GetCachedBuffer", UnsafePointer[NoneType]
    ](
        Int(buffer_slot),
        ctx.ctx_ptr,
        UnsafePointer.address_of(buffer_size),
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
fn mgp_buffer_get_size(buf: NDBuffer[DType.uint8, 1]) -> Int:
    return buf.num_elements()


@register_internal("destruct_async_refs")
@no_inline
fn destruct_async_refs(
    storage_ref_addr: UnsafePointer[UnsafePointer[NoneType]],
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
    bRawDType: UInt8,
    aRawDims: DimList,
    aRawDimsRank: Int,
](*runtimeDims: Int) -> StaticTensorSpec[aRawDimsRank]:
    var type = DType._from_ui8(bRawDType.value)
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
    return StaticTensorSpec[aRawDimsRank](shape, type)


@register_internal("mgp.tensor_spec.equal.static")
@no_inline
fn mgp_tensor_spec_equal_static[
    spec_rank: Int, *rawDims: Dim
](spec: StaticTensorSpec[spec_rank]) -> Bool:
    var dims: VariadicList[Dim] = rawDims
    var numDims = len(dims)
    if spec_rank != numDims:
        return False
    for i in range(numDims):
        var dim = dims[i]
        var expectedDim = spec.shape[i]
        if dim and dim != -1 and dim != expectedDim:
            return False

    return True


@register_internal("mgp.tensor_spec.get_dim")
@no_inline
fn mgp_tensor_spec_get_dim[
    spec_rank: Int, axis: UInt64
](spec: StaticTensorSpec[spec_rank]) -> Int:
    constrained[
        axis < spec_rank,
        "axis for get_dim must be less than rank of TensorSpec",
    ]()
    return spec.shape[Int(axis)]


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
    aDebugString: StringLiteral,
    bLabel: StringLiteral,
](ctx: StateContext,) raises:
    prefix = ""
    if bLabel:
        prefix = "[" + bLabel + "] "
    print(prefix + aDebugString)


@register_internal("mgp.debug.tensor.print")
@no_inline
fn mgp_debug_tensor_print[
    spec_rank: Int,
    type: DType,
](
    buffer: NDBuffer[DType.uint8, 1],
    spec: StaticTensorSpec[spec_rank],
    label_ptr: UnsafePointer[Byte],
    label_len: UInt,
) raises:
    external_call["KGEN_CompilerRT_DebugTensorPrint", NoneType](
        label_ptr,
        label_len,
        type,
        UnsafePointer.address_of(spec.shape.data.array),
        spec_rank,
        buffer.data,
        len(buffer),
    )


# ===-----------------------------------------------------------------------===#
# Opaque Test Primitives
# ===-----------------------------------------------------------------------===#


struct MyInt(Movable):
    var val: Int

    @implicit
    fn __init__(out self, val: Int):
        self.val = val

    fn __moveinit__(out self, owned other: MyInt):
        print("MyInt.__moveinit__", other.val)
        self.val = other.val

    fn __del__(owned self):
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


@value
@register_passable("trivial")
struct MyIntReg:
    var val: Int

    @implicit
    fn __init__(out self, val: Int):
        self.val = val


@register_internal("testfuse.my_int_reg.square")
@no_inline
fn test_my_int_reg_square(x: MyIntReg) -> MyIntReg:
    return MyIntReg(x.val * x.val)


@value
@register_passable
struct MyIntReg2:
    var val: Int

    @implicit
    fn __init__(out self, val: Int):
        self.val = val

    fn __del__(owned self):
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
