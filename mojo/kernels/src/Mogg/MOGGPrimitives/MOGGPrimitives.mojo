# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from collections.vector import InlinedFixedVector
from math import ceildiv
from os import abort
from sys import alignof, external_call, sizeof

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu.host import Context as CudaContext
from gpu.host import (
    CudaInstance,
    Device,
    DeviceBuffer,
    DeviceContext,
    Event,
)
from gpu.host._utils import _check_error
from gpu.host.cuda_instance import *
from memory import UnsafePointer, memcpy
from memory.memory import _malloc as _malloc_cpu
from MOGGIntList import IntList
from nn.concat import concat
from register import *
from runtime.asyncrt import MojoCallContextPtr
from weights_registry import WeightsRegistry

from utils import Index, IndexList

# ===----------------------------------------------------------------------===#
# Helper Structures
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct StaticTensorSpec[rank: Int]():
    """Defines a static-rank tensor spec which - has a static-rank shape
    in a IndexList and dtype. This is analagous to TensorSpec from
    tensor_spec, but this is fully static and will not have any allocations."""

    var shape: IndexList[rank]
    var type: DType

    @always_inline
    fn __init__(inout self, shape: IndexList[rank, **_], type: DType):
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
    fn __init__(inout self, num_slots: Int, ctx_ptr: UnsafePointer[NoneType]):
        self.num_slots = num_slots
        self.ctx_ptr = ctx_ptr

    @always_inline
    fn __getitem__(self, index: Int) -> UnsafePointer[NoneType]:
        debug_assert(0 <= index < self.num_slots, "index must be within bounds")
        return external_call[
            "KGEN_CompilerRT_GetContextPayloadPtr",
            UnsafePointer[NoneType],
        ](index, self.ctx_ptr)


# ===----------------------------------------------------------------------===#
# Helper functions
# ===----------------------------------------------------------------------===#


@always_inline
fn byte_buffer_alloc[
    target: StringLiteral,
    alignment: Int,
](
    byte_size: Int,
    device_context: UnsafePointer[DeviceContext],
    call_ctx: MojoCallContextPtr,
) raises -> NDBuffer[DType.int8, 1]:
    """Function will allocate a 1-D buffer with the specified size/alignment on device.
    """
    # This primitive has a byte-size input, so always assume a byte format
    var shape = IndexList[1](byte_size)

    @parameter
    if "cuda" in target:
        # For now, only cuda targets can use device context directly
        var buf = device_context[].create_buffer[DType.int8](byte_size)
        return NDBuffer[DType.int8, 1](buf^.take_ptr(), shape)
    else:
        return NDBuffer[DType.int8, 1](
            call_ctx.alloc(byte_size, alignment).bitcast[Int8](),
            shape,
        )


# ===----------------------------------------------------------------------===#
# Async Packing/Unpacking functions
# ===----------------------------------------------------------------------===#


@mogg_register("builtin.create_errror_async_values_and_destruct_error")
@always_inline
@export
fn create_errror_async_values_and_destruct_error[
    len: Int
](
    ctx: MojoCallContextPtr,
    ptr: UnsafePointer[UnsafePointer[NoneType]],
    runtime: UnsafePointer[NoneType],
    owned err: Error,
):
    """Indicates to the C++ runtime that the kernel has failed."""
    var str = err.__str__()
    var strslice = str.as_string_slice()
    external_call["KGEN_CompilerRT_AsyncRT_CreateAsyncs_Error", NoneType](
        ctx, ptr, len, runtime, strslice.unsafe_ptr(), strslice.byte_length()
    )


@mogg_register("builtin.create_index_async")
@always_inline
@export
fn create_index_async(
    value: Int,
    async_ptr: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
):
    external_call["KGEN_CompilerRT_CreateAsync_ssizet", NoneType](
        value, async_ptr, runtime
    )


@mogg_register("builtin.create_chain_async")
@always_inline
@export
fn create_chain_async(
    async_ptr: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
):
    external_call["KGEN_CompilerRT_CreateAsync_chain", NoneType](
        async_ptr, runtime
    )


@mogg_register("builtin.create_i1_async")
@always_inline
@export
fn create_i1_async(
    value: Bool,
    async_ptr: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
):
    external_call["KGEN_CompilerRT_CreateAsync_bool", NoneType](
        value, async_ptr, runtime
    )


# TODO: this should contain a pointer or reference to the DeviceContext, NOT
# a copy of it. This is not possible until GRA-902 is resolved.
alias DeviceBufferMojoValueType = Tuple[DeviceContext, UnsafePointer[Int8]]


fn _destroy_device_buffer(ptr: UnsafePointer[NoneType]):
    var cast_ptr = ptr.bitcast[DeviceBufferMojoValueType]()
    var ctx = cast_ptr[][0]
    var data = cast_ptr[][1]
    _ = DeviceBuffer(ctx, data, 0, owning=True)


@mogg_register("builtin.create_buffer_ref_async")
@always_inline
@export
fn create_buffer_ref_async[
    target: StringLiteral
](
    buffer: NDBuffer[DType.int8, 1],
    async_ptr: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
    call_ctx: MojoCallContextPtr,
):
    # DeviceContext does not support CPU so handle this specially.
    # We could also use the MojoValue approach below for CPU, but it is harder
    # to destroy the buffer in mojo because the runtime (which holds the allocator)
    # is not currently available in mojo.
    @parameter
    if target == "cpu":
        external_call["KGEN_CompilerRT_CreateAsyncBufferRef", NoneType](
            buffer.data, len(buffer), async_ptr, runtime
        )
        return

    # Otherwise, create a MojoValue containing the DeviceContext, which is used
    # to free the data pointer.
    alias size = sizeof[DeviceBufferMojoValueType]()
    alias align = alignof[DeviceBufferMojoValueType]()

    var mojo_value_ptr = external_call[
        "KGEN_CompilerRT_MojoValueAllocateBuffer",
        UnsafePointer[DeviceBufferMojoValueType],
    ](size, align)
    # Note: We need to make a copy of the DeviceContext here because the graph
    # compiler does not share the same DeviceContext object as the MAX Driver
    # (GRA-902). The members are shared so this is OK.
    # The DeviceContext is currently really big (1700B) so this needs to be fixed.
    mojo_value_ptr.init_pointee_move(
        (call_ctx.get_device_context(), buffer.data)
    )

    external_call["KGEN_CompilerRT_CreateAsyncMojoValueBufferRef", NoneType](
        buffer.data,
        len(buffer),
        mojo_value_ptr,
        _destroy_device_buffer,
        async_ptr,
        runtime,
    )


@mogg_register("builtin.create_non_tracked_buffer_ref_async")
@always_inline
@export
fn create_non_tracked_buffer_ref_async[
    target: StringLiteral
](
    buffer: NDBuffer[DType.int8, 1],
    async_ptr: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
    call_ctx: MojoCallContextPtr,
):
    external_call["KGEN_CompilerRT_CreateAsyncNonTrackedBufferRef", NoneType](
        buffer.data, len(buffer), async_ptr, runtime
    )


@mogg_register("builtin.create_buffer_ref_with_borrow_async")
@always_inline
@export
fn create_buffer_ref_with_borrow_async[
    borrow_from_storage_handle: Bool, target: StringLiteral
](
    buffer: NDBuffer[DType.int8, 1],
    async_to_borrow: UnsafePointer[NoneType],
    output_async: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
):
    external_call["KGEN_CompilerRT_CreateAsyncBufferWithBorrow", NoneType](
        buffer.data,
        len(buffer),
        async_to_borrow,
        borrow_from_storage_handle,
        output_async,
        runtime,
    )


@mogg_register("builtin.create_tensor_spec_async")
@always_inline
@export
fn create_tensor_spec_async[
    spec_rank: Int
](
    spec: StaticTensorSpec[spec_rank],
    async_ptr: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
):
    # Mojo impl is bitwise compatible with cpp variant, can construct TensorSpec in mojo
    # and pass it back to C++ -- However, this is an issue for the heap allocated dims.
    # For the benefit of simplicity, allocate the shapes and ptrs and free explicitly after
    var shape_ptr = UnsafePointer[Int].alloc(spec_rank)
    for i in range(spec_rank):
        shape_ptr[i] = spec.shape[i]
    external_call["KGEN_CompilerRT_CreateAsyncTensorSpec", NoneType](
        shape_ptr, spec_rank, spec.type, async_ptr, runtime
    )
    shape_ptr.free()


@mogg_register("builtin.create_tensor_with_borrow_async")
@always_inline
@export
fn create_tensor_async[
    tensor_rank: Int,
    buffer_rank: Int,
    type: DType,
    target: StringLiteral,
    borrow_from_storage_handle: Bool,
](
    buffer: NDBuffer[type, buffer_rank],
    async_to_borrow: UnsafePointer[NoneType],
    output_async: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
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
        borrow_from_storage_handle,
        output_async,
        runtime,
    )
    pass


@export
fn empty_destructor(ptr: UnsafePointer[UInt8]):
    pass


@mogg_register("builtin.create_mojo_value_async")
@always_inline
@export
fn create_mojo_value_async(
    val_ptr: UnsafePointer[UInt8],
    async_ptr: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
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
            runtime,
        )
        return
    var dst_ptr = external_call[
        "KGEN_CompilerRT_MojoValueAllocateBuffer", UnsafePointer[UInt8]
    ](size, align)
    move_fn(dst_ptr, val_ptr)

    external_call["KGEN_CompilerRT_CreateOwnedAsyncMojoValue", NoneType](
        dst_ptr,
        destructor_fn,
        async_ptr,
        runtime,
    )


@mogg_register("builtin.create_python_mojo_value_async")
@always_inline
@export
fn create_python_mojo_value_async(
    val_ptr: UnsafePointer[UInt8],
    async_ptr: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
    size: Int,
    align: Int,
    destructor_fn: fn (UnsafePointer[UInt8]) -> None,
    move_fn: fn (UnsafePointer[UInt8], UnsafePointer[UInt8]) -> None,
):
    var dst_ptr = external_call[
        "KGEN_CompilerRT_MojoValueAllocateBuffer", UnsafePointer[UInt8]
    ](size, align)
    move_fn(dst_ptr, val_ptr)

    external_call["KGEN_CompilerRT_CreateOwnedAsyncPythonMojoValue", NoneType](
        dst_ptr,
        destructor_fn,
        async_ptr,
        runtime,
    )


@mogg_register("builtin.unpack_async")
@always_inline
@export
fn unpack_async(
    async_ptr: UnsafePointer[NoneType],
) -> UnsafePointer[NoneType]:
    return external_call[
        "KGEN_CompilerRT_GetValueFromAsync",
        UnsafePointer[NoneType],
    ](async_ptr)


@mogg_register("builtin.unpack_buffer_ref")
@always_inline
@export
fn unpack_buffer_ref[
    target: StringLiteral
](async_ptr: UnsafePointer[NoneType],) -> NDBuffer[DType.uint8, 1]:
    var size: UInt64 = 0
    var data_ptr = external_call[
        "KGEN_CompilerRT_GetDataFromBuffer",
        UnsafePointer[NoneType],
    ](async_ptr, UnsafePointer.address_of(size))
    var shape = IndexList[1](int(size))
    return NDBuffer[DType.uint8, 1](data_ptr.bitcast[UInt8](), shape)


@mogg_register("builtin.unpack_tensor")
@always_inline
@export
fn unpack_tensor[
    buffer_rank: Int,
    tensor_rank: Int,
    type: DType,
    target: StringLiteral,
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

    return NDBuffer[type, buffer_rank](buffer_ptr.bitcast[type](), shapes)


@mogg_register("builtin.unpack_tensor_spec")
@always_inline
@export
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
        shape[i] = int(shape_ptr[i])
    shape_ptr.free()
    return StaticTensorSpec[spec_rank](shape, DType._from_ui8(raw_dtype.value))


@mogg_register("builtin.unpack_context")
@always_inline
@export
fn unpack_context(
    async_ptr: UnsafePointer[NoneType],
) -> StateContext:
    # We want to construct this because we want all payloads to be implemented
    var num_slots: UInt64 = 0
    var ctx_ptr: UnsafePointer[NoneType] = external_call[
        "KGEN_CompilerRT_GetContextAndSizeFromAsync",
        UnsafePointer[NoneType],
    ](UnsafePointer.address_of(num_slots), async_ptr)
    return StateContext(int(num_slots), ctx_ptr)


@mogg_register("builtin.get_buffer_data")
@always_inline
@export
fn get_buffer_data(buffer: NDBuffer[DType.uint8, 1]) -> UnsafePointer[UInt8]:
    return buffer.data


# ===----------------------------------------------------------------------===#
# MGP Common Primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mgp.assert")
@always_inline
@export
fn mgp_assert[message: StringLiteral](cond: Bool) raises -> Int:
    if not cond:
        raise Error(message)
    return 0


# ===----------------------------------------------------------------------===#
# MGP Tensor Primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mgp.tensor.create")
@always_inline
fn mgp_tensor_create[
    spec_rank: Int,
    buffer_rank: Int,
    type: DType,
](
    dummy_chain: Int,
    buffer: NDBuffer[DType.uint8, 1],
    spec: StaticTensorSpec[spec_rank],
) -> NDBuffer[type, buffer_rank]:
    debug_assert(type == spec.type)

    @parameter
    if spec_rank == 0:
        # We promote scalar tensor to tensor<[1]>
        constrained[buffer_rank == 1]()
        return NDBuffer[type, buffer_rank](
            buffer.data.bitcast[type](),
            rebind[IndexList[buffer_rank]](IndexList[1](1)),
        )
    else:
        constrained[spec_rank == buffer_rank]()
        return NDBuffer[type, buffer_rank](
            buffer.data.bitcast[type](),
            rebind[IndexList[buffer_rank]](spec.shape),
        )


@mogg_register("mgp.tensor.extract.tensor_spec")
@always_inline
fn mgp_tensor_extract_tensor_spec[
    tensor_rank: Int,
    buffer_rank: Int,
    type: DType,
](buffer: NDBuffer[type, buffer_rank]) -> StaticTensorSpec[
    tensor_rank
] as result:
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


# ===----------------------------------------------------------------------===#
# MGP Buffer Primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mgp.buffer.alloc")
@always_inline
fn mgp_buffer_alloc[
    aRuntimeSlot: UInt64,
    bRawAlign: UInt64,
    cDevice: StringLiteral,
](
    dummy_chain: Int,
    ctx: StateContext,
    byte_size: Int,
    dev_context: UnsafePointer[DeviceContext],
    call_ctx: MojoCallContextPtr,
) raises -> NDBuffer[DType.int8, 1]:
    # Default to alignment of 0 which means kPreferredMemoryAlignment if cRawAlign is kUnknownSize (SizeUtils.h).
    alias alignment = 0 if bRawAlign == UInt64.MAX else int(bRawAlign)
    return byte_buffer_alloc[cDevice, alignment=alignment](
        byte_size, dev_context, call_ctx
    )


@mogg_register("mgp.buffer.constant.external")
@export
fn mgp_buffer_constant_external[
    aRuntimeSlot: UInt64,
    bName: StringLiteral,
    cSize: UInt64,
    dAlign: UInt64,
    eDevice: StringLiteral,
](
    dummy_chain: Int,
    ctx: StateContext,
    weights: UnsafePointer[WeightsRegistry],
    call_ctx: MojoCallContextPtr,
) raises -> NDBuffer[DType.int8, 1]:
    constrained[dAlign > 0, "dAlign must be a positive integer value"]()

    if not weights:
        raise Error(
            "received null weights registry in mgp.buffer.constant.external"
        )

    var weight_ptr = weights[][bName]
    if (int(weight_ptr) % dAlign) != 0:
        raise Error(
            "invalid alignment for address "
            + str(weight_ptr)
            + " and align "
            + str(dAlign)
        )

    return NDBuffer[DType.int8, 1](weight_ptr.bitcast[Int8](), DimList(cSize))


@always_inline
fn fill_buffer[
    type: DType
](buf: NDBuffer[DType.uint8, 1], vals: VariadicList[Int]):
    var ptr = buf.data.bitcast[type]()
    var offset: Int = 0
    for val in vals:
        ptr.store(offset, val)
        offset += 1


@mogg_register("mgp.buffer.set_with_index")
@always_inline
@export
fn mgp_buffer_set_with_index[
    aRuntimeSlot: UInt64, bDevice: StringLiteral
](
    ctx: StateContext, buffer: NDBuffer[DType.uint8, 1], *vals: Int
) raises -> Int:
    debug_assert(
        bDevice == "cpu", "set_with_index can only work on cpu buffers"
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
    return 1  # Dummy int for output chain on DeviceOp.td


@mogg_register("mgp.buffer.to_bool")
@always_inline
fn mgp_buffer_to_bool[
    aRuntimeSlot: UInt64, bDevice: StringLiteral
](
    dummy_chain: Int, ctx: StateContext, buffer: NDBuffer[DType.uint8, 1]
) -> Bool:
    debug_assert(bDevice == "cpu", "to_bool can only work on cpu buffers")
    var bufSize = buffer.num_elements()
    debug_assert(
        bufSize == 1,
        "buffer size must be a size of 1",
    )
    return buffer[0] != 0


@mogg_register("mgp.buffer.to_index")
@always_inline
@export
fn mgp_buffer_to_index(
    dummy_chain: Int, buffer: NDBuffer[DType.uint8, 1]
) raises -> Int:
    var bufSize = buffer.num_elements()
    if bufSize == 4:
        return int(buffer.data.bitcast[Int32]()[0])
    if bufSize == 8:
        return int(buffer.data.bitcast[Int64]()[0])

    raise Error(
        "mgp.buffer.to_index must be called on either a 4- or 8-byte buffer"
    )


@mogg_register("mgp.buffer.slice")
@always_inline
fn mgp_buffer_slice(
    buffer: NDBuffer[DType.uint8, 1], offset: Int, size: Int
) -> NDBuffer[DType.uint8, 1]:
    return NDBuffer[DType.uint8, 1](buffer.data.offset(offset), Index(size))


@mogg_register("mgp.buffer.concat")
@always_inline
fn mgp_buffer_concat[
    aRuntimeSlot: UInt64, bDevice: StringLiteral
](
    dummy_chain: Int,
    ctx: StateContext,
    output: NDBuffer[DType.uint8, 1],
    inputs: StaticTuple[NDBuffer[DType.uint8, 1], *_],
    call_ctx: MojoCallContextPtr,
) raises -> Int:
    if len(output) < 4096:
        concat[1, DType.uint8, True, bDevice, None](
            output, 0, inputs, context=call_ctx
        )
    else:
        concat[1, DType.uint8, False, bDevice, None](
            output, 0, inputs, context=call_ctx
        )

    return 0


@mogg_register("mgp.buffer.device_to_host")
@always_inline
fn mgp_buffer_device_to_host[
    aOtherRuntimeSlot: UInt64,
    bHostRuntimeSlot: UInt64,
    cOtherDevice: StringLiteral,
    dHostDevice: StringLiteral,
](
    in_chain: Int,
    ctx: StateContext,
    dev_buf: NDBuffer[DType.uint8, 1],
    host_buf: NDBuffer[DType.uint8, 1],
    dev_ctx: UnsafePointer[DeviceContext],
    call_ctx: MojoCallContextPtr,
) raises -> Int:
    @parameter
    if (dHostDevice == "cpu") and ("cuda" in cOtherDevice):
        dev_ctx[].enqueue_copy_from_device[DType.uint8](
            host_buf.data,
            DeviceBuffer[DType.uint8](
                dev_ctx[],
                dev_buf.data,
                dev_buf.size(),
                owning=False,
            ),
        )
    else:
        raise Error(
            "mgp.buffer.device_to_host must be scheduled on cuda device"
        )
    return 0


@mogg_register("mgp.buffer.device_to_device")
@always_inline
fn mgp_buffer_device_to_device[
    aSrcRuntimeSlot: UInt64,
    bDstRuntimeSlot: UInt64,
    cSrcDevice: StringLiteral,
    dDstDevice: StringLiteral,
](
    in_chain: Int,
    ctx: StateContext,
    src_buf: NDBuffer[DType.uint8, 1],
    dst_buf: NDBuffer[DType.uint8, 1],
    src_dev_ctx: UnsafePointer[DeviceContext],
    dst_dev_ctx: UnsafePointer[DeviceContext],
    call_ctx: MojoCallContextPtr,
) raises -> Int:
    @parameter
    if ("cuda" in cSrcDevice) and ("cuda" in dDstDevice):
        dst_dev_ctx[].enqueue_copy_device_to_device[DType.uint8](
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
    elif cSrcDevice == dDstDevice == "cpu":
        memcpy(dst_buf.data, src_buf.data, src_buf.size())
    else:
        raise Error(
            "mgp.buffer.device_to_device can be scheduled between same device"
            " types (cpu-cpu) or (cuda-cuda)"
        )
    return 0


@mogg_register("mgp.buffer.host_to_device")
@always_inline
fn mgp_buffer_host_to_device[
    aHostRuntimeSlot: UInt64,
    bOtherRuntimeSlot: UInt64,
    cHostDevice: StringLiteral,
    dOtherDevice: StringLiteral,
](
    in_chain: Int,
    ctx: StateContext,
    host_buf: NDBuffer[DType.uint8, 1],
    dev_buf: NDBuffer[DType.uint8, 1],
    dev_ctx: UnsafePointer[DeviceContext],
    call_ctx: MojoCallContextPtr,
) raises -> Int:
    @parameter
    if ("cuda" in dOtherDevice) and (cHostDevice == "cpu"):
        dev_ctx[].enqueue_copy_to_device[DType.uint8](
            DeviceBuffer[DType.uint8](
                dev_ctx[],
                dev_buf.data,
                dev_buf.size(),
                owning=False,
            ),
            host_buf.data,
        )
    else:
        raise Error(
            "mgp.buffer.host_to_device must be scheduled on cuda device"
        )
    return 0


@mogg_register("mgp.buffer.get_cached")
@always_inline
@export
fn mgp_buffer_get_cached[
    aRuntimeSlot: UInt64,
    bBufferSlot: UInt64,
](
    dummy_chain: Int,
    ctx: StateContext,
    storage_ref_addr: UnsafePointer[UnsafePointer[NoneType]],
) raises -> NDBuffer[DType.uint8, 1]:
    var buffer_size: UInt64 = 0
    var buffer_data: UnsafePointer[NoneType] = external_call[
        "KGEN_CompilerRT_GetCachedBuffer", UnsafePointer[NoneType]
    ](
        int(bBufferSlot),
        ctx.ctx_ptr,
        UnsafePointer.address_of(buffer_size),
        storage_ref_addr,
    )

    if not buffer_data:
        raise Error("failed in mgp.buffer.get_cached")

    return NDBuffer[DType.uint8, 1](
        buffer_data.bitcast[UInt8](), Index(buffer_size)
    )


@mogg_register("destruct_async_refs")
@always_inline
fn destruct_async_refs[
    size: Int
](storage_ref_addr: StaticTuple[UnsafePointer[UnsafePointer[NoneType]], size]):
    external_call["KGEN_CompilerRT_DestructAsyncRefs", NoneType](
        size, UnsafePointer.address_of(storage_ref_addr.array).address
    )


# ===----------------------------------------------------------------------===#
# MGP Tensor Spec Primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mgp.tensor_spec.create")
@always_inline
@export
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


@mogg_register("mgp.tensor_spec.size")
@always_inline
@export
fn mgp_tensor_spec_size[
    spec_rank: Int
](spec: StaticTensorSpec[spec_rank]) -> Int:
    return spec.bytecount()


@mogg_register("mgp.tensor_spec.equal.static")
@always_inline
@export
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


# ===----------------------------------------------------------------------===#
# MGP Device Context Primitives
# ===----------------------------------------------------------------------===#


@export
fn mgp_device_context_destroy(dev_ctx: UnsafePointer[DeviceContext]):
    _ = dev_ctx.destroy_pointee()


@mogg_register("mgp.device.context.profile.start")
@always_inline
@export
fn mgp_device_context_profile_start[
    aDeviceRuntimeSlot: UInt64,
    bDevice: StringLiteral,
    cTag: StringLiteral,
    dFilePath: StringLiteral,
](
    in_chain: Int,
    ctx: StateContext,
    dev_ctx: UnsafePointer[DeviceContext],
    call_ctx: MojoCallContextPtr,
) -> Int:
    # Call into device_context here.
    return 1


@mogg_register("mgp.device.context.profile.end")
@always_inline
@export
fn mgp_device_context_profile_end[
    aDeviceRuntimeSlot: UInt64,
    bDevice: StringLiteral,
    cTag: StringLiteral,
    dFilePath: StringLiteral,
](
    in_chain: Int,
    ctx: StateContext,
    dev_ctx: UnsafePointer[DeviceContext],
    call_ctx: MojoCallContextPtr,
) raises -> Int:
    # Call into device_context here....
    return 1


@mogg_register("mgp.sync")
@always_inline
fn mgp_sync[
    aRuntimeSlot: UInt64,
    bDevice: StringLiteral,
](
    in_chain: Int,
    ctx: StateContext,
    dev_ctx: UnsafePointer[DeviceContext],
    call_ctx: MojoCallContextPtr,
) raises -> Int:
    dev_ctx[].synchronize()
    return 0


@mogg_register("mgp.debug.print")
@always_inline
fn mg_debug_print[
    aDebugString: StringLiteral,
    bLabel: StringLiteral,
](in_chain: Int, ctx: StateContext,) raises -> Int:
    prefix = ""
    if bLabel:
        prefix = "[" + bLabel + "] "
    print(prefix + aDebugString)
    return 0


# ===----------------------------------------------------------------------===#
# Opaque Test Primitives
# ===----------------------------------------------------------------------===#


struct MyInt(Movable):
    var val: Int

    fn __init__(inout self, val: Int):
        self.val = val

    fn __moveinit__(inout self, owned other: MyInt):
        print("MyInt.__moveinit__", other.val)
        self.val = other.val

    fn __del__(owned self):
        print("MyInt.__del__", self.val)


@mogg_register("testfuse.my_int.from_index")
@always_inline
@export
fn test_my_int_from_index(x: Int) -> MyInt:
    return MyInt(x)


@mogg_register("testfuse.my_int.square")
@always_inline
@export
fn test_my_int_square(x: MyInt) -> MyInt:
    return MyInt(x.val * x.val)


@mogg_register("testfuse.my_int.to_index")
@always_inline
@export
fn test_my_int_to_index(x: MyInt) -> Int:
    return x.val


@value
@register_passable("trivial")
struct MyIntReg:
    var val: Int

    fn __init__(inout self, val: Int):
        self.val = val


@mogg_register("testfuse.my_int_reg.square")
@always_inline
@export
fn test_my_int_reg_square(x: MyIntReg) -> MyIntReg:
    return MyIntReg(x.val * x.val)


@value
@register_passable
struct MyIntReg2:
    var val: Int

    fn __init__(inout self, val: Int):
        self.val = val

    fn __del__(owned self):
        print("MyIntReg2.__del__", self.val)


@mogg_register("testfuse.my_int_reg2.from_index")
@always_inline
@export
fn test_my_int_reg2_from_index(x: Int) -> MyIntReg2:
    return MyIntReg2(x)


@mogg_register("testfuse.my_int_reg2.square")
@always_inline
@export
fn test_my_int_reg2_square(x: MyIntReg2) -> MyIntReg2:
    return MyIntReg2(x.val * x.val)


@mogg_register("testfuse.my_int_reg2.to_index")
@always_inline
@export
fn test_my_int_reg2_to_index(x: MyIntReg2) -> Int:
    return x.val
