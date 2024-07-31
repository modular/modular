# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.vector import InlinedFixedVector
from math import ceildiv

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from extensibility import Tensor as ExtensibilityTensor
from gpu.host import Context as CudaContext
from gpu.host import (
    CudaInstance,
    Device,
    DeviceBuffer,
    DeviceContext,
    KernelProfilingInfo,
)
from gpu.host.memory import _free, _malloc
from memory import UnsafePointer
from memory.memory import _malloc as _malloc_cpu
from MOGGIntList import IntList
from register import *
from runtime.asyncrt import MojoCallContextPtr
from utils import StaticIntTuple
from sys import alignof

# ===----------------------------------------------------------------------===#
# Helper Structures
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct StaticTensorSpec[rank: Int]():
    """Defines a static-rank tensor spec which - has a static-rank shape
    in a StaticIntTuple and dtype. This is analagous to TensorSpec from
    tensor_spec, but this is fully static and will not have any allocations."""

    var shape: StaticIntTuple[rank]
    var dType: DType

    @always_inline
    fn __init__(inout self, shape: StaticIntTuple[rank], dType: DType):
        """Constructs a static tensor spec with a static rank shape and dType.

        Args:
            shape: The shape of static rank we are creating tensor spec with.
            dType: The DType we are creating tensor spec with.
        """
        self.shape = shape
        self.dType = dType

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
        return len(self) * self.dType.sizeof()

    @always_inline
    fn __eq__(self, rhs: StaticTensorSpec[rank]) -> Bool:
        """Compares this StaticTensorSpec to another StaticTensorSpec for equality.

        The StaticTensorSpec are equal if the shapes are equal and the dTypes are
        also equal.

        Args:
            rhs: The other StaticTensorSpec.

        Returns:
            The comparison result.
        """
        return self.shape == rhs.shape and self.dType == rhs.dType


@register_passable("trivial")
struct StateContext:
    """Defines a StateContext structure which holds a ptr to context and has accessors that go to external calls
    This is currently meant as a mojo-side container for GML::StateContext."""

    var numSlots: Int
    var ctxPtr: UnsafePointer[NoneType]

    @always_inline
    fn __init__(inout self, numSlots: Int, ctxPtr: UnsafePointer[NoneType]):
        self.numSlots = numSlots
        self.ctxPtr = ctxPtr

    @always_inline
    fn __getitem__(self, index: Int) -> UnsafePointer[NoneType]:
        debug_assert(0 <= index < self.numSlots, "index must be within bounds")
        return external_call[
            "KGEN_CompilerRT_GetContextPayloadPtr",
            UnsafePointer[NoneType],
        ](index, self.ctxPtr)


# ===----------------------------------------------------------------------===#
# Helper functions
# ===----------------------------------------------------------------------===#


@always_inline
fn byte_buffer_alloc[
    target: StringLiteral,
    alignment: Int,
](byte_size: Int, callCtx: MojoCallContextPtr) raises -> NDBuffer[
    DType.int8, 1
]:
    """Function will allocate a 1-D buffer with the specified size/alignment on device.
    """
    # This primitive has a byte-size input, so always assume a byte format
    var shape = StaticIntTuple[1](byte_size)

    @parameter
    if target == "cuda":
        # For now, only cuda targets can use device context directly
        return NDBuffer[DType.int8, 1](
            callCtx.get_cuda_device().cuda_context.malloc_async[Int8](
                byte_size, callCtx.get_cuda_device().cuda_stream
            ),
            shape,
        )
    else:
        return NDBuffer[DType.int8, 1](
            UnsafePointer[Int8].alloc[alignment=alignment](byte_size),
            shape,
        )


# ===----------------------------------------------------------------------===#
# Async Packing/Unpacking functions
# ===----------------------------------------------------------------------===#


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


@mogg_register("builtin.create_buffer_ref_async")
@always_inline
@export
fn create_buffer_ref_async[
    target: StringLiteral
](
    buffer: NDBuffer[DType.int8, 1],
    async_ptr: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
    callCtx: MojoCallContextPtr,
):
    @parameter
    if target == "cuda":
        external_call["KGEN_CompilerRT_CreateAsyncCUDABufferRef", NoneType](
            buffer.data, len(buffer), async_ptr, runtime, callCtx.ptr
        )
    else:
        external_call["KGEN_CompilerRT_CreateAsyncBufferRef", NoneType](
            buffer.data, len(buffer), async_ptr, runtime
        )


@mogg_register("builtin.create_buffer_ref_with_borrow_async")
@always_inline
@export
fn create_buffer_ref_with_borrow_async[
    target: StringLiteral
](
    buffer: NDBuffer[DType.int8, 1],
    async_to_borrow: UnsafePointer[NoneType],
    output_async: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
    callCtx: MojoCallContextPtr,
):
    external_call["KGEN_CompilerRT_CreateAsyncBufferWithBorrow", NoneType](
        buffer.data, len(buffer), async_to_borrow, output_async, runtime
    )


@mogg_register("builtin.create_tensor_spec_async")
@always_inline
@export
fn create_tensor_spec_async[
    rank: Int
](
    spec: StaticTensorSpec[rank],
    async_ptr: UnsafePointer[NoneType],
    runtime: UnsafePointer[NoneType],
):
    # Mojo impl is bitwise compatible with cpp variant, can construct TensorSpec in mojo
    # and pass it back to C++ -- However, this is an issue for the heap allocated dims.
    # For the benefit of simplicity, allocate the shapes and ptrs and free explicitly after
    var shape_ptr = UnsafePointer[Int].alloc(rank)
    for i in range(rank):
        shape_ptr[i] = spec.shape[i]
    external_call["KGEN_CompilerRT_CreateAsyncTensorSpec", NoneType](
        shape_ptr, rank, spec.dType, async_ptr, runtime
    )
    shape_ptr.free()


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


@mogg_register("builtin.unpack_buffer")
@always_inline
@export
fn unpack_buffer[
    target: StringLiteral
](async_ptr: UnsafePointer[NoneType],) -> NDBuffer[DType.uint8, 1]:
    var size: UInt64 = 0
    var data_ptr = external_call[
        "KGEN_CompilerRT_GetDataFromBuffer",
        UnsafePointer[NoneType],
    ](async_ptr, UnsafePointer.address_of(size))
    var shape = StaticIntTuple[1](int(size))
    return NDBuffer[DType.uint8, 1](data_ptr.bitcast[UInt8](), shape)


@mogg_register("builtin.unpack_tensor_spec")
@always_inline
@export
fn unpack_tensor_spec[
    rank: Int
](async_ptr: UnsafePointer[NoneType]) -> StaticTensorSpec[rank]:
    var shape_ptr = UnsafePointer[Int].alloc(rank)
    var raw_dtype = external_call[
        "KGEN_CompilerRT_GetTensorSpecFromAsync",
        UInt8,
    ](shape_ptr, rank, async_ptr)
    var shape = StaticIntTuple[rank]()
    for i in range(rank):
        shape[i] = int(shape_ptr[i])
    shape_ptr.free()
    return StaticTensorSpec[rank](shape, DType._from_ui8(raw_dtype.value))


@mogg_register("builtin.unpack_context")
@always_inline
@export
fn unpack_context(
    async_ptr: UnsafePointer[NoneType],
) -> StateContext:
    # We want to construct this because we want all payloads to be implemented
    var numSlots: UInt64 = 0
    var ctxPtr: UnsafePointer[NoneType] = external_call[
        "KGEN_CompilerRT_GetContextAndSizeFromAsync",
        UnsafePointer[NoneType],
    ](UnsafePointer.address_of(numSlots), async_ptr)
    return StateContext(int(numSlots), ctxPtr)


@mogg_register("builtin.get_buffer_data")
@always_inline
@export
fn get_buffer_data(buffer: NDBuffer[DType.uint8, 1]) -> UnsafePointer[UInt8]:
    return buffer.data


# ===----------------------------------------------------------------------===#
# MIP Index Primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mip.constant.index")
@always_inline
@export
fn mip_constant_index[value: Int64]() -> Int:
    return value.value


@mogg_register("mip.print.index")
@always_inline
@export
fn mip_print_index(x: Int, dummy_chain: Int) -> Int:
    print("index = ", x)
    return x


@mogg_register("mip.add")
@always_inline
@export
fn mip_add(x: Int, y: Int) -> Int:
    return x + y


@mogg_register("mip.mul")
@always_inline
@export
fn mip_mul(lhs: Int, rhs: Int) -> Int:
    return lhs * rhs


@mogg_register("mip.div")
@always_inline
@export
fn mip_div(numerator: Int, denominator: Int) -> Int:
    debug_assert(denominator != 0, "mip.div divide by zero")
    return numerator // denominator


@mogg_register("mip.div.ceil")
@always_inline
@export
fn mip_div_ceil(numerator: Int, denominator: Int) -> Int:
    debug_assert(denominator != 0, "mip.div.ceil divide by zero")
    return ceildiv(numerator, denominator)


@mogg_register("mip.cmp.eq")
@always_inline
@export
fn mip_cmp_eq(x: Int, y: Int) -> Bool:
    return x == y


@mogg_register("mip.cmp.lt")
@always_inline
@export
fn mip_cmp_lt(x: Int, y: Int) -> Bool:
    return x < y


@mogg_register("mip.cmp.le")
@always_inline
@export
fn mip_cmp_le(x: Int, y: Int) -> Bool:
    return x <= y


@mogg_register("mip.max")
@always_inline
@export
fn mip_max(x: Int, y: Int) -> Int:
    return max(x, y)


@mogg_register("mip.min")
@always_inline
@export
fn mip_min(x: Int, y: Int) -> Int:
    return min(x, y)


@mogg_register("mip.mod")
@always_inline
@export
fn mip_mod(numerator: Int, denominator: Int) -> Int:
    debug_assert(denominator != 0, "mip.mod divide by zero")
    return numerator % denominator


# ===----------------------------------------------------------------------===#
# MIP Bool Primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mip.constant.bool")
@always_inline
@export
fn mip_constant_bool[value: Bool]() -> Bool:
    return value


@mogg_register("mip.bool.and")
@always_inline
@export
fn mip_and(x: Bool, y: Bool) -> Bool:
    return x & y


@mogg_register("mip.bool.or")
@always_inline
@export
fn mip_or(x: Bool, y: Bool) -> Bool:
    return x | y


@mogg_register("mip.bool.xor")
@always_inline
@export
fn mip_xor(x: Bool, y: Bool) -> Bool:
    return x ^ y


@mogg_register("mip.select")
@always_inline
@export
fn mip_select[T: AnyTrivialRegType](cond: Bool, true: T, false: T) -> T:
    return true if cond else false


@mogg_register("mip.nary.mul")
@always_inline
@export
fn mip_nary_mul[constInt: Int64](*vals: Int) -> Int:
    var product = Int(constInt.value)
    for val in vals:
        product *= val
    return product


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
# MGP Buffer Primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mgp.buffer.alloc.static")
@always_inline
@export
fn mgp_buffer_alloc_static[
    aRuntimeSlot: UInt64,
    bSize: UInt64,
    cRawAlign: UInt64,
    dDevice: StringLiteral,
](
    dummy_chain: Int, stateCtx: StateContext, callCtx: MojoCallContextPtr
) raises -> NDBuffer[DType.int8, 1]:
    # Default to alignment of 1 if cRawAlign is kUnknownSize (SizeUtils.h).
    alias alignment = alignof[DType.int8]() if cRawAlign == UInt64.MAX else int(
        cRawAlign
    )
    return byte_buffer_alloc[dDevice, alignment=alignment](int(bSize), callCtx)


@mogg_register("mgp.buffer.alloc.dynamic")
@always_inline
@export
fn mgp_buffer_alloc_dynamic[
    aRuntimeSlot: UInt64,
    bRawAlign: UInt64,
    cDevice: StringLiteral,
](
    dummy_chain: Int,
    ctx: StateContext,
    byte_size: Int,
    callCtx: MojoCallContextPtr,
) raises -> NDBuffer[DType.int8, 1]:
    # Default to alignment of 1 if cRawAlign is kUnknownSize (SizeUtils.h).
    alias alignment = alignof[DType.int8]() if bRawAlign == UInt64.MAX else int(
        bRawAlign
    )
    return byte_buffer_alloc[cDevice, alignment=alignment](byte_size, callCtx)


@always_inline
@export
fn get_mgp_buffer[
    rank: Int, type: DType
](tensor: ExtensibilityTensor[type, rank]) -> NDBuffer[DType.uint8, 1]:
    var bufferRef = NDBuffer[DType.uint8, 1](
        tensor.data.bitcast[DType.uint8](),
        tensor.nelems() * sizeof[type](),
    )
    return bufferRef


@always_inline
fn fillBuffer[
    type: DType
](buf: NDBuffer[DType.uint8, 1], vals: VariadicList[Int]):
    var ptr = buf.data.bitcast[type]()
    var offset: Int = 0
    for val in vals:
        ptr.store[width=1](offset, val)
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
        fillBuffer[DType.int32](buffer, vals)
    elif elSize == 8:
        fillBuffer[DType.int64](buffer, vals)
    else:
        raise Error("unsupported element size")
    return 1  # Dummy int for output chain on DeviceOp.td


@mogg_register("mgp.buffer.to_bool")
@always_inline
@export
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


@mogg_register("mgp.buffer.slice")
@always_inline
@export
fn mgp_buffer_slice(
    buffer: NDBuffer[DType.uint8, 1], offset: Int, size: Int
) -> NDBuffer[DType.uint8, 1]:
    return NDBuffer[DType.uint8, 1](buffer.data.offset(offset), size)


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
    var dType = DType._from_ui8(bRawDType.value)
    var static_shape = IntList[aRawDims]()
    var shape = StaticIntTuple[aRawDimsRank]()
    var runtimeIndex = 0
    # Update Shape with runtime elements.
    for i in range(aRawDimsRank):
        if static_shape[i] > -1:
            shape[i] = static_shape[i]
        else:
            shape[i] = runtimeDims[runtimeIndex]
            runtimeIndex = runtimeIndex + 1
    return StaticTensorSpec[aRawDimsRank](shape, dType)


@mogg_register("mgp.tensor_spec.size")
@always_inline
@export
fn mgp_tensor_spec_size[rank: Int](spec: StaticTensorSpec[rank]) -> Int:
    return spec.bytecount()


@mogg_register("mgp.tensor_spec.equal.static")
@always_inline
@export
fn mgp_tensor_spec_equal_static[
    rank: Int, *rawDims: Dim
](spec: StaticTensorSpec[rank]) -> Bool:
    var dims: VariadicList[Dim] = rawDims
    var numDims = len(dims)
    if rank != numDims:
        return False
    for i in range(numDims):
        var dim = dims[i]
        var expectedDim = spec.shape[i]
        if dim and dim != -1 and dim != expectedDim:
            return False

    return True


# ===----------------------------------------------------------------------===#
# MGP Chain Primitives
# ===----------------------------------------------------------------------===#

# TODO: Enable MGP chain primitives once either fusion is enabled or perf issues
# are resolved
# See https://github.com/modularml/modular/issues/38548
# Issue tracking reenablement: https://github.com/modularml/modular/issues/38551


# @mogg_register("mgp.chain.create")
@always_inline
@export
fn mgp_chain_create[
    aOtherRuntimeSlot: UInt64,
    bHostDevice: StringLiteral,
    cOtherDevice: StringLiteral,
]():
    return


# @mogg_register("mgp.chain.device_to_host")
@always_inline
@export
fn mgp_chain_device_to_host[aRuntimeSlot: UInt64, bDevice: StringLiteral]():
    return


# @mogg_register("mgp.chain.host_to_device")
@always_inline
@export
fn mgp_chain_host_to_device[aRuntimeSlot: UInt64, bDevice: StringLiteral]():
    return


# ===----------------------------------------------------------------------===#
# MGP Device Context Primitives
# ===----------------------------------------------------------------------===#

alias dev_ty = Tuple[CudaContext, CudaInstance, KernelProfilingInfo]


@mogg_register("mgp.device.context.create")
@always_inline
@export
fn mgp_device_context_create[
    aDeviceRuntimeSlot: UInt64, bDevice: StringLiteral
](dummy_chain: Int, ctx: StateContext) -> Int:
    @parameter
    if bDevice == "cuda":
        var dev_ctx = external_call[
            "KGEN_CompilerRT_MojoValueAllocateBuffer", UnsafePointer[dev_ty]
        ](sizeof[dev_ty](), alignof[dev_ty]())
        try:
            var instance = CudaInstance()
            var context = CudaContext(Device(instance))
            var kernel_profiling_info = KernelProfilingInfo()
            dev_ctx.init_pointee_move(
                (context, instance, kernel_profiling_info)
            )
        except e:
            abort(e)
        external_call[
            "KGEN_CompilerRT_CudaContextSetDevice",
            NoneType._mlir_type,
        ](
            dev_ctx,
            mgp_device_context_destroy,
            ctx[Int(aDeviceRuntimeSlot.cast[DType.int64]().value)],
        )
    return 1


@export
fn mgp_device_context_destroy(
    dev_ctx: UnsafePointer[
        Tuple[StateContext, CudaInstance, KernelProfilingInfo]
    ]
):
    _ = UnsafePointer.address_of(dev_ctx[][0]).destroy_pointee()
    _ = UnsafePointer.address_of(dev_ctx[][1]).destroy_pointee()
    # TODO: The below fails because dev_ctx[][2] is already freed.
    # dev_ctx.destroy_pointee()


@mogg_register("mgp.device.context.profile.start")
@always_inline
@export
fn mgp_device_context_profile_start[
    aDeviceRuntimeSlot: UInt64,
    bDevice: StringLiteral,
    cTag: StringLiteral,
    dFilePath: StringLiteral,
](ctx: StateContext, callCtx: MojoCallContextPtr) -> Int:
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
](ctx: StateContext, callCtx: MojoCallContextPtr) -> Int:
    # Call into device_context here....
    try:
        callCtx.get_cuda_device().dump_kernel_timing_info()
    except e:
        abort(e)
    return 1


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
@register_passable
struct MyIntReg:
    var val: Int

    fn __init__(inout self, val: Int):
        self.val = val

    fn __del__(owned self):
        print("MyIntReg.__del__", self.val)


@mogg_register("testfuse.my_int_reg.from_index")
@always_inline
@export
fn test_my_int_reg_from_index(x: Int) -> MyIntReg:
    return MyIntReg(x)


@mogg_register("testfuse.my_int_reg.square")
@always_inline
@export
fn test_my_int_reg_square(x: MyIntReg) -> MyIntReg:
    return MyIntReg(x.val * x.val)


@mogg_register("testfuse.my_int_reg.to_index")
@always_inline
@export
fn test_my_int_reg_to_index(x: MyIntReg) -> Int:
    return x.val
