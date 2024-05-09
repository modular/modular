# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import ceildiv

from register import *
from buffer import NDBuffer
from buffer.list import DimList
from tensor import TensorSpec
from MOGGIntList import IntList
from extensibility import Tensor as ExtensibilityTensor

from builtin.dtype import _get_runtime_dtype_size


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
        return len(self) * _get_runtime_dtype_size(self.dType)

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


# ===----------------------------------------------------------------------===#
# Async Packing/Unpacking functions
# ===----------------------------------------------------------------------===#


@mogg_register("builtin.create_index_async")
@always_inline
@export
fn create_index_async(
    value: Int,
    async_ptr: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
    runtime: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
):
    external_call["KGEN_CompilerRT_CreateAsync_ssizet", NoneType](
        value, async_ptr, runtime
    )


@mogg_register("builtin.create_i1_async")
@always_inline
@export
fn create_i1_async(
    value: Bool,
    async_ptr: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
    runtime: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
):
    external_call["KGEN_CompilerRT_CreateAsync_bool", NoneType](
        value, async_ptr, runtime
    )


@mogg_register("builtin.create_buffer_ref_async")
@always_inline
@export
fn create_buffer_ref_async(
    buffer: NDBuffer[DType.int8, 1],
    async_ptr: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
    runtime: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
):
    external_call["KGEN_CompilerRT_CreateAsyncBufferRef", NoneType](
        buffer.data, len(buffer), async_ptr, runtime
    )


@mogg_register("builtin.create_tensor_spec_async")
@always_inline
@export
fn create_tensor_spec_async[
    rank: Int
](
    spec: StaticTensorSpec[rank],
    async_ptr: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
    runtime: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
):
    # Mojo impl is bitwise compatible with cpp variant, can construct TensorSpec in mojo
    # and pass it back to C++ -- However, this is an issue for the heap allocated dims.
    # For the benefit of simplicity, allocate the shapes and ptrs and free explicitly after
    var shape_ptr = DTypePointer[DType.index].alloc(rank)
    for i in range(rank):
        shape_ptr[i] = spec.shape[i]
    external_call["KGEN_CompilerRT_CreateAsyncTensorSpec", NoneType](
        shape_ptr, rank, spec.dType, async_ptr, runtime
    )
    shape_ptr.free()


@mogg_register("builtin.unpack_async")
@always_inline
@export
fn unpack_async(
    async_ptr: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
) -> __mlir_type.`!kgen.pointer<none>`:
    return external_call[
        "KGEN_CompilerRT_GetValueFromAsync",
        __mlir_type.`!kgen.pointer<none>`,
    ](async_ptr)


@mogg_register("builtin.unpack_buffer")
@always_inline
@export
fn unpack_buffer(
    async_ptr: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
) -> NDBuffer[DType.uint8, 1]:
    var size: UInt64 = 0
    var data_ptr = external_call[
        "KGEN_CompilerRT_GetDataFromBuffer",
        __mlir_type.`!kgen.pointer<scalar<invalid>>`,
    ](async_ptr, Pointer.address_of(size))

    var shape = StaticIntTuple[1](int(size))
    return NDBuffer[DType.uint8, 1](Pointer(data_ptr).bitcast[UInt8](), shape)


@mogg_register("builtin.unpack_tensor_spec")
@always_inline
@export
fn unpack_tensor_spec[
    rank: Int
](
    async_ptr: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
) -> StaticTensorSpec[
    rank
]:
    var shape_ptr = DTypePointer[DType.index].alloc(rank)
    var raw_dtype = external_call[
        "KGEN_CompilerRT_GetTensorSpecFromAsync",
        UInt8,
    ](shape_ptr, rank, async_ptr)
    var shape = StaticIntTuple[rank]()
    for i in range(rank):
        shape[i] = int(shape_ptr[i])
    shape_ptr.free()
    return StaticTensorSpec[rank](shape, DType._from_ui8(raw_dtype.value))


@mogg_register("builtin.get_buffer_data")
@always_inline
@export
fn get_buffer_data(
    buffer: NDBuffer[DType.uint8, 1]
) -> DTypePointer[DType.uint8]:
    return buffer.data


# ===----------------------------------------------------------------------===#
# MIP Index Primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mip.constant.index")
@always_inline
@export
fn mip_constant_index[value: Int64]() -> Int:
    return value.value


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
fn mip_select[T: AnyRegType](cond: Bool, true: T, false: T) -> T:
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
# MGP Primitives
# ===----------------------------------------------------------------------===#


@mogg_register("mgp.assert")
@always_inline
@export
fn mgp_assert[message: StringLiteral](cond: Bool) raises:
    if not cond:
        raise Error(message)


@mogg_register("mgp.buffer.alloc.static")
@always_inline
@export
fn mgp_buffer_alloc_static[
    aRuntimeSlot: UInt64,
    bSize: UInt64,
    cRawAlign: UInt64,
    dDevice: StringLiteral,
]() -> NDBuffer[DType.int8, 1]:
    var shape = StaticIntTuple[1](int(bSize))
    if cRawAlign == UInt64.MAX:
        return NDBuffer[DType.int8, 1](
            DTypePointer[DType.int8].alloc(int(bSize)), shape
        )
    else:
        return NDBuffer[DType.int8, 1](
            DTypePointer[DType.int8].alloc(
                int(bSize), alignment=int(cRawAlign)
            ),
            shape,
        )


@mogg_register("mgp.buffer.alloc.dynamic")
@always_inline
@export
fn mgp_buffer_alloc_dynamic[
    aRuntimeSlot: UInt64,
    bRawAlign: UInt64,
    cDevice: StringLiteral,
](byte_size: Int) -> NDBuffer[DType.int8, 1]:
    var shape = StaticIntTuple[1](byte_size)
    if bRawAlign == UInt64.MAX:
        return NDBuffer[DType.int8, 1](
            DTypePointer[DType.int8].alloc(byte_size), shape
        )
    else:
        return NDBuffer[DType.int8, 1](
            DTypePointer[DType.int8].alloc(byte_size, alignment=int(bRawAlign)),
            shape,
        )


@always_inline
@export
fn get_mgp_buffer[
    rank: Int, type: DType
](tensor: ExtensibilityTensor[type, rank]) -> NDBuffer[DType.uint8, 1]:
    var bufferRef = NDBuffer[DType.uint8, 1](
        tensor.data.bitcast[DType.uint8](),
        tensor.nelems() * type.sizeof(),
    )
    return bufferRef


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
fn mgp_tensor_spec_size[
    rank: Int
](borrowed spec: StaticTensorSpec[rank]) -> Int:
    return spec.bytecount()


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


@always_inline
fn fillBuffer[
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
fn mgp_buffer_set_with_index(buffer: NDBuffer[DType.uint8, 1], *vals: Int):
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
        debug_assert(False, "unsupported element size")
