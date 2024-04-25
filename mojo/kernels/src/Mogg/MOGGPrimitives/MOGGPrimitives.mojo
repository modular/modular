# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import (
    div_ceil,
    max,
    min,
)
from register import *
from buffer import NDBuffer
from buffer.list import DimList
from buffer.buffer import (
    _MAX_RANK,
)
from tensor import TensorSpec, TensorShape
from MOGGIntList import IntList


# ===----------------------------------------------------------------------===#
# Helper Structures
# ===----------------------------------------------------------------------===#


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
    spec: TensorSpec,
    async_ptr: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
    runtime: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
):
    # Mojo impl is bitwise compatible with cpp variant, can construct TensorSpec in mojo
    # and pass it back to C++
    external_call["KGEN_CompilerRT_CreateAsyncTensorSpec", NoneType](
        Pointer.address_of(spec), async_ptr, runtime
    )


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


@mogg_register("builtin.unpack_buffer_data")
@always_inline
@export
fn unpack_buffer_data(
    async_ptr: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
) -> __mlir_type.`!kgen.pointer<scalar<invalid>>`:
    return external_call[
        "KGEN_CompilerRT_GetDataFromBuffer",
        __mlir_type.`!kgen.pointer<scalar<invalid>>`,
    ](async_ptr)


@mogg_register("builtin.unpack_tensor_spec")
@always_inline
@export
fn unpack_tensor_spec(
    async_ptr: __mlir_type.`!kgen.pointer<scalar<invalid>>`,
) -> TensorSpec:
    var raw_spec_ptr = external_call[
        "KGEN_CompilerRT_GetValueFromAsync",
        __mlir_type.`!kgen.pointer<scalar<invalid>>`,
    ](async_ptr)
    var spec = Pointer(raw_spec_ptr).bitcast[TensorSpec]()
    return spec[]


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
    return div_ceil(numerator, denominator)


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


@mogg_register("mgp.tensor_spec.create")
@always_inline
@export
fn mgp_tensor_spec_create[
    bRawDType: UInt8,
    aRawDims: DimList,
    aRawDimsRank: Int,
](*runtimeDims: Int) -> TensorSpec:
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
    return TensorSpec(dType, shape)


@mogg_register("mgp.tensor_spec.size")
@always_inline
@export
fn mgp_tensor_spec_size(spec: TensorSpec) -> Int:
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
