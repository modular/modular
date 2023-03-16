# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s | FileCheck %s
# ===----------------------------------------------------------------------=== #
#
# Standard Prolog(ue) for Lit.
#
# This file implements basic datatypes used commonly across lit programs.
#
# ===----------------------------------------------------------------------=== #


from Int import Int
from SIMD import SIMD
from F32 import F32
from Assert import assert_param
from Range import range
from IO import print

# CHECK: lit.func @"printInt
fn printInt(x: Int):
    print(x.value)


# ===----------------------------------------------------------------------=== #
# WIP Types
# ===----------------------------------------------------------------------=== #

# NOTE: This would be more naturally modeled as an enum, but this works for now.
@register_passable
struct DType:
    var value: __mlir_type.`!kgen.dtype`

    fn __new__(value: __mlir_type.`!kgen.dtype`) -> DType:
        return DType {value: value}

    alias f32 = DType(__mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`)
    alias f64 = DType(__mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`)
    alias si32 = DType(__mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`)
    alias si64 = DType(__mlir_attr.`#kgen.dtype.constant<si64> : !kgen.dtype`)


@register_passable
struct Scalar[type: DType]:
    pass


# ===----------------------------------------------------------------------=== #
# Error
# ===----------------------------------------------------------------------=== #


@register_passable
struct StringRef:
    var data: __mlir_type.`!pop.pointer<!pop.scalar<si8>>`
    var size: Int

    fn __clone__(self&) -> Self:
        return Self {data: self.data, size: self.size}

    fn __new__(
        data: __mlir_type.`!pop.pointer<!pop.scalar<si8>>`, size: Int
    ) -> StringRef:
        return StringRef {data: data, size: size}

    @staticmethod
    fn empty() -> StringRef:
        let nullptr = __mlir_op.`pop.cast_from_builtin`[
            _type : __mlir_type.`!pop.scalar<index>`
        ](__mlir_op.`index.constant`[value:0]())
        let size = Int(0)
        return StringRef(
            __mlir_op.`pop.index_to_pointer`[
                _type : __mlir_type.`!pop.pointer<!pop.scalar<si8>>`
            ](nullptr),
            size,
        )


@register_passable
struct Error:
    var msg: StringRef

    fn __clone__(self&) -> Self:
        return Self {msg: self.msg}

    fn __new__(msg: StringRef) -> Error:
        return Error {msg: msg}


struct ErrorOr[type: __mlir_type.`!kgen.mlirtype`]:
    var value: __mlir_type[`!pop.variant<`, Error, `, `, type, `>>`]

    fn __new__(
        value: __mlir_type[`!pop.variant<`, Error, `, `, type, `>>`]
    ) -> ErrorOr[type]:
        return ErrorOr[type] {value: value}

    fn __new__(err: Error) -> ErrorOr[type]:
        return ErrorOr[type] {
            value: __mlir_op.`pop.variant.create`[
                _type : __mlir_type[`!pop.variant<`, Error, `, `, type, `>`]
            ](err)
        }

    fn __new__(value: type) -> ErrorOr[type]:
        return ErrorOr[type] {
            value: __mlir_op.`pop.variant.create`[
                _type : __mlir_type[`!pop.variant<`, Error, `, `, type, `>`]
            ](value)
        }

    fn __bool__(self: ErrorOr[type]) -> Bool:
        return (
            Bool.false()
            if __mlir_op.`pop.variant.is`[testType : __mlir_attr[Error]](
                self.value
            )
            else Bool.true()
        )

    fn getValue(self: ErrorOr[type]) -> type:
        return __mlir_op.`pop.variant.get`[_type:type](self.value)

    fn getError(self: ErrorOr[type]) -> Error:
        return __mlir_op.`pop.variant.get`[_type:Error](self.value)


# ===----------------------------------------------------------------------=== #
# object
# ===----------------------------------------------------------------------=== #

# TODO: This should eventually model a dynamic object base class.  For now, this
# is just a placeholder to be used by untyped 'def' operands.
@register_passable
struct object:
    pass
