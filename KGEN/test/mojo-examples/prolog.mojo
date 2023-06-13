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


from SIMD import SIMD
from SIMD import Float32
from Assert import assert_param, assert_param
from Range import range
from IO import print
from DType import DType
from Object import object

# CHECK: lit.func @"printInt
fn printInt(x: Int):
    print(x.value)


# ===----------------------------------------------------------------------=== #
# WIP Types
# ===----------------------------------------------------------------------=== #


@register_passable
struct Scalar[type: DType]:
    fn __copyinit__(self) -> Self:
        return Self {}


# ===----------------------------------------------------------------------=== #
# Error
# ===----------------------------------------------------------------------=== #


@register_passable
struct StringRef:
    var data: __mlir_type.`!pop.pointer<!pop.scalar<si8>>`
    var size: Int

    fn __copyinit__(self) -> Self:
        return Self {data: self.data, size: self.size}

    fn __init__(
        data: __mlir_type.`!pop.pointer<!pop.scalar<si8>>`, size: Int
    ) -> StringRef:
        return StringRef {data: data, size: size}

    @staticmethod
    fn empty() -> StringRef:
        let nullptr = __mlir_op.`pop.cast_from_builtin`[
            _type : __mlir_type.`!pop.scalar<index>`
        ]((0).value)
        let size = 0
        return StringRef(
            __mlir_op.`pop.index_to_pointer`[
                _type : __mlir_type.`!pop.pointer<!pop.scalar<si8>>`
            ](nullptr),
            size,
        )
