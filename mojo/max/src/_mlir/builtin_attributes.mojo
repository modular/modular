# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import List
from collections.string import StaticString


import ._c
import ._c.BuiltinAttributes
import ._c.BuiltinTypes
from ._c.ffi import MLIR_func
from .ir import Attribute, Context, DialectAttribute, Type


@value
struct BoolAttr(DialectAttribute):
    var ctx: Context
    var value: Bool

    fn to_mlir(self) -> Attribute:
        return _c.BuiltinAttributes.mlirBoolAttrGet(self.ctx.c, Int(self.value))

    @staticmethod
    fn from_mlir(attr: Attribute) raises -> Self:
        if not _c.BuiltinAttributes.mlirAttributeIsABool(attr.c):
            raise "Attribute is not a Bool"
        return Self(
            attr.context(),
            _c.BuiltinAttributes.mlirBoolAttrGetValue(attr.c),
        )


@value
struct TypeAttr(DialectAttribute):
    var type: Type

    @implicit
    fn __init__(out self, type: Type):
        self.type = type

    fn to_mlir(self) -> Attribute:
        return _c.BuiltinAttributes.mlirTypeAttrGet(self.type.c)

    @staticmethod
    fn from_mlir(attr: Attribute) raises -> Self:
        if not _c.BuiltinAttributes.mlirAttributeIsAType(attr.c):
            raise "Attribute is not a Type"
        return Self(_c.BuiltinAttributes.mlirTypeAttrGetValue(attr.c))


@value
struct StringAttr(DialectAttribute):
    var ctx: Context
    var value: String

    fn to_mlir(self) -> Attribute:
        var result = _c.BuiltinAttributes.mlirStringAttrGet(
            self.ctx.c,
            StaticString(
                ptr=self.value.unsafe_ptr(), length=self.value.byte_length()
            ),
        )
        return result

    @staticmethod
    fn from_mlir(attr: Attribute) raises -> Self:
        if not _c.BuiltinAttributes.mlirAttributeIsAString(attr.c):
            raise "Attribute is not a String"
        return Self(
            attr.context(),
            String(_c.BuiltinAttributes.mlirStringAttrGetValue(attr.c)),
        )
