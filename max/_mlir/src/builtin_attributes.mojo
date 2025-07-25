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

from collections.string import StaticString

import ._c
from .ir import Attribute, Context, DialectAttribute, Type


@fieldwise_init
struct BoolAttr(Copyable, DialectAttribute, Movable):
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


struct TypeAttr(Copyable, DialectAttribute, Movable):
    var type: Type

    fn __init__(out self, type: Type):
        self.type = type

    fn to_mlir(self) -> Attribute:
        return _c.BuiltinAttributes.mlirTypeAttrGet(self.type.c)

    @staticmethod
    fn from_mlir(attr: Attribute) raises -> Self:
        if not _c.BuiltinAttributes.mlirAttributeIsAType(attr.c):
            raise "Attribute is not a Type"
        return Self(_c.BuiltinAttributes.mlirTypeAttrGetValue(attr.c))


@fieldwise_init
struct StringAttr(Copyable, DialectAttribute, Movable):
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
