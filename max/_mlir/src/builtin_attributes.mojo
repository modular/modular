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
