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

from ._c.ffi import MLIR_func
from .ir import Context, DialectType, Type


@fieldwise_init
struct FunctionType(Copyable, DialectType, Movable):
    var ctx: Context
    var inputs: List[Type]
    var results: List[Type]

    @implicit
    fn __init__(out self, ctx: Context):
        self = Self(ctx, List[Type](), List[Type]())

    fn __init__(out self, inputs: List[Type], results: List[Type]):
        debug_assert(
            len(inputs).__bool__() or len(results).__bool__(),
            "nullary functions must use the context constructor",
        )
        var ctx = (inputs if len(inputs) else results)[0].context()
        self = Self(ctx, inputs, results)

    fn to_mlir(self) -> Type:
        return _c.BuiltinTypes.mlirFunctionTypeGet(
            self.ctx.c,
            len(self.inputs),
            self.inputs.data.bitcast[Type.cType](),
            len(self.results),
            self.results.data.bitcast[Type.cType](),
        )

    @staticmethod
    fn from_mlir(type: Type) raises -> Self:
        if not _c.BuiltinTypes.mlirTypeIsAFunction(type.c):
            raise "Type is not a Function"
        var inputs = List[Type]()
        var results = List[Type]()
        for i in range(_c.BuiltinTypes.mlirFunctionTypeGetNumInputs(type.c)):
            inputs.append(_c.BuiltinTypes.mlirFunctionTypeGetInput(type.c, i))
        for i in range(_c.BuiltinTypes.mlirFunctionTypeGetNumResults(type.c)):
            results.append(_c.BuiltinTypes.mlirFunctionTypeGetResult(type.c, i))
        return Self(type.context(), inputs, results)
