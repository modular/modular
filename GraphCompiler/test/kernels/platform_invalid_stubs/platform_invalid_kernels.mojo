# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
#
# Kernel stubs with forged platforms. Mojo accepts each of them, and the
# graph compiler must reject each of them when it imports the package.

from std.collections.string import StaticString
from max.gpu.host import DeviceContext

import extensibility
from extensibility import (
    PlatformExpr,
    PlatformPredicate,
    Input,
    Output,
    TileTensorable,
)

comptime Field = type_of("".value)

# Compares two device fields with each other instead of with a string literal.
comptime _LabelIsApi[label: Field, api: Field, arch: Field, model: Field] = (
    __mlir_attr[
        `#kgen.param.identical<`, label, `, `, api, `> : !kgen.scalar<bool>`
    ]
)
comptime LABEL_IS_API = PlatformPredicate[_LabelIsApi[_, _, _, _]]()

# Compares two device fields through `StringLiteral`, with no MLIR syntax.
comptime _LabelEqualsApi[
    label: Field, api: Field, arch: Field, model: Field
] = (StringLiteral[label]() == StringLiteral[api]())._mlir_value
comptime LABEL_EQUALS_API = PlatformPredicate[_LabelEqualsApi[_, _, _, _]]()

# Tests a prefix, which no atom can express.
comptime _ArchIsSm[label: Field, api: Field, arch: Field, model: Field] = (
    StaticString(StringLiteral[arch]()).startswith("sm_")
)._mlir_value
comptime ARCH_IS_SM = PlatformPredicate[_ArchIsSm[_, _, _, _]]()

# Takes integers instead of the device fields, and `rebind` hides the type.
comptime _IntFields[label: Int, api: Int, arch: Int, model: Int] = (
    label == api
)._mlir_value
comptime INT_FIELDS = PlatformPredicate[
    rebind[PlatformExpr](_IntFields[_, _, _, _])
]()


@extensibility.register("pred_label_is_api", LABEL_IS_API)
struct PredLabelIsApi:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_label_equals_api", LABEL_EQUALS_API)
struct PredLabelEqualsApi:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_arch_is_sm", ARCH_IS_SM)
struct PredArchIsSm:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@extensibility.register("pred_int_fields", INT_FIELDS)
struct PredIntFields:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & TileTensorable,
        X: Input & TileTensorable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass
