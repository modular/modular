# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Platform predicates for the `cond` argument of `@extensibility.register`.

A platform predicate selects the devices that a kernel registration applies
to. Build one from the `type`, `api`, `arch`, and `model` atoms below, and
combine predicates with `&` and `|`. The `CPU`, `ACCELERATOR`, `CUDA`, `HIP`,
and `METAL` aliases in `extensibility.platforms` cover the common cases.
Name a predicate for a specific device with `comptime`, and pass the name to
`register`:

```mojo
comptime B200 = ACCELERATOR & CUDA & model("NVIDIA B200")

@extensibility.register("mo.matmul", B200)
struct B200Matmul:
    ...
```

`&` and `|` flatten and order their operands, so `CUDA | HIP` and
`HIP | CUDA` are the same type. Predicates that select the same devices need
not be spelled the same way, or have the same type, as `CUDA` and
`ACCELERATOR & CUDA` show: the graph compiler compares predicates by the
devices they select, not by their type.

A platform predicate is not a `Bool`: only `register` accepts it.

When more than one registration of an op matches a device, the graph compiler
selects the registration whose predicate selects the fewest devices: its set
of devices must be a strict subset of the set of every other match. The graph
compiler compares predicates only through string equality on the four device
fields. Thus a predicate must state every field that it depends on. For
example, `CUDA` is `type("gpu") & api("cuda")`, so a CUDA kernel is more
specific than a kernel registered for all accelerators.
"""

from std.builtin.enum_like import KGENString


comptime PlatformExpr = __generator_type[
    label: KGENString, api: KGENString, arch: KGENString, model: KGENString
] __mlir_type.`!kgen.scalar<bool>`
"""The expression of a platform predicate: what it tests on a device.

A predicate may compare the label, API, architecture, or model of a device
with a string literal, and combine those tests with `&` and `|`. The graph
compiler reads the expression as data when it imports a kernel, so it rejects
anything else, and it rejects a predicate that holds for no device.
"""

# Each alias below has the four device fields as its last parameters. The
# atoms and the operators bind the other parameters and leave the fields
# unbound (`_`). The result is a `PlatformExpr`.

comptime _Type[
    v: KGENString,
    label: KGENString,
    api: KGENString,
    arch: KGENString,
    model: KGENString,
] = __mlir_attr[
    `#kgen.param.identical<`, label, `, `, v, `> : !kgen.scalar<bool>`
]

comptime _Api[
    v: KGENString,
    label: KGENString,
    api: KGENString,
    arch: KGENString,
    model: KGENString,
] = __mlir_attr[
    `#kgen.param.identical<`, api, `, `, v, `> : !kgen.scalar<bool>`
]

comptime _Arch[
    v: KGENString,
    label: KGENString,
    api: KGENString,
    arch: KGENString,
    model: KGENString,
] = __mlir_attr[
    `#kgen.param.identical<`, arch, `, `, v, `> : !kgen.scalar<bool>`
]

comptime _Model[
    v: KGENString,
    label: KGENString,
    api: KGENString,
    arch: KGENString,
    model: KGENString,
] = __mlir_attr[
    `#kgen.param.identical<`, model, `, `, v, `> : !kgen.scalar<bool>`
]

comptime _And[
    A: PlatformExpr,
    B: PlatformExpr,
    label: KGENString,
    api: KGENString,
    arch: KGENString,
    model: KGENString,
] = __mlir_attr[
    `#kgen.param.expr<and, `,
    A[label, api, arch, model],
    `, `,
    B[label, api, arch, model],
    `> : !kgen.scalar<bool>`,
]

comptime _Or[
    A: PlatformExpr,
    B: PlatformExpr,
    label: KGENString,
    api: KGENString,
    arch: KGENString,
    model: KGENString,
] = __mlir_attr[
    `#kgen.param.expr<or, `,
    A[label, api, arch, model],
    `, `,
    B[label, api, arch, model],
    `> : !kgen.scalar<bool>`,
]


@fieldwise_init
struct PlatformPredicate[expr: PlatformExpr](TrivialRegisterPassable):
    """A set of devices that a kernel registration applies to.

    The type holds the expression of the predicate, so predicates with the same
    meaning have the same type.

    Parameters:
        expr: The expression of the platform predicate.
    """

    @always_inline
    def __and__(
        self, other: PlatformPredicate
    ) -> PlatformPredicate[_And[Self.expr, type_of(other).expr, _, _, _, _]]:
        """Returns a platform predicate that holds when both predicates hold.

        Args:
            other: The other platform predicate.

        Returns:
            The conjunction of the two platform predicates.
        """
        return {}

    @always_inline
    def __or__(
        self, other: PlatformPredicate
    ) -> PlatformPredicate[_Or[Self.expr, type_of(other).expr, _, _, _, _]]:
        """Returns a platform predicate that holds when either predicate holds.

        Args:
            other: The other platform predicate.

        Returns:
            The disjunction of the two platform predicates.
        """
        return {}


@always_inline
def type(
    value: StringLiteral,
) -> PlatformPredicate[_Type[type_of(value).value, _, _, _, _]]:
    """Returns a platform predicate that holds when the device label equals
    `value`.

    Args:
        value: The device label, such as `"cpu"` or `"gpu"`. It is the label of
            the `DeviceRef`, for example `DeviceRef.GPU()`.

    Returns:
        The platform predicate.
    """
    return {}


@always_inline
def api(
    value: StringLiteral,
) -> PlatformPredicate[_Api[type_of(value).value, _, _, _, _]]:
    """Returns a platform predicate that holds when the device API equals
    `value`.

    Args:
        value: The device API, such as `"cuda"` or `"hip"`.

    Returns:
        The platform predicate.
    """
    return {}


@always_inline
def arch(
    value: StringLiteral,
) -> PlatformPredicate[_Arch[type_of(value).value, _, _, _, _]]:
    """Returns a platform predicate that holds when the device architecture
    equals `value`.

    Args:
        value: The device architecture, such as `"sm_100a"` or `"gfx942"`.

    Returns:
        The platform predicate.
    """
    return {}


@always_inline
def model(
    value: StringLiteral,
) -> PlatformPredicate[_Model[type_of(value).value, _, _, _, _]]:
    """Returns a platform predicate that holds when the device model equals
    `value`.

    Args:
        value: The device model, such as `"NVIDIA B200"`.

    Returns:
        The platform predicate.
    """
    return {}
