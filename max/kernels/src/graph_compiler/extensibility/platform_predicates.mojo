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
comptime B200 = ACCELERATOR & CUDA & arch("sm_100a") & model("NVIDIA B200")

@extensibility.register("mo.matmul", B200)
struct B200Matmul:
    ...
```

The four device fields form levels, from the broadest to the narrowest: the
type, the API, the architecture, and the model. A test of one field narrows a
condition at the level above it, so `api()` narrows a type-level condition
such as `ACCELERATOR`, `arch()` an API-level one such as `CUDA`, and `model()`
an arch-level one. `api()`, `arch()`, and `model()` alone are not conditions:
`register` accepts a predicate only when every test in it has its parent, and
a test can't skip a level or attach to the wrong one. Thus the graph compiler
always knows that a narrower predicate is inside a broader one.

`&` and `|` take their operands in either order, so `arch("sm_90a") & CUDA`
is the same as `CUDA & arch("sm_90a")`. A chain of field tests can wait for
its parent, as in `ACCELERATOR & (api("hip") | (api("cuda") & arch("sm_90a")))`.
A union of conditions that end at different levels, such as `CUDA | HOPPER`,
can be registered, but it can't take more field tests: add the test to the
branch that it narrows.

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
fields.
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


@always_inline("builtin")
def _min(a: Int, b: Int) -> Int:
    return a if a < b else b


@always_inline("builtin")
def _max(a: Int, b: Int) -> Int:
    return a if a > b else b


struct _AndRules[
    top: Int,
    low: Int,
    high: Int,
    other_top: Int,
    other_low: Int,
    other_high: Int,
]:
    """The rules that `&` checks on its two operands.

    `&` is commutative, so the operand with the broader `top` is the parent
    and the other operand is the child. Each rule holds unless one mistake
    occurs, and at most one rule fails for a call.
    """

    comptime _swap = Self.other_top < Self.top
    comptime _parent_top = Self.other_top if Self._swap else Self.top
    comptime _parent_low = Self.other_low if Self._swap else Self.low
    comptime _parent_high = Self.other_high if Self._swap else Self.high
    comptime _child_top = Self.top if Self._swap else Self.other_top
    comptime _extends = Self._child_top != Self._parent_top
    comptime _even = Self._parent_low == Self._parent_high
    comptime _fits[level: Int] = not (
        Self._parent_top == 0
        and Self._child_top == level
        and Self._even
        and Self._parent_low != level - 1
    )

    comptime no_test_under_a_mixed_union = not (
        Self._extends and not Self._even
    )
    comptime api_under_a_type = Self._fits[1]
    comptime arch_under_an_api = Self._fits[2]
    comptime model_under_an_arch = Self._fits[3]
    comptime tests_chain_one_level_at_a_time = not (
        Self._parent_top != 0
        and Self._extends
        and Self._even
        and Self._child_top != Self._parent_low + 1
    )


@fieldwise_init
struct _Platform[expr: PlatformExpr, top: Int, low: Int, high: Int](
    TrivialRegisterPassable
):
    """A platform as the atoms and the operators build it: a set of devices,
    or a chain of field tests that still needs its parent.

    The levels count the device fields from the broadest to the narrowest: 0
    is the type, 1 the API, 2 the architecture, and 3 the model. A platform
    with `top == 0` is a full condition, which converts to a
    `PlatformPredicate`. Any other platform is a chain of field tests that
    needs a parent one level above `top`.

    `low` and `high` record where the branches of the platform end. A branch
    is one way for the expression to hold, read as one path through its `|`
    operators, and it ends at the narrowest level that it tests. `CUDA` has
    one branch, which ends at level 1. `CUDA | HOPPER` has two, which end at
    levels 1 and 2, so its `low` is 1 and its `high` is 2. The platform
    keeps only the two extremes because the rules ask one question of its
    ends: do all branches end at the same level? `low == high` answers it,
    and then `low` is the level that a new field test attaches under.

    Each form of expression computes the ends of its result from the ends of
    its operands. The examples show `(top, low, high)`:

    - An atom has one branch, which ends at its own level. `api("cuda")` is
      `(1, 1, 1)`.
    - `A | B` keeps the branches of both operands, so its ends are all the
      ends of both: `low` is the smaller `low`, and `high` is the larger
      `high`. `CUDA | HIP` is `(0, 1, 1)`, because two branches that end at
      the same level keep the union even. `CUDA | HOPPER` is `(0, 1, 2)`.
    - `A & B` with the same `top` is an intersection. Each branch of the
      result pairs a branch of `A` with a branch of `B` and tests the levels
      of both, so it ends at the narrower of their two ends. Thus `low` is
      the larger `low`, and `high` is the larger `high`.
      `ACCELERATOR & CUDA` is `(0, 1, 1)`. `(CUDA | HOPPER) & HOPPER` is
      `(0, 2, 2)`: an intersection with an even platform can make a mixed
      union even again.
    - `A & B` with different `top` values is an extension. The child
      attaches under the single end of the parent, so each branch of the
      result ends where a branch of the child ends, and the result takes the
      `low` and `high` of the child. `CUDA & arch("sm_90a")` is `(0, 2, 2)`,
      and `CUDA & (arch("sm_90a") | (arch("sm_100a") & model("X")))` is
      `(0, 2, 3)`.

    Only an extension checks the ends. The parent must be even, and the
    child must start at the parent's `low + 1`. A mixed parent, such as
    `CUDA | HOPPER`, has no single level for a child to start under, so it
    can be registered but not extended. Put the test in the branch that it
    narrows instead, as in `CUDA | (HOPPER & model("X"))`. An intersection
    checks nothing, because each operand already holds its own levels
    correctly, and a union checks only that both operands start at the same
    `top`.

    A chain of field tests without its parent follows the same rules, one
    level lower. In `api("hip") | (api("cuda") & arch("sm_90a"))`, both
    branches start at level 1, so `|` accepts them, and the result is the
    mixed chain `(1, 1, 2)`. `ACCELERATOR` is `(0, 0, 0)`, which is even and
    ends at level 0, so it can take this chain as a child, and
    `ACCELERATOR & (api("hip") | (api("cuda") & arch("sm_90a")))` is
    `(0, 1, 2)`. Because the chain is mixed, nothing can extend it, before
    or after it gets its parent.

    `low` and `high` don't decide whether a platform converts to a
    `PlatformPredicate`: only `top == 0` does. They decide only what can
    extend the platform.

    Parameters:
        expr: The expression of the platform.
        top: The broadest level that the platform tests.
        low: The broadest level at which a branch of the platform ends.
        high: The narrowest level at which a branch of the platform ends.
            `low` and `high` differ for a union such as `CUDA | HOPPER`.
    """

    @always_inline
    def __and__[
        E: PlatformExpr, T: Int, L: Int, H: Int
    ](self, other: _Platform[E, T, L, H]) -> _Platform[
        _And[Self.expr, E, _, _, _, _],
        _min(Self.top, T),
        # An intersection keeps the narrower end; an extension takes the ends
        # of the child.
        _max(Self.low, L) if Self.top
        == T else (L if Self.top < T else Self.low),
        _max(Self.high, H) if Self.top
        == T else (H if Self.top < T else Self.high),
    ] where (
        _AndRules[
            Self.top, Self.low, Self.high, T, L, H
        ].no_test_under_a_mixed_union
    ) else (
        "add the field test to each branch of this union"
    ) where (
        _AndRules[Self.top, Self.low, Self.high, T, L, H].api_under_a_type
    ) else (
        'api() narrows a type-level condition: ACCELERATOR & api("cuda")'
    ) where (
        _AndRules[Self.top, Self.low, Self.high, T, L, H].arch_under_an_api
    ) else (
        'arch() narrows an API-level condition: CUDA & arch("sm_90a")'
    ) where (
        _AndRules[Self.top, Self.low, Self.high, T, L, H].model_under_an_arch
    ) else (
        'model() narrows an arch-level condition: SM100 & model("NVIDIA B200")'
    ) where (
        _AndRules[
            Self.top, Self.low, Self.high, T, L, H
        ].tests_chain_one_level_at_a_time
    ) else (
        'field tests chain one level at a time: api("cuda") & arch("sm_90a")'
    ):
        """Returns a platform that holds when both platforms hold.

        Parameters:
            E: The expression of the other platform.
            T: The broadest level that the other platform tests.
            L: The broadest level at which a branch of the other platform
                ends.
            H: The narrowest level at which a branch of the other platform
                ends.

        Args:
            other: The other platform.

        Returns:
            The conjunction of the two platforms.
        """
        return {}

    @always_inline
    def __or__[
        E: PlatformExpr, T: Int, L: Int, H: Int
    ](self, other: _Platform[E, T, L, H]) -> _Platform[
        _Or[Self.expr, E, _, _, _, _],
        Self.top,
        _min(Self.low, L),
        _max(Self.high, H),
    ] where (Self.top == T) else (
        "| joins two conditions, or two field tests of the same level"
    ):
        """Returns a platform that holds when either platform holds.

        Parameters:
            E: The expression of the other platform.
            T: The broadest level that the other platform tests.
            L: The broadest level at which a branch of the other platform
                ends.
            H: The narrowest level at which a branch of the other platform
                ends.

        Args:
            other: The other platform.

        Returns:
            The disjunction of the two platforms.
        """
        return {}


@fieldwise_init
struct PlatformPredicate[expr: PlatformExpr](TrivialRegisterPassable):
    """A set of devices that a kernel registration applies to.

    The type holds the expression of the predicate. A full condition, such as
    `CUDA` or `CUDA & arch("sm_90a")`, converts to a `PlatformPredicate`. A
    field test without its parent, such as `arch("sm_90a")`, does not.

    Parameters:
        expr: The expression of the platform predicate.
    """

    @implicit
    def __init__[
        low: Int, high: Int
    ](out self, platform: _Platform[Self.expr, 0, low, high]):
        """Converts a full condition to a platform predicate.

        Parameters:
            low: The broadest level at which a branch of `platform` ends.
            high: The narrowest level at which a branch of `platform` ends.

        Args:
            platform: The full condition.
        """
        pass


@always_inline
def type(
    value: StringLiteral,
) -> _Platform[_Type[type_of(value).value, _, _, _, _], 0, 0, 0]:
    """Returns a platform that holds when the device label equals
    `value`.

    Args:
        value: The device label, such as `"cpu"` or `"gpu"`. It is the label of
            the `DeviceRef`, for example `DeviceRef.GPU()`.

    Returns:
        The platform.
    """
    return {}


@always_inline
def api(
    value: StringLiteral,
) -> _Platform[_Api[type_of(value).value, _, _, _, _], 1, 1, 1]:
    """Returns a field test that holds when the device API equals `value`.

    The test narrows a type-level condition, as in `ACCELERATOR & api("cuda")`.

    Args:
        value: The device API, such as `"cuda"` or `"hip"`.

    Returns:
        The field test.
    """
    return {}


@always_inline
def arch(
    value: StringLiteral,
) -> _Platform[_Arch[type_of(value).value, _, _, _, _], 2, 2, 2]:
    """Returns a field test that holds when the device architecture equals
    `value`.

    The test narrows an API-level condition, as in `CUDA & arch("sm_90a")`.

    Args:
        value: The device architecture, such as `"sm_100a"` or `"gfx942"`.

    Returns:
        The field test.
    """
    return {}


@always_inline
def model(
    value: StringLiteral,
) -> _Platform[_Model[type_of(value).value, _, _, _, _], 3, 3, 3]:
    """Returns a field test that holds when the device model equals `value`.

    The test narrows an arch-level condition, as in
    `CUDA & arch("sm_100a") & model("NVIDIA B200")`.

    Args:
        value: The device model, such as `"NVIDIA B200"`.

    Returns:
        The field test.
    """
    return {}
