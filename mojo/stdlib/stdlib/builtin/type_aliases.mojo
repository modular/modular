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
"""Defines some type aliases.

These are Mojo built-ins, so you don't need to import them.
"""

comptime AnyTrivialRegType = __mlir_type.`!kgen.type`
"""Represents any register passable Mojo data type."""


@deprecated(use=ImmutOrigin)
comptime ImmutableOrigin = ImmutOrigin
"""Immutable origin reference type."""

comptime ImmutOrigin = Origin[False]
"""Immutable origin reference type."""


@deprecated(use=MutOrigin)
comptime MutableOrigin = MutOrigin
"""Mutable origin reference type."""

comptime MutOrigin = Origin[True]
"""Mutable origin reference type."""


@deprecated(use=ImmutAnyOrigin)
comptime ImmutableAnyOrigin = ImmutAnyOrigin
"""The immutable origin that might access any memory value."""

comptime ImmutAnyOrigin = __mlir_attr.`#lit.any.origin : !lit.origin<0>`
"""The immutable origin that might access any memory value."""


@deprecated(use=MutAnyOrigin)
comptime MutableAnyOrigin = MutAnyOrigin
"""The mutable origin that might access any memory value."""

comptime MutAnyOrigin = __mlir_attr.`#lit.any.origin<1>: !lit.origin<1>`
"""The mutable origin that might access any memory value."""

# Static constants are a named subset of the global origin.
comptime StaticConstantOrigin = __mlir_attr[
    `#lit.origin.field<`,
    `#lit.static.origin : !lit.origin<0>`,
    `, "__constants__"> : !lit.origin<0>`,
]
"""An origin for strings and other always-immutable static constants."""

comptime OriginSet = __mlir_type.`!lit.origin.set`
"""A set of origin parameters."""

comptime Never = __mlir_type.`!kgen.never`
"""A type that can never have an instance constructed, used as a function result
by functions that never return."""


@register_passable("trivial")
struct Origin[mut: Bool]:
    """This represents a origin reference for a memory value.

    Parameters:
        mut: Whether the origin is mutable.
    """

    comptime _mlir_type = __mlir_type[
        `!lit.origin<`,
        Self.mut._mlir_value,
        `>`,
    ]

    comptime cast_from[o: Origin] = __mlir_attr[
        `#lit.origin.mutcast<`,
        o._mlir_origin,
        `> : !lit.origin<`,
        Self.mut._mlir_value,
        `>`,
    ]
    """Cast an existing Origin to be of the specified mutability.

    This is a low-level way to coerce Origin mutability. This should be used
    rarely, typically when building low-level fundamental abstractions. Strongly
    consider alternatives before reaching for this "escape hatch".

    Parameters:
        o: The origin to cast.

    Safety:
        This is an UNSAFE operation if used to cast an immutable origin to
        a mutable origin.

    Examples:

    Cast a mutable origin to be immutable:

    ```mojo
    struct Container[mut: Bool, //, origin: Origin[mut]]:
        var data: Int

        fn imm_borrow(self) -> Container[ImmutOrigin.cast_from[origin]]:
            pass
    ```
    """

    comptime external = Self.cast_from[origin_of()]
    """An external origin of the given mutability. The external origin is
    guaranteed not to alias any existing origins.

    An external origin implies there is no previously existing value that this
    origin aliases. Therefore, the compiler cannot track the origin or the
    value's lifecycle. The external origin is useful when interfacing with
    memory that comes from outside the current Mojo program.
    """

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _mlir_origin: Self._mlir_type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_private
    @implicit
    @always_inline("builtin")
    fn __init__(out self, mlir_origin: Self._mlir_type):
        """Initialize an Origin from a raw MLIR `!lit.origin` value.

        Args:
            mlir_origin: The raw MLIR origin value.
        """
        self._mlir_origin = mlir_origin
