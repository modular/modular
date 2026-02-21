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
"""Defines some type aliases.

These are Mojo built-ins, so you don't need to import them.
"""

comptime __TypeOfAllTypes = __mlir_type.`!kgen.type`
"""Represents any register passable Mojo data type."""

comptime ImmutOrigin = Origin[mut=False]
"""Immutable origin reference type."""

comptime MutOrigin = Origin[mut=True]
"""Mutable origin reference type."""

comptime AnyOrigin[*, mut: Bool] = Origin[
    _mlir_origin = __mlir_attr[
        `#lit.any.origin : !lit.origin<`, +mut._mlir_value, `>`
    ],
]()
"""An origin that might access any memory value.

Parameters:
    mut: Whether the origin is mutable.
"""

comptime ImmutAnyOrigin = AnyOrigin[mut=False]
"""The immutable origin that might access any memory value."""

comptime MutAnyOrigin = AnyOrigin[mut=True]
"""The mutable origin that might access any memory value."""

comptime ExternalOrigin[*, mut: Bool] = Origin[
    _mlir_origin = __mlir_attr[
        `#lit.origin.union<> : !lit.origin<`,
        +mut._mlir_value,
        `>`,
    ],
]()
"""A parameterized external origin guaranteed not to alias any existing origins.

Parameters:
    mut: Whether the origin is mutable.

An external origin implies there is no previously existing value that this
origin aliases. The compiler cannot track the origin or the value's lifecycle.
Useful when interfacing with memory from outside the current Mojo program.
"""

comptime ImmutExternalOrigin = ExternalOrigin[mut=False]
"""An immutable external origin guaranteed not to alias any existing origins.

An external origin implies there is no previously existing value that this
origin aliases. The compiler cannot track the origin or the value's lifecycle.
Useful when interfacing with memory from outside the current Mojo program.
"""

comptime MutExternalOrigin = ExternalOrigin[mut=True]
"""A mutable external origin guaranteed not to alias any existing origins.

An external origin implies there is no previously existing value that this
origin aliases. The compiler cannot track the origin or the value's lifecycle.
Useful when interfacing with memory from outside the current Mojo program.
"""

# Static constants are a named subset of the global origin.
comptime StaticConstantOrigin = Origin[
    _mlir_origin = __mlir_attr[
        `#lit.origin.field<`,
        `#lit.static.origin : !lit.origin<0>`,
        `, "__constants__"> : !lit.origin<0>`,
    ]
]()
"""An origin for strings and other always-immutable static constants."""

comptime OriginSet = __mlir_type.`!lit.origin.set`
"""A set of origin parameters."""

comptime Never = __mlir_type.`!kgen.never`
"""A type that can never have an instance constructed, used as a function result
by functions that never return."""

comptime EllipsisType = __mlir_type.`!lit.ellipsis`
"""The type of the `...` literal."""


comptime _lit_origin_type_of_mut[mut: Bool] = __mlir_type[
    `!lit.origin<`, mut._mlir_value, `>`
]


struct Origin[mut: Bool, _mlir_origin: _lit_origin_type_of_mut[mut], //](
    TrivialRegisterPassable
):
    """This represents a origin reference for a memory value.

    Parameters:
        mut: Whether the origin is mutable.
        _mlir_origin: The raw MLIR origin value.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __init__(out self):
        """Construct an Origin."""
        pass

    @always_inline("builtin")
    @implicit
    fn __init__(v: Origin) -> ImmutOrigin[_mlir_origin = v._mlir_origin]:
        """Implicitly convert an origin to an immutable one.

        Args:
            v: The origin to convert.
        """
        return {}

    @always_inline("builtin")
    fn __init__(
        *, unsafe_mut_cast: Origin
    ) -> Origin[
        _mlir_origin = __mlir_attr[
            `#lit.origin.mutcast<`,
            unsafe_mut_cast._mlir_origin,
            `> : !lit.origin<`,
            Self.mut._mlir_value,
            `>`,
        ]
    ]:
        """Cast an origin to a different mutability, potentially introducing more
        mutability, which is an unsafe operation.

        Args:
            unsafe_mut_cast: The origin to cast.
        """
        return {}
