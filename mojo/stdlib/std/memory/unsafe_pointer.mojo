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
"""Defines `UnsafePointer` and related aliases as backward-compatible names
for `Pointer`.

`UnsafePointer` is a `comptime` alias for `Pointer`, kept for code written
before the two pointer types were unified. Prefer `Pointer` directly in new
code; see `std.memory.pointer` for the type's docstring and its `unsafe_`
method surface.
"""

from std.collections import OptionalReg
from std.memory.pointer import AddressSpace, OptionalPointer, Pointer

# ===----------------------------------------------------------------------=== #
# unsafe_cast
# ===----------------------------------------------------------------------=== #


@always_inline
@doc_hidden
def unsafe_cast[
    from_mut: Bool,
    from_type: AnyType,
    from_origin: Origin[mut=from_mut],
    from_address_space: AddressSpace,
    mut: Bool = from_mut,
    //,
    *,
    Type: AnyType = from_type,
    origin: Origin[mut=mut] = from_origin,
    address_space: AddressSpace = from_address_space,
](
    pointer: OptionalPointer[
        from_type, from_origin, address_space=from_address_space
    ],
    out result: OptionalPointer[Type, origin, address_space=address_space],
):
    result = Pointer(to=pointer).unsafe_bitcast[type_of(result)]()[]


@always_inline
@doc_hidden
def unsafe_cast[
    from_mut: Bool,
    from_type: AnyType,
    from_origin: Origin[mut=from_mut],
    from_address_space: AddressSpace,
    mut: Bool = from_mut,
    //,
    *,
    Type: AnyType = from_type,
    origin: Origin[mut=mut] = from_origin,
    address_space: AddressSpace = from_address_space,
](
    pointer: OptionalReg[
        Pointer[
            from_type,
            from_origin,
            address_space=from_address_space,
        ]
    ],
    out result: OptionalReg[Pointer[Type, origin, address_space=address_space]],
):
    result = Pointer(to=pointer).unsafe_bitcast[type_of(result)]()[]


@always_inline("nodebug")
@doc_hidden
def pointer_to_int(pointer: OptionalPointer[...]) -> Int:
    return Pointer(to=pointer).unsafe_bitcast[Int]()[]


# ===----------------------------------------------------------------------=== #
# UnsafePointer aliases
# ===----------------------------------------------------------------------=== #


comptime MutUnsafePointer[
    T: AnyType,
    origin: MutOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = Pointer[mut=True, T, origin, address_space=address_space]
"""A mutable unsafe pointer.

Parameters:
    T: The pointee type.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime ImmUnsafePointer[
    T: AnyType,
    origin: ImmOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = Pointer[T, origin, address_space=address_space]
"""An immutable unsafe pointer.

Parameters:
    T: The pointee type.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""


@doc_hidden
@deprecated(use=ImmUnsafePointer)
comptime ImmutUnsafePointer = ImmUnsafePointer

comptime OpaquePointer[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = Pointer[NoneType, origin, address_space=address_space]
"""An opaque pointer, equivalent to the C `(const) void*` type.

Parameters:
    mut: Whether the pointer is mutable.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime MutOpaquePointer[
    origin: MutOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = OpaquePointer[origin, address_space=address_space]
"""A mutable opaque pointer, equivalent to the C `void*` type.

Parameters:
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime ImmOpaquePointer[
    origin: ImmOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = OpaquePointer[origin, address_space=address_space]
"""An immutable opaque pointer, equivalent to the C `const void*` type.

Parameters:
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""


@doc_hidden
@deprecated(use=ImmOpaquePointer)
comptime ImmutOpaquePointer = ImmOpaquePointer

comptime OptionalUnsafePointer[
    mut: Bool,
    //,
    T: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = Optional[Pointer[T, origin, address_space=address_space]]
"""An optional (nullable) `UnsafePointer`.

Parameters:
    mut: The mutability of the pointer.
    T: The type of the pointee.
    origin: The origin of the pointer.
    address_space: The address space of the pointer.
"""

comptime UnsafePointer[
    mut: Bool,
    //,
    T: AnyType,
    origin: Origin[mut=mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = Pointer[T, origin, address_space=address_space]
"""An indirect reference to one or more values of `T` consecutively in
memory, and can refer to uninitialized memory.

Parameters:
    mut: Whether the pointer is mutable.
    T: The type the pointer points to.
    origin: The origin of the memory being addressed.
    address_space: The address space of the pointer.
"""


comptime _UnsafeDanglingPluginHookFnType = def[alignment: Int]() thin -> Int
"""Plugin-hook signature for `PluginHooks.unsafe_dangling_fn`; keep in sync with `Pointer.unsafe_dangling`."""
