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
"""Provides the unified `reflect[T]` / `Reflected[T]` reflection API.

This module exposes a single entry point `reflect[T]()` which returns a
`Reflected[T]` handle. The handle exposes struct introspection through methods,
without the `struct_` prefix used by the legacy free-function API:

- `is_struct()` - whether `T` is a Mojo struct type.
- `field_count()` - number of fields.
- `field_names()` - `InlineArray[StaticString, N]` of field names.
- `field_types()` - a `TypeList` of field types.
- `field_index[name]()` - index of the named field.
- `field_type[name]()` - a `Reflected[FieldT]` handle for the named field's type.
- `field_offset[name=...]()` / `field_offset[index=...]()` - byte offset.
- `field_ref[idx](s)` - reference to field at index `idx` in value `s`.

`reflect` is auto-imported via the prelude, so it is available without
an explicit import. `Reflected[T]` must be imported from `std.reflection`
when named in signatures.

Example:

```mojo
struct Point:
    var x: Int
    var y: Float64

def print_fields[T: AnyType]():
    comptime r = reflect[T]()
    comptime names = r.field_names()
    comptime for i in range(r.field_count()):
        print(names[i])

def main():
    print_fields[Point]()
```

The wrapped type is exposed as the `T` parameter, so the result of
`field_type[name]()` can be used as a type:

```mojo
def main():
    comptime r = reflect[Point]()
    comptime y_type = r.field_type["y"]()
    var v: y_type.T = 3.14  # y_type.T is Float64
```
"""

from std.builtin.variadics import ParameterList, TypeList
from std.sys.info import _TargetType, _current_target


# ===----------------------------------------------------------------------=== #
# Implementation primitives
# ===----------------------------------------------------------------------=== #
#
# These KGEN attributes implement `ContextuallyEvaluatedAttrInterface`, which
# allows them to be evaluated during elaboration after generic type parameters
# have been specialized to concrete types. This is what lets reflection work in
# generic code:
#
#   def foo[T: AnyType]():
#       comptime count = reflect[T]().field_count()
# ===----------------------------------------------------------------------=== #


comptime _field_types_of[T: AnyType] = TypeList[
    __mlir_attr[
        `#kgen.struct_field_types<`, T, `> : !kgen.param_list<`, AnyType, `>`
    ]
]

comptime _field_names_of[T: AnyType] = ParameterList[
    __mlir_attr[
        `#kgen.struct_field_names<`, T, `> : !kgen.param_list<!kgen.string>`
    ]
]


# ===----------------------------------------------------------------------=== #
# `reflect` / `Reflected[T]`
# ===----------------------------------------------------------------------=== #


@always_inline("builtin")
def reflect[T: AnyType]() -> Reflected[T]:
    """Returns a compile-time reflection handle for type `T`.

    Parameters:
        T: The type to introspect.

    Returns:
        A `Reflected[T]` handle exposing introspection methods.

    Example:
        ```mojo
        struct Point:
            var x: Int
            var y: Float64

        def main():
            comptime r = reflect[Point]()
            print(r.field_count())  # 2
        ```
    """
    return {}


struct Reflected[T: AnyType](TrivialRegisterPassable):
    """A compile-time reflection handle for a type.

    `Reflected[T]` is a zero-sized handle that exposes compile-time
    introspection of `T` through methods. Construct it via `reflect[T]()`.

    For best performance, assign the result of `reflect[T]()` and any methods
    that return type-level values (such as `field_names`, `field_types`,
    `field_count`) to `comptime` variables so the work happens at compile time.

    Parameters:
        T: The type being introspected. The wrapped type is exposed via this
            parameter, so `reflect[T]().T` is `T`.

    Example:
        ```mojo
        struct Point:
            var x: Int
            var y: Float64

        def main():
            comptime r = reflect[Point]()
            comptime if r.is_struct():
                comptime names = r.field_names()
                comptime for i in range(r.field_count()):
                    print(names[i])
        ```
    """

    @always_inline("builtin")
    def __init__(out self):
        """Constructs a reflection handle. Prefer `reflect[T]()`."""
        pass

    @always_inline("builtin")
    def is_struct(self) -> Bool:
        """Returns `True` if `T` is a Mojo struct type, `False` otherwise.

        This distinguishes Mojo struct types from MLIR primitive types (such as
        `__mlir_type.index` or `__mlir_type.i64`). The other reflection methods
        produce a compile error on non-struct types, so `is_struct` is useful
        as a `comptime if` guard when iterating over field types that may
        contain MLIR primitives.

        Returns:
            `True` if `T` is a Mojo struct type, `False` if it is an MLIR type.

        Example:
            ```mojo
            def process_type[T: AnyType]():
                comptime r = reflect[T]()
                comptime if r.is_struct():
                    print("struct with", r.field_count(), "fields")
                else:
                    print("non-struct:", r.name())
            ```
        """
        return __mlir_attr[`#kgen.is_struct_type<`, Self.T, `> : i1`]

    def name[*, qualified_builtins: Bool = False](self) -> StaticString:
        """Returns the struct name of `T`.

        Parameters:
            qualified_builtins: Whether to print fully qualified builtin type
                names (e.g. `std.builtin.int.Int`) or shorten them
                (e.g. `Int`).

        Returns:
            Type name.

        Example:
            ```mojo
            struct Point:
                var x: Int
                var y: Float64

            def main():
                comptime r = reflect[Point]()
                print(r.name())  # "Point" (or module-qualified if defined)
            ```
        """
        return StaticString(
            __mlir_attr[
                `#kgen.get_type_name<`,
                Self.T,
                `, `,
                qualified_builtins._mlir_value,
                `> : !kgen.string`,
            ]
        )

    def base_name(self) -> StaticString:
        """Returns the name of the base type of a parameterized type.

        For parameterized types like `List[Int]`, this returns `"List"`.
        For non-parameterized types, it returns the type's simple name.

        Unlike `name`, this method strips type parameters and returns only the
        unqualified base type name.

        Returns:
            The unqualified name of the base type as a `StaticString`.

        Example:
            ```mojo
            from std.collections import List, Dict

            def main():
                print(reflect[List[Int]]().base_name())          # "List"
                print(reflect[Dict[String, Int]]().base_name())  # "Dict"
                print(reflect[Int]().base_name())                # "Int"
            ```
        """
        return StaticString(
            __mlir_attr[
                `#kgen.get_base_type_name<`,
                Self.T,
                `> : !kgen.string`,
            ]
        )

    @always_inline("builtin")
    def field_count(self) -> Int:
        """Returns the number of fields in struct `T`.

        Constraints:
            `T` must be a struct type.

        Returns:
            The number of fields in the struct.
        """
        return _field_types_of[Self.T]().size

    def field_types(self) -> _field_types_of[Self.T]:
        """Returns the types of all fields in struct `T` as a `TypeList`.

        For nested structs this returns the struct type itself, not its
        flattened fields.

        Constraints:
            `T` must be a struct type.

        Returns:
            A `TypeList` with one entry per field in the struct.

        Example:
            ```mojo
            struct Point:
                var x: Int
                var y: Float64

            def main():
                comptime r = reflect[Point]()
                comptime types = r.field_types()
                comptime for i in range(r.field_count()):
                    print(reflect[types[i]]().name())
            ```
        """
        return {}

    def field_names(
        self,
    ) -> InlineArray[StaticString, _field_types_of[Self.T]().size]:
        """Returns the names of all fields in struct `T`.

        Constraints:
            `T` must be a struct type.

        Returns:
            An `InlineArray` of `StaticString`, one entry per field.
        """
        comptime count = _field_types_of[Self.T]().size
        comptime raw = _field_names_of[Self.T]()

        # Safety: uninitialized=True is safe because the comptime for loop
        # below initializes every element.
        var result = InlineArray[StaticString, count](uninitialized=True)

        comptime for i in range(raw.size):
            result[i] = comptime (StaticString(raw[i]))

        return result^

    def field_index[name: StringLiteral](self) -> Int:
        """Returns the index of the field with the given name in struct `T`.

        Note: `T` must be a concrete type, not a generic type parameter, when
        looking up by name.

        Parameters:
            name: The name of the field to look up.

        Returns:
            The zero-based index of the field.
        """
        comptime str_value = name.value
        return Int(
            mlir_value=__mlir_attr[
                `#kgen.struct_field_index_by_name<`,
                Self.T,
                `, `,
                str_value,
                `> : index`,
            ]
        )

    def field_type[
        name: StringLiteral,
    ](
        self,
    ) -> Reflected[
        __mlir_attr[
            `#kgen.struct_field_type_by_name<`,
            Self.T,
            `, `,
            name.value,
            `> : `,
            AnyType,
        ]
    ]:
        """Returns a reflection handle for the type of the named field.

        The returned handle's `T` parameter is the field's type, so
        `r.field_type["x"]().T` can be used in type position.

        Note: `T` must be a concrete type, not a generic type parameter, when
        looking up by name.

        Parameters:
            name: The name of the field.

        Returns:
            A `Reflected` handle whose `T` is the named field's type.

        Example:
            ```mojo
            struct Point:
                var x: Int
                var y: Float64

            def main():
                comptime r = reflect[Point]()
                comptime y_type = r.field_type["y"]()
                var v: y_type.T = 3.14  # y_type.T is Float64
            ```
        """
        return {}

    # `nodebug` (not `builtin`) because the body emits a `lit.ref.struct.ger`
    # MLIR op, which is not legal inside a `@always_inline("builtin")` function.
    @always_inline("nodebug")
    def field_ref[
        idx: Int
    ](self, ref s: Self.T) -> ref[s] _field_types_of[Self.T]()[idx]:
        """Returns a reference to the field at the given index in `s`.

        Returns a reference rather than a copy, so this works with
        non-copyable field types and supports mutation through the result.

        Parameters:
            idx: The zero-based index of the field.

        Args:
            s: The struct value to access.

        Constraints:
            `T` must be a struct type. `idx` must be in range
            `[0, field_count())`.

        Returns:
            A reference to the field at the specified index, with the same
            mutability as `s`.

        Example:
            ```mojo
            @fieldwise_init
            struct Container:
                var id: Int
                var name: String

            def main():
                var c = Container(id=1, name="test")
                comptime r = reflect[Container]()
                r.field_ref[0](c) = 42  # mutates c.id
            ```
        """
        # Emit `lit.ref.struct.ger` with index-access form. The op accepts
        # StructType, ParamType, and ClosureType element types, so this works
        # for concrete structs, generic type parameters, and closures alike.
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.struct.ger`[
                index=idx._int_mlir_index(),
                _type=__mlir_type[
                    `!lit.ref<`,
                    _field_types_of[Self.T]()[idx],
                    `, `,
                    origin_of(s)._mlir_origin,
                    `>`,
                ],
            ](__get_mvalue_as_litref(s))
        )

    def field_offset[
        *,
        name: StringLiteral,
        target: _TargetType = _current_target(),
    ](self) -> Int:
        """Returns the byte offset of the named field within struct `T`.

        Accounts for alignment padding between fields. Computed using the
        target's data layout.

        Parameters:
            name: The name of the field.
            target: The target architecture (defaults to the current target).

        Constraints:
            `T` must be a struct type with a field of the given name.

        Returns:
            The byte offset of the field from the start of the struct.

        Example:
            ```mojo
            struct Point:
                var x: Int      # offset 0
                var y: Float64  # offset 8

            def main():
                comptime r = reflect[Point]()
                comptime x_off = r.field_offset[name="x"]()  # 0
                comptime y_off = r.field_offset[name="y"]()  # 8
            ```
        """
        comptime str_value = name.value
        return Int(
            mlir_value=__mlir_attr[
                `#kgen.struct_field_offset_by_name<`,
                Self.T,
                `, `,
                str_value,
                `, `,
                target,
                `> : index`,
            ]
        )

    def field_offset[
        *,
        index: Int,
        target: _TargetType = _current_target(),
    ](self) -> Int:
        """Returns the byte offset of the field at the given index.

        Accounts for alignment padding between fields. Computed using the
        target's data layout.

        Parameters:
            index: The zero-based index of the field.
            target: The target architecture (defaults to the current target).

        Constraints:
            `T` must be a struct type. `index` must be in range
            `[0, field_count())`.

        Returns:
            The byte offset of the field from the start of the struct.
        """
        return Int(
            mlir_value=__mlir_attr[
                `#kgen.struct_field_offset_by_index<`,
                Self.T,
                `, `,
                index._int_mlir_index(),
                `, `,
                target,
                `> : index`,
            ]
        )
