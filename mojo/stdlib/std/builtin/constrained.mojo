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
"""Implements compile-time constraints.

These are Mojo built-ins, so you don't need to import them.
"""
from collections.string.string_slice import _get_kgen_string
from reflection import (
    get_type_name,
    struct_field_names,
    struct_field_types,
)
from reflection.type_info import _unqualified_type_name


@deprecated("Use `comptime assert` instead")
@always_inline("nodebug")
fn constrained[cond: Bool, msg: StaticString, *extra: StaticString]():
    """Asserts that the condition must be true at compile time.

    The `constrained()` function introduces a compile-time constraint on the
    enclosing function. If the condition is true at compile time, the constraint
    has no effect. If the condition is false, compilation fails and the message
    is displayed.

    This is similar to `static_assert` in C++. It differs from
    [`debug_assert()`](/mojo/std/builtin/debug_assert/debug_assert), which
    is a run-time assertion.

    Example:

    ```mojo
    fn half[dtype: DType](a: Scalar[dtype]) -> Scalar[dtype]:
        comptime assert dtype.is_numeric(), "dtype must be numeric."
        return a / 2

    def main():
        print(half(UInt8(5)))  # prints 2
        print(half(Scalar[DType.bool](True)))  # constraint failed:
                                               #     dtype must be numeric.
    ```

    Parameters:
        cond: The bool value to assert.
        msg: The message to display on failure.
        extra: Additional messages to concatenate to msg.

    """
    __mlir_op.`kgen.param.assert`[
        cond = cond.__mlir_i1__(),
        message = _get_kgen_string[msg, extra](),
    ]()


@deprecated("Use `comptime assert` instead")
@always_inline("nodebug")
fn constrained[cond: Bool]():
    """Asserts that the condition must be true at compile time.

    The `constrained()` function introduces a compile-time constraint on the
    enclosing function. If the condition is true at compile time, the constraint
    has no effect. If the condition is false, compilation fails and a generic
    message is displayed.

    This is similar to `static_assert` in C++. It differs from
    [`debug_assert()`](/mojo/std/builtin/debug_assert/debug_assert), which
    is a run-time assertion.

    For an example, see the
    [first overload](/mojo/std/builtin/constrained/constrained).

    Parameters:
        cond: The bool value to assert.
    """
    comptime assert cond, "param assertion failed"


@always_inline("nodebug")
fn _constrained_conforms_to[
    cond: Bool,
    *,
    Parent: AnyType,
    Element: AnyType,
    ParentConformsTo: StaticString,
    ElementConformsTo: StaticString = ParentConformsTo,
]():
    comptime parent_type_name = get_type_name[Parent]()
    comptime elem_type_name = get_type_name[Element]()
    # TODO(MOCO-2901): Support traits in get_type_name
    #   comptime trait_name = get_type_name[ParentConformsTo]()
    comptime parent_conforms_to_trait_name = ParentConformsTo
    comptime elem_conforms_to_trait_name = ElementConformsTo

    # Construct a message like:
    #     List(Equatable) conformance requires Foo(Equatable) conformance, which
    #     is not satisfied.
    comptime assert cond, StaticString(
        _get_kgen_string[
            parent_type_name,
            "(",
            parent_conforms_to_trait_name,
            ") conformance requires ",
            elem_type_name,
            "(",
            elem_conforms_to_trait_name,
            ") conformance, which is not satisfied.",
        ]()
    )


@always_inline("nodebug")
fn _constrained_field_conforms_to[
    cond: Bool,
    *,
    Parent: AnyType,
    FieldIndex: Int,
    ParentConformsTo: StaticString,
    FieldConformsTo: StaticString = ParentConformsTo,
]():
    """Asserts that a struct field conforms to a trait at compile time.

    This helper is used in default trait implementations that use reflection
    to operate on all fields of a struct. It produces a clear error message
    when a field doesn't conform to the required trait.

    Parameters:
        cond: The conformance condition (e.g., `conforms_to(FieldType, Trait)`).
        Parent: The struct type being checked.
        FieldIndex: The index of the field in the struct.
        ParentConformsTo: The trait the parent is trying to conform to.
        FieldConformsTo: The trait the field must conform to
            (defaults to ParentConformsTo).
    """
    comptime names = struct_field_names[Parent]()
    comptime types = struct_field_types[Parent]()
    comptime field_name = names[FieldIndex]
    comptime FieldType = types[FieldIndex]
    comptime parent_type_name = _unqualified_type_name[Parent]()
    comptime field_type_name = _unqualified_type_name[FieldType]()

    # Construct a message like:
    #     Could not derive Equatable for Point - member field `x: Int` does not
    #     implement Equatable
    comptime assert cond, StaticString(
        _get_kgen_string[
            "Could not derive ",
            ParentConformsTo,
            " for ",
            parent_type_name,
            " - member field `",
            field_name,
            ": ",
            field_type_name,
            "` does not implement ",
            FieldConformsTo,
        ]()
    )
