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
"""Tests for type and function name introspection APIs."""

from sys.info import CompilationTarget, _current_target, is_64bit

from collections import List, Optional

from reflection import (
    get_linkage_name,
    get_type_name,
    get_function_name,
    get_base_type_name,
)
from reflection.type_info import _unqualified_type_name
from testing import assert_equal, assert_true, assert_false
from testing import TestSuite


fn my_func() -> Int:
    return 0


def test_get_linkage_name():
    var name = get_linkage_name[my_func]()
    assert_equal(name, "test_type_info::my_func()")


def test_get_linkage_name_nested():
    fn nested_func(x: Int) -> Int:
        return x

    var name = get_linkage_name[nested_func]()
    assert_equal(
        name,
        "test_type_info::test_get_linkage_name_nested()_nested_func(::Int)",
    )


fn your_func[x: Int]() raises -> Int:
    return x


def test_get_linkage_name_parameterized():
    var name = get_linkage_name[your_func[7]]()
    assert_equal(name, "test_type_info::your_func[::Int](),x=7")


def test_get_linkage_name_on_itself():
    var name = get_linkage_name[_current_target]()
    assert_equal(name, "std::sys::info::_current_target()")


def test_get_function_name():
    var name = get_function_name[my_func]()
    assert_equal(name, "my_func")


def test_get_function_name_nested():
    fn nested_func(x: Int) -> Int:
        return x

    var name2 = get_function_name[nested_func]()
    assert_equal(name2, "nested_func")


def test_get_function_name_parameterized():
    var name = get_function_name[your_func]()
    assert_equal(name, "your_func")

    var name2 = get_function_name[your_func[7]]()
    assert_equal(name2, "your_func")


struct WhatsMyName:
    pass


struct ImGeneric[T: AnyType]:
    pass


def test_unqualified_type_name():
    assert_equal(_unqualified_type_name[WhatsMyName](), "WhatsMyName")
    assert_equal(_unqualified_type_name[Int](), "Int")
    assert_equal(_unqualified_type_name[String](), "String")
    assert_equal(_unqualified_type_name[Int8](), "SIMD[DType.int8, 1]")

    # TODO: strip the module name from the inner type name
    assert_equal(
        _unqualified_type_name[ImGeneric[WhatsMyName]](),
        "ImGeneric[test_type_info.WhatsMyName]",
    )


def test_get_type_name():
    var name = get_type_name[Int]()
    assert_equal(name, "Int")

    name = get_type_name[Int, qualified_builtins=True]()
    assert_equal(name, "std.builtin.int.Int")


def test_get_type_name_nested():
    fn nested_func[T: AnyType]() -> StaticString:
        return get_type_name[T]()

    var name = nested_func[String]()
    assert_equal(name, "String")


def test_get_type_name_simd():
    var name = get_type_name[Float32]()
    assert_equal(name, "SIMD[DType.float32, 1]")

    name = get_type_name[SIMD[DType.uint16, 4], qualified_builtins=True]()
    assert_equal(
        name, "std.builtin.simd.SIMD[std.builtin.dtype.DType.uint16, 4]"
    )


@fieldwise_init
struct Bar[x: Int, f: Float32 = 1.3](Intable):
    fn __int__(self) -> Int:
        return self.x

    var y: Int
    var z: Float64


@fieldwise_init
struct Foo[
    T: Intable, //, b: T, c: Bool, d: NoneType = None, e: StaticString = "hello"
]:
    pass


def test_get_type_name_non_scalar_simd_value():
    var name = get_type_name[
        Foo[SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0), True]
    ]()
    assert_equal(
        name,
        (
            "test_type_info.Foo[SIMD[DType.float32, 4], "
            '[1, 2, 3, 4] : SIMD[DType.float32, 4], True, None, {"hello\0", 5}]'
        ),
    )

    name = get_type_name[
        Foo[SIMD[DType.bool, 4](True, False, True, False), True]
    ]()
    assert_equal(
        name,
        (
            "test_type_info.Foo[SIMD[DType.bool, 4], "
            "[True, False, True, False] : SIMD[DType.bool, 4], "
            'True, None, {"hello\0", 5}]'
        ),
    )


def test_get_type_name_struct():
    var name = get_type_name[Foo[Bar[2](y=3, z=4.1), True]]()
    assert_equal(
        name,
        (
            "test_type_info.Foo["
            "test_type_info.Bar[2, 1.29999995 : SIMD[DType.float32, 1]], "
            "{3, 4.0999999999999996 : SIMD[DType.float64, 1]}, "
            'True, None, {"hello\0", 5}]'
        ),
    )


def test_get_type_name_partially_bound_type():
    var name = get_type_name[Foo[Bar[2](y=3, z=0.125)]]()
    assert_equal(
        name,
        (
            "test_type_info.Foo["
            "test_type_info.Bar[2, 1.29999995 : SIMD[DType.float32, 1]], "
            '{3, 0.125 : SIMD[DType.float64, 1]}, ?, None, {"hello\0", 5}]'
        ),
    )


def test_get_type_name_unprintable():
    var name = get_type_name[CompilationTarget[_current_target()]]()
    assert_equal(name, "std.sys.info.CompilationTarget[<unprintable>]")


# Generic struct for testing types with constructor calls in parameters
struct WrapperWithValue[T: AnyType, //, value: T]:
    pass


@fieldwise_init
struct SimpleParam(TrivialRegisterType):
    var b: Bool


def test_get_type_name_ctor_param_direct():
    """Test that direct usage of types with constructor calls works."""
    var name = get_type_name[WrapperWithValue[SimpleParam(True)]]()
    assert_equal(
        name,
        "test_type_info.WrapperWithValue[test_type_info.SimpleParam, True]",
    )


# Generic struct for testing nested parametric types
struct GenericWrapper[T: AnyType]:
    pass


# Generic struct with multiple type parameters
struct Pair[T: AnyType, U: AnyType]:
    pass


def test_get_type_name_nested_parametric_direct():
    """Test that directly using nested parametric types works."""
    # Direct usage works fine
    var name = get_type_name[GenericWrapper[List[String]]]()
    assert_equal(name, "test_type_info.GenericWrapper[List[String]]")

    # Deeply nested direct usage also works
    name = get_type_name[GenericWrapper[GenericWrapper[Int]]]()
    assert_equal(
        name,
        "test_type_info.GenericWrapper[test_type_info.GenericWrapper[Int]]",
    )


def test_get_type_name_alias():
    comptime T = Bar[5]
    var name = get_type_name[T]()
    assert_equal(
        name, "test_type_info.Bar[5, 1.29999995 : SIMD[DType.float32, 1]]"
    )

    # Also test parametric aliases (i.e. unbound parameters).
    comptime R = Bar[_]
    name = get_type_name[R]()
    assert_equal(
        name, "test_type_info.Bar[?, 1.29999995 : SIMD[DType.float32, 1]]"
    )


fn _get_type_name_generic[T: AnyType]() -> StaticString:
    """Helper function to test get_type_name through generic parameter."""
    return get_type_name[T]()


def test_get_type_name_through_generic():
    """Test get_type_name through a generic function."""
    assert_equal(_get_type_name_generic[Int](), "Int")
    assert_equal(_get_type_name_generic[String](), "String")


struct UIndexParam[value: Scalar[DType.uint]]:
    pass


struct IndexParam[value: Scalar[DType.int]]:
    pass


def test_get_type_name_uindex_index_simd_value():
    """Test that DType.uint and DType.int SIMD values are printed correctly."""

    # Test unsigned uindex value - should print as large positive number
    comptime uint_max: Scalar[DType.uint] = Scalar[DType.uint].MAX
    var name = get_type_name[UIndexParam[uint_max]]()
    if is_64bit():
        assert_equal(
            name,
            (
                "test_type_info.UIndexParam[18446744073709551615 :"
                " SIMD[DType.uindex, 1]]"
            ),
        )
    else:
        assert_equal(
            name,
            "test_type_info.UIndexParam[4294967295 : SIMD[DType.uindex, 1]]",
        )

    # Test signed index value for comparison - should print as -1
    comptime neg_one: Scalar[DType.int] = -1
    name = get_type_name[IndexParam[neg_one]]()
    assert_equal(
        name,
        "test_type_info.IndexParam[-1 : SIMD[DType.index, 1]]",
    )


# ===----------------------------------------------------------------------=== #
# Base Type Reflection Tests (Issue #5735)
# ===----------------------------------------------------------------------=== #


def test_get_base_type_name_basic():
    """Test get_base_type_name with simple and parameterized types."""
    # For non-parameterized types, get_base_type_name returns the type's name
    assert_equal(get_base_type_name[Int](), "Int")

    # For parameterized types, get_base_type_name returns the base type name
    assert_equal(get_base_type_name[List[Int]](), "List")

    # Different parameterizations of the same type have the same base name
    assert_equal(
        get_base_type_name[List[Int]](), get_base_type_name[List[String]]()
    )


def test_get_base_type_name_user_defined():
    """Test get_base_type_name with user-defined generic types."""
    # User-defined generics should have the same base name
    assert_equal(
        get_base_type_name[GenericWrapper[Int]](),
        get_base_type_name[GenericWrapper[String]](),
    )

    # Types with multiple parameters
    assert_equal(get_base_type_name[Pair[Int, String]](), "Pair")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
