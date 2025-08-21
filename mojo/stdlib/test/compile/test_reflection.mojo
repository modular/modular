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

from compile.reflection import get_linkage_name, get_type_name
from testing import assert_equal
from sys.info import _current_target, CompilationTarget


fn my_func() -> Int:
    return 0


def test_get_linkage_name():
    var name = get_linkage_name[my_func]()
    assert_equal(name, "test_reflection::my_func()")


def test_get_linkage_name_nested():
    fn nested_func(x: Int) -> Int:
        return x

    var name2 = get_linkage_name[nested_func]()
    assert_equal(
        name2,
        "test_reflection::test_get_linkage_name_nested()_nested_func(::Int)",
    )


fn your_func[x: Int]() raises -> Int:
    return x


def test_get_linkage_name_parameterized():
    var name = get_linkage_name[your_func[7]]()
    assert_equal(name, "test_reflection::your_func[::Int](),x=7")


def test_get_linkage_name_on_itself():
    var name = get_linkage_name[_current_target]()
    assert_equal(name, "stdlib::sys::info::_current_target()")


def test_get_type_name():
    var name = get_type_name[Int]()
    assert_equal(name, "Int")

    name = get_type_name[Int, qualified_builtins=True]()
    assert_equal(name, "stdlib.builtin.int.Int")


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
        name, "stdlib.builtin.simd.SIMD[stdlib.builtin.dtype.DType.uint16, 4]"
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
            "test_reflection.Foo[SIMD[DType.float32, 4], "
            '[1, 2, 3, 4] : SIMD[DType.float32, 4], True, None, {"hello", 5}]'
        ),
    )

    name = get_type_name[
        Foo[SIMD[DType.bool, 4](True, False, True, False), True]
    ]()
    assert_equal(
        name,
        (
            "test_reflection.Foo[SIMD[DType.bool, 4], "
            "[True, False, True, False] : SIMD[DType.bool, 4], "
            'True, None, {"hello", 5}]'
        ),
    )


def test_get_type_name_struct():
    var name = get_type_name[Foo[Bar[2](y=3, z=4.1), True]]()
    assert_equal(
        name,
        (
            "test_reflection.Foo["
            "test_reflection.Bar[2, 1.29999995 : SIMD[DType.float32, 1]], "
            "{3, 4.0999999999999996 : SIMD[DType.float64, 1]}, "
            'True, None, {"hello", 5}]'
        ),
    )


def test_get_type_name_partially_bound_type():
    var name = get_type_name[Foo[Bar[2](y=3, z=0.125)]]()
    assert_equal(
        name,
        (
            "test_reflection.Foo["
            "test_reflection.Bar[2, 1.29999995 : SIMD[DType.float32, 1]], "
            '{3, 0.125 : SIMD[DType.float64, 1]}, ?, None, {"hello", 5}]'
        ),
    )


def test_get_type_name_unprintable():
    var name = get_type_name[CompilationTarget[_current_target()]]()
    assert_equal(name, "stdlib.sys.info.CompilationTarget[<unprintable>]")


def test_get_type_name_alias():
    alias T = Bar[5]
    var name = get_type_name[T]()
    assert_equal(
        name, "test_reflection.Bar[5, 1.29999995 : SIMD[DType.float32, 1]]"
    )

    # Also test parametric aliases (i.e. unbound parameters).
    alias R = Bar[_]
    name = get_type_name[R]()
    assert_equal(
        name, "test_reflection.Bar[?, 1.29999995 : SIMD[DType.float32, 1]]"
    )


def main():
    test_get_linkage_name()
    test_get_linkage_name_nested()
    test_get_linkage_name_parameterized()
    test_get_linkage_name_on_itself()
    test_get_type_name()
    test_get_type_name_nested()
    test_get_type_name_simd()
    test_get_type_name_non_scalar_simd_value()
    test_get_type_name_struct()
    test_get_type_name_partially_bound_type()
    test_get_type_name_unprintable()
    test_get_type_name_alias()
