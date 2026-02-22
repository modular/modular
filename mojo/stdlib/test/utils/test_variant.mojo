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

from os import abort
from ffi import _Global

from test_utils import (
    MoveCopyCounter,
    ObservableDel,
    ConfigureTrivial,
    MoveOnly,
    ExplicitDelOnly,
    NonMovable,
    check_write_to,
)
from testing import TestSuite, assert_equal, assert_false, assert_true
from benchmark import keep

from utils import Variant

comptime TEST_VARIANT_POISON = _Global[
    "TEST_VARIANT_POISON", _initialize_poison
]


fn _initialize_poison() -> Bool:
    return False


fn _poison_ptr() -> UnsafePointer[Bool, MutExternalOrigin]:
    try:
        return TEST_VARIANT_POISON.get_or_create_ptr()
    except:
        abort("Failed to get or create TEST_VARIANT_POISON")


fn assert_no_poison() raises:
    assert_false(_poison_ptr().take_pointee())


struct Poison(ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __init__(out self, *, copy: Self):
        _poison_ptr().init_pointee_move(True)

    fn __init__(out self, *, deinit take: Self):
        _poison_ptr().init_pointee_move(True)

    fn __del__(deinit self):
        _poison_ptr().init_pointee_move(True)


comptime TestVariant = Variant[MoveCopyCounter, Poison]


def test_basic():
    comptime IntOrString = Variant[Int, String]
    var i = IntOrString(4)
    var s = IntOrString("4")

    # isa
    assert_true(i.isa[Int]())
    assert_false(i.isa[String]())
    assert_true(s.isa[String]())
    assert_false(s.isa[Int]())

    # get
    assert_equal(4, i[Int])
    assert_equal("4", s[String])
    # we don't test what happens when you `get` the wrong type.
    # have fun!

    # set
    i.set[String]("i")
    assert_false(i.isa[Int]())
    assert_true(i.isa[String]())
    assert_equal("i", i[String])


def test_copy():
    var v1 = TestVariant(MoveCopyCounter())
    var v2 = v1
    # didn't call copyinit
    assert_equal(v1[MoveCopyCounter].copied, 0)
    assert_equal(v2[MoveCopyCounter].copied, 1)
    # test that we didn't call the other copyinit too!
    assert_no_poison()


def test_explicit_copy():
    var v1 = TestVariant(MoveCopyCounter())

    # Perform explicit copy
    var v2 = v1.copy()

    # Test copy counts
    assert_equal(v1[MoveCopyCounter].copied, 0)
    assert_equal(v2[MoveCopyCounter].copied, 1)

    # test that we didn't call the other copyinit too!
    assert_no_poison()


def test_move():
    var v1 = TestVariant(MoveCopyCounter())
    var v2 = v1
    # didn't call moveinit
    assert_equal(v1[MoveCopyCounter].moved, 1)
    assert_equal(v2[MoveCopyCounter].moved, 1)
    # test that we didn't call the other moveinit too!
    assert_no_poison()


def test_del():
    comptime TestDeleterVariant = Variant[ObservableDel[], Poison]
    var deleted: Bool = False
    var v1 = TestDeleterVariant(ObservableDel(UnsafePointer(to=deleted)))
    _ = v1^  # call __del__
    assert_true(deleted)
    # test that we didn't call the other deleter too!
    assert_no_poison()


def test_set_calls_deleter():
    comptime TestDeleterVariant = Variant[ObservableDel[], Poison]
    var deleted: Bool = False
    var deleted2: Bool = False
    var v1 = TestDeleterVariant(ObservableDel(UnsafePointer(to=deleted)))
    v1.set(ObservableDel(UnsafePointer(to=deleted2)))
    assert_true(deleted)
    assert_false(deleted2)
    _ = v1^
    assert_true(deleted2)
    # test that we didn't call the poison deleter too!
    assert_no_poison()


def test_replace():
    var v1: Variant[Int, String] = 998
    var x = v1.replace[String, Int]("hello")

    assert_equal(x, 998)


def test_take_doesnt_call_deleter():
    comptime TestDeleterVariant = Variant[ObservableDel[], Poison]
    var deleted: Bool = False
    var v1 = TestDeleterVariant(ObservableDel(UnsafePointer(to=deleted)))
    assert_false(deleted)
    var v2 = v1.unsafe_take[ObservableDel[]]()
    assert_false(deleted)
    _ = v2
    assert_true(deleted)
    # test that we didn't call the poison deleter too!
    assert_no_poison()


def test_get_returns_mutable_reference():
    var v1: Variant[Int, String] = 42
    var x = v1[Int]
    assert_equal(42, x)
    x = 100
    assert_equal(100, x)
    v1.set[String]("hello")
    assert_equal(100, x)  # the x reference is still valid

    var v2: Variant[Int, String] = "something"
    v2[String] = "something else"
    assert_equal(v2[String], "something else")


def test_is_type_supported():
    var _x: Variant[Float64, Int32]
    assert_equal(_x.is_type_supported[Float64](), True)
    assert_equal(_x.is_type_supported[Int32](), True)
    assert_equal(_x.is_type_supported[Float32](), False)
    assert_equal(_x.is_type_supported[UInt32](), False)
    var _y: Variant[SIMD[DType.uint8, 2], SIMD[DType.uint8, 4]]
    assert_equal(_y.is_type_supported[SIMD[DType.uint8, 2]](), True)
    assert_equal(_y.is_type_supported[SIMD[DType.uint8, 4]](), True)
    assert_equal(_y.is_type_supported[SIMD[DType.uint8, 8]](), False)


def test_variant_works_with_move_only_types():
    var v1 = Variant[MoveOnly[Int], MoveOnly[String]](MoveOnly[Int](42))
    var v2 = v1^
    assert_equal(v2[MoveOnly[Int]].data, 42)


def test_variant_linear_type_take():
    var v = Variant[ExplicitDelOnly, String](ExplicitDelOnly(5))

    var x = v^.take[ExplicitDelOnly]()

    var data = x.data
    # Destroy before potentially raising after assert
    x^.destroy()
    assert_equal(data, 5)


def test_variant_linear_type_destroy_with():
    # Test destroying a linear variant element in-place
    var v1 = Variant[ExplicitDelOnly, String](ExplicitDelOnly(5))
    v1^.destroy_with(ExplicitDelOnly.destroy)

    # Test destroying a non-linear variant element in-place
    var v2 = Variant[ExplicitDelOnly, String]("notlinear")
    v2^.destroy_with(String.__del__)


def test_variant_linear_type_move():
    var v1 = Variant[ExplicitDelOnly, String](ExplicitDelOnly(5))
    var v2 = v1^

    v2^.destroy_with(ExplicitDelOnly.destroy)


def test_variant_trivial_del():
    comptime yes = ConfigureTrivial[del_is_trivial=True]
    comptime no = ConfigureTrivial[del_is_trivial=False]

    assert_true(Variant[yes].__del__is_trivial)
    assert_false(Variant[no].__del__is_trivial)
    assert_false(Variant[yes, no].__del__is_trivial)

    # TODO (MOCO-3016):
    # check variant of linear type
    # assert_false(Variant[LinearType].__del__is_trivial)


def test_variant_trivial_copyinit():
    comptime yes = ConfigureTrivial[copyinit_is_trivial=True]
    comptime no = ConfigureTrivial[copyinit_is_trivial=False]

    assert_true(Variant[yes].__copyinit__is_trivial)
    assert_false(Variant[no].__copyinit__is_trivial)
    assert_false(Variant[yes, no].__copyinit__is_trivial)

    # check variant of move-only type
    assert_false(Variant[MoveOnly[Int]].__copyinit__is_trivial)


def test_variant_trivial_moveinit():
    comptime yes = ConfigureTrivial[moveinit_is_trivial=True]
    comptime no = ConfigureTrivial[moveinit_is_trivial=False]

    assert_true(Variant[yes].__moveinit__is_trivial)
    assert_false(Variant[no].__moveinit__is_trivial)
    assert_false(Variant[yes, no].__moveinit__is_trivial)

    # check variant of non-movable type
    assert_false(Variant[NonMovable].__moveinit__is_trivial)


def test_variant_write_to():
    var v = Variant[Int, String](42)
    check_write_to(v, expected="42", is_repr=False)
    v = "hello"
    check_write_to(v, expected="hello", is_repr=False)


def test_variant_write_repr_to():
    var v = Variant[Int, String](42)
    check_write_to(v, expected="Variant[Int, String](Int(42))", is_repr=True)
    v = "hello"
    check_write_to(v, expected="Variant[Int, String]('hello')", is_repr=True)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
