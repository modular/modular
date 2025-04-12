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
# RUN: %mojo %s

from sys.ffi import _Global

from memory import UnsafePointer
from test_utils import MoveCopyCounter, ObservableDel
from testing import assert_equal, assert_false, assert_true

from utils import Variant

alias TEST_VARIANT_POISON = _Global[
    "TEST_VARIANT_POISON", Bool, _initialize_poison
]


fn _initialize_poison() -> Bool:
    return False


fn _poison_ptr() -> UnsafePointer[Bool]:
    return TEST_VARIANT_POISON.get_or_create_ptr()


fn assert_no_poison() raises:
    assert_false(_poison_ptr().take_pointee())


struct Poison(CollectionElement):
    fn __init__(out self):
        pass

    fn __init__(out self, *, other: Self):
        _poison_ptr().init_pointee_move(True)

    fn __copyinit__(out self, other: Self):
        _poison_ptr().init_pointee_move(True)

    fn copy(self) -> Self:
        # Invokes __copyinit__, which sets the poision value.
        return self

    fn __moveinit__(out self, owned other: Self):
        _poison_ptr().init_pointee_move(True)

    fn __del__(owned self):
        _poison_ptr().init_pointee_move(True)


alias TestVariant = Variant[MoveCopyCounter, Poison]


def test_basic():
    alias IntOrString = Variant[Int, String]
    var i = IntOrString(4)
    var s = IntOrString(String("4"))

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
    assert_equal(v2[MoveCopyCounter].moved, 2)
    # test that we didn't call the other moveinit too!
    assert_no_poison()


def test_del():
    alias TestDeleterVariant = Variant[ObservableDel, Poison]
    var deleted: Bool = False
    var v1 = TestDeleterVariant(
        ObservableDel(UnsafePointer.address_of(deleted))
    )
    _ = v1^  # call __del__
    assert_true(deleted)
    # test that we didn't call the other deleter too!
    assert_no_poison()


def test_set_calls_deleter():
    alias TestDeleterVariant = Variant[ObservableDel, Poison]
    var deleted: Bool = False
    var deleted2: Bool = False
    var v1 = TestDeleterVariant(
        ObservableDel(UnsafePointer.address_of(deleted))
    )
    v1.set[ObservableDel](ObservableDel(UnsafePointer.address_of(deleted2)))
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
    alias TestDeleterVariant = Variant[ObservableDel, Poison]
    var deleted: Bool = False
    var v1 = TestDeleterVariant(
        ObservableDel(UnsafePointer.address_of(deleted))
    )
    assert_false(deleted)
    var v2 = v1.unsafe_take[ObservableDel]()
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

    var v2: Variant[Int, String] = String("something")
    v2[String] = "something else"
    assert_equal(v2[String], "something else")


def test_is_type_supported():
    var x = Variant[Float64, Int32](Int32(0))
    assert_equal(x.is_type_supported[Float64](), True)
    assert_equal(x.is_type_supported[Int32](), True)
    assert_equal(x.is_type_supported[Float32](), False)
    assert_equal(x.is_type_supported[UInt32](), False)
    var y: Variant[SIMD[DType.uint8, 2], SIMD[DType.uint8, 4]]
    assert_equal(y.is_type_supported[SIMD[DType.uint8, 2]](), True)
    assert_equal(y.is_type_supported[SIMD[DType.uint8, 4]](), True)
    assert_equal(y.is_type_supported[SIMD[DType.uint8, 8]](), False)


def main():
    test_basic()
    test_get_returns_mutable_reference()
    test_copy()
    test_explicit_copy()
    test_move()
    test_del()
    test_take_doesnt_call_deleter()
    test_set_calls_deleter()
    test_replace()
    test_is_type_supported()
