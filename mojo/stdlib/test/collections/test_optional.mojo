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

from std.builtin.device_passable import DevicePassable
from std.collections import OptionalReg
from std.sys import size_of

from std.testing import *
from std.testing import TestSuite
from test_utils import MoveOnly, check_write_to


def test_basic() raises:
    # Assign to vars to remove compiler warnings.
    var false = False
    var true = True

    var a = Optional(1)
    var b = Optional[Int](None)

    assert_true(a)
    assert_false(b)

    assert_true(a and true)
    assert_true(true and a)
    assert_false(a and false)

    assert_false(b and true)
    assert_false(b and false)

    assert_true(a or true)
    assert_true(a or false)

    assert_true(b or true)
    assert_false(b or false)

    assert_equal(1, a.value())

    # Test invert operator
    assert_false(~a)
    assert_true(~b)

    # TODO(27776): can't inline these, they need to be mutable lvalues
    var a1 = a.or_else(2)
    var b1 = b.or_else(2)

    assert_equal(1, a1)
    assert_equal(2, b1)

    assert_equal(1, a.unsafe_take())

    # TODO: this currently only checks for mutable references.
    # We may want to come back and add an immutable test once
    # there are the language features to do so.
    var a2 = Optional(1)
    a2.value() = 2
    assert_equal(a2.value(), 2)


def test_optional_reg_basic() raises:
    # Assign to vars to remove compiler warnings
    var false = False
    var true = True

    var val: OptionalReg[Int] = None
    assert_false(val.__bool__())

    val = 15
    assert_true(val.__bool__())

    assert_equal(val.value(), 15)

    assert_true(val or False)
    assert_true(val and True)

    assert_true(false or val)
    assert_true(true and val)

    assert_equal(OptionalReg[Int]().or_else(33), 33)
    assert_equal(OptionalReg[Int](42).or_else(33), 42)


def test_optional_is() raises:
    a = Optional(1)
    assert_false(a is None)

    a = Optional[Int](None)
    assert_true(a is None)


def test_optional_isnot() raises:
    a = Optional(1)
    assert_true(a is not None)

    a = Optional[Int](None)
    assert_false(a is not None)


def test_optional_reg_is() raises:
    a = OptionalReg(1)
    assert_false(a is None)

    a = OptionalReg[Int](None)
    assert_true(a is None)


def test_optional_reg_isnot() raises:
    a = OptionalReg(1)
    assert_true(a is not None)

    a = OptionalReg[Int](None)
    assert_false(a is not None)


def test_optional_take_mutates() raises:
    var opt1 = Optional[Int](5)

    assert_true(opt1)

    var value: Int = opt1.take()

    assert_equal(value, 5)
    # The optional should now be empty
    assert_false(opt1)


def test_optional_explicit_copy() raises:
    var v1 = Optional[String]("test")

    var v2 = v1.copy()

    assert_equal(v1.value(), "test")
    assert_equal(v2.value(), "test")

    v2.value() += "ing"

    assert_equal(v1.value(), "test")
    assert_equal(v2.value(), "testing")


def test_optional_conformance() raises:
    assert_true(conforms_to(Optional[Int], Writable))


def test_optional_conditional_conformances() raises:
    assert_true(conforms_to(Optional[Int], Writable))
    assert_true(conforms_to(Optional[String], Writable))
    assert_false(conforms_to(Optional[MoveOnly[Int]], Writable))

    assert_true(conforms_to(Optional[Int], Copyable))
    assert_true(conforms_to(Optional[String], Copyable))
    assert_false(conforms_to(Optional[MoveOnly[Int]], Copyable))

    assert_true(conforms_to(Optional[Int], Hashable))
    assert_true(conforms_to(Optional[String], Hashable))
    assert_false(conforms_to(Optional[MoveOnly[Int]], Hashable))


def test_optional_write_to() raises:
    check_write_to(Optional[Int](None), expected="None", is_repr=False)
    check_write_to(Optional[Int](42), expected="42", is_repr=False)


def test_optional_write_repr_to() raises:
    check_write_to(
        Optional[Int](None), expected="Optional[Int](None)", is_repr=True
    )
    check_write_to(
        Optional[Int](42), expected="Optional[Int](Int(42))", is_repr=True
    )


def test_optional_hash() raises:
    # Equal optionals should produce the same hash.
    assert_equal(hash(Optional(1)), hash(Optional(1)))
    assert_equal(hash(Optional("hello")), hash(Optional("hello")))

    # None optionals should hash equally.
    assert_equal(hash(Optional[Int](None)), hash(Optional[Int](None)))

    # Different values should (likely) produce different hashes.
    assert_not_equal(hash(Optional(1)), hash(Optional(2)))

    # Present vs None should produce different hashes.
    assert_not_equal(hash(Optional(0)), hash(Optional[Int](None)))

    # A value stored in an Optional should have a different hash than that
    # value on its own.
    assert_not_equal(hash(1), hash(Optional(1)))


def test_optional_equality() raises:
    o = Optional(10)
    n = Optional[Int]()
    assert_true(o == 10)
    assert_true(o != 11)
    assert_true(o != n)
    assert_true(o != None)
    assert_true(n != 11)
    assert_true(n == n)
    assert_true(n == None)


def test_optional_copied() raises:
    var data = "foo"

    var opt_ref: Optional[Pointer[String, origin_of(data)]] = Optional(
        Pointer(to=data)
    )

    # Copy the optional Pointer value.
    var opt_owned: Optional[String] = opt_ref.copied()

    assert_equal(opt_owned.value(), "foo")


def test_optional_unwrap() raises:
    var a = Optional(123)
    assert_true(a)
    assert_equal(123, a[])
    a = Optional[Int](None)
    with assert_raises(contains="EmptyOptionalError"):
        _ = a[]


def test_optional_repr_wrap() raises:
    var o = Optional(10)
    assert_equal(repr(o), "Optional[Int](Int(10))")
    o = None
    assert_equal(repr(o), "Optional[Int](None)")


def test_optional_iter() raises:
    var o = Optional(10)
    var it = o.__iter__()
    assert_equal(it.bounds()[0], 1)
    assert_equal(it.bounds()[1].value(), 1)
    assert_equal(it.__next__(), 10)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration

    var called = False
    var o2 = Optional(20)
    for _value in o2:
        called = True
    assert_true(called)


def test_optional_iter_empty() raises:
    var o = Optional[Int](None)
    var it = o.__iter__()
    assert_equal(it.bounds()[0], 0)
    assert_equal(it.bounds()[1].value(), 0)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration

    var called = False
    var o2 = Optional[Int](None)
    for _value in o2:
        called = True
    assert_false(called)


def test_optional_of_move_only_type() raises:
    var opt1 = Optional(MoveOnly[Int](5))
    # Test moving the optional
    var opt2 = opt1^
    # Test moving out of the optional
    var val: MoveOnly[Int] = opt2.take()

    assert_equal(val.data, 5)
    assert_false(opt2)

    # Test move-only default value
    val = opt2^.or_else(MoveOnly(10))
    assert_equal(val.data, 10)


def test_nicheable_size() raises:
    comptime PointerType = Pointer[Int, AnyOrigin[mut=True]]

    assert_equal(size_of[Optional[PointerType]](), size_of[PointerType]())
    assert_true(size_of[Optional[Int]]() > size_of[Int]())


struct NotEquatable(Movable):
    pass


def test_optional_conforms_to_equatable() raises:
    assert_true(conforms_to(Optional[Int], Equatable))
    assert_false(conforms_to(Optional[NotEquatable], Equatable))


def test_optional_conditional_register_passable() raises:
    assert_true(conforms_to(Optional[Int], RegisterPassable))
    assert_true(conforms_to(Optional[Bool], RegisterPassable))
    assert_false(conforms_to(Optional[List[Int]], RegisterPassable))
    assert_false(conforms_to(Optional[String], RegisterPassable))


def test_optional_conditional_device_passable() raises:
    assert_true(conforms_to(Optional[Int], DevicePassable))
    assert_true(conforms_to(Optional[Scalar[DType.float32]], DevicePassable))
    assert_false(conforms_to(Optional[Bool], DevicePassable))
    assert_false(conforms_to(Optional[String], DevicePassable))
    assert_false(conforms_to(Optional[MoveOnly[Int]], DevicePassable))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
