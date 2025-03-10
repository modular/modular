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

from sys import os_is_macos

from memory import UnsafePointer
from testing import assert_false, assert_true, assert_equal
from test_utils import DelRecorder


def test_tuple_contains():
    var a = (123, True, "Mojo is awesome")

    assert_true("Mojo is awesome" in a)
    assert_true(a.__contains__("Mojo is awesome"))

    assert_false("Hello world" in a)
    assert_false(a.__contains__("Hello world"))

    assert_true(123 in a)
    assert_true(a.__contains__(123))

    assert_true(True in a)
    assert_true(a.__contains__(True))

    assert_false(False in a)
    assert_false(a.__contains__(False))

    assert_false(a.__contains__(1))
    assert_false(a.__contains__(0))
    assert_false(1 in a)
    assert_false(0 in a)

    var b = (False, True)
    assert_true(True in b)
    assert_true(b.__contains__(True))
    assert_true(False in b)
    assert_true(b.__contains__(False))
    assert_false(b.__contains__(1))
    assert_false(b.__contains__(0))

    var c = (1, 0)
    assert_false(c.__contains__(True))
    assert_false(c.__contains__(False))
    assert_false(True in c)
    assert_false(False in c)

    var d = (123, True, String("Mojo is awesome"))

    assert_true(String("Mojo is awesome") in d)
    assert_false("Mojo is awesome" in d)
    assert_true(d.__contains__(String("Mojo is awesome")))

    assert_false(String("Hello world") in d)
    assert_false(d.__contains__(String("Hello world")))

    alias a_alias = (123, True, "Mojo is awesome")

    assert_true("Mojo is awesome" in a_alias)
    assert_true(a_alias.__contains__("Mojo is awesome"))

    assert_false("Hello world" in a_alias)
    assert_false(a_alias.__contains__("Hello world"))

    assert_true(123 in a_alias)
    assert_true(a_alias.__contains__(123))

    assert_true(True in a_alias)
    assert_true(a_alias.__contains__(True))

    assert_false(False in a_alias)
    assert_false(a_alias.__contains__(False))

    assert_false(a_alias.__contains__(1))
    assert_false(a_alias.__contains__(0))
    assert_false(1 in a_alias)
    assert_false(0 in a_alias)

    alias b_alias = (False, True)
    assert_true(True in b_alias)
    assert_true(b_alias.__contains__(True))
    assert_true(False in b_alias)
    assert_true(b_alias.__contains__(False))
    assert_false(b_alias.__contains__(1))
    assert_false(b_alias.__contains__(0))

    alias c_alias = (1, 0)
    assert_false(c_alias.__contains__(True))
    assert_false(c_alias.__contains__(False))
    assert_false(True in c_alias)
    assert_false(False in c_alias)

    alias d_alias = (123, True, String("Mojo is awesome"))
    # Ensure `contains` itself works in comp-time domain
    alias ok = 123 in d_alias
    assert_true(ok)

    assert_true(String("Mojo is awesome") in d_alias)
    assert_false("Mojo is awesome" in d_alias)
    assert_true(d_alias.__contains__(String("Mojo is awesome")))

    assert_false(String("Hello world") in d_alias)
    assert_false(d_alias.__contains__(String("Hello world")))


def test_tuple_init():
    var destructor_counter = List[Int]()
    var pointer_to_destructor_counter = UnsafePointer.address_of(
        destructor_counter
    )
    var a = DelRecorder(1, pointer_to_destructor_counter)
    var b = DelRecorder(2, pointer_to_destructor_counter)
    var t = (a^, b^)
    assert_equal(t[0].value, 1)
    assert_equal(t[1].value, 2)
    __type_of(t).__del__(t^)
    assert_equal(len(destructor_counter), 2)
    assert_equal(destructor_counter[0], 1)
    assert_equal(destructor_counter[1], 2)


def test_tuple_del():
    var destructor_counter = List[Int]()
    var pointer_to_destructor_counter = UnsafePointer.address_of(
        destructor_counter
    )
    var t = (
        DelRecorder(1, pointer_to_destructor_counter),
        DelRecorder(2, pointer_to_destructor_counter),
    )
    assert_equal(t[0].value, 1)
    assert_equal(t[1].value, 2)
    __type_of(t).__del__(t^)
    assert_equal(len(destructor_counter), 2)
    assert_equal(destructor_counter[0], 1)
    assert_equal(destructor_counter[1], 2)


def test_tuple_assigned_into_multiple_variables():
    var destructor_counter = List[Int]()
    var pointer_to_destructor_counter = UnsafePointer.address_of(
        destructor_counter
    )
    var t = (
        DelRecorder(1, pointer_to_destructor_counter),
        DelRecorder(2, pointer_to_destructor_counter),
    )
    a, b = t^
    assert_equal(a.value, 1)
    assert_equal(b.value, 2)
    assert_equal(len(destructor_counter), 2)  # FIXME (result: 4)


def main():
    test_tuple_contains()
    test_tuple_init()
    test_tuple_del()
    # test_tuple_assigned_into_multiple_variables() #FIXME
