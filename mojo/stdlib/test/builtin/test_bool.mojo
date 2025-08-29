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

from python import PythonObject
from testing import assert_equal, assert_false, assert_true


def test_default():
    assert_equal(Bool(), False)


def test_min_max():
    assert_equal(Bool.MIN, False)
    assert_equal(Bool.MAX, True)


def test_bool_cast_to_int():
    assert_equal(False.__int__(), 0)
    assert_equal(True.__int__(), 1)

    assert_equal(Int(False), 0)
    assert_equal(Int(True), 1)


def test_bool_none():
    var test = None
    assert_equal(Bool(None), False)
    assert_equal(Bool(test), False)


@fieldwise_init
struct MyTrue(ImplicitlyBoolable):
    fn __bool__(self) -> Bool:
        return True

    fn __as_bool__(self) -> Bool:
        return self.__bool__()


fn takes_bool(cond: Bool) -> Bool:
    return cond


def test_convert_from_implicitly_boolable():
    assert_true(takes_bool(MyTrue()))
    assert_true(Bool(MyTrue()))


def test_bool_representation():
    assert_equal(repr(True), "True")
    assert_equal(repr(False), "False")


def test_bitwise():
    var value: Bool

    # and
    value = False
    value &= False
    assert_false(value)
    value = False
    value &= True
    assert_false(value)
    value = True
    value &= False
    assert_false(value)
    value = True
    value &= True
    assert_true(value)

    # or
    value = False
    value |= False
    assert_false(value)
    value = False
    value |= True
    assert_true(value)
    value = True
    value |= False
    assert_true(value)
    value = True
    value |= True
    assert_true(value)

    # xor
    value = False
    value ^= False
    assert_false(value)
    value = False
    value ^= True
    assert_true(value)
    value = True
    value ^= False
    assert_true(value)
    value = True
    value ^= True
    assert_false(value)


def test_indexer():
    assert_true(1 == index(Bool(True)))
    assert_true(0 == index(Bool(False)))


def test_comparisons():
    assert_true(False == False)
    assert_true(True == True)
    assert_false(False == True)
    assert_false(True == False)

    assert_true(False != True)
    assert_true(True != False)
    assert_false(False != False)
    assert_false(True != True)

    assert_true(True > False)
    assert_false(False > True)
    assert_false(False > False)
    assert_false(True > True)

    assert_true(True >= False)
    assert_false(False >= True)
    assert_true(False >= False)
    assert_true(True >= True)

    assert_false(True < False)
    assert_true(False < True)
    assert_false(False < False)
    assert_false(True < True)

    assert_false(True <= False)
    assert_true(False <= True)
    assert_true(False <= False)
    assert_true(True <= True)


def test_float_conversion():
    assert_equal((True).__float__(), 1.0)
    assert_equal((False).__float__(), 0.0)


def main():
    test_default()
    test_min_max()
    test_bool_cast_to_int()
    test_bool_none()
    test_convert_from_implicitly_boolable()
    test_bool_representation()
    test_bitwise()
    test_indexer()
    test_comparisons()
    test_float_conversion()
