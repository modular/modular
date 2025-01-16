# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo-no-debug %s

from collections import LinkedList
from testing import assert_equal


def test_construction():
    var l1 = LinkedList[Int]()
    assert_equal(len(l1), 0)

    var l2 = LinkedList[Int](1, 2, 3)
    assert_equal(len(l2), 3)
    assert_equal(l2[0], 1)
    assert_equal(l2[1], 2)
    assert_equal(l2[2], 3)


def test_append():
    var l1 = LinkedList[Int]()
    l1.append(1)
    l1.append(2)
    l1.append(3)
    assert_equal(len(l1), 3)
    assert_equal(l1[0], 1)
    assert_equal(l1[1], 2)
    assert_equal(l1[2], 3)


def test_prepend():
    var l1 = LinkedList[Int]()
    l1.prepend(1)
    l1.prepend(2)
    l1.prepend(3)
    assert_equal(len(l1), 3)
    assert_equal(l1[0], 3)
    assert_equal(l1[1], 2)
    assert_equal(l1[2], 1)


def test_copy():
    var l1 = LinkedList[Int](1, 2, 3)
    var l2 = l1.copy()
    assert_equal(len(l2), 3)
    assert_equal(l2[0], 1)
    assert_equal(l2[1], 2)
    assert_equal(l2[2], 3)


def test_reverse():
    var l1 = LinkedList[Int](1, 2, 3)
    l1.reverse()
    assert_equal(len(l1), 3)
    assert_equal(l1[0], 3)
    assert_equal(l1[1], 2)
    assert_equal(l1[2], 1)


def test_pop():
    var l1 = LinkedList[Int](1, 2, 3)
    assert_equal(l1.pop(), 3)
    assert_equal(len(l1), 2)
    assert_equal(l1[0], 1)
    assert_equal(l1[1], 2)


def test_getitem():
    var l1 = LinkedList[Int](1, 2, 3)
    assert_equal(l1[0], 1)
    assert_equal(l1[1], 2)
    assert_equal(l1[2], 3)

    assert_equal(l1[-1], 3)
    assert_equal(l1[-2], 2)
    assert_equal(l1[-3], 1)


def test_setitem():
    var l1 = LinkedList[Int](1, 2, 3)
    l1[0] = 4
    assert_equal(l1[0], 4)
    assert_equal(l1[1], 2)
    assert_equal(l1[2], 3)

    l1[-1] = 5
    assert_equal(l1[0], 4)
    assert_equal(l1[1], 2)
    assert_equal(l1[2], 5)


def test_str():
    var l1 = LinkedList[Int](1, 2, 3)
    assert_equal(String(l1), "[1, 2, 3]")


def test_repr():
    var l1 = LinkedList[Int](1, 2, 3)
    assert_equal(repr(l1), "LinkedList(1, 2, 3)")


def main():
    test_construction()
    test_append()
    test_prepend()
    test_copy()
    test_reverse()
    test_pop()
    test_getitem()
    test_setitem()
    test_str()
    test_repr()
