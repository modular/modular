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

from collections import LinkedList

from test_utils import (
    CopyCountedStruct,
    CopyCounter,
    DelCounter,
    MoveCounter,
)
from testing import assert_equal, assert_false, assert_raises, assert_true


def test_construction():
    var l1 = LinkedList[Int]()
    assert_equal(len(l1), 0)

    var l2 = LinkedList[Int](1, 2, 3)
    assert_equal(len(l2), 3)
    assert_equal(l2[0], 1)
    assert_equal(l2[1], 2)
    assert_equal(l2[2], 3)


def test_linkedlist_literal():
    var l: LinkedList[Int] = [1, 2, 3]
    assert_equal(3, len(l))
    assert_equal(1, l[0])
    assert_equal(2, l[1])
    assert_equal(3, l[2])

    var l2: LinkedList[Float64] = [1, 2.5]
    assert_equal(2, len(l2))
    assert_equal(1.0, l2[0])
    assert_equal(2.5, l2[1])

    var l3: LinkedList[Int] = []
    assert_equal(0, len(l3))


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
    assert_equal(l1.__str__(), "[1, 2, 3]")


def test_repr():
    var l1 = LinkedList[Int](1, 2, 3)
    assert_equal(l1.__repr__(), "LinkedList(1, 2, 3)")


def test_pop_on_empty_list():
    with assert_raises():
        var ll = LinkedList[Int]()
        _ = ll.pop()


def test_optional_pop_on_empty_linked_list():
    var ll = LinkedList[Int]()
    var result = ll.maybe_pop()
    assert_false(Bool(result))


def test_list():
    var list = LinkedList[Int]()

    for i in range(5):
        list.append(i)

    assert_equal(5, len(list))
    assert_equal(0, list[0])
    assert_equal(1, list[1])
    assert_equal(2, list[2])
    assert_equal(3, list[3])
    assert_equal(4, list[4])

    assert_equal(0, list[-5])
    assert_equal(3, list[-2])
    assert_equal(4, list[-1])

    list[2] = -2
    assert_equal(-2, list[2])

    list[-5] = 5
    assert_equal(5, list[-5])
    list[-2] = 3
    assert_equal(3, list[-2])
    list[-1] = 7
    assert_equal(7, list[-1])


def test_list_clear():
    var list = LinkedList[Int](1, 2, 3)
    assert_equal(len(list), 3)
    list.clear()

    assert_equal(len(list), 0)


def test_list_to_bool_conversion():
    assert_false(LinkedList[String]())
    assert_true(LinkedList[String]("a"))
    assert_true(LinkedList[String]("", "a"))
    assert_true(LinkedList[String](""))


def test_list_pop():
    var list = LinkedList[Int]()
    # Test pop with index
    for i in range(6):
        list.append(i)

    assert_equal(6, len(list))

    # try popping from index 3 for 3 times
    for i in range(3, 6):
        assert_equal(i, list.pop(3))

    # list should have 3 elements now
    assert_equal(3, len(list))
    assert_equal(0, list[0])
    assert_equal(1, list[1])
    assert_equal(2, list[2])

    # Test pop with negative index
    for i in range(0, 2):
        var popped: Int = list.pop(-len(list))
        assert_equal(i, popped)

    # test default index as well
    assert_equal(2, list.pop())
    list.append(2)
    assert_equal(2, list.pop())

    # list should be empty now
    assert_equal(0, len(list))


def test_list_variadic_constructor():
    var l = LinkedList[Int](2, 4, 6)
    assert_equal(3, len(l))
    assert_equal(2, l[0])
    assert_equal(4, l[1])
    assert_equal(6, l[2])

    l.append(8)
    assert_equal(4, len(l))
    assert_equal(8, l[3])

    #
    # Test variadic construct copying behavior
    #

    var l2 = LinkedList[CopyCounter](
        CopyCounter(), CopyCounter(), CopyCounter()
    )

    assert_equal(len(l2), 3)
    assert_equal(l2[0].copy_count, 0)
    assert_equal(l2[1].copy_count, 0)
    assert_equal(l2[2].copy_count, 0)


def test_list_reverse():
    #
    # Test reversing the list []
    #

    var vec = LinkedList[Int]()

    assert_equal(len(vec), 0)

    vec.reverse()

    assert_equal(len(vec), 0)

    #
    # Test reversing the list [123]
    #

    vec = LinkedList[Int]()

    vec.append(123)

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    vec.reverse()

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    #
    # Test reversing the list ["one", "two", "three"]
    #

    var vec2 = LinkedList[String]("one", "two", "three")

    assert_equal(len(vec2), 3)
    assert_equal(vec2[0], "one")
    assert_equal(vec2[1], "two")
    assert_equal(vec2[2], "three")

    vec2.reverse()

    assert_equal(len(vec2), 3)
    assert_equal(vec2[0], "three")
    assert_equal(vec2[1], "two")
    assert_equal(vec2[2], "one")

    #
    # Test reversing the list [5, 10]
    #

    vec = LinkedList[Int]()
    vec.append(5)
    vec.append(10)

    assert_equal(len(vec), 2)
    assert_equal(vec[0], 5)
    assert_equal(vec[1], 10)

    vec.reverse()

    assert_equal(len(vec), 2)
    assert_equal(vec[0], 10)
    assert_equal(vec[1], 5)


def test_list_insert():
    #
    # Test the list [1, 2, 3] created with insert
    #

    var v1 = LinkedList[Int]()
    v1.insert(len(v1), 1)
    v1.insert(len(v1), 3)
    v1.insert(1, 2)

    assert_equal(len(v1), 3)
    assert_equal(v1[0], 1)
    assert_equal(v1[1], 2)
    assert_equal(v1[2], 3)

    #
    # Test the list [1, 2, 3, 4, 5] created with negative and positive index
    #

    var v2 = LinkedList[Int]()
    v2.insert(-1729, 2)
    v2.insert(len(v2), 3)
    v2.insert(len(v2), 5)
    v2.insert(-1, 4)
    v2.insert(-len(v2), 1)

    assert_equal(len(v2), 5)
    assert_equal(v2[0], 1)
    assert_equal(v2[1], 2)
    assert_equal(v2[2], 3)
    assert_equal(v2[3], 4)
    assert_equal(v2[4], 5)

    #
    # Test the list [1, 2, 3, 4] created with negative index
    #

    var v3 = LinkedList[Int]()
    v3.insert(-11, 4)
    v3.insert(-13, 3)
    v3.insert(-17, 2)
    v3.insert(-19, 1)

    assert_equal(len(v3), 4)
    assert_equal(v3[0], 1)
    assert_equal(v3[1], 2)
    assert_equal(v3[2], 3)
    assert_equal(v3[3], 4)

    #
    # Test the list [1, 2, 3, 4, 5, 6, 7, 8] created with insert
    #

    var v4 = LinkedList[Int]()
    for i in range(4):
        v4.insert(0, 4 - i)
        v4.insert(len(v4), 4 + i + 1)

    for i in range(len(v4)):
        assert_equal(v4[i], i + 1)


def test_list_extend_non_trivial():
    # Tests three things:
    #   - extend() for non-plain-old-data types
    #   - extend() with mixed-length self and other lists
    #   - extend() using optimal number of __moveinit__() calls
    var v1 = LinkedList[MoveCounter[String]]()
    v1.append(MoveCounter[String]("Hello"))
    v1.append(MoveCounter[String]("World"))

    var v2 = LinkedList[MoveCounter[String]]()
    v2.append(MoveCounter[String]("Foo"))
    v2.append(MoveCounter[String]("Bar"))
    v2.append(MoveCounter[String]("Baz"))

    v1.extend(v2^)

    assert_equal(len(v1), 5)
    assert_equal(v1[0].value, "Hello")
    assert_equal(v1[1].value, "World")
    assert_equal(v1[2].value, "Foo")
    assert_equal(v1[3].value, "Bar")
    assert_equal(v1[4].value, "Baz")

    assert_equal(v1[0].move_count, 1)
    assert_equal(v1[1].move_count, 1)
    assert_equal(v1[2].move_count, 1)
    assert_equal(v1[3].move_count, 1)
    assert_equal(v1[4].move_count, 1)


def test_2d_dynamic_list():
    var list = LinkedList[LinkedList[Int]]()

    for i in range(2):
        var v = LinkedList[Int]()
        for j in range(3):
            v.append(i + j)
        list.append(v)

    assert_equal(0, list[0][0])
    assert_equal(1, list[0][1])
    assert_equal(2, list[0][2])
    assert_equal(1, list[1][0])
    assert_equal(2, list[1][1])
    assert_equal(3, list[1][2])

    assert_equal(2, len(list))

    assert_equal(3, len(list[0]))

    list[0].clear()
    assert_equal(0, len(list[0]))

    list.clear()
    assert_equal(0, len(list))


def test_list_explicit_copy():
    var list = LinkedList[CopyCounter]()
    list.append(CopyCounter())
    var list_copy = list.copy()
    assert_equal(0, list[0].copy_count)
    assert_equal(1, list_copy[0].copy_count)

    var l2 = LinkedList[Int]()
    for i in range(10):
        l2.append(i)

    var l2_copy = l2.copy()
    assert_equal(len(l2), len(l2_copy))
    for i in range(len(l2)):
        assert_equal(l2[i], l2_copy[i])


def test_no_extra_copies_with_sugared_set_by_field():
    var list = LinkedList[LinkedList[CopyCountedStruct]]()
    var child_list = LinkedList[CopyCountedStruct]()
    child_list.append(CopyCountedStruct("Hello"))
    child_list.append(CopyCountedStruct("World"))

    # No copies here.  Constructing with LinkedList[CopyCountedStruct](CopyCountedStruct("Hello")) is a copy.
    assert_equal(0, child_list[0].counter.copy_count)
    assert_equal(0, child_list[1].counter.copy_count)

    list.append(child_list^)

    assert_equal(0, list[0][0].counter.copy_count)
    assert_equal(0, list[0][1].counter.copy_count)

    # list[0][1] makes a copy for reasons I cannot determine
    list.__getitem__(0).__getitem__(1).value = "Mojo"

    assert_equal(0, list[0][0].counter.copy_count)
    assert_equal(0, list[0][1].counter.copy_count)

    assert_equal("Mojo", list[0][1].value)

    assert_equal(0, list[0][0].counter.copy_count)
    assert_equal(0, list[0][1].counter.copy_count)


def test_list_boolable():
    assert_true(LinkedList[Int](1))
    assert_false(LinkedList[Int]())


def test_list_count():
    var list = LinkedList[Int](1, 2, 3, 2, 5, 6, 7, 8, 9, 10)
    assert_equal(1, list.count(1))
    assert_equal(2, list.count(2))
    assert_equal(0, list.count(4))

    var list2 = LinkedList[Int]()
    assert_equal(0, list2.count(1))


def test_list_contains():
    var x = LinkedList[Int](1, 2, 3)
    assert_false(0 in x)
    assert_true(1 in x)
    assert_false(4 in x)

    # TODO: implement LinkedList.__eq__ for Self[Copyable & Movable & Comparable]
    # var y = LinkedList[LinkedList[Int]]()
    # y.append(LinkedList(1,2))
    # assert_equal(LinkedList(1,2) in y,True)
    # assert_equal(LinkedList(0,1) in y,False)


def test_list_eq_ne():
    var l1 = LinkedList[Int](1, 2, 3)
    var l2 = LinkedList[Int](1, 2, 3)
    assert_true(l1 == l2)
    assert_false(l1 != l2)

    var l3 = LinkedList[Int](1, 2, 3, 4)
    assert_false(l1 == l3)
    assert_true(l1 != l3)

    var l4 = LinkedList[Int]()
    var l5 = LinkedList[Int]()
    assert_true(l4 == l5)
    assert_true(l1 != l4)

    var l6 = LinkedList[String]("a", "b", "c")
    var l7 = LinkedList[String]("a", "b", "c")
    var l8 = LinkedList[String]("a", "b")
    assert_true(l6 == l7)
    assert_false(l6 != l7)
    assert_false(l6 == l8)


def test_indexing():
    var l = LinkedList[Int](1, 2, 3)
    assert_equal(l[Int(1)], 2)
    assert_equal(l[False], 1)
    assert_equal(l[True], 2)
    assert_equal(l[2], 3)


# ===-------------------------------------------------------------------===#
# LinkedList dtor tests
# ===-------------------------------------------------------------------===#


def test_list_dtor():
    var dtor_count = 0

    var l = LinkedList[DelCounter]()
    assert_equal(dtor_count, 0)

    l.append(DelCounter(UnsafePointer(to=dtor_count)))
    assert_equal(dtor_count, 0)

    l^.__del__()
    assert_equal(dtor_count, 1)


def test_iter():
    var l = LinkedList[Int](1, 2, 3)
    var iter = l.__iter__()
    assert_true(iter.__has_next__(), "Expected iter to have next")
    assert_equal(iter.__next_ref__(), 1)
    assert_equal(iter.__next_ref__(), 2)
    assert_equal(iter.__next_ref__(), 3)
    assert_false(iter.__has_next__(), "Expected iter to not have next")

    var riter = l.__reversed__()
    assert_true(riter.__has_next__(), "Expected iter to have next")
    assert_equal(riter.__next_ref__(), 3)
    assert_equal(riter.__next_ref__(), 2)
    assert_equal(riter.__next_ref__(), 1)
    assert_false(riter.__has_next__(), "Expected iter to not have next")

    var i = 0
    for el in l:
        assert_equal(el, l[i])
        i += 1

    i = 2
    for el in l.__reversed__():
        assert_equal(el, l[i])
        i -= 1


def test_repr_wrap():
    var l1 = LinkedList[Int](1, 2, 3)
    assert_equal(repr(l1), "LinkedList(1, 2, 3)")


def main():
    test_construction()
    test_linkedlist_literal()
    test_append()
    test_prepend()
    test_copy()
    test_reverse()
    test_pop()
    test_getitem()
    test_setitem()
    test_str()
    test_repr()
    test_pop_on_empty_list()
    test_optional_pop_on_empty_linked_list()
    test_list()
    test_list_clear()
    test_list_to_bool_conversion()
    test_list_pop()
    test_list_variadic_constructor()
    test_list_reverse()
    test_list_extend_non_trivial()
    test_list_explicit_copy()
    test_no_extra_copies_with_sugared_set_by_field()
    test_2d_dynamic_list()
    test_list_boolable()
    test_list_count()
    test_list_contains()
    test_indexing()
    test_list_dtor()
    test_list_insert()
    test_list_eq_ne()
    test_iter()
    test_repr_wrap()
