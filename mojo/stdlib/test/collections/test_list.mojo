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

from sys.info import size_of

from test_utils import (
    CopyCountedStruct,
    CopyCounter,
    DelCounter,
    MoveCounter,
    TriviallyCopyableMoveCounter,
)
from testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
    TestSuite,
)


def test_mojo_issue_698():
    var list = List[Float64]()
    for i in range(5):
        list.append(i)

    assert_equal(0.0, list[0])
    assert_equal(1.0, list[1])
    assert_equal(2.0, list[2])
    assert_equal(3.0, list[3])
    assert_equal(4.0, list[4])


def test_list():
    var list = List[Int]()

    for i in range(5):
        list.append(i)

    assert_equal(5, len(list))
    assert_equal(5 * size_of[Int](), list.byte_length())
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


struct WeirdList[T: AnyType]:
    fn __init__(out self, var *values: T, __list_literal__: ()):
        pass


fn take_generic_weird_list(list: WeirdList[_]):
    pass


def test_list_literal():
    var list: List[Int] = [1, 2, 3]
    assert_equal(3, len(list))
    assert_equal(1, list[0])
    assert_equal(2, list[1])
    assert_equal(3, list[2])

    var list2 = [1, 2.5]
    assert_equal(2, len(list2))
    assert_equal(1.0, list2[0])
    assert_equal(2.5, list2[1])

    # Test parameter inference of the T element type.
    take_generic_weird_list([1.0, 2.0])

    # Heterogenous lists
    # take_generic_weird_list([1.0, 2])
    # take_generic_weird_list([1, 2.0])


def test_list_unsafe_get():
    var list = List[Int]()

    for i in range(5):
        list.append(i)

    assert_equal(5, len(list))
    assert_equal(0, list.unsafe_get(0))
    assert_equal(1, list.unsafe_get(1))
    assert_equal(2, list.unsafe_get(2))
    assert_equal(3, list.unsafe_get(3))
    assert_equal(4, list.unsafe_get(4))

    list[2] = -2
    assert_equal(-2, list.unsafe_get(2))

    list.clear()
    list.append(2)
    assert_equal(2, list.unsafe_get(0))


def test_list_unsafe_set():
    var list = List[Int]()

    for i in range(5):
        list.append(i)

    assert_equal(5, len(list))
    list.unsafe_set(0, 0)
    list.unsafe_set(1, 10)
    list.unsafe_set(2, 20)
    list.unsafe_set(3, 30)
    list.unsafe_set(4, 40)

    assert_equal(list[0], 0)
    assert_equal(list[1], 10)
    assert_equal(list[2], 20)
    assert_equal(list[3], 30)
    assert_equal(list[4], 40)


def test_list_clear():
    var list = [1, 2, 3]
    assert_equal(len(list), 3)
    assert_equal(list.capacity, 3)
    list.clear()

    assert_equal(len(list), 0)
    assert_equal(list.capacity, 3)


def test_list_to_bool_conversion():
    assert_false(List[String]())
    assert_true(List[String]("a"))
    assert_true(List[String]("", "a"))
    assert_true(List[String](""))


def test_list_pop():
    var list = List[Int]()
    # Test pop with index
    for i in range(6):
        list.append(i)

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
        assert_equal(i, list.pop(-len(list)))

    # test default index as well
    assert_equal(2, list.pop())
    list.append(2)
    assert_equal(2, list.pop())

    # list should be empty now
    assert_equal(0, len(list))


def test_list_variadic_constructor():
    var l = [2, 4, 6]
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

    var l2 = List[CopyCounter](CopyCounter(), CopyCounter(), CopyCounter())

    assert_equal(len(l2), 3)
    assert_equal(l2[0].copy_count, 0)
    assert_equal(l2[1].copy_count, 0)
    assert_equal(l2[2].copy_count, 0)


def test_list_resize():
    var l = List[Int](1)
    assert_equal(1, len(l))
    l.resize(2, 0)
    assert_equal(2, len(l))
    assert_equal(l[1], 0)
    l.shrink(0)
    assert_equal(len(l), 0)


def test_list_reverse():
    #
    # Test reversing the list []
    #

    var vec = List[Int]()

    assert_equal(len(vec), 0)

    vec.reverse()

    assert_equal(len(vec), 0)

    #
    # Test reversing the list [123]
    #

    vec = []

    vec.append(123)

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    vec.reverse()

    assert_equal(len(vec), 1)
    assert_equal(vec[0], 123)

    #
    # Test reversing the list ["one", "two", "three"]
    #

    vec2 = ["one", "two", "three"]

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

    vec = []
    vec.append(5)
    vec.append(10)

    assert_equal(len(vec), 2)
    assert_equal(vec[0], 5)
    assert_equal(vec[1], 10)

    vec.reverse()

    assert_equal(len(vec), 2)
    assert_equal(vec[0], 10)
    assert_equal(vec[1], 5)


def test_list_reverse_move_count():
    # Create this vec with enough capacity to avoid moves due to resizing.
    var vec = List[MoveCounter[Int]](capacity=5)
    vec.append(MoveCounter(1))
    vec.append(MoveCounter(2))
    vec.append(MoveCounter(3))
    vec.append(MoveCounter(4))
    vec.append(MoveCounter(5))

    assert_equal(len(vec), 5)
    assert_equal(vec[0].value, 1)
    assert_equal(vec[1].value, 2)
    assert_equal(vec[2].value, 3)
    assert_equal(vec[3].value, 4)
    assert_equal(vec[4].value, 5)

    assert_equal(vec[0].move_count, 1)
    assert_equal(vec[1].move_count, 1)
    assert_equal(vec[2].move_count, 1)
    assert_equal(vec[3].move_count, 1)
    assert_equal(vec[4].move_count, 1)

    vec.reverse()

    assert_equal(len(vec), 5)
    assert_equal(vec[0].value, 5)
    assert_equal(vec[1].value, 4)
    assert_equal(vec[2].value, 3)
    assert_equal(vec[3].value, 2)
    assert_equal(vec[4].value, 1)

    # NOTE:
    # Earlier elements went through 2 moves and later elements went through 3
    # moves because the implementation of List.reverse arbitrarily
    # chooses to perform the swap of earlier and later elements by moving the
    # earlier element to a temporary (+1 move), directly move the later element
    # into the position the earlier element was in, and then move from the
    # temporary into the later position (+1 move).
    assert_equal(vec[0].move_count, 2)
    assert_equal(vec[1].move_count, 2)
    assert_equal(vec[2].move_count, 1)
    assert_equal(vec[3].move_count, 3)
    assert_equal(vec[4].move_count, 3)


def test_list_insert():
    #
    # Test the list [1, 2, 3] created with insert
    #

    v1 = List[Int]()
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

    v2 = List[Int]()
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

    v3 = List[Int]()
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

    v4 = List[Int]()
    for i in range(4):
        v4.insert(0, 4 - i)
        v4.insert(len(v4), 4 + i + 1)

    for i, value in enumerate(v4):
        assert_equal(value, i + 1)


def test_list_index():
    var test_list_a = [10, 20, 30, 40, 50]

    # Basic Functionality Tests
    assert_equal(test_list_a.index(10), 0)
    assert_equal(test_list_a.index(30), 2)
    assert_equal(test_list_a.index(50), 4)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(60)

    # Tests With Start Parameter
    assert_equal(test_list_a.index(30, start=1), 2)
    assert_equal(test_list_a.index(30, start=-4), 2)
    assert_equal(test_list_a.index(30, start=-1000), 2)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(30, start=3)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(30, start=5)

    # Tests With Start and End Parameters
    assert_equal(test_list_a.index(30, start=1, stop=3), 2)
    assert_equal(test_list_a.index(30, start=-4, stop=-2), 2)
    assert_equal(test_list_a.index(30, start=-1000, stop=1000), 2)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(30, start=1, stop=2)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(30, start=3, stop=1)

    # Tests With End Parameter Only
    assert_equal(test_list_a.index(30, stop=3), 2)
    assert_equal(test_list_a.index(30, stop=-2), 2)
    assert_equal(test_list_a.index(30, stop=1000), 2)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(30, stop=1)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(30, stop=2)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(60, stop=50)

    # Edge Cases and Special Conditions
    assert_equal(test_list_a.index(10, start=-5, stop=-1), 0)
    assert_equal(test_list_a.index(10, start=0, stop=50), 0)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(50, start=-5, stop=-1)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(50, start=0, stop=-1)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(10, start=-4, stop=-1)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(10, start=5, stop=50)
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = List[Int]().index(10)

    # Test empty slice
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(10, start=1, stop=1)
    # Test empty slice with 0 start and end
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_a.index(10, start=0, stop=0)

    var test_list_b = [10, 20, 30, 20, 10]

    # Test finding the first occurrence of an item
    assert_equal(test_list_b.index(10), 0)
    assert_equal(test_list_b.index(20), 1)

    # Test skipping the first occurrence with a start parameter
    assert_equal(test_list_b.index(20, start=2), 3)

    # Test constraining search with start and end, excluding last occurrence
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_b.index(10, start=1, stop=4)

    # Test search within a range that includes multiple occurrences
    assert_equal(test_list_b.index(20, start=1, stop=4), 1)

    # Verify error when constrained range excludes occurrences
    with assert_raises(contains="ValueError: Given element is not in list"):
        _ = test_list_b.index(20, start=4, stop=5)


def test_list_append():
    var items = List[UInt32]()
    items.append(1)
    items.append(2)
    items.append(3)
    assert_equal(items, List[UInt32](1, 2, 3))


def test_list_extend():
    var items = List[UInt32](1, 2, 3)
    var copy = items.copy()
    items.extend(copy^)
    assert_equal(items, List[UInt32](1, 2, 3, 1, 2, 3))

    items = [1, 2, 3]
    copy = [1, 2, 3]

    # Extend with span
    items.extend(Span(copy))
    assert_equal(items, List[UInt32](1, 2, 3, 1, 2, 3))

    # Extend with whole SIMD
    items = List[UInt32](1, 2, 3)
    items.extend(SIMD[DType.uint32, 4](1, 2, 3, 4))
    assert_equal(items, List[UInt32](1, 2, 3, 1, 2, 3, 4))
    # Extend with part of SIMD
    items = List[UInt32](1, 2, 3)
    items.extend(SIMD[DType.uint32, 4](1, 2, 3, 4), count=3)
    assert_equal(items, List[UInt32](1, 2, 3, 1, 2, 3))


def test_list_extend_non_trivial():
    # Tests three things:
    #   - extend() for non-plain-old-data types
    #   - extend() with mixed-length self and other lists
    #   - extend() using optimal number of __moveinit__() calls

    # Preallocate with enough capacity to avoid reallocation making the
    # move count checks below flaky.
    var v1 = List[MoveCounter[String]](capacity=5)
    v1.append(MoveCounter[String]("Hello"))
    v1.append(MoveCounter[String]("World"))

    var v2 = List[MoveCounter[String]](capacity=3)
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
    assert_equal(v1[2].move_count, 2)
    assert_equal(v1[3].move_count, 2)
    assert_equal(v1[4].move_count, 2)


def test_list_extend_trivial_copy_nontrivial_move():
    var v1 = List[TriviallyCopyableMoveCounter](capacity=1)
    var v2 = List(TriviallyCopyableMoveCounter(0))

    assert_equal(v2[0].move_count, 1)

    v1.extend(v2^)

    # `extend()` should call __moveinit__, not perform even a trivially copy.
    assert_equal(v1[0].move_count, 2)


def test_2d_dynamic_list():
    var list = List[List[Int]]()

    for i in range(2):
        var v = List[Int]()
        for j in range(3):
            v.append(i + j)
        list.append(v^)

    assert_equal(0, list[0][0])
    assert_equal(1, list[0][1])
    assert_equal(2, list[0][2])
    assert_equal(1, list[1][0])
    assert_equal(2, list[1][1])
    assert_equal(3, list[1][2])

    assert_equal(2, len(list))
    assert_equal(2, list.capacity)

    assert_equal(3, len(list[0]))

    list[0].clear()
    assert_equal(0, len(list[0]))
    assert_equal(4, list[0].capacity)

    list.clear()
    assert_equal(0, len(list))
    assert_equal(2, list.capacity)


def test_list_explicit_copy():
    var list = List[CopyCounter]()
    list.append(CopyCounter())
    var list_copy = list.copy()
    assert_equal(0, list[0].copy_count)
    assert_equal(1, list_copy[0].copy_count)

    var l2 = List[Int]()
    for i in range(10):
        l2.append(i)

    var l2_copy = l2.copy()
    assert_equal(len(l2), len(l2_copy))
    for val1, val2 in zip(l2, l2_copy):
        assert_equal(val1, val2)


def test_no_extra_copies_with_sugared_set_by_field():
    var list = List[List[CopyCountedStruct]](capacity=1)
    var child_list = List[CopyCountedStruct](capacity=2)
    child_list.append(CopyCountedStruct("Hello"))
    child_list.append(CopyCountedStruct("World"))

    # No copies here.  Constructing with List[CopyCountedStruct](CopyCountedStruct("Hello")) is a copy.
    assert_equal(0, child_list[0].counter.copy_count)
    assert_equal(0, child_list[1].counter.copy_count)
    list.append(child_list^)

    list[0][1].value = "Mojo"
    assert_equal("Mojo", list[0][1].value)

    assert_equal(0, list[0][0].counter.copy_count)
    assert_equal(0, list[0][1].counter.copy_count)


# Ensure correct behavior of __copyinit__
# as reported in GH issue 27875 internally and
# https://github.com/modular/modular/issues/1493
def test_list_copy_constructor():
    var vec = List[Int](capacity=1)
    var vec_copy = vec.copy()
    vec_copy.append(1)  # Ensure copy constructor doesn't crash
    _ = vec^  # To ensure previous one doesn't invoke move constructor


def test_list_iter():
    var vs = List[Int]()
    vs.append(1)
    vs.append(2)
    vs.append(3)

    # Borrow immutably
    fn sum(vs: List[Int]) -> Int:
        var sum = 0
        for v in vs:
            sum += v
        return sum

    assert_equal(6, sum(vs))


def test_list_iter_mutable():
    var vs = [1, 2, 3]

    for ref v in vs:
        v += 1

    var sum = 0
    for v in vs:
        sum += v

    assert_equal(9, sum)


def _test_list_iter_bounds[I: Iterator](var list_iter: I, list_len: Int):
    var iter = list_iter^

    for i in range(list_len):
        var lower, upper = iter.bounds()
        assert_equal(list_len - i, lower)
        assert_equal(list_len - i, upper.value())
        _ = iter.__next__()

    var lower, upper = iter.bounds()
    assert_equal(0, lower)
    assert_equal(0, upper.value())


def test_list_iter_bounds():
    var list = [1, 2, 3]
    _test_list_iter_bounds(iter(list), len(list))
    _test_list_iter_bounds(reversed(list), len(list))


def test_list_span():
    var vs = [1, 2, 3]

    var es = vs[1:]
    assert_equal(es[0], 2)
    assert_equal(es[1], 3)
    assert_equal(len(es), 2)

    es = vs[:-1]
    assert_equal(es[0], 1)
    assert_equal(es[1], 2)
    assert_equal(len(es), 2)

    es = vs[1:-1:1]
    assert_equal(es[0], 2)
    assert_equal(len(es), 1)

    es = vs[::-1]
    assert_equal(es[0], 3)
    assert_equal(es[1], 2)
    assert_equal(es[2], 1)
    assert_equal(len(es), 3)

    es = vs[:]
    assert_equal(es[0], 1)
    assert_equal(es[1], 2)
    assert_equal(es[2], 3)
    assert_equal(len(es), 3)

    assert_equal(vs[1:0:-1][0], 2)
    assert_equal(vs[2:1:-1][0], 3)
    es = vs[:0:-1]
    assert_equal(es[0], 3)
    assert_equal(es[1], 2)
    assert_equal(vs[2::-1][0], 3)

    assert_equal(len(vs[1:2:-1]), 0)

    assert_equal(0, len(vs[:-1:-2]))
    assert_equal(0, len(vs[-50::-1]))
    es = vs[-50::]
    assert_equal(3, len(es))
    assert_equal(es[0], 1)
    assert_equal(es[1], 2)
    assert_equal(es[2], 3)
    es = vs[:-50:-1]
    assert_equal(3, len(es))
    assert_equal(es[0], 3)
    assert_equal(es[1], 2)
    assert_equal(es[2], 1)
    es = vs[:50:]
    assert_equal(3, len(es))
    assert_equal(es[0], 1)
    assert_equal(es[1], 2)
    assert_equal(es[2], 3)
    es = vs[::50]
    assert_equal(1, len(es))
    assert_equal(es[0], 1)
    es = vs[::-50]
    assert_equal(1, len(es))
    assert_equal(es[0], 3)
    es = vs[50::-50]
    assert_equal(1, len(es))
    assert_equal(es[0], 3)
    es = vs[-50::50]
    assert_equal(1, len(es))
    assert_equal(es[0], 1)


def test_list_realloc_trivial_types():
    a = List[Int]()
    for i in range(100):
        a.append(i)

    assert_equal(len(a), 100)
    for i in range(100):
        assert_equal(a[i], i)

    b = List[Int8]()
    for i in range(100):
        b.append(Int8(i))

    assert_equal(len(b), 100)
    for i in range(100):
        assert_equal(b[i], Int8(i))


def test_list_realloc_trivial_copy_nontrivial_move():
    var lst = List[TriviallyCopyableMoveCounter](capacity=1)

    lst.append(TriviallyCopyableMoveCounter(0))
    assert_equal(lst[0].move_count, 1)

    lst.reserve(10)

    # Reallocating the list should call __moveinit__(), not perform a copy.
    assert_equal(lst[0].move_count, 2)


def test_list_boolable():
    assert_true(List[Int](1))
    assert_false(List[Int]())


def test_converting_list_to_string():
    # This is also testing the method `to_format` because
    # essentially, `List.__str__()` just creates a String and applies `to_format` to it.
    # If we were to write unit tests for `to_format`, we would essentially copy-paste the code
    # of `List.__str__()`
    var my_list = [1, 2, 3]
    assert_equal(my_list.__str__(), "[1, 2, 3]")

    var my_list4 = ["a", "b", "c", "foo"]
    assert_equal(my_list4.__str__(), "['a', 'b', 'c', 'foo']")


def test_list_count():
    var list = [1, 2, 3, 2, 5, 6, 7, 8, 9, 10]
    assert_equal(1, list.count(1))
    assert_equal(2, list.count(2))
    assert_equal(0, list.count(4))

    var list2 = List[Int]()
    assert_equal(0, list2.count(1))


def test_list_add():
    var a = [1, 2, 3]
    var b = [4, 5, 6]
    var c = a + b.copy()
    assert_equal(len(c), 6)
    # check that original values aren't modified
    assert_equal(len(a), 3)
    assert_equal(len(b), 3)
    assert_equal(c.__str__(), "[1, 2, 3, 4, 5, 6]")

    a += b.copy()
    assert_equal(len(a), 6)
    assert_equal(a.__str__(), "[1, 2, 3, 4, 5, 6]")
    assert_equal(len(b), 3)

    a = [1, 2, 3]
    a += b^
    assert_equal(len(a), 6)
    assert_equal(a.__str__(), "[1, 2, 3, 4, 5, 6]")

    var d = [1, 2, 3]
    var e = [4, 5, 6]
    var f = d + e^
    assert_equal(len(f), 6)
    assert_equal(f.__str__(), "[1, 2, 3, 4, 5, 6]")

    var l = [1, 2, 3]
    l += []
    assert_equal(len(l), 3)


def test_list_mult():
    var a = [1, 2, 3]
    var b = a * 2
    assert_equal(len(b), 6)
    assert_equal(b.__str__(), "[1, 2, 3, 1, 2, 3]")
    b = a * 3
    assert_equal(len(b), 9)
    assert_equal(b.__str__(), "[1, 2, 3, 1, 2, 3, 1, 2, 3]")
    a *= 2
    assert_equal(len(a), 6)
    assert_equal(a.__str__(), "[1, 2, 3, 1, 2, 3]")

    var l = [1, 2]
    l *= 1
    assert_equal(len(l), 2)

    l *= 0
    assert_equal(len(l), 0)
    assert_equal(len(List[Int](1, 2, 3) * 0), 0)


def test_list_contains():
    var x = [1, 2, 3]
    assert_false(0 in x)
    assert_true(1 in x)
    assert_false(4 in x)

    # TODO: implement List.__eq__ for Self[Copyable & Movable & Comparable]
    # var y = List[List[Int]]()
    # y.append([1, 2])
    # assert_equal([1, 2] in y,True)
    # assert_equal([0, 1] in y,False)


def test_list_eq_ne():
    var l1 = [1, 2, 3]
    var l2 = [1, 2, 3]
    assert_true(l1 == l2)
    assert_false(l1 != l2)

    var l3 = [1, 2, 3, 4]
    assert_false(l1 == l3)
    assert_true(l1 != l3)

    var l4 = List[Int]()
    var l5 = List[Int]()
    assert_true(l4 == l5)
    assert_true(l1 != l4)

    var l6 = ["a", "b", "c"]
    var l7 = ["a", "b", "c"]
    var l8 = ["a", "b"]
    assert_true(l6 == l7)
    assert_false(l6 != l7)
    assert_false(l6 == l8)


def test_list_init_span():
    var l = [String("a"), "bb", "cc", "def"]
    var sp = Span(l)
    var l2 = List[String](sp)
    for val1, val2 in zip(l, l2):
        assert_equal(val1, val2)


def test_indexing():
    var l = [1, 2, 3]
    assert_equal(l[Int(1)], 2)
    assert_equal(l[False], 1)
    assert_equal(l[True], 2)
    assert_equal(l[2], 3)


# ===-------------------------------------------------------------------===#
# List dtor tests
# ===-------------------------------------------------------------------===#


def test_list_dtor():
    var dtor_count = 0

    var ptr = UnsafePointer(to=dtor_count).origin_cast[False]()
    var l = List[DelCounter[ptr.origin]]()
    assert_equal(dtor_count, 0)

    l.append(DelCounter(ptr))
    assert_equal(dtor_count, 0)

    l^.__del__()
    assert_equal(dtor_count, 1)


def test_destructor_trivial_elements():
    var dtor_count = 0

    var ptr = UnsafePointer(to=dtor_count).origin_cast[False]()
    var l = List[DelCounter[ptr.origin, trivial_del=True]]()
    l.append(DelCounter[ptr.origin, trivial_del=True](ptr))

    l^.__del__()

    assert_equal(dtor_count, 0)


def test_list_repr():
    var l = [1, 2, 3]
    assert_equal(l.__repr__(), "[1, 2, 3]")
    var empty = List[Int]()
    assert_equal(empty.__repr__(), "[]")


def test_list_fill_constructor():
    var l = List[Int32](length=10, fill=3)
    assert_equal(len(l), 10)

    for i in range(10):
        assert_equal(l[i], 3)

    var l2 = List[String](length=20, fill="hi")
    assert_equal(len(l2), 20)

    for i in range(20):
        assert_equal(l2[i], "hi")


def test_uninit_ctor():
    var list = List[String](unsafe_uninit_length=2)

    UnsafePointer(to=list[0]).init_pointee_move("hello ")
    UnsafePointer(to=list[1]).init_pointee_move("world")
    assert_equal(list[0], "hello ")
    assert_equal(list[1], "world")

    # Resize with uninitialized memory.
    var list2 = List[String]()
    list2.resize(unsafe_uninit_length=2)
    (list2.unsafe_ptr() + 0).init_pointee_move("hello ")
    (list2.unsafe_ptr() + 1).init_pointee_move("world")
    assert_equal(list2[0], "hello ")
    assert_equal(list2[1], "world")


def _test_copyinit_trivial_types[dt: DType]():
    alias sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    assert_equal(len(sizes), 10)
    var test_current_size = 1

    @parameter
    for sizes_index in range(len(sizes)):
        alias current_size = sizes[sizes_index]
        x = List[Scalar[dt]]()
        for i in range(current_size):
            x.append(i)
        y = x.copy()
        assert_equal(test_current_size, current_size)
        assert_equal(len(y), current_size)
        assert_not_equal(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()))
        for i in range(current_size):
            assert_equal(Scalar[dt](i), x[i])
            assert_equal(y[i], x[i])
        test_current_size *= 2
    assert_equal(test_current_size, 1024)


def test_copyinit_trivial_types_dtypes():
    alias dtypes = (
        DType.int64,
        DType.int32,
        DType.float64,
        DType.float32,
        DType.uint8,
        DType.int8,
        DType.bool,
    )

    @parameter
    for index_dtype in range(len(dtypes)):
        _test_copyinit_trivial_types[dtypes[index_dtype]]()


def test_list_comprehension():
    var l1 = [x * x for x in range(10) if x & 1]
    assert_equal(l1, [1, 9, 25, 49, 81])

    var l2 = [x * y for x in range(3) for y in l1]
    assert_equal(l2, [0, 0, 0, 0, 0, 1, 9, 25, 49, 81, 2, 18, 50, 98, 162])


def test_list_repr_wrap():
    assert_equal(repr(List("Hello", "World")), "['Hello', 'World']")
    assert_equal(
        repr(List[UInt8](0, 1)),
        "[SIMD[DType.uint8, 1](0), SIMD[DType.uint8, 1](1)]",
    )
    var l = List(DType.int8, DType.int16)
    assert_equal(
        repr(l),
        "[DType.int8, DType.int16]",
    )


# ===-------------------------------------------------------------------===#
# main
# ===-------------------------------------------------------------------===#
def main():
    var suite = TestSuite()

    suite.test[test_mojo_issue_698]()
    suite.test[test_list]()
    suite.test[test_list_literal]()
    suite.test[test_list_unsafe_get]()
    suite.test[test_list_unsafe_set]()
    suite.test[test_list_clear]()
    suite.test[test_list_to_bool_conversion]()
    suite.test[test_list_pop]()
    suite.test[test_list_variadic_constructor]()
    suite.test[test_list_resize]()
    suite.test[test_list_reverse]()
    suite.test[test_list_reverse_move_count]()
    suite.test[test_list_insert]()
    suite.test[test_list_index]()
    suite.test[test_list_append]()
    suite.test[test_list_extend]()
    suite.test[test_list_extend_non_trivial]()
    suite.test[test_list_extend_trivial_copy_nontrivial_move,]()
    suite.test[test_list_explicit_copy]()
    suite.test[test_no_extra_copies_with_sugared_set_by_field,]()
    suite.test[test_list_copy_constructor]()
    suite.test[test_2d_dynamic_list]()
    suite.test[test_list_iter]()
    suite.test[test_list_iter_mutable]()
    suite.test[test_list_iter_bounds]()
    suite.test[test_list_span]()
    suite.test[test_list_realloc_trivial_types]()
    suite.test[test_list_realloc_trivial_copy_nontrivial_move,]()
    suite.test[test_list_boolable]()
    suite.test[test_converting_list_to_string]()
    suite.test[test_list_count]()
    suite.test[test_list_add]()
    suite.test[test_list_mult]()
    suite.test[test_list_contains]()
    suite.test[test_indexing]()
    suite.test[test_list_dtor]()
    suite.test[test_destructor_trivial_elements]()
    suite.test[test_list_repr]()
    suite.test[test_list_fill_constructor]()
    suite.test[test_uninit_ctor]()
    suite.test[test_copyinit_trivial_types_dtypes]()
    suite.test[test_list_comprehension]()
    suite.test[test_list_repr_wrap]()

    suite^.run()
