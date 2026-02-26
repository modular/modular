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

from testing import TestSuite
from testing import assert_equal, assert_raises, assert_true, assert_false
from test_utils import MoveOnly, check_write_to
from math import iota
from memory import ImmutSpan, MutSpan


def test_span_list_int():
    var l = [1, 2, 3, 4, 5, 6, 7]
    var s = Span(list=l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])
    assert_equal(s[-1], l[-1])

    # Test mutation
    s[0] = 9
    assert_equal(s[0], 9)
    assert_equal(l[0], 9)

    s[-1] = 0
    assert_equal(s[-1], 0)
    assert_equal(l[-1], 0)


def test_span_list_str():
    var l = ["a", "b", "c", "d", "e", "f", "g"]
    var s = Span(l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])

    # Test mutation
    s[0] = "h"
    assert_equal(s[0], "h")
    assert_equal(l[0], "h")

    s[-1] = "i"
    assert_equal(s[-1], "i")
    assert_equal(l[-1], "i")


def test_span_array_int():
    var l: InlineArray[Int, 7] = [1, 2, 3, 4, 5, 6, 7]
    var s = Span[Int](array=l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])

    # Test mutation
    s[0] = 9
    assert_equal(s[0], 9)
    assert_equal(l[0], 9)

    s[-1] = 0
    assert_equal(s[-1], 0)
    assert_equal(l[-1], 0)


def test_span_array_str():
    var l: InlineArray[String, 7] = ["a", "b", "c", "d", "e", "f", "g"]
    var s = Span[String](array=l)
    assert_equal(len(s), len(l))
    for i in range(len(s)):
        assert_equal(l[i], s[i])
    # subslice
    var s2 = s[2:]
    assert_equal(s2[0], l[2])
    assert_equal(s2[1], l[3])
    assert_equal(s2[2], l[4])
    assert_equal(s2[3], l[5])

    # Test mutation
    s[0] = "h"
    assert_equal(s[0], "h")
    assert_equal(l[0], "h")

    s[-1] = "i"
    assert_equal(s[-1], "i")
    assert_equal(l[-1], "i")


def test_indexing():
    var l: InlineArray[Int, 7] = [1, 2, 3, 4, 5, 6, 7]
    var s = Span[Int](array=l)
    assert_equal(s[True], 2)
    assert_equal(s[Int(0)], 1)
    assert_equal(s[3], 4)


def test_span_slice():
    def compare(s: Span[Int], l: List[Int]) -> Bool:
        if len(s) != len(l):
            return False
        for i in range(len(s)):
            if s[i] != l[i]:
                return False
        return True

    var l = [1, 2, 3, 4, 5]
    var s = Span(l)
    var res = s[1:2]
    assert_equal(res[0], 2)
    res = s[1:-1]
    assert_equal(res[0], 2)
    assert_equal(res[1], 3)
    assert_equal(res[2], 4)


def test_copy_from():
    var a = [0, 1, 2, 3]
    var b = [4, 5, 6, 7, 8, 9, 10]
    var s = Span(a)
    var s2 = Span(b)
    s.copy_from(s2[: len(a)])
    for i, val in enumerate(a):
        assert_equal(val, b[i])
        assert_equal(s[i], s2[i])


def test_bool():
    var l: InlineArray[String, 7] = ["a", "b", "c", "d", "e", "f", "g"]
    var s = Span[String](l)
    assert_true(s)
    assert_true(not s[0:0])


def test_contains():
    items: List[Byte] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    span = Span(items)
    assert_true(0 not in span)
    assert_true(16 not in span)
    for item in items:
        assert_true(item in span)


def test_equality():
    var l: InlineArray[String, 7] = ["a", "b", "c", "d", "e", "f", "g"]
    var l2 = [String("a"), "b", "c", "d", "e", "f", "g"]
    var sp = Span[String](l)
    var sp2 = Span[String](l)
    var sp3 = Span(l2)
    # same pointer
    assert_true(sp == sp2)
    # different pointer
    assert_true(sp == sp3)
    # different length
    assert_true(sp != sp3[:-1])
    # empty
    assert_true(sp[0:0] == sp3[0:0])


def test_fill():
    var a = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    var s = Span(a)

    s.fill(2)

    for i, val in enumerate(a):
        assert_equal(val, 2)
        assert_equal(s[i], 2)


def test_ref():
    var l: InlineArray[Int, 3] = [1, 2, 3]
    var s = Span[Int](array=l)
    assert_true(s.as_ref() == Pointer(to=l.unsafe_ptr()[]))


def test_reversed():
    var forward: InlineArray[Int, 3] = [1, 2, 3]
    var backward: InlineArray[Int, 3] = [3, 2, 1]
    var s = Span[Int](forward)
    var i = 0
    for num in reversed(s):
        assert_equal(num, backward[i])
        i += 1


# We don't actually need to call this test
# but we want to make sure it compiles
def test_span_coerce():
    var l = [1, 2, 3]
    var a: InlineArray[Int, 3] = [1, 2, 3]

    fn takes_span(s: Span[Int]):
        pass

    takes_span(l)
    takes_span(a)


def test_swap_elements():
    var l = [1, 2, 3, 4, 5]
    var s = Span(l)
    s.swap_elements(1, 4)
    assert_equal(l[1], 5)
    assert_equal(l[4], 2)

    var l2 = ["hi", "hello", "hey"]
    var s2 = Span(l2)
    s2.swap_elements(0, 2)
    assert_equal(l2[0], "hey")
    assert_equal(l2[2], "hi")

    with assert_raises(contains="index out of bounds"):
        s2.swap_elements(0, 4)


def test_merge():
    var a = [1, 2, 3]
    var b = [4, 5, 6]

    fn inner(cond: Bool, mut a: List[Int], mut b: List[Int]):
        var either = Span(a) if cond else Span(b)
        either[0] = 0
        either[-1] = 10

    inner(True, a, b)
    inner(False, a, b)

    assert_equal(a, [0, 2, 10])
    assert_equal(b, [0, 5, 10])


def test_reverse():
    def _test_dtype[D: DType]():
        var forward: InlineArray[Scalar[D], 11] = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ]
        var backward: InlineArray[Scalar[D], 11] = [
            11,
            10,
            9,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            1,
        ]
        s = Span(forward)
        s.reverse()
        i = 0
        for num in s:
            assert_equal(num, backward[i])
            i += 1

    _test_dtype[DType.uint8]()
    _test_dtype[DType.uint16]()
    _test_dtype[DType.uint32]()
    _test_dtype[DType.uint64]()
    _test_dtype[DType.int8]()
    _test_dtype[DType.int16]()
    _test_dtype[DType.int32]()
    _test_dtype[DType.int64]()
    _test_dtype[DType.float16]()
    _test_dtype[DType.float32]()
    _test_dtype[DType.float64]()


def test_apply():
    @parameter
    fn _twice[D: DType, w: Int](x: SIMD[D, w]) -> SIMD[D, w]:
        return x * 2

    @parameter
    fn _where[D: DType, w: Int](x: SIMD[D, w]) -> SIMD[DType.bool, w]:
        return (x % 2).eq(0)

    def _test[D: DType]():
        items: List[Scalar[D]] = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ]
        twice = items.copy()
        span = Span(twice)
        span.apply[func = _twice[D]]()
        for i, item in enumerate(items):
            assert_true(span[i] == item * 2)

        # twice only even numbers
        twice = items.copy()
        span = Span(twice)
        span.apply[func = _twice[D], cond = _where[D]]()
        for i, item in enumerate(items):
            if item % 2 == 0:
                assert_true(span[i] == item * 2)
            else:
                assert_true(span[i] == item)

    _test[DType.uint8]()
    _test[DType.uint16]()
    _test[DType.uint32]()
    _test[DType.uint64]()
    _test[DType.int8]()
    _test[DType.int16]()
    _test[DType.int32]()
    _test[DType.int64]()
    _test[DType.float16]()
    _test[DType.float32]()
    _test[DType.float64]()


def test_count_func():
    @parameter
    fn is_2[w: Int](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
        return v.eq(2)

    var data = Span[Byte]([Byte(0), 1, 2, 1, 2, 1, 2])
    assert_equal(3, Int(data.count[func=is_2]()))
    assert_equal(2, Int(data[:-1].count[func=is_2]()))
    assert_equal(1, Int(data[:3].count[func=is_2]()))


def test_unsafe_subspan():
    var data = Span[Int]([0, 1, 2, 3, 4])

    var subspan1 = data.unsafe_subspan(offset=0, length=4)
    assert_equal(List(subspan1), [0, 1, 2, 3])

    var subspan2 = data.unsafe_subspan(offset=1, length=3)
    assert_equal(List(subspan2), [1, 2, 3])


def test_binary_search():
    def _test[dtype: DType]():
        comptime max_val = Int(Scalar[dtype].MAX)
        var data = List[Scalar[dtype]](unsafe_uninit_length=max_val + 1)
        iota(data)

        # make sure we aren't reading an empty pointer
        var view = Span(data)[:0]
        assert_true(view._binary_search_index(0) is None)
        view = Span(data)[:1]
        assert_true(view._binary_search_index(0))
        assert_equal(view._binary_search_index(0).value(), 0)
        view = Span(data)[: len(data) - 1]
        assert_true(view._binary_search_index(1))
        assert_equal(view._binary_search_index(1).value(), 1)
        view = Span(data)
        assert_true(view._binary_search_index(Scalar[dtype](max_val)))
        assert_equal(
            view._binary_search_index(Scalar[dtype](max_val)).value(),
            UInt(max_val),
        )
        view = Span(data)[: len(data) - 1]
        assert_true(view._binary_search_index(Scalar[dtype](max_val - 1)))
        assert_equal(
            view._binary_search_index(Scalar[dtype](max_val - 1)).value(),
            UInt(max_val - 1),
        )

    _test[DType.uint8]()
    _test[DType.int8]()
    _test[DType.uint16]()
    _test[DType.int16]()


def test_binary_sarch_by():
    var data: List[Int] = [1, 3, 5, 7, 9, 11, 13]
    var span = Span(data)

    fn cmp_7(x: Int) -> Int:
        return x - 7

    var result = span.binary_search_by[cmp_7]()
    assert_equal(3, result.value())

    fn cmp_6(x: Int) -> Int:
        return x - 6

    var result2 = span.binary_search_by[cmp_6]()
    assert_true(not result2)

    fn cmp_1(x: Int) -> Int:
        return x - 1

    var result3 = span.binary_search_by[cmp_1]()
    assert_equal(0, result3.value())

    fn cmp_13(x: Int) -> Int:
        return x - 13

    var result4 = span.binary_search_by[cmp_13]()
    assert_equal(6, result4.value())


def test_iter():
    var data = [1, 2, 3, 4, 5]
    var span = Span(data)
    var it = iter(span)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    assert_equal(next(it), 4)
    assert_equal(next(it), 5)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_iter_empty():
    var data: List[Int] = []
    var span = Span(data)
    var it = iter(span)
    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_mut_span_alias():
    var data = [1, 2, 3, 4, 5]

    fn fill_span(span: MutSpan[Int, _]):
        span.fill(42)

    fill_span(data)
    for val in data:
        assert_equal(val, 42)


def test_immut_span_alias():
    var data: List[Int] = [1, 2, 3, 4, 5]

    fn sum_span(span: ImmutSpan[Int, _]) -> Int:
        var total = 0
        for i in range(len(span)):
            total += span[i]
        return total

    # ImmutSpan works with both mutable and immutable data
    assert_equal(sum_span(data), 15)


def test_span_write_to():
    check_write_to(Span([1, 2, 3]), expected="[1, 2, 3]", is_repr=False)
    check_write_to(Span(List[Int]()), expected="[]", is_repr=False)
    check_write_to(Span([42]), expected="[42]", is_repr=False)


def test_span_write_repr_to():
    check_write_to(
        Span([1, 2, 3]),
        expected="Span[mut=False, Int]([Int(1), Int(2), Int(3)])",
        is_repr=True,
    )
    check_write_to(
        Span(List[Int]()), expected="Span[mut=False, Int]([])", is_repr=True
    )
    check_write_to(
        Span([42]), expected="Span[mut=False, Int]([Int(42)])", is_repr=True
    )


def test_span_with_move_only_type():
    var ptr = alloc[MoveOnly[Int]](1)
    ptr.init_pointee_move(MoveOnly(42))
    var span = Span(ptr=ptr, length=1)
    assert_equal(span[0].data, 42)
    ptr.destroy_pointee()
    ptr.free()


struct NonMovable:
    var x: Int


def test_span_with_non_movable_type():
    var ptr = alloc[NonMovable](1)
    var _span = Span(ptr=ptr, length=0)
    ptr.free()


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
