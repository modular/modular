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

from std.collections.counter import Counter

from test_utils import check_write_to
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.testing import TestSuite


def test_construction_from_iterable() raises:
    # From a list literal, inferring `Counter[String]`.
    var c = Counter(["a", "a", "a", "b", "b", "c", "d", "c", "c"])
    assert_equal(c["a"], 3)
    assert_equal(c["b"], 2)
    assert_equal(c["c"], 3)
    assert_equal(c["d"], 1)
    assert_equal(c["z"], 0)  # missing key defaults to 0
    assert_equal(len(c), 4)
    assert_equal(c.total(), 9)

    # From a `List` bound to a variable.
    var items = [String("x"), "y", "x"]
    var c2 = Counter(items)
    assert_equal(c2["x"], 2)
    assert_equal(c2["y"], 1)

    # Empty iterable.
    var empty = Counter(List[String]())
    assert_equal(len(empty), 0)


def test_construction_from_iterable_unicode() raises:
    # Multi-byte grapheme keys counted via owned `String` values.
    var c = Counter(["a", "b", "a", "é", "🎉", "é", "a"])
    assert_equal(c["a"], 3)
    assert_equal(c["b"], 1)
    assert_equal(c["é"], 2)
    assert_equal(c["🎉"], 1)
    assert_equal(c["z"], 0)
    assert_equal(len(c), 4)

    # Counting the bytes of a string, inferring `Counter[Byte]`.
    var by_byte = Counter(String("aaabbaacdd").bytes())
    assert_equal(by_byte[Byte(ord("a"))], 5)
    assert_equal(by_byte[Byte(ord("b"))], 2)
    assert_equal(by_byte[Byte(ord("c"))], 1)
    assert_equal(by_byte[Byte(ord("d"))], 2)

    # Byte-level tally of a non-ASCII string: "é" is U+00E9 -> UTF-8 0xC3 0xA9.
    var cafe = Counter(String("café").bytes())
    assert_equal(cafe[Byte(ord("c"))], 1)
    assert_equal(cafe[Byte(ord("a"))], 1)
    assert_equal(cafe[Byte(ord("f"))], 1)
    assert_equal(cafe[Byte(0xC3)], 1)
    assert_equal(cafe[Byte(0xA9)], 1)
    assert_equal(len(cafe), 5)


def test_and() raises:
    var c1 = Counter[String]()
    c1["a"] = 1
    c1["b"] = 2

    var c2 = Counter[String]()
    c2["b"] = 3
    c2["c"] = 4

    var c3 = c1 & c2

    assert_equal(c3["a"], 0)
    assert_equal(c3["b"], 2)
    assert_equal(c3["c"], 0)

    c1 &= c2

    assert_equal(c1["a"], 0)
    assert_equal(c1["b"], 2)
    assert_equal(c1["c"], 0)


# def test_bool() raises:
#     var c = Counter[String]()
#     assert_false(c)
#     c["a"] = 1
#     assert_true(c)
#     _ = c.pop("a")
#     assert_false(c)


# def test_clear() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     c.clear()

#     assert_equal(len(c), 0)
#     assert_false(c)


# def test_contains() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     assert_true("a" in c)
#     assert_true("b" in c)
#     assert_false("c" in c)


# def test_copy() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     var copy = c.copy()

#     assert_equal(copy["a"], 1)
#     assert_equal(copy["b"], 2)
#     assert_equal(len(copy), 2)

#     c["c"] = 3

#     assert_equal(copy["c"], 0)
#     assert_equal(len(copy), 2)


# def test_counter_construction() raises:
#     _ = Counter[Int]()
#     _ = Counter[Int](List[Int]())
#     _ = Counter[String](List[String]())
#     _ = Counter[Int](1, 2)


# def test_counter_getitem() raises:
#     c = Counter[Int](1, 2, 2, 3, 3, 3, 4)
#     assert_equal(c[1], 1)
#     assert_equal(c[2], 2)
#     assert_equal(c[3], 3)
#     assert_equal(c[4], 1)
#     assert_equal(c[5], 0)


# def test_fromkeys() raises:
#     var keys = [String("a"), "b", "c"]
#     var c = Counter[String].fromkeys(keys, 3)

#     assert_equal(c["a"], 3)
#     assert_equal(c["b"], 3)
#     assert_equal(c["c"], 3)
#     assert_equal(len(c), 3)


# def test_get() raises:
#     var counter = Counter[String]()
#     counter["a"] = 1
#     counter["b"] = 2

#     var a: Int = counter.get("a").value()
#     var b: Int = counter.get("b").value()
#     var c: Int = counter.get("c", 3)

#     var d: Optional[Int] = counter.get("d")

#     assert_equal(a, 1)
#     assert_equal(b, 2)
#     assert_equal(c, 3)
#     assert_false(d)


# def test_iter() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     var keys = String()
#     for key in c:
#         keys += key

#     assert_equal(keys, "ab")

#     var keys_list = ["a", "b"]
#     var keys_count = 0
#     for i, e in enumerate(c):
#         assert_equal(i, keys_count)
#         assert_equal(e, keys_list[i])
#         keys_count += 1


# def test_iter_keys() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     var keys = String()
#     for key in c.keys():
#         keys += key

#     assert_equal(keys, "ab")


# def test_iter_values() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     var sum = 0
#     for value in c.values():
#         sum += value

#     assert_equal(sum, 3)


# def test_iter_values_mut() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     for ref value in c.values():
#         value += 1

#     assert_equal(2, c["a"])
#     assert_equal(3, c["b"])
#     assert_equal(2, len(c))


# def test_iter_items() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     var keys = String()
#     var sum = 0
#     for entry in c.items():
#         keys += entry.key
#         sum += entry.value

#     assert_equal(keys, "ab")
#     assert_equal(sum, 3)


# def test_len() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     assert_equal(len(c), 2)
#     _ = c.pop("a")
#     assert_equal(len(c), 1)
#     c.clear()
#     assert_equal(len(c), 0)


# def test_total() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2

#     assert_equal(c.total(), 3)


# def test_most_common() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2
#     c["c"] = 3

#    var most_common = c.most_common(2)
#    assert_equal(len(most_common), 2)
#    assert_equal(most_common[0][0][String], "c")
#    # TODO: these asserts crash the compiler due to MOCO-3763
#    # assert_equal(most_common[0][1][Int], 3)
#    assert_equal(most_common[1][0][String], "b")
#    # assert_equal(most_common[1][1][Int], 2)


def test_most_common_clamps() raises:
    var c = Counter[Int](1, 1, 2)

    # Asking for more elements than the counter holds returns all of them,
    # like Python's `Counter.most_common`.
    var top = c.most_common(5)
    assert_equal(len(top), 2)
    assert_equal(top[0]._value, 1)
    assert_equal(top[0]._count, 2)
    assert_equal(top[1]._value, 2)
    assert_equal(top[1]._count, 1)

    assert_equal(len(c.most_common(0)), 0)
    assert_equal(len(c.most_common(-1)), 0)


# def test_eq_and_ne() raises:
#     var c1 = Counter[String]()
#     c1["a"] = 1
#     c1["b"] = 2
#     c1["d"] = 0

#     var c2 = Counter[String]()
#     c2["a"] = 1
#     c2["b"] = 2
#     c2["c"] = 0

#     assert_true(c1.__eq__(c2))
#     assert_false(c1.__ne__(c2))

#     c2["b"] = 3
#     assert_false(c1.__eq__(c2))
#     assert_true(c1.__ne__(c2))


# def test_lt_le_gt_and_ge() raises:
#     var c1 = Counter[String]()
#     c1["a"] = 1
#     c1["b"] = 2
#     c1["d"] = 0

#     var c2 = Counter[String]()
#     c2["a"] = 1
#     c2["b"] = 2
#     c2["c"] = 0

#     assert_false(c1.lt(c2))
#     assert_false(c2.lt(c1))
#     assert_true(c1.le(c2))
#     assert_true(c2.le(c1))
#     assert_false(c1.gt(c2))
#     assert_false(c2.gt(c1))
#     assert_true(c1.ge(c2))
#     assert_true(c2.ge(c1))

#     c2["a"] = 2
#     assert_false(c1.lt(c2))
#     assert_false(c2.lt(c1))
#     assert_true(c1.le(c2))
#     assert_false(c2.le(c1))
#     assert_false(c1.gt(c2))
#     assert_false(c2.gt(c1))
#     assert_false(c1.ge(c2))
#     assert_true(c2.ge(c1))

#     c2["b"] = 3
#     c2["c"] = 1
#     c2["d"] = 1
#     assert_true(c1.lt(c2))
#     assert_false(c2.lt(c1))
#     assert_true(c1.le(c2))
#     assert_false(c2.le(c1))
#     assert_false(c1.gt(c2))
#     assert_true(c2.gt(c1))
#     assert_false(c1.ge(c2))
#     assert_true(c2.ge(c1))


# def test_elements() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = 2
#     c["c"] = 3

#     var elements = c.elements()

#     assert_equal(len(elements), 6)
#     assert_equal(elements[0], "a")
#     assert_equal(elements[1], "b")
#     assert_equal(elements[2], "b")
#     assert_equal(elements[3], "c")
#     assert_equal(elements[4], "c")
#     assert_equal(elements[5], "c")


# def test_update() raises:
#     var c1 = Counter[String]()
#     c1["a"] = 1
#     c1["b"] = 2

#     var c2 = Counter[String]()
#     c2["b"] = 3
#     c2["c"] = 4

#     c1.update(c2)

#     assert_equal(c1["a"], 1)
#     assert_equal(c1["b"], 5)
#     assert_equal(c1["c"], 4)


def test_add() raises:
    var c1 = Counter[String]()
    c1["a"] = 3
    c1["b"] = 2
    c1["d"] = -1  # should be ignored

    var c2 = Counter[String]()
    c2["a"] = -2
    c2["b"] = 3
    c2["c"] = 4
    c2["e"] = 0  # should be ignored

    var c3 = c1 + c2

    assert_equal(c3["a"], 1)
    assert_equal(c3["b"], 5)
    assert_equal(c3["c"], 4)
    # Check that the original counters are not modified
    assert_equal(c1["a"], 3)
    assert_equal(c1["b"], 2)
    assert_equal(c1["d"], -1)

    c2 += c1

    assert_equal(c2["a"], 1)
    assert_equal(c2["b"], 5)
    assert_equal(c2["c"], 4)


def test_subtract() raises:
    var c1 = Counter[String]()
    c1["a"] = 4
    c1["b"] = 2
    c1["c"] = 0

    var c2 = Counter[String]()
    c2["a"] = 1
    c2["b"] = -2
    c2["c"] = 3

    c1.subtract(c2)

    assert_equal(c1["a"], 3)
    assert_equal(c1["b"], 4)
    assert_equal(c1["c"], -3)


def test_sub() raises:
    var c1 = Counter[String]()
    c1["a"] = 4
    c1["b"] = 2
    c1["c"] = 0

    var c2 = Counter[String]()
    c2["a"] = 1
    c2["b"] = -2
    c2["c"] = 3

    var c3 = c1 - c2

    assert_equal(c3["a"], 3)
    assert_equal(c3["b"], 4)
    assert_equal(c3["c"], 0)

    # Check that the original counters are not modified
    assert_equal(c1["a"], 4)
    assert_equal(c1["b"], 2)
    assert_equal(c1["c"], 0)

    c2 -= c1

    assert_equal(c2["a"], 0)
    assert_equal(c2["b"], 0)
    assert_equal(c2["c"], 3)


# def test_counter_setitem() raises:
#     c = Counter[Int]()
#     c[1] = 1
#     c[2] = 2
#     assert_equal(c[1], 1)
#     assert_equal(c[2], 2)
#     assert_equal(c[3], 0)


# def test_neg() raises:
#     var c = Counter[String]()
#     c["a"] = 1
#     c["b"] = -2
#     c["c"] = 3

#     var neg = -c

#     assert_equal(neg["a"], 0)
#     assert_equal(neg["b"], 2)
#     assert_equal(neg["c"], 0)


# def test_or() raises:
#     var c1 = Counter[String]()
#     c1["a"] = 1
#     c1["b"] = 2

#     var c2 = Counter[String]()
#     c2["b"] = 3
#     c2["c"] = 4
#     c2["d"] = -1

#     var c3 = c1 | c2

#     assert_equal(c3["a"], 1)
#     assert_equal(c3["b"], 3)
#     assert_equal(c3["c"], 4)
#     assert_equal(c3["d"], 0)

#     c1 |= c2

#     assert_equal(c1["a"], 1)
#     assert_equal(c1["b"], 3)
#     assert_equal(c1["c"], 4)
#     assert_equal(c1["d"], 0)


# def test_pop() raises:
#     var counter = Counter[String]()
#     counter["a"] = 1
#     counter["b"] = 2

#     var a = counter.pop("a")
#     var b = counter.pop("b")
#     var c = counter.pop("c", 3)

#     assert_equal(a, 1)
#     assert_equal(b, 2)
#     assert_equal(c, 3)

#     with assert_raises(contains="KeyError"):
#         _ = counter.pop("not_a_key")


# def test_popitem() raises:
#     var counter = Counter[String]()
#     counter["a"] = 1
#     counter["b"] = 2

#     var item = counter.popitem()
#     assert_equal(item[0][String], "b")
#     assert_equal(item[1][Int], 2)

#     item = counter.popitem()
#     assert_equal(item[0][String], "a")
#     assert_equal(item[1][Int], 1)

#     with assert_raises():
#         _ = counter.popitem()


# def test_write_to() raises:
#     var c = Counter[String]()
#     c["a"] = 3
#     c["b"] = 2
#     check_write_to(c, expected="{a: 3, b: 2}", is_repr=False)

#     var empty = Counter[String]()
#     check_write_to(empty, expected="{}", is_repr=False)

#     var single = Counter[String]()
#     single["x"] = 1
#     check_write_to(single, expected="{x: 1}", is_repr=False)


# def test_write_repr_to() raises:
#     var c = Counter[String]()
#     c["a"] = 3
#     c["b"] = 2
#     check_write_to(
#         c,
#         expected="Counter[String]({'a': Int(3), 'b': Int(2)})",
#         is_repr=True,
#     )

#     var empty = Counter[String]()
#     check_write_to(empty, expected="Counter[String]({})", is_repr=True)

#     var single = Counter[String]()
#     single["x"] = 1
#     check_write_to(
#         single, expected="Counter[String]({'x': Int(1)})", is_repr=True
#     )


# def test_counter_iter_owned() raises:
#     var c = Counter[String]("a", "a", "b", "c")
#     var keys = List[String]()
#     for key in c^:
#         keys.append(key)

#     assert_equal(len(keys), 3)
#     assert_true(String("a") in keys)
#     assert_true(String("b") in keys)
#     assert_true(String("c") in keys)


# def test_counter_iter_owned_bounds() raises:
#     var c = Counter[String]("a", "b", "c")
#     var it = c^.__iter__()
#     assert_equal(it.bounds()[0], 3)
#     _ = it.__next__()
#     assert_equal(it.bounds()[0], 2)
#     _ = it.__next__()
#     assert_equal(it.bounds()[0], 1)
#     _ = it.__next__()
#     assert_equal(it.bounds()[0], 0)


# ===----------------------------------------------------------------------=== #
# In-place operators that remove keys
# ===----------------------------------------------------------------------=== #


def _assert_keys_and_counts(
    c: Counter[Int], keys: List[Int], counts: List[Int]
) raises:
    """Checks that `c` holds exactly `keys`, in order, with `counts`."""
    assert_equal(len(c), len(keys))
    var i = 0
    for key in c.keys():
        assert_true(i < len(keys), "more keys than expected")
        assert_equal(key, keys[i])
        assert_equal(c[key], counts[i])
        i += 1
    assert_equal(i, len(keys))


def _check_inplace_removal(n: Int, remove: List[Int]) raises:
    """Runs `&=`, `-=` and `+=` on a counter with keys `0..n-1`, so that the
    keys in `remove` end up with a non-positive count or are excluded, and
    checks that exactly the other keys remain, with the right counts and in
    insertion order.
    """
    var removed = List[Bool](length=n, fill=False)
    for k in remove:
        removed[k] = True

    var kept_keys = List[Int]()
    var kept_counts = List[Int]()
    for k in range(n):
        if not removed[k]:
            kept_keys.append(k)
            kept_counts.append(3)

    # Every key starts at 5, and every kept key ends at 3.
    var base = Counter[Int]()
    for k in range(n):
        base[k] = 5

    # `&=`: drop the removed keys from `other`, so they are excluded.
    var and_other = Counter[Int]()
    for k in range(n):
        if not removed[k]:
            and_other[k] = 3
    var a = base.copy()
    a &= and_other
    _assert_keys_and_counts(a, kept_keys, kept_counts)
    _assert_keys_and_counts(base & and_other, kept_keys, kept_counts)

    # `-=`: removed keys alternate between landing on 0 and going negative.
    var sub_other = Counter[Int]()
    for k in range(n):
        sub_other[k] = 5 + (k % 2) if removed[k] else 2
    var s = base.copy()
    s -= sub_other
    _assert_keys_and_counts(s, kept_keys, kept_counts)
    _assert_keys_and_counts(base - sub_other, kept_keys, kept_counts)

    # `+=`: same, with negative counts in `other`.
    var add_other = Counter[Int]()
    for k in range(n):
        add_other[k] = -5 - (k % 2) if removed[k] else -2
    var p = base.copy()
    p += add_other
    _assert_keys_and_counts(p, kept_keys, kept_counts)
    _assert_keys_and_counts(base + add_other, kept_keys, kept_counts)


def _key_range(start: Int, end: Int) -> List[Int]:
    var keys = List[Int]()
    for k in range(start, end):
        keys.append(k)
    return keys^


def test_iand_issue_7641() raises:
    # Reproducer from https://github.com/modular/modular/issues/7641.
    var left = Counter[Int].fromkeys([1, 2, 3], 1)
    var right = Counter[Int].fromkeys([2], 1)
    var ordinary = left & right
    _assert_keys_and_counts(ordinary, [2], [1])
    left &= right
    _assert_keys_and_counts(left, [2], [1])


def test_inplace_ops_issue_7658() raises:
    # Reproducer from https://github.com/modular/modular/issues/7658.
    var a = Counter[Int]()
    var b = Counter[Int]()
    for i in range(100):
        a[i] = 1
        b[i] = 5
    assert_equal(len(a - b), 0)
    a -= b
    assert_equal(len(a), 0)

    var c = Counter[Int]()
    var d = Counter[Int]()
    for i in range(100):
        c[i] = 5
    for i in range(90, 100):
        d[i] = 3
    assert_equal(len(c & d), 10)
    c &= d
    assert_equal(len(c), 10)
    for i in range(90, 100):
        assert_equal(c[i], 3)

    var e = Counter[Int]()
    var f = Counter[Int]()
    for i in range(100):
        e[i] = 1
        f[i] = -1
    assert_equal(len(e + f), 0)
    e += f
    assert_equal(len(e), 0)


def test_inplace_ops_remove_single_key() raises:
    _check_inplace_removal(1, [0])
    _check_inplace_removal(2, [0])
    _check_inplace_removal(2, [1])
    _check_inplace_removal(10, [5])


def test_inplace_ops_remove_first_or_last_key() raises:
    _check_inplace_removal(10, [0])
    _check_inplace_removal(10, [9])
    _check_inplace_removal(10, [0, 9])
    _check_inplace_removal(100, [0])
    _check_inplace_removal(100, [99])
    _check_inplace_removal(100, [0, 99])


def test_inplace_ops_remove_adjacent_keys() raises:
    _check_inplace_removal(10, [3, 4])
    _check_inplace_removal(100, _key_range(0, 50))
    _check_inplace_removal(100, _key_range(50, 100))
    _check_inplace_removal(100, _key_range(20, 80))
    _check_inplace_removal(1000, _key_range(0, 999))
    _check_inplace_removal(1000, _key_range(1, 1000))


def test_inplace_ops_remove_many_keys() raises:
    var evens = List[Int]()
    var odds = List[Int]()
    for k in range(1000):
        if k % 2 == 0:
            evens.append(k)
        else:
            odds.append(k)
    _check_inplace_removal(1000, evens)
    _check_inplace_removal(1000, odds)

    var most = List[Int]()
    for k in range(1000):
        if k % 10 != 7:
            most.append(k)
    _check_inplace_removal(1000, most)


def test_inplace_ops_remove_every_key() raises:
    for n in [1, 2, 3, 8, 9, 100, 1000]:
        _check_inplace_removal(n, _key_range(0, n))


def test_iand_after_pop() raises:
    # Popping leaves stale slots in the underlying `Dict`, so the next insert
    # into it compacts its storage. `&=` must still visit every key.
    var c = Counter[Int]()
    for i in range(100):
        c[i] = 5
    for i in range(60):
        _ = c.pop(i)

    var other = Counter[Int]()
    for i in range(60, 100):
        other[i] = 3
    c &= other
    _assert_keys_and_counts(
        c, _key_range(60, 100), List[Int](length=40, fill=3)
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
