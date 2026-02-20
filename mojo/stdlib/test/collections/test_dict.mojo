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

from collections.dict import (
    Dict,
    DictKeyError,
    EmptyDictError,
    OwnedKwargsDict,
    _GROUP_WIDTH,
)

from hashlib import Hasher, default_comp_time_hasher

from test_utils import CopyCounter, check_write_to
from testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def test_dict_construction():
    _ = Dict[Int, Int]()
    _ = Dict[String, Int]()


def test_dict_literals():
    a = {"foo": 1, "bar": 2}
    assert_equal(a["foo"], 1)

    b = {1: 4, 2: 7, 3: 18}
    assert_equal(b[1], 4)
    assert_equal(b[2], 7)
    assert_equal(b[3], 18)
    assert_false(4 in b)


def test_dict_fromkeys():
    comptime keys = [String("a"), "b"]
    var expected_dict = Dict[String, Int]()
    expected_dict["a"] = 1
    expected_dict["b"] = 1
    var dict = Dict.fromkeys(materialize[keys](), 1)

    assert_equal(len(dict), len(expected_dict))

    for k_v in expected_dict.items():
        var k = k_v.key
        var v = k_v.value
        assert_true(k in dict)
        assert_equal(dict[k], v)


def test_dict_fromkeys_optional():
    comptime keys = [String("a"), "b", "c"]
    var expected_dict: Dict[String, Optional[Int]] = {
        "a": None,
        "b": None,
        "c": None,
    }
    var dict = Dict[_, Int].fromkeys(materialize[keys]())

    assert_equal(len(dict), len(expected_dict))

    for k_v in expected_dict.items():
        var k = k_v.key
        var v = k_v.value
        assert_true(k in dict)
        assert_false(v)


def test_basic():
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    assert_equal(1, dict["a"])
    assert_equal(2, dict["b"])

    ptr = Pointer(to=dict["a"])
    assert_equal(1, ptr[])
    ptr[] = 17
    assert_equal(17, dict["a"])


def test_basic_no_copies():
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2

    assert_equal(1, dict["a"])
    assert_equal(2, dict["b"])


def test_multiple_resizes():
    var dict: Dict[String, Int] = {}
    for i in range(20):
        dict[String("key", i)] = i + 1
    assert_equal(11, dict["key10"])
    assert_equal(20, dict["key19"])


def test_bool_conversion():
    var dict: Dict[String, Int] = {}
    assert_false(dict)
    dict["a"] = 1
    assert_true(dict)
    dict["b"] = 2
    assert_true(dict)
    _ = dict.pop("a")
    assert_true(dict)
    _ = dict.pop("b")
    assert_false(dict)


def test_big_dict():
    var dict: Dict[String, Int] = {}
    for i in range(2000):
        dict[String("key", i)] = i + 1
    assert_equal(2000, len(dict))


def test_dict_string_representation_string_int():
    var d: Dict[String, Int] = {"a": 1, "b": 2}
    check_write_to(d, expected="{a: 1, b: 2}", is_repr=False)
    check_write_to(
        d,
        expected="Dict[String, Int]({'a': Int(1), 'b': Int(2)})",
        is_repr=True,
    )


def test_dict_string_representation_int_int():
    var d: Dict[Int, Int] = {1: 2, 3: 4}
    check_write_to(d, expected="{1: 2, 3: 4}", is_repr=False)
    check_write_to(
        d,
        expected="Dict[Int, Int]({Int(1): Int(2), Int(3): Int(4)})",
        is_repr=True,
    )


def test_dict_string_representation_custom_hasher():
    var some_dict: Dict[Int, Int, default_comp_time_hasher] = {}
    some_dict[1] = 10
    some_dict[2] = 20

    var dict_as_string = some_dict.__str__()

    # Just check that __str__ is callable and produces something
    assert_true(
        some_dict._minimum_size_of_string_representation()
        <= len(dict_as_string)
    )


def test_compact():
    var dict: Dict[String, Int] = {}
    for i in range(20):
        var key = String("key", i)
        dict[key] = i + 1
        _ = dict.pop(key)
    assert_equal(0, len(dict))


def test_compact_with_elements():
    var dict: Dict[String, Int] = {}
    for i in range(5):
        var key = String("key", i)
        dict[key] = i + 1
    for i in range(5, 20):
        var key = String("key", i)
        dict[key] = i + 1
        _ = dict.pop(key)
    assert_equal(5, len(dict))


def test_pop_default():
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    assert_equal(1, dict.pop("a", -1))
    assert_equal(2, dict.pop("b", -1))
    assert_equal(-1, dict.pop("c", -1))


def test_key_error():
    var dict: Dict[String, Int] = {}

    with assert_raises(contains="KeyError"):
        _ = dict["a"]
    with assert_raises(contains="KeyError"):
        _ = dict.pop("a")


def _test_iter_bounds[
    I: Iterator, //
](var dict_iter: I, dict_len: Int,):
    var iter = dict_iter^
    for i in range(dict_len):
        var lower, upper = iter.bounds()
        assert_equal(dict_len - i, lower)
        assert_equal(dict_len - i, upper.value())
        _ = trait_downcast_var[Movable & ImplicitlyDestructible](
            iter.__next__()
        )

    var lower, upper = iter.bounds()
    assert_equal(0, lower)
    assert_equal(0, upper.value())


def test_iter():
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    var keys = String()
    for key in dict:
        keys += key

    assert_equal(keys, "ab")
    _test_iter_bounds(dict.__iter__(), len(dict))

    var empty_dict: Dict[String, Int] = {}
    with assert_raises():
        var it = iter(empty_dict)
        _ = it.__next__()  # raises StopIteration


def test_iter_keys():
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    var keys = String()
    for key in dict.keys():
        keys += key

    assert_equal(keys, "ab")
    _test_iter_bounds(dict.keys(), len(dict))


def test_iter_values():
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    var sum = 0
    for value in dict.values():
        sum += value

    assert_equal(sum, 3)
    _test_iter_bounds(dict.values(), len(dict))


def test_iter_values_mut():
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    for ref value in dict.values():
        value += 1

    assert_equal(2, dict["a"])
    assert_equal(3, dict["b"])
    assert_equal(2, len(dict))
    _test_iter_bounds(dict.values(), len(dict))


def test_iter_items():
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    var keys = String()
    var sum = 0
    for entry in dict.items():
        keys += entry.key
        sum += entry.value

    assert_equal(keys, "ab")
    assert_equal(sum, 3)

    # TODO: _DictItemIter does not conform to `Iterator` yet
    # _test_iter_bounds(dict.values(), len(dict))


def test_iter_take_items():
    var dict: Dict[Int, String] = {0: "a", 1: "b", 2: "c"}

    var values = String()
    var keys = 0

    for entry in dict.take_items():
        keys += entry.key
        values += entry.value

    assert_equal(values, "abc")
    assert_equal(keys, 3)
    assert_equal(len(dict), 0)
    with assert_raises():
        var it = dict.take_items()
        _ = it.__next__()  # raises StopIteration

    for i in range(3):
        with assert_raises(contains="KeyError"):
            _ = dict[i]


def test_iter_take_items_empty():
    var dict: Dict[Int, String] = {}

    var count = 0
    for _ in dict.take_items():
        count += 1
    assert_equal(len(dict), 0)
    assert_equal(count, 0)


def test_dict_contains():
    var dict: Dict[String, Int] = {}
    dict["abc"] = 1
    dict["def"] = 2
    assert_true("abc" in dict)
    assert_true("def" in dict)
    assert_false("c" in dict)


def test_dict_copy():
    var orig: Dict[String, Int] = {}
    orig["a"] = 1

    # test values copied to new Dict
    var copy = orig.copy()
    assert_equal(1, copy["a"])

    # test there are two copies of dict and
    # they don't share underlying memory
    copy["a"] = 2
    assert_equal(2, copy["a"])
    assert_equal(1, orig["a"])


def test_dict_copy_delete_original():
    var orig: Dict[String, Int] = {}
    orig["a"] = 1

    # test values copied to new Dict
    var copy = orig.copy()
    # don't access the original dict, anymore, confirm that
    # deleting the original doesn't violate the integrity of the copy
    assert_equal(1, copy["a"])


def test_dict_copy_add_new_item():
    var orig: Dict[String, Int] = {}
    orig["a"] = 1

    # test values copied to new Dict
    var copy = orig.copy()
    assert_equal(1, copy["a"])

    # test there are two copies of dict and
    # they don't share underlying memory
    copy["b"] = 2
    assert_false(String(2) in orig)


def test_dict_copy_calls_copy_constructor():
    var orig: Dict[String, CopyCounter[]] = {}
    orig["a"] = CopyCounter()

    # test values copied to new Dict
    var copy = orig.copy()
    assert_equal(0, orig["a"].copy_count)
    assert_equal(1, copy["a"].copy_count)
    assert_equal(0, orig._find_ref("a").copy_count)
    assert_equal(1, copy._find_ref("a").copy_count)


def test_dict_update_nominal():
    var orig: Dict[String, Int] = {}
    orig["a"] = 1
    orig["b"] = 2

    var new: Dict[String, Int] = {}
    new["b"] = 3
    new["c"] = 4

    orig.update(new)

    assert_equal(orig["a"], 1)
    assert_equal(orig["b"], 3)
    assert_equal(orig["c"], 4)


def test_dict_update_empty_origin():
    var orig: Dict[String, Int] = {}
    var new: Dict[String, Int] = {}
    new["b"] = 3
    new["c"] = 4

    orig.update(new)

    assert_equal(orig["b"], 3)
    assert_equal(orig["c"], 4)


def test_dict_or():
    var orig: Dict[String, Int] = {}
    var new: Dict[String, Int] = {}

    new["b"] = 3
    new["c"] = 4
    orig["d"] = 5
    orig["b"] = 8

    var out = orig | new

    assert_equal(out["b"], 3)
    assert_equal(out["c"], 4)
    assert_equal(out["d"], 5)

    orig |= new

    assert_equal(orig["b"], 3)
    assert_equal(orig["c"], 4)
    assert_equal(orig["d"], 5)

    orig = Dict[String, Int]()
    new = Dict[String, Int]()
    new["b"] = 3
    new["c"] = 4

    orig |= new

    assert_equal(orig["b"], 3)
    assert_equal(orig["c"], 4)

    orig = Dict[String, Int]()
    orig["a"] = 1
    orig["b"] = 2

    new = Dict[String, Int]()

    orig = orig | new

    assert_equal(orig["a"], 1)
    assert_equal(orig["b"], 2)
    assert_equal(len(orig), 2)

    orig = Dict[String, Int]()
    new = Dict[String, Int]()
    orig["a"] = 1
    orig["b"] = 2
    new["c"] = 3
    new["d"] = 4
    orig |= new
    assert_equal(orig["a"], 1)
    assert_equal(orig["b"], 2)
    assert_equal(orig["c"], 3)
    assert_equal(orig["d"], 4)

    orig = Dict[String, Int]()
    new = Dict[String, Int]()
    assert_equal(len(orig | new), 0)


def test_dict_update_empty_new():
    var orig: Dict[String, Int] = {}
    orig["a"] = 1
    orig["b"] = 2

    var new: Dict[String, Int] = {}

    orig.update(new)

    assert_equal(orig["a"], 1)
    assert_equal(orig["b"], 2)
    assert_equal(len(orig), 2)


@fieldwise_init("implicit")
struct DummyKey(KeyElement):
    var value: Int

    fn __hash__[H: Hasher](self, mut hasher: H):
        return hasher.update(self.value)

    fn __eq__(self, other: DummyKey) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: DummyKey) -> Bool:
        return self.value != other.value


def test_mojo_issue_1729():
    var keys = [
        7005684093727295727,
        2833576045803927472,
        -446534169874157203,
        -5597438459201014662,
        -7007119737006385570,
        7237741981002255125,
        -649171104678427962,
        -6981562940350531355,
    ]
    var d: Dict[DummyKey, Int] = {}
    for i, key in enumerate(keys):
        d[DummyKey(key)] = i
    assert_equal(len(d), len(keys))
    for i, key in enumerate(keys):
        assert_equal(i, d[DummyKey(key)])


def _test_taking_owned_kwargs_dict(var kwargs: OwnedKwargsDict[Int]):
    assert_equal(len(kwargs), 2)

    assert_true("fruit" in kwargs)
    assert_equal(kwargs["fruit"], 8)
    assert_equal(kwargs["fruit"], 8)

    assert_true("dessert" in kwargs)
    assert_equal(kwargs["dessert"], 9)
    assert_equal(kwargs["dessert"], 9)

    var keys = String()
    for key in kwargs.keys():
        keys += key
    assert_equal(keys, "fruitdessert")

    var sum = 0
    for val in kwargs.values():
        sum += val
    assert_equal(sum, 17)

    assert_false(kwargs.find("salad").__bool__())
    with assert_raises(contains="KeyError"):
        _ = kwargs["salad"]

    kwargs["salad"] = 10
    assert_equal(kwargs["salad"], 10)

    assert_equal(kwargs.pop("fruit"), 8)
    assert_equal(kwargs.pop("fruit", 2), 2)
    with assert_raises(contains="KeyError"):
        _ = kwargs.pop("fruit")

    keys = String()
    sum = 0
    for entry in kwargs.items():
        keys += entry.key
        sum += entry.value
    assert_equal(keys, "dessertsalad")
    assert_equal(sum, 19)


def test_owned_kwargs_dict():
    var owned_kwargs = OwnedKwargsDict[Int]()
    owned_kwargs._insert("fruit", 8)
    owned_kwargs._insert("dessert", 9)
    _test_taking_owned_kwargs_dict(owned_kwargs^)


def test_find_get():
    var some_dict: Dict[String, Int] = {}
    some_dict["key"] = 1
    assert_equal(some_dict.find("key").value(), 1)
    assert_equal(some_dict.get("key").value(), 1)
    assert_equal(some_dict.find("not_key").or_else(0), 0)
    assert_equal(some_dict.get("not_key", 0), 0)


def test_dict_popitem():
    var dict: Dict[String, Int] = {}
    dict["a"] = 1
    dict["b"] = 2

    assert_equal(len(dict), 2)
    var item = dict.popitem()
    assert_equal(item.key, "b")
    assert_equal(item.value, 2)
    assert_equal(len(dict), 1)
    item = dict.popitem()
    assert_equal(item.key, "a")
    assert_equal(item.value, 1)
    assert_equal(len(dict), 0)
    with assert_raises(contains="EmptyDictError"):
        _ = dict.popitem()


def test_pop_string_values():
    var dict: Dict[String, String] = {}
    dict["mojo"] = "lang"
    dict["max"] = "engine"
    dict["a"] = ""
    dict[""] = "a"

    assert_equal(dict.pop("mojo"), "lang")
    assert_equal(dict.pop("max"), "engine")
    assert_equal(dict.pop("a"), "")
    assert_equal(dict.pop(""), "a")
    with assert_raises(contains="KeyError"):
        _ = dict.pop("absent")


fn test_clear() raises:
    var some_dict: Dict[String, Int] = {}
    some_dict["key"] = 1
    some_dict.clear()
    assert_equal(len(some_dict), 0)
    assert_false(some_dict.get("key"))

    some_dict = Dict[String, Int]()
    some_dict.clear()
    assert_equal(len(some_dict), 0)


def test_init_initial_capacity():
    var initial_capacity = 16
    var x = Dict[Int, Int](capacity=initial_capacity)
    assert_equal(x._reserved(), initial_capacity)
    for i in range(initial_capacity):
        x[i] = i
    for i in range(initial_capacity):
        assert_equal(i, x[i])

    var y = Dict[Int, Int](capacity=64)
    assert_equal(y._reserved(), 64)

    # Non-power-of-two capacity is rounded up
    var z = Dict[Int, Int](capacity=50)
    assert_equal(z._reserved(), 64)

    # Small capacity is clamped to minimum (16)
    var w = Dict[Int, Int](capacity=3)
    assert_equal(w._reserved(), 16)


fn test_dict_setdefault() raises:
    var some_dict: Dict[String, Int] = {}
    some_dict["key1"] = 1
    some_dict["key2"] = 2
    assert_equal(some_dict.setdefault("key1", 0), 1)
    assert_equal(some_dict.setdefault("key2", 0), 2)
    assert_equal(some_dict.setdefault("not_key", 0), 0)
    assert_equal(some_dict["not_key"], 0)

    # Check that there is no copy of the default value, so it's performant
    var other_dict: Dict[String, CopyCounter[]] = {}
    var a = CopyCounter()
    var a_def = CopyCounter()
    var b_def = CopyCounter()
    other_dict["a"] = a^
    assert_equal(0, other_dict["a"].copy_count)
    _ = other_dict.setdefault("a", a_def^)
    _ = other_dict.setdefault("b", b_def^)
    assert_equal(0, other_dict["a"].copy_count)
    assert_equal(0, other_dict["b"].copy_count)


def test_compile_time_dict():
    comptime N = 10

    fn _get_dict() -> Dict[String, Int32, default_comp_time_hasher]:
        var res = Dict[String, Int32, default_comp_time_hasher]()
        for i in range(N):
            res[String(i)] = Int32(i)
        return res^

    comptime my_dict = _get_dict()

    comptime for i in range(N):
        comptime val = my_dict.get(String(i)).value()
        assert_equal(val, Int32(i))


# FIXME: Dictionaries should be equatable when their keys/values are.
def is_equal[
    K: KeyElement, V: Equatable & Copyable
](a: Dict[K, V], b: Dict[K, V]) -> Bool:
    if len(a) != len(b):
        return False
    for k in a.keys():
        if a[k] != b[k]:
            return False
    return True


def test_dict_comprehension():
    var d1 = {x: x * x for x in range(10) if x & 1}
    assert_true(is_equal(d1, {1: 1, 3: 9, 5: 25, 7: 49, 9: 81}))

    var s2 = {a * b: b for a in ["foo", "bar"] for b in [1, 2]}
    var d1reference = {
        "foo": 1,
        "bar": 1,
        "foofoo": 2,
        "barbar": 2,
    }
    assert_true(is_equal(s2, d1reference))


def test_dict_repr_wrap():
    var tmp_dict = {"one": 1.0, "two": 2.0}
    assert_equal(
        repr(tmp_dict),
        (
            "Dict[String, SIMD[DType.float64, 1]]"
            "({'one': SIMD[DType.float64, 1](1.0), "
            "'two': SIMD[DType.float64, 1](2.0)})"
        ),
    )


def test_popitem_no_copies():
    var dict: Dict[String, CopyCounter[]] = {}
    dict["a"] = CopyCounter()
    dict["b"] = CopyCounter()

    assert_equal(len(dict), 2)
    var item = dict.popitem()
    assert_equal(item.key, "b")
    assert_equal(item.value.copy_count, 0)
    assert_equal(len(dict), 1)
    item = dict.popitem()
    assert_equal(item.key, "a")
    assert_equal(item.value.copy_count, 0)
    assert_equal(len(dict), 0)
    with assert_raises(contains="EmptyDictError"):
        _ = dict.popitem()


def test_dict_key_error_repr():
    var e = DictKeyError[Int]()
    check_write_to(e, expected="DictKeyError[Int]()", is_repr=False)
    check_write_to(e, expected="DictKeyError[Int]()", is_repr=True)


def test_empty_dict_error_repr():
    var e = EmptyDictError()
    check_write_to(e, expected="EmptyDictError()", is_repr=False)
    check_write_to(e, expected="EmptyDictError()", is_repr=True)


def test_high_fill():
    """Fill a dict near its 7/8 load factor to exercise resize triggers."""
    var d = Dict[Int, Int]()
    # Insert enough to trigger multiple resizes (initial capacity is 16,
    # 7/8 load factor means resize at 14, then at 28, 56, 112, etc.)
    for i in range(200):
        d[i] = i * 2
    assert_equal(len(d), 200)
    # Verify all entries survived resizes
    for i in range(200):
        assert_equal(d[i], i * 2)


def test_tombstone_accumulation():
    """Repeatedly insert and delete to accumulate tombstones without resize."""
    var d = Dict[Int, Int]()
    # Pre-fill with 10 entries
    for i in range(10):
        d[i] = i
    # Insert and delete many transient entries to create tombstones
    for i in range(100, 500):
        d[i] = i
        _ = d.pop(i)
    # Original entries must still be found correctly despite tombstones
    assert_equal(len(d), 10)
    for i in range(10):
        assert_equal(d[i], i)


def test_ctrl_mirroring_boundary():
    """Keys landing near the end of the ctrl array exercise mirror bytes."""
    var d = Dict[Int, Int](capacity=16)
    # Insert keys that, when hashed, are likely to probe at positions
    # near the end of the 16-slot table (positions 14, 15) where SIMD
    # loads read into the mirror region.
    for i in range(16):
        d[i] = i
    # All entries must be findable
    for i in range(16):
        assert_equal(d[i], i)
    # Delete some near-boundary entries and re-verify
    _ = d.pop(14)
    _ = d.pop(15)
    assert_false(14 in d)
    assert_false(15 in d)
    for i in range(14):
        assert_equal(d[i], i)


def test_delete_and_relookup():
    """Delete entries then look them up to ensure correct miss detection."""
    var d = Dict[Int, Int]()
    for i in range(50):
        d[i] = i
    # Delete every other entry
    for i in range(0, 50, 2):
        _ = d.pop(i)
    assert_equal(len(d), 25)
    # Deleted keys must not be found
    for i in range(0, 50, 2):
        assert_false(i in d)
    # Remaining keys must still be found
    for i in range(1, 50, 2):
        assert_equal(d[i], i)


def test_order_preserved_after_heavy_deletion():
    """Insertion order is preserved even after many deletions."""
    var d = Dict[Int, Int]()
    for i in range(20):
        d[i] = i
    # Delete first 10
    for i in range(10):
        _ = d.pop(i)
    # Iteration should yield 10..19 in insertion order
    var idx = 10
    for k in d:
        assert_equal(k, idx)
        idx += 1
    assert_equal(idx, 20)


def test_order_compaction():
    """The _order array is compacted when it has too many stale entries."""
    # Use a large initial capacity so inserts+deletes don't trigger resize,
    # allowing stale entries to accumulate in _order until compaction fires.
    var d = Dict[Int, Int](capacity=1024)
    # Insert 100 entries (well under 7/8 * 1024 = 896)
    for i in range(100):
        d[i] = i
    # Delete 90, leaving 10 live entries but 100 stale _order entries
    for i in range(90):
        _ = d.pop(i)
    assert_equal(len(d), 10)
    # Now insert new entries. Each insert calls _maybe_resize which checks
    # compaction (len(_order) > 2 * _len). With 100 order entries and 10
    # live, compaction should trigger on the next insert.
    d[1000] = 1000
    assert_equal(len(d), 11)
    # Verify all live entries are intact and iteration order is correct
    for i in range(90, 100):
        assert_equal(d[i], i)
    assert_equal(d[1000], 1000)
    # Verify iteration yields entries in insertion order
    var keys = List[Int]()
    for k in d:
        keys.append(k)
    # Should be 90..99 (surviving originals) then 1000
    for i in range(10):
        assert_equal(keys[i], 90 + i)
    assert_equal(keys[10], 1000)


def test_reversed_items():
    """Reversed item iteration must use _order, not _capacity."""
    # Fresh dict: reversed items should be reverse insertion order
    var d = Dict[String, Int]()
    d["a"] = 1
    d["b"] = 2
    d["c"] = 3
    var keys = List[String]()
    var vals = List[Int]()
    for item in reversed(d.items()):
        keys.append(item.key)
        vals.append(item.value)
    assert_equal(len(keys), 3)
    assert_equal(keys[0], "c")
    assert_equal(keys[1], "b")
    assert_equal(keys[2], "a")
    assert_equal(vals[0], 3)
    assert_equal(vals[1], 2)
    assert_equal(vals[2], 1)

    # After deletions: stale _order entries skipped in reverse
    _ = d.pop("b")
    keys = List[String]()
    for item in reversed(d.items()):
        keys.append(item.key)
    assert_equal(len(keys), 2)
    assert_equal(keys[0], "c")
    assert_equal(keys[1], "a")

    # After delete + re-insert: new entries appear at end of _order
    d["x"] = 10
    keys = List[String]()
    vals = List[Int]()
    for item in reversed(d.items()):
        keys.append(item.key)
        vals.append(item.value)
    assert_equal(len(keys), 3)
    assert_equal(keys[0], "x")
    assert_equal(keys[1], "c")
    assert_equal(keys[2], "a")
    assert_equal(vals[0], 10)
    assert_equal(vals[1], 3)
    assert_equal(vals[2], 1)

    # Empty dict: reversed items yields nothing
    var empty = Dict[String, Int]()
    var count = 0
    for _ in reversed(empty.items()):
        count += 1
    assert_equal(count, 0)


def test_minimum_capacity():
    """The minimum capacity is _GROUP_WIDTH (16) for SIMD correctness."""
    var d = Dict[Int, Int](capacity=16)
    assert_true(d._capacity >= _GROUP_WIDTH)
    # Default constructor also gets at least _GROUP_WIDTH capacity
    var d2 = Dict[Int, Int]()
    assert_true(d2._capacity >= _GROUP_WIDTH)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
