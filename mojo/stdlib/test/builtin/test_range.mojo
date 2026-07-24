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

from std.testing import assert_equal, assert_false, assert_true, TestSuite


comptime DTYPES = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
]


# Regression test for cyclic dependency bug in MSTDL-2217
# This helper must be declared before any test function that calls it.
# The bug was triggered when a function using range(Int, Int) was declared
# before main/test functions, causing a cyclic dependency during overload
# resolution: range -> Int -> Equatable.__eq__ -> range.
def _range_with_int_params_helper(start: Int, end: Int) -> Int:
    var sum = 0
    for i in range(start, end):
        sum += i
    return sum


def test_range_with_int_params_declaration_order() raises:
    assert_equal(_range_with_int_params_helper(0, 5), 10)  # 0+1+2+3+4
    assert_equal(_range_with_int_params_helper(1, 4), 6)  # 1+2+3
    assert_equal(_range_with_int_params_helper(5, 5), 0)  # empty range


def _test_range_iter_bounds[
    I: Iterator
](var range_iter: I, len: Int) raises where conforms_to(
    I.Element, ImplicitlyDeletable
):
    var iter = range_iter^

    for i in range(len):
        var lower, upper = iter.bounds()
        assert_equal(len - i, lower)
        assert_equal(len - i, upper.value())
        _ = iter.__next__()

    var lower, upper = iter.bounds()
    assert_equal(0, lower)
    assert_equal(0, upper.value())


def test_range_int_bounds() raises:
    _test_range_iter_bounds(range(0), 0)
    _test_range_iter_bounds(range(10), 10)
    _test_range_iter_bounds(range(0, 10), 10)
    _test_range_iter_bounds(range(5, 10), 5)
    _test_range_iter_bounds(range(10, 0, -1), 10)
    _test_range_iter_bounds(range(0, 10, 2), 5)
    _test_range_iter_bounds(range(0, 11, 2), 6)
    _test_range_iter_bounds(range(38, -13, -23), 3)


def test_range_uint_bounds() raises:
    _test_range_iter_bounds(range(UInt(0)), 0)
    _test_range_iter_bounds(range(UInt(10)), 10)
    _test_range_iter_bounds(range(UInt(0), UInt(10)), 10)
    _test_range_iter_bounds(range(UInt(5), UInt(10)), 5)
    _test_range_iter_bounds(range(UInt(0), UInt(10), UInt(2)), 5)
    _test_range_iter_bounds(range(UInt(0), UInt(11), UInt(2)), 6)


def _test_range_scalar_bounds[dtype: DType]() raises:
    comptime scalar = Scalar[dtype]

    _test_range_iter_bounds(range(scalar(0)), 0)
    _test_range_iter_bounds(range(scalar(10)), 10)
    _test_range_iter_bounds(range(scalar(0), scalar(10)), 10)
    _test_range_iter_bounds(range(scalar(5), scalar(10)), 5)
    _test_range_iter_bounds(range(scalar(0), scalar(10), scalar(2)), 5)
    _test_range_iter_bounds(range(scalar(0), scalar(11), scalar(2)), 6)

    comptime if dtype.is_signed():
        _test_range_iter_bounds(range(scalar(10), scalar(0), scalar(-1)), 10)
        _test_range_iter_bounds(range(scalar(38), scalar(-13), scalar(-23)), 3)


def test_range_scalar_bounds() raises:
    comptime for dtype in DTYPES:
        _test_range_scalar_bounds[dtype]()


def test_larger_than_int_max_bounds() raises:
    def test[I: Iterator](iter: I) raises:
        var lower, upper = iter.bounds()
        assert_equal(lower, Int.MAX)
        assert_false(upper)

    # UInt
    test(range(UInt.MAX))
    test(range(UInt(1), UInt.MAX))
    test(range(UInt(1), UInt.MAX, UInt(1)))

    # UInt64
    test(range(UInt64.MAX))
    test(range(UInt64(1), UInt64.MAX))
    test(range(UInt64(1), UInt64.MAX, UInt64(1)))


def test_range_len() raises:
    # Usual cases
    assert_equal(range(10).__len__(), 10, "len(range(10))")
    assert_equal(range(0, 10).__len__(), 10, "len(range(0, 10))")
    assert_equal(range(5, 10).__len__(), 5, "len(range(5, 10))")
    assert_equal(range(10, 0, -1).__len__(), 10, "len(range(10, 0, -1))")
    assert_equal(range(0, 10, 2).__len__(), 5, "len(range(0, 10, 2))")
    assert_equal(range(38, -13, -23).__len__(), 3, "len(range(38, -13, -23))")

    # Edge cases
    assert_equal(range(0).__len__(), 0, "len(range(0))")
    assert_equal(range(-10).__len__(), 0, "len(range(-10))")
    assert_equal(range(0, 0).__len__(), 0, "len(range(0, 0))")
    assert_equal(range(10, 0).__len__(), 0, "len(range(10, 0))")
    assert_equal(range(0, 0, 1).__len__(), 0, "len(range(0, 0, 1))")

    assert_equal(range(5, 10, -1).__len__(), 0, "len(range(5, 10, -1))")
    assert_equal(range(10, 5, 1).__len__(), 0, "len(range(10, 5, 1))")
    assert_equal(range(5, 10, -10).__len__(), 0, "len(range(5, 10, -10))")
    assert_equal(range(10, 5, 10).__len__(), 0, "len(range(10, 5, 10))")
    assert_equal(range(5, 10, 20).__len__(), 1, "len(range(5, 10, 20))")
    assert_equal(range(10, 5, -20).__len__(), 1, "len(range(10, 5, -20))")


def test_range_len_uint_maxuint() raises:
    assert_equal(
        range(UInt(0), UInt.MAX).__len__(), UInt.MAX, "len(range(0, UInt.MAX))"
    )
    assert_equal(
        range(UInt.MAX, UInt(0), UInt(1)).__len__(),
        0,
        "len(range(UInt.MAX, 0, 1))",
    )


def test_range_len_uint_empty() raises:
    assert_equal(
        range(UInt(0), UInt(0), UInt(1)).__len__(), 0, "len(range(0, 0, 1))"
    )
    assert_equal(
        range(UInt(10), UInt(10), UInt(1)).__len__(), 0, "len(range(10, 10, 1))"
    )


def test_range_len_uint() raises:
    assert_equal(range(UInt(10)).__len__(), 10, "len(range(10))")

    # start < end
    assert_equal(range(UInt(0), UInt(10)).__len__(), 10, "len(range(0, 10))")
    assert_equal(range(UInt(5), UInt(10)).__len__(), 5, "len(range(5, 10))")
    assert_equal(
        range(UInt(0), UInt(10), UInt(2)).__len__(), 5, "len(range(0, 10, 2))"
    )
    # start > end
    assert_equal(
        range(UInt(10), UInt(0), UInt(1)).__len__(), 0, "len(range(10, 0, 1))"
    )


def _test_range_len_scalar[dtype: DType]() raises:
    comptime scalar = Scalar[dtype]

    # empty
    assert_equal(range(scalar(0), scalar(0), scalar(1)).__len__(), 0)
    assert_equal(range(scalar(10), scalar(10), scalar(1)).__len__(), 0)

    # start = 0
    assert_equal(range(scalar(10)).__len__(), 10)

    # start < end
    assert_equal(range(scalar(0), scalar(10)).__len__(), 10)
    assert_equal(range(scalar(5), scalar(10)).__len__(), 5)
    assert_equal(range(scalar(0), scalar(10), scalar(2)).__len__(), 5)

    # start > end
    assert_equal(range(scalar(10), scalar(0), scalar(1)).__len__(), 0)


def test_range_len_scalar() raises:
    comptime for dtype in DTYPES:
        _test_range_len_scalar[dtype]()


def test_range_getitem() raises:
    # Usual cases
    assert_equal(range(10)[3], 3, "range(10)[3]")
    assert_equal(range(0, 10)[3], 3, "range(0, 10)[3]")
    assert_equal(range(5, 10)[3], 8, "range(5, 10)[3]")
    assert_equal(range(5, 10)[4], 9, "range(5, 10)[4]")
    assert_equal(range(10, 0, -1)[2], 8, "range(10, 0, -1)[2]")
    assert_equal(range(0, 10, 2)[4], 8, "range(0, 10, 2)[4]")
    assert_equal(range(38, -13, -23)[1], 15, "range(38, -13, -23)[1]")


def test_range_getitem_uint() raises:
    assert_equal(range(UInt(10))[3], 3, "range(10)[3]")

    assert_equal(range(UInt(0), UInt(10))[3], 3, "range(0, 10)[3]")
    assert_equal(range(UInt(5), UInt(10))[3], 8, "range(5, 10)[3]")
    assert_equal(range(UInt(5), UInt(10))[4], 9, "range(5, 10)[4]")

    # Specify the step size > 1
    assert_equal(range(UInt(0), UInt(10), UInt(2))[4], 8, "range(0, 10, 2)[4]")

    # start > end
    var bad_strided_uint_range = range(UInt(10), UInt(5), UInt(1))
    var bad_strided_uint_range_iter = bad_strided_uint_range.__iter__()
    assert_equal(UInt(0), UInt(bad_strided_uint_range_iter.__len__()))


def test_range_reversed() raises:
    # Zero starting
    assert_equal(
        range(10).__reversed__().start, 9, "range(10).__reversed__().start"
    )
    assert_equal(
        range(10).__reversed__().end, -1, "range(10).__reversed__().end"
    )
    assert_equal(
        range(10).__reversed__().step, -1, "range(10).__reversed__().step"
    )
    # Sequential
    assert_equal(
        range(5, 10).__reversed__().start, 9, "range(5,10).__reversed__().start"
    )
    assert_equal(
        range(5, 10).__reversed__().end, 4, "range(5,10).__reversed__().end"
    )
    assert_equal(
        range(5, 10).__reversed__().step, -1, "range(5,10).__reversed__().step"
    )
    # Strided
    assert_equal(
        range(38, -13, -23).__reversed__().start,
        -8,
        "range(38, -13, -23).__reversed__().start",
    )
    assert_equal(
        range(38, -13, -23).__reversed__().end,
        61,
        "range(38, -13, -23).__reversed__().end",
    )
    assert_equal(
        range(38, -13, -23).__reversed__().step,
        23,
        "range(38, -13, -23).__reversed__().step",
    )

    # Test a reversed range's sum and length compared to the original
    @parameter
    def test_sum_reversed(start: Int, end: Int, step: Int) raises:
        var forward = range(start, end, step)
        var iforward = forward.__iter__()
        var ibackward = forward.__reversed__()
        var backward = range(ibackward.start, ibackward.end, ibackward.step)
        assert_equal(
            forward.__len__(), backward.__len__(), "len(forward), len(backward)"
        )
        var forward_sum = 0
        var backward_sum = 0
        for _ in forward:
            forward_sum += iforward.__next__()
            backward_sum += ibackward.__next__()
        assert_equal(forward_sum, backward_sum, "forward_sum, backward_sum")

    # Test using loops and reversed
    for end in range(10, 13):
        test_sum_reversed(1, end, 3)

    for end in range(10, 13).__reversed__():
        test_sum_reversed(20, end, -3)


def test_range_reversed_float() raises:
    # `reversed()` must equal forward in reverse order, element-for-element,
    # exact even for steps not representable in binary.
    def assert_reversed_matches(
        start: Float64, end: Float64, step: Float64
    ) raises:
        var forward = List[Float64]()
        for x in range(start, end, step):
            forward.append(x)
        var backward = List[Float64]()
        for x in reversed(range(start, end, step)):
            backward.append(x)
        assert_equal(len(backward), len(forward))
        for i in range(len(forward)):
            assert_equal(backward[i], forward[len(forward) - 1 - i])

    # Exact-binary steps.
    assert_reversed_matches(5.0, 0.0, -0.5)  # fractional negative step
    assert_reversed_matches(0.0, 5.0, 0.5)  # ascending fractional step
    assert_reversed_matches(5.0, 0.6, -0.5)  # end not aligned to the grid
    assert_reversed_matches(0.0, 10.0, 3.0)  # step magnitude greater than one
    assert_reversed_matches(0.0, 5.0, -0.5)  # empty: step points the wrong way
    # Steps not exactly representable in binary.
    assert_reversed_matches(0.0, 1.0, 0.1)
    assert_reversed_matches(1.0, 0.0, -0.1)
    assert_reversed_matches(2.0, -1.0, -0.3)

    # Accumulation hazards: hundreds to thousands of steps.
    assert_reversed_matches(0.0, 100.0, 0.1)  # ~1000 steps
    assert_reversed_matches(-5.0, 5.0, 0.1)  # crosses zero, ~100 steps
    assert_reversed_matches(0.0, 10.0, 0.3)  # repr-nasty step
    assert_reversed_matches(10.0, 0.0, -0.7)  # descending repr-nasty step

    # Spot-check the exact reversed values for the originally reported case.
    var expected: List[Float64] = [
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
    ]
    var actual = List[Float64]()
    for x in reversed(range(5.0, 0.0, -0.5)):
        actual.append(x)
    assert_equal(actual, expected)


def test_range_float_forward_count() raises:
    # Non-representable step used to drift to 11 elements for [0, 1) by 0.1.
    var values = List[Float64]()
    for x in range(0.0, 1.0, 0.1):
        values.append(x)
    assert_equal(len(values), 10)
    assert_equal(values[0], 0.0)
    assert_equal(values[1], 0.1)


def test_range_float_zero_step() raises:
    # Zero step is empty both directions, not an infinite loop.
    var count = 0
    for _ in range(5.0, 0.0, 0.0):
        count += 1
    assert_equal(count, 0)
    var reverse_count = 0
    for _ in reversed(range(5.0, 0.0, 0.0)):
        reverse_count += 1
    assert_equal(reverse_count, 0)


def test_range_float_grid() raises:
    # On-grid `end` is excluded, no `// + 1` overcount (1.0 = 4 * 0.25).
    var v = List[Float64]()
    for x in range(0.0, 1.0, 0.25):
        v.append(x)
    assert_equal(v, [0.0, 0.25, 0.5, 0.75])


def test_range_float_empty() raises:
    # Wrong-direction ranges are empty for either step sign.
    var forward = 0
    for _ in range(5.0, 0.0, 0.5):  # positive step, end < start
        forward += 1
    assert_equal(forward, 0)
    var backward = 0
    for _ in range(0.0, 5.0, -0.5):  # negative step, end > start
        backward += 1
    assert_equal(backward, 0)


def test_indexing() raises:
    var r = range(10)
    assert_equal(r[Int(4)], 4)
    assert_equal(r[3], 3)


def test_range_bounds() raises:
    var start = 0
    var end = 10

    # verify loop iteration
    var r = range(start, end)
    var last_seen = -1
    for x in r:
        last_seen = x
    assert_equal(last_seen, end - 1)

    # verify index lookup
    var ln = r.__len__()
    assert_equal(r[ln - 1], last_seen)


def test_scalar_range() raises:
    r = range(UInt8(2), 16, 4)
    assert_equal(r.start, 2)
    assert_equal(r.end, 16)
    assert_equal(r.step, 4)

    def append_many[T: Copyable, //](mut list: List[T], *values: T):
        for value in values:
            list.append(value.copy())

    expected_elements = List[UInt8]()
    append_many(expected_elements, 2, 6, 10, 14)
    actual_elements = List[UInt8]()
    for e in r:
        actual_elements.append(UInt8(e))
    assert_equal(actual_elements, expected_elements)


def test_range_compile_time() raises:
    """Tests that verify compile-time parameter loops work correctly with
    various scalar types.
    """

    comptime for i in range(10):
        assert_true(i >= 0)

    comptime for i in reversed(range(10)):
        assert_true(i >= 0)

    comptime for i in range(UInt8(10)):
        assert_true(i >= 0)

    comptime for i in range(Int32(10)):
        assert_true(i >= 0)

    comptime for i in range(UInt16(1), 10, 2):
        assert_true(i >= 0)

    comptime for i in range(Int16(1), 10, 2):
        assert_true(i >= 0)

    comptime for i in reversed(range(Int16(1), 10, 2)):
        assert_true(i >= 0)

    comptime for i in range(Int64(10), 1, -2):
        assert_true(i > 0)
        assert_true(i <= 10)


def test_range_iterable() raises:
    var ai = 0
    var bi = UInt8(0)
    var ci = 0
    for a, b, c in zip(range(0, 10), range(UInt8(10)), range(0, 20, 2)):
        assert_equal(a, ai)
        assert_equal(b, bi)
        assert_equal(c, ci)
        ai += 1
        bi += 1
        ci += 2


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
