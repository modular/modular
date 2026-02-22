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

from builtin.builtin_slice import ContiguousSlice, StridedSlice
from test_utils import check_write_to
from testing import assert_equal, assert_true, TestSuite


def test_none_end_folds():
    var all_def_slice = slice(0, None, 1)
    assert_equal(all_def_slice.start.value(), 0)
    assert_true(all_def_slice.end is None)
    assert_equal(all_def_slice.step.value(), 1)


# This requires parameter inference of StartT.
@fieldwise_init
struct FunnySlice(ImplicitlyCopyable):
    var start: Int
    var upper: String
    var stride: Float64
    var __slice_literal__: ()


@fieldwise_init
struct BoringSlice(ImplicitlyCopyable):
    var a: Int
    var b: Int
    var c: String
    var __slice_literal__: ()


struct Sliceable:
    fn __init__(out self):
        pass

    fn __getitem__(self, a: FunnySlice) -> FunnySlice:
        return a

    fn __getitem__(self, a: BoringSlice) -> BoringSlice:
        return a


def test_sliceable():
    var sliceable = Sliceable()

    var new_slice = sliceable[1:"hello":4.0]
    assert_equal(new_slice.start, 1)
    assert_equal(new_slice.upper, "hello")
    assert_equal(new_slice.stride, 4.0)

    var boring_slice = sliceable[1:2:"foo"]
    assert_equal(boring_slice.a, 1)
    assert_equal(boring_slice.b, 2)
    assert_equal(boring_slice.c, "foo")


struct SliceStringable:
    fn __init__(out self):
        pass

    fn __getitem__(self, a: Slice) -> String:
        return String(a)


def test_slice_stringable():
    var s = SliceStringable()
    assert_equal(s[2::-1], "slice(2, None, -1)")
    assert_equal(s[1:-1:2], "slice(1, -1, 2)")
    assert_equal(s[:-1], "slice(None, -1, None)")
    assert_equal(s[::], "slice(None, None, None)")
    assert_equal(s[::4], "slice(None, None, 4)")
    assert_equal(repr(slice(None, 2, 3)), "slice(None, 2, 3)")
    assert_equal(repr(slice(10)), "slice(None, 10, None)")


struct StridedSliceWritable:
    fn __init__(out self):
        pass

    fn __getitem__(self, a: StridedSlice) -> String:
        var s = String()
        a.write_to(s)
        return s


def test_strided_slice_write_to():
    var s = StridedSliceWritable()
    # Positive stride
    check_write_to(StridedSlice(1, 10, 2), expected="slice(1, 10, 2)", is_repr=False)
    check_write_to(StridedSlice(0, 5, 1), expected="slice(0, 5, 1)", is_repr=False)
    # Negative stride
    check_write_to(StridedSlice(2, None, -1), expected="slice(2, None, -1)", is_repr=False)
    check_write_to(StridedSlice(1, -1, 2), expected="slice(1, -1, 2)", is_repr=False)
    # None start or end
    check_write_to(StridedSlice(None, None, 3), expected="slice(None, None, 3)", is_repr=False)
    check_write_to(StridedSlice(None, 5, 2), expected="slice(None, 5, 2)", is_repr=False)
    check_write_to(StridedSlice(1, None, 2), expected="slice(1, None, 2)", is_repr=False)
    # Also verify via helper struct using slice literal syntax
    assert_equal(s[1:10:2], "slice(1, 10, 2)")
    assert_equal(s[2::-1], "slice(2, None, -1)")
    assert_equal(s[1:-1:2], "slice(1, -1, 2)")
    assert_equal(s[::3], "slice(None, None, 3)")


def test_strided_slice_write_repr_to():
    # repr == str for StridedSlice
    check_write_to(StridedSlice(1, 10, 2), expected="slice(1, 10, 2)", is_repr=True)
    check_write_to(StridedSlice(2, None, -1), expected="slice(2, None, -1)", is_repr=True)
    check_write_to(StridedSlice(None, None, 3), expected="slice(None, None, 3)", is_repr=True)


struct ContiguousSliceWritable:
    fn __init__(out self):
        pass

    fn __getitem__(self, a: ContiguousSlice) -> String:
        var s = String()
        a.write_to(s)
        return s


def test_contiguous_slice_write_to():
    var s = ContiguousSliceWritable()
    # Both bounds present
    check_write_to(ContiguousSlice(1, 5, None), expected="slice(1, 5, None)", is_repr=False)
    check_write_to(ContiguousSlice(0, 10, None), expected="slice(0, 10, None)", is_repr=False)
    # Only end
    check_write_to(ContiguousSlice(None, 3, None), expected="slice(None, 3, None)", is_repr=False)
    check_write_to(ContiguousSlice(None, 5, None), expected="slice(None, 5, None)", is_repr=False)
    # Only start
    check_write_to(ContiguousSlice(3, None, None), expected="slice(3, None, None)", is_repr=False)
    # Neither
    check_write_to(ContiguousSlice(None, None, None), expected="slice(None, None, None)", is_repr=False)
    # Also verify via helper struct using slice literal syntax
    assert_equal(s[1:5], "slice(1, 5, None)")
    assert_equal(s[:3], "slice(None, 3, None)")
    assert_equal(s[3:], "slice(3, None, None)")
    assert_equal(s[:], "slice(None, None, None)")


def test_contiguous_slice_write_repr_to():
    # repr == str for ContiguousSlice
    check_write_to(ContiguousSlice(1, 5, None), expected="slice(1, 5, None)", is_repr=True)
    check_write_to(ContiguousSlice(None, 3, None), expected="slice(None, 3, None)", is_repr=True)
    check_write_to(ContiguousSlice(3, None, None), expected="slice(3, None, None)", is_repr=True)
    check_write_to(ContiguousSlice(None, None, None), expected="slice(None, None, None)", is_repr=True)


def test_slice_eq():
    assert_equal(slice(1, 2, 3), slice(1, 2, 3))
    assert_equal(slice(None, 1, None), slice(1))
    assert_true(slice(2, 3) != slice(4, 5))
    assert_equal(slice(1, None, None), slice(1, None, None))
    assert_equal(slice(1, 2), slice(1, 2, None))


def test_slice_indices():
    var start: Int
    var end: Int
    var step: Int
    var s = slice(1, 10)
    start, end, step = s.indices(9)
    assert_equal(slice(start, end, step), slice(1, 9, 1))
    s = slice(1, None, 1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(1, 5, 1))
    s = slice(1, None, -1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(1, -1, -1))
    s = slice(-1, None, 1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(4, 5, 1))
    s = slice(None, 2, 1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(0, 2, 1))
    s = slice(None, 2, -1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(4, 2, -1))
    s = slice(0, -1, 1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(0, 4, 1))
    s = slice(None, None, 1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(0, 5, 1))
    s = slice(20)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(0, 5, 1))
    s = slice(10, -10, 1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(5, 0, 1))
    assert_equal(len(range(start, end, step)), 0)
    s = slice(-12, -10, -1)
    start, end, step = s.indices(5)
    assert_equal(slice(start, end, step), slice(-1, -1, -1))
    assert_equal(len(range(start, end, step)), 0)
    # TODO: Decide how to handle 0 step
    # s = slice(-10, -2, 0)
    # start, end, step = s.indices(5)
    # assert_equal(slice(start, end, step), slice(-1, 3, 0))
    # assert_equal(len(range(start, end, step)), 0)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
