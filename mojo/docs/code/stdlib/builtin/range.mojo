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
# Not tested here (compile errors, traps, or a filed bug — can't be asserted
# at runtime):
#   - Unsigned reverse of the zero-start or sequential form:
#     reversed(range(UInt8(4))) is a compile-time error ("cannot reverse an
#     unsigned range"). The strided form does reverse unsigned (tested below).
#   - Float len(): len() of any float range is a compile-time error. No float
#     range conforms to Sized, and the strided __len__ also asserts an integral
#     dtype.
#   - Strided float indexing: a strided float range[i] is a compile-time error
#     (strided __getitem__ goes through the integral-only __len__). The
#     zero-start and sequential float forms DO index (tested below).
#   - Zero-start and sequential scalar len(): len(range(Int32(4))) doesn't
#     compile — those two structs don't conform to Sized (only the strided
#     scalar range does). Indexing and reversed() do work on them (tested below).
#   - Out-of-bounds index: r[999] traps under -D ASSERT=all.
#   - Scalar strided zero step: iterating can loop forever (the value never
#     advances toward end), so it can't be asserted. The Int strided form makes
#     a zero step an empty range (tested below).
#   - Strided float reverse: reversed(range(5.0, 0.0, -0.5)) compiles but drops
#     the boundary element (the __reversed__ shift-by-sign math assumes integer
#     steps). Filed as a stdlib bug; the strided float form is iteration-only,
#     so this is left unasserted.
from std.testing import assert_almost_equal, assert_equal


# ===-------------------------------------------------------------------=== #
# Integer ranges (Int / Indexer)
# ===-------------------------------------------------------------------=== #


def test_int_zero_based() raises:
    var seq = List[Int]()
    for i in range(5):
        seq.append(i)
    assert_equal(seq, [0, 1, 2, 3, 4])

    var steps = range(1000)
    assert_equal(steps[499], 499)
    assert_equal(len(steps), 1000)

    var rev = List[Int]()
    for i in reversed(range(5)):
        rev.append(i)
    assert_equal(rev, [4, 3, 2, 1, 0])

    # A negative end produces an empty range
    assert_equal(len(range(-5)), 0)


def test_int_sequential() raises:
    var fwd = List[Int]()
    for i in range(3, 7):
        fwd.append(i)
    assert_equal(fwd, [3, 4, 5, 6])

    var neg = List[Int]()
    for i in range(-3, 4):
        neg.append(i)
    assert_equal(neg, [-3, -2, -1, 0, 1, 2, 3])

    # end <= start is empty, not a countdown (covers both < and ==)
    assert_equal(len(range(7, 3)), 0)
    assert_equal(len(range(5, 5)), 0)


def test_int_strided() raises:
    var ups = List[Int]()
    for i in range(0, 10, 2):
        ups.append(i)
    assert_equal(ups, [0, 2, 4, 6, 8])

    var downs = List[Int]()
    for i in range(7, 3, -1):
        downs.append(i)
    assert_equal(downs, [7, 6, 5, 4])

    var evens = range(0, 2_000_000, 2)
    assert_equal(evens[999_999], 1_999_998)
    assert_equal(len(evens), 1_000_000)

    # A zero step is an empty range, not an error, for the Int strided form
    assert_equal(len(range(0, 10, 0)), 0)
    var empty = List[Int]()
    for i in range(0, 10, 0):
        empty.append(i)
    assert_equal(len(empty), 0)

    # The documented idiom for picking the step direction when you don't know
    # which bound is larger.
    var start = 0
    var stop = 5
    var up = List[Int]()
    for i in range(start, stop, 1 if stop > start else -1):
        up.append(i)
    assert_equal(up, [0, 1, 2, 3, 4])

    var down = List[Int]()
    for i in range(stop, start, 1 if start > stop else -1):
        down.append(i)
    assert_equal(down, [5, 4, 3, 2, 1])


# ===-------------------------------------------------------------------=== #
# Scalar ranges (Scalar[dtype])
# ===-------------------------------------------------------------------=== #


def test_scalar_zero_based() raises:
    # Typed elements
    var u = List[UInt8]()
    for i in range(UInt8(4)):
        u.append(i)
    assert_equal(u, [0, 1, 2, 3])

    # Float zero-start ranges iterate, index, and reverse (they just have no
    # builtin len()).
    var f = List[Float64]()
    for x in range(Float64(4.0)):
        f.append(x)
    assert_equal(f, [0.0, 1.0, 2.0, 3.0])
    assert_equal(range(Float64(4.0))[Float64(1.0)], 1.0)
    var frev = List[Float64]()
    for x in reversed(range(Float64(4.0))):
        frev.append(x)
    assert_equal(frev, [3.0, 2.0, 1.0, 0.0])

    # Integer scalar: indexing and reversed() (no builtin len() on this form)
    assert_equal(range(Int32(4))[Int32(2)], 2)
    var rev = List[Int32]()
    for x in reversed(range(Int32(4))):
        rev.append(x)
    assert_equal(rev, [3, 2, 1, 0])

    # A negative end produces an empty range (no len() here, so iterate)
    var neg = List[Int32]()
    for x in range(Int32(-5)):
        neg.append(x)
    assert_equal(len(neg), 0)


def test_scalar_sequential() raises:
    var got = List[Int32]()
    for i in range(Int32(3), Int32(7)):
        got.append(i)
    assert_equal(got, [3, 4, 5, 6])

    # Indexing and reversed() (no builtin len() on this form)
    assert_equal(range(Int32(3), Int32(7))[Int32(1)], 4)
    var rev = List[Int32]()
    for x in reversed(range(Int32(3), Int32(7))):
        rev.append(x)
    assert_equal(rev, [6, 5, 4, 3])

    # Float sequential ranges index and reverse too
    assert_equal(range(Float64(2.0), Float64(6.0))[Float64(1.0)], 3.0)
    var frev = List[Float64]()
    for x in reversed(range(Float64(2.0), Float64(6.0))):
        frev.append(x)
    assert_equal(frev, [5.0, 4.0, 3.0, 2.0])

    # end <= start is empty (no len() here, so iterate)
    var empty = List[Int32]()
    for x in range(Int32(7), Int32(3)):
        empty.append(x)
    assert_equal(len(empty), 0)


def test_scalar_strided_integer() raises:
    # len() and O(1) indexing work on the strided scalar range
    var r = range(Int32(0), Int32(20), Int32(2))
    assert_equal(len(r), 10)
    assert_equal(r[Int32(3)], 6)

    var rev = List[Int32]()
    for x in reversed(range(Int32(1), Int32(10), Int32(2))):
        rev.append(x)
    assert_equal(rev, [9, 7, 5, 3, 1])

    # The strided form supports len(), indexing, and reversed() for unsigned
    # ranges too — unlike the zero-start and sequential forms.
    var ur = range(UInt8(0), UInt8(8), UInt8(2))
    assert_equal(len(ur), 4)
    assert_equal(ur[UInt8(1)], 2)
    var urev = List[UInt8]()
    for x in reversed(range(UInt8(0), UInt8(8), UInt8(2))):
        urev.append(x)
    assert_equal(urev, [6, 4, 2, 0])


def test_scalar_strided_float() raises:
    # Endpoint is exclusive
    var open = List[Float64]()
    for t in range(Float64(0.0), Float64(1.0), Float64(0.25)):
        open.append(t)
    assert_equal(open, [0.0, 0.25, 0.5, 0.75])

    # Push end past the endpoint by a fraction of the step to include it
    var tolerance = Float64(0.1)
    var closed = List[Float64]()
    for t in range(Float64(0.0), Float64(1) + tolerance, Float64(0.25)):
        closed.append(t)
    assert_equal(closed, [0.0, 0.25, 0.5, 0.75, 1.0])

    # Steps in both directions
    var down = List[Float64]()
    for t in range(Float64(1.0), Float64(0.0), Float64(-0.25)):
        down.append(t)
    assert_equal(down, [1.0, 0.75, 0.5, 0.25])


def main() raises:
    test_int_zero_based()
    test_int_sequential()
    test_int_strided()
    test_scalar_zero_based()
    test_scalar_sequential()
    test_scalar_strided_integer()
    test_scalar_strided_float()
