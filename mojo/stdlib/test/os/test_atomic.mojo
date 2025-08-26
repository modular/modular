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

from os.atomic import Atomic, Consistency

from testing import assert_equal, assert_not_equal, assert_false, assert_true


def test_consistency_equality_comparable():
    var ordering = Consistency.SEQUENTIAL

    assert_not_equal(ordering, Consistency(42))
    assert_not_equal(ordering, Consistency.NOT_ATOMIC)
    assert_not_equal(ordering, Consistency.UNORDERED)
    assert_not_equal(ordering, Consistency.MONOTONIC)
    assert_not_equal(ordering, Consistency.ACQUIRE)
    assert_not_equal(ordering, Consistency.RELEASE)
    assert_not_equal(ordering, Consistency.ACQUIRE_RELEASE)
    assert_equal(ordering, Consistency.SEQUENTIAL)


def test_consistency_representable():
    assert_equal(repr(Consistency(42)), "Consistency.UNKNOWN")
    assert_equal(repr(Consistency.NOT_ATOMIC), "Consistency.NOT_ATOMIC")
    assert_equal(repr(Consistency.UNORDERED), "Consistency.UNORDERED")
    assert_equal(repr(Consistency.MONOTONIC), "Consistency.MONOTONIC")
    assert_equal(repr(Consistency.ACQUIRE), "Consistency.ACQUIRE")
    assert_equal(repr(Consistency.RELEASE), "Consistency.RELEASE")
    assert_equal(
        repr(Consistency.ACQUIRE_RELEASE), "Consistency.ACQUIRE_RELEASE"
    )
    assert_equal(repr(Consistency.SEQUENTIAL), "Consistency.SEQUENTIAL")


def test_consistency_stringable():
    assert_equal(String(Consistency(42)), "UNKNOWN")
    assert_equal(String(Consistency.NOT_ATOMIC), "NOT_ATOMIC")
    assert_equal(String(Consistency.UNORDERED), "UNORDERED")
    assert_equal(String(Consistency.MONOTONIC), "MONOTONIC")
    assert_equal(String(Consistency.ACQUIRE), "ACQUIRE")
    assert_equal(String(Consistency.RELEASE), "RELEASE")
    assert_equal(String(Consistency.ACQUIRE_RELEASE), "ACQUIRE_RELEASE")
    assert_equal(String(Consistency.SEQUENTIAL), "SEQUENTIAL")


fn test_atomic() raises:
    var atom = Atomic[DType.index](3)

    assert_equal(atom.load(), 3)

    assert_equal(atom.value, 3)

    atom += 4
    assert_equal(atom.value, 7)

    atom -= 4
    assert_equal(atom.value, 3)

    atom.max(0)
    assert_equal(atom.value, 3)

    atom.max(42)
    assert_equal(atom.value, 42)

    atom.min(3)
    assert_equal(atom.value, 3)

    atom.min(0)
    assert_equal(atom.value, 0)


fn test_atomic_floating_point() raises:
    var atom = Atomic(Float32(3.0))

    assert_equal(atom.value, 3.0)

    atom += 4
    assert_equal(atom.value, 7.0)

    atom -= 4
    assert_equal(atom.value, 3.0)

    atom.max(0)
    assert_equal(atom.value, 3.0)

    atom.max(42)
    assert_equal(atom.value, 42.0)

    atom.min(3)
    assert_equal(atom.value, 3.0)

    atom.min(0)
    assert_equal(atom.value, 0.0)


def test_compare_exchange_weak():
    var atom = Atomic[DType.int64](3)
    var expected = Int64(3)
    var desired = Int64(3)
    var ok = atom.compare_exchange_weak(expected, desired)

    assert_equal(expected, 3)
    assert_true(ok)

    expected = Int64(4)
    ok = atom.compare_exchange_weak(expected, desired)

    assert_equal(expected, 3)
    assert_false(ok)

    expected = Int64(4)
    desired = Int64(6)
    ok = atom.compare_exchange_weak(expected, desired)

    assert_equal(expected, 3)
    assert_false(ok)


def test_comptime_atomic():
    fn comptime_fn() -> Int:
        var atom = Atomic[DType.index](3)
        atom += 4
        atom -= 4
        return Int(atom.load())

    alias value = comptime_fn()
    assert_equal(value, 3)


def main():
    test_consistency_equality_comparable()
    test_consistency_representable()
    test_consistency_stringable()
    test_atomic()
    test_atomic_floating_point()
    test_compare_exchange_weak()
    test_comptime_atomic()
