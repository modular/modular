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

from os.atomic import Atomic, Consistency, fence

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


def test_atomic[dtype: DType]():
    alias scalar = Scalar[dtype]
    var atom = Atomic[dtype](3)

    assert_equal(atom.load(), scalar(3))

    assert_equal(atom.value, scalar(3))

    atom += scalar(4)
    assert_equal(atom.value, scalar(7))

    atom -= scalar(4)
    assert_equal(atom.value, scalar(3))

    atom.max(scalar(0))
    assert_equal(atom.value, scalar(3))

    atom.max(scalar(42))
    assert_equal(atom.value, scalar(42))

    atom.min(scalar(3))
    assert_equal(atom.value, scalar(3))

    atom.min(scalar(0))
    assert_equal(atom.value, scalar(0))


def test_compare_exchange[dtype: DType]():
    alias scalar = Scalar[dtype]
    var atom = Atomic[dtype](3)

    # Successful cmpxchg
    var expected = scalar(3)
    var success = atom.compare_exchange(expected, scalar(42))

    assert_true(success)
    assert_equal(expected, scalar(3))
    assert_equal(atom.load(), scalar(42))

    # Failure cmpxchg
    expected = scalar(3)
    var failure = atom.compare_exchange(expected, scalar(99))

    assert_false(failure)
    assert_equal(expected, scalar(42))
    assert_equal(atom.load(), scalar(42))


def test_comptime_atomic():
    fn comptime_fn() -> Int:
        var atom = Atomic[DType.index](3)
        atom += 4
        atom -= 4
        return Int(atom.load())

    alias value = comptime_fn()
    assert_equal(value, 3)


def test_comptime_fence():
    fn comptime_fn() -> Int:
        fence()
        return 1

    alias value = comptime_fn()
    assert_equal(value, 1)


def test_comptime_compare_exchange():
    fn comptime_fn(expected_in: Int32) -> Tuple[Bool, Int32, Int32]:
        var expected = expected_in
        var atom = Atomic[DType.int32](0)
        var success = atom.compare_exchange(expected, 42)
        return (success, expected, atom.load())

    alias result_success = comptime_fn(0)
    assert_true(result_success[0])
    assert_equal(result_success[1], 0)
    assert_equal(result_success[2], 42)

    alias result_failure = comptime_fn(1)
    assert_false(result_failure[0])
    assert_equal(result_failure[1], 0)
    assert_equal(result_failure[2], 0)


def main():
    test_consistency_equality_comparable()
    test_consistency_representable()
    test_consistency_stringable()
    test_atomic[DType.int32]()
    test_atomic[DType.float64]()
    test_compare_exchange[DType.int32]()
    test_compare_exchange[DType.float64]()
    test_comptime_atomic()
    test_comptime_fence()
    test_comptime_compare_exchange()
