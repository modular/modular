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

from testing import *


fn all_equal[
    T: Equatable & Copyable
](ref lhs: List[T], ref rhs: List[T]) -> Bool:
    if len(lhs) != len(rhs):
        return False

    for left, right in zip(lhs, rhs):
        if left != right:
            return False
    return True


@fieldwise_init
struct NotEquatable(Copyable):
    var value: Int

    fn __eq__(self, other: NotEquatable) -> Bool:
        return self.value == other.value


comptime CanCompare = Equatable & ImplicitlyCopyable


@fieldwise_init
struct CanBeEquatable[T: CanCompare](CanCompare):
    var value: Self.T

    fn __eq__(self, other: CanBeEquatable[Self.T]) -> Bool:
        return self.value == other.value


comptime ComparableValue = Equatable & ImplicitlyCopyable


@fieldwise_init
struct Pair[T: ComparableValue](ComparableValue):
    var left: Self.T
    var right: Self.T

    fn __eq__(self, other: Pair[Self.T]) -> Bool:
        return self.left == other.left and self.right == other.right


fn main() raises:
    # Basic types
    print("Int (Expect True):\t", all_equal([1, 2, 3], [1, 2, 3]))  # True
    print("Int (Expect False):\t", all_equal([1, 2, 3], [4, 5, 6]))  # False
    print(
        "String (Expect True):\t",
        all_equal(["hello", "world"], ["hello", "world"]),
    )  # True
    print(
        "String (Expect False):\t",
        all_equal(["hello", "world"], ["goodbye", "world"]),
    )  # False

    # Bad practice: floating point equality
    print("Float (Expect True):\t", all_equal([9.8, 7.6], [9.8, 7.6]))
    # True, but bad practice, use math.is_close() instead

    # Another floating point sample, pathological case
    var big_num = 1.0e16
    var bigger_num = big_num + 1.0
    print(
        "Floating point (equal, pathological):\t",
        all_equal([big_num], [bigger_num]),
    )
    # True due to floating point precision

    # Custom type that declares `Equatable`
    print(
        "CanBeEquatable (Expect False):\t",
        all_equal(
            [CanBeEquatable(1), CanBeEquatable(2)],
            [CanBeEquatable(1), CanBeEquatable(3)],
        ),
    )  # False

    # Equal but can't be compiled and run. Does not explicitly conform
    # var l1: List[NotEquatable] = [NotEquatable(1), NotEquatable(2)]
    # var l2: List[NotEquatable] = [NotEquatable(1), NotEquatable(2)]
    # print(all_equal(l1, l2))

    # Tests
    assert_true(all_equal([1, 2, 3], [1, 2, 3]), "Lists should be equal")
    assert_false(all_equal([1, 2, 3], [4, 5, 6]), "Lists should not be equal")

    assert_true(all_equal([9.8, 7.6], [9.8, 7.6]), "Lists should be equal")
    assert_false(all_equal([9.8, 7.6], [1.2, 3.4]), "Lists should not be equal")

    assert_true(
        all_equal(["hello", "world"], ["hello", "world"]),
        "Lists should be equal",
    )
    assert_false(
        all_equal(["hello", "world"], ["goodbye", "world"]),
        "Lists should not be equal",
    )
    assert_true(
        all_equal(
            [CanBeEquatable(1), CanBeEquatable(2)],
            [CanBeEquatable(1), CanBeEquatable(2)],
        ),
        "Lists should be equal",
    )
    assert_true(
        all_equal([big_num], [bigger_num]),
        (
            "Lists should not be equal but will be equal due to floating point"
            " precision"
        ),
    )
    assert_false(
        all_equal(
            [CanBeEquatable(1), CanBeEquatable(2)],
            [CanBeEquatable(1), CanBeEquatable(3)],
        ),
        "Lists should not be equal",
    )
