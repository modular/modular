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

comptime CanCompare = Equatable & ImplicitlyCopyable


fn all_equal[T: Copyable](lhs: List[T], rhs: List[T]) -> Bool:
    # Must be same length
    if len(lhs) != len(rhs):
        return False

    @parameter
    if conforms_to(T, Equatable):
        for left, right in zip(lhs, rhs):
            if trait_downcast[Equatable](left) != trait_downcast[Equatable](
                right
            ):
                return False
    else:
        return False

    # All elements matched, T is Equatable
    return True


@fieldwise_init
struct CanBeEquatable[T: CanCompare](CanCompare):
    var value: Self.T

    fn __eq__(self, other: CanBeEquatable[Self.T]) -> Bool:
        return self.value == other.value


fn main() raises:
    var list1 = [CanBeEquatable(1), CanBeEquatable(2)]
    var list2 = [CanBeEquatable(1), CanBeEquatable(2)]
    var list3 = [CanBeEquatable(1), CanBeEquatable(3)]

    print("CanBeEquatable (Expect False):\t", all_equal(list1, list3))  # False
    print("CanBeEquatable (Expect True):\t", all_equal(list1, list2))  # True

    assert_true(all_equal(list1, list2), "Lists should be equal")
    assert_false(all_equal(list1, list3), "Lists should not be equal")
