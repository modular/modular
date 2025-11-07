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


trait Comparable(EqualityComparable):
    """A type which can be compared for order with other instances of itself.

    Implementers of this trait must define the `__lt__` and `__eq__` methods.

    The default implementations of the default comparison methods can be
    potentially inefficent for types where comparison is expensive. For such
    types, it is recommended to override all the default implementations.
    """

    fn __lt__(self, rhs: Self) -> Bool:
        """Define whether `self` is less than `rhs`.

        Args:
            rhs: The value to compare with.

        Returns:
            True if `self` is less than `rhs`.
        """
        ...

    @always_inline
    fn __gt__(self, rhs: Self) -> Bool:
        """Define whether `self` is greater than `rhs`.

        Args:
            rhs: The value to compare with.

        Returns:
            True if `self` is greater than `rhs`.
        """
        return rhs < self

    @always_inline
    fn __le__(self, rhs: Self) -> Bool:
        """Define whether `self` is less than or equal to `rhs`.

        Args:
            rhs: The value to compare with.

        Returns:
            True if `self` is less than or equal to `rhs`.
        """
        return not rhs < self

    @always_inline
    fn __ge__(self, rhs: Self) -> Bool:
        """Define whether `self` is greater than or equal to `rhs`.

        Args:
            rhs: The value to compare with.

        Returns:
            True if `self` is greater than or equal to `rhs`.
        """
        return not self < rhs
