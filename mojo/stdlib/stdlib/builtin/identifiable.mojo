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


trait Identifiable:
    """The Identifiable trait denotes a type with an identity
    which can be compared with other instances of itself.
    """

    fn __is__(self, rhs: Self) -> Bool:
        """Define whether `self` has the same identity as `rhs`.

        Args:
            rhs: The right hand side of the comparison.

        Returns:
            True if `self` is `rhs`.
        """
        ...

    fn __isnot__(self, rhs: Self) -> Bool:
        """Define whether `self` has a different identity than `rhs`.

        Args:
            rhs: The right hand side of the comparison.

        Returns:
            True if `self` is not `rhs`.
        """
        ...


# TODO(MSTDL-1580): Replace with compiler-provided type ID.
trait TypeIdentifiable:
    """Denotes a type that can be uniquely identified.

    This trait is intended to be usable for implementing "type map" based
    functionality.

    This type will eventually be replaced with a generic compiler interface."""

    alias TYPE_ID: StaticString
    """The unique identifier."""
