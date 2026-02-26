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

from builtin.constrained import _constrained_field_conforms_to
from builtin.range import _ZeroStartingRange
from reflection import struct_field_names, struct_field_types


trait Equatable(ImplicitlyDestructible):
    """A type which can be compared for equality with other instances of itself.

    The `Equatable` trait has a default implementation of `__eq__()` that uses
    reflection to compare all fields. This means simple structs can conform to
    `Equatable` without implementing any methods:

    ```mojo
    @fieldwise_init
    struct Point(Equatable):
        var x: Int
        var y: Int

    var p1 = Point(1, 2)
    var p2 = Point(1, 2)
    print(p1 == p2)  # True
    ```

    All fields must conform to `Equatable`. Override `__eq__()` for custom
    equality semantics.

    Note: The default implementation performs memberwise equality comparison.
    This may not be appropriate for types containing floating-point fields
    (due to NaN semantics) or types requiring custom equality logic.
    """

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        """Define whether two instances of the object are equal to each other.

        The default implementation uses reflection to compare all fields for
        equality. All fields must conform to `Equatable`.

        Args:
            other: Another instance of the same type.

        Returns:
            True if the instances are equal according to the type's definition
            of equality, False otherwise.
        """

        # Default implementation using reflection: compare all fields
        comptime names = struct_field_names[Self]()
        comptime types = struct_field_types[Self]()

        comptime for i in range(names.size):
            comptime T = types[i]
            _constrained_field_conforms_to[
                conforms_to(T, Equatable),
                Parent=Self,
                FieldIndex=i,
                ParentConformsTo="Equatable",
            ]()
            if trait_downcast[Equatable](
                __struct_field_ref(i, self)
            ) != trait_downcast[Equatable](__struct_field_ref(i, other)):
                return False
        return True

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        """Define whether two instances of the object are not equal to each
        other.

        Args:
            other: Another instance of the same type.

        Returns:
            True if the instances are not equal according to the type's
            definition of equality, False otherwise.
        """
        return not self == other


trait Comparable(Equatable):
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
