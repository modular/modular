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
"""Implements the `Hashable` trait and `hash()` built-in function.

There are a few main tools in this module:

- `Hashable` trait for types implementing `__hash__(self) -> UInt`
- `hash[T: Hashable](hashable: T) -> Int` built-in function.
- A `hash()` implementation for arbitrary byte strings,
  `hash(data: UnsafePointer[mut=False, UInt8], n: Int) -> Int`,
  is the workhorse function, which implements efficient hashing via SIMD
  vectors. See the documentation of this function for more details on the hash
  implementation.
- `hash(SIMD)` and `hash(UInt8)` implementations
    These are useful helpers to specialize for the general bytes implementation.
"""

from builtin.constrained import _constrained_field_conforms_to
from memory import Span
from reflection import get_type_name, struct_field_names, struct_field_types

from .hasher import Hasher, default_hasher

# ===----------------------------------------------------------------------=== #
# Implementation
# ===----------------------------------------------------------------------=== #


trait Hashable:
    """A trait for types which specify a function to hash their data.

    This hash function will be used for applications like hash maps, and
    don't need to be cryptographically secure. A good hash function will
    hash similar / common types to different values, and in particular
    the _low order bits_ of the hash, which are used in smaller dictionaries,
    should be sensitive to any changes in the data structure. If your type's
    hash function doesn't meet this criteria it will get poor performance in
    common hash map implementations.

    The `Hashable` trait has a default implementation of `__hash__()` that uses
    reflection to hash all fields. This means simple structs can conform to
    `Hashable` without implementing any methods:

    ```mojo
    @fieldwise_init
    struct Point(Hashable):
        var x: Int
        var y: Int

    var p = Point(1, 2)
    print(hash(p))
    ```

    All fields must conform to `Hashable`. Override `__hash__()` for custom
    hashing behavior.

    Note: When implementing both `Hashable` and `Equatable`, ensure that
    equal values produce equal hashes (i.e., if `a == b` then `hash(a) == hash(b)`).
    The default implementations of both traits satisfy this property when all
    fields implement both traits correctly.
    """

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Accepts a hasher and contributes to the hash value
        by calling the update function of the hasher.

        Parameters:
            H: Any Hasher type.

        Args:
            hasher: The hasher instance to contribute to.
        """
        comptime names = struct_field_names[Self]()
        comptime types = struct_field_types[Self]()

        comptime for i in range(names.size):
            comptime T = types[i]
            _constrained_field_conforms_to[
                conforms_to(T, Hashable),
                Parent=Self,
                FieldIndex=i,
                ParentConformsTo="Hashable",
            ]()
            hasher.update(trait_downcast[Hashable](__struct_field_ref(i, self)))


fn hash[
    T: Hashable, HasherType: Hasher = default_hasher
](hashable: T) -> UInt64:
    """Hash a Hashable type using its underlying hash implementation.

    Parameters:
        T: Any Hashable type.
        HasherType: Type of the hasher which should be used for hashing.

    Args:
        hashable: The input data to hash.

    Returns:
        A 64-bit integer hash based on the underlying implementation.
    """
    var hasher = HasherType()
    hasher.update(hashable)
    var value = hasher^.finish()
    return value


fn hash[
    HasherType: Hasher = default_hasher
](bytes: UnsafePointer[mut=False, UInt8], n: Int) -> UInt64:
    """Hash a sequence of bytes using the specified hasher.

    Parameters:
        HasherType: Type of the hasher to use for hashing (default: `AHasher`).

    Args:
        bytes: Pointer to the byte sequence to hash.
        n: The number of bytes to hash.

    Returns:
        A 64-bit integer hash value.
    """
    var hasher = HasherType()
    hasher._update_with_bytes(Span(ptr=bytes, length=n))
    var value = hasher^.finish()
    return value
