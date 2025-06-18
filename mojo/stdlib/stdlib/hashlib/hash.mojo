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
"""Implements the `Hashable` trait and `hash()` built-in function.

There are a few main tools in this module:

- `Hashable` trait for types implementing `__hash__(self) -> UInt`
- `hash[T: Hashable](hashable: T) -> Int` built-in function.
- A `hash()` implementation for arbitrary byte strings,
  `hash(data: UnsafePointer[UInt8], n: Int) -> Int`,
  is the workhorse function, which implements efficient hashing via SIMD
  vectors. See the documentation of this function for more details on the hash
  implementation.
- `hash(SIMD)` and `hash(UInt8)` implementations
    These are useful helpers to specialize for the general bytes implementation.
"""

import ._djbx33a
from memory import UnsafePointer
from ._djbx33a import _hash_simd

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

    ```mojo
    @fieldwise_init
    struct Foo(Hashable):
        fn __hash__(self) -> UInt:
            return 4  # chosen by fair random dice roll

    var foo = Foo()
    print(hash(foo))
    ```
    """

    fn __hash__(self) -> UInt:
        """Return a 64-bit hash of the type's data.

        Returns:
            A 64-bit integer hash of this instance's data.
        """
        ...


fn hash[T: Hashable](hashable: T) -> UInt:
    """Hash a Hashable type using its underlying hash implementation.

    Parameters:
        T: Any Hashable type.

    Args:
        hashable: The input data to hash.

    Returns:
        A 64-bit integer hash based on the underlying implementation.
    """
    return hashable.__hash__()


fn hash(
    bytes: UnsafePointer[
        UInt8, address_space = AddressSpace.GENERIC, mut=False, **_
    ],
    n: Int,
) -> UInt:
    return _djbx33a.hash(bytes, n)
