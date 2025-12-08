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
from memory import Span

from ._ahash import AHasher
from ._fnv1a import Fnv1a

comptime default_hasher = AHasher[SIMD[DType.uint64, 4](0)]
"""The default hasher for runtime hashing."""

comptime default_comp_time_hasher = Fnv1a
"""The default hasher for compile-time hashing."""


trait Hasher:
    """A trait for types that can incrementally compute hash values.

    The `Hasher` trait defines the interface for hash algorithms that can process
    data in chunks and produce a final hash value.

    Implementers must provide methods to initialize the hasher, update it with
    data, and finalize the hash computation.
    """

    fn __init__(out self):
        """Initialize a new hasher instance."""
        ...

    fn _update_with_bytes(mut self, data: Span[Byte, _]):
        ...

    fn _update_with_simd(mut self, value: SIMD[_, _]):
        ...

    fn update[T: Hashable](mut self, value: T):
        """Update the hash with a value.

        Parameters:
            T: The type of value to hash, which must implement `Hashable`.

        Args:
            value: The value to incorporate into the hash.
        """
        ...

    fn finish(var self) -> UInt64:
        """Finalize the hash computation and return the hash value.

        Returns:
            The computed hash value as a 64-bit unsigned integer.
        """
        ...
