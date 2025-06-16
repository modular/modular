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

"""Implements the [Fnv1a 64 bit variant](https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function) algorithm as a _Hasher type."""

from ._hasher import _Hasher, _HashableWithHasher
from memory import UnsafePointer


struct Fnv1a(Defaultable, _Hasher):
    """Fnv1a is a very simple algorithm with good quality, but sub optimal runtime for long inputs.
    It can be used for comp time hash value generation.

    References:

    - [Fnv1a 64 bit variant](https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function)
    """

    var _value: UInt64

    fn __init__(out self):
        """Initialize the hasher."""
        self._value = 0xCBF29CE484222325

    fn _update_with_bytes(
        mut self, data: UnsafePointer[UInt8, mut=False, **_], length: Int
    ):
        """Consume provided data to update the internal buffer.

        Args:
            data: Pointer to the byte array.
            length: The length of the byte array.
        """
        for i in range(length):
            self._value ^= data[i].cast[DType.uint64]()
            self._value *= 0x100000001B3

    fn _update_with_simd(mut self, value: SIMD[_, _]):
        """Update the buffer value with new data.

        Args:
            value: Value used for update.
        """

        # number of rounds a single vector value will contribute to a hash
        # values smaller than 8 bytes contribute only once
        # values which are multiple of 8 bytes contribute multiple times
        # e.g. int128 is 16 bytes long and evaluates to 2 rounds
        alias rounds = value.dtype.sizeof() // 8 + (
            value.dtype.sizeof() % 8 > 0
        )

        @parameter
        for i in range(value.size):
            var v = value[i]

            @parameter
            for r in range(rounds):
                var u64: UInt64

                @parameter
                if value.dtype.is_floating_point():
                    u64 = v.to_bits().cast[DType.uint64]()
                elif value.dtype.is_integral():
                    u64 = (v >> (r * 64)).cast[DType.uint64]()
                else:
                    u64 = v.cast[DType.uint64]()
                self._value ^= u64.cast[DType.uint64]()
                self._value *= 0x100000001B3

    fn update[T: _HashableWithHasher](mut self, value: T):
        """Update the buffer value with new hashable value.

        Parameters:
            T: Hashable type.

        Args:
            value: Value used for update.
        """
        value.__hash__(self)

    fn finish(owned self) -> UInt64:
        """Computes the hash value based on all the previously provided data.

        Returns:
            Final hash value.
        """
        return self._value
