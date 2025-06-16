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

from bit import byte_swap, rotate_bits_left
from memory import bitcast

from ._hasher import _HashableWithHasher, _Hasher

alias U256 = SIMD[DType.uint64, 4]
alias U128 = SIMD[DType.uint64, 2]
alias MULTIPLE = 6364136223846793005
alias ROT = 23


@always_inline
fn _folded_multiply(lhs: UInt64, rhs: UInt64) -> UInt64:
    """A fast function to emulate a folded multiply of two 64 bit uints.

    Args:
        lhs: 64 bit uint.
        rhs: 64 bit uint.

    Returns:
        A value which is similar in its bitpattern to result of a folded multiply.
    """
    # Extend to 128 bits and multiply.
    m = lhs.cast[DType.uint128]() * rhs.cast[DType.uint128]()
    # Extract the high and low 64 bits.
    res = bitcast[DType.uint64, 2](m)
    return res[0] ^ res[1]


@always_inline
fn _read_small(data: UnsafePointer[UInt8, mut=False, **_], length: Int) -> U128:
    """Produce a `SIMD[DType.uint64, 2]` value from data which is smaller than or equal to `8` bytes.

    Args:
        data: Pointer to the byte array.
        length: The byte array length.

    Returns:
        Returns a SIMD[DType.uint64, 2] value.
    """
    if length >= 2:
        if length >= 4:
            # len 4-8
            var a = (
                data.bitcast[Scalar[DType.uint32]]().load().cast[DType.uint64]()
            )
            var b = (
                data.offset(length - 4)
                .bitcast[Scalar[DType.uint32]]()
                .load()
                .cast[DType.uint64]()
            )
            return U128(a, b)
        else:
            # len 2-3
            var a = (
                data.bitcast[Scalar[DType.uint16]]().load().cast[DType.uint64]()
            )
            var b = data.offset(length - 1).load().cast[DType.uint64]()
            return U128(a, b)
    else:
        # len 0-1
        if length > 0:
            var a = data.load().cast[DType.uint64]()
            return U128(a, a)
        else:
            return U128(0, 0)


struct AHasher[key: U256](Defaultable, _Hasher):
    """Adopted AHash algorithm which produces fast and high quality hash value by
    implementing `_Hasher` trait.

    References:

    - [AHasher Implementation in Rust](https://github.com/tkaitchuck/aHash)
    """

    var buffer: UInt64
    var pad: UInt64
    var extra_keys: U128

    fn __init__(out self):
        """Initialize the hasher."""
        alias pi_key = key ^ U256(
            0x243F_6A88_85A3_08D3,
            0x1319_8A2E_0370_7344,
            0xA409_3822_299F_31D0,
            0x082E_FA98_EC4E_6C89,
        )
        self.buffer = pi_key[0]
        self.pad = pi_key[1]
        self.extra_keys = U128(pi_key[2], pi_key[3])

    @always_inline
    fn _update(mut self, new_data: UInt64):
        """Update the buffer value with new data.

        Args:
            new_data: Value used for update.
        """
        self.buffer = _folded_multiply(new_data ^ self.buffer, MULTIPLE)

    @always_inline
    fn _large_update(mut self, new_data: U128):
        """Update the buffer value with new data.

        Args:
            new_data: Value used for update.
        """
        var xored = new_data ^ self.extra_keys
        var combined = _folded_multiply(xored[0], xored[1])
        self.buffer = rotate_bits_left[ROT]((self.buffer + self.pad) ^ combined)

    fn _update_with_bytes(
        mut self, data: UnsafePointer[UInt8, mut=False, **_], length: Int
    ):
        """Consume provided data to update the internal buffer.

        Args:
            data: Pointer to the byte array.
            length: The length of the byte array.
        """
        self.buffer = (self.buffer + length) * MULTIPLE
        if length > 8:
            if length > 16:
                var tail = (
                    data.offset(length - 16)
                    .bitcast[Scalar[DType.uint64]]()
                    .load[width=2]()
                )
                self._large_update(tail)
                var offset = 0
                while length - offset > 16:
                    var block = (
                        data.offset(offset)
                        .bitcast[Scalar[DType.uint64]]()
                        .load[width=2]()
                    )
                    self._large_update(block)
                    offset += 16
            else:
                var a = data.bitcast[Scalar[DType.uint64]]().load()
                var b = (
                    data.offset(length - 8)
                    .bitcast[Scalar[DType.uint64]]()
                    .load()
                )
                self._large_update(U128(a, b))
        else:
            var value = _read_small(data, length)
            self._large_update(value)

    fn _update_with_simd(mut self, new_data: SIMD[_, _]):
        """Update the buffer value with new data.

        Args:
            new_data: Value used for update.
        """

        # number of rounds a single vector value will contribute to a hash
        # values smaller than 8 bytes contribute only once
        # values which are multiple of 8 bytes contribute multiple times
        # e.g. int128 is 16 bytes long and evaluates to 2 rounds
        alias rounds = new_data.dtype.sizeof() // 8 + (
            new_data.dtype.sizeof() % 8 > 0
        )

        @parameter
        if rounds == 1:
            # vector values are not bigger than 8 bytes each
            var u64: SIMD[DType.uint64, new_data.size]

            @parameter
            if new_data.dtype.is_floating_point():
                u64 = new_data.to_bits().cast[DType.uint64]()
            else:
                u64 = new_data.cast[DType.uint64]()

            @parameter
            if u64.size == 1:
                self._update(u64[0])
            else:

                @parameter
                for i in range(0, u64.size, 2):
                    self._large_update(U128(u64[i], u64[i + 1]))
        else:
            # vector values will contribute to hash in multiple rounds
            @parameter
            for i in range(new_data.size):
                var v = new_data[i]
                constrained[v.dtype.sizeof() > 8 and v.dtype.is_integral()]()

                @parameter
                for r in range(0, rounds, 2):
                    var u64_1 = (v >> (r * 64)).cast[DType.uint64]()
                    var u64_2 = (v >> ((r + 1) * 64)).cast[DType.uint64]()
                    self._large_update(U128(u64_1, u64_2))

    fn update[T: _HashableWithHasher](mut self, value: T):
        """Update the buffer value with new hashable value.

        Args:
            value: Value used for update.
        """
        value.__hash__(self)

    @always_inline
    fn finish(owned self) -> UInt64:
        """Computes the hash value based on all the previously provided data.

        Returns:
            Final hash value.
        """
        var rot = self.buffer & 63
        var folded = _folded_multiply(self.buffer, self.pad)
        return (folded << rot) | (folded >> ((64 - rot) & 63))
