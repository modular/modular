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
"""Provides a compact, grow-only set of non-negative integers.

Optimized for space (1 bit per element) and speed (O(1) operations).
Offers set/clear/test/toggle and fast population count. The underlying
storage grows automatically but does not shrink unless `shrink_to_fit`
is called (not implemented yet).

Example:
```mojo
from collections import BitSet

var bs = BitSet[128]()  # 128-bit set, all clear
bs.set(42)              # Mark value 42 as present.
if bs.test(42):         # Check membership.
    print("hit")        # Prints "hit".
bs.clear(42)            # Remove 42.
print(len(bs))          # Prints 0.
```
"""
# ---------------------------------------------------------------------------


from math import ceildiv
from sys import simd_width_of, bit_width_of

from algorithm import vectorize
from bit import log2_floor, pop_count
from format._utils import FormatStruct
from memory import pack_bits

from .inline_array import InlineArray

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#

comptime _WORD_BITS = bit_width_of[DType.int64]()
comptime _WORD_BITS_LOG2 = log2_floor(_WORD_BITS)


@always_inline
fn _word_index(idx: Int) -> Int:
    """Computes the 0-based index of the 64-bit word containing bit `idx`."""
    return idx >> _WORD_BITS_LOG2


@always_inline
fn _bit_mask(idx: Int) -> Int:
    """Returns a Int64 mask with only the bit corresponding to `idx` set."""
    return 1 << (idx & _WORD_BITS - 1)


@always_inline
fn _check_index_bounds[operation_name: StaticString](idx: Int, max_size: Int):
    """Checks if the index is within bounds for a BitSet operation.

    Parameters:
        operation_name: The name of the operation for error reporting.

    Args:
        idx: The index to check.
        max_size: The maximum size of the BitSet.
    """
    debug_assert(
        idx < max_size,
        "BitSet index out of bounds when ",
        operation_name,
        " bit: ",
        idx,
        " >= ",
        max_size,
    )


# ===-----------------------------------------------------------------------===#
# BitSet
# ===-----------------------------------------------------------------------===#


struct BitSet[size: Int](Boolable, Copyable, Defaultable, Sized, Writable):
    """A grow-only set storing non-negative integers efficiently using bits.

    Parameters:
        size: The maximum number of bits the bitset can store.

    Each integer element is represented by a single bit within an array
    of 64-bit words (`Int64`). This structure is optimized for:

    *   **Compactness:** Uses 64 times less memory than `List[Bool]`.
    *   **Speed:** Offers O(1) time complexity for `set`, `clear`, `test`,
        and `toggle` operations (single word load/store).

    Ideal for applications like data-flow analysis, graph algorithms, or
    any task requiring dense sets of small integers where memory and
    lookup speed are critical.
    """

    comptime _words_size = max(1, ceildiv(Self.size, _WORD_BITS))
    var _words: InlineArray[Int64, Self._words_size]  # Payload storage.

    # --------------------------------------------------------------------- #
    # Constructors
    # --------------------------------------------------------------------- #

    fn __init__(out self):
        """Initializes an empty BitSet with zero capacity and size."""
        self._words = type_of(self._words)(fill=0)

    fn __init__(init: SIMD[DType.bool, _], out self: BitSet[init.size]):
        """Initializes a BitSet with the given SIMD vector of booleans.

        Args:
            init: A SIMD vector of booleans to initialize the bitset with.
        """
        comptime assert (
            max(init.size, _WORD_BITS) // _WORD_BITS == Self._words_size
        )
        self._words = type_of(self._words)(uninitialized=True)
        comptime step = min(init.size, _WORD_BITS)

        comptime for i in range(Self._words_size):
            self._words.unsafe_get(i) = pack_bits(
                init.slice[step, offset = i * step]()
            ).cast[DType.int64]()

    # --------------------------------------------------------------------- #
    # Capacity queries
    # --------------------------------------------------------------------- #

    @always_inline
    fn __len__(self) -> Int:
        """Counts the total number of bits that are set to 1 in the bitset.

        Uses the efficient `pop_count` intrinsic for each underlying word.
        The complexity is proportional to the number of words used by the
        bitset's capacity (`_words_size`), not the logical size (`len`).

        Returns:
            The total count of set bits (population count).
        """
        var total = 0

        comptime for i in range(self._words_size):
            total += Int(pop_count(self._words.unsafe_get(i)))

        return total

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the bitset is non-empty (contains at least one set bit).

        Returns:
            True if at least one bit is set, False otherwise.
        """
        return len(self) != 0

    # --------------------------------------------------------------------- #
    # Bit manipulation
    # --------------------------------------------------------------------- #

    @always_inline
    fn set(mut self, idx: Int):
        """Sets the bit at the specified index `idx` to 1.

        If `idx` is greater than or equal to the current logical size,
        the logical size is updated. Aborts if `idx` is negative or
        greater than or equal to the compile-time `size`.

        Args:
            idx: The non-negative index of the bit to set (must be < `size`).
        """
        _check_index_bounds["set"](idx, Self.size)
        var w = _word_index(idx)
        self._words.unsafe_get(w) |= Int64(_bit_mask(idx))

    @always_inline
    fn clear(mut self, idx: Int):
        """Clears the bit at the specified index `idx` (sets it to 0).

        Aborts if `idx` is negative or greater than or equal to the
        compile-time `size`. Does not change the logical size.

        Args:
            idx: The non-negative index of the bit to clear (must be < `size`).
        """
        _check_index_bounds["clearing"](idx, Self.size)
        var w = _word_index(idx)
        self._words.unsafe_get(w) &= Int64(~_bit_mask(idx))

    @always_inline
    fn toggle(mut self, idx: Int):
        """Toggles (inverts) the bit at the specified index `idx`.

        If the bit becomes 1 and `idx` is greater than or equal to the
        current logical size, the logical size is updated. Aborts if `idx`
        is negative or greater than or equal to the compile-time `size`.

        Args:
            idx: The non-negative index of the bit to toggle (must be < `size`).
        """
        _check_index_bounds["toggling"](idx, Self.size)
        var w = _word_index(idx)
        self._words.unsafe_get(w) ^= Int64(_bit_mask(idx))

    @always_inline
    fn test(self, idx: Int) -> Bool:
        """Tests if the bit at the specified index `idx` is set (is 1).

        Aborts if `idx` is negative or greater than or equal to the
        compile-time `size`.

        Args:
            idx: The non-negative index of the bit to test (must be < `size`).

        Returns:
            True if the bit at `idx` is set, False otherwise.
        """
        _check_index_bounds["testing"](idx, Self.size)
        var w = _word_index(idx)
        return (self._words.unsafe_get(w) & Int64(_bit_mask(idx))) != 0

    fn clear_all(mut self):
        """Clears all bits in the set, resetting the logical size to 0.

        The allocated storage capacity remains unchanged. Equivalent to
        re-initializing the set with `Self()`.
        """
        self = Self()

    fn toggle_all(mut self):
        """Toggles (inverts) all bits in the set up to the compile-time `size`.
        """

        comptime for i in range(self._words_size):
            self._words.unsafe_get(i) ^= ~0

    fn set_all(mut self):
        """Sets all bits in the set up to the compile-time `size`."""

        comptime for i in range(self._words_size):
            self._words.unsafe_get(i) = ~0

    # --------------------------------------------------------------------- #
    # Set operations
    # --------------------------------------------------------------------- #
    @always_inline
    @staticmethod
    fn _vectorize_apply[
        func: fn[simd_width: Int](
            SIMD[DType.int64, simd_width],
            SIMD[DType.int64, simd_width],
        ) capturing -> SIMD[DType.int64, simd_width],
    ](left: Self, right: Self) -> Self:
        """Applies a vectorized binary operation between two bitsets.

        This internal utility function optimizes set operations by processing
        multiple words in parallel using SIMD instructions when possible. It
        applies the provided function to corresponding words from both bitsets
        and returns a new bitset with the results.

        The vectorized operation is applied to each word in the bitsets but only
        if the number of words in the bitsets is greater than or equal to the
        SIMD width.

        Parameters:
            func: A function that takes two SIMD vectors of UInt64 values and
                returns a SIMD vector with the result of the operation. The
                function should implement the desired set operation (e.g.,
                union, intersection).

        Args:
            left: The first bitset operand.
            right: The second bitset operand.

        Returns:
            A new bitset containing the result of applying the function to each
            corresponding pair of words from the input bitsets.
        """
        comptime simd_width = simd_width_of[Int64]()
        var res = Self()

        # Define a vectorized operation that processes multiple words at once
        @always_inline
        fn _intersect[
            simd_width: Int
        ](offset: Int) unified {mut res, read left, read right}:
            # Initialize SIMD vectors to hold multiple words from each bitset
            var left_vec = SIMD[DType.int64, simd_width]()
            var right_vec = SIMD[DType.int64, simd_width]()

            # Load a batch of words from both bitsets into SIMD vectors
            comptime for i in range(simd_width):
                left_vec[i] = left._words.unsafe_get(offset + i)
                right_vec[i] = right._words.unsafe_get(offset + i)

            # Apply the provided operation (union, intersection, etc.) to the
            # vectors
            var result_vec = func(left_vec, right_vec)

            # Store the results back into the result bitset
            comptime for i in range(simd_width):
                res._words.unsafe_get(offset + i) = result_vec[i]

        # Choose between vectorized or scalar implementation based on word count
        comptime if Self._words_size >= simd_width:
            # If we have enough words, use SIMD vectorization for better
            # performance
            vectorize[simd_width](Self._words_size, _intersect)
        else:
            # For small bitsets, use a simple scalar implementation
            comptime for i in range(Self._words_size):
                res._words.unsafe_get(i) = func(
                    left._words.unsafe_get(i),
                    right._words.unsafe_get(i),
                )

        return res^

    fn union(self, other: Self) -> Self:
        """Returns a new bitset that is the union of `self` and `other`.

        Args:
            other: The bitset to union with.

        Returns:
            A new bitset containing all elements from both sets.
        """

        @parameter
        @always_inline
        fn _union[
            simd_width: Int
        ](
            left: SIMD[DType.int64, simd_width],
            right: SIMD[DType.int64, simd_width],
        ) -> SIMD[DType.int64, simd_width]:
            return left | right

        return Self._vectorize_apply[_union](self, other)

    fn intersection(self, other: Self) -> Self:
        """Returns a new bitset that is the intersection of `self` and `other`.

        Args:
            other: The bitset to intersect with.

        Returns:
            A new bitset containing only the elements present in both sets.
        """

        @parameter
        @always_inline
        fn _intersection[
            simd_width: Int
        ](
            left: SIMD[DType.int64, simd_width],
            right: SIMD[DType.int64, simd_width],
        ) -> SIMD[DType.int64, simd_width]:
            return left & right

        return Self._vectorize_apply[_intersection](self, other)

    fn difference(self, other: Self) -> Self:
        """Returns a new bitset that is the difference of `self` and `other`.

        Args:
            other: The bitset to subtract from `self`.

        Returns:
            A new bitset containing elements from `self` that are not in `other`.
        """

        @parameter
        @always_inline
        fn _difference[
            simd_width: Int
        ](
            left: SIMD[DType.int64, simd_width],
            right: SIMD[DType.int64, simd_width],
        ) -> SIMD[DType.int64, simd_width]:
            return left & ~right

        return Self._vectorize_apply[_difference](self, other)

    # --------------------------------------------------------------------- #
    # Representation helpers
    # --------------------------------------------------------------------- #

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of the set bits to the given writer.
        Outputs the indices of the set bits in ascending order, enclosed in
        curly braces and separated by commas (e.g., "{1, 5, 42}"). Uses
        efficient bitwise operations to find set bits without iterating
        through every possible bit.

        Args:
            writer: The writer instance to output the representation to.
        """
        writer.write("{")
        var first = True

        # Iterate through words rather than individual bits
        for word_idx in range(self._words_size):
            var word = self._words.unsafe_get(word_idx)

            # Skip empty words entirely
            if word == 0:
                continue

            var bit_idx_base = word_idx << _WORD_BITS_LOG2

            # Process bits efficiently using bit manipulation
            while word != 0:
                # Find position of rightmost 1-bit using binary tricks
                # (x & -x) isolates the rightmost 1-bit
                var rightmost_bit = word & (~word + 1)

                # Calculate the position of the bit within the word
                var bit_pos = pop_count(rightmost_bit - 1)

                # Write the absolute bit index
                var abs_idx = Int64(bit_idx_base) + bit_pos

                # Skip bits that would be beyond the maximum size
                if abs_idx >= Int64(Self.size):
                    break

                if not first:
                    writer.write(", ")
                writer.write(abs_idx)
                first = False

                # Clear the rightmost bit
                word &= ~rightmost_bit

        writer.write("}")

    @no_inline
    fn write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of the `BitSet` to the writer.

        Args:
            writer: The value to write to.
        """
        FormatStruct(writer, "BitSet").params(Self.size).fields(self)

    @deprecated("Representable is deprecated. Use Writable instead.")
    fn __repr__(self) -> String:
        """Returns a developer-friendly string representation of the bitset.

        Currently equivalent to `__str__`.

        Returns:
            A string showing the set bits (e.g., "{1, 5, 42}").
        """
        return String(self)

    @deprecated("Stringable is deprecated. Use Writable instead.")
    fn __str__(self) -> String:
        """Returns a user-friendly string representation of the bitset.

        Formats the set bits as a comma-separated list within curly braces,
        like "{1, 5, 42}". Uses the `write_to` method internally.

        Returns:
            A string showing the set bits.
        """
        return String.write(self)
