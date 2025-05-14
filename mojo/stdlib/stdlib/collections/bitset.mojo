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
"""Provides a compact, grow-only set of non-negative integers.

Optimized for space (1 bit per element) and speed (O(1) operations).
Offers set/clear/test/toggle and fast population count. The underlying
storage grows automatically but does not shrink unless `shrink_to_fit`
is called (not implemented yet).

Example:
```mojo
    var bs = BitSet[128]()      # 128-bit set, all clear
    bs.set(42)                  # Mark value 42 as present.
    if bs.test(42):             # Check membership.
        print("hit")            # Prints "hit".
    bs.clear(42)                # Remove 42.
    print(bs.count())           # Prints 0.
```
"""
# ---------------------------------------------------------------------------


from .inline_array import InlineArray
from sys import bitwidthof, simdwidthof
from math import ceildiv
from bit import pop_count, log2_floor
from algorithm import vectorize

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#

alias _WORD_BITS = bitwidthof[UInt64]()
alias _WORD_BITS_LOG2 = log2_floor(_WORD_BITS)


@always_inline
fn _word_index(idx: UInt) -> UInt:
    """Computes the 0-based index of the 64-bit word containing bit `idx`."""
    return Int(idx >> _WORD_BITS_LOG2)


@always_inline
fn _bit_mask(idx: UInt) -> UInt64:
    """Returns a UInt64 mask with only the bit corresponding to `idx` set."""
    return UInt64(1) << (idx & (_WORD_BITS - 1))


@always_inline
fn _check_index_bounds[operation_name: StaticString](idx: UInt, max_size: Int):
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


struct BitSet[size: UInt](
    Stringable, Writable, Boolable, Sized, Copyable, Movable
):
    """A grow-only set storing non-negative integers efficiently using bits.

    Parameters:
        size: The maximum number of bits the bitset can store.

    Each integer element is represented by a single bit within an array
    of 64-bit words (`UInt64`). This structure is optimized for:

    *   **Compactness:** Uses 64 times less memory than `List[Bool]`.
    *   **Speed:** Offers O(1) time complexity for `set`, `clear`, `test`,
        and `toggle` operations (single word load/store).

    Ideal for applications like data-flow analysis, graph algorithms, or
    any task requiring dense sets of small integers where memory and
    lookup speed are critical.
    """

    alias _words_size = max(1, ceildiv(size, _WORD_BITS))
    var _words: InlineArray[UInt64, Self._words_size]  # Payload storage.

    # --------------------------------------------------------------------- #
    # Constructors
    # --------------------------------------------------------------------- #

    fn __init__(out self):
        """Initializes an empty BitSet with zero capacity and size."""
        self._words = __type_of(self._words)(0)

    fn __init__(init: SIMD[DType.bool], out self: BitSet[UInt(init.size)]):
        """Initializes a BitSet with the given SIMD vector of booleans.

        Args:
            init: A SIMD vector of booleans to initialize the bitset with.
        """
        self._words = __type_of(self._words)(0)

        @parameter
        for i in range(Int(size)):
            if init[i]:
                self.set(i)

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
        var total: UInt = 0

        @parameter
        for i in range(Int(self._words_size)):
            total += UInt(pop_count(self._words.unsafe_get(i)))

        return total

    @always_inline
    fn is_empty(self) -> Bool:
        """Checks if the bitset contains any set bits.

        Equivalent to `len(self) == 0`. Note that this checks the logical
        size, not the allocated capacity.

        Returns:
            True if no bits are set (logical size is 0), False otherwise.
        """
        return len(self) == 0

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the bitset is non-empty (contains at least one set bit).

        Equivalent to `len(self) != 0` or `not self.is_empty()`.

        Returns:
            True if at least one bit is set, False otherwise.
        """
        return not self.is_empty()

    # --------------------------------------------------------------------- #
    # Bit manipulation
    # --------------------------------------------------------------------- #

    @always_inline
    fn set(mut self, idx: UInt):
        """Sets the bit at the specified index `idx` to 1.

        If `idx` is greater than or equal to the current logical size,
        the logical size is updated. Aborts if `idx` is negative or
        greater than or equal to the compile-time `size`.

        Args:
            idx: The non-negative index of the bit to set (must be < `size`).
        """
        _check_index_bounds["set"](idx, size)
        var w = _word_index(idx)
        self._words.unsafe_get(w) |= _bit_mask(idx)

    @always_inline
    fn clear(mut self, idx: UInt):
        """Clears the bit at the specified index `idx` (sets it to 0).

        Aborts if `idx` is negative or greater than or equal to the
        compile-time `size`. Does not change the logical size.

        Args:
            idx: The non-negative index of the bit to clear (must be < `size`).
        """
        _check_index_bounds["clearing"](idx, size)
        var w = _word_index(idx)
        self._words.unsafe_get(w) &= ~_bit_mask(idx)

    @always_inline
    fn toggle(mut self, idx: UInt):
        """Toggles (inverts) the bit at the specified index `idx`.

        If the bit becomes 1 and `idx` is greater than or equal to the
        current logical size, the logical size is updated. Aborts if `idx`
        is negative or greater than or equal to the compile-time `size`.

        Args:
            idx: The non-negative index of the bit to toggle (must be < `size`).
        """
        _check_index_bounds["toggling"](idx, size)
        var w = _word_index(idx)
        self._words.unsafe_get(w) ^= _bit_mask(idx)

    @always_inline
    fn test(self, idx: UInt) -> Bool:
        """Tests if the bit at the specified index `idx` is set (is 1).

        Aborts if `idx` is negative or greater than or equal to the
        compile-time `size`.

        Args:
            idx: The non-negative index of the bit to test (must be < `size`).

        Returns:
            True if the bit at `idx` is set, False otherwise.
        """
        _check_index_bounds["testing"](idx, size)
        var w = _word_index(idx)
        return (self._words.unsafe_get(w) & _bit_mask(idx)) != 0

    fn clear_all(mut self):
        """Clears all bits in the set, resetting the logical size to 0.

        The allocated storage capacity remains unchanged. Equivalent to
        re-initializing the set with `Self()`.
        """
        self = Self()

    # --------------------------------------------------------------------- #
    # Set operations
    # --------------------------------------------------------------------- #
    @always_inline
    @staticmethod
    fn _vectorize_apply[
        func: fn[simd_width: Int] (
            SIMD[DType.uint64, simd_width],
            SIMD[DType.uint64, simd_width],
        ) capturing -> SIMD[DType.uint64, simd_width],
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
        alias simd_width = simdwidthof[UInt64]()
        var res = Self()

        # Define a vectorized operation that processes multiple words at once
        @parameter
        @always_inline
        fn _intersect[simd_width: Int](offset: Int):
            # Initialize SIMD vectors to hold multiple words from each bitset
            var left_vec = SIMD[DType.uint64, simd_width]()
            var right_vec = SIMD[DType.uint64, simd_width]()

            # Load a batch of words from both bitsets into SIMD vectors
            @parameter
            for i in range(simd_width):
                left_vec[i] = left._words.unsafe_get(offset + i)
                right_vec[i] = right._words.unsafe_get(offset + i)

            # Apply the provided operation (union, intersection, etc.) to the
            # vectors
            var result_vec = func(left_vec, right_vec)

            # Store the results back into the result bitset
            @parameter
            for i in range(simd_width):
                res._words.unsafe_get(offset + i) = result_vec[i]

        # Choose between vectorized or scalar implementation based on word count
        @parameter
        if Self._words_size >= simd_width:
            # If we have enough words, use SIMD vectorization for better
            # performance
            vectorize[_intersect, simd_width, size = Self._words_size]()
        else:
            # For small bitsets, use a simple scalar implementation
            @parameter
            for i in range(Int(Self._words_size)):
                res._words.unsafe_get(i) = func(
                    left._words.unsafe_get(i),
                    right._words.unsafe_get(i),
                )

        return res

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
            left: SIMD[DType.uint64, simd_width],
            right: SIMD[DType.uint64, simd_width],
        ) -> SIMD[DType.uint64, simd_width]:
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
            left: SIMD[DType.uint64, simd_width],
            right: SIMD[DType.uint64, simd_width],
        ) -> SIMD[DType.uint64, simd_width]:
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
            left: SIMD[DType.uint64, simd_width],
            right: SIMD[DType.uint64, simd_width],
        ) -> SIMD[DType.uint64, simd_width]:
            return left & ~right

        return Self._vectorize_apply[_difference](self, other)

    # --------------------------------------------------------------------- #
    # Representation helpers
    # --------------------------------------------------------------------- #

    @no_inline
    fn write_to[W: Writer, //](self, mut writer: W):
        """Writes a string representation of the set bits to the given writer.
        Outputs the indices of the set bits in ascending order, enclosed in
        curly braces and separated by commas (e.g., "{1, 5, 42}"). Uses
        efficient bitwise operations to find set bits without iterating
        through every possible bit.

        Parameters:
            W: The type of the writer, conforming to the `Writer` trait.

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
                var abs_idx = bit_idx_base + bit_pos

                # Skip bits that would be beyond the maximum size
                if abs_idx >= size:
                    break

                if not first:
                    writer.write(", ")
                writer.write(abs_idx)
                first = False

                # Clear the rightmost bit
                word &= ~rightmost_bit

        writer.write("}")

    fn __repr__(self) -> String:
        """Returns a developer-friendly string representation of the bitset.

        Currently equivalent to `__str__`.

        Returns:
            A string showing the set bits (e.g., "{1, 5, 42}").
        """
        return String(self)

    fn __str__(self) -> String:
        """Returns a user-friendly string representation of the bitset.

        Formats the set bits as a comma-separated list within curly braces,
        like "{1, 5, 42}". Uses the `write_to` method internally.

        Returns:
            A string showing the set bits.
        """
        return String.write(self)
