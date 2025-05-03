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
from sys import bitwidthof
from math import ceildiv
from bit import pop_count, log2_floor

# ===-----------------------------------------------------------------------===#
# BitSet
# ===-----------------------------------------------------------------------===#

alias _WORD_BITS = bitwidthof[UInt64]()
alias _WORD_BITS_LOG2 = log2_floor(_WORD_BITS)


@always_inline
fn _word_index(idx: UInt) -> UInt:
    """Computes the 0-based index of the 64-bit word containing bit `idx`."""
    return Int(idx) >> _WORD_BITS_LOG2


@always_inline
fn _bit_mask(idx: UInt) -> UInt64:
    """Returns a UInt64 mask with only the bit corresponding to `idx` set."""
    return UInt64(1) << (idx & (_WORD_BITS - 1))


@value
struct BitSet[size: Int](Stringable, Writable, Boolable, Sized):
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

    alias _words_size = ceildiv(size, _WORD_BITS)
    var _words: InlineArray[UInt64, Self._words_size]  # Payload storage.
    var _size_bits: Int  # Highest observed bit index + 1.

    # --------------------------------------------------------------------- #
    # Constructors
    # --------------------------------------------------------------------- #

    fn __init__(out self):
        """Initializes an empty BitSet with zero capacity and size."""
        self._words = __type_of(self._words)(0)
        self._size_bits = 0

    fn __init__(out self, init: SIMD[DType.bool, size]):
        """Initializes a BitSet with the given SIMD vector of booleans.

        Args:
            init: A SIMD vector of booleans to initialize the bitset with.
        """
        self._words = __type_of(self._words)(0)
        self._size_bits = 0

        @parameter
        for i in range(size):
            if init[i]:
                self.set(i)

    # --------------------------------------------------------------------- #
    # Capacity queries
    # --------------------------------------------------------------------- #

    @always_inline
    fn __len__(self) -> Int:
        """Returns the logical size of the bitset.

        This is defined as the index of the highest bit ever set plus one.
        Returns 0 if the bitset is empty or has never had any bits set.

        Returns:
            The logical size (highest set bit index + 1).
        """
        return self._size_bits

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
        debug_assert(
            idx < size,
            String(
                "BitSet index out of bounds when setting bit: ",
                idx,
                " >= ",
                size,
            ),
        )
        var w = _word_index(idx)
        self._words[w] |= _bit_mask(idx)
        if idx + 1 > len(self):
            self._size_bits = idx + 1

    @always_inline
    fn clear(mut self, idx: UInt):
        """Clears the bit at the specified index `idx` (sets it to 0).

        Aborts if `idx` is negative or greater than or equal to the
        compile-time `size`. Does not change the logical size.

        Args:
            idx: The non-negative index of the bit to clear (must be < `size`).
        """
        debug_assert(
            idx < size,
            String(
                "BitSet index out of bounds when clearing bit: ",
                idx,
                " >= ",
                size,
            ),
        )
        var w = _word_index(idx)
        self._words[w] &= ~_bit_mask(idx)

    @always_inline
    fn toggle(mut self, idx: UInt):
        """Toggles (inverts) the bit at the specified index `idx`.

        If the bit becomes 1 and `idx` is greater than or equal to the
        current logical size, the logical size is updated. Aborts if `idx`
        is negative or greater than or equal to the compile-time `size`.

        Args:
            idx: The non-negative index of the bit to toggle (must be < `size`).
        """
        debug_assert(
            idx < size,
            String(
                "BitSet index out of bounds when toggling bit: ",
                idx,
                " >= ",
                size,
            ),
        )
        var w = _word_index(idx)
        var mask = _bit_mask(idx)
        self._words[w] ^= mask
        if (self._words[w] & mask) != 0 and idx + 1 > len(self):
            self._size_bits = idx + 1

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
        debug_assert(
            idx < size,
            String(
                "BitSet index out of bounds when testing bit: ",
                idx,
                " >= ",
                size,
            ),
        )
        var w = _word_index(idx)
        return (self._words[w] & _bit_mask(idx)) != 0

    fn clear_all(mut self):
        """Clears all bits in the set, resetting the logical size to 0.

        The allocated storage capacity remains unchanged. Equivalent to
        re-initializing the set with `Self()`.
        """
        self = Self()

    fn count(self) -> UInt:
        """Counts the total number of bits that are set to 1 in the bitset.

        Uses the efficient `pop_count` intrinsic for each underlying word.
        The complexity is proportional to the number of words used by the
        bitset's capacity (`_words_size`), not the logical size (`len`).

        Returns:
            The total count of set bits (population count).
        """
        var total: UInt = 0

        @parameter
        for i in range(self._words_size):
            # TODO (MSTDL-1485): remove the cast to Int
            total += UInt(Int(pop_count(self._words[i])))

        return total

    # --------------------------------------------------------------------- #
    # Representation helpers
    # --------------------------------------------------------------------- #

    @no_inline
    fn write_to[W: Writer, //](self, mut writer: W):
        """Writes a string representation of the set bits to the given writer.

        Outputs the indices of the set bits in ascending order, enclosed in
        curly braces and separated by commas (e.g., "{1, 5, 42}"). Only
        checks bits up to `len(self) - 1`.

        Parameters:
            W: The type of the writer, conforming to the `Writer` trait.

        Args:
            writer: The writer instance to output the representation to.
        """
        writer.write("{")
        var first = True
        for idx in range(len(self)):
            if not self.test(idx):
                continue
            if not first:
                writer.write(", ")
            writer.write(idx)
            first = False
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
