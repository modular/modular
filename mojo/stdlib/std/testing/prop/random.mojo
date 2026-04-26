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
"""Implements random number generation for property-based testing."""

from std.random import random_ui64, seed
from ._errors import PLAYBACK_EXHAUSTED


struct Rng(Movable):
    """A seeded pseudo-random number generator.

    Users should not need to create this type directly, instead use the `Rng`
    value provided by the `Strategy` trait.
    """

    var history: List[UInt64]
    """The recorded history of values generated."""

    var playback_mode: Bool
    """Whether this RNG producer is currently in playback mode."""

    var playback_index: Int
    """The current index in the history during playback mode."""

    @doc_hidden
    def __init__(out self, var history: List[UInt64]):
        self.history = history^
        self.playback_mode = True
        self.playback_index = 0

    @doc_hidden
    def __init__(out self, *, seed: Int):
        # TODO: Figure out how to ensure this 'global' seed value is not
        # accidentally overwritten by the user in their test code.
        random.seed(seed)
        self.history = []
        self.playback_mode = False
        self.playback_index = 0

    @doc_hidden
    def _next(mut self, max: UInt64 = UInt64.MAX, out value: UInt64) raises:
        """If in playback mode, returns the next value in the history, otherwise
        generates a random value and records it.

        All random values are built on top of a random `UInt64` so we can record
        and keep a history of all generated values. When a test failure is
        encountered, this history is used to "shrink" the generated input values
        from Strategies by reordering, reducing, and removing the individual
        `UInt64` values in the history.

        Args:
            max: The maximum value.

        Returns:
            The next value in the history, or a random value if in live mode.

        Raises:
            If in playback mode and the history is exhausted.
        """
        if self.playback_mode:
            if self.playback_index >= len(self.history):
                raise materialize[PLAYBACK_EXHAUSTED]()
            value = self.history[self.playback_index]
            self.playback_index += 1
        else:
            value = random_ui64(0, max)
            self.history.append(value)

    def _xoshiro_float(mut self) raises -> Float64:
        """Returns a random `Float64` between `[0.0, 1.0]` using the Xoshiro
        algorithm.

        References:
            https://prng.di.unimi.it/#remarks
        """
        var uint64 = self._next()
        # C++ equivalent (uint64 >> 11) * 0x1.0p-53
        var float64 = Float64(uint64 >> 11) * (2.0**-53)
        return float64

    def rand_bool(
        mut self,
        *,
        true_probability: Float64 = 0.5,
    ) raises -> Bool:
        """Returns a random `Bool` with the given probability of being True.

        Args:
            true_probability: The probability of being `True` (between 0.0 and 1.0).

        Returns:
            A random `Bool`.

        Raises:
            If the underlying random number generator raises an error.
        """
        if true_probability < 0.0:
            return False
        if true_probability > 1.0:
            return True

        var percentage = self._xoshiro_float()
        return percentage > (1.0 - true_probability)

    # TODO: Revisit when we have a better random module.
    def rand_scalar[
        dtype: DType
    ](
        mut self,
        *,
        min: Scalar[dtype] = Scalar[dtype].MIN_FINITE,
        max: Scalar[dtype] = Scalar[dtype].MAX_FINITE,
    ) raises -> Scalar[dtype]:
        """Returns a random `Scalar` from the given range.

        Parameters:
            dtype: The `DType` of the scalar.

        Args:
            min: The minimum value.
            max: The maximum value.

        Returns:
            A random number in the range [min, max].

        Raises:
            If the minimum value is greater than the maximum value or if the
            underlying random number generator raises an error.
        """
        if min > max:
            raise Error("invalid min/max")

        if min == max:
            return min

        comptime if dtype == DType.bool:
            return rebind[Scalar[dtype]](Scalar[DType.bool](self.rand_bool()))
        elif dtype.is_integral():
            var offset = UInt64(0) - UInt64(Scalar[dtype].MIN)
            var a = UInt64(min) + offset
            var b = UInt64(max) + offset
            var diff = a - b if a > b else b - a
            var uint64 = self._next(diff)
            return Scalar[dtype](uint64) + min
        elif dtype.is_floating_point():
            # Shrink-monotonic decoding: smaller UInt64 must produce a float
            # with smaller (or equal) |result|, with u=0 producing 0.0. The
            # shrinker reduces the raw stream bits, so the encoding is what
            # makes shrinking converge to small magnitudes rather than to the
            # midpoint of [min, max].
            #
            # Layout: low bit is sign; upper 63 bits are a positive IEEE 754
            # Float64 bit pattern. For non-negative finite doubles, bit-pattern
            # ordering matches numeric ordering, so |result| is monotonic in
            # the upper 63 bits.
            var u = self._next()
            var sign_bit = u & 1
            var mag_bits = u >> 1

            # Bit patterns above +inf (0x7FF0000000000000) are NaN, which isn't
            # comparable and would break the clamp below; cap at +inf's pattern.
            # +inf itself is kept so callers who pass max=+inf can actually see
            # infinity; the final clamp removes it otherwise.
            comptime _POS_INF_F64_BITS = UInt64(0x7FF0000000000000)
            if mag_bits > _POS_INF_F64_BITS:
                mag_bits = _POS_INF_F64_BITS

            var magnitude = Float64(from_bits=mag_bits)
            var result = -magnitude if sign_bit else magnitude

            var min_f64 = Float64(min)
            var max_f64 = Float64(max)
            if result < min_f64:
                result = min_f64
            elif result > max_f64:
                result = max_f64

            return Scalar[dtype](result)
        else:
            comptime assert (
                False
            ), "rand_scalar expected bool, integral, or floating point"

    # TODO (MSTDL-1185): Can remove when UInt and SIMD are unified.
    def rand_uint(
        mut self,
        *,
        min: UInt = UInt.MIN,
        max: UInt = UInt.MAX,
    ) raises -> UInt:
        """Returns a random `UInt` from the given range.

        Args:
            min: The minimum value.
            max: The maximum value.

        Returns:
            A random `UInt` in the range [min, max].

        Raises:
            If the underlying random number generator raises an error.
        """
        return self.rand_scalar(min=min, max=max)

    # TODO (MSTDL-1185): Can remove when Int and SIMD are unified.
    def rand_int(
        mut self,
        *,
        min: Int = Int.MIN,
        max: Int = Int.MAX,
    ) raises -> Int:
        """Returns a random `Int` from the given range.

        Args:
            min: The minimum value.
            max: The maximum value.

        Returns:
            A random `Int` in the range [min, max].

        Raises:
            If the underlying random number generator raises an error.
        """
        return Int(
            self.rand_scalar[DType.int](
                min=Scalar[DType.int](min),
                max=Scalar[DType.int](max),
            )
        )
