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
"""Provides a seeded pseudo-random generator for reproducible test inputs."""

from std.sys import bit_width_of
from std.utils.numerics import FPUtils


struct SplitMix64:
    """A splitmix64 pseudo-random generator.

    It gives every platform the same sequence for a seed, so a failing input
    reproduces anywhere.
    """

    var state: UInt64
    """The generator state."""

    def __init__(out self, seed: UInt64):
        """Creates a generator.

        Args:
            seed: The initial state.
        """
        self.state = seed

    def next(mut self) -> UInt64:
        """Advances the generator.

        Returns:
            The next 64 random bits.
        """
        self.state += 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def bits[
        dtype: DType
    ](mut self) -> Scalar[dtype] where dtype.is_floating_point():
        """Returns a value with a uniformly random bit pattern.

        Parameters:
            dtype: The floating-point type to return.

        Returns:
            A random `dtype` value, which may be a NaN, an infinity or a
            subnormal.
        """
        comptime nbits = bit_width_of[dtype]()
        var v = self.next() >> UInt64(64 - nbits)
        return Scalar[dtype](from_bits=v.cast[FPUtils[dtype].integral_type]())

    def uniform(mut self) -> Float64:
        """Returns a uniformly random value in [0, 1).

        Returns:
            A float64 with 53 random bits.
        """
        return Float64(self.next() >> 11) / Float64(1 << 53)


def xorshift64(mut state: UInt64) -> UInt64:
    """Advances a xorshift64 generator.

    Args:
        state: The generator state, which must not be zero.

    Returns:
        The new state, which is also the next 64 random bits.
    """
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return state
