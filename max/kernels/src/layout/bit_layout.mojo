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
"""Defines `BitLayout`, a layout that is linear over F2.

A layout whose extents are powers of two is linear over F2: each domain bit
goes to a fixed set of codomain bits, and a point goes to the XOR of the
images of its set bits. Strided modes and swizzles are both linear in this
sense, so one `BitLayout` describes either. Points are `UInt64`s in whatever
encoding the domain and codomain spaces define.

You can import these APIs from the `layout.bit_layout` module:

```mojo
from layout.bit_layout import BitLayout, placement

comptime pairs = BitLayout().run(0, 1, 0).run(16, 5, 1)
```
"""

from std.bit import count_trailing_zeros

from layout.layout import TensorLayout

comptime BIT_LAYOUT_BITS = 64
"""The number of domain bits a `BitLayout` can map."""


@always_inline
def bit(i: Int) -> UInt64:
    """Returns the point with only bit `i` set.

    Args:
        i: The bit index.

    Returns:
        `1 << i`.
    """
    return UInt64(1) << UInt64(i)


@always_inline
def bits(lo: Int, n: Int) -> UInt64:
    """Returns the mask of bits `lo` through `lo + n - 1`.

    Args:
        lo: The lowest bit.
        n: The number of bits.

    Returns:
        The mask.
    """
    return ((UInt64(1) << UInt64(n)) - 1) << UInt64(lo)


@fieldwise_init
struct BitLayout(Equatable, TrivialRegisterPassable, Writable):
    """A linear map over F2: domain bit `i` goes to `img[i]`.

    `dom` marks the domain bits the map has. A bit outside it does not
    address the codomain at all. `dead` marks domain bits whose points
    hold nothing, such as lanes an instruction leaves idle; their images are
    zero.
    """

    var img: SIMD[DType.uint64, BIT_LAYOUT_BITS]
    """The image of each domain bit; zero outside `dom` and on `dead`."""

    var dom: UInt64
    """The domain bits the map has."""

    var dead: UInt64
    """The domain bits whose points hold nothing."""

    def __init__(out self):
        """Creates the empty map."""
        self.img = 0
        self.dom = 0
        self.dead = 0

    def put(self, d: Int, image: UInt64) -> Self:
        """Returns this map with domain bit `d` going to `image`.

        Args:
            d: The domain bit.
            image: Its image.

        Returns:
            The extended map.
        """
        var m = self
        m.img[d] = image
        m.dom |= bit(d)
        m.dead &= ~bit(d)
        return m

    def kill(self, d: Int) -> Self:
        """Returns this map with domain bit `d` holding nothing.

        Args:
            d: The domain bit.

        Returns:
            The extended map.
        """
        var m = self
        m.img[d] = 0
        m.dom |= bit(d)
        m.dead |= bit(d)
        return m

    def run(self, d0: Int, n: Int, c0: Int) -> Self:
        """Returns this map with domain bits `d0 .. d0 + n - 1` going to
        codomain bits `c0 .. c0 + n - 1`: the strided mode `(2^n : 2^c0)`.

        Args:
            d0: The lowest domain bit.
            n: The number of bits.
            c0: The codomain bit `d0` goes to.

        Returns:
            The extended map.
        """
        var m = self
        for i in range(n):
            m = m.put(d0 + i, bit(c0 + i))
        return m

    def has(self, d: Int) -> Bool:
        """Returns whether domain bit `d` is in the map.

        Args:
            d: The domain bit.

        Returns:
            True when `d` is in `dom`.
        """
        return (self.dom >> UInt64(d)) & 1 == 1

    def is_dead(self, d: Int) -> Bool:
        """Returns whether domain bit `d` holds nothing.

        Args:
            d: The domain bit.

        Returns:
            True when `d` is in `dead`.
        """
        return (self.dead >> UInt64(d)) & 1 == 1

    def live(self) -> UInt64:
        """Returns the domain bits that address the codomain.

        Returns:
            `dom` without `dead`.
        """
        return self.dom & ~self.dead

    def __call__(self, x: UInt64) -> UInt64:
        """Maps a point.

        Args:
            x: A point of the domain; bits outside `dom` are ignored.

        Returns:
            The XOR of the images of `x`'s set domain bits.
        """
        var out: UInt64 = 0
        var rest = x & self.dom
        while rest != 0:
            out ^= self.img[Int(count_trailing_zeros(rest))]
            rest &= rest - 1
        return out

    def __eq__(self, other: Self) -> Bool:
        """Returns whether two maps have the same domain, dead bits and
        images.

        Args:
            other: The map to compare with.

        Returns:
            True when the maps are equal.
        """
        return (
            self.dom == other.dom
            and self.dead == other.dead
            and self.img == other.img
        )

    def write_to(self, mut writer: Some[Writer]):
        """Writes the map as `{bit: image, ...}`, with `x` for a dead bit.

        Args:
            writer: The writer to write to.
        """
        writer.write("{")
        var rest = self.dom
        var first = True
        while rest != 0:
            var d = Int(count_trailing_zeros(rest))
            rest &= rest - 1
            if not first:
                writer.write(", ")
            first = False
            if self.is_dead(d):
                writer.write(d, ": x")
            else:
                writer.write(d, ": ", hex(self.img[d]))
        writer.write("}")


def _log2(n: Int) -> Int:
    return Int(count_trailing_zeros(UInt64(n)))


def _power_of_two(n: Int) -> Bool:
    return n > 0 and n & (n - 1) == 0


def is_bit_linear[L: TensorLayout]() -> Bool:
    """Returns whether a static layout is linear over F2.

    It is when every mode's extent is a power of two, every stride is zero or
    a power of two, and no two coordinate bits reach the same offset bit, so
    adding offsets never carries.

    Parameters:
        L: The layout.

    Returns:
        True when `placement[L]()` describes `L` exactly.
    """
    if not L.all_dims_known:
        return False
    var reached: UInt64 = 0
    comptime for i in range(L.rank):
        comptime extent = L.static_shape[i]
        comptime stride = L.static_stride[i]
        if not _power_of_two(extent):
            return False
        if stride != 0:
            if not _power_of_two(stride):
                return False
            var span = bits(_log2(stride), _log2(extent))
            if reached & span != 0:
                return False
            reached |= span
    return True


def placement[L: TensorLayout]() -> BitLayout:
    """Returns a static layout as a `BitLayout` from coordinate bits to
    offset bits.

    Coordinate bits number the layout's points colexicographically: mode 0's
    bits are the lowest. Bit `b` of mode `i` goes to `stride_i << b`, or
    nowhere for a stride of zero.

    Parameters:
        L: The layout, which `is_bit_linear`.

    Returns:
        The layout's placement.
    """
    var m = BitLayout()
    var d = 0
    comptime for i in range(L.rank):
        comptime extent = L.static_shape[i]
        comptime stride = L.static_stride[i]
        for b in range(_log2(extent)):
            m = m.put(d, UInt64(stride) << UInt64(b))
            d += 1
    return m
