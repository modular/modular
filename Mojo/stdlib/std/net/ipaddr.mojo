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
"""Defines the IPv4 and IPv6 address types."""


struct Ipv4Addr(Equatable, ImplicitlyCopyable, Writable):
    """Represents an IPv4 address.

    The address is stored as four octets in network byte order.
    """

    var _octets: Array[Byte, 4]

    comptime localhost: Self = Self(127, 0, 0, 1)
    """The loopback address, 127.0.0.1."""
    comptime broadcast: Self = Self(255, 255, 255, 255)
    """The limited broadcast address, 255.255.255.255."""
    comptime unspecified: Self = Self(0, 0, 0, 0)
    """The unspecified address, 0.0.0.0."""

    def __init__(out self, a: Byte, b: Byte, c: Byte, d: Byte, /):
        """Constructs an address from its four octets, first octet first.

        Args:
            a: The first octet.
            b: The second octet.
            c: The third octet.
            d: The fourth octet.
        """
        self._octets = [a, b, c, d]

    def __init__(out self, *, bits: UInt32):
        """Constructs an address from its 32-bit numeric value.

        The most significant byte of `bits` becomes the first octet, so
        `Ipv4Addr(bits=0x7F000001)` is 127.0.0.1. A value that is already in
        network byte order, such as C's `in_addr.s_addr`, needs
        `UInt32.from_endian[.big]()` first.

        Args:
            bits: The address as an integer in host byte order.
        """
        self._octets = rebind_var[Array[Byte, 4]](bits.as_bytes[endian=.big]())

    def __init__(out self, *, var octets: Array[Byte, 4]):
        """Constructs an address from its octets in network byte order.

        Args:
            octets: The four octets, first octet first.
        """
        self._octets = octets^

    def __init__(out self, *, copy: Self):
        """Copies an address.

        Args:
            copy: The address to copy.
        """
        self._octets = copy._octets.copy()

    def bits(self) -> UInt32:
        """Returns the address as a 32-bit integer.

        This is the inverse of the `bits` initializer: the first octet
        becomes the most significant byte.

        Returns:
            The address as an integer in host byte order.
        """
        return UInt32.from_bytes[endian=.big](self._octets)

    def write_to(self, mut writer: Some[Writer]):
        """Writes the address in dotted-decimal notation, such as `127.0.0.1`.

        Args:
            writer: The writer to write to.
        """
        ref octets = self._octets
        t"{octets[0]}.{octets[1]}.{octets[2]}.{octets[3]}".write_to(writer)


struct Ipv6Addr(Equatable, ImplicitlyCopyable, Writable):
    """Represents an IPv6 address.

    The address is stored as sixteen octets in network byte order.
    """

    var _octets: Array[Byte, 16]

    comptime localhost: Self = Self(0, 0, 0, 0, 0, 0, 0, 1)
    """The loopback address, `::1`."""
    comptime unspecified: Self = Self(0, 0, 0, 0, 0, 0, 0, 0)
    """The unspecified address, `::`."""

    def __init__(
        out self,
        a: UInt16,
        b: UInt16,
        c: UInt16,
        d: UInt16,
        e: UInt16,
        f: UInt16,
        g: UInt16,
        h: UInt16,
        /,
    ):
        """Constructs an address from its eight 16-bit segments, first
        segment first.

        Args:
            a: The first segment.
            b: The second segment.
            c: The third segment.
            d: The fourth segment.
            e: The fifth segment.
            f: The sixth segment.
            g: The seventh segment.
            h: The eighth segment.
        """
        self._octets = rebind_var[Array[Byte, 16]](
            SIMD[.uint16, 8](a, b, c, d, e, f, g, h).as_bytes[endian=.big]()
        )

    def __init__(out self, *, bits: UInt128):
        """Constructs an address from its 128-bit numeric value.

        The most significant byte of `bits` becomes the first octet, so
        `Ipv6Addr(bits=1)` is `::1`.

        Args:
            bits: The address as an integer in host byte order.
        """
        self._octets = rebind_var[Array[Byte, 16]](bits.as_bytes[endian=.big]())

    def __init__(out self, *, segments: Array[UInt16, 8]):
        """Constructs an address from an array of its eight 16-bit segments.

        Args:
            segments: The segments in host byte order, first segment first.
        """
        self = Self(
            segments[0],
            segments[1],
            segments[2],
            segments[3],
            segments[4],
            segments[5],
            segments[6],
            segments[7],
        )

    def __init__(out self, *, var octets: Array[Byte, 16]):
        """Constructs an address from its octets in network byte order.

        Args:
            octets: The sixteen octets, first octet first.
        """
        self._octets = octets^

    def __init__(out self, *, copy: Self):
        """Copies an address.

        Args:
            copy: The address to copy.
        """
        self._octets = copy._octets.copy()

    def bits(self) -> UInt128:
        """Returns the address as a 128-bit integer.

        This is the inverse of the `bits` initializer: the first octet
        becomes the most significant byte.

        Returns:
            The address as an integer in host byte order.
        """
        return UInt128.from_bytes[endian=.big](self._octets)

    def write_to(self, mut writer: Some[Writer]):
        """Writes the address in RFC 5952 notation, such as `2001:db8::1`.

        Each segment is lowercase hexadecimal without leading zeros, and the
        longest run of two or more zero segments (the first, on a tie) becomes
        `::`. An IPv4-mapped address keeps its IPv4 part in dotted-decimal
        notation, such as `::ffff:192.0.2.1`.

        Args:
            writer: The writer to write to.
        """
        var segments = Array[UInt16, 8](
            fill_with=lambda (i: Int) -> UInt16: (
                self._octets[2 * i].cast[.uint16]() << 8
            )
            | (self._octets[2 * i + 1].cast[.uint16]())
        )

        if case [0, 0, 0, 0, 0, 0xFFFF, _, _] = segments:
            var ipv4 = Ipv4Addr(
                self._octets[12],
                self._octets[13],
                self._octets[14],
                self._octets[15],
            )
            t"::ffff:{ipv4}".write_to(writer)
            return

        var gap_start = -1
        var gap_end = -1
        var run_start = 0
        var run_len = 0
        for i, segment in enumerate(segments):
            if segment != 0:
                run_len = 0
                continue
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len >= 2 and run_len > gap_end - gap_start:
                gap_start = run_start
                gap_end = run_start + run_len

        var separator: StaticString = ""
        for i, segment in enumerate(segments):
            if gap_start <= i < gap_end:
                separator = "::"
                continue
            var digits = hex(segment, prefix="")
            t"{separator}{digits}".write_to(writer)
            separator = ":"
        # A zero run at the end has no segment after it to carry its `::`.
        if gap_end == len(segments):
            writer.write("::")


__enum IpAddr(Equatable, ImplicitlyCopyable, Writable):
    """Represents either an IPv4 or an IPv6 address."""

    case v4(Ipv4Addr)
    """An IPv4 address."""
    case v6(Ipv6Addr)
    """An IPv6 address."""

    def __eq__(self, other: Self) -> Bool:
        """Returns whether two addresses are equal.

        Args:
            other: The address to compare against.

        Returns:
            True if both addresses have the same family and value.
        """
        match (self, other):
            case (.v4(addr), .v4(other_addr)):
                return addr == other_addr
            case (.v6(addr), .v6(other_addr)):
                return addr == other_addr
            case _:
                return False

    def write_to(self, mut writer: Some[Writer]):
        """Writes the address in the notation of its family.

        Args:
            writer: The writer to write to.
        """
        match self:
            case .v4(addr):
                addr.write_to(writer)
            case .v6(addr):
                addr.write_to(writer)
