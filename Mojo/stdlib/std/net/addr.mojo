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
"""Defines the IP address and socket address types."""

from std.hash.hasher import Hasher


struct Ipv4Addr(Equatable, Hashable, ImplicitlyCopyable, Writable):
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


struct Ipv6Addr(Equatable, Hashable, ImplicitlyCopyable, Writable):
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


__enum IpAddr(Equatable, Hashable, ImplicitlyCopyable, Writable):
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

    def __hash__(self, mut hasher: Some[Hasher]):
        """Hashes the address using the given hasher.

        Args:
            hasher: The hasher instance.
        """
        match self:
            case .v4(addr):
                addr.__hash__(hasher)
            case .v6(addr):
                addr.__hash__(hasher)

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


struct SocketAddrV4(Equatable, Hashable, ImplicitlyCopyable, Writable):
    """Represents an IPv4 socket address: an IPv4 address and a port."""

    var _ip: Ipv4Addr
    var _port: UInt16

    def __init__(out self, ip: Ipv4Addr, port: UInt16):
        """Constructs a socket address from an IPv4 address and a port.

        Args:
            ip: The IPv4 address.
            port: The port number in host byte order.
        """
        self._ip = ip
        self._port = port

    def ip(self) -> Ipv4Addr:
        """Returns the IPv4 address.

        Returns:
            The IPv4 address.
        """
        return self._ip

    def port(self) -> UInt16:
        """Returns the port number.

        Returns:
            The port number in host byte order.
        """
        return self._port

    def write_to(self, mut writer: Some[Writer]):
        """Writes the socket address as `address:port`, such as
        `127.0.0.1:8080`.

        Args:
            writer: The writer to write to.
        """
        t"{self._ip}:{self._port}".write_to(writer)


struct SocketAddrV6(Equatable, Hashable, ImplicitlyCopyable, Writable):
    """Represents an IPv6 socket address."""

    var _ip: Ipv6Addr
    var _port: UInt16
    var _flowinfo: UInt32
    var _scope_id: UInt32

    def __init__(
        out self,
        ip: Ipv6Addr,
        port: UInt16,
        flowinfo: UInt32 = 0,
        scope_id: UInt32 = 0,
    ):
        """Constructs a socket address from an IPv6 address and a port.

        Args:
            ip: The IPv6 address.
            port: The port number in host byte order.
            flowinfo: The flow information, which combines the flow label and
                the traffic class. See `flowinfo()`.
            scope_id: The scope ID.
        """
        self._ip = ip
        self._port = port
        self._flowinfo = flowinfo
        self._scope_id = scope_id

    def ip(self) -> Ipv6Addr:
        """Returns the IPv6 address.

        Returns:
            The IPv6 address.
        """
        return self._ip

    def port(self) -> UInt16:
        """Returns the port number.

        Returns:
            The port number in host byte order.
        """
        return self._port

    def flowinfo(self) -> UInt32:
        """Returns the flow information.

        This is the `sin6_flowinfo` field of C's `struct sockaddr_in6`, from
        RFC 2553, section 3.3. It isn't just the flow label: its 32 bits
        combine the 20-bit flow label and the 8-bit traffic class, which
        RFC 8200 defines in sections 6 and 7.

        Returns:
            The flow information.
        """
        return self._flowinfo

    def scope_id(self) -> UInt32:
        """Returns the scope ID.

        Returns:
            The scope ID.
        """
        return self._scope_id

    def write_to(self, mut writer: Some[Writer]):
        """Writes the socket address as `[address]:port`, such as
        `[::1]:8080`.

        A nonzero scope ID follows the address after a `%`, such as
        `[fe80::1%2]:8080`. The flow information isn't written.

        Args:
            writer: The writer to write to.
        """
        if self._scope_id == 0:
            t"[{self._ip}]:{self._port}".write_to(writer)
        else:
            t"[{self._ip}%{self._scope_id}]:{self._port}".write_to(writer)


__enum SocketAddr(Equatable, Hashable, ImplicitlyCopyable, Writable):
    """Represents either an IPv4 or an IPv6 socket address."""

    case v4(SocketAddrV4)
    """An IPv4 socket address."""
    case v6(SocketAddrV6)
    """An IPv6 socket address."""

    def __init__(out self, ip: IpAddr, port: UInt16):
        """Constructs a socket address in the family of `ip`.

        An IPv6 socket address gets zero for its flow information and scope ID.

        Args:
            ip: The IP address.
            port: The port number in host byte order.
        """
        match ip:
            case .v4(addr):
                self = .v4(SocketAddrV4(addr, port))
            case .v6(addr):
                self = .v6(SocketAddrV6(addr, port))

    def ip(self) -> IpAddr:
        """Returns the IP address.

        Returns:
            The IP address.
        """
        match self:
            case .v4(addr):
                return .v4(addr.ip())
            case .v6(addr):
                return .v6(addr.ip())

    def port(self) -> UInt16:
        """Returns the port number.

        Returns:
            The port number in host byte order.
        """
        match self:
            case .v4(addr):
                return addr.port()
            case .v6(addr):
                return addr.port()

    def __eq__(self, other: Self) -> Bool:
        """Returns whether two socket addresses are equal.

        Args:
            other: The socket address to compare against.

        Returns:
            True if both socket addresses have the same family and value.
        """
        match (self, other):
            case (.v4(addr), .v4(other_addr)):
                return addr == other_addr
            case (.v6(addr), .v6(other_addr)):
                return addr == other_addr
            case _:
                return False

    def __hash__(self, mut hasher: Some[Hasher]):
        """Hashes the socket address using the given hasher.

        Args:
            hasher: The hasher instance.
        """
        match self:
            case .v4(addr):
                addr.__hash__(hasher)
            case .v6(addr):
                addr.__hash__(hasher)

    def write_to(self, mut writer: Some[Writer]):
        """Writes the socket address in the notation of its family.

        Args:
            writer: The writer to write to.
        """
        match self:
            case .v4(addr):
                addr.write_to(writer)
            case .v6(addr):
                addr.write_to(writer)
