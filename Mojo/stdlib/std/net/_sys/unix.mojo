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
"""Socket constants, types, and syscall bindings shared by the Unix
platforms, Linux and macOS.
"""

from std.ffi import (
    c_int,
    c_long,
    c_size_t,
    c_ssize_t,
    c_uint,
    external_call,
)
from std.sys.info import platform_map, platform_type_map

from .inet import SockaddrIn, SockaddrIn6
from .linux import (
    AF_INET6 as linux_AF_INET6,
    sockaddr as linux_sockaddr,
    sockaddr_in as linux_sockaddr_in,
    sockaddr_in6 as linux_sockaddr_in6,
    sockaddr_storage as linux_sockaddr_storage,
    sockaddr_un as linux_sockaddr_un,
)
from .macos import (
    AF_INET6 as macos_AF_INET6,
    sockaddr as macos_sockaddr,
    sockaddr_in as macos_sockaddr_in,
    sockaddr_in6 as macos_sockaddr_in6,
    sockaddr_storage as macos_sockaddr_storage,
    sockaddr_un as macos_sockaddr_un,
)

comptime socklen_t = c_uint
"""C `socklen_t`: the size of a socket address or option value, in bytes."""


def constant[name: StaticString, *, linux: c_int, macos: c_int]() -> c_int:
    return platform_map[T=c_int, String(name), linux=linux, macos=macos]()


# ===----------------------------------------------------------------------=== #
# Address families
# ===----------------------------------------------------------------------=== #

comptime AF_UNSPEC: c_int = 0
"""Unspecified address family."""
comptime AF_UNIX: c_int = 1
"""Unix domain (local) sockets."""
comptime AF_INET: c_int = 2
"""IPv4 internet protocols."""
comptime AF_INET6: c_int = constant[
    "AF_INET6", linux=linux_AF_INET6, macos=macos_AF_INET6
]()
"""IPv6 internet protocols."""

# ===----------------------------------------------------------------------=== #
# Socket types
# ===----------------------------------------------------------------------=== #

comptime SOCK_STREAM: c_int = 1
"""Sequenced, reliable, connection-based byte stream."""
comptime SOCK_DGRAM: c_int = 2
"""Connectionless, unreliable datagrams."""
comptime SOCK_RAW: c_int = 3
"""Raw protocol access."""
comptime SOCK_SEQPACKET: c_int = 5
"""Sequenced, reliable, connection-based messages that keep their boundaries."""

# ===----------------------------------------------------------------------=== #
# Socket option level and names
# ===----------------------------------------------------------------------=== #

comptime SOL_SOCKET: c_int = constant["SOL_SOCKET", linux=1, macos=0xFFFF]()
"""Option level for socket-level options."""
comptime SO_DEBUG: c_int = 1
"""Enable socket debugging."""
comptime SO_REUSEADDR: c_int = constant["SO_REUSEADDR", linux=2, macos=0x0004]()
"""Allow reuse of local addresses."""
comptime SO_TYPE: c_int = constant["SO_TYPE", linux=3, macos=0x1008]()
"""Get the socket type (read-only)."""
comptime SO_ERROR: c_int = constant["SO_ERROR", linux=4, macos=0x1007]()
"""Get and clear the pending socket error (read-only)."""
comptime SO_DONTROUTE: c_int = constant["SO_DONTROUTE", linux=5, macos=0x0010]()
"""Bypass routing; send directly to the interface."""
comptime SO_BROADCAST: c_int = constant["SO_BROADCAST", linux=6, macos=0x0020]()
"""Permit sending broadcast datagrams."""
comptime SO_SNDBUF: c_int = constant["SO_SNDBUF", linux=7, macos=0x1001]()
"""Send buffer size."""
comptime SO_RCVBUF: c_int = constant["SO_RCVBUF", linux=8, macos=0x1002]()
"""Receive buffer size."""
comptime SO_KEEPALIVE: c_int = constant["SO_KEEPALIVE", linux=9, macos=0x0008]()
"""Enable keep-alive probes on connection-oriented sockets."""
comptime SO_OOBINLINE: c_int = constant[
    "SO_OOBINLINE", linux=10, macos=0x0100
]()
"""Leave received out-of-band data in the normal stream."""
comptime SO_LINGER: c_int = constant["SO_LINGER", linux=13, macos=0x0080]()
"""Linger on close if unsent data is present (takes `struct linger`).

macOS measures this option's timeout in clock ticks, not seconds.
"""
comptime SO_REUSEPORT: c_int = constant[
    "SO_REUSEPORT", linux=15, macos=0x0200
]()
"""Allow multiple sockets to bind the same address and port."""
comptime SO_RCVTIMEO: c_int = constant["SO_RCVTIMEO", linux=20, macos=0x1006]()
"""Receive timeout (takes `struct timeval`)."""
comptime SO_SNDTIMEO: c_int = constant["SO_SNDTIMEO", linux=21, macos=0x1005]()
"""Send timeout (takes `struct timeval`)."""
comptime SO_ACCEPTCONN: c_int = constant[
    "SO_ACCEPTCONN", linux=30, macos=0x0002
]()
"""Whether the socket is listening (read-only)."""

# ===----------------------------------------------------------------------=== #
# TCP option names
# ===----------------------------------------------------------------------=== #

comptime TCP_NODELAY: c_int = 1
"""Disable Nagle's algorithm (level `IPPROTO_TCP`)."""

# ===----------------------------------------------------------------------=== #
# IP option names
# ===----------------------------------------------------------------------=== #

comptime IP_TTL: c_int = constant["IP_TTL", linux=2, macos=4]()
"""Time-to-live of outgoing unicast packets (level `IPPROTO_IP`)."""
comptime IP_MULTICAST_TTL: c_int = constant[
    "IP_MULTICAST_TTL", linux=33, macos=10
]()
"""Time-to-live of outgoing multicast packets; takes a `u_char`."""
comptime IP_MULTICAST_LOOP: c_int = constant[
    "IP_MULTICAST_LOOP", linux=34, macos=11
]()
"""Loop outgoing multicast packets back to local sockets; takes a `u_char`."""

# ===----------------------------------------------------------------------=== #
# IPv6 option names
# ===----------------------------------------------------------------------=== #

comptime IPV6_MULTICAST_HOPS: c_int = constant[
    "IPV6_MULTICAST_HOPS", linux=18, macos=10
]()
"""Hop limit of outgoing multicast packets (level `IPPROTO_IPV6`)."""
comptime IPV6_MULTICAST_LOOP: c_int = constant[
    "IPV6_MULTICAST_LOOP", linux=19, macos=11
]()
"""Loop outgoing multicast packets back to local sockets."""
comptime IPV6_V6ONLY: c_int = constant["IPV6_V6ONLY", linux=26, macos=27]()
"""Restrict an IPv6 socket to IPv6 traffic."""

# ===----------------------------------------------------------------------=== #
# `send` and `recv` flags
# ===----------------------------------------------------------------------=== #

comptime MSG_OOB: c_int = 0x1
"""Send or receive out-of-band data."""
comptime MSG_PEEK: c_int = 0x2
"""Peek at incoming data without consuming it."""
comptime MSG_DONTROUTE: c_int = 0x4
"""Send directly to the interface, bypassing routing."""
comptime MSG_CTRUNC: c_int = constant["MSG_CTRUNC", linux=0x8, macos=0x20]()
"""Ancillary data was truncated (reported in `recvmsg`'s `msg_flags`)."""
comptime MSG_TRUNC: c_int = constant["MSG_TRUNC", linux=0x20, macos=0x10]()
"""The datagram was truncated (reported in `recvmsg`'s `msg_flags`)."""
comptime MSG_DONTWAIT: c_int = constant[
    "MSG_DONTWAIT", linux=0x40, macos=0x80
]()
"""Non-blocking operation for this call only."""
comptime MSG_EOR: c_int = constant["MSG_EOR", linux=0x80, macos=0x8]()
"""Terminate a record, on socket types that have records."""
comptime MSG_WAITALL: c_int = constant["MSG_WAITALL", linux=0x100, macos=0x40]()
"""Block until the full request is received."""
comptime MSG_NOSIGNAL: c_int = constant[
    "MSG_NOSIGNAL", linux=0x4000, macos=0x80000
]()
"""Don't raise `SIGPIPE` on send to a closed peer."""

# ===----------------------------------------------------------------------=== #
# Control message types
# ===----------------------------------------------------------------------=== #

comptime SCM_RIGHTS: c_int = 1
"""Control message carrying file descriptors (level `SOL_SOCKET`)."""

# ===----------------------------------------------------------------------=== #
# `shutdown` directions
# ===----------------------------------------------------------------------=== #

comptime SHUT_RD: c_int = 0
"""Disallow further receives."""
comptime SHUT_WR: c_int = 1
"""Disallow further sends."""
comptime SHUT_RDWR: c_int = 2
"""Disallow further sends and receives."""

# ===----------------------------------------------------------------------=== #
# File status flags
# ===----------------------------------------------------------------------=== #

comptime O_NONBLOCK: c_int = constant[
    "O_NONBLOCK", linux=0o4000, macos=0x0004
]()
"""Non-blocking file status flag."""

# ===----------------------------------------------------------------------=== #
# `fcntl` commands
# ===----------------------------------------------------------------------=== #

comptime F_GETFL: c_int = 3
"""Get the file status flags."""
comptime F_SETFL: c_int = 4
"""Set the file status flags."""
comptime F_DUPFD_CLOEXEC: c_int = constant[
    "F_DUPFD_CLOEXEC", linux=1030, macos=67
]()
"""Duplicate a descriptor with close-on-exec set on the copy."""

# ===----------------------------------------------------------------------=== #
# Socket option values
# ===----------------------------------------------------------------------=== #

comptime time_t = c_long
"""C `time_t`: a time in whole seconds."""
comptime suseconds_t = Scalar[
    platform_map[
        T=DType, "suseconds_t", linux=c_long.dtype, macos=DType.int32
    ]()
]
"""C `suseconds_t`: a time in microseconds."""


@fieldwise_init
struct timeval(TrivialRegisterPassable, Writable):
    """C `struct timeval`: the value of `SO_RCVTIMEO` and `SO_SNDTIMEO`."""

    var tv_sec: time_t
    var tv_usec: suseconds_t


@fieldwise_init
struct linger(TrivialRegisterPassable, Writable):
    """C `struct linger`: the value of `SO_LINGER`."""

    var l_onoff: c_int
    var l_linger: c_int


# ===----------------------------------------------------------------------=== #
# Socket addresses
# ===----------------------------------------------------------------------=== #

comptime sockaddr = platform_type_map[
    Trait=Copyable,
    linux=linux_sockaddr,
    macos=macos_sockaddr,
]
"""The compilation target's C `struct sockaddr`."""

comptime sockaddr_in = platform_type_map[
    Trait=SockaddrIn,
    linux=linux_sockaddr_in,
    macos=macos_sockaddr_in,
]
"""The compilation target's C `struct sockaddr_in`."""

comptime sockaddr_in6 = platform_type_map[
    Trait=SockaddrIn6,
    linux=linux_sockaddr_in6,
    macos=macos_sockaddr_in6,
]
"""The compilation target's C `struct sockaddr_in6`."""

comptime sockaddr_un = platform_type_map[
    Trait=Copyable,
    linux=linux_sockaddr_un,
    macos=macos_sockaddr_un,
]
"""The compilation target's C `struct sockaddr_un`."""

comptime sockaddr_storage = platform_type_map[
    Trait=Copyable,
    linux=linux_sockaddr_storage,
    macos=macos_sockaddr_storage,
]
"""The compilation target's C `struct sockaddr_storage`."""


# ===----------------------------------------------------------------------=== #
# Syscall bindings (`<sys/socket.h>`)
# ===----------------------------------------------------------------------=== #


def socket(domain: c_int, type: c_int, protocol: c_int) -> c_int:
    """C `socket(2)`: creates an endpoint for communication.

    Args:
        domain: The protocol family (`AF_*`).
        type: The socket type (`SOCK_*`).
        protocol: The protocol, or 0 for the type's default.

    Returns:
        A new socket descriptor, or -1 with `errno` set.
    """
    return external_call["socket", c_int](domain, type, protocol)


def bind(
    sockfd: c_int, addr: ImmPointer[sockaddr, _], addrlen: socklen_t
) -> c_int:
    """C `bind(2)`: assigns a local address to a socket.

    Args:
        sockfd: The socket descriptor.
        addr: Pointer to the address to bind.
        addrlen: The valid size of the address behind `addr`, in bytes.

    Returns:
        0 on success, or -1 with `errno` set.
    """
    return external_call["bind", c_int](sockfd, addr, addrlen)


def listen(sockfd: c_int, backlog: c_int) -> c_int:
    """C `listen(2)`: marks a socket as accepting connections.

    Args:
        sockfd: The socket descriptor.
        backlog: The maximum length of the pending-connection queue.

    Returns:
        0 on success, or -1 with `errno` set.
    """
    return external_call["listen", c_int](sockfd, backlog)


def accept(
    sockfd: c_int,
    addr: OptionalMutPointer[sockaddr, _],
    addrlen: OptionalMutPointer[socklen_t, _],
) -> c_int:
    """C `accept(2)`: accepts a connection on a listening socket.

    Args:
        sockfd: The listening socket descriptor.
        addr: Out-pointer to storage for the peer address, typically a
            `sockaddr_storage` cast to `sockaddr`, or `None` when the
            caller does not want it.
        addrlen: In/out pointer holding the storage capacity on entry and
            the written address size on return; `None` exactly when `addr`
            is `None`.

    Returns:
        The new connection's descriptor, or -1 with `errno` set.
    """
    return external_call["accept", c_int](sockfd, addr, addrlen)


def connect(
    sockfd: c_int, addr: ImmPointer[sockaddr, _], addrlen: socklen_t
) -> c_int:
    """C `connect(2)`: initiates a connection on a socket.

    Args:
        sockfd: The socket descriptor.
        addr: Pointer to the peer address.
        addrlen: The valid size of the address behind `addr`, in bytes.

    Returns:
        0 on success, or -1 with `errno` set.
    """
    return external_call["connect", c_int](sockfd, addr, addrlen)


def send(
    sockfd: c_int, buf: ImmPointer[UInt8, _], len: c_size_t, flags: c_int
) -> c_ssize_t:
    """C `send(2)`: sends bytes on a connected socket.

    Args:
        sockfd: The socket descriptor.
        buf: Pointer to the bytes to send.
        len: The number of bytes to send.
        flags: `MSG_*` flags, or 0.

    Returns:
        The number of bytes sent (possibly less than `len`), or -1 with
        `errno` set.
    """
    return external_call["send", c_ssize_t](sockfd, buf, len, flags)


def recv(
    sockfd: c_int, buf: MutPointer[UInt8, _], len: c_size_t, flags: c_int
) -> c_ssize_t:
    """C `recv(2)`: receives bytes from a connected socket.

    Args:
        sockfd: The socket descriptor.
        buf: Pointer to the buffer to receive into.
        len: The buffer capacity, in bytes.
        flags: `MSG_*` flags, or 0.

    Returns:
        The number of bytes received, 0 on orderly peer shutdown when
        `len` is nonzero, or -1 with `errno` set.
    """
    return external_call["recv", c_ssize_t](sockfd, buf, len, flags)


def shutdown(sockfd: c_int, how: c_int) -> c_int:
    """C `shutdown(2)`: shuts down part of a full-duplex connection.

    Args:
        sockfd: The socket descriptor.
        how: One of `SHUT_RD`, `SHUT_WR`, or `SHUT_RDWR`.

    Returns:
        0 on success, or -1 with `errno` set.
    """
    return external_call["shutdown", c_int](sockfd, how)


def getsockname(
    sockfd: c_int,
    addr: MutPointer[sockaddr, _],
    addrlen: MutPointer[socklen_t, _],
) -> c_int:
    """C `getsockname(2)`: returns the socket's local address.

    Args:
        sockfd: The socket descriptor.
        addr: Out-pointer to storage for the address, typically a
            `sockaddr_storage` cast to `sockaddr`.
        addrlen: In/out pointer holding the storage capacity on entry and
            the written address size on return.

    Returns:
        0 on success, or -1 with `errno` set.
    """
    return external_call["getsockname", c_int](sockfd, addr, addrlen)


def getpeername(
    sockfd: c_int,
    addr: MutPointer[sockaddr, _],
    addrlen: MutPointer[socklen_t, _],
) -> c_int:
    """C `getpeername(2)`: returns the connected peer's address.

    Args:
        sockfd: The socket descriptor.
        addr: Out-pointer to storage for the address, typically a
            `sockaddr_storage` cast to `sockaddr`.
        addrlen: In/out pointer holding the storage capacity on entry and
            the written address size on return.

    Returns:
        0 on success, or -1 with `errno` set.
    """
    return external_call["getpeername", c_int](sockfd, addr, addrlen)


def setsockopt(
    sockfd: c_int,
    level: c_int,
    optname: c_int,
    optval: ImmPointer[NoneType, _],
    optlen: socklen_t,
) -> c_int:
    """C `setsockopt(2)`: sets a socket option.

    Args:
        sockfd: The socket descriptor.
        level: The option's protocol level (`SOL_SOCKET`, `IPPROTO_TCP`, ...).
        optname: The option name (`SO_*`, `TCP_*`, ...).
        optval: Pointer to the option value.
        optlen: The size of the value behind `optval`, in bytes.

    Returns:
        0 on success, or -1 with `errno` set.
    """
    return external_call["setsockopt", c_int](
        sockfd, level, optname, optval, optlen
    )


def getsockopt(
    sockfd: c_int,
    level: c_int,
    optname: c_int,
    optval: MutPointer[NoneType, _],
    optlen: MutPointer[socklen_t, _],
) -> c_int:
    """C `getsockopt(2)`: gets a socket option.

    Args:
        sockfd: The socket descriptor.
        level: The option's protocol level (`SOL_SOCKET`, `IPPROTO_TCP`, ...).
        optname: The option name (`SO_*`, `TCP_*`, ...).
        optval: Out-pointer to storage for the option value.
        optlen: In/out pointer holding the storage capacity on entry and
            the written value size on return.

    Returns:
        0 on success, or -1 with `errno` set.
    """
    return external_call["getsockopt", c_int](
        sockfd, level, optname, optval, optlen
    )


def sendto(
    sockfd: c_int,
    buf: ImmPointer[UInt8, _],
    len: c_size_t,
    flags: c_int,
    dest_addr: ImmPointer[sockaddr, _],
    addrlen: socklen_t,
) -> c_ssize_t:
    """C `sendto(2)`: sends bytes to the given address."""
    return external_call["sendto", c_ssize_t](
        sockfd, buf, len, flags, dest_addr, addrlen
    )


def recvfrom(
    sockfd: c_int,
    buf: MutPointer[UInt8, _],
    len: c_size_t,
    flags: c_int,
    src_addr: OptionalMutPointer[sockaddr, _],
    addrlen: OptionalMutPointer[socklen_t, _],
) -> c_ssize_t:
    """C `recvfrom(2)`: receives bytes and the address they came from."""
    return external_call["recvfrom", c_ssize_t](
        sockfd, buf, len, flags, src_addr, addrlen
    )


def socketpair(
    domain: c_int,
    type: c_int,
    protocol: c_int,
    sv: MutPointer[Array[c_int, 2], _],
) -> c_int:
    """C `socketpair(2)`: creates a pair of connected sockets."""
    return external_call["socketpair", c_int](domain, type, protocol, sv)
