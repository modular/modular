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
"""Linux (glibc) socket address layouts and constants.

The layouts and values are transcribed from `<sys/socket.h>`,
`<netinet/in.h>`, and `<sys/un.h>` on x86-64 and aarch64 Linux.
"""

from std.ffi import c_char, c_int, c_uint, c_ulong, c_ushort, external_call
from std.sys import size_of
from std.sys.info import CompilationTarget

from .inet import SockaddrIn, SockaddrIn6, in6_addr, in_addr, in_port_t

comptime sa_family_t = c_ushort
"""C `sa_family_t`: the address family (`AF_*`) of a socket address."""

comptime socklen_t = c_uint
"""C `socklen_t`: the size of a socket address or option value, in bytes."""

comptime AF_INET: c_int = 2
"""IPv4 internet protocols."""
comptime AF_INET6: c_int = 10
"""IPv6 internet protocols."""
comptime SOCK_NONBLOCK: c_int = 0o4000
"""Create the socket non-blocking (or-ed into the socket type)."""
comptime SOCK_CLOEXEC: c_int = 0o2000000
"""Create the socket close-on-exec (or-ed into the socket type)."""
comptime SO_PROTOCOL: c_int = 38
"""Get the socket protocol (read-only)."""
comptime SO_DOMAIN: c_int = 39
"""Get the socket domain (read-only)."""
comptime SO_ZEROCOPY: c_int = 60
"""Allow `MSG_ZEROCOPY` sends on the socket."""
comptime MSG_ERRQUEUE: c_int = 0x2000
"""Receive from the error queue, where `MSG_ZEROCOPY` completions arrive."""
comptime MSG_MORE: c_int = 0x8000
"""More data follows; hold back a partial packet until it arrives."""
comptime MSG_ZEROCOPY: c_int = 0x4000000
"""Send from the caller's buffer without copying; needs `SO_ZEROCOPY` set."""
comptime MSG_CMSG_CLOEXEC: c_int = 0x40000000
"""Set close-on-exec on file descriptors received with `SCM_RIGHTS`."""


@fieldwise_init
struct sockaddr(Copyable, Writable):
    """C `struct sockaddr`: the generic socket address header."""

    var sa_family: sa_family_t
    """The address family (`AF_*`)."""
    var sa_data: Array[c_char, 14]
    """Padding to the historical 16-byte `sockaddr` size."""


@fieldwise_init
struct sockaddr_in(ImplicitlyCopyable, SockaddrIn, Writable):
    """C `struct sockaddr_in`: an IPv4 socket address."""

    var sin_family: sa_family_t
    """The address family (`AF_INET`)."""
    var sin_port: in_port_t
    """The port in network byte order."""
    var sin_addr: in_addr
    """The IPv4 address."""
    var sin_zero: Array[UInt8, 8]
    """Padding to the 16-byte `sockaddr` size; always zero."""

    def __init__(out self, *, port: in_port_t, addr: in_addr):
        self.sin_family = AF_INET.cast[sa_family_t.dtype]()
        self.sin_port = port
        self.sin_addr = addr
        self.sin_zero = Array[UInt8, 8](fill=0)

    def __init__(out self, *, copy: Self):
        self.sin_family = copy.sin_family
        self.sin_port = copy.sin_port
        self.sin_addr = copy.sin_addr
        self.sin_zero = copy.sin_zero.copy()

    def port(self) -> in_port_t:
        return self.sin_port

    def addr(self) -> in_addr:
        return self.sin_addr


@fieldwise_init
struct sockaddr_in6(ImplicitlyCopyable, SockaddrIn6, Writable):
    """C `struct sockaddr_in6`: an IPv6 socket address."""

    var sin6_family: sa_family_t
    """The address family (`AF_INET6`)."""
    var sin6_port: in_port_t
    """The port in network byte order."""
    var sin6_flowinfo: UInt32
    """IPv6 flow information."""
    var sin6_addr: in6_addr
    """The IPv6 address."""
    var sin6_scope_id: UInt32
    """Scope zone index for link-local addresses."""

    def __init__(
        out self,
        *,
        port: in_port_t,
        var addr: in6_addr,
        flowinfo: UInt32,
        scope_id: UInt32,
    ):
        self.sin6_family = AF_INET6.cast[sa_family_t.dtype]()
        self.sin6_port = port
        self.sin6_flowinfo = flowinfo
        self.sin6_addr = addr^
        self.sin6_scope_id = scope_id

    def __init__(out self, *, copy: Self):
        self.sin6_family = copy.sin6_family
        self.sin6_port = copy.sin6_port
        self.sin6_flowinfo = copy.sin6_flowinfo
        self.sin6_addr = copy.sin6_addr.copy()
        self.sin6_scope_id = copy.sin6_scope_id

    def port(self) -> in_port_t:
        return self.sin6_port

    def addr(self) -> in6_addr:
        return self.sin6_addr.copy()

    def flowinfo(self) -> UInt32:
        return self.sin6_flowinfo

    def scope_id(self) -> UInt32:
        return self.sin6_scope_id


comptime _SUN_PATH_LEN = 108


@fieldwise_init
struct sockaddr_un(Copyable, Writable):
    """C `struct sockaddr_un`: a Unix domain socket address."""

    var sun_family: sa_family_t
    """The address family (`AF_UNIX`)."""
    var sun_path: Array[c_char, _SUN_PATH_LEN]
    """The filesystem path, NUL-terminated unless it fills the array."""


# Sizing for `sockaddr_storage`, from glibc
comptime __ss_aligntype = c_ulong
comptime __SOCKADDR_COMMON_SIZE = size_of[c_ushort]()
comptime _SS_SIZE = 128
comptime _SS_PADSIZE = _SS_SIZE - __SOCKADDR_COMMON_SIZE - size_of[
    __ss_aligntype
]()


@fieldwise_init
struct sockaddr_storage(Copyable, Writable):
    """C `struct sockaddr_storage`: sized and aligned to hold any address.

    Used as the out-parameter for calls that return a peer address
    (`accept`, `getsockname`, ...); inspect `ss_family` and reinterpret
    as the matching concrete type.
    """

    var ss_family: sa_family_t
    """The address family of the stored address (`AF_*`)."""
    var __ss_padding: Array[c_char, _SS_PADSIZE]
    var __ss_align: __ss_aligntype


def accept4(
    sockfd: c_int,
    addr: OptionalMutPointer[sockaddr, _],
    addrlen: OptionalMutPointer[socklen_t, _],
    flags: c_int,
) -> c_int:
    """C `accept4(2)`: `accept` that applies `SOCK_*` flags atomically."""
    comptime assert (
        CompilationTarget.is_linux()
    ), "`accept4` is only available on Linux"
    return external_call["accept4", c_int](sockfd, addr, addrlen, flags)
