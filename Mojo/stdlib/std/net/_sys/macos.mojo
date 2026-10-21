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
"""macOS (Darwin) socket address layouts and constants.

The layouts and values are transcribed from `<sys/socket.h>`,
`<netinet/in.h>`, and `<sys/un.h>` on macOS.
"""

from std.ffi import c_char, c_int
from std.sys import size_of

from .inet import SockaddrIn, SockaddrIn6, in6_addr, in_addr, in_port_t

comptime sa_family_t = UInt8
"""C `sa_family_t`: the address family (`AF_*`) of a socket address."""

comptime AF_INET: c_int = 2
"""IPv4 internet protocols."""
comptime AF_INET6: c_int = 30
"""IPv6 internet protocols."""
comptime SO_NOSIGPIPE: c_int = 0x1022
"""Don't raise `SIGPIPE` on write to a closed peer."""


@fieldwise_init
struct sockaddr(Copyable, Writable):
    """C `struct sockaddr`: the generic socket address header."""

    var sa_len: UInt8
    """The total length of the address."""
    var sa_family: sa_family_t
    """The address family (`AF_*`)."""
    var sa_data: Array[c_char, 14]
    """Padding to the historical 16-byte `sockaddr` size."""


@fieldwise_init
struct sockaddr_in(ImplicitlyCopyable, SockaddrIn, Writable):
    """C `struct sockaddr_in`: an IPv4 socket address."""

    var sin_len: UInt8
    """The total length of the address (16)."""
    var sin_family: sa_family_t
    """The address family (`AF_INET`)."""
    var sin_port: in_port_t
    """The port in network byte order."""
    var sin_addr: in_addr
    """The IPv4 address."""
    var sin_zero: Array[c_char, 8]
    """Padding to the 16-byte `sockaddr` size; always zero."""

    def __init__(out self, *, port: in_port_t, addr: in_addr):
        self.sin_len = UInt8(size_of[Self]())
        self.sin_family = AF_INET.cast[sa_family_t.dtype]()
        self.sin_port = port
        self.sin_addr = addr
        self.sin_zero = Array[c_char, 8](fill=0)

    def __init__(out self, *, copy: Self):
        self.sin_len = copy.sin_len
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

    var sin6_len: UInt8
    """The total length of the address (28)."""
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
        self.sin6_len = UInt8(size_of[Self]())
        self.sin6_family = AF_INET6.cast[sa_family_t.dtype]()
        self.sin6_port = port
        self.sin6_flowinfo = flowinfo
        self.sin6_addr = addr^
        self.sin6_scope_id = scope_id

    def __init__(out self, *, copy: Self):
        self.sin6_len = copy.sin6_len
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


comptime _SUN_PATH_LEN = 104


@fieldwise_init
struct sockaddr_un(Copyable, Writable):
    """C `struct sockaddr_un`: a Unix domain socket address."""

    var sun_len: UInt8
    """The total length of the address."""
    var sun_family: sa_family_t
    """The address family (`AF_UNIX`)."""
    var sun_path: Array[c_char, _SUN_PATH_LEN]
    """The filesystem path, NUL-terminated unless it fills the array."""


# Sizing for `sockaddr_storage`, from Darwin's `<sys/socket.h>`
comptime _SS_MAXSIZE = 128
comptime _SS_ALIGNSIZE = size_of[Int64]()
comptime _SS_PAD1SIZE = (
    _SS_ALIGNSIZE - size_of[UInt8]() - size_of[sa_family_t]()
)
comptime _SS_PAD2SIZE = (
    _SS_MAXSIZE
    - size_of[UInt8]()
    - size_of[sa_family_t]()
    - _SS_PAD1SIZE
    - _SS_ALIGNSIZE
)


@fieldwise_init
struct sockaddr_storage(Copyable, Writable):
    """C `struct sockaddr_storage`: sized and aligned to hold any address.

    Used as the out-parameter for calls that return a peer address
    (`accept`, `getsockname`, ...); inspect `ss_family` and reinterpret
    as the matching concrete type.
    """

    var ss_len: UInt8
    """The total length of the stored address."""
    var ss_family: sa_family_t
    """The address family of the stored address (`AF_*`)."""
    var __ss_pad1: Array[c_char, _SS_PAD1SIZE]
    var __ss_align: Int64
    var __ss_pad2: Array[c_char, _SS_PAD2SIZE]
