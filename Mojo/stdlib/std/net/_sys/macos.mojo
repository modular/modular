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

from .inet import in6_addr, in_addr, in_port_t

comptime SO_NOSIGPIPE: c_int = 0x1022
"""Don't raise `SIGPIPE` on write to a closed peer."""


@fieldwise_init
struct sockaddr(Copyable, Writable):
    """C `struct sockaddr`: the generic socket address header."""

    var sa_len: UInt8
    """The total length of the address."""
    var sa_family: UInt8
    """The address family (`AF_*`)."""
    var sa_data: Array[c_char, 14]
    """Padding to the historical 16-byte `sockaddr` size."""


@fieldwise_init
struct sockaddr_in(Copyable, Writable):
    """C `struct sockaddr_in`: an IPv4 socket address."""

    var sin_len: UInt8
    """The total length of the address (16)."""
    var sin_family: UInt8
    """The address family (`AF_INET`)."""
    var sin_port: in_port_t
    """The port in network byte order."""
    var sin_addr: in_addr
    """The IPv4 address."""
    var sin_zero: Array[c_char, 8]
    """Padding to the 16-byte `sockaddr` size; always zero."""


@fieldwise_init
struct sockaddr_in6(Copyable, Writable):
    """C `struct sockaddr_in6`: an IPv6 socket address."""

    var sin6_len: UInt8
    """The total length of the address (28)."""
    var sin6_family: UInt8
    """The address family (`AF_INET6`)."""
    var sin6_port: in_port_t
    """The port in network byte order."""
    var sin6_flowinfo: UInt32
    """IPv6 flow information."""
    var sin6_addr: in6_addr
    """The IPv6 address."""
    var sin6_scope_id: UInt32
    """Scope zone index for link-local addresses."""


comptime _SUN_PATH_LEN = 104


@fieldwise_init
struct sockaddr_un(Copyable, Writable):
    """C `struct sockaddr_un`: a Unix domain socket address."""

    var sun_len: UInt8
    """The total length of the address."""
    var sun_family: UInt8
    """The address family (`AF_UNIX`)."""
    var sun_path: Array[c_char, _SUN_PATH_LEN]
    """The filesystem path, NUL-terminated unless it fills the array."""


# Sizing for `sockaddr_storage`, from Darwin's `<sys/socket.h>`
comptime _SS_MAXSIZE = 128
comptime _SS_ALIGNSIZE = size_of[Int64]()
comptime _SS_PAD1SIZE = _SS_ALIGNSIZE - size_of[UInt8]() - size_of[UInt8]()
comptime _SS_PAD2SIZE = (
    _SS_MAXSIZE
    - size_of[UInt8]()
    - size_of[UInt8]()
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
    var ss_family: UInt8
    """The address family of the stored address (`AF_*`)."""
    var __ss_pad1: Array[c_char, _SS_PAD1SIZE]
    var __ss_align: Int64
    var __ss_pad2: Array[c_char, _SS_PAD2SIZE]
