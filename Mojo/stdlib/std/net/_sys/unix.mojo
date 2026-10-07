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
"""Socket constants shared by the Unix platforms, Linux and macOS.

The values are transcribed from `<sys/socket.h>`, `<netinet/tcp.h>`, and
`<fcntl.h>`. Every name here is defined on both platforms. A value they
share is a plain literal, and one that differs resolves for the compilation
target, so using it on any other target is a compile error. Names that only
one platform defines live in `linux` or `macos`.
"""

from std.ffi import c_int
from std.sys.info import platform_map


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
comptime AF_INET6: c_int = constant["AF_INET6", linux=10, macos=30]()
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
# `send` and `recv` flags
# ===----------------------------------------------------------------------=== #

comptime MSG_PEEK: c_int = 0x2
"""Peek at incoming data without consuming it."""
comptime MSG_DONTWAIT: c_int = constant[
    "MSG_DONTWAIT", linux=0x40, macos=0x80
]()
"""Non-blocking operation for this call only."""
comptime MSG_NOSIGNAL: c_int = constant[
    "MSG_NOSIGNAL", linux=0x4000, macos=0x80000
]()
"""Don't raise `SIGPIPE` on send to a closed peer."""

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
