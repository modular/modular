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
"""Internet types and constants from `<netinet/in.h>` shared by Linux and macOS.

The constants are IANA-assigned or wire-format values, identical everywhere.
The struct mirrors are the shared POSIX layouts.
"""

from std.ffi import c_int

comptime in_port_t = UInt16
"""C `in_port_t`: a TCP/UDP port in network byte order."""

comptime in_addr_t = UInt32
"""C `in_addr_t`: an IPv4 address as a 32-bit integer."""

comptime IPPROTO_IP: c_int = 0
"""Dummy protocol for IP-level options."""
comptime IPPROTO_TCP: c_int = 6
"""Transmission Control Protocol."""
comptime IPPROTO_UDP: c_int = 17
"""User Datagram Protocol."""
comptime IPPROTO_IPV6: c_int = 41
"""IPv6-level options."""

comptime INADDR_ANY: in_addr_t = 0
"""Bind to all local interfaces (0.0.0.0), in host byte order."""
comptime INADDR_LOOPBACK: in_addr_t = 0x7F000001
"""The loopback address (127.0.0.1), in host byte order."""


@fieldwise_init
struct in_addr(TrivialRegisterPassable):
    """C `struct in_addr`: an IPv4 address."""

    var s_addr: in_addr_t
    """The address in network byte order."""


@align(4)
@fieldwise_init
struct in6_addr(Copyable):
    """C `struct in6_addr`: an IPv6 address.

    POSIX declares the storage as a union of u8/u16/u32 views, which gives
    the struct 4-byte alignment.
    """

    var s6_addr: Array[UInt8, 16]
    """The address bytes, in network byte order."""
