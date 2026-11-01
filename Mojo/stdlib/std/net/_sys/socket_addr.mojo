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
"""Converts socket addresses to and from the C `sockaddr_in` and
`sockaddr_in6` the socket syscalls take and return."""

from std.net.addr import Ipv4Addr, Ipv6Addr, SocketAddrV4, SocketAddrV6

from .inet import in6_addr, in_addr
from .unix import sockaddr_in, sockaddr_in6


def to_sockaddr_in(addr: SocketAddrV4) -> sockaddr_in:
    return sockaddr_in(
        port=addr.port().to_endian[.big](),
        addr=in_addr(addr.ip().bits().to_endian[.big]()),
    )


def from_sockaddr_in(addr: sockaddr_in) -> SocketAddrV4:
    return SocketAddrV4(
        Ipv4Addr(bits=.from_endian[.big](addr.addr().s_addr)),
        UInt16.from_endian[.big](addr.port()),
    )


# The flow info and scope ID pass through unconverted: the scope ID is a
# host-order interface index, and RFC 3493 leaves the flow info's byte order
# unspecified.
def to_sockaddr_in6(addr: SocketAddrV6) -> sockaddr_in6:
    return sockaddr_in6(
        port=addr.port().to_endian[.big](),
        addr=in6_addr(addr.ip().octets()),
        flowinfo=addr.flowinfo(),
        scope_id=addr.scope_id(),
    )


def from_sockaddr_in6(addr: sockaddr_in6) -> SocketAddrV6:
    return SocketAddrV6(
        Ipv6Addr(octets=addr.addr().s6_addr.copy()),
        .from_endian[.big](addr.port()),
        flowinfo=addr.flowinfo(),
        scope_id=addr.scope_id(),
    )
