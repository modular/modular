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
"""Converts socket addresses to and from the C `sockaddr_in`,
`sockaddr_in6`, and `sockaddr_storage` the socket syscalls take and
return."""

from std.io import IOError
from std.net.addr import (
    Ipv4Addr,
    Ipv6Addr,
    SocketAddr,
    SocketAddrV4,
    SocketAddrV6,
)
from std.sys import size_of

from .inet import in6_addr, in_addr
from .unix import (
    AF_INET,
    AF_INET6,
    sockaddr_in,
    sockaddr_in6,
    sockaddr_storage,
    socklen_t,
)


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


def to_sockaddr_storage(addr: SocketAddr) -> sockaddr_storage:
    var storage = sockaddr_storage()
    match addr:
        case .v4(v4):
            Pointer(to=storage).unsafe_bitcast[
                sockaddr_in
            ]()[] = to_sockaddr_in(v4)
        case .v6(v6):
            Pointer(to=storage).unsafe_bitcast[
                sockaddr_in6
            ]()[] = to_sockaddr_in6(v6)
    return storage^


def sockaddr_len(addr: SocketAddr) -> socklen_t:
    """Returns the size of the C address `to_sockaddr_storage` writes."""
    match addr:
        case .v4(_):
            return socklen_t(size_of[sockaddr_in]())
        case .v6(_):
            return socklen_t(size_of[sockaddr_in6]())


def from_sockaddr_storage(
    storage: sockaddr_storage, operation: StaticString
) raises IOError -> SocketAddr:
    """Returns the IPv4 or IPv6 address in `storage`, and raises an
    `invalid_input` error for `operation` on any other address family."""
    match storage.family():
        case == AF_INET:
            return SocketAddr.v4(
                from_sockaddr_in(
                    Pointer(to=storage).unsafe_bitcast[sockaddr_in]()[]
                )
            )
        case == AF_INET6:
            return SocketAddr.v6(
                from_sockaddr_in6(
                    Pointer(to=storage).unsafe_bitcast[sockaddr_in6]()[]
                )
            )
        case _:
            raise IOError(
                .invalid_input,
                "address isn't IPv4 or IPv6",
                operation=operation,
            )
