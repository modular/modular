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
"""Provides the traits that describe socket options, and the options that more
than one kind of socket accepts.

Each protocol's `option` module, such as `std.net.tcp.option`, re-exports the
shared options that its sockets accept.
"""

from std.ffi import c_int
from std.io import IOError

from ._socket import Socket as _Socket
from ._sys.unix import (
    SOL_SOCKET as _SOL_SOCKET,
    SO_REUSEADDR as _SO_REUSEADDR,
    SO_REUSEPORT as _SO_REUSEPORT,
)


trait SocketOption:
    """Describes a socket option by the type of its value.

    An option conforms to `GettableSocketOption` if it can be read, to
    `SettableSocketOption` if it can be set, and to a marker trait, such as
    `TCPSocketOption`, for each kind of socket that accepts it.
    """

    comptime OptionType: Movable & Deinitable
    """The type of the option's value."""


trait GettableSocketOption(SocketOption):
    """Describes a socket option that can be read from a socket."""

    @staticmethod
    def get_option(socket: _Socket) raises IOError -> Self.OptionType:
        """Returns the option's value on a socket.

        Args:
            socket: The socket to read it from.

        Returns:
            The option's value.

        Raises:
            If the option can't be read.
        """
        ...


trait SettableSocketOption(SocketOption):
    """Describes a socket option that can be set on a socket."""

    @staticmethod
    def set_option(socket: _Socket, var option: Self.OptionType) raises IOError:
        """Sets the option on a socket.

        Args:
            socket: The socket to set it on.
            option: The value to set.

        Raises:
            If the option can't be set.
        """
        ...


trait TCPSocketOption(SocketOption):
    """Marks a socket option that `TCPSocket` accepts."""

    pass


struct FlagOption[
    name: StaticString,
    level: c_int,
    number: c_int,
    *,
    tcp_socket: Bool = False,
](GettableSocketOption, SettableSocketOption, TCPSocketOption where tcp_socket):
    """Represents a socket option that is either on or off.

    Each option of this kind is a `comptime` alias of this type, such as
    `ReuseAddress`.

    Parameters:
        name: The option's name, such as `"ReuseAddress"`.
        level: The level the option belongs to, as passed to `setsockopt`.
        number: The option's number within its level, which `setsockopt`
            takes as `option_name`.
        tcp_socket: Whether `TCPSocket` accepts the option.
    """

    comptime OptionType = Bool
    """Whether the option is on."""

    @staticmethod
    def set_option(socket: _Socket, var option: Bool) raises IOError:
        """Turns the option on or off.

        Args:
            socket: The socket to set it on.
            option: Whether to turn it on.

        Raises:
            If the option can't be set.
        """
        socket.set_flag(level=Self.level, name=Self.number, enabled=option)

    @staticmethod
    def get_option(socket: _Socket) raises IOError -> Bool:
        """Returns whether the option is on.

        Args:
            socket: The socket to read it from.

        Returns:
            Whether it's on.

        Raises:
            If the option can't be read.
        """
        return socket.flag(level=Self.level, name=Self.number)


comptime ReuseAddress = FlagOption[
    "ReuseAddress", _SOL_SOCKET, _SO_REUSEADDR, tcp_socket=True
]
"""Represents `SO_REUSEADDR`: whether the socket can bind a local address that
connections from an earlier socket still hold.

A server that restarts needs it to bind its port again while connections from
its last run are still closing.
"""

comptime ReusePort = FlagOption[
    "ReusePort", _SOL_SOCKET, _SO_REUSEPORT, tcp_socket=True
]
"""Represents `SO_REUSEPORT`: whether other sockets can bind the same address
and port."""
