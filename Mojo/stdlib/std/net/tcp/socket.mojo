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
"""Provides `TCPSocket`, a TCP socket to configure before it listens.

Examples:

```mojo
from std.net import SocketAddrV4
from std.net.tcp import TCPSocket
from std.net.tcp.option import NoDelay

def main() raises:
    var socket = TCPSocket.v4()
    socket.set_option[NoDelay](True)
    socket.bind(.v4(SocketAddrV4(.localhost, 0)))
    var listener = socket^.listen()
```
"""

from std.ffi import c_int
from std.io import IOError

from ..addr import SocketAddr
from .._socket import Socket as _Socket
from .._sys.unix import (
    AF_INET as _AF_INET,
    AF_INET6 as _AF_INET6,
    SOCK_STREAM as _SOCK_STREAM,
)
from ..option import GettableSocketOption, SettableSocketOption
from .listener import TCPListener


struct TCPSocket(Movable):
    """A TCP socket to configure before it listens.

    Create one with `TCPSocket.v4()` or `TCPSocket.v6()`, then set its options
    with `set_option()`, such as `set_option[NoDelay](True)`. `bind()` then
    binds it to a local address, and `listen()` turns it into a
    `TCPListener`.

    Examples:

    ```mojo
    from std.net import SocketAddrV4
    from std.net.tcp import TCPSocket
    from std.net.tcp.option import ReusePort

    def main() raises:
        var socket = TCPSocket.v4()
        socket.set_option[ReusePort](True)
        socket.bind(.v4(SocketAddrV4(.localhost, 0)))
        var listener = socket^.listen()
    ```
    """

    var _socket: _Socket

    @doc_hidden
    def __init__(out self, var socket: _Socket):
        """Takes ownership of an open TCP socket.

        Args:
            socket: The socket.
        """
        self._socket = socket^

    @staticmethod
    def v4() raises IOError -> Self:
        """Opens an IPv4 TCP socket.

        Returns:
            The socket.

        Raises:
            If the socket can't be opened.
        """
        return Self(_Socket(domain=_AF_INET, type=_SOCK_STREAM))

    @staticmethod
    def v6() raises IOError -> Self:
        """Opens an IPv6 TCP socket.

        Returns:
            The socket.

        Raises:
            If the socket can't be opened.
        """
        return Self(_Socket(domain=_AF_INET6, type=_SOCK_STREAM))

    @staticmethod
    def _for_addr(addr: SocketAddr) raises IOError -> Self:
        match addr:
            case .v4(_):
                return Self.v4()
            case .v6(_):
                return Self.v6()

    def __deinit__(deinit self):
        """Closes the socket, ignoring any error."""
        try:
            self._socket^.close()
        except:
            pass

    def close(deinit self) raises IOError:
        """Closes the socket, consuming it.

        Raises:
            If the close fails.
        """
        self._socket^.close()

    def set_option[
        Option: SettableSocketOption
    ](self, var option: Option.OptionType) raises IOError:
        """Sets a socket option.

        Parameters:
            Option: The option to set, such as `NoDelay`.

        Args:
            option: The option's new value.

        Raises:
            If the option can't be set.
        """
        Option.set_option(self._socket, option^)

    def get_option[
        Option: GettableSocketOption
    ](self) raises IOError -> Option.OptionType:
        """Returns a socket option's value.

        Parameters:
            Option: The option to read, such as `NoDelay`.

        Returns:
            The option's value.

        Raises:
            If the option can't be read.
        """
        return Option.get_option(self._socket)

    def bind(self, addr: SocketAddr) raises IOError:
        """Binds the socket to a local address.

        Unlike `TCPListener(bind=)`, it leaves `ReuseAddress` as it is, so set
        that option first to bind a port that connections from an earlier
        socket still hold.

        Args:
            addr: The local address, of the socket's IP version. Port 0 picks a
                free port, which `TCPListener.local_addr()` returns.

        Raises:
            If the bind fails, such as when another socket holds the address
            or `addr` isn't of the socket's IP version.
        """
        self._socket.bind(addr)

    def listen(
        deinit self, *, backlog: Int = 128
    ) raises IOError -> TCPListener:
        """Listens for connections, consuming the socket.

        Args:
            backlog: How many connections can wait to be accepted.

        Returns:
            The listener.

        Raises:
            If the listen fails.
        """
        # The listener owns the socket from here, so it closes on any error.
        var listener = TCPListener(self._socket^)
        # TODO(https://github.com/modular/modular/issues/7509): Reject a
        # `backlog` outside the `c_int` range instead of wrapping it.
        listener._socket.listen(backlog=c_int(backlog))
        return listener^
