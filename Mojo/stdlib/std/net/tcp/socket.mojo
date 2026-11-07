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
"""Provides `TCPSocket`, a TCP socket to configure.

Examples:

```mojo
from std.net.tcp import TCPSocket
from std.net.tcp.option import NoDelay

def main() raises:
    var socket = TCPSocket.v4()
    socket.set_option[NoDelay](True)
```
"""

from std.io import IOError

from .._socket import Socket as _Socket
from .._sys.unix import (
    AF_INET as _AF_INET,
    AF_INET6 as _AF_INET6,
    SOCK_STREAM as _SOCK_STREAM,
)
from ..option import (
    GettableSocketOption,
    SettableSocketOption,
    TCPSocketOption,
)


struct TCPSocket(Movable):
    """A TCP socket to configure.

    Create one with `TCPSocket.v4()` or `TCPSocket.v6()`, then set its options
    with `set_option()`, such as `set_option[NoDelay](True)`.

    Examples:

    ```mojo
    from std.net.tcp import TCPSocket
    from std.net.tcp.option import ReusePort

    def main() raises:
        var socket = TCPSocket.v4()
        socket.set_option[ReusePort](True)
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
        Option: TCPSocketOption & SettableSocketOption
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
        Option: TCPSocketOption & GettableSocketOption
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
