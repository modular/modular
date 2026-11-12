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
"""Provides `TCPListener`, a TCP socket that listens for connections.

Examples:

```mojo
from std.net import SocketAddrV4
from std.net.tcp import TCPListener

def main() raises:
    var listener = TCPListener(bind=.v4(SocketAddrV4(.localhost, 0)))
    var addr = listener.local_addr()
```
"""

from std.io import IOError

from ..addr import SocketAddr
from .._socket import Socket as _Socket
from ..option import GettableSocketOption, ReuseAddress, SettableSocketOption
from .socket import TCPSocket


struct TCPListener(Movable):
    """A TCP socket that listens for connections.

    Examples:

    ```mojo
    from std.net import SocketAddrV4
    from std.net.tcp import TCPListener

    def main() raises:
        var listener = TCPListener(bind=.v4(SocketAddrV4(.localhost, 0)))
        print(listener.local_addr())
    ```
    """

    var _socket: _Socket

    @doc_hidden
    def __init__(out self, var socket: _Socket):
        """Takes ownership of a listening TCP socket.

        Args:
            socket: The socket.
        """
        self._socket = socket^

    def __init__(
        out self, *, bind: SocketAddr, backlog: Int = 128
    ) raises IOError:
        """Binds to a local address and listens for connections.

        The socket's IP version matches `bind`. It turns on `ReuseAddress`, so
        a server that restarts can bind its port again while connections from
        its last run are still closing. To choose the options yourself, use
        `TCPSocket` instead.

        Args:
            bind: The local address. Port 0 picks a free port, which
                `local_addr()` returns.
            backlog: How many connections can wait to be accepted.

        Raises:
            If the bind or listen fails, such as when another socket holds the
            address.
        """
        var socket = TCPSocket._for_addr(bind)
        socket.set_option[ReuseAddress](True)
        socket.bind(bind)
        self = socket^.listen(backlog=backlog)

    def __deinit__(deinit self):
        """Closes the listener, ignoring any error."""
        try:
            self._socket^.close()
        except:
            pass

    def close(deinit self) raises IOError:
        """Closes the listener, consuming it.

        Raises:
            If the close fails.
        """
        self._socket^.close()

    def local_addr(self) raises IOError -> SocketAddr:
        """Returns the address the listener is bound to.

        Returns:
            The address, including the port the system picked if the listener
            was bound to port 0.

        Raises:
            If the address can't be read.
        """
        return self._socket.local_addr()

    def set_option[
        Option: SettableSocketOption
    ](self, var option: Option.OptionType) raises IOError:
        """Sets a socket option.

        Options that take effect when the socket binds, such as
        `ReuseAddress`, need to be set on a `TCPSocket` before `bind()`.

        Parameters:
            Option: The option to set, such as `KeepAlive`.

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
            Option: The option to read, such as `ReuseAddress`.

        Returns:
            The option's value.

        Raises:
            If the option can't be read.
        """
        return Option.get_option(self._socket)
