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
"""Provides `TCPStream`, a TCP connection.

Examples:

```mojo
from std.net import SocketAddrV4
from std.net.tcp import TCPListener, TCPStream

def main() raises:
    var listener = TCPListener(bind=.v4(SocketAddrV4(.localhost, 0)))
    var client = TCPStream(connect=listener.local_addr())
    var server, _ = listener.accept()
    client.write_all("ping".as_bytes())
    var buffer = List[Byte](length=4, fill=0)
    var n = server.read(buffer)
```
"""

from std.io import IOError

from ..addr import SocketAddr
from .._socket import Socket as _Socket
from ..option import GettableSocketOption, SettableSocketOption
from .socket import TCPSocket


struct TCPStream(Movable):
    """A TCP connection.

    Create one with `TCPStream(connect=)`, `TCPSocket.connect()`, or
    `TCPListener.accept()`.

    Examples:

    ```mojo
    from std.net import SocketAddrV4
    from std.net.tcp import TCPStream

    def main() raises:
        var stream = TCPStream(connect=.v4(SocketAddrV4(.localhost, 8080)))
        stream.write_all("ping".as_bytes())
        var buffer = List[Byte](length=1024, fill=0)
        var n = stream.read(buffer)
        stream^.close()
    ```
    """

    var _socket: _Socket

    @doc_hidden
    def __init__(out self, var socket: _Socket):
        """Takes ownership of a connected TCP socket.

        Args:
            socket: The socket.
        """
        self._socket = socket^

    def __init__(out self, *, connect: SocketAddr) raises IOError:
        """Connects to a remote address.

        Args:
            connect: The remote address.

        Raises:
            If the connection fails, such as when nothing listens at
            `connect`.
        """
        self = TCPSocket._for_addr(connect).connect(connect)

    def __deinit__(deinit self):
        """Closes the connection, ignoring any error."""
        try:
            self._socket^.close()
        except:
            pass

    def close(deinit self) raises IOError:
        """Closes the connection, consuming the stream.

        Raises:
            If the close fails.
        """
        self._socket^.close()

    def read(self, buffer: MutSpan[Byte, _]) raises IOError -> Int:
        """Reads bytes into `buffer`, waiting until at least one arrives.

        Args:
            buffer: Where to put the bytes.

        Returns:
            How many bytes it read: 0 once the peer has closed its end, or if
            `buffer` is empty.

        Raises:
            If the read fails, such as when the peer resets the connection.
        """
        # TODO: on Linux, a `recv` into an empty buffer waits for data to
        # arrive instead of returning 0. Return 0 without calling `recv` when
        # `buffer` is empty.
        return self._socket.recv(buffer)

    def write(self, bytes: ImmSpan[Byte, _]) raises IOError -> Int:
        """Writes some of `bytes`, waiting until at least one can be written.

        To write all of them, use `write_all()`.

        Args:
            bytes: The bytes to write.

        Returns:
            How many bytes it wrote, which can be fewer than `len(bytes)`.

        Raises:
            If the write fails, such as when the peer has closed the
            connection.
        """
        # TODO(https://github.com/modular/modular/issues/7475): a write to a
        # peer that has closed raises `SIGPIPE`, which ends the process instead
        # of raising an error, until `Socket.send()` suppresses it.
        return self._socket.send(bytes)

    def write_all(self, bytes: ImmSpan[Byte, _]) raises IOError:
        """Writes all of `bytes` by calling `write()` in a loop.

        Args:
            bytes: The bytes to write.

        Raises:
            If a write fails, such as when the peer has closed the connection.
            Some of `bytes` may have been written before the error.
        """
        var written = 0
        while written < len(bytes):
            var n = self.write(bytes[written:])
            if n == 0:
                raise IOError(.write_zero, "wrote 0 bytes", operation="send")
            written += n

    def local_addr(self) raises IOError -> SocketAddr:
        """Returns the local address of the connection.

        Returns:
            The address.

        Raises:
            If the address can't be read.
        """
        return self._socket.local_addr()

    def peer_addr(self) raises IOError -> SocketAddr:
        """Returns the address of the peer.

        Returns:
            The address.

        Raises:
            If the address can't be read, such as when the peer has reset the
            connection.
        """
        return self._socket.peer_addr()

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
