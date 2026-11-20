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
"""Owns a socket descriptor and applies the setup every socket needs."""

from std.ffi import c_int, c_size_t
from std.io import FileDescriptor, IOError
from std.io._syscall import check, retry_io_interrupted
from std.net.addr import SocketAddr
from std.sys import size_of

from ._sys import unix
from ._sys.socket_addr import (
    from_sockaddr_storage,
    sockaddr_len,
    to_sockaddr_storage,
)
from ._sys.unix import (
    accept_socket,
    getsockopt,
    open_socket,
    setsockopt,
    sockaddr,
    sockaddr_storage,
    socklen_t,
)


struct Socket(not Deinitable, Movable):
    """An open socket, closed when the value is destroyed.

    New sockets are close-on-exec; see `open_socket` for the full setup.
    """

    var _fd: FileDescriptor

    def __init__(out self, var fd: FileDescriptor):
        """Takes ownership of an open socket descriptor."""
        self._fd = fd^

    def __init__(
        out self, *, domain: c_int, type: c_int, protocol: c_int = 0
    ) raises IOError:
        """Opens a socket of the given domain, type, and protocol, where
        protocol 0 picks the default for the domain and type."""
        self._fd = open_socket(domain, type, protocol)

    def close(deinit self) raises IOError:
        """Closes the socket."""
        self._fd^.close()

    # TODO(https://github.com/modular/modular/issues/7477): this really
    # shouldn't be "unsafe". Getting the `c_int` of the underlying descriptor is
    # perfectly safe. The only unsafe thing would be initializing a Socket/FD
    # from a raw `c_int`.
    def unsafe_fd(self) -> c_int:
        """Returns the descriptor, which stays owned by this socket.

        It's valid only while the socket is alive, and Mojo destroys the socket
        at its last use, which can be the `unsafe_fd()` call itself.
        """
        return self._fd._value._value

    def set_option(
        self, *, level: c_int, name: c_int, value: c_int
    ) raises IOError:
        """Sets an integer socket option."""
        var v = value
        _ = check(
            setsockopt(
                self.unsafe_fd(),
                level,
                name,
                Pointer(to=v).unsafe_bitcast[NoneType](),
                socklen_t(size_of[c_int]()),
            ),
            "setsockopt",
        )

    def option(self, *, level: c_int, name: c_int) raises IOError -> c_int:
        """Returns an integer socket option."""
        var value: c_int = 0
        var len = socklen_t(size_of[c_int]())
        _ = check(
            getsockopt(
                self.unsafe_fd(),
                level,
                name,
                Pointer(to=value).unsafe_bitcast[NoneType](),
                Pointer(to=len),
            ),
            "getsockopt",
        )
        return value

    def set_flag(
        self, *, level: c_int, name: c_int, enabled: Bool
    ) raises IOError:
        """Turns a boolean socket option on or off."""
        self.set_option(
            level=level, name=name, value=c_int(1) if enabled else c_int(0)
        )

    def flag(self, *, level: c_int, name: c_int) raises IOError -> Bool:
        """Returns whether a boolean socket option is on."""
        # macOS reports a set flag as its option bit, not 1.
        return self.option(level=level, name=name) != 0

    def bind(self, addr: SocketAddr) raises IOError:
        """Binds the socket to a local address."""
        var storage = to_sockaddr_storage(addr)
        if (
            unix.bind(
                self.unsafe_fd(),
                Pointer(to=storage).unsafe_bitcast[sockaddr](),
                sockaddr_len(addr),
            )
            == -1
        ):
            raise IOError.last_os_error("bind")._with_subject(.addr(addr))

    def listen(self, *, backlog: c_int) raises IOError:
        """Accepts connections, queueing up to `backlog` of them."""
        _ = check(unix.listen(self.unsafe_fd(), backlog), "listen")

    def connect(self, addr: SocketAddr) raises IOError:
        """Connects the socket to a remote address."""
        var storage = to_sockaddr_storage(addr)

        # TODO(https://github.com/modular/modular/issues/7478): return nothing
        # once the compiler can infer `None` as the result type of a closure
        # passed to `retry_io_interrupted`.
        def attempt() raises IOError {imm} -> c_int:
            var result = unix.connect(
                self.unsafe_fd(),
                Pointer(to=storage).unsafe_bitcast[sockaddr](),
                sockaddr_len(addr),
            )
            if result == -1:
                raise IOError.last_os_error("connect")._with_subject(
                    .addr(addr)
                )
            return result

        # TODO(https://github.com/modular/modular/issues/7474): an interrupted
        # `connect` carries on in the kernel. On macOS the retry fails with
        # `EISCONN` even while the connection is still pending, so this can
        # raise for a connection that succeeds; treating `EISCONN` as success,
        # as Rust's std does, would instead return before the socket is
        # connected. After `EINTR`, `poll` for `POLLOUT` and read the outcome
        # from `SO_ERROR`, as CPython does.
        _ = retry_io_interrupted(attempt)

    def accept(self) raises IOError -> Tuple[Socket, SocketAddr]:
        """Accepts a connection, and returns its socket and the peer's
        address."""
        match accept_socket(self.unsafe_fd()):
            case (var fd, var addr):
                return (Socket(fd^), addr)

    def local_addr(self) raises IOError -> SocketAddr:
        """Returns the address the socket is bound to."""
        var storage = sockaddr_storage()
        var len = socklen_t(size_of[sockaddr_storage]())
        _ = check(
            unix.getsockname(
                self.unsafe_fd(),
                Pointer(to=storage).unsafe_bitcast[sockaddr](),
                Pointer(to=len),
            ),
            "getsockname",
        )
        return from_sockaddr_storage(storage, "getsockname")

    def peer_addr(self) raises IOError -> SocketAddr:
        """Returns the address of the connected peer."""
        var storage = sockaddr_storage()
        var len = socklen_t(size_of[sockaddr_storage]())
        _ = check(
            unix.getpeername(
                self.unsafe_fd(),
                Pointer(to=storage).unsafe_bitcast[sockaddr](),
                Pointer(to=len),
            ),
            "getpeername",
        )
        return from_sockaddr_storage(storage, "getpeername")

    def recv(
        self, buffer: MutSpan[Byte, _], *, flags: c_int = 0
    ) raises IOError -> Int:
        """Reads into `buffer` with the given `MSG_*` flags, and returns how
        many bytes it read: 0 once the peer has closed its end."""

        # TODO(https://github.com/modular/modular/issues/7476): macOS fails a
        # `recv` of more than `INT_MAX` bytes with `EINVAL` instead of reading
        # less. Cap the length there.
        def attempt() raises IOError {imm} -> Int:
            return check(
                unix.recv(
                    self.unsafe_fd(),
                    buffer.unsafe_ptr(),
                    c_size_t(len(buffer)),
                    flags,
                ),
                "recv",
            )

        return retry_io_interrupted(attempt)

    def send(
        self, bytes: ImmSpan[Byte, _], *, flags: c_int = 0
    ) raises IOError -> Int:
        """Writes some of `bytes` with the given `MSG_*` flags, and returns how
        many it wrote.

        macOS ignores `MSG_DONTWAIT` here, and blocks on a full buffer anyway.
        """

        # TODO(https://github.com/modular/modular/issues/7475): a send to a peer
        # that has closed raises `SIGPIPE`, which ends the process by default.
        # Suppress it with `MSG_NOSIGNAL` on each send on Linux, and with
        # `SO_NOSIGPIPE` on macOS, set in `open_socket` and `accept_socket`.
        # TODO(https://github.com/modular/modular/issues/7476): macOS fails a
        # `send` of more than `INT_MAX` bytes with `EINVAL` instead of writing
        # less. Cap the length there.
        def attempt() raises IOError {imm} -> Int:
            return check(
                unix.send(
                    self.unsafe_fd(),
                    bytes.unsafe_ptr(),
                    c_size_t(len(bytes)),
                    flags,
                ),
                "send",
            )

        return retry_io_interrupted(attempt)
