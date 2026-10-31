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

from std.ffi import c_int
from std.io import FileDescriptor, IOError
from std.io._syscall import check
from std.sys import size_of

from ._sys.unix import getsockopt, open_socket, setsockopt, socklen_t


struct Socket(not Deinitable, Movable):
    """An open socket, closed when the value is destroyed.

    New sockets are close-on-exec; see `open_socket` for the full setup.
    """

    var _fd: FileDescriptor

    def __init__(out self, var fd: FileDescriptor):
        """Takes ownership of an open socket descriptor."""
        self._fd = fd^

    def __init__(out self, *, domain: c_int, type: c_int) raises IOError:
        """Opens a socket of the given domain and type."""
        self._fd = open_socket(domain, type, 0)

    def close(deinit self) raises IOError:
        """Closes the socket."""
        self._fd^.close()

    def unsafe_fd(self) -> c_int:
        """Returns the descriptor, which stays owned by this socket.

        It's valid only while the socket is alive, and Mojo destroys the socket
        at its last use, which can be the `unsafe_fd()` call itself.
        """
        return self._fd._value._value

    def set_option(
        self, level: c_int, name: c_int, value: c_int
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

    def option(self, level: c_int, name: c_int) raises IOError -> c_int:
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
