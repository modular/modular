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
"""Defines `IOError`, the error type for failed I/O operations, and
`IOErrorKind`, its portable classification.

You can import these APIs from the `io` package. For example:

```mojo
from std.io import IOError, IOErrorKind
```
"""

import std.format._utils as fmt
from std.ffi import ErrNo, get_errno
from std.hash.hasher import Hasher


__enum IOErrorKind(Equatable, Hashable, ImplicitlyCopyable, Writable):
    """Classifies an `IOError` independently of the platform.

    The enum is open: later releases can add kinds, so a `match` over it is
    exhaustive only with a `case _` arm.
    """

    case not_found
    """An entity, such as a file, wasn't found."""
    case permission_denied
    """The operation lacked the necessary privileges."""
    case already_exists
    """An entity, such as a file, already exists."""
    case invalid_input
    """An argument was invalid."""
    case write_zero
    """A write that had to make progress wrote zero bytes."""
    case connection_refused
    """The remote host refused the connection."""
    case connection_reset
    """The remote host reset the connection."""
    case connection_aborted
    """The local host aborted the connection."""
    case not_connected
    """The operation needs a connected socket, and the socket isn't
    connected."""
    case address_in_use
    """The local address is already in use."""
    case address_not_available
    """The requested address isn't local or doesn't exist."""
    case host_unreachable
    """The remote host couldn't be reached."""
    case network_unreachable
    """The network containing the remote host couldn't be reached."""
    case network_down
    """The network is down."""
    case broken_pipe
    """The other end of a pipe or connection has closed."""
    case would_block
    """The operation would block, and blocking wasn't requested."""
    case timed_out
    """The operation timed out."""
    case interrupted
    """The operation was interrupted, typically by a signal."""
    case uncategorized
    """An operating system error that no other kind covers."""
    # TODO: Add more ErrNo/IOErrorKind mappings

    comptime _enum_is_exhaustive = False

    def __eq__(self, other: Self) -> Bool:
        """Returns whether two kinds are the same.

        Args:
            other: The kind to compare against.

        Returns:
            True if both are the same kind.
        """
        return self._get_enum_discriminant() == other._get_enum_discriminant()

    def __hash__(self, mut hasher: Some[Hasher]):
        """Hashes the kind using the given hasher.

        Args:
            hasher: The hasher instance.
        """
        self._get_enum_discriminant().__hash__(hasher)


def _kind_from_errno(errno: ErrNo) -> IOErrorKind:
    match errno:
        case == ErrNo.ENOENT:
            return .not_found
        case == ErrNo.EPERM | == ErrNo.EACCES:
            return .permission_denied
        case == ErrNo.EEXIST:
            return .already_exists
        case == ErrNo.EINVAL:
            return .invalid_input
        case == ErrNo.ECONNREFUSED:
            return .connection_refused
        case == ErrNo.ECONNRESET:
            return .connection_reset
        case == ErrNo.ECONNABORTED:
            return .connection_aborted
        case == ErrNo.ENOTCONN:
            return .not_connected
        case == ErrNo.EADDRINUSE:
            return .address_in_use
        case == ErrNo.EADDRNOTAVAIL:
            return .address_not_available
        case == ErrNo.EHOSTUNREACH:
            return .host_unreachable
        case == ErrNo.ENETUNREACH:
            return .network_unreachable
        case == ErrNo.ENETDOWN:
            return .network_down
        case == ErrNo.EPIPE:
            return .broken_pipe
        case == ErrNo.EAGAIN:
            return .would_block
        case == ErrNo.ETIMEDOUT:
            return .timed_out
        case == ErrNo.EINTR:
            return .interrupted
        # TODO: Add more ErrNo cases
        case _:
            return .uncategorized


@fieldwise_init
struct _Custom(ImplicitlyCopyable):
    var kind: IOErrorKind
    var message: StaticString


__enum _Cause(ImplicitlyCopyable):
    case os(ErrNo)
    case custom(_Custom)


struct IOError(ImplicitlyCopyable, Writable):
    """Represents a failed I/O operation.

    Examples:

    ```mojo
    from std.io import IOError

    def read_file(path: String) raises IOError -> String:
        raise IOError(.not_found, "file not found", operation="open")

    try:
        _ = read_file("foo.txt")
    except e:
        match e.kind():
            case .not_found:
                print("missing:", e)  # missing: open: file not found
            case .permission_denied:
                print("access denied:", e)
            case _:
                print("failed:", e)
    ```
    """

    var _cause: _Cause
    var _operation: StaticString

    def __init__(
        out self,
        kind: IOErrorKind,
        message: StaticString,
        *,
        operation: StaticString,
    ):
        """Constructs an error that didn't come from the operating system.

        Args:
            kind: The kind of error.
            message: A description of the error.
            operation: The name of the operation that failed.
        """
        self._cause = .custom(_Custom(kind, message))
        self._operation = operation

    @doc_hidden
    def __init__(out self, *, errno: ErrNo, operation: StaticString):
        """Constructs an error from an operating system error code.

        Args:
            errno: The error code.
            operation: The name of the operation that failed.
        """
        self._cause = .os(errno)
        self._operation = operation

    @staticmethod
    def last_os_error(operation: StaticString) -> Self:
        """Returns an error for the calling thread's current `errno`.

        Args:
            operation: The name of the operation that failed.

        Returns:
            The error for the current `errno`.
        """
        return Self(errno=get_errno(), operation=operation)

    def kind(self) -> IOErrorKind:
        """Returns the kind of error.

        Returns:
            The kind passed to the constructor, or for an operating system
            error, the kind that its platform error maps to.
        """
        match self._cause:
            case .os(errno):
                return _kind_from_errno(errno)
            case .custom(custom):
                return custom.kind

    def write_to(self, mut writer: Some[Writer]):
        """Writes the error to a writer.

        Args:
            writer: The writer to write to.
        """
        t"{self._operation}: ".write_to(writer)
        match self._cause:
            case .os(errno):
                t"{errno} (os error {errno.value})".write_to(writer)
            case .custom(custom):
                custom.message.write_to(writer)

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the error's debug representation to a writer.

        Args:
            writer: The writer to write to.
        """
        match self._cause:
            case .os(errno):
                fmt.FormatStruct(writer, "IOError").fields(
                    fmt.Named("operation", fmt.Repr(self._operation)),
                    fmt.Named("errno", errno.value),
                    fmt.Named("kind", fmt.Repr(_kind_from_errno(errno))),
                    fmt.Named("message", t"'{errno}'"),
                )
            case .custom(custom):
                fmt.FormatStruct(writer, "IOError").fields(
                    fmt.Named("operation", fmt.Repr(self._operation)),
                    fmt.Named("kind", fmt.Repr(custom.kind)),
                    fmt.Named("message", fmt.Repr(custom.message)),
                )
