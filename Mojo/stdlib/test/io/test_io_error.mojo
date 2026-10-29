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

from std.ffi import ErrNo, set_errno
from std.io import IOError, IOErrorKind
from std.net.addr import IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr
from std.testing import (
    TestSuite,
    assert_equal,
    assert_not_equal,
    assert_true,
)


def assert_kind(errno: ErrNo, expected: IOErrorKind) raises:
    assert_equal(
        IOError(errno=errno, operation="op").kind(), expected, msg=String(errno)
    )


def test_errno_kinds() raises:
    assert_kind(ErrNo.ENOENT, IOErrorKind.not_found)
    assert_kind(ErrNo.EPERM, IOErrorKind.permission_denied)
    assert_kind(ErrNo.EACCES, IOErrorKind.permission_denied)
    assert_kind(ErrNo.EEXIST, IOErrorKind.already_exists)
    assert_kind(ErrNo.EINVAL, IOErrorKind.invalid_input)
    assert_kind(ErrNo.ECONNREFUSED, IOErrorKind.connection_refused)
    assert_kind(ErrNo.ECONNRESET, IOErrorKind.connection_reset)
    assert_kind(ErrNo.ECONNABORTED, IOErrorKind.connection_aborted)
    assert_kind(ErrNo.ENOTCONN, IOErrorKind.not_connected)
    assert_kind(ErrNo.EADDRINUSE, IOErrorKind.address_in_use)
    assert_kind(ErrNo.EADDRNOTAVAIL, IOErrorKind.address_not_available)
    assert_kind(ErrNo.EHOSTUNREACH, IOErrorKind.host_unreachable)
    assert_kind(ErrNo.ENETUNREACH, IOErrorKind.network_unreachable)
    assert_kind(ErrNo.ENETDOWN, IOErrorKind.network_down)
    assert_kind(ErrNo.EPIPE, IOErrorKind.broken_pipe)
    assert_kind(ErrNo.EAGAIN, IOErrorKind.would_block)
    assert_kind(ErrNo.EWOULDBLOCK, IOErrorKind.would_block)
    assert_kind(ErrNo.ETIMEDOUT, IOErrorKind.timed_out)
    assert_kind(ErrNo.EINTR, IOErrorKind.interrupted)


def test_unmapped_errno_is_uncategorized() raises:
    assert_kind(ErrNo.ECHILD, IOErrorKind.uncategorized)
    assert_kind(ErrNo.EISCONN, IOErrorKind.uncategorized)
    assert_kind(ErrNo.EBADF, IOErrorKind.uncategorized)


def test_last_os_error() raises:
    set_errno(ErrNo.ENOENT)
    var err = IOError.last_os_error(operation="open")
    assert_equal(err.kind(), IOErrorKind.not_found)


def test_os_error_write_to() raises:
    var code = ErrNo.ECONNREFUSED
    var operation: StaticString = "connect"
    var cause = String(t"{code} (os error {code.value})")
    assert_true(cause.startswith("Connection refused (os error "))
    assert_equal(
        String(IOError(errno=code, operation=operation)),
        String(t"{operation}: {cause}"),
    )


def test_error_write_to() raises:
    var kind = IOErrorKind.write_zero
    var message: StaticString = "failed to write whole buffer"
    var operation: StaticString = "write"
    assert_equal(
        String(IOError(kind, message, operation=operation)),
        String(t"{operation}: {message}"),
    )


def test_os_error_write_repr_to() raises:
    var code = ErrNo.ENOENT
    var kind = IOErrorKind.not_found
    var operation: StaticString = "open"
    assert_equal(
        repr(IOError(errno=code, operation=operation)),
        String(
            t"IOError(operation={repr(operation)}, errno={code.value},"
            t" kind={repr(kind)}, message={repr(String(code))})"
        ),
    )


def test_error_write_repr_to() raises:
    var kind = IOErrorKind.write_zero
    var message: StaticString = "failed to write whole buffer"
    var operation: StaticString = "write"
    assert_equal(
        repr(IOError(kind, message, operation=operation)),
        String(
            t"IOError(operation={repr(operation)}, kind={repr(kind)},"
            t" message={repr(message)})"
        ),
    )


def test_write_to_with_path() raises:
    var code = ErrNo.ENOENT
    var operation: StaticString = "open"
    var path = "/tmp/x"
    assert_equal(
        String(
            IOError(errno=code, operation=operation)._with_subject(.path(path))
        ),
        String(t"{operation} {path}: {code} (os error {code.value})"),
    )


def test_write_to_with_addr() raises:
    var code = ErrNo.ECONNREFUSED
    var operation: StaticString = "connect"
    var cause = String(t"{code} (os error {code.value})")
    for addr in [
        SocketAddr(IpAddr.v4(Ipv4Addr.localhost), 8080),
        SocketAddr(IpAddr.v6(Ipv6Addr.localhost), 8080),
    ]:
        assert_equal(
            String(
                IOError(errno=code, operation=operation)._with_subject(
                    .addr(addr)
                )
            ),
            String(t"{operation} {addr}: {cause}"),
        )


def test_subject_keeps_kind() raises:
    set_errno(ErrNo.ENOENT)
    var err = IOError.last_os_error("open")._with_subject(.path("/tmp/x"))
    assert_equal(err.kind(), IOErrorKind.not_found)


def test_later_subject_replaces_earlier() raises:
    var message: StaticString = "failed to write whole buffer"
    var operation: StaticString = "write"
    var addr = SocketAddr(.v4(.localhost), 8080)
    assert_equal(
        String(
            IOError(IOErrorKind.write_zero, message, operation=operation)
            ._with_subject(.path("/tmp/x"))
            ._with_subject(.addr(addr))
        ),
        String(t"{operation} {addr}: {message}"),
    )


def test_write_repr_to_with_subject() raises:
    var code = ErrNo.ENOENT
    var kind = IOErrorKind.not_found
    var operation: StaticString = "connect"
    var path = "/tmp/x"
    var addr = SocketAddr(.v4(.localhost), 8080)
    var os_fields = String(
        t"errno={code.value}, kind={repr(kind)}, message={repr(String(code))}"
    )
    assert_equal(
        repr(
            IOError(errno=code, operation=operation)._with_subject(.path(path))
        ),
        String(
            t"IOError(operation={repr(operation)}, path={repr(path)},"
            t" {os_fields})"
        ),
    )
    assert_equal(
        repr(
            IOError(errno=code, operation=operation)._with_subject(.addr(addr))
        ),
        String(
            t"IOError(operation={repr(operation)},"
            t" addr={repr(String(addr))}, {os_fields})"
        ),
    )


def test_kind_write_to() raises:
    assert_equal(String(IOErrorKind.not_found), "IOErrorKind.not_found")
    assert_equal(String(IOErrorKind.uncategorized), "IOErrorKind.uncategorized")


def test_kind_equality() raises:
    assert_equal(IOErrorKind.not_found, IOErrorKind.not_found)
    assert_not_equal(IOErrorKind.not_found, IOErrorKind.permission_denied)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
