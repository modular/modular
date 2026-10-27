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
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
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
    var expected = String(t"{code} (os error {code.value})")
    assert_true(expected.startswith("Connection refused (os error "))
    assert_equal(
        String(IOError(errno=code, operation="connect")),
        String(t"connect: {expected}"),
    )


def test_error_write_to() raises:
    assert_equal(
        String(
            IOError(
                IOErrorKind.write_zero,
                "failed to write whole buffer",
                operation="write",
            )
        ),
        "write: failed to write whole buffer",
    )


def test_os_error_write_repr_to() raises:
    var code = ErrNo.ENOENT
    assert_equal(
        repr(IOError(errno=code, operation="open")),
        String(
            t"IOError(operation='open', errno={code.value},"
            t" kind=IOErrorKind.not_found, message='{code}')"
        ),
    )


def test_error_write_repr_to() raises:
    assert_equal(
        repr(
            IOError(
                IOErrorKind.write_zero,
                "failed to write whole buffer",
                operation="write",
            )
        ),
        (
            "IOError(operation='write', kind=IOErrorKind.write_zero,"
            " message='failed to write whole buffer')"
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
