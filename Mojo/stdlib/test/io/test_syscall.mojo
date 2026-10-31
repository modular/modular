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

from std.ffi import ErrNo
from std.io import IOError, IOErrorKind
from std.io._syscall import retry_io, retry_io_interrupted
from std.testing import TestSuite, assert_equal


def test_retries_listed_kinds() raises:
    var calls = 0

    def attempt() raises IOError {mut calls} -> Int:
        calls += 1
        if calls == 1:
            raise IOError(errno=ErrNo.EINTR, operation="read")
        if calls == 2:
            raise IOError(errno=ErrNo.EAGAIN, operation="read")
        return 7

    assert_equal(retry_io[retry_kinds=[.interrupted, .would_block]](attempt), 7)
    assert_equal(calls, 3)


def test_raises_unlisted_kinds() raises:
    var calls = 0

    def attempt() raises IOError {mut calls} -> Int:
        calls += 1
        raise IOError(errno=ErrNo.EINTR, operation="read")

    var kind = IOErrorKind.uncategorized
    try:
        _ = retry_io[retry_kinds=[.would_block]](attempt)
    except e:
        kind = e.kind()
    assert_equal(kind, IOErrorKind.interrupted)
    assert_equal(calls, 1)


def test_retry_io_interrupted() raises:
    var calls = 0

    def attempt() raises IOError {mut calls} -> Int:
        calls += 1
        if calls < 3:
            raise IOError(errno=ErrNo.EINTR, operation="read")
        return 7

    assert_equal(retry_io_interrupted(attempt), 7)
    assert_equal(calls, 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
