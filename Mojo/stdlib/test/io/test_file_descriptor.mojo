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

from std.collections import Optional
from std.ffi import c_int
from std.io import RawFileDescriptor
from std.sys import size_of
from std.sys._io import stdin, stdout, stderr

from std.testing import TestSuite, assert_equal, assert_false, assert_true


def test_value_is_c_int() raises:
    assert_true(type_of(stdout._value) == c_int)
    assert_equal(stdout._value, 1)
    assert_equal(stderr._value, 2)
    assert_equal(RawFileDescriptor()._value, 1)


def test_isatty_with_standard_descriptors() raises:
    assert_true(type_of(stdin.isatty()) == Bool)
    assert_true(type_of(stdout.isatty()) == Bool)
    assert_true(type_of(stderr.isatty()) == Bool)


def test_optional_uses_fd_niche() raises:
    comptime assert (
        size_of[Optional[RawFileDescriptor]]() == size_of[RawFileDescriptor]()
    )
    comptime assert size_of[RawFileDescriptor]() == size_of[c_int]()

    var fd: Optional[RawFileDescriptor] = RawFileDescriptor(7)
    assert_true(fd)
    assert_equal(fd[]._value, 7)

    fd = None
    assert_false(fd)

    fd = RawFileDescriptor(3)
    assert_true(fd)
    assert_equal(fd[]._value, 3)


def test_isatty_with_invalid_fd() raises:
    # A descriptor that is not open is not a tty.
    assert_false(RawFileDescriptor(9999).isatty())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
