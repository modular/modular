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

from std.ffi import c_int
from std.io import RawFileDescriptor
from std.sys._io import stdin, stdout, stderr

from std.testing import TestSuite, assert_equal, assert_false, assert_true


def test_value_is_c_int() raises:
    assert_true(type_of(stdout.value) == c_int)
    assert_equal(stdout.value, 1)
    assert_equal(stderr.value, 2)
    assert_equal(RawFileDescriptor().value, 1)


def test_isatty_with_standard_descriptors() raises:
    assert_true(type_of(stdin.isatty()) == Bool)
    assert_true(type_of(stdout.isatty()) == Bool)
    assert_true(type_of(stderr.isatty()) == Bool)


def test_isatty_with_invalid_fd() raises:
    # Test with an invalid file descriptor
    # isatty should return False for invalid file descriptors
    assert_false(RawFileDescriptor(-1).isatty())

    # Test with a very large invalid file descriptor
    assert_false(RawFileDescriptor(9999).isatty())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
