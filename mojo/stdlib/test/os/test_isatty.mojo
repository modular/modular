# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from os import isatty
from sys._io import stdin, stdout, stderr
from testing import TestSuite, assert_equal, assert_false


def test_isatty_matches_file_descriptor():
    """Test that os.isatty(fd) matches FileDescriptor.isatty() for standard streams.
    """
    assert_equal(isatty(0), stdin.isatty())
    assert_equal(isatty(1), stdout.isatty())
    assert_equal(isatty(2), stderr.isatty())


def test_isatty_with_invalid_fd():
    """Test that isatty returns False for invalid file descriptors."""
    # Test with negative file descriptor
    assert_false(isatty(-1))

    # Test with a very large invalid file descriptor
    assert_false(isatty(9999))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
