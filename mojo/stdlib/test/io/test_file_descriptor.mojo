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

from io import FileDescriptor
from sys._io import stdin, stdout, stderr

from sys.intrinsics import _type_is_eq
from testing import TestSuite, assert_false, assert_true


def test_isatty_with_standard_descriptors():
    assert_true(_type_is_eq[type_of(stdin.isatty()), Bool]())
    assert_true(_type_is_eq[type_of(stdout.isatty()), Bool]())
    assert_true(_type_is_eq[type_of(stderr.isatty()), Bool]())


def test_isatty_with_invalid_fd():
    # Test with an invalid file descriptor
    # isatty should return False for invalid file descriptors
    assert_false(FileDescriptor(-1).isatty())

    # Test with a very large invalid file descriptor
    assert_false(FileDescriptor(9999).isatty())


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
