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

from std.sys import Endian
from std.testing import TestSuite, assert_equal, assert_not_equal


def test_equality() raises:
    assert_equal(Endian.big, Endian.big)
    assert_equal(Endian.little, Endian.little)
    assert_not_equal(Endian.big, Endian.little)


def test_write_to() raises:
    assert_equal(String(Endian.big), "Endian.big")
    assert_equal(String(Endian.little), "Endian.little")

    __match Endian.native():
    case .big:
        assert_equal(String(Endian.native()), "Endian.big")
    case .little:
        assert_equal(String(Endian.native()), "Endian.little")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
