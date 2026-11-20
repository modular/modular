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

from std.testing import assert_equal, TestSuite


def test_pow() raises:
    assert_equal(Int(SIMDLength(2) ** SIMDLength(10)), 1024)
    assert_equal(Int(SIMDLength(3) ** SIMDLength(0)), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
