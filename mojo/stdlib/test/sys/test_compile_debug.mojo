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

from sys.compile import DebugLevel, OptimizationLevel

from testing import assert_equal
from testing import TestSuite


def test_compile_debug_options():
    assert_equal(Int(OptimizationLevel), 0)
    assert_equal(String(DebugLevel), "none")


def main():
    var suite = TestSuite()

    suite.test[test_compile_debug_options]()

    suite^.run()
