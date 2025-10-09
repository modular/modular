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

from testing import assert_true, TestSuite

from utils.lock import SpinWaiter


def test_spin_waiter():
    var waiter = SpinWaiter()
    alias RUNS = 1000
    for _ in range(RUNS):
        waiter.wait()
    assert_true(True)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
