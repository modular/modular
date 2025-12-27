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

from testing import TestSuite, assert_true
from time import perf_counter_ns
from utils import ReadOnly


@fieldwise_init
struct TimeHolder(ImplicitlyCopyable):
    var time: UInt


def test_read_only():
    var current_time = TimeHolder(perf_counter_ns())
    ref start_time = ReadOnly(current_time)[]
    current_time.time = 0  # does not affect start_time
    # start_time.time = 0  # compile error
    assert_true(
        start_time.time > 0,
        msg="failed to use runtime constant value",
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
