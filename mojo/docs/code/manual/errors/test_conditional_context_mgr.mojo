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
from conditional_context_mgr import ConditionalTimer, flaky_identity
from testing import assert_raises
import time


def test_conditional_timer_no_error():
    with ConditionalTimer():
        print("Beginning no-error execution")
        time.sleep(0.1)
        i = 1
        _ = flaky_identity(i)
        print("Ending no-error execution")


def test_conditional_timer_suppressed_error():
    i = 2
    with assert_raises(contains="just a warning"):
        _ = flaky_identity(i)

    with ConditionalTimer():
        print("Beginning no-error execution")
        time.sleep(0.1)
        _ = flaky_identity(i)
        print("Ending no-error execution")


def test_conditional_timer_propagated_error():
    with assert_raises(contains="really bad"):
        with ConditionalTimer():
            print("Beginning propagated error execution")
            time.sleep(0.1)
            i = 4
            _ = flaky_identity(i)
            # We should not reach this line


def main():
    test_conditional_timer_no_error()
    test_conditional_timer_suppressed_error()
    test_conditional_timer_propagated_error()
