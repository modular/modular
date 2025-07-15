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
from context_mgr import Timer
from testing import assert_raises
import time


def test_timer_no_error():
    with Timer():
        print("Beginning no-error execution")
        time.sleep(0.1)
        print("Ending no-error execution")


def test_timer_error():
    with assert_raises(contains="simulated error"):
        with Timer():
            print("Beginning error execution")
            time.sleep(0.1)
            raise "simulated error"
            # We should not reach this line


def main():
    test_timer_no_error()
    test_timer_error()
