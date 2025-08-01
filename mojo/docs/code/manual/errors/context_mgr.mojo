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
import sys
import time


@fieldwise_init
struct Timer(Copyable, Movable):
    var start_time: Int

    fn __init__(out self):
        self.start_time = 0

    fn __enter__(mut self) -> Self:
        self.start_time = Int(time.perf_counter_ns())
        return self

    fn __exit__(mut self):
        end_time = time.perf_counter_ns()
        elapsed_time_ms = round(((end_time - self.start_time) / 1e6), 3)
        print("Elapsed time:", elapsed_time_ms, "milliseconds")


def main():
    with Timer():
        print("Beginning execution")
        time.sleep(1.0)
        if len(sys.argv()) > 1:
            raise "simulated error"
        time.sleep(1.0)
        print("Ending execution")
