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
import time


@fieldwise_init
struct ConditionalTimer(ImplicitlyCopyable, Movable):
    var start_time: Int

    fn __init__(out self):
        self.start_time = 0

    fn __enter__(mut self) -> Self:
        self.start_time = Int(time.perf_counter_ns())
        return self

    fn __exit__(mut self):
        end_time = time.perf_counter_ns()
        elapsed_time_ms = round(((end_time - UInt(self.start_time)) / 1e6), 3)
        print("Elapsed time:", elapsed_time_ms, "milliseconds")

    fn __exit__(mut self, e: Error) raises -> Bool:
        if String(e) == "just a warning":
            print("Suppressing error:", e)
            self.__exit__()
            return True
        else:
            print("Propagating error")
            self.__exit__()
            return False


def flaky_identity(n: Int) -> Int:
    if (n % 4) == 0:
        raise "really bad"
    elif (n % 2) == 0:
        raise "just a warning"
    else:
        return n


def main():
    for i in range(1, 9):
        with ConditionalTimer():
            print("\nBeginning execution")

            print("i =", i)
            time.sleep(0.1)

            if i == 3:
                print("continue executed")
                continue

            j = flaky_identity(i)
            print("j =", j)

            print("Ending execution")
