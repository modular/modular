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

from os import Atomic
from time import sleep, time_function

from runtime.asyncrt import TaskGroup
from testing import assert_equal

from utils.lock import BlockingScopedLock, BlockingSpinLock


fn test_basic_lock() raises:
    var lock = BlockingSpinLock()
    var rawCounter = 0
    var counter = Atomic[DType.int64](0)
    alias maxI = 100
    alias maxJ = 100

    @parameter
    async fn inc():
        with BlockingScopedLock(lock):
            rawCounter += 1
            _ = counter.fetch_add(1)

    # CHECK: PRE::Atomic counter is 0 , and raw counter, 0
    print(
        "PRE::Atomic counter is ",
        counter.load(),
        ", and raw counter, ",
        rawCounter,
    )

    @parameter
    fn test_atomic() -> None:
        var tg = TaskGroup()
        for _ in range(0, maxI):
            for _ in range(0, maxJ):
                tg.create_task(inc())
        tg.wait[__origin_of(lock)]()

    _ = time_function[test_atomic]()
    _ = lock^
    # print("Total time taken ", time_ns / (1_000_000_000), " s")

    # CHECK: POST::Atomic counter is 10000 , and raw counter, 10000
    print(
        "POST::Atomic counter is ",
        counter.load(),
        ", and raw counter, ",
        rawCounter,
    )
    assert_equal(counter.load(), rawCounter, "atomic stress test failed")

    return


def main():
    test_basic_lock()
