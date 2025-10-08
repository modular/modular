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

from time import sleep, time_function

from benchmark import Report, clobber_memory, keep, run
from testing import TestSuite


# CHECK-LABEL: test_stopping_criteria
def test_stopping_criteria():
    print("== test_stopping_criteria")
    # Stop when min_runtime_secs has elapsed and either max_runtime_secs or max_iters
    # is reached

    @always_inline
    @parameter
    fn time_me():
        sleep(0.002)
        clobber_memory()
        return

    var lb = 0.02  # 20ms
    var ub = 0.1  # 100ms

    # stop after ub (max_runtime_secs)
    var max_iters_1 = 1000_000_000

    @__copy_capture(lb, ub)
    @parameter
    fn timer() raises:
        var report = run[time_me](
            max_iters=max_iters_1, min_runtime_secs=lb, max_runtime_secs=ub
        )
        # CHECK: True
        print(report.mean() > 0)
        # CHECK: True
        print(report.iters() != max_iters_1)

    var t1 = time_function[timer]()
    # CHECK: True
    print(t1 / 1e9 >= ub)

    # stop after lb (min_runtime_secs)
    var ub_big = 1  # 1s
    var max_iters_2 = 1

    @__copy_capture(ub_big, lb)
    @parameter
    fn timer2() raises:
        var report = run[time_me](
            max_iters=max_iters_2, min_runtime_secs=lb, max_runtime_secs=ub_big
        )
        # CHECK: True
        print(report.mean() > 0)
        # CHECK: True
        print(report.iters() >= max_iters_2)

    var t2 = time_function[timer2]()

    # CHECK: True
    print(t2 / 1e9 >= lb and t2 / 1e9 <= ub_big)

    # stop on or before max_iters
    var max_iters_3 = 3

    @__copy_capture(ub_big)
    @parameter
    fn timer3() raises:
        var report = run[time_me](
            max_iters=max_iters_3, min_runtime_secs=0, max_runtime_secs=ub_big
        )
        # CHECK: True
        print(report.mean() > 0)
        # CHECK: True
        print(report.iters() <= max_iters_3)

    var t3 = time_function[timer3]()

    # CHECK: True
    print(t3 / 1e9 <= ub_big)


@register_passable("trivial")
struct SomeStruct:
    var x: Int
    var y: Int

    @always_inline
    fn __init__(out self):
        self.x = 5
        self.y = 4


@register_passable("trivial")
struct SomeTrivialStruct:
    var x: Int
    var y: Int

    @always_inline
    fn __init__(out self):
        self.x = 3
        self.y = 5


# CHECK-LABEL: test_keep
# There is nothing to test here other than the code executes and does not crash.
def test_keep():
    print("== test_keep")

    keep(False)
    keep(33)

    var val = SIMD[DType.int, 4](1, 2, 3, 4)
    keep(val)

    var ptr = UnsafePointer(to=val)
    keep(ptr)

    var s0 = SomeStruct()
    keep(s0)

    var s1 = SomeTrivialStruct()
    keep(s1)


fn sleeper():
    sleep(0.001)


# CHECK-LABEL: test_non_capturing
def test_non_capturing():
    print("== test_non_capturing")
    var report = run[sleeper](min_runtime_secs=0.1, max_runtime_secs=0.3)
    # CHECK: True
    print(report.mean() > 0.001)


# CHECK-LABEL: test_change_units
def test_change_units():
    print("== test_change_units")
    var report = run[sleeper](min_runtime_secs=0.1, max_runtime_secs=0.3)
    # CHECK: True
    print(report.mean("ms") > 1.0)
    # CHECK: True
    print(report.mean("ns") > 1_000_000.0)


# CHECK-LABEL: test_report
def test_report():
    print("== test_report")
    var report = run[sleeper](min_runtime_secs=0.1, max_runtime_secs=0.3)

    # CHECK: Benchmark Report (s)
    report.print()


def main():
    var suite = TestSuite()

    suite.test[test_stopping_criteria]()
    suite.test[test_keep]()
    suite.test[test_non_capturing]()
    suite.test[test_change_units]()
    suite.test[test_report]()

    suite^.run()
