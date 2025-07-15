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


# CHECK-LABEL: test_benchmark
fn test_benchmark():
    print("== test_benchmark")

    @always_inline
    @parameter
    fn time_me():
        sleep(0.002)
        clobber_memory()
        return

    # check that benchmark_function returns after max_time_ns is hit.
    var lb = 0.02  # 20ms
    var ub = 0.1  # 100ms
    var max_iters = 1000_000_000

    @__copy_capture(max_iters, lb, ub)
    @parameter
    fn timer():
        var b3 = run[time_me](
            0, max_iters, min_runtime_secs=lb, max_runtime_secs=ub
        )
        # CHECK: True
        print(b3.mean() > 0)

    var t3 = time_function[timer]()
    # CHECK: True
    print(t3 / 1e9 >= lb)

    var ub_big = 1  # 1s

    @__copy_capture(ub_big, lb)
    @parameter
    fn timer2():
        var b4 = run[time_me](
            0, 1, min_runtime_secs=lb, max_runtime_secs=ub_big
        )
        # CHECK: True
        print(b4.mean() > 0)

    var t4 = time_function[timer2]()
    # CHECK: True
    print(t4 / 1e9 >= lb and t4 / 1e9 <= ub_big)


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
fn test_keep():
    print("== test_keep")

    keep(False)
    keep(33)

    var val = SIMD[DType.index, 4](1, 2, 3, 4)
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
fn test_non_capturing():
    print("== test_non_capturing")
    var report = run[sleeper](min_runtime_secs=0.1, max_runtime_secs=0.3)
    # CHECK: True
    print(report.mean() > 0.001)


# CHECK-LABEL: test_change_units
fn test_change_units():
    print("== test_change_units")
    var report = run[sleeper](min_runtime_secs=0.1, max_runtime_secs=0.3)
    # CHECK: True
    print(report.mean("ms") > 1.0)
    # CHECK: True
    print(report.mean("ns") > 1_000_000.0)


# CHECK-LABEL: test_report
fn test_report():
    print("== test_report")
    var report = run[sleeper](min_runtime_secs=0.1, max_runtime_secs=0.3)

    # CHECK: Benchmark Report (s)
    report.print()


def main():
    test_benchmark()
    test_keep()
    test_non_capturing()
    test_change_units()
    test_report()
