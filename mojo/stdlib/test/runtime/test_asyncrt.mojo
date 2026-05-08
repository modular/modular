# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.runtime.asyncrt import (
    create_raising_task,
    create_task,
    _create_task,
    task_id_for_device,
)

from std.testing import assert_equal, TestSuite


# CHECK-LABEL: test_runtime_task
def test_runtime_task() raises:
    print("== test_runtime_task")

    @parameter
    async def test_asyncrt_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    @parameter
    async def test_asyncrt_add_two_of_them(a: Int, b: Int) -> Int:
        return await create_task(test_asyncrt_add[1](a)) + await create_task(
            test_asyncrt_add[2](b)
        )

    var task = create_task(test_asyncrt_add_two_of_them(10, 20))
    # CHECK: 33
    print(task.wait())


# CHECK-LABEL: test_runtime_taskgroup
def test_runtime_taskgroup() raises:
    print("== test_runtime_taskgroup")

    @parameter
    async def return_value[value: Int]() -> Int:
        return value

    @parameter
    async def run_as_group() -> Int:
        var t0 = create_task(return_value[1]())
        var t1 = create_task(return_value[2]())
        return await t0 + await t1

    var t0 = create_task(run_as_group())
    var t1 = create_task(run_as_group())
    # CHECK: 6
    print(t0.wait() + t1.wait())


# CHECK-LABEL: test_runtime_unified_async_memory_result_raises
def test_runtime_unified_async_memory_result_raises() raises:
    print("== test_runtime_unified_async_memory_result_raises")
    var prefix = String("hello")

    async def build_message() raises {mut prefix} -> String:
        return prefix + String(" world")

    var task = create_raising_task(build_message())
    var result = task^.wait()
    # CHECK: hello world
    print(result)
    assert_equal(result, "hello world")


# CHECK-LABEL: test_task_id_for_device_returns_int
def test_task_id_for_device_returns_int() raises:
    """Verify task_id_for_device returns a non-negative or -1 integer.

    The function delegates to KGEN_CompilerRT_TaskIdForDevice.  Without a
    configured affinity map, the runtime may return -1 (no hint) or a valid
    worker index.  Either way, the return type must be Int and be >= -1.
    """
    print("== test_task_id_for_device_returns_int")
    var result = task_id_for_device(0)
    # CHECK: task_id_for_device(0) >= -1: True
    print("task_id_for_device(0) >= -1:", result >= -1)


# CHECK-LABEL: test_create_task_with_affinity_runs_coroutine
def test_create_task_with_affinity_runs_coroutine() raises:
    """Verify _create_task executes the coroutine to completion.

    The desired_worker_id hint is advisory; correctness must hold regardless
    of whether the runtime honours the hint.
    """
    print("== test_create_task_with_affinity_runs_coroutine")

    @parameter
    async def compute() -> Int:
        return 42

    var worker_id = task_id_for_device(0)
    var task = _create_task(compute(), desired_worker_id=worker_id)
    # CHECK: affinity task result: 42
    print("affinity task result:", task.wait())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
