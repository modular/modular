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

from os.atomic import Atomic

from runtime.asyncrt import create_task, run
from testing import assert_true


# CHECK-LABEL: test_sync_coro
fn test_sync_coro():
    print("== test_sync_coro")

    @parameter
    async fn test_asyncrt_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    @parameter
    async fn test_asyncrt_add_two_of_them(a: Int, b: Int) -> Int:
        return await test_asyncrt_add[5](a) + await test_asyncrt_add[2](b)

    # CHECK: 57
    print(run(test_asyncrt_add_two_of_them(20, 30)))


fn test_sync_raising_coro():
    # CHECK: == test_sync_raising_coro
    print("== test_sync_raising_coro")

    # FIXME(#26008): Raising async functions do not work.
    # @parameter
    # async fn might_throw(a: Int) raises -> Int:
    #    if a > 10:
    #        raise Error("oops")
    #    return a + 1

    # @parameter
    # async fn also_might_throw(a: Int) raises -> Int:
    #    if a == 20:
    #        raise Error("doh!")
    #    return await might_throw(a) + 100

    # try:
    #    print(also_might_throw(20)())
    # except e:
    #    # XCHECK-NEXT: doh!
    #    print(e)
    # try:
    #    print(also_might_throw(25)())
    # except e:
    #    # XCHECK-NEXT: oops
    #    print(e)
    # try:
    #    # XCHECK-NEXT: 102
    #    print(also_might_throw(1)())
    # except:
    #    pass


# CHECK-LABEL: test_runtime_task
fn test_runtime_task():
    print("== test_runtime_task")

    @parameter
    async fn test_asyncrt_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs

    @parameter
    async fn test_asyncrt_add_two_of_them(a: Int, b: Int) -> Int:
        return await create_task(test_asyncrt_add[1](a)) + await create_task(
            test_asyncrt_add[2](b)
        )

    var task = create_task(test_asyncrt_add_two_of_them(10, 20))
    # CHECK: 33
    print(task.wait())


# CHECK-LABEL: test_runtime_taskgroup
fn test_runtime_taskgroup():
    print("== test_runtime_taskgroup")

    @parameter
    async fn return_value[value: Int]() -> Int:
        return value

    @parameter
    async fn run_as_group() -> Int:
        var t0 = create_task(return_value[1]())
        var t1 = create_task(return_value[2]())
        return await t0 + await t1

    var t0 = create_task(run_as_group())
    var t1 = create_task(run_as_group())
    # CHECK: 6
    print(t0.wait() + t1.wait())


def main():
    test_sync_coro()
    test_sync_raising_coro()
    test_runtime_task()
    test_runtime_taskgroup()
