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

from runtime.asyncrt import create_task

from testing import TestSuite


# CHECK-LABEL: test_runtime_task
def test_runtime_task():
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
def test_runtime_taskgroup():
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
    var suite = TestSuite()

    suite.test[test_runtime_task]()
    suite.test[test_runtime_taskgroup]()

    suite^.run()
