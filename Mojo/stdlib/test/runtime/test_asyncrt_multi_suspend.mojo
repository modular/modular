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

# Reproducer for https://github.com/modular/modular/issues/7368: an awaited
# coroutine (hot ramp) that suspends twice never stores its second state in
# `_resume`, so it re-dispatches to its first suspension point forever. Until
# the bug is fixed this test hangs and is killed by the bazel test timeout.

from std.runtime._asyncrt import _run, create_task
from std.testing import assert_equal, TestSuite


__async def leaf(value: Int) -> Int:
    return value


__async def inner() -> Int:
    var a = __await create_task(leaf(1))
    var b = __await create_task(leaf(2))
    return a + b


__async def outer() -> Int:
    return __await inner()


def test_awaited_coroutine_suspends_twice() raises:
    assert_equal(_run(outer()), 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
