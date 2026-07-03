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

from std.benchmark import Bencher
from std.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal, assert_true


def test_iter_custom_device_context_closure() raises:
    """Tests `iter_custom(func, ctx)` for `def(DeviceContext) raises` closures.
    """
    var bencher = Bencher(3)
    var ctx = DeviceContext(api="cpu")
    var count = 0

    @always_inline
    def launch(ctx: DeviceContext) raises {mut count}:
        _ = ctx
        count += 1

    bencher.iter_custom(launch, ctx)

    assert_equal(count, 3)
    assert_true(bencher.elapsed >= 0)


def test_iter_custom_device_context_iteration_closure() raises:
    """Tests `iter_custom(func, ctx)` for `def(DeviceContext, Int) raises` closures.
    """
    var bencher = Bencher(4)
    var ctx = DeviceContext(api="cpu")
    var iteration_sum = 0

    @always_inline
    def launch(ctx: DeviceContext, iteration: Int) raises {mut iteration_sum}:
        _ = ctx
        iteration_sum += iteration

    bencher.iter_custom(launch, ctx)

    assert_equal(iteration_sum, 6)
    assert_true(bencher.elapsed >= 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
