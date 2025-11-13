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

from testing import (
    assert_equal,
    assert_true,
    TestSuite,
)
from test_utils import DelCounter
from testing.prop import Rng, Strategy
from sys.intrinsics import _type_is_eq


@fieldwise_init
struct TestStruct(Copyable, Movable):
    var n: Int

    @staticmethod
    fn strategy() -> TestStructStrategy:
        return TestStructStrategy(0, 10)


@fieldwise_init
struct TestStructStrategy(Movable, Strategy):
    alias Value = TestStruct

    var min: Int
    var max: Int

    fn value(mut self, mut rng: Rng) raises -> Self.Value:
        return {rng.rand_int(min=self.min, max=self.max)}


def test_strategy_returns_correct_value():
    var strategy = TestStruct.strategy()
    var rng = Rng(seed=1234)
    for _ in range(10):
        var n = strategy.value(rng).n
        assert_true(n >= 0 and n <= 10)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
