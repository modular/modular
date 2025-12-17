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
    assert_raises,
    TestSuite,
)
from testing.prop import PropTest, PropTestConfig, Rng, Strategy


@fieldwise_init
struct SimpleStrategy(Movable, Strategy):
    comptime Value = Int

    fn value(mut self, mut rng: Rng) raises -> Self.Value:
        return 42


def test_prop_test_runner_propagates_error():
    @parameter
    def properties(_n: Int):
        raise Error("prop test error 123")

    with assert_raises(contains="prop test error 123"):
        PropTest().test[properties](SimpleStrategy())


@fieldwise_init
struct RecordingStrategy[origin: MutOrigin](Movable, Strategy):
    comptime Value = Int

    var list: UnsafePointer[List[Int], origin = Self.origin]

    fn value(self, mut rng: Rng) raises -> Self.Value:
        var random = rng.rand_int()
        self.list[].append(random)
        return random


def test_prop_test_runner_executes_specified_number_of_runs():
    @parameter
    def properties(_n: Int):
        pass

    var list = List[Int]()
    var strategy = RecordingStrategy(UnsafePointer(to=list))

    PropTest(config=PropTestConfig(runs=10)).test[properties](strategy^)
    assert_equal(10, len(list))


def test_prop_test_runner_using_same_seed_produces_deterministic_results():
    @parameter
    def properties(_n: Int):
        pass

    var config = PropTestConfig(runs=5, seed=1234)

    var initial_list = List[Int]()
    PropTest(config=config.copy()).test[properties](
        RecordingStrategy(UnsafePointer(to=initial_list))
    )

    var second_list = List[Int]()
    PropTest(config=config^).test[properties](
        RecordingStrategy(UnsafePointer(to=second_list))
    )

    assert_equal(initial_list, second_list)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
