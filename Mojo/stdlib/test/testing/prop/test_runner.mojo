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

from std.testing import assert_equal, assert_raises, TestSuite
from std.testing.prop import PropTest, PropTestConfig, Rng, Strategy


@fieldwise_init
struct SimpleStrategy(Movable, Strategy):
    comptime Value = Int

    def value(mut self, mut rng: Rng) raises -> Self.Value:
        return 42


def test_prop_test_runner_propagates_error() raises:
    def properties(_n: Int) raises:
        raise Error("prop test error 123")

    with assert_raises(contains="prop test error 123"):
        PropTest().test(SimpleStrategy(), properties)


@fieldwise_init
struct RecordingStrategy[origin: MutOrigin](Movable, Strategy):
    comptime Value = Int

    var list: Pointer[List[Int], origin=Self.origin]

    def value(self, mut rng: Rng) raises -> Self.Value:
        var random = rng.rand_int()
        self.list[].append(random)
        return random


def test_prop_test_runner_executes_specified_number_of_runs() raises:
    def properties(_n: Int) raises:
        pass

    var list = List[Int]()
    var strategy = RecordingStrategy(Pointer(to=list))

    PropTest(config=PropTestConfig(runs=10)).test(strategy^, properties)
    assert_equal(10, len(list))


def test_prop_test_runner_using_same_seed_produces_deterministic_results() raises:
    def properties(_n: Int) raises:
        pass

    var config = PropTestConfig(runs=5, seed=1234)

    var initial_list = List[Int]()
    PropTest(config=config.copy()).test(
        RecordingStrategy(Pointer(to=initial_list)), properties
    )

    var second_list = List[Int]()
    PropTest(config=config^).test(
        RecordingStrategy(Pointer(to=second_list)), properties
    )

    assert_equal(initial_list, second_list)


def test_prop_test_runner_accepts_capturing_closure() raises:
    var total = 0

    def properties(n: Int) raises {mut total}:
        total += n

    PropTest(config=PropTestConfig(runs=3)).test(SimpleStrategy(), properties)
    assert_equal(total, 3 * 42)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
