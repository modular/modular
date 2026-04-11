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

from std.testing import (
    assert_equal,
    assert_true,
    assert_raises,
    TestSuite,
)
from std.testing.prop import PropTest, PropTestConfig, Rng, Strategy
from std.testing.prop._shrinking import Shrinker
from std.testing.prop.strategy.simd_strategy import *


@fieldwise_init
struct SimpleStrategy(Movable, Strategy):
    comptime Value = Int

    def value(mut self, mut rng: Rng) raises -> Self.Value:
        return 42


def test_prop_test_runner_propagates_error() raises:
    @parameter
    def properties(_n: Int) raises:
        raise Error("prop test error 123")

    with assert_raises(contains="prop test error 123"):
        PropTest().test[properties](SimpleStrategy())


@fieldwise_init
struct RecordingStrategy[origin: MutOrigin](Movable, Strategy):
    comptime Value = Int

    var list: UnsafePointer[List[Int], origin=Self.origin]

    def value(self, mut rng: Rng) raises -> Self.Value:
        var random = rng.rand_int()
        self.list[].append(random)
        return random


def test_prop_test_runner_executes_specified_number_of_runs() raises:
    @parameter
    def properties(_n: Int) raises:
        pass

    var list = List[Int]()
    var strategy = RecordingStrategy(UnsafePointer(to=list))

    PropTest(config=PropTestConfig(runs=10)).test[properties](strategy^)
    assert_equal(10, len(list))


def test_prop_test_runner_using_same_seed_produces_deterministic_results() raises:
    @parameter
    def properties(_n: Int) raises:
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


def test_simple_reduce() raises:
    @parameter
    def do_test(var value: UInt64) raises:
        assert_true(
            UInt64(0) < value <= UInt64(10), "value should be between 1 and 10"
        )

    var err = Error("")

    try:
        do_test(300)
    except e:
        err = e^

    var strat = UInt64.strategy()
    var shrinker = Shrinker[type_of(strat), do_test](strat^, [300], err^)

    assert_equal(shrinker.shrink().value, 0)


def test_list_reduce() raises:
    # Property: all lists of positive integers sum to less than 100
    @parameter
    def prop(values: List[Int64]) raises:
        var total: Int64 = 0
        for v in values:
            total += v
        if total >= 100:
            raise Error("invalid total")

    var config = PropTestConfig(seed=1235)

    with assert_raises(contains="[100]"):
        PropTest(config=config^).test[prop](
            List[Int64].strategy(Int64.strategy(min=1, max=50))
        )


@fieldwise_init
struct CompoundCase(Copyable, Writable):
    var a: Int
    var b: Float64
    var c: List[UInt64]


struct CompoundCaseStrategy(Strategy):
    comptime Value = CompoundCase
    """The type the strategy produces."""

    var l_strat: type_of(List[UInt64].strategy(UInt64.strategy()))

    def __init__(out self) raises:
        self.l_strat = List[UInt64].strategy(UInt64.strategy())

    def value(mut self, mut rng: Rng) raises -> Self.Value:
        return Self.Value(
            rng.rand_int(),
            rng.rand_scalar[DType.float64](),
            self.l_strat.value(rng),
        )


def test_complex_reduct() raises:
    @parameter
    def test(p: CompoundCase) raises:
        assert_true(p.a < 10 or p.b < 1.0 or len(p.c) < 5)

    var config = PropTestConfig(seed=1235)

    with assert_raises(contains="CompoundCase(a=10, b=1.0, c=[0, 0, 0, 0, 0])"):
        PropTest(config=config^).test[test](CompoundCaseStrategy())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
