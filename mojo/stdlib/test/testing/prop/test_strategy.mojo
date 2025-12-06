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
from testing.prop.strategy.simd_strategy import *
from testing.prop.strategy.string_strategy import *
from testing.prop.strategy.list_strategy import *
from sys.intrinsics import _type_is_eq
from collections.string._utf8 import _is_valid_utf8


@fieldwise_init
struct TestStruct(Copyable):
    var n: Int

    @staticmethod
    fn strategy() -> TestStructStrategy:
        return TestStructStrategy(0, 10)


@fieldwise_init
struct TestStructStrategy(Movable, Strategy):
    comptime Value = TestStruct

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


def test_simd_strategy():
    var min = Int64(-10)
    var max = Int64(30)
    var strat = SIMD[DType.int64, 8].strategy(min=min, max=max)
    var rng = Rng(seed=1234)
    for _ in range(10):
        var v = strat.value(rng)
        assert_true(all(v.ge(min)))
        assert_true(all(v.le(max)))


def test_string_ascii_strategy():
    var s = String.ascii_strategy(min_len=1, max_len=10, only_printable=True)
    var rng = Rng(seed=1234)
    for _ in range(10):
        var s = s.value(rng)
        assert_true(1 <= len(s) <= 10)
        assert_true(s.as_string_slice().is_ascii_printable())

    s = String.ascii_strategy(min_len=1, max_len=10, only_printable=False)
    rng = Rng(seed=1234)
    for _ in range(10):
        var s = s.value(rng)
        assert_true(1 <= len(s) <= 10)
        for c in s.codepoints():
            assert_true(c.to_u32() <= UInt32(127))


def test_string_utf8_strategy():
    var s = String.utf8_strategy(min_len=1, max_len=10)
    var rng = Rng(seed=1234)
    for _ in range(10):
        var s = s.value(rng)
        assert_true(1 <= len(s.codepoints()) <= 10, s)
        assert_true(_is_valid_utf8(s.as_bytes()), s)


def test_list_strategy():
    var s = List[Int8].strategy(
        Int8.strategy(min=0, max=100), min_len=2, max_len=45
    )
    var rng = Rng(seed=1234)

    for _ in range(10):
        var l = s.value(rng)

        assert_true(2 <= len(l) <= 45)

        for el in l:
            assert_true(Int8(0) <= el <= Int8(100))


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
