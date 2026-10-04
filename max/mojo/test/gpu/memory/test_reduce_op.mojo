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

from std.testing import TestSuite, assert_equal

from max.gpu import CacheEviction, Consistency, Fill, ReduceOp


def _matched_case(op: ReduceOp) -> Int:
    """Returns the 1-based case index selected by `__match` for `op`."""
    __match op:
        case .ADD:
            return 1
        case .MIN:
            return 2
        case .MAX:
            return 3
        case .AND:
            return 4
        case .OR:
            return 5
        case .XOR:
            return 6


def test_match_selects_case() raises:
    # Materialize each case into a runtime value so the `__match` in
    # `_matched_case` is exercised on a runtime `ReduceOp`.
    var cases = List[ReduceOp]()
    cases.append(ReduceOp.ADD)
    cases.append(ReduceOp.MIN)
    cases.append(ReduceOp.MAX)
    cases.append(ReduceOp.AND)
    cases.append(ReduceOp.OR)
    cases.append(ReduceOp.XOR)

    for i in range(len(cases)):
        assert_equal(_matched_case(cases[i]), Int(i) + 1, "wrong case matched")


def test_enum_discriminants() raises:
    # `ADD` is the first case (discriminant 0) and `XOR` the last (5); the
    # backing `_value` is what `_get_enum_discriminant` reads, and it is
    # observable through equality with the case's comptime value.
    assert_equal(ReduceOp.ADD._value, 0)
    assert_equal(ReduceOp.MIN._value, 1)
    assert_equal(ReduceOp.MAX._value, 2)
    assert_equal(ReduceOp.AND._value, 3)
    assert_equal(ReduceOp.OR._value, 4)
    assert_equal(ReduceOp.XOR._value, 5)


def test_formatting_keeps_custom_mnemonics() raises:
    # The EnumLike conformance must not change the custom `write_to`, which
    # prints the reduction mnemonic rather than the case name.
    assert_equal(String(ReduceOp.ADD), "add")
    assert_equal(String(ReduceOp.MIN), "min")
    assert_equal(String(ReduceOp.MAX), "max")
    assert_equal(String(ReduceOp.AND), "and")
    assert_equal(String(ReduceOp.OR), "or")
    assert_equal(String(ReduceOp.XOR), "xor")


def _cache_eviction_rank(eviction: CacheEviction) -> Int:
    """Returns the 0-based case index selected by `__match`."""
    __match eviction:
        case .EVICT_NORMAL:
            return 0
        case .EVICT_FIRST:
            return 1
        case .EVICT_LAST:
            return 2
        case .EVICT_UNCHANGED:
            return 3
        case .NO_ALLOCATE:
            return 4


def _fill_rank(fill: Fill) -> Int:
    """Returns the 0-based case index selected by `__match`."""
    __match fill:
        case .NONE:
            return 0
        case .ZERO:
            return 1
        case .NAN:
            return 2


def _consistency_rank(consistency: Consistency) -> Int:
    """Returns the 0-based case index selected by `__match`."""
    __match consistency:
        case .WEAK:
            return 0
        case .RELAXED:
            return 1
        case .ACQUIRE:
            return 2
        case .RELEASE:
            return 3


def test_cache_eviction_match() raises:
    var evictions = List[CacheEviction]()
    evictions.append(CacheEviction.EVICT_NORMAL)
    evictions.append(CacheEviction.EVICT_FIRST)
    evictions.append(CacheEviction.EVICT_LAST)
    evictions.append(CacheEviction.EVICT_UNCHANGED)
    evictions.append(CacheEviction.NO_ALLOCATE)
    for i in range(len(evictions)):
        assert_equal(
            _cache_eviction_rank(evictions[i]), i, "wrong CacheEviction case"
        )


def test_fill_match() raises:
    var fills = List[Fill]()
    fills.append(Fill.NONE)
    fills.append(Fill.ZERO)
    fills.append(Fill.NAN)
    for i in range(len(fills)):
        assert_equal(_fill_rank(fills[i]), i, "wrong Fill case")


def test_consistency_match() raises:
    var consistencies = List[Consistency]()
    consistencies.append(Consistency.WEAK)
    consistencies.append(Consistency.RELAXED)
    consistencies.append(Consistency.ACQUIRE)
    consistencies.append(Consistency.RELEASE)
    for i in range(len(consistencies)):
        assert_equal(
            _consistency_rank(consistencies[i]), i, "wrong Consistency case"
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
