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

from max.gpu import ReduceOp


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
