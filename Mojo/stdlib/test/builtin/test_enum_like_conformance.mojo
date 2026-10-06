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
"""Exercises runtime `__match` on the stdlib's enum-like structs.

Each helper names every case so the match fails to compile if
`_enum_case_names` drifts from the comptime case constants.
"""

from std.testing import TestSuite, assert_equal

from std._gpu.intrinsics import Scope
from std.simd import FastMathFlag
from std.testing.suite import TestResult


def _scope_rank(scope: Scope) -> Int:
    __match scope:
        case .NONE:
            return 0
        case .THREAD:
            return 1
        case .WARP:
            return 2
        case .BLOCK:
            return 3
        case .CLUSTER:
            return 4
        case .GPU:
            return 5
        case .SYSTEM:
            return 6


def _fast_math_rank(flag: FastMathFlag) -> Int:
    __match flag:
        case .NONE:
            return 0
        case .NNAN:
            return 1
        case .NINF:
            return 2
        case .NSZ:
            return 3
        case .ARCP:
            return 4
        case .CONTRACT:
            return 5
        case .AFN:
            return 6
        case .REASSOC:
            return 7
        case .FAST:
            return 8


def _test_result_rank(result: TestResult) -> Int:
    __match result:
        case .PASS:
            return 0
        case .FAIL:
            return 1
        case .SKIP:
            return 2


def test_scope_match() raises:
    var scopes = List[Scope]()
    scopes.append(Scope.NONE)
    scopes.append(Scope.THREAD)
    scopes.append(Scope.WARP)
    scopes.append(Scope.BLOCK)
    scopes.append(Scope.CLUSTER)
    scopes.append(Scope.GPU)
    scopes.append(Scope.SYSTEM)

    for i in range(len(scopes)):
        assert_equal(_scope_rank(scopes[i]), i, "wrong Scope case matched")


def test_fast_math_match() raises:
    var flags = List[FastMathFlag]()
    flags.append(FastMathFlag.NONE)
    flags.append(FastMathFlag.NNAN)
    flags.append(FastMathFlag.NINF)
    flags.append(FastMathFlag.NSZ)
    flags.append(FastMathFlag.ARCP)
    flags.append(FastMathFlag.CONTRACT)
    flags.append(FastMathFlag.AFN)
    flags.append(FastMathFlag.REASSOC)
    flags.append(FastMathFlag.FAST)

    for i in range(len(flags)):
        assert_equal(
            _fast_math_rank(flags[i]), i, "wrong FastMathFlag case matched"
        )


def test_result_match() raises:
    var results = List[TestResult]()
    results.append(TestResult.PASS)
    results.append(TestResult.FAIL)
    results.append(TestResult.SKIP)

    for i in range(len(results)):
        assert_equal(
            _test_result_rank(results[i]), i, "wrong TestResult case matched"
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
