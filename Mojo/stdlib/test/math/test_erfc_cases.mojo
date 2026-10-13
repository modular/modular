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

# Inputs where erfc once had more than 1.5 ulp of error, checked against
# correctly rounded results from mpmath.

from std.math import erfc, ulp
from std.testing import assert_true, TestSuite


def _within_one_ulp[
    dtype: DType
](
    got: Scalar[dtype], want: Scalar[dtype]
) -> Bool where dtype.is_floating_point():
    return abs(got - want) <= ulp(want)


def test_erfc_hard_cases() raises:
    var x: List[Float64] = [
        0.8291682292057292,
        0.8324116746883992,
        1.1728488886441404,
        2.834710867771694,
    ]
    var want: List[Float64] = [
        0.2409478422146139,
        0.23911254431355505,
        0.09718456490977953,
        6.10057131957997e-05,
    ]
    for i in range(len(x)):
        assert_true(_within_one_ulp(erfc(x[i]), want[i]), String(t"x = {x[i]}"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
