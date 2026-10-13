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

# The signs of j1 and y1 at the infinities.

from std.math import inf, isnan, j1, y1
from std.testing import assert_equal, assert_true, TestSuite


def _check_j1_y1_infinities[
    dtype: DType
]() raises where dtype.is_floating_point():
    var pos_zero = Scalar[dtype](0).to_bits()
    assert_equal(j1(inf[dtype]()).to_bits(), pos_zero)
    assert_equal(j1(-inf[dtype]()).to_bits(), pos_zero)
    assert_equal(y1(inf[dtype]()).to_bits(), pos_zero)
    assert_true(isnan(y1(-inf[dtype]())))


def test_j1_y1_infinities() raises:
    _check_j1_y1_infinities[.float32]()
    _check_j1_y1_infinities[.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
