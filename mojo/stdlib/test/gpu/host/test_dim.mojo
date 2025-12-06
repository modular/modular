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

from testing import assert_equal, TestSuite
from gpu.host import Dim


def test_dim_set_x_y_z():
    var dim: Dim = (10, 20, 30)
    dim.set_x(40)
    dim.set_y(50)
    dim.set_z(60)
    assert_equal(dim.x(), 40)
    assert_equal(dim.y(), 50)
    assert_equal(dim.z(), 60)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
