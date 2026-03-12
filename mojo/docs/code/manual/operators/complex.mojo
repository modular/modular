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

from std.testing import assert_almost_equal


@fieldwise_init
struct Complex(Copyable, RegisterPassable):
    var re: Float64
    var im: Float64

    def __init__(out self, re: Float64):
        self.re = re
        self.im = 0.0


def main() raises:
    var c1 = Complex(-1.2, 6.5)
    assert_almost_equal(c1.re, -1.2)
    assert_almost_equal(c1.im, 6.5)
    var c_copy = c1.copy()
    assert_almost_equal(c_copy.re, -1.2)
    assert_almost_equal(c_copy.im, 6.5)
    var c2 = Complex(-1.2)
    assert_almost_equal(c2.re, -1.2)
    assert_almost_equal(c2.im, 0.0)
