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

from internal_utils import env_get_shape, parse_shape
from testing import assert_true


fn print_static_shape[x: List[Int]]():
    comptime for i in range(len(x)):
        comptime xi = x[i]
        print("dim", i, "=", xi)


def main():
    comptime shape_mnk = parse_shape["10x20x30"]()
    print_static_shape[shape_mnk]()
    comptime assert shape_mnk[0] == 10
    comptime assert shape_mnk[1] == 20
    comptime assert shape_mnk[2] == 30

    comptime shape = env_get_shape["shape", "1x2x3"]()
    print_static_shape[shape]()

    comptime assert shape[0] == 1
    comptime assert shape[1] == 2
    comptime assert shape[2] == 3
