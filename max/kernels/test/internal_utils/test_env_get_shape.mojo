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

from internal_utils import env_get_shape, parse_shape
from testing import assert_true


fn print_static_shape[x: List[Int]]():
    @parameter
    for i in range(len(x)):
        alias xi = x[i]
        print("dim", i, "=", xi)


def main():
    alias shape_mnk = parse_shape["10x20x30"]()
    print_static_shape[shape_mnk]()
    constrained[shape_mnk[0] == 10]()
    constrained[shape_mnk[1] == 20]()
    constrained[shape_mnk[2] == 30]()

    alias shape = env_get_shape["shape", "1x2x3"]()
    print_static_shape[shape]()

    constrained[shape[0] == 1]()
    constrained[shape[1] == 2]()
    constrained[shape[2] == 3]()
