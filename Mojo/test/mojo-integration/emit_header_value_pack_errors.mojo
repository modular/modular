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


# RUN: kgen %s -emit=header -verify-diagnostics

from std.utils.coord import Coord


# A value pack is not split into one argument per element. Three fields are
# passed indirectly by the platform C ABI, so they cannot be spelled as one C
# parameter each.
# expected-error @+4 {{cannot declare a C prototype taking struct}}
# expected-note @+3 {{see current operation}}
# expected-error @+2 {{during header emission for this function}}
@export
def sum_coord(c: Coord[Int, Int, Int]) abi("C") -> Int:
    return c[0].value() + c[1].value() + c[2].value()


def main():
    pass
