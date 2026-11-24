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

# `CoordLike` derives from `Intable`, so `Int(coord)` compiles for any
# coordinate. The conversion is only defined for scalar values: converting a
# tuple `Coord` must fail `Coord.value()`'s constraint at compile time.

# RUN: not %mojo -D test=1 %s 2>&1 | FileCheck --check-prefix CHECK_1 %s
# RUN: not %mojo -D test=2 %s 2>&1 | FileCheck --check-prefix CHECK_2 %s

from std.sys import get_defined_int
from std.utils.coord import Coord, CoordLike, Idx


def to_int[T: CoordLike](c: T) -> Int:
    return Int(c)


def main():
    var coord_tuple = Coord(Idx[2], Int(3))

    comptime if get_defined_int["test"]() == 1:
        # Direct `Int` construction from a tuple `Coord`.
        # CHECK_1: constraint failed: Coord is not a value type
        print(Int(coord_tuple))
    elif get_defined_int["test"]() == 2:
        # The generic path layout code takes: a `CoordLike` bound converted
        # straight to `Int`.
        # CHECK_2: constraint failed: Coord is not a value type
        print(to_int(coord_tuple))
