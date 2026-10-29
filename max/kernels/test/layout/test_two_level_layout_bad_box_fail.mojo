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

# RUN: not %mojo-no-debug %s 2>&1 | FileCheck %s
#
# A tile that is neither whole footprints nor an aligned power-of-two box
# inside one footprint is rejected at compile time, naming the mode.

from layout import ComptimeInt, Coord, row_major, two_level_layout_from_swizzle
from layout.swizzle import make_swizzle
from std.testing import assert_true


def main() raises:
    comptime swizzle = make_swizzle[num_rows=8, row_size=64, access_size=8]()
    comptime plain = row_major[64, 64]()
    comptime layout = two_level_layout_from_swizzle[type_of(plain), swizzle](
        plain
    )
    # 3 is not a multiple of the 8-row footprint and not a power of two.
    comptime box = Coord[ComptimeInt[3], ComptimeInt[64]].element_types
    # CHECK: TwoLevelLayout cannot tile mode 0 (footprint 8) with tile size 3
    assert_true(type_of(layout).require_tile_shape[box]())
