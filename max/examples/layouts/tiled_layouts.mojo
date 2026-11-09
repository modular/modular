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

# DOC: layout/layouts.mdx

from layout import Coord, coord, Idx
from layout.layout import (
    Layout,
    blocked_product,
    col_major,
    col_major_nested,
    row_major,
    row_major_nested,
    zipped_divide,
)


def use_layout_constructor():
    print("layout constructor")
    var tiled_layout = Layout(
        Coord(coord[3, 2], coord[2, 5]),  # shape
        Coord(coord[1, 6], coord[3, 12]),  # strides
    )
    print(tiled_layout)
    print()


def use_blocked_product():
    print("blocked product")
    # Define a 3x2 tile
    var tile = col_major[3, 2]()
    # Define a 2x5 tiler
    var tiler = col_major[2, 5]()
    var blocked = blocked_product(tile, tiler)

    print("Tile:")
    print(tile)
    print("\nTiler:")
    print(tiler)
    print("\nTiled layout:")
    print(blocked)
    print()


def use_zipped_divide():
    print("zipped divide")
    comptime base = row_major[6, 4]()
    comptime result = zipped_divide[coord[2, 2]](base)
    print(base)
    print(result)
    print(result(coord[1, 1]))
    print(result(coord[1]))
    print(result(Coord(coord[1, 0], coord[1, 0])))
    print()


def use_nested_layouts():
    print("nested layouts")
    # A re-nested row-major layout: a 2x2 outer grid of 3x4 inner
    # fragments. Shape ((2, 3), (2, 4)) flattens to (2, 3, 2, 4),
    # producing row-major strides (24, 8, 4, 1), re-nested as
    # ((24, 8), (4, 1)).
    var nested = row_major_nested(
        Coord(Coord(Idx[2], Idx[3]), Coord(Idx[2], Idx[4]))
    )
    print(nested)

    # The column-major variant re-nests col-major strides over the same
    # flattened shape: (1, 2, 6, 12), re-nested as ((1, 2), (6, 12)).
    var nested_col = col_major_nested(
        Coord(Coord(Idx[2], Idx[3]), Coord(Idx[2], Idx[4]))
    )
    print(nested_col)
    print()


def main() raises:
    use_layout_constructor()
    use_blocked_product()
    use_zipped_divide()
    use_nested_layouts()
