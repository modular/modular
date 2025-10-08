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

from algorithm import (
    tile,
    tile_and_unswitch,
    tile_middle_unswitch_boundaries,
    unswitch,
)
from testing import TestSuite

from utils.index import Index, IndexList


# Helper workgroup function to test dynamic workgroup tiling.
@always_inline
@parameter
fn print_number_dynamic(data_idx: Int, tile_size: Int):
    # Print out the range of workload that this launched instance is
    #  processing, in (begin, end).
    print(Index(data_idx, data_idx + tile_size))


# Helper workgroup function to test static workgroup tiling.
@always_inline
@parameter
fn print_number_static[tile_size: Int](data_idx: Int):
    print_number_dynamic(data_idx, tile_size)


# Helper workgroup function to test static workgroup tiling.
@always_inline
@parameter
fn print_tile2d_static[
    tile_size_x: Int, tile_size_y: Int
](offset_x: Int, offset_y: Int):
    print(Index(tile_size_x, tile_size_y, offset_x, offset_y))


# CHECK-LABEL: test_static_tile
def test_static_tile():
    print("test_static_tile")
    # CHECK: (0, 4)
    # CHECK: (4, 6)
    tile[print_number_static, VariadicList[Int](4, 3, 2, 1)](0, 6)
    # CHECK: (0, 4)
    # CHECK: (4, 8)
    tile[print_number_static, VariadicList[Int](4, 3, 2, 1)](0, 8)
    # CHECK: (1, 5)
    # CHECK: (5, 6)
    tile[print_number_static, VariadicList[Int](4, 3, 2, 1)](1, 6)


# CHECK-LABEL: test_static_tile2d
def test_static_tile2d():
    print("test_static_tile2d")
    # CHECK: (2, 2, 0, 0)
    # CHECK: (2, 2, 2, 0)
    # CHECK: (2, 2, 4, 0)
    # CHECK: (2, 2, 0, 2)
    # CHECK: (2, 2, 2, 2)
    # CHECK: (2, 2, 4, 2)
    # CHECK: (2, 2, 0, 4)
    # CHECK: (2, 2, 2, 4)
    # CHECK: (2, 2, 4, 4)
    # CHECK: ========
    tile[print_tile2d_static, VariadicList(2), VariadicList(2)](0, 0, 6, 6)
    print("========")
    # CHECK: (4, 4, 4, 4)
    # CHECK: (4, 4, 8, 4)
    # CHECK: (4, 4, 12, 4)
    # CHECK: (4, 4, 4, 8)
    # CHECK: (4, 4, 8, 8)
    # CHECK: (4, 4, 12, 8)
    # CHECK: (4, 4, 4, 12)
    # CHECK: (4, 4, 8, 12)
    # CHECK: (4, 4, 12, 12)
    # CHECK: ========
    tile[print_tile2d_static, VariadicList(4), VariadicList(4)](4, 4, 16, 16)
    print("========")
    # CHECK: (3, 4, 1, 1)
    # CHECK: (3, 4, 4, 1)
    # CHECK: (3, 4, 7, 1)
    # CHECK: (1, 4, 10, 1)
    # CHECK: (1, 4, 11, 1)
    # CHECK: (3, 1, 1, 5)
    # CHECK: (3, 1, 4, 5)
    # CHECK: (3, 1, 7, 5)
    # CHECK: (1, 1, 10, 5)
    # CHECK: (1, 1, 11, 5)
    # CHECK: (3, 1, 1, 6)
    # CHECK: (3, 1, 4, 6)
    # CHECK: (3, 1, 7, 6)
    # CHECK: (1, 1, 10, 6)
    # CHECK: (1, 1, 11, 6)
    tile[print_tile2d_static, VariadicList(3, 1), VariadicList(4, 1)](
        1, 1, 12, 7
    )


# CHECK-LABEL: test_dynamic_tile
def test_dynamic_tile():
    print("test_dynamic_tile")
    # CHECK: (1, 4)
    # CHECK: (4, 5)
    tile[print_number_dynamic](1, 5, VariadicList[Int](3, 2))
    # CHECK: (0, 4)
    # CHECK: (4, 5)
    # CHECK: (5, 6)
    tile[print_number_dynamic](0, 6, VariadicList[Int](4, 1))
    # CHECK: (2, 7)
    # CHECK: (7, 12)
    # CHECK: (12, 15)
    # CHECK: (15, 16)
    tile[print_number_dynamic](2, 16, VariadicList[Int](5, 3))


# CHECK-LABEL: test_unswitched_tile
def test_unswitched_tile():
    print("test_unswitched_tile")

    # A tiled function that takes a start and a dynamic boundary.
    @always_inline
    @parameter
    fn switched_tile[tile_size: Int](start: Int, bound: Int):
        # Inside each unit there's either a per-element check or a unswitched
        #  tile level check.
        @always_inline
        @parameter
        fn switched_tile_unit[static_switch: Bool]():
            for i in range(start, start + tile_size):
                if static_switch or i < bound:
                    print(i)

        # Use unswitch on the tiled unit.
        unswitch[switched_tile_unit](start + tile_size <= bound)

    # CHECK: 5
    # CHECK: 6
    # CHECK: 7
    switched_tile[4](5, 8)

    # CHECK: 5
    # CHECK: 6
    switched_tile[2](5, 8)


# CHECK-LABEL: test_unswitched_2d_tile
def test_unswitched_2d_tile():
    print("test_unswitched_2d_tile")

    # A tiled function that takes a start and a dynamic boundary.
    @parameter
    @always_inline
    fn switched_tile[
        tile_size_x: Int, tile_size_y: Int
    ](start: IndexList[2], bound: IndexList[2]):
        var tile_size = Index(tile_size_x, tile_size_y)

        # Inside each unit there's either a per-element check or a unswitched
        #  tile level check.
        @always_inline
        @__copy_capture(tile_size)
        @parameter
        fn switched_tile_unit[static_switch0: Bool, static_switch1: Bool]():
            for i in range(start[0], start[0] + tile_size[0]):
                for j in range(start[1], start[1] + tile_size[1]):
                    if static_switch0 or i < bound[0]:
                        if static_switch1 or j < bound[1]:
                            print(Index(i, j))

        # Use unswitch on the tiled unit.
        var tile_end_point = start + tile_size
        unswitch[switched_tile_unit](
            tile_end_point[0] <= bound[0], tile_end_point[1] <= bound[1]
        )

    # CHECK: (1, 2)
    # CHECK: (1, 3)
    # CHECK: (1, 4)
    switched_tile[2, 3](Index(1, 2), Index(2, 6))
    # CHECK: (1, 2)
    # CHECK: (1, 3)
    # CHECK: (2, 2)
    # CHECK: (2, 3)
    switched_tile[2, 3](Index(1, 2), Index(4, 4))


# CHECK-LABEL: test_tile_and_unswitch
def test_tile_and_unswitch():
    print("test_tile_and_unswitch")

    @parameter
    # Helper workgroup function to test static workgroup tiling.
    @always_inline
    fn print_number_static_unswitched[
        tile_size: Int, static_switch: Bool
    ](data_idx: Int, upperbound: Int):
        print(Index(data_idx, tile_size, upperbound))
        print("Unswitched:", static_switch)

    # CHECK: (0, 4, 6)
    # CHECK: Unswitched: True
    # CHECK: (4, 2, 6)
    # CHECK: Unswitched: True
    tile_and_unswitch[
        print_number_static_unswitched, VariadicList[Int](4, 3, 2)
    ](0, 6)
    # CHECK: (0, 4, 8)
    # CHECK: Unswitched: True
    # CHECK: (4, 4, 8)
    # CHECK: Unswitched: True
    tile_and_unswitch[
        print_number_static_unswitched, VariadicList[Int](4, 3, 2)
    ](0, 8)
    # CHECK: (1, 4, 6)
    # CHECK: Unswitched: True
    # CHECK: (5, 2, 6)
    # CHECK: Unswitched: False
    tile_and_unswitch[
        print_number_static_unswitched, VariadicList[Int](4, 3, 2)
    ](1, 6)
    # CHECK: (4, 4, 8)
    # CHECK: Unswitched: True
    tile_and_unswitch[print_number_static_unswitched, VariadicList[Int](4, 1)](
        4, 8
    )


def test_tile_middle_unswitch_boundaries():
    print("test_tile_middle_unswitch_boundaries")

    @parameter
    @always_inline
    fn print_wrapper[tile_size: Int, switch: Bool](offset: Int):
        print(offset, tile_size, switch)

    # CHECK: 0 1 True
    # CHECK: 1 4 False
    # CHECK: 5 2 False
    # CHECK: 7 1 True
    tile_middle_unswitch_boundaries[
        print_wrapper,
        VariadicList[Int](4, 3, 2, 1),
    ](0, 1, 7, 8)

    # CHECK: 1 1 True
    # CHECK: 2 1 True
    # CHECK: 3 3 False
    # CHECK: 6 1 True
    # CHECK: 7 1 True
    tile_middle_unswitch_boundaries[
        print_wrapper,
        VariadicList[Int](4, 3, 1),
    ](1, 3, 6, 8)

    # CHECK: 0 2 True
    # CHECK: 2 6 False
    # CHECK: 8 2 False
    # CHECK: 10 2 True
    # CHECK: 12 2 True
    tile_middle_unswitch_boundaries[
        print_wrapper,
        VariadicList[Int](6, 4, 2, 1),
        left_tile_size=2,
        right_tile_size=2,
    ](0, 2, 10, 14)


def test_tile_middle_unswitch_boundaries_static():
    print("test_tile_middle_unswitch_boundaries_static")

    @parameter
    @always_inline
    fn print_wrapper[tile_size: Int, lflag: Bool, rflag: Bool](offset: Int):
        print(offset, tile_size, lflag, rflag)

    # CHECK: 0 2 True True
    tile_middle_unswitch_boundaries[print_wrapper, 2, 2]()

    # CHECK: 0 2 True False
    # CHECK: 2 3 False True
    tile_middle_unswitch_boundaries[print_wrapper, 4, 5]()

    # CHECK: 0 3 True False
    # CHECK: 3 3 False False
    # CHECK: 6 2 False False
    # CHECK: 8 3 False True
    tile_middle_unswitch_boundaries[print_wrapper, 3, 11]()


def main():
    var suite = TestSuite()

    suite.test[test_static_tile]()
    suite.test[test_static_tile2d]()
    suite.test[test_dynamic_tile]()
    suite.test[test_unswitched_tile]()
    suite.test[test_unswitched_2d_tile]()
    suite.test[test_tile_and_unswitch]()
    suite.test[test_tile_middle_unswitch_boundaries]()
    suite.test[test_tile_middle_unswitch_boundaries_static]()

    suite^.run()
