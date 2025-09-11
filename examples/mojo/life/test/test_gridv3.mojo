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

from gridv3 import Grid
from testing import *

alias data4x4: List[List[Int8]] = [
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1],
]


def grid4x4() -> Grid[4, 4]:
    grid = Grid[4, 4]()
    for row in range(4):
        for col in range(4):
            grid[row, col] = materialize[data4x4]()[row][col]
    return grid


def test_gridv3_init():
    grid = Grid[4, 4]()
    assert_equal(4, grid.rows)
    assert_equal(4, grid.cols)
    for row in range(4):
        for col in range(4):
            assert_equal(Int8(0), grid[row, col])


def test_gridv3_index():
    grid = grid4x4()
    for row in range(4):
        for col in range(4):
            assert_equal(materialize[data4x4]()[row][col], grid[row, col])
            grid[row, col] = 1
            assert_equal(Int8(1), grid[row, col])
            grid[row, col] = 0
            assert_equal(Int8(0), grid[row, col])


def test_gridv3_str():
    grid = grid4x4()
    grid_str = String(grid)
    var str4x4 = " ** \n**  \n  **\n*  *"
    assert_equal(str4x4, grid_str)


def test_gridv3_evolve():
    data_gen2: List[List[Int8]] = [
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ]
    data_gen3: List[List[Int8]] = [
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
    ]

    grid_gen1 = grid4x4()

    grid_gen2 = grid_gen1.evolve()
    for row in range(4):
        for col in range(4):
            assert_equal(data_gen2[row][col], grid_gen2[row, col])

    grid_gen3 = grid_gen2.evolve()
    for row in range(4):
        for col in range(4):
            assert_equal(data_gen3[row][col], grid_gen3[row, col])


def main():
    test_gridv3_init()
    test_gridv3_index()
    test_gridv3_str()
    test_gridv3_evolve()
