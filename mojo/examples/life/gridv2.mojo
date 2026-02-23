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

import random
from collections import Optional

from memory import memcpy, memset_zero


struct Grid[rows: Int, cols: Int](Copyable, Stringable):
    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    comptime num_cells = Self.rows * Self.cols
    var data: UnsafePointer[Int8, MutExternalOrigin]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        self.data = alloc[Int8](self.num_cells)
        memset_zero(self.data, self.num_cells)

    fn __init__(out self, *, copy: Self):
        self.data = alloc[Int8](self.num_cells)
        memcpy(dest=self.data, src=copy.data, count=self.num_cells)
        # The lifetime of `existing` continues unchanged

    fn __del__(deinit self):
        self.data.free()

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    fn random(seed: Optional[Int] = None) -> Self:
        if seed:
            random.seed(seed.value())
        else:
            random.seed()

        grid = Self()
        random.randint(grid.data, grid.num_cells, 0, 1)

        return grid^

    # ===-------------------------------------------------------------------===#
    # Indexing
    # ===-------------------------------------------------------------------===#

    fn __getitem__(self, row: Int, col: Int) -> Int8:
        return (self.data + row * Self.cols + col)[]

    fn __setitem__(mut self, row: Int, col: Int, value: Int8) -> None:
        (self.data + row * Self.cols + col)[] = value

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __str__(self) -> String:
        str = String()
        for row in range(Self.rows):
            for col in range(Self.cols):
                if self[row, col] == 1:
                    str += "*"
                else:
                    str += " "
            if row != Self.rows - 1:
                str += "\n"
        return str

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn evolve(self) -> Self:
        next_generation = Self()

        for row in range(Self.rows):
            # Calculate neighboring row indices, handling "wrap-around"
            row_above = (row - 1) % Self.rows
            row_below = (row + 1) % Self.rows

            for col in range(Self.cols):
                # Calculate neighboring column indices, handling "wrap-around"
                col_left = (col - 1) % Self.cols
                col_right = (col + 1) % Self.cols

                # Determine number of populated cells around the current cell
                num_neighbors = (
                    self[row_above, col_left]
                    + self[row_above, col]
                    + self[row_above, col_right]
                    + self[row, col_left]
                    + self[row, col_right]
                    + self[row_below, col_left]
                    + self[row_below, col]
                    + self[row_below, col_right]
                )

                if num_neighbors | self[row, col] == 3:
                    next_generation[row, col] = 1

        return next_generation^
