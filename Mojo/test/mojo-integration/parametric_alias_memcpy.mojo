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

# RUN: %mojo %s | FileCheck %s

# `lit.memcpy` accepts element types that only match canonically, such as a
# parametric alias and its expansion. Make sure that still lowers and runs.


struct Tile[n: Int](ImplicitlyCopyable):
    var x: Int
    var y: Int

    def __init__(out self):
        self.x = Self.n
        self.y = Self.n + 1


comptime DoubledTile[n: Int] = Tile[n * 2]


def make_tile[n: Int]() -> DoubledTile[n]:
    return {}


def copy_into_array_generic[n: Int]() -> Int:
    var tile = make_tile[n]()
    var tiles: Array[type_of(tile), 1] = [tile]
    return tiles[0].x + tiles[0].y


def main():
    var tile = make_tile[2]()
    var tiles: Array[type_of(tile), 1] = [tile]
    # CHECK: 4 5
    print(tiles[0].x, tiles[0].y)
    # CHECK: 13
    print(copy_into_array_generic[3]())
