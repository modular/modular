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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# A trivially copyable, memory-only value is copied with `lit.memcpy`. When the
# source is typed by a parametric alias and the destination slot by its
# expansion, the element types only match canonically, which `lit.memcpy`
# accepts without a rebind.


struct Tile[n: Int](ImplicitlyCopyable):
    var x: Int
    var y: Int

    def __init__(out self):
        self.x = 0
        self.y = 0


comptime DoubledTile[n: Int] = Tile[n * 2]


def make_tile[n: Int]() -> DoubledTile[n]:
    return {}


# CHECK-LABEL: lit.fn @"copy_into_array()"
def copy_into_array():
    var tile = make_tile[2]()
    # CHECK-NOT: kgen.rebind %tile
    # CHECK: lit.memcpy %tile, %{{.*}} : <:meta<!lit.struct<#Tile {{.*}}sugar_preserved({{.*}} -> <!lit.struct<#Tile <:!Int {:scalar<index> 4}>>
    var tiles: Array[type_of(tile), 1] = [tile]
    _ = tiles^


# CHECK-LABEL: lit.fn @"copy_into_array_generic
def copy_into_array_generic[n: Int]():
    var tile = make_tile[n]()
    # CHECK-NOT: kgen.rebind %tile
    # CHECK: lit.memcpy %tile, %{{.*}} : {{.*}}sugar_preserved({{.*}} -> <!lit.struct<#Tile
    var tiles: Array[type_of(tile), 1] = [tile]
    _ = tiles^
