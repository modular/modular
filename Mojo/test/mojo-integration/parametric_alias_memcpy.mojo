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
# RUN: %mojo -O0 %s | FileCheck %s
# RUN: %mojo -O3 %s | FileCheck %s

# `lit.memcpy` accepts element types that only match canonically, such as a
# parametric alias and its expansion. Make sure that still lowers and runs.

from std.benchmark import black_box


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


# Copying a value that may be the destination itself, as in `v = v` or a
# ternary that may pick `v`, must read the value before it is replaced, at
# every optimization level.
# Issue #7760: https://github.com/modular/modular/issues/7760
@fieldwise_init
struct One(ImplicitlyCopyable):
    var x: Int


@fieldwise_init
struct Three(ImplicitlyCopyable):
    var x: Int
    var y: Int
    var z: Int


def self_assign_one(a: Int) -> Int:
    var v = One(a)
    v = v
    return v.x


def self_assign_three(a: Int) -> Int:
    var v = Three(a, a, a)
    v = v
    return v.x + v.y + v.z


def keep_or_other(a: Int, c: Bool) -> Int:
    var v = Three(a, a, a)
    var w = Three(a + 1, a, a)
    v = v if c else w
    return v.x + v.y + v.z


def keep_or_new(a: Int, c: Bool) -> Int:
    var v = Three(a, a, a)
    v = v if c else Three(9, 9, 9)
    return v.x + v.y + v.z


def other_or_keep(a: Int, c: Bool) -> Int:
    var v = Three(a, a, a)
    var w = Three(a + 1, a, a)
    v = w if c else v
    return v.x + v.y + v.z


def main():
    var tile = make_tile[2]()
    var tiles: Array[type_of(tile), 1] = [tile]
    # CHECK: 4 5
    print(tiles[0].x, tiles[0].y)
    # CHECK: 13
    print(copy_into_array_generic[3]())

    var a = black_box(4)
    var c = black_box(True)
    # CHECK: self_assign 4 12
    print("self_assign", self_assign_one(a), self_assign_three(a))
    # CHECK: ternary 12 12 12
    print(
        "ternary", keep_or_other(a, c), keep_or_new(a, c), other_or_keep(a, not c)
    )
