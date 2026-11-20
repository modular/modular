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

from std.os import abort

import extensibility
from extensibility import Input, Output, Tensor, TileTensorable

from std.utils.coord import Coord


struct Counter[stride: Int](Movable):
    var a: Int
    var b: Int

    def __init__(out self):
        self.a = 0
        self.b = 0
        print("counter init (no arg)")

    def __init__(out self, a: Int, b: Int):
        self.a = a
        self.b = b
        print("counter init", a, b)

    def __deinit__(deinit self):
        print("counter deinit")

    def bump(mut self):
        self.a += Self.stride
        self.b += self.a
        print("bumped", self.a, self.b)


@extensibility.register("make_counter_from_tensor")
struct MakeCounterFromTensor:
    @staticmethod
    def execute[
        stride: Int,
    ](init: Some[Input & Tensor]) -> Counter[stride]:
        comptime assert conforms_to(type_of(init), TileTensorable)
        comptime assert type_of(init).dtype == .int32
        comptime assert type_of(init).rank == 1
        var init_tile = init.to_tile_tensor[.int32]().as_imm()
        var a = init_tile.load[width=1](Coord(0))
        var b = init_tile.load[width=1](Coord(1))
        print("making. init:", a, b)
        return Counter[stride](Int(a), Int(b))


@extensibility.register("make_counter")
struct MakeCounter:
    @staticmethod
    def execute[stride: Int]() -> Counter[stride]:
        print("making")
        return Counter[stride]()


@extensibility.register("bump_counter")
struct BumpCounter:
    @staticmethod
    def execute[
        stride: Int,
    ](mut c: Counter[stride]) -> None:
        print("bumping")
        c.bump()


@extensibility.register("read_counter")
struct ReadCounter:
    @staticmethod
    def execute[stride: Int](output: Some[Output & Tensor], c: Counter[stride]):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert type_of(output).dtype == .int32
        comptime assert type_of(output).rank == 1
        var output_tile = output.to_tile_tensor[.int32]()
        output_tile.store[width=1](Coord(0), Int32(c.a))
        output_tile.store[width=1](Coord(1), Int32(c.b))
