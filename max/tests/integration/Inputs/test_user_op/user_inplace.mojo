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

import extensibility
from extensibility import MutableInput, Tensor, TileTensorable
from std.utils.coord import Coord


@extensibility.register("mutable_test_op")
struct MutableTestOp:
    @staticmethod
    def execute(in_place_tensor: Some[MutableInput & Tensor]) raises:
        comptime assert conforms_to(type_of(in_place_tensor), TileTensorable)
        comptime dtype = type_of(in_place_tensor).dtype
        var tile = in_place_tensor.to_tile_tensor[dtype]()
        var x = tile.load[width=1](Coord(0))
        x += 1
        tile.store(Coord(0), x)


@extensibility.register("foo")
struct FooKernel:
    @staticmethod
    def execute(in_place_tensor: Some[MutableInput & Tensor]) raises:
        comptime assert conforms_to(type_of(in_place_tensor), TileTensorable)
        comptime dtype = type_of(in_place_tensor).dtype
        in_place_tensor.to_tile_tensor[dtype]().store(
            Coord(0), Scalar[dtype](0)
        )


@extensibility.register("bar")
struct BarKernel:
    @staticmethod
    def execute(in_place_tensor: Some[MutableInput & Tensor]) raises:
        comptime assert conforms_to(type_of(in_place_tensor), TileTensorable)
        comptime dtype = type_of(in_place_tensor).dtype
        in_place_tensor.to_tile_tensor[dtype]().store(
            Coord(0), Scalar[dtype](0)
        )


@extensibility.register("baz")
struct BazKernel:
    @staticmethod
    def execute(in_place_tensor: Some[MutableInput & Tensor]) raises:
        comptime assert conforms_to(type_of(in_place_tensor), TileTensorable)
        comptime dtype = type_of(in_place_tensor).dtype
        in_place_tensor.to_tile_tensor[dtype]().store(
            Coord(0), Scalar[dtype](0)
        )
