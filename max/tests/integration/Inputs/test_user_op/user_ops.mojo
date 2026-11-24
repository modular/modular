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


from std.math import iota
from extensibility import *
import extensibility
from layout import Coord

from std.utils.index import IndexList


@extensibility.register("reduce_buffers")
struct ReduceBuffers:
    @staticmethod
    def execute[
        Ins: TypeList[Trait=MutableInput & Tensor, ...], //
    ](output: Some[Output & Tensor], inputs: TensorTuple[*Ins]) -> None:
        comptime assert type_of(output).dtype == .float32
        comptime assert type_of(output).rank == 1
        comptime for i in range(len(Ins)):
            comptime assert Ins[i].dtype == .float32
            comptime assert Ins[i].rank == 1
        print("Success!")


@fieldwise_init
struct SIMDPair[S0: Int, S1: Int](ImplicitlyCopyable, RegisterPassable):
    var x: SIMD[.int32, Self.S0]
    var y: SIMD[.int32, Self.S1]


@extensibility.register("make_simd_pair")
struct MakeSimdPair:
    @staticmethod
    def execute[P0: Int, P1: Int]() -> SIMDPair[P0, P1]:
        return SIMDPair[P0, P1](iota[.int32, P0](), iota[.int32, P1](Int32(P0)))


@extensibility.register("kernel_with_parameterized_opaque")
struct ParameterizedOpaqueType:
    @staticmethod
    def execute[P0: Int](output: Some[Output & Tensor], x: SIMDPair[P0, _]):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert type_of(output).dtype == .int32
        comptime assert type_of(output).rank == 1
        var out_tile = output.to_tile_tensor[.int32]()
        out_tile.store(Coord(0), x.x)
        out_tile.store(Coord(P0), x.y)


@extensibility.register_shape_function("kernel_with_parameterized_opaque")
def kernel_with_parameterized_opaque_shape[
    P0: Int
](x: SIMDPair[P0, _]) -> IndexList[1]:
    return IndexList[1](x.S0 + x.S1)
