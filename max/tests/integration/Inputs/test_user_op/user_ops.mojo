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
from layout import Coord, row_major
from max.gpu.host import DeviceContext
from nn.concat import fused_concat

from std.utils import StaticTuple
from std.utils.coord import DynamicCoord, coord_to_index_list
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


# `concat_opaque` computes exactly what `mo.concat` does, through the same
# `fused_concat`, but as a custom op the graph compiler can only treat as an
# opaque kernel: tests of fusion into an opaque kernel's prologue or epilogue
# use it where they need one shaped like a concat.
@extensibility.register("concat_opaque")
struct ConcatOpaque:
    @staticmethod
    def execute[
        Ins: TypeList[Trait=Input & Fused & LoadAccess, ...],
        //,
        axis: Int,
    ](
        output: Some[Output & Fused & StoreAccess],
        inputs: TensorTuple[*Ins],
        ctx: DeviceContext,
    ) raises:
        comptime dtype = type_of(output).dtype
        comptime rank = type_of(output).rank
        comptime assert 0 <= axis < rank, "axis out of range"
        var input_shapes = StaticTuple[DynamicCoord[.int, rank], len(Ins)]()
        comptime for i in range(len(Ins)):
            comptime assert Ins[i].dtype == dtype, "inputs must share a dtype"
            comptime assert Ins[i].rank == rank, "inputs must share a rank"
            # `fused_concat` threads the input shapes as dynamic `Coord`s. One
            # is built explicitly per input because the tensors' shape types
            # can differ in their static-vs-dynamic split, while the tuple's
            # element type must be uniform.
            var input_shape = DynamicCoord[.int, rank]()
            comptime for d in range(rank):
                input_shape.replace_dynamic[d](Int(inputs[i].shape()[d]))
            input_shapes[i] = input_shape

        # Copy-captured, as in `mo.concat`'s kernel: a by-reference capture
        # leaves a device kernel holding a host pointer on Metal.
        var input_views = TensorTuple._fused_view(inputs)
        comptime assert conforms_to(type_of(output), UnsafeFusedView)
        var out_view = unsafe_fused_view(output)
        comptime assert conforms_to(type_of(out_view), Storable)

        @inline(.always)
        def inputs_lambda[
            input_index: Int, width: Int, alignment: Int = 1
        ](coords: Coord) {var input_views} -> SIMD[dtype, width]:
            comptime assert conforms_to(
                type_of(input_views[input_index]), Loadable
            )
            return rebind[SIMD[dtype, width]](
                input_views[input_index].load[
                    width=width, element_alignment=alignment
                ](coords)
            )

        @inline(.always)
        def epilogue_wrapper[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](coords: Coord, value: SIMD[_dtype, width]) {var out_view}:
            out_view.store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[type_of(out_view).dtype, width]](value),
            )

        fused_concat[dtype, rank, axis=axis](
            input_shapes,
            row_major(output.shape()),
            inputs_lambda,
            epilogue_wrapper,
            ctx,
        )


@extensibility.register_shape_function("concat_opaque")
def concat_opaque_shape_fn[
    Ins: TypeList[Trait=Tensor, ...],
    //,
    axis: Int,
](inputs: TensorTuple[*Ins]) raises -> IndexList[Ins[0].rank]:
    comptime rank = Ins[0].rank
    var shape = rebind[IndexList[rank]](coord_to_index_list(inputs[0].shape()))
    var extent = 0
    comptime for i in range(len(Ins)):
        var input_shape = rebind[IndexList[rank]](
            coord_to_index_list(inputs[i].shape())
        )
        comptime for d in range(rank):
            if d != axis and input_shape[d] != shape[d]:
                raise Error(
                    "[concat_opaque] input shapes must match except at the"
                    " concat axis"
                )
        extent += input_shape[axis]
    shape[axis] = extent
    return shape
