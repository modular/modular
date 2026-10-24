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
from extensibility import OutputTensor
from extensibility import (
    _FusedInputVariadicTensors as FusedInputVariadicTensors,
    _FusedOutputTensor as FusedOutputTensor,
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)
from layout import Coord
from max.gpu.host import DeviceContext
from nn.concat import fused_concat

from std.utils import StaticTuple
from std.utils.coord import coord_to_index_list
from std.utils.index import IndexList


@extensibility.register("reduce_buffers")
struct ReduceBuffers:
    @staticmethod
    def execute(
        output: OutputTensor[dtype=.float32, rank=1, ...],
        inputs: MutableInputVariadicTensors[dtype=DType.float32, rank=1, ...],
    ) -> None:
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
    def execute[
        P0: Int
    ](output: OutputTensor[dtype=.int32, rank=1, ...], x: SIMDPair[P0, _],):
        output.store((0,), x.x)
        output.store((P0,), x.y)


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
        dtype: DType,
        rank: Int,
        target: StaticString,
        axis: Int,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        inputs: FusedInputVariadicTensors[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert 0 <= axis < rank, "axis out of range"
        var input_shapes = StaticTuple[IndexList[rank], inputs.size]()
        comptime for i in range(inputs.size):
            input_shapes[i] = inputs[i].shape()

        # Copy-captured, as in `mo.concat`'s kernel: a by-reference capture
        # leaves a device kernel holding a host pointer on Metal.
        @inline(.always)
        def inputs_lambda[
            input_index: Int, width: Int, alignment: Int = 1
        ](coords: Coord) {var inputs} -> SIMD[dtype, width]:
            return inputs[input_index]._lambda_load[
                width=width, element_alignment=alignment
            ](coords)

        @inline(.always)
        def epilogue_wrapper[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](coords: Coord, value: SIMD[_dtype, width]) {var output}:
            output._lambda_store[width=width, element_alignment=alignment](
                coords,
                rebind[SIMD[output.dtype, width]](value),
            )

        fused_concat[dtype, rank, axis=axis, target=target](
            input_shapes,
            output.runtime_layout(),
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
