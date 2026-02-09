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


from math import iota
from register import *
import compiler_internal as compiler
from tensor import OutputTensor
from tensor.managed_tensor_slice import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)

from utils.index import IndexList


@compiler.register("reduce_buffers")
struct ReduceBuffers:
    @staticmethod
    fn execute(
        output: OutputTensor[dtype = DType.float32, rank=1, ...],
        inputs: MutableInputVariadicTensors[dtype = DType.float32, rank=1, ...],
    ) -> None:
        print("Success!")


@fieldwise_init
struct SIMDPair[S0: Int, S1: Int](ImplicitlyCopyable, RegisterPassable):
    var x: SIMD[DType.int32, Self.S0]
    var y: SIMD[DType.int32, Self.S1]


@compiler.register("make_simd_pair")
struct MakeSimdPair:
    @staticmethod
    fn execute[P0: Int, P1: Int]() -> SIMDPair[P0, P1]:
        return SIMDPair[P0, P1](
            iota[DType.int32, P0](), iota[DType.int32, P1](Int32(P0))
        )


@compiler.register("kernel_with_parameterized_opaque")
struct ParameterizedOpaqueType:
    @staticmethod
    fn execute[
        P0: Int
    ](
        output: OutputTensor[dtype = DType.int32, rank=1], x: SIMDPair[P0, _]
    ) capturing:
        output.store(IndexList[1](0), x.x)
        output.store(IndexList[1](P0), x.y)

    @staticmethod
    fn shape[P0: Int](x: SIMDPair[P0, _]) -> IndexList[1]:
        return IndexList[1](x.S0 + x.S1)
