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
"""Empirical-verification fixture: registers the same op-names as built-in
MOGGKernelAPI kernels with deliberately wrong behavior, so a passing test
can only mean the user registration shadowed the built-in.
"""

import compiler_internal as compiler
from std.gpu.host import DeviceContext
from tensor import (
    ElementwiseBinaryOp,
    InputTensor,
)
from tensor.managed_tensor_slice import (
    _FusedInputTensor as FusedInputTensor,
)
from tensor.managed_tensor_slice import (
    _FusedOutputTensor as FusedOutputTensor,
)
from std.utils.index import IndexList


# Built-in `mo.add` returns `lhs + rhs`.  This override returns
# `lhs + rhs + 1000`, an observable sentinel that no real add would produce.
@compiler.register("mo.add")
struct AddOverride(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: Int,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs + rhs + 1000


# Built-in `mo.reduce.layer_norm` computes the actual normalization.  This
# override raises immediately; the test asserts the error surfaces, which is
# only possible if the user registration shadowed the built-in.
@compiler.register("mo.reduce.layer_norm")
struct LayerNormOverride:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma: FusedInputTensor[dtype=dtype, rank=1, ...],
        beta: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        raise Error("LAYER_NORM_OVERRIDE_FIRED")

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        beta: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
    ) -> IndexList[rank]:
        return input.shape()
