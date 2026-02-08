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
"""RMSNorm fused residual op registration for state space models."""

import compiler_internal as compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

from utils.index import IndexList

from state_space.rms_norm_fused_residual import rms_norm_fused_residual


# ===----------------------------------------------------------------------=== #
# RMSNorm Fused Residual Operation
# ===----------------------------------------------------------------------=== #


@compiler.register("rms_norm_fused_residual")
struct RMSNormFusedResidual:
    """RMS normalization with fused residual connection for Mamba blocks.

    Performs RMS normalization on (input + residual), returning both the
    normalized output and the pre-normalized input (residual output).
    This matches the fused residual + norm pattern used in Mamba models.

    Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layer_norm.py

    Tensor Shapes:
        - output: (..., hidden_size) - Normalized output tensor (same shape as input).
        - residual_output: (..., hidden_size) - Pre-normalized input (input + residual).
        - input: (..., hidden_size) - Input tensor to normalize.
        - residual_input: (..., hidden_size) - Residual tensor to add before normalization.
        - weight: (hidden_size,) - Weight tensor (gamma) for normalization.
        - eps: Scalar - Epsilon value for numerical stability (default: 1e-6).
        - weight_offset: Scalar - Offset added to weight before normalization (default: 0.0).
        - dropout_p: Scalar - Dropout probability (default: 0.0).
        - seed: Scalar[uint64] - Random seed for dropout (default: 0).

    Compile-time Options:
        - multiply_before_cast: If True, multiplies by weight before casting to output dtype.
          If False, casts to output dtype before multiplying by weight.
    """

    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        residual_output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        residual_input: InputTensor[dtype=dtype, rank=rank],
        gamma: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        dropout_p: Scalar[dtype=dtype],
        seed: Scalar[dtype = DType.uint64],
        ctx: DeviceContextPtr,
    ) capturing raises:
        # Validate shapes
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        if input.shape() != residual_input.shape():
            raise Error("Input and residual input buffers are not same shape")

        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input.load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn residual_input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return residual_input.load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            output._fused_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        @parameter
        @always_inline
        fn residual_output_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            residual_output._fused_store[
                width=width, element_alignment=alignment
            ](
                rebind[IndexList[residual_output.rank]](coords),
                rebind[SIMD[residual_output.dtype, width]](val),
            )

        rms_norm_fused_residual[
            input_fn,
            residual_input_fn,
            output_fn,
            residual_output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            input.shape(),
            gamma.to_layout_tensor(),
            epsilon,
            weight_offset,
            ctx,
            dropout_p,
            UInt64(seed),
        )

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        residual_input: InputTensor[dtype=dtype, rank=rank],
        gamma: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        dropout_p: Scalar[dtype=dtype],
        seed: Scalar[dtype = DType.uint64],
    ) -> IndexList[rank]:
        return input.shape()
