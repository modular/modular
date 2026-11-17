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

import extensibility

from max.gpu.host import DeviceContext
from max.gpu.host.info import is_gpu
from extensibility import Input, Output, Tensor, TileTensorable

from layout import Coord
from std.sys import align_of
from std.utils.coord import coord_to_index_list
from std.utils.index import IndexList

from state_space.rms_norm_fused_residual import (
    _rms_norm_fused_residual_cpu_entry,
    rms_norm_fused_residual,
)


# ===----------------------------------------------------------------------=== #
# RMSNorm Fused Residual Operation
# ===----------------------------------------------------------------------=== #


@extensibility.register("rms_norm_fused_residual")
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
    def execute[
        dtype: DType,
        multiply_before_cast: Bool = True,
    ](
        output: Some[Output & Tensor],
        residual_output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        residual_input: Some[Input & Tensor],
        gamma: Some[Input & Tensor],
        epsilon: Float32,
        weight_offset: Scalar[dtype=dtype],
        dropout_p: Scalar[dtype=dtype],
        seed: Scalar[dtype=DType.uint64],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(residual_output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(residual_input), TileTensorable)
        comptime assert conforms_to(type_of(gamma), TileTensorable)
        comptime rank = type_of(input).rank
        comptime assert type_of(output).rank == rank
        comptime assert type_of(residual_output).rank == rank
        comptime assert type_of(residual_input).rank == rank
        comptime assert type_of(gamma).rank == 1

        # Validate shapes
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        if input.shape() != residual_input.shape():
            raise Error("Input and residual input buffers are not same shape")

        if residual_output.shape() != input.shape():
            raise Error("Input and residual output buffers are not same shape")

        var input_tt = input.to_tile_tensor[dtype]()
        var residual_input_tt = residual_input.to_tile_tensor[dtype]()
        var output_tt = output.to_tile_tensor[dtype]()
        var residual_output_tt = residual_output.to_tile_tensor[dtype]()

        comptime if ctx.target.is_gpu():

            @inline(.always)
            def input_fn[
                width: Int
            ](coords: Coord) {var input_tt} -> SIMD[dtype, width]:
                return input_tt.load[width=width, alignment=align_of[dtype]()](
                    coords
                )

            @inline(.always)
            def residual_input_fn[
                width: Int
            ](coords: Coord) {var residual_input_tt} -> SIMD[dtype, width]:
                return residual_input_tt.load[
                    width=width, alignment=align_of[dtype]()
                ](coords)

            @inline(.always)
            def output_fn[
                width: SIMDLength, alignment: Int
            ](coords: Coord, val: SIMD[dtype, width]) {var output_tt} -> None:
                output_tt.store[
                    width=width,
                    alignment=min(
                        type_of(output).alignment,
                        alignment * align_of[dtype](),
                    ),
                ](coords, val)

            @inline(.always)
            def residual_output_fn[
                width: SIMDLength, alignment: Int
            ](coords: Coord, val: SIMD[dtype, width]) {
                var residual_output_tt
            } -> None:
                residual_output_tt.store[
                    width=width,
                    alignment=min(
                        type_of(residual_output).alignment,
                        alignment * align_of[dtype](),
                    ),
                ](coords, val)

            rms_norm_fused_residual[multiply_before_cast=multiply_before_cast,](
                input_fn,
                residual_input_fn,
                output_fn,
                residual_output_fn,
                input.shape(),
                gamma.to_tile_tensor[dtype](),
                epsilon,
                weight_offset,
                ctx,
                dropout_p,
                UInt64(seed),
            )
        else:
            # CPU path: pass the callbacks as unified closures (runtime args).
            # They capture the tensors directly, which is what lets the migrated
            # CPU kernel take runtime closures end to end.
            @inline(.always)
            def input_fn_cpu[
                width: Int
            ](coords: Coord) {var input_tt} -> SIMD[dtype, width]:
                return input_tt.load[width=width, alignment=align_of[dtype]()](
                    coords
                )

            @inline(.always)
            def residual_input_fn_cpu[
                width: Int
            ](coords: Coord) {var residual_input_tt} -> SIMD[dtype, width]:
                return residual_input_tt.load[
                    width=width, alignment=align_of[dtype]()
                ](coords)

            @inline(.always)
            def output_fn_cpu[
                width: SIMDLength, alignment: Int
            ](coords: Coord, val: SIMD[dtype, width]) {var output_tt} -> None:
                output_tt.store[
                    width=width,
                    alignment=min(
                        type_of(output).alignment,
                        alignment * align_of[dtype](),
                    ),
                ](coords, val)

            @inline(.always)
            def residual_output_fn_cpu[
                width: SIMDLength, alignment: Int
            ](coords: Coord, val: SIMD[dtype, width]) {
                var residual_output_tt
            } -> None:
                residual_output_tt.store[
                    width=width,
                    alignment=min(
                        type_of(residual_output).alignment,
                        alignment * align_of[dtype](),
                    ),
                ](coords, val)

            _rms_norm_fused_residual_cpu_entry[
                multiply_before_cast=multiply_before_cast
            ](
                input_fn_cpu,
                residual_input_fn_cpu,
                output_fn_cpu,
                residual_output_fn_cpu,
                input.shape(),
                gamma.to_tile_tensor[dtype](),
                epsilon,
                weight_offset,
                dropout_p,
                UInt64(seed),
            )


@extensibility.register_shape_function("rms_norm_fused_residual")
def rms_norm_fused_residual_shape[
    dtype: DType,
](
    input: Some[Tensor],
    residual_input: Some[Tensor],
    gamma: Some[Tensor],
    epsilon: Float32,
    # The scalars bind their own `dtype` parameter. The graph compiler's
    # scalar-attribute binding cannot resolve a trait projection like
    # `type_of(input).dtype`. The `comptime assert`s below restore the tie.
    weight_offset: Scalar[dtype=dtype],
    dropout_p: Scalar[dtype=dtype],
    seed: Scalar[dtype=DType.uint64],
) -> IndexList[type_of(input).rank]:
    """Returns the output shape for the `rms_norm_fused_residual` op.

    The fused RMS-norm-plus-residual op preserves the input shape; both the
    normalized output and the updated residual have the same shape as `input`.

    Args:
        input: Input tensor to normalize.
        residual_input: Residual tensor added before normalization.
        gamma: Scale (weight) vector with shape `(last_dim,)`.
        epsilon: Small constant added to the RMS for numerical stability.
        weight_offset: Scalar offset added to `gamma` before scaling.
        dropout_p: Dropout probability applied to `input` before the residual
            add (0.0 disables dropout).
        seed: RNG seed for dropout.

    Returns:
        The output shape, equal to `input.shape()`.
    """
    comptime assert (
        type_of(input).dtype == dtype
    ), "input dtype must match the scalar dtype"
    comptime assert (
        type_of(residual_input).rank == type_of(input).rank
    ), "residual_input rank must match input rank"
    comptime assert (
        type_of(residual_input).dtype == dtype
    ), "residual_input dtype must match the scalar dtype"
    comptime assert type_of(gamma).rank == 1, "gamma must be rank 1"
    comptime assert (
        type_of(gamma).dtype == dtype
    ), "gamma dtype must match the scalar dtype"
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape())
    )
