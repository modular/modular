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


# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

from std.sys.info import simd_width_of
import extensibility as compiler

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from std.algorithm import max as reduce_max
from std.algorithm import mean
from std.algorithm import min as reduce_min
from std.algorithm import product, sum
from std.algorithm.reduction import _reduce_generator

from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.host.info import is_gpu
from nn import arg_nonzero
from nn.argmaxmin import argmax, argmin
from nn.argmaxmin_gpu import argmax_gpu, argmin_gpu
from nn.argsort import argsort
from nn.cumsum import cumsum
from nn.gather_scatter import _unsafe_normalize_neg_index, normalize_neg_index
from nn.normalization import (
    apply_qk_rms_norm,
    group_norm,
    layer_norm,
    rms_norm,
    rms_norm_fused_residual_add,
    rms_norm_rope_gpu,
    row_mean_of_squares,
    row_mean_of_squares_qk,
)
from nn.softmax import logsoftmax, softmax
from nn.topk import top_k, top_k_shape_impl
from extensibility import InputTensor, OutputTensor
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _FusedOutputTensor as FusedOutputTensor,
)
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList, StaticTuple
from std.utils.coord import Coord
from std.utils.index import Index

# ===-----------------------------------------------------------------------===#
from .kernels import *


@compiler.register("mo.reduce.arg_max")
struct ArgMax:
    @staticmethod
    def execute[
        target: StaticString,
        rank: Int,
        axis: Int,
        _trace_name: StaticString,
    ](
        output: OutputTensor[rank=rank, ...],
        input: InputTensor[rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        var axis_val = normalize_neg_index(axis, rank)

        comptime if target == "cpu":
            argmax(
                input.to_tile_tensor[DType.int64](),
                axis_val,
                output.to_tile_tensor[DType.int64](),
                Optional[DeviceContext](ctx),
            )
        else:
            if axis_val != rank - 1:
                raise Error("axis other than -1 not supported on GPU")

            # Has no static shape info

            # TODO(KERN-1045): Add support for taking advantage of static_shapes
            argmax_gpu(
                ctx,
                input.to_tile_tensor[DType.int64](),
                output.to_tile_tensor[DType.int64](),
            )


@compiler.register("mo.reduce.arg_min")
struct ArgMin:
    @staticmethod
    def execute[
        target: StaticString,
        rank: Int,
        axis: Int,
        _trace_name: StaticString,
    ](
        output: OutputTensor[rank=rank, ...],
        input: InputTensor[rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        var axis_val = normalize_neg_index(axis, rank)

        comptime if target == "cpu":
            argmin(
                input.to_tile_tensor[DType.int64](),
                axis_val,
                output.to_tile_tensor[DType.int64](),
                Optional[DeviceContext](ctx),
            )
        else:
            if axis_val != rank - 1:
                raise Error("axis other than -1 not supported on GPU")

            # TODO(KERN-1045): Add support for taking advantage of static_shapes
            argmin_gpu(
                ctx,
                input.to_tile_tensor[DType.int64](),
                output.to_tile_tensor[DType.int64](),
            )


@compiler.register("mo.arg_nonzero")
struct ArgNonZero:
    @staticmethod
    def execute(
        output_buffer: OutputTensor[rank=2, ...],
        input_buffer: InputTensor,
    ) raises:
        arg_nonzero.arg_nonzero(
            input_buffer.to_tile_tensor[DType.int64](),
            output_buffer.to_tile_tensor[DType.int64](),
        )


@compiler.register_shape_function("mo.arg_nonzero")
def arg_nonzero_shape(input_buffer: InputTensor) -> IndexList[2]:
    return arg_nonzero.arg_nonzero_shape(
        input_buffer.to_tile_tensor[DType.int64]()
    )


@compiler.register("mo.reduce.mean")
struct Mean:
    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        mean[
            output.dtype,
            input_fn,
            output_fn,
            target=target,
            reduce_dim=axis,
        ](Coord(input.shape()), Coord(output.shape()), ctx)


@compiler.register_shape_function("mo.reduce.mean")
def reduce_mean_shape[
    input_rank: Int,
    input_type: DType,
    axis: Int,
](
    input: InputTensor[dtype=input_type, rank=input_rank, ...],
) raises -> IndexList[input_rank]:
    return reduce_shape(input, axis)


@compiler.register("mo.reduce.row_mean_of_squares")
struct RowMeanOfSquares:
    """Per-row mean of squares over the last axis, accumulated in float32.

    For input `x` of shape `[M, N]` computes `out[m, 0] = sum_n(x[m,n]^2) / N`
    and writes a `[M, 1]` `output.dtype` result (typically float32). The square
    and accumulation always run in the input's accumulation type (float32 for
    bfloat16/float16/float32 inputs), independent of the output dtype.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=2, ...],
        input: FusedInputTensor[rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape()[0] != input.shape()[0] or output.shape()[1] != 1:
            raise Error("output must have shape [input_rows, 1]")

        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn(row: Int, val: Scalar[output.dtype]):
            output.store[width=1](Index(row, 0), val)

        row_mean_of_squares[input_fn, output_fn, target=target](
            input.shape(), ctx
        )


@compiler.register_shape_function("mo.reduce.row_mean_of_squares")
def reduce_row_mean_of_squares_shape(
    input: InputTensor[rank=2, ...],
) -> IndexList[2]:
    return Index(input.shape()[0], 1)


@compiler.register("mo.reduce.row_mean_of_squares_qk")
struct RowMeanOfSquaresQK:
    """Fused per-row mean of squares for two operands Q and K.

    For `q` of shape `[M, Nq]` and `k` of shape `[M, Nk]` (sharing rows but with
    possibly different column counts), computes `out[m, 0] = mean_n(q[m,n]^2)`
    and `out[m, 1] = mean_n(k[m,n]^2)` into a `[M, 2]` output. The square and
    accumulation always run in float32. This is a single-launch fusion of two
    `mo.reduce.row_mean_of_squares` ops plus a concat, used for cross-head
    QK-RMSNorm statistics under tensor parallelism.
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=2, ...],
        q: InputTensor[rank=2, ...],
        k: InputTensor[rank=2, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert q.dtype == k.dtype, "q and k must share a dtype"
        if (
            output.shape()[0] != q.shape()[0]
            or output.shape()[0] != k.shape()[0]
            or output.shape()[1] != 2
        ):
            raise Error("output must have shape [rows, 2] matching q/k rows")

        # `k` is bitcast to `q.dtype` to unify the single `in_dtype` kernel
        # parameter (q and k share a dtype, asserted above).
        row_mean_of_squares_qk[target=target](
            output.to_tile_tensor[DType.int64](),
            q.to_tile_tensor[DType.int64](),
            k.to_tile_tensor[DType.int64]().bitcast[q.dtype](),
            q.shape()[0],
            q.shape()[1],
            k.shape()[1],
            ctx,
        )


@compiler.register_shape_function("mo.reduce.row_mean_of_squares_qk")
def reduce_row_mean_of_squares_qk_shape(
    q: InputTensor[rank=2, ...],
    k: InputTensor[rank=2, ...],
) -> IndexList[2]:
    return Index(q.shape()[0], 2)


@compiler.register("mo.norm.apply_qk_rms_norm")
struct ApplyQKRMSNorm:
    """Fused per-element QK-RMSNorm apply for two operands Q and K.

    Given the already cross-rank-reduced per-row statistics `qk_var [M, 2]`
    (col 0 = mean(q^2), col 1 = mean(k^2), float32) and per-column float32
    scales `gamma_q [Nq]` / `gamma_k [Nk]`, applies in a single launch:

    `q_out[m,c] = cast((cast(q[m,c], f32) * rsqrt(qk_var[m,0] + eps)) * gamma_q[c], q.dtype)`
    and likewise for K with column 1. The grouping `((x * rs) * gamma)` then
    cast matches the unfused graph this replaces for bit-accuracy. This fuses
    the QK-RMSNorm apply chain (~7 tiny elementwise/View kernels) into one
    launch, used for cross-head QK-RMSNorm under tensor parallelism.

    Outputs (in order): `q_out [M, Nq]`, `k_out [M, Nk]` (both q/k dtype).
    Inputs (in order): `q [M, Nq]`, `k [M, Nk]` (activation dtype),
    `qk_var [M, 2]` (float32), `gamma_q [Nq]` (float32),
    `gamma_k [Nk]` (float32). Attribute: `epsilon` (float32 host scalar).
    """

    @staticmethod
    def execute[
        target: StaticString,
    ](
        q_out: OutputTensor[rank=2, ...],
        k_out: OutputTensor[rank=2, ...],
        q: InputTensor[rank=2, ...],
        k: InputTensor[rank=2, ...],
        qk_var: InputTensor[dtype=DType.float32, rank=2, ...],
        gamma_q: InputTensor[dtype=DType.float32, rank=1, ...],
        gamma_k: InputTensor[dtype=DType.float32, rank=1, ...],
        epsilon: Scalar[DType.float32],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert q.dtype == k.dtype, "q and k must share a dtype"
        comptime assert (
            q_out.dtype == q.dtype and k_out.dtype == q.dtype
        ), "outputs must match the q/k dtype"
        if q_out.shape()[0] != q.shape()[0] or q_out.shape()[1] != q.shape()[1]:
            raise Error("q_out must have shape [rows, Nq] matching q")
        if k_out.shape()[0] != k.shape()[0] or k_out.shape()[1] != k.shape()[1]:
            raise Error("k_out must have shape [rows, Nk] matching k")
        if (
            qk_var.shape()[0] != q.shape()[0]
            or qk_var.shape()[0] != k.shape()[0]
            or qk_var.shape()[1] != 2
        ):
            raise Error("qk_var must have shape [rows, 2] matching q/k rows")
        if gamma_q.shape()[0] != q.shape()[1]:
            raise Error("gamma_q must have shape [Nq] matching q cols")
        if gamma_k.shape()[0] != k.shape()[1]:
            raise Error("gamma_k must have shape [Nk] matching k cols")

        # `out_dtype` is inferred from `q_out`; `k_out` shares the same dtype
        # (asserted above), so bitcast its tile tensor to `q_out.dtype` to
        # unify the single `out_dtype` parameter. Likewise `in_dtype` is
        # inferred from `q`, so bitcast `k` to `q.dtype`.
        apply_qk_rms_norm[target=target,](
            q_out.to_tile_tensor[DType.int64](),
            k_out.to_tile_tensor[DType.int64]().bitcast[q_out.dtype](),
            gamma_q.to_tile_tensor[DType.int64](),
            gamma_k.to_tile_tensor[DType.int64](),
            qk_var.to_tile_tensor[DType.int64](),
            q.to_tile_tensor[DType.int64](),
            k.to_tile_tensor[DType.int64]().bitcast[q.dtype](),
            epsilon,
            q.shape()[0],
            q.shape()[1],
            k.shape()[1],
            ctx,
        )


@compiler.register("mo.reduce.add")
struct ReduceAdd:
    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        sum[
            output.dtype,
            input_fn,
            output_fn,
            target=target,
            reduce_dim=axis,
        ](Coord(input.shape()), ctx)


@compiler.register_shape_function("mo.reduce.add")
def reduce_add_shape[
    input_rank: Int,
    input_type: DType,
    axis: Int,
](
    input: InputTensor[dtype=input_type, rank=input_rank, ...],
) raises -> IndexList[input_rank]:
    return reduce_shape(input, axis)


@compiler.register("mo.reduce.mul")
struct ReduceMul:
    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        product[
            output.dtype,
            input_fn,
            output_fn,
            target=target,
            reduce_dim=axis,
        ](Coord(input.shape()), ctx)


@compiler.register_shape_function("mo.reduce.mul")
def reduce_mul_shape[
    input_rank: Int,
    input_type: DType,
    axis: Int,
](
    input: InputTensor[dtype=input_type, rank=input_rank, ...],
) raises -> IndexList[input_rank]:
    return reduce_shape(input, axis)


@compiler.register("mo.reduce.max")
struct ReduceMax:
    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        reduce_max[
            output.dtype,
            input_fn,
            output_fn,
            target=target,
            reduce_dim=axis,
        ](Coord(input.shape()), ctx)


@compiler.register_shape_function("mo.reduce.max")
def reduce_max_shape[
    input_rank: Int,
    input_type: DType,
    axis: Int,
](
    input: InputTensor[dtype=input_type, rank=input_rank, ...],
) raises -> IndexList[input_rank]:
    return reduce_shape(input, axis)


@compiler.register("mo.reduce.min")
struct ReduceMin:
    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        reduce_min[
            output.dtype,
            input_fn,
            output_fn,
            target=target,
            reduce_dim=axis,
        ](Coord(input.shape()), ctx)


@compiler.register_shape_function("mo.reduce.min")
def reduce_min_shape[
    input_rank: Int,
    input_type: DType,
    axis: Int,
](
    input: InputTensor[dtype=input_type, rank=input_rank, ...],
) raises -> IndexList[input_rank]:
    return reduce_shape(input, axis)


@compiler.register("mo.reduce.layer_norm")
struct LayerNorm:
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
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def gamma_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return gamma._lambda_load[width=width, element_alignment=alignment](
                rebind[IndexList[1]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            output._lambda_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        layer_norm[dtype, rank, input_fn, gamma_fn, output_fn, target=target](
            input.shape(),
            gamma.shape(),
            beta.to_tile_tensor[DType.int64](),
            epsilon,
            ctx,
        )


@compiler.register_shape_function("mo.reduce.layer_norm")
def reduce_layer_norm_shape[
    dtype: DType,
    rank: Int,
](
    input: InputTensor[dtype=dtype, rank=rank, ...],
    gamma: InputTensor[dtype=dtype, rank=1, ...],
    beta: InputTensor[dtype=dtype, rank=1, ...],
    epsilon: Scalar[dtype=dtype],
) -> IndexList[rank]:
    return input.shape()


@compiler.register("mo.reduce.rms_norm")
struct ReduceRMSNorm:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        # `IndexList` -> `Coord` boundary migration (mirror of softmax,
        # `Softmax.execute` above). The input fusion lambda takes a `Coord`
        # (the `_lambda_load` Coord overload erases to `IndexList` internally)
        # and the shape is passed via `input.shape_coord()`, which preserves
        # statically-known dims in the `Coord` type instead of erasing them to
        # an all-runtime `IndexList` as `input.shape()` would. `output_fn`
        # keeps its n-D `IndexList` form to match `rms_norm`'s `output_0_fn`.
        @parameter
        @always_inline
        def input_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
            return input._lambda_load[width=width, element_alignment=width](
                coords
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            output._lambda_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        rms_norm[
            dtype,
            rank,
            input_fn,
            output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            input.shape_coord(),
            gamma.to_tile_tensor[DType.int64](),
            epsilon,
            weight_offset,
            ctx,
        )


@compiler.register_shape_function("mo.reduce.rms_norm")
def reduce_rms_norm_shape[
    dtype: DType,
    rank: Int,
](
    input: InputTensor[dtype=dtype, rank=rank, ...],
    gamma: InputTensor[dtype=dtype, rank=1, ...],
    epsilon: Scalar[dtype=dtype],
    weight_offset: Scalar[dtype=dtype],
) -> IndexList[rank]:
    return input.shape()


@compiler.register("mo.composite.rms_norm_rope")
struct ReduceRMSNormRoPE:
    """Fuses RMS normalization and Rotary Position Embedding (RoPE) into one operation.

    Computes per-row RMS normalization scaled by `weight`, then applies RoPE to
    the normalized values using the provided cosine and sine tables.  The last
    dimension of the input must be an even number.
    """

    @staticmethod
    def execute[
        dtype: DType,
        output_dtype: DType,
        cos_sin_dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: FusedOutputTensor[dtype=output_dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        weight: InputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        cos_vals: FusedInputTensor[dtype=cos_sin_dtype, rank=rank, ...],
        sin_vals: FusedInputTensor[dtype=cos_sin_dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input._lambda_load[width=width, element_alignment=alignment](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def cos_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank]) -> SIMD[cos_sin_dtype, width]:
            return cos_vals._fused_load[
                width=width, element_alignment=alignment
            ](rebind[IndexList[cos_vals.rank]](coords))

        @parameter
        @always_inline
        def sin_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank]) -> SIMD[cos_sin_dtype, width]:
            return sin_vals._fused_load[
                width=width, element_alignment=alignment
            ](rebind[IndexList[sin_vals.rank]](coords))

        @parameter
        @always_inline
        def output_fn[
            width: Int, alignment: Int
        ](coords: IndexList[rank], val: SIMD[output_dtype, width]):
            output._lambda_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        rms_norm_rope_gpu[
            input_fn,
            cos_fn,
            sin_fn,
            output_fn,
            multiply_before_cast,
        ](
            input.shape(),
            weight.to_tile_tensor[DType.int64](),
            epsilon,
            weight_offset,
            cos_vals.to_tile_tensor[DType.int64](),
            sin_vals.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register_shape_function("mo.composite.rms_norm_rope")
def composite_rms_norm_rope_shape[
    dtype: DType,
    cos_sin_dtype: DType,
    rank: Int,
](
    input: InputTensor[dtype=dtype, rank=rank, ...],
    weight: InputTensor[dtype=dtype, rank=1, ...],
    epsilon: Scalar[dtype=dtype],
    weight_offset: Scalar[dtype=dtype],
    cos_vals: InputTensor[dtype=cos_sin_dtype, rank=rank, ...],
    sin_vals: InputTensor[dtype=cos_sin_dtype, rank=rank, ...],
) -> IndexList[rank]:
    return input.shape()


@compiler.register("mo.reduce.group_norm")
struct ReduceGroupNorm:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma: FusedInputTensor[dtype=dtype, rank=1, ...],
        beta: FusedInputTensor[dtype=dtype, rank=1, ...],
        epsilon: Scalar[dtype=dtype],
        num_groups: Int32,
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def gamma_fn[width: Int](coords: IndexList[1]) -> SIMD[dtype, width]:
            return gamma._lambda_load[width=width](coords)

        @parameter
        @always_inline
        def beta_fn[width: Int](coords: IndexList[1]) -> SIMD[dtype, width]:
            return beta._lambda_load[width=width](coords)

        group_norm[dtype, rank, input_fn, gamma_fn, beta_fn, target](
            shape=input.shape(),
            epsilon=epsilon,
            groups=num_groups,
            output=output.to_tile_tensor[DType.int64](),
            ctx=ctx,
        )


@compiler.register_shape_function("mo.reduce.group_norm")
def reduce_group_norm_shape[
    dtype: DType,
    rank: Int,
](
    input: InputTensor[dtype=dtype, rank=rank, ...],
    gamma: InputTensor[dtype=dtype, rank=1, ...],
    beta: InputTensor[dtype=dtype, rank=1, ...],
    epsilon: Scalar[dtype=dtype],
    num_groups: Int32,
) -> IndexList[rank]:
    return input.shape()


@compiler.register("mo.reduce.reduce_min_and_max")
struct ReduceMinAndMax:
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
        dtype: DType,
        rank: Int,
        axis: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        """Given a tensor of shape [A, B, C, D] and reducing along dimension 'C'
        writes to a tensor of shape [A, B, 2, D] where [:, :, 0, :] contains
        the minimum reduction and [:, :, 1, :] contains the maximum reduction.
        """

        comptime num_reductions = 2
        comptime norm_axis = axis + rank if axis < 0 else axis
        comptime assert (
            0 <= norm_axis < rank
        ), "axis must be between [0, <input rank>)"

        @parameter
        @always_inline
        def input_0_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_0_fn[
            width: SIMDSize, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._fused_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        @always_inline
        @parameter
        def input_0_fn_wrapper[
            _type: DType, width: Int, rank: Int
        ](idx: IndexList[rank]) -> SIMD[_type, width]:
            return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

        @always_inline
        @parameter
        def output_0_fn_wrapper[
            _type: DType,
            width: SIMDSize,
            rank: Int,
        ](
            indices: IndexList[rank],
            val: StaticTuple[SIMD[_type, width], num_reductions],
        ):
            # TODO: once we support multiple outputs, change this to route to
            # TODO: multiple output tensors.
            var indices_min = indices
            indices_min[norm_axis] = 0
            output_0_fn[width, rank](
                indices_min, rebind[SIMD[dtype, width]](val[0])
            )

            var indices_max = indices
            indices_max[norm_axis] = 1
            output_0_fn[width, rank](
                indices_max, rebind[SIMD[dtype, width]](val[1])
            )

        @always_inline
        @parameter
        def reduce_fn[
            ty: DType,
            width: SIMDSize,
            reduction_idx: Int,
        ](left: SIMD[ty, width], right: SIMD[ty, width]) -> SIMD[ty, width]:
            comptime assert reduction_idx < num_reductions, "reduction_idx OOB"

            comptime if reduction_idx == 0:
                return min(left, right)
            else:
                return max(left, right)

        var init_min = Scalar[dtype].MAX
        var init_max = Scalar[dtype].MIN
        var init = StaticTuple[Scalar[dtype], num_reductions](
            init_min, init_max
        )

        _reduce_generator[
            num_reductions,
            dtype,
            input_0_fn_wrapper,
            output_0_fn_wrapper,
            reduce_fn,
            target=target,
            reduce_dim=norm_axis,
        ](
            Coord(input.shape()),
            init=init,
            context=Optional[DeviceContext](ctx),
        )


@compiler.register_shape_function("mo.reduce.reduce_min_and_max")
def reduce_reduce_min_and_max_shape[
    axis: Int,
](input: InputTensor) -> IndexList[input.rank]:
    var new_shape = input.shape()
    new_shape[_unsafe_normalize_neg_index(axis, input.rank)] = 2

    return new_shape


@compiler.register("mo.composite.rms_norm_fused_residual_add")
struct ReduceRMSNormFusedResidualAdd:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        residual_output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        residual_input: FusedInputTensor[dtype=dtype, rank=rank, ...],
        gamma1: InputTensor[dtype=dtype, rank=1, ...],
        gamma2: InputTensor[dtype=dtype, rank=1, ...],
        epsilon1: Scalar[dtype=dtype],
        epsilon2: Scalar[dtype=dtype],
        weight_offset1: Scalar[dtype=dtype],
        weight_offset2: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        if input.shape() != residual_input.shape():
            raise Error("Input and residual input buffers are not same shape")

        @parameter
        @always_inline
        def input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input._lambda_load[width=width, element_alignment=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def residual_input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return residual_input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        def output_fn[
            width: SIMDSize, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            output._fused_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        @parameter
        @always_inline
        def residual_output_fn[
            width: SIMDSize, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            residual_output._fused_store[
                width=width, element_alignment=alignment
            ](
                rebind[IndexList[residual_output.rank]](coords),
                rebind[SIMD[residual_output.dtype, width]](val),
            )

        rms_norm_fused_residual_add[
            input_fn,
            residual_input_fn,
            output_fn,
            residual_output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            input.shape(),
            gamma1.to_tile_tensor[DType.int64](),
            epsilon1,
            weight_offset1,
            gamma2.to_tile_tensor[DType.int64](),
            epsilon2,
            weight_offset2,
            ctx,
        )


@compiler.register_shape_function("mo.composite.rms_norm_fused_residual_add")
def composite_rms_norm_fused_residual_add_shape[
    dtype: DType,
    rank: Int,
](
    input: InputTensor[dtype=dtype, rank=rank, ...],
    residual_input: InputTensor[dtype=dtype, rank=rank, ...],
    gamma1: InputTensor[dtype=dtype, rank=1, ...],
    gamma2: InputTensor[dtype=dtype, rank=1, ...],
    epsilon1: Scalar[dtype=dtype],
    epsilon2: Scalar[dtype=dtype],
    weight_offset1: Scalar[dtype=dtype],
    weight_offset2: Scalar[dtype=dtype],
) -> IndexList[rank]:
    return input.shape()


@compiler.register("mo.bottom_k")
struct BottomK:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        values: OutputTensor[dtype=dtype, rank=rank, ...],
        indices: OutputTensor[dtype=DType.int64, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[DType.bool],
        ctx: DeviceContext,
    ) raises:
        top_k[largest=False, target=target](
            input.to_tile_tensor[DType.int64](),
            Int(k),
            Int(axis),
            values.to_tile_tensor[DType.int64](),
            indices.to_tile_tensor[DType.int64](),
            sorted,
            ctx,
        )


@compiler.register_shape_function("mo.bottom_k")
def bottom_k_shape(
    input: InputTensor,
    k: Scalar,
    axis: Scalar,
    sorted: Scalar[DType.bool],
) raises -> IndexList[input.rank]:
    return rebind[IndexList[input.rank]](
        top_k_shape_impl(
            input.to_tile_tensor[DType.int64](),
            Int(k),
            Int(axis),
        )
    )


@compiler.register("mo.top_k")
struct TopK:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        values: OutputTensor[dtype=dtype, rank=rank, ...],
        indices: OutputTensor[dtype=DType.int64, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[DType.bool],
        ctx: DeviceContext,
    ) raises:
        top_k[largest=True, target=target](
            input.to_tile_tensor[DType.int64](),
            Int(k),
            Int(axis),
            values.to_tile_tensor[DType.int64](),
            indices.to_tile_tensor[DType.int64](),
            sorted,
            ctx,
        )


@compiler.register_shape_function("mo.top_k")
def top_k_shape(
    input: InputTensor,
    k: Scalar,
    axis: Scalar,
    sorted: Scalar[DType.bool],
) raises -> IndexList[input.rank]:
    return rebind[IndexList[input.rank]](
        top_k_shape_impl(
            input.to_tile_tensor[DType.int64](),
            Int(k),
            Int(axis),
        )
    )


@compiler.register("mo.reduce.softmax")
struct Softmax:
    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
        has_prologue_fusion: Bool,
    ](
        output: OutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # For adapting input fusion lambda required by call
        @parameter
        @always_inline
        def input_fn[width: Int](coords: Coord) -> SIMD[output.dtype, width]:
            return input._lambda_load[width=width](coords)

        comptime simd_width = simd_width_of[
            output.dtype, target=get_gpu_target()
        ]() if is_gpu[target]() else simd_width_of[output.dtype]()

        softmax[
            output.dtype,
            simd_width,
            output.rank,
            input_fn,
            target,
            has_prologue_fusion=has_prologue_fusion,
        ](
            Coord(output.shape()),
            output.to_tile_tensor[DType.int64](),
            axis,
            context=ctx,
        )


@compiler.register("mo.reduce.logsoftmax")
struct LogSoftmax:
    @staticmethod
    def execute[
        target: StaticString,
        axis: Int,
        has_prologue_fusion: Bool,
    ](
        output: OutputTensor,
        input: FusedInputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) capturing raises:
        # For adapting input fusion lambda required by call
        @parameter
        @always_inline
        def input_fn[width: Int](coords: Coord) -> SIMD[output.dtype, width]:
            return input._lambda_load[width=width](coords)

        logsoftmax[
            output.dtype,
            simd_width_of[output.dtype](),
            output.rank,
            input_fn,
            target,
            has_prologue_fusion=has_prologue_fusion,
        ](
            Coord(output.shape()),
            output.to_tile_tensor[DType.int64](),
            axis,
            context=ctx,
        )


@compiler.register("mo.cumsum")
struct CumSum:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        exclusive: Int,
        reverse: Int,
        axis: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        cumsum[dtype, Bool(exclusive), Bool(reverse), axis=axis](
            output.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
        )


@compiler.register("mx.argsort")
struct ArgSort[*, ascending: Bool]:
    @staticmethod
    def execute[
        target: StaticString
    ](
        indices: OutputTensor[rank=1, ...],
        input: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var indices_tensor = indices.to_tile_tensor[DType.int64]()
        var input_tensor = input.to_tile_tensor[DType.int64]()

        comptime if target == "cpu":
            argsort[ascending=Self.ascending](indices_tensor, input_tensor)
        else:
            argsort[ascending=Self.ascending, target=target](
                indices_tensor, input_tensor, ctx
            )
