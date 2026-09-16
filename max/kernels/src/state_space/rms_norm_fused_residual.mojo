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
"""RMSNorm with fused residual connection for state space models."""

from std.math import align_down, align_up, ceildiv, rsqrt
from std.sys.info import align_of, simd_width_of, size_of

from std.algorithm import vectorize
from max.algorithm.functional import _get_start_indices_of_nth_subvolume
from max.gpu import (
    WARP_SIZE,
    block_dim,
    block_idx,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from max.gpu.host.info import is_gpu
from max.gpu.memory import external_memory
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from layout import TensorEngine, TensorLayout, TileTensor
from std.memory import AddressSpace
from std.random import Random

from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import IndexList
from std.utils.numerics import get_accum_type

from nn.normalization import _rms_norm_gpu_block_subkernel, _sum_to_mean


# ===----------------------------------------------------------------------=== #
# CPU Implementations
# ===----------------------------------------------------------------------=== #


def _rms_norm_fused_residual_cpu_2d[
    dtype: DType,
    InputFnType: ImplicitlyCopyable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    ResidualInputFnType: ImplicitlyCopyable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
    OutputResidualFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
    ResidualReadFnType: ImplicitlyCopyable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    //,
    multiply_before_cast: Bool = True,
](
    input_fn: InputFnType,
    residual_input_fn: ResidualInputFnType,
    output_fn: OutputFnType,
    output_residual_fn: OutputResidualFnType,
    residual_read_fn: ResidualReadFnType,
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    out_shape: IndexList[2],
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
):
    """Core 2D implementation of RMSNorm with fused residual.

    Uses simple (row, col) indexing to avoid compile-time evaluation issues.
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    var num_rows = out_shape[0]
    var num_cols = out_shape[1]

    comptime simd_width = simd_width_of[dtype]()
    var simd_loop_end = align_down(num_cols, simd_width)
    comptime intermediate_type = get_accum_type[dtype]()

    # Calculate dropout scale if needed
    var dropout_scale = Scalar[dtype](1.0)
    var zero_scalar = Scalar[dtype](0.0)
    if dropout_p > zero_scalar:
        var one_scalar = Scalar[dtype](1.0)
        dropout_scale = one_scalar / (one_scalar - dropout_p)

    for var row in range(num_rows):
        # First compute sum of squared (input + residual) for RMSNorm
        var sum_simd = SIMD[intermediate_type, simd_width]()

        # SIMD loop
        for col in range(0, simd_loop_end, simd_width):
            var input_vals = input_fn[simd_width](row, col)
            var residual_vals = residual_input_fn[simd_width](row, col)

            # Apply dropout if enabled
            if dropout_p > zero_scalar:
                comptime for i in range(simd_width):
                    var element_offset = row * num_cols + col + i
                    var generator = Random(
                        seed=seed, offset=UInt64(element_offset)
                    )
                    var rng = generator.step_uniform()
                    var rng_val = rng[0].cast[dtype]()
                    if rng_val >= dropout_p:
                        input_vals[i] = input_vals[i] * dropout_scale
                    else:
                        input_vals[i] = zero_scalar

            var sum_vals = input_vals + residual_vals

            # Output pre-normalized value (x + residual)
            output_residual_fn[simd_width, 1](row, col, sum_vals)

            # Accumulate for RMSNorm
            sum_simd += sum_vals.cast[intermediate_type]() ** 2

        # Scalar loop for remainder
        var sum_val = sum_simd.reduce_add()
        for col in range(simd_loop_end, num_cols):
            var input_val = input_fn[1](row, col)[0]
            var residual_val = residual_input_fn[1](row, col)[0]

            # Apply dropout if enabled
            if dropout_p > zero_scalar:
                var element_offset = row * num_cols + col
                var generator = Random(seed=seed, offset=UInt64(element_offset))
                var rng = generator.step_uniform()
                var rng_val = rng[0].cast[dtype]()
                if rng_val >= dropout_p:
                    input_val = input_val * dropout_scale
                else:
                    input_val = zero_scalar

            var sum_val_scalar = input_val + residual_val

            # Output pre-normalized value
            output_residual_fn[1, 1](row, col, sum_val_scalar)

            # Accumulate for RMSNorm
            sum_val += sum_val_scalar.cast[intermediate_type]() ** 2

        # Compute normalization factor
        var mean_val = _sum_to_mean(sum_val, num_cols)
        var norm_factor = rsqrt(mean_val + epsilon.cast[intermediate_type]())

        # Second pass: apply normalization
        def _normalize[
            sw: Int
        ](col: Int) {gamma, weight_offset, residual_read_fn, output_fn, mut}:
            # Read the pre-computed sum values (input + residual) from first pass
            var sum_vals = residual_read_fn[sw](row, col).cast[
                intermediate_type
            ]()

            var gamma_val = gamma.raw_load[width=sw](col)
            var norm_val: SIMD[dtype, sw]

            if multiply_before_cast:
                var gamma_offset = gamma_val + weight_offset
                norm_val = (sum_vals * norm_factor).cast[dtype]() * gamma_offset
            else:
                norm_val = (sum_vals * norm_factor).cast[dtype]() * (
                    gamma_val + weight_offset
                )

            output_fn[sw, 1](row, col, norm_val)

        vectorize[simd_width](num_cols, _normalize)


def rms_norm_fused_residual_cpu[
    dtype: DType,
    rank: Int,
    InputFnType: ImplicitlyCopyable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    ResidualInputFnType: ImplicitlyCopyable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) -> None,
    OutputResidualFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) -> None,
    ResidualReadFnType: ImplicitlyCopyable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    //,
    multiply_before_cast: Bool = True,
](
    input_fn: InputFnType,
    residual_input_fn: ResidualInputFnType,
    output_fn: OutputFnType,
    output_residual_fn: OutputResidualFnType,
    residual_read_fn: ResidualReadFnType,
    shape: IndexList[rank],
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    """Generic rank wrapper that delegates to the 2D core implementation.

    Creates 2D wrapper lambdas that translate (row, col) to IndexList[rank]
    at runtime, avoiding compile-time evaluation issues with _lambda_load.

    Parameters:
        dtype: Element data type of the tensors (inferred).
        rank: Tensor rank of `shape` (inferred).
        InputFnType: Type of the `input_fn` lambda (inferred).
        ResidualInputFnType: Type of the `residual_input_fn` lambda (inferred).
        OutputFnType: Type of the `output_fn` lambda (inferred).
        OutputResidualFnType: Type of the `output_residual_fn` lambda
            (inferred).
        ResidualReadFnType: Type of the `residual_read_fn` lambda (inferred).
        multiply_before_cast: When `True`, multiplies by `gamma` before
            casting to the output dtype (defaults to `True`).

    Args:
        input_fn: Lambda that loads the primary input `SIMD[dtype, width]`
            at a given `IndexList[rank]`.
        residual_input_fn: Lambda that loads the residual input at a given
            index.
        output_fn: Lambda that stores the normalized output at a given
            index.
        output_residual_fn: Lambda that stores the summed residual (input
            plus residual) before normalization.
        residual_read_fn: Lambda that re-reads the summed residual during
            the second normalization pass.
        shape: Shape of the tensors; the last dimension is normalized over.
        gamma: Scale (weight) vector with shape `(last_dim,)`.
        epsilon: Small constant added to the RMS denominator.
        weight_offset: Scalar added to each `gamma` element before scaling.
        dropout_p: Dropout probability applied to the primary input before
            the residual add (defaults to 0.0, which disables dropout).
        seed: RNG seed for dropout.
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim

    # Create 2D wrapper lambdas that translate indices at runtime
    @inline(.always)
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) {var shape, var input_fn} -> SIMD[dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width, rank](indices)

    @inline(.always)
    def residual_input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) {var shape, var residual_input_fn} -> SIMD[
        dtype, simd_width
    ]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return residual_input_fn[simd_width, rank](indices)

    @inline(.always)
    def output_fn_2d[
        simd_width: SIMDLength, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) {
        var shape, var output_fn
    } -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](indices, val)

    @inline(.always)
    def output_residual_fn_2d[
        simd_width: SIMDLength, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) {
        var shape, var output_residual_fn
    } -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_residual_fn[simd_width, alignment](indices, val)

    @inline(.always)
    def residual_read_fn_2d[
        sw: Int
    ](row: Int, col: Int) {var shape, var residual_read_fn} -> SIMD[dtype, sw]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return residual_read_fn[sw, rank](indices)

    # Call the 2D core implementation
    _rms_norm_fused_residual_cpu_2d[multiply_before_cast=multiply_before_cast](
        input_fn_2d,
        residual_input_fn_2d,
        output_fn_2d,
        output_residual_fn_2d,
        residual_read_fn_2d,
        gamma,
        epsilon,
        weight_offset,
        out_shape=IndexList[2](prod_all_but_last_dim, last_dim),
        dropout_p=dropout_p,
        seed=seed,
    )


def _rms_norm_fused_residual_cpu_entry[
    dtype: DType,
    rank: Int,
    InputFnType: ImplicitlyCopyable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    ResidualInputFnType: ImplicitlyCopyable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) -> None,
    OutputResidualFnType: ImplicitlyCopyable
    & def[width: SIMDLength, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) -> None,
    //,
    multiply_before_cast: Bool = True,
](
    input_fn: InputFnType,
    residual_input_fn: ResidualInputFnType,
    output_fn: OutputFnType,
    output_residual_fn: OutputResidualFnType,
    shape: IndexList[rank],
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    """CPU entry point that builds the residual read closure and dispatches.

    The op registration calls this for the CPU target, passing unified closures
    that capture the underlying tensors. The CPU kernel reads back the
    `input + residual` values it wrote in its first pass; since we have no
    direct handle to that buffer here, we recompute them from the input
    closures, matching the first pass exactly. Keeping this on the CPU path lets
    the whole chain use runtime closures end to end. The GPU path takes the
    same value-taking `FuncType` callbacks (see `_rms_norm_fused_residual_impl`).
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    # Note: we only support reduction along the last dimension
    if Int(gamma.dim[0]()) != shape[rank - 1]:
        raise Error(
            "Gamma size "
            + String(gamma.dim[0]())
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if shape.flattened_length() == 0:
        # Nothing to do.
        return

    @inline(.always)
    def residual_read_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) {
        var input_fn,
        var residual_input_fn,
        var dropout_p,
        var seed,
        var shape,
    } -> SIMD[dtype, width]:
        var input_vals = input_fn[width, _rank](coords)
        var residual_vals = residual_input_fn[width, _rank](coords)

        # Apply dropout if enabled (matching first pass exactly)
        var zero_scalar = Scalar[dtype](0.0)
        if dropout_p > zero_scalar:
            var one_scalar = Scalar[dtype](1.0)
            var dropout_scale = one_scalar / (one_scalar - dropout_p)
            var last_dim = shape[_rank - 1]
            var row = coords.flattened_length() // last_dim

            comptime for i in range(width):
                var col_idx = coords[_rank - 1] + i
                var element_offset = row * last_dim + col_idx
                var generator = Random(seed=seed, offset=UInt64(element_offset))
                var rng = generator.step_uniform()
                var rng_val = rng[0].cast[dtype]()
                if rng_val >= dropout_p:
                    input_vals[i] = input_vals[i] * dropout_scale
                else:
                    input_vals[i] = zero_scalar

        return input_vals + residual_vals

    rms_norm_fused_residual_cpu[multiply_before_cast=multiply_before_cast](
        input_fn,
        residual_input_fn,
        output_fn,
        output_residual_fn,
        residual_read_fn,
        shape,
        gamma,
        epsilon,
        weight_offset,
        dropout_p,
        seed,
    )


# ===----------------------------------------------------------------------=== #
# GPU Implementations
# ===----------------------------------------------------------------------=== #


# TODO(josh): Port this to the structured rowwise kernel API in
# `algorithm/gpu/rowwise.mojo`, which picks the dispatch tier from the
# shape and should be faster than this hand-rolled block-per-row kernel.
@fieldwise_init
struct _FusedResidualBlockKernel[
    gamma_mut: Bool,
    dtype: DType,
    GammaLayout: TensorLayout,
    gamma_origin: Origin[mut=gamma_mut],
    GammaEngine: TensorEngine,
    gamma_address_space: AddressSpace,
    gamma_linear_idx_type: DType,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    ResidualInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
    OutputResidualFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    multiply_before_cast: Bool,
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    """Block-per-row fused-residual RMSNorm kernel, holding its four callbacks
    as value fields.

    The callbacks have to cross the launch inside a struct rather than as
    `enqueue_function` arguments. An argument that conforms to `DevicePassable`
    is re-encoded field by field through each capture's `device_type`, and a
    callback here captures an output `ManagedTensorSlice`, whose `device_type`
    is `LayoutTensor` -- so the kernel would read its layout and fusion state
    back out of the wrong bytes. This struct is deliberately not
    `DevicePassable`, which selects the `enqueue_function` overload that
    bit-copies the payload verbatim. `rowwise`'s `_BlockKernel` crosses its own
    value closures the same way.
    """

    var gamma: TileTensor[
        Self.dtype,
        Self.GammaLayout,
        Self.gamma_origin,
        Engine=Self.GammaEngine,
        address_space=Self.gamma_address_space,
        linear_idx_type=Self.gamma_linear_idx_type,
    ]
    var epsilon: Float32
    var weight_offset: Float32
    var num_cols: Int32
    var dropout_p: Float32
    var seed: UInt64
    var input_fn: Self.InputFnType
    var residual_input_fn: Self.ResidualInputFnType
    var output_fn: Self.OutputFnType
    var output_residual_fn: Self.OutputResidualFnType

    @__name(
        t"rms_norm_fused_residual_gpu_block_{Self.dtype}_{Self.multiply_before_cast}",
    )
    def __call__(self) capturing:
        var _num_cols = Int(self.num_cols)
        var _weight_offset = Scalar[Self.dtype](self.weight_offset)
        var _dropout_p = Scalar[Self.dtype](self.dropout_p)

        var shared_mem = external_memory[
            Scalar[Self.dtype],
            address_space=.SHARED,
            alignment=align_of[SIMD[Self.dtype, Self.simd_width]](),
            name="intermediate_shared_memory",
        ]()
        with PDL():
            # First stage: apply dropout, add residual to input and store in
            # shared memory. Loop to handle cases where
            # `_num_cols > block_dim * simd_width`, matching the loop
            # structure in `_rms_norm_gpu_block_subkernel`.
            var tid = thread_idx.x
            var row = block_idx.x

            for x in range(ceildiv(_num_cols // Self.simd_width, block_dim.x)):
                var idx = (
                    x * block_dim.x * Self.simd_width + tid * Self.simd_width
                )

                if idx < _num_cols:
                    var input_val = self.input_fn[Self.simd_width](row, idx)

                    # Apply dropout if enabled
                    var zero_scalar = Scalar[Self.dtype](0.0)
                    if _dropout_p > zero_scalar:
                        var one_scalar = Scalar[Self.dtype](1.0)
                        var dropout_scale = one_scalar / (
                            one_scalar - _dropout_p
                        )

                        for i in range(Self.simd_width):
                            if idx + i < _num_cols:
                                var element_offset = (
                                    UInt64(row) * UInt64(_num_cols)
                                    + UInt64(idx)
                                    + UInt64(i)
                                )
                                var generator = Random(
                                    seed=self.seed, offset=element_offset
                                )
                                var rng = generator.step_uniform()
                                var rng_val = rng[0].cast[Self.dtype]()
                                if rng_val >= _dropout_p:
                                    input_val[i] = input_val[i] * dropout_scale
                                else:
                                    input_val[i] = zero_scalar

                    var residual_val = self.residual_input_fn[Self.simd_width](
                        row, idx
                    )
                    var residual_add_val = input_val + residual_val

                    self.output_residual_fn[
                        Self.simd_width,
                        align_of[SIMD[Self.dtype, Self.simd_width]](),
                    ](row, idx, residual_add_val)

                    shared_mem.store[
                        width=Self.simd_width,
                        alignment=align_of[SIMD[Self.dtype, Self.simd_width]](),
                    ](idx, residual_add_val)

            barrier()

            @inline(.always)
            def shared_mem_input_fn[
                width: Int
            ](row: Int, col: Int) {var shared_mem} -> SIMD[Self.dtype, width]:
                return shared_mem.load[width=width](col)

            _rms_norm_gpu_block_subkernel[
                Self.simd_width,
                Self.max_warps_per_block,
                Self.multiply_before_cast,
            ](
                shared_mem_input_fn,
                self.output_fn,
                self.gamma,
                self.epsilon,
                _weight_offset,
                _num_cols,
            )


def _enqueue_rms_norm_fused_residual_gpu_block[
    dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    multiply_before_cast: Bool,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    ResidualInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int](Int, Int) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
    OutputResidualFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) -> None,
](
    input_fn: InputFnType,
    residual_input_fn: ResidualInputFnType,
    output_fn: OutputFnType,
    output_residual_fn: OutputResidualFnType,
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Float32,
    num_cols: Int32,
    dropout_p: Float32,
    seed: UInt64,
    ctx: DeviceContext,
    launch_grid_dim: Int,
    launch_block_dim: Int,
    shared_mem_size: Int,
) raises:
    ctx.enqueue_function(
        _FusedResidualBlockKernel[
            simd_width=simd_width,
            max_warps_per_block=max_warps_per_block,
            multiply_before_cast=multiply_before_cast,
        ](
            gamma,
            epsilon,
            weight_offset,
            num_cols,
            dropout_p,
            seed,
            input_fn,
            residual_input_fn,
            output_fn,
            output_residual_fn,
        ),
        grid_dim=launch_grid_dim,
        block_dim=launch_block_dim,
        attributes=pdl_launch_attributes(PDLLevel.ON),
        shared_mem_bytes=shared_mem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(
                ctx.default_device_info.shared_memory_per_multiprocessor - 4096
            )
        ),
    )


def rms_norm_fused_residual_gpu[
    dtype: DType,
    rank: Int,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    ResidualInputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) -> None,
    OutputResidualFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) -> None,
    //,
    multiply_before_cast: Bool,
](
    input_fn: InputFnType,
    residual_input_fn: ResidualInputFnType,
    output_residual_fn: OutputResidualFnType,
    output_fn: OutputFnType,
    shape: IndexList[rank, ...],
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    """Dispatches the fused RMS-norm-plus-residual kernel on GPU.

    Selects the optimal SIMD width and warp count for `shape`, then enqueues
    `rms_norm_fused_residual_gpu_block` on `ctx`. Input, residual, normalized
    output, and residual output are all accessed through caller-supplied
    lambdas, enabling fusion with arbitrary prologue/epilogue patterns.

    Parameters:
        dtype: Element data type.
        rank: Tensor rank of `shape`.
        InputFnType: Type of the `input_fn` lambda (inferred).
        ResidualInputFnType: Type of the `residual_input_fn` lambda (inferred).
        OutputFnType: Type of the `output_fn` lambda (inferred).
        OutputResidualFnType: Type of the `output_residual_fn` lambda
            (inferred).
        multiply_before_cast: When `True`, multiplies by `gamma` before
            casting to the output dtype.

    Args:
        input_fn: Lambda that loads a `SIMD[dtype, width]` for a given index.
        residual_input_fn: Lambda that loads the residual value for a given index.
        output_residual_fn: Lambda that stores the summed residual value.
        output_fn: Lambda that stores the normalized output value.
        shape: Shape of the input tensor (the last dim is normalized over).
        gamma: Scale vector with shape `(last_dim,)`.
        epsilon: Small constant added to the RMS denominator.
        weight_offset: Scalar added to each `gamma` element before scaling.
        ctx: Device context for GPU execution.
        dropout_p: Dropout probability; 0.0 disables dropout.
        seed: RNG seed for dropout.

    Raises:
        If the GPU kernel launch fails.
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    if rank == 0:
        return

    var last_dim = shape[rank - 1]

    if last_dim == 0:
        return

    var rows = shape.flattened_length() // last_dim
    var cols = last_dim

    @inline(.always)
    def output_fn_2d[
        simd_width: SIMDLength, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) {
        var shape, var output_fn
    } -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](indices.canonicalize(), val)

    @inline(.always)
    def output_residual_fn_2d[
        simd_width: SIMDLength, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) {
        var shape, var output_residual_fn
    } -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_residual_fn[simd_width, alignment](indices.canonicalize(), val)

    @inline(.always)
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) {var shape, var input_fn} -> SIMD[dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width](indices.canonicalize())

    @inline(.always)
    def residual_input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) {var shape, var residual_input_fn} -> SIMD[
        dtype, simd_width
    ]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return residual_input_fn[simd_width](indices.canonicalize())

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    var grid_dim = rows
    var block_dim = min(
        align_up(ceildiv(cols, simd_width), WARP_SIZE),
        WARP_SIZE * max_warps_per_block,
    )

    var shared_mem_size = align_up(cols, simd_width) * size_of[dtype]()

    _enqueue_rms_norm_fused_residual_gpu_block[
        simd_width=simd_width,
        max_warps_per_block=max_warps_per_block,
        multiply_before_cast=multiply_before_cast,
    ](
        input_fn_2d,
        residual_input_fn_2d,
        output_fn_2d,
        output_residual_fn_2d,
        gamma,
        epsilon.cast[.float32](),
        weight_offset.cast[.float32](),
        Int32(cols),
        dropout_p.cast[.float32](),
        seed,
        ctx,
        grid_dim,
        block_dim,
        shared_mem_size,
    )


def _rms_norm_fused_residual_impl[
    dtype: DType,
    rank: Int,
    Input0FnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    Input1FnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) -> None,
    OutputResidualFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) -> None,
    //,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    input_0_fn: Input0FnType,
    input_1_fn: Input1FnType,
    output_fn: OutputFnType,
    output_residual_fn: OutputResidualFnType,
    shape: IndexList[rank],
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert is_gpu[target](), (
        "`_rms_norm_fused_residual_impl` only handles the GPU path; the CPU"
        " path is dispatched directly from the op registration via"
        " `_rms_norm_fused_residual_cpu_entry`, where the runtime closures can"
        " capture the underlying tensors."
    )

    # Note: we only support reduction along the last dimension
    if Int(gamma.dim[0]()) != shape[rank - 1]:
        raise Error(
            "Gamma size "
            + String(gamma.dim[0]())
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if shape.flattened_length() == 0:
        # Nothing to do.
        return

    rms_norm_fused_residual_gpu[multiply_before_cast=multiply_before_cast](
        input_0_fn,
        input_1_fn,
        output_residual_fn,
        output_fn,
        shape,
        gamma,
        epsilon,
        weight_offset,
        ctx,
        dropout_p,
        seed,
    )


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


@inline(.always)
def rms_norm_fused_residual[
    dtype: DType,
    rank: Int,
    Input0FnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    Input1FnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[dtype, width],
    Output0FnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) -> None,
    OutputResidualFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDLength, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) -> None,
    //,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    input_0_fn: Input0FnType,
    input_1_fn: Input1FnType,
    output_0_fn: Output0FnType,
    output_residual_fn: OutputResidualFnType,
    shape: IndexList[rank],
    gamma: TileTensor[dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
    dropout_p: Scalar[dtype] = Scalar[dtype](0.0),
    seed: UInt64 = 0,
) raises:
    """Applies fused residual add and RMS layer normalization.

    Computes `output = rms_norm(input + residual) * (gamma + weight_offset)`
    and writes the updated residual `input + residual` to `output_residual_fn`.
    Dispatches to a CPU or GPU implementation based on `target`.

    All tensor accesses go through caller-supplied lambdas, which lets the
    graph compiler fuse adjacent elementwise prologue/epilogue operations
    without materializing intermediate buffers.

    Parameters:
        dtype: Element data type.
        rank: Tensor rank of `shape`.
        Input0FnType: Type of the `input_0_fn` lambda (inferred).
        Input1FnType: Type of the `input_1_fn` lambda (inferred).
        Output0FnType: Type of the `output_0_fn` lambda (inferred).
        OutputResidualFnType: Type of the `output_residual_fn` lambda
            (inferred).
        target: Compilation target, e.g. `"cpu"` or `"gpu"`.
        multiply_before_cast: When `True`, multiplies by `gamma` before
            casting to the output dtype.

    Args:
        input_0_fn: Lambda that loads the primary input.
        input_1_fn: Lambda that loads the residual input.
        output_0_fn: Lambda that stores the normalized output.
        output_residual_fn: Lambda that stores the summed residual before
            normalization.
        shape: Shape of the tensors; the last dimension is normalized over.
        gamma: Scale (weight) vector with shape `(last_dim,)`.
        epsilon: Small constant added to the RMS denominator.
        weight_offset: Scalar added to each `gamma` element before scaling.
        ctx: Device context for GPU execution; unused on CPU.
        dropout_p: Dropout probability applied to the primary input before the
            residual add; 0.0 disables dropout.
        seed: RNG seed for dropout.

    Raises:
        If the GPU kernel launch fails.
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    @inline(.always)
    def description_fn() {imm} -> String:
        return trace_arg("input", shape, dtype)

    with Trace[TraceLevel.OP, target=target](
        "rms_norm_fused_residual",
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        _rms_norm_fused_residual_impl[
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            input_0_fn,
            input_1_fn,
            output_0_fn,
            output_residual_fn,
            shape,
            gamma,
            epsilon,
            weight_offset,
            ctx,
            dropout_p,
            seed,
        )
