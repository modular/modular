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

"""Correctness test for the fused reduce-scatter + RMSNorm kernel (bf16).

Runs `reducescatter_rmsnorm` on TP4 shards over several M (incl. the M=6
non-divisible ragged edge) and gates each rank's `[rank_units, H]` outputs
against host references (full-H divisor, multiply-before-cast, bf16 last):

  * TIGHT — vs the bf16-rounded ref (sum rounded to bf16 pre-norm, as the kernel
    does): only mean-square associativity wobbles the norm. Gate: frac(ULP>1) <=
    1% and max_ulp <= 4.
  * LOOSE — vs the f32-sum ref. Gate: max_ulp <= 4. The ~24% 1-ULP mismatch is
    the expected f32-vs-bf16-sum gap (deliberate mid-way rounding); not gated.
  * `sum_out` — bit-identical to a standalone `reducescatter` shard on AMD
    non-multimem (tolerance on NVIDIA), given simd==8, f32 accum, and peer
    rotation `(my_rank+i)%ngpus == circular_add`.
"""

from std.sys import (
    has_amd_gpu_accelerator,
    simd_width_of,
    size_of,
)

from std.math import rsqrt
from std.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from std.utils.index import Index
from std.utils.numerics import get_accum_type
from std.testing import assert_true

from layout import Coord, TileTensor, row_major

from comm import Signal, MAX_GPUS, group_start, group_end
from comm.reducescatter_rmsnorm import (
    RS_NORM_FUSE_THRESHOLD,
    _dispatch_rs_norm,
    reducescatter_rmsnorm,
)
from comm.reducescatter import reducescatter, ReduceScatterConfig
from nn.normalization import rms_norm_gpu
from comm.sync import (
    circular_add,
    enable_p2p,
    init_signal_buffer,
    is_p2p_enabled,
)


def _run_case[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
    use_dispatch: Bool = False,
](
    num_rows: Int,
    list_of_ctx: List[DeviceContext],
    dispatch_threshold: Int = RS_NORM_FUSE_THRESHOLD,
) raises:
    """Run the fused kernel (or `_dispatch_rs_norm`) and gate both oracles.

    With `use_dispatch`, drive the auto dispatch with a caller `two_launch`
    closure and overridable `dispatch_threshold` (the collective-split-brain
    guard): at a non-divisible `num_rows` straddling the threshold, a
    rank-variant gate would split ranks across fused/`two_launch` on the same
    `rank_sigs` and DEADLOCK; the rank-invariant gate (`rank_units(0)`) makes all
    ranks agree. A regression hangs this case."""
    comptime simd_width = simd_width_of[in_dtype, target=get_gpu_target()]()

    # Preconditions for bit-identical `sum_out` on AMD non-multimem: peer
    # rotation `(my_rank+i)%ngpus == circular_add` and f32 accum (asserted so a
    # future change fails here, not as silent drift); simd==8 is AMD-only.
    comptime assert (
        get_accum_type[in_dtype]() == DType.float32
    ), "sum_out bit-identity assumes an f32 accumulator"
    comptime for _r in range(ngpus):
        comptime for _i in range(ngpus):
            comptime assert (
                circular_add[ngpus](_r, _i) == (_r + _i) % ngpus
            ), "kernel peer rotation must equal RS circular_add"
    comptime if has_amd_gpu_accelerator():
        comptime assert (
            simd_width == 8
        ), "sum_out bit-identity assumes simd=8 on AMD"

    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )
    var length = num_rows * num_cols
    var epsilon = Float32(1e-6)
    var weight_offset = Scalar[in_dtype](0.0)

    # Per-GPU inputs, per-device gamma, signals, and three output shards.
    var in_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var host_bufs = List[List[Scalar[in_dtype]]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_shard = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var rs_ref = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    # Shared gamma values (replicated per device so each rank reads locally).
    var gamma_host = List(length=num_cols, fill=Scalar[in_dtype](0))
    for c in range(num_cols):
        gamma_host[c] = (Float64(c + num_cols) / Float64(num_cols)).cast[
            in_dtype
        ]()

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](length))

        # Distinct positive per-GPU data: the peer sum reaches ~1010 (bf16
        # granule 4), so the mid-way bf16 store is genuinely lossy (the ~24%
        # loose headroom). Positive keeps bf16-ULP-via-bits clean.
        var h = List[Scalar[in_dtype]](length=length, fill=Scalar[in_dtype](0))
        for j in range(length):
            h[j] = Scalar[in_dtype](i + 1) + Scalar[in_dtype](j % 251)
        list_of_ctx[i].enqueue_copy(in_dev[i], h)
        host_bufs.append(h^)

        gamma_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](num_cols)
        )
        list_of_ctx[i].enqueue_copy(gamma_dev[i], gamma_host)

        # >= 1 row of storage so 0-row ranks (e.g. ranks 1-3 at M=1) aren't
        # zero-size; the true row count drives the launch and the readback.
        var alloc_i = config.rank_num_elements(i)
        if alloc_i < 1:
            alloc_i = 1
        normed.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))
        sum_shard.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i)
        )
        rs_ref.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[DType.uint8](size_of[Signal]())
        )
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Tensor views: full [rows, cols] inputs and [rank_units, cols] output
    # shards (true row count; the storage is >= 1 row).
    comptime InTensorType = TileTensor[
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        ImmutAnyOrigin,
    ]
    comptime OutShardType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]
    var in_bufs = InlineArray[InTensorType, ngpus](uninitialized=True)
    comptime for i in range(ngpus):
        in_bufs[i] = InTensorType(
            rebind[UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin]](
                in_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_rows, num_cols))),
        )

    # --- Run the fused kernel (or the auto dispatch) on every rank. ---
    group_start()
    for i in range(ngpus):
        var normed_view = OutShardType(
            normed[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
        var sum_view = OutShardType(
            sum_shard[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
        var gamma_view = GammaType(
            rebind[UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin]](
                gamma_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_cols))),
        )
        comptime if use_dispatch:
            # Production two-launch fallback (== the op's closure): standalone
            # reduce-scatter into `sum_view`, then `rms_norm_gpu` into
            # `normed_view`. Writing both outputs lets it hit the same oracles.
            @parameter
            @always_inline
            def two_launch() raises:
                reducescatter[dtype=in_dtype, ngpus=ngpus, axis=0](
                    in_bufs, sum_view, rank_sigs, list_of_ctx[i]
                )
                _rms_norm_shard[in_dtype, num_cols](
                    config.rank_units(i),
                    sum_shard[i],
                    normed[i],
                    gamma_dev[i],
                    epsilon,
                    weight_offset,
                    list_of_ctx[i],
                )

            _dispatch_rs_norm[two_launch=two_launch](
                in_bufs,
                normed_view,
                sum_view,
                gamma_view,
                epsilon,
                weight_offset,
                rank_sigs,
                list_of_ctx[i],
                threshold=dispatch_threshold,
            )
        else:
            reducescatter_rmsnorm(
                in_bufs,
                normed_view,
                sum_view,
                gamma_view,
                epsilon,
                weight_offset,
                rank_sigs,
                list_of_ctx[i],
            )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Standalone reduce-scatter into rs_ref, for the sum_out compare. ---
    # Re-init the signal buffers so this collective gets clean barrier state.
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    group_start()
    for i in range(ngpus):
        var rs_view = OutShardType(
            rs_ref[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
        reducescatter[dtype=in_dtype, ngpus=ngpus, axis=0](
            in_bufs, rs_view, rank_sigs, list_of_ctx[i]
        )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Host oracle + comparison over all owned rows of all ranks. ---
    var woff = weight_offset.cast[DType.float32]()
    var total_elems = 0
    var max_ulp_bf = 0  # fused vs bf16 ref (tight — kernel norms the bf16 sum)
    var gt1_ulp_bf = 0
    var max_ulp_f = 0  # fused vs f32 ref (loose — the wider f32-sum path)
    var mismatch_f = 0  # fused-vs-f32 exact-mismatch (headroom, reported)
    var sum_mismatch = 0  # sum_out vs standalone RS shard
    var sum_max_ulp = 0
    var accum = List[Scalar[DType.float32]](
        length=num_cols, fill=Scalar[DType.float32](0)
    )

    for i in range(ngpus):
        var local_rows = config.rank_units(i)
        if local_rows == 0:
            continue
        var start = config.rank_unit_start(i)
        var n = local_rows * num_cols

        var normed_h = List[Scalar[in_dtype]](
            length=n, fill=Scalar[in_dtype](0)
        )
        var sum_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        var rs_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        list_of_ctx[i].enqueue_copy(normed_h, normed[i])
        list_of_ctx[i].enqueue_copy(sum_h, sum_shard[i])
        list_of_ctx[i].enqueue_copy(rs_h, rs_ref[i])
        list_of_ctx[i].synchronize()
        total_elems += n

        for rr in range(local_rows):
            var grow = start + rr
            var base = grow * num_cols

            # Pass 1: f32 peer sum + both mean-square accumulations.
            var m2_bf = Float32(0)
            var m2_f = Float32(0)
            for c in range(num_cols):
                var s = Float32(0)
                for g in range(ngpus):
                    s += host_bufs[g][base + c].cast[DType.float32]()
                accum[c] = s
                var xb = s.cast[DType.bfloat16]().cast[DType.float32]()
                m2_bf += xb * xb
                m2_f += s * s

            var nf_bf = rsqrt(m2_bf / Float32(num_cols) + epsilon)
            var nf_f = rsqrt(m2_f / Float32(num_cols) + epsilon)

            # Pass 2: normalize, fold gamma in f32, cast bf16 last, compare.
            for c in range(num_cols):
                var s = accum[c]
                var xb = s.cast[DType.bfloat16]().cast[DType.float32]()
                var g_f = gamma_host[c].cast[DType.float32]() + woff
                var ref_bf16 = ((xb * nf_bf) * g_f).cast[DType.bfloat16]()
                var ref_f16 = ((s * nf_f) * g_f).cast[DType.bfloat16]()
                var gpu_normed = normed_h[rr * num_cols + c].cast[
                    DType.bfloat16
                ]()

                # Outputs positive, so the uint16 bit pattern is
                # magnitude-monotonic and |bits_a - bits_b| is the bf16 ULP.
                var gpu_bits = Int(gpu_normed.to_bits())
                var ulp_f = abs(gpu_bits - Int(ref_f16.to_bits()))
                var ulp_bf = abs(gpu_bits - Int(ref_bf16.to_bits()))
                if ulp_bf > max_ulp_bf:
                    max_ulp_bf = ulp_bf
                if ulp_bf > 1:
                    gt1_ulp_bf += 1
                if ulp_f > max_ulp_f:
                    max_ulp_f = ulp_f
                if (
                    gpu_normed.cast[DType.float32]()
                    != ref_f16.cast[DType.float32]()
                ):
                    mismatch_f += 1

                # sum_out vs standalone RS shard (bit-identical on AMD).
                var sum_bits = Int(
                    sum_h[rr * num_cols + c].cast[DType.bfloat16]().to_bits()
                )
                var rs_bits = Int(
                    rs_h[rr * num_cols + c].cast[DType.bfloat16]().to_bits()
                )
                var sum_ulp = abs(sum_bits - rs_bits)
                if sum_ulp != 0:
                    sum_mismatch += 1
                if sum_ulp > sum_max_ulp:
                    sum_max_ulp = sum_ulp

        _ = normed_h^
        _ = sum_h^
        _ = rs_h^

    var frac_gt1_bf = Float32(gt1_ulp_bf) / Float32(total_elems)
    var rate_f = Float32(mismatch_f) / Float32(total_elems)

    comptime mode_tag = "[dispatch straddle] " if use_dispatch else ""
    print(
        String(
            "  ",
            mode_tag,
            "M=",
            num_rows,
            ": TIGHT(fused vs bf16 ref) frac>1ULP=",
            frac_gt1_bf * 100.0,
            "% max_ulp=",
            max_ulp_bf,
            " | LOOSE(vs f32 ref) exact-mismatch=",
            rate_f * 100.0,
            "% (headroom) max_ulp=",
            max_ulp_f,
            " | sum_out mismatches=",
            sum_mismatch,
            " max_ulp=",
            sum_max_ulp,
        )
    )

    # Tight gate: kernel norms the bf16-rounded sum, matching the bf16 ref
    # within the block-reduce-vs-serial wobble (<=1 ULP); a wrong-eps /
    # wrong-divisor / gamma-after-cast bug is >> 4 ULP.
    if frac_gt1_bf > 0.01 or max_ulp_bf > 4:
        raise Error(
            String(
                "tight bf16-ref gate failed at M=",
                num_rows,
                ": frac>1ULP=",
                frac_gt1_bf * 100.0,
                "% max_ulp=",
                max_ulp_bf,
            )
        )
    # Loose gate: bounded bf16-ULP magnitude vs the f32-sum ref (~24% 1-ULP
    # divergence is the expected mid-way rounding).
    if max_ulp_f > 4:
        raise Error(
            String(
                "loose f32-ref gate failed at M=",
                num_rows,
                ": max_ulp=",
                max_ulp_f,
            )
        )
    # sum_out: bit-identical on AMD non-multimem; 1-ULP tolerance on NVIDIA
    # (RS may take the multimem path, undefined reduction order).
    comptime if has_amd_gpu_accelerator():
        if sum_mismatch != 0:
            raise Error(
                String(
                    "sum_out not bit-identical to standalone RS on AMD at M=",
                    num_rows,
                    ": mismatches=",
                    sum_mismatch,
                    " max_ulp=",
                    sum_max_ulp,
                )
            )
    else:
        if sum_max_ulp > 1:
            raise Error(
                String(
                    "sum_out exceeds 1-ULP tolerance vs standalone RS at M=",
                    num_rows,
                    ": max_ulp=",
                    sum_max_ulp,
                )
            )

    _ = in_dev^
    _ = host_bufs^
    _ = gamma_dev^
    _ = gamma_host^
    _ = normed^
    _ = sum_shard^
    _ = rs_ref^
    _ = signal_buffers^
    _ = accum^


def _rms_norm_shard[
    in_dtype: DType,
    num_cols: Int,
](
    rows: Int,
    src: DeviceBuffer[in_dtype],
    dst: DeviceBuffer[in_dtype],
    gamma: DeviceBuffer[in_dtype],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    ctx: DeviceContext,
) raises:
    """Standalone `rms_norm_gpu` on one `[rows, num_cols]` shard, M3-config
    (multiply_before_cast=True). This is the RMSNorm half of the production
    two-launch path (`reducescatter` -> `rms_norm_gpu`); reads `src`, writes
    `dst`."""
    comptime ShardType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    var src_view = ShardType(
        rebind[UnsafePointer[Scalar[in_dtype], MutAnyOrigin]](src.unsafe_ptr()),
        row_major(Coord(Index(rows, num_cols))),
    )
    var dst_view = ShardType(
        rebind[UnsafePointer[Scalar[in_dtype], MutAnyOrigin]](dst.unsafe_ptr()),
        row_major(Coord(Index(rows, num_cols))),
    )
    var gamma_view = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ](
        rebind[UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin]](
            gamma.unsafe_ptr()
        ),
        row_major(Coord(Index(num_cols))),
    )

    @always_inline
    @__copy_capture(src_view)
    @parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[in_dtype, width]:
        return src_view.raw_load[width=width](src_view.layout(coords))

    @always_inline
    @__copy_capture(dst_view)
    @parameter
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[in_dtype, width]) -> None:
        dst_view.raw_store[width=width, alignment=alignment](
            dst_view.layout(coords), val
        )

    rms_norm_gpu[2, input_fn, output_fn, multiply_before_cast=True](
        Coord(Index(rows, num_cols)), gamma_view, epsilon, weight_offset, ctx
    )


def _run_prod_oracle_case[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
    use_dispatch: Bool = False,
](num_rows: Int, list_of_ctx: List[DeviceContext]) raises -> Int:
    """Compare the fused `normed_out` to the ACTUAL M3 production norm; return the
    fused-vs-production exact-mismatch count.

    Production is the fused op's two-launch fallback: standalone `reducescatter`
    then the real `rms_norm_gpu` KERNEL at M3 config (weight_offset=1.0, mbc=True).
    Load-bearing because it runs the real kernel, not a host reduction, so a 1-ULP
    block-reduce-geometry gap between the fused `block.sum` and `rms_norm_gpu`
    shows up as a real bf16 mismatch; sweeping M locates the crossover M* below
    which fused == production.

    When `use_dispatch`, drive the auto dispatch (real `RS_NORM_FUSE_THRESHOLD`)
    with the op's `two_launch` closure: fused below M*, two-launch above, so
    `normed_out` must be bit-identical to production at EVERY M (the routing
    invariant, gated). This is the exact dispatch `distributed.mojo` uses.

    Also asserts `sum_out` == the standalone RS shard at every M (the
    residual-stream contract)."""
    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )
    var length = num_rows * num_cols
    var epsilon = Float32(1e-6)
    # M3 post-attention layernorm is Gemma-style: weight_offset=1.0, mbc=True.
    var weight_offset = Scalar[in_dtype](1.0)

    var in_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var host_bufs = List[List[Scalar[in_dtype]]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_shard = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var rs_ref = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var prod = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    var gamma_host = List(length=num_cols, fill=Scalar[in_dtype](0))
    for c in range(num_cols):
        gamma_host[c] = (Float64(c + num_cols) / Float64(num_cols)).cast[
            in_dtype
        ]()

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](length))
        var h = List[Scalar[in_dtype]](length=length, fill=Scalar[in_dtype](0))
        for j in range(length):
            h[j] = Scalar[in_dtype](i + 1) + Scalar[in_dtype](j % 251)
        list_of_ctx[i].enqueue_copy(in_dev[i], h)
        # Keep the host buffer alive until after the launch (the copy is async).
        host_bufs.append(h^)

        gamma_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](num_cols)
        )
        list_of_ctx[i].enqueue_copy(gamma_dev[i], gamma_host)

        var alloc_i = config.rank_num_elements(i)
        if alloc_i < 1:
            alloc_i = 1
        normed.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))
        sum_shard.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i)
        )
        rs_ref.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))
        prod.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[DType.uint8](size_of[Signal]())
        )
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    comptime InTensorType = TileTensor[
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        ImmutAnyOrigin,
    ]
    comptime OutShardType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]
    var in_bufs = InlineArray[InTensorType, ngpus](uninitialized=True)
    comptime for i in range(ngpus):
        in_bufs[i] = InTensorType(
            rebind[UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin]](
                in_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_rows, num_cols))),
        )

    # --- Fused kernel directly, or the op's dispatch (auto-route at the real
    # threshold) with the production `two_launch` fallback. ---
    group_start()
    for i in range(ngpus):
        var normed_view = OutShardType(
            normed[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
        var sum_view = OutShardType(
            sum_shard[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
        var gamma_view = GammaType(
            rebind[UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin]](
                gamma_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_cols))),
        )
        comptime if use_dispatch:
            # Mirror the graph op's fallback (distributed.mojo): standalone
            # reduce-scatter into `sum_view`, then `rms_norm_gpu` into
            # `normed_view`.
            @parameter
            @always_inline
            def two_launch() raises:
                reducescatter[dtype=in_dtype, ngpus=ngpus, axis=0](
                    in_bufs, sum_view, rank_sigs, list_of_ctx[i]
                )
                _rms_norm_shard[in_dtype, num_cols](
                    config.rank_units(i),
                    sum_shard[i],
                    normed[i],
                    gamma_dev[i],
                    epsilon,
                    weight_offset,
                    list_of_ctx[i],
                )

            _dispatch_rs_norm[two_launch=two_launch](
                in_bufs,
                normed_view,
                sum_view,
                gamma_view,
                epsilon,
                weight_offset,
                rank_sigs,
                list_of_ctx[i],
            )
        else:
            reducescatter_rmsnorm(
                in_bufs,
                normed_view,
                sum_view,
                gamma_view,
                epsilon,
                weight_offset,
                rank_sigs,
                list_of_ctx[i],
            )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Production two-launch: standalone reduce-scatter -> rms_norm_gpu. ---
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    group_start()
    for i in range(ngpus):
        var rs_view = OutShardType(
            rs_ref[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
        reducescatter[dtype=in_dtype, ngpus=ngpus, axis=0](
            in_bufs, rs_view, rank_sigs, list_of_ctx[i]
        )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    for i in range(ngpus):
        var local_rows = config.rank_units(i)
        if local_rows == 0:
            continue
        _rms_norm_shard[in_dtype, num_cols](
            local_rows,
            rs_ref[i],
            prod[i],
            gamma_dev[i],
            epsilon,
            weight_offset,
            list_of_ctx[i],
        )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Compare fused normed vs production normed (bit-for-bit). ---
    var total_elems = 0
    var normed_mismatch = 0  # fused vs production rms_norm_gpu output
    var normed_max_ulp = 0
    var sum_mismatch = 0  # fused sum_out vs standalone RS shard
    var sum_max_ulp = 0

    for i in range(ngpus):
        var local_rows = config.rank_units(i)
        if local_rows == 0:
            continue
        var n = local_rows * num_cols
        var normed_h = List[Scalar[in_dtype]](
            length=n, fill=Scalar[in_dtype](0)
        )
        var prod_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        var sum_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        var rs_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        list_of_ctx[i].enqueue_copy(normed_h, normed[i])
        list_of_ctx[i].enqueue_copy(prod_h, prod[i])
        list_of_ctx[i].enqueue_copy(sum_h, sum_shard[i])
        list_of_ctx[i].enqueue_copy(rs_h, rs_ref[i])
        list_of_ctx[i].synchronize()
        total_elems += n

        for e in range(n):
            var f_bits = Int(normed_h[e].cast[DType.bfloat16]().to_bits())
            var p_bits = Int(prod_h[e].cast[DType.bfloat16]().to_bits())
            if f_bits != p_bits:
                normed_mismatch += 1
            var ulp = abs(f_bits - p_bits)
            if ulp > normed_max_ulp:
                normed_max_ulp = ulp

            var s_bits = Int(sum_h[e].cast[DType.bfloat16]().to_bits())
            var r_bits = Int(rs_h[e].cast[DType.bfloat16]().to_bits())
            if s_bits != r_bits:
                sum_mismatch += 1
            var s_ulp = abs(s_bits - r_bits)
            if s_ulp > sum_max_ulp:
                sum_max_ulp = s_ulp

        _ = normed_h^
        _ = prod_h^
        _ = sum_h^
        _ = rs_h^

    var rate = Float32(normed_mismatch) / Float32(total_elems) * 100.0
    comptime mode_tag = "dispatched-op-vs" if use_dispatch else "fused-vs"
    print(
        String(
            "  M=",
            num_rows,
            " (rank0 units=",
            config.rank_units(0),
            "): ",
            mode_tag,
            "-PRODUCTION mismatch=",
            normed_mismatch,
            "/",
            total_elems,
            " (",
            rate,
            "%) max_ulp=",
            normed_max_ulp,
            " | sum_out mismatch=",
            sum_mismatch,
            (
                "  <-- BIT-IDENTICAL" if normed_mismatch
                == 0 else "  <-- diverges"
            ),
        )
    )

    # sum_out must be bit-identical to the standalone RS shard on AMD at every M
    # (residual stream is plain reduce-scatter on either branch).
    comptime if has_amd_gpu_accelerator():
        if sum_mismatch != 0:
            raise Error(
                String(
                    "sum_out not bit-identical to standalone RS at M=",
                    num_rows,
                    ": mismatches=",
                    sum_mismatch,
                    " max_ulp=",
                    sum_max_ulp,
                )
            )

    # Routing invariant: the dispatched op must be bit-identical to production at
    # EVERY M (fused below the threshold, two-launch above); a wrong sense would
    # fuse a diverging shape and fail here. AMD-scoped (gfx950-calibrated).
    comptime if use_dispatch and has_amd_gpu_accelerator():
        if normed_mismatch != 0:
            raise Error(
                String(
                    "dispatched op NOT bit-identical to production at M=",
                    num_rows,
                    ": mismatches=",
                    normed_mismatch,
                    " max_ulp=",
                    normed_max_ulp,
                    " (dispatch routed a diverging shape to the fused kernel)",
                )
            )

    _ = in_dev^
    _ = host_bufs^
    _ = gamma_dev^
    _ = gamma_host^
    _ = normed^
    _ = sum_shard^
    _ = rs_ref^
    _ = prod^
    _ = signal_buffers^

    return normed_mismatch


def _run_suite[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
]() raises:
    """Run the full correctness suite on `ngpus` GPUs (caller ensures >= ngpus
    are present and P2P is on). Driven at TP2 and TP4 from `main` so a 2-GPU CI
    lane exercises the kernel instead of skipping to a false green."""
    var list_of_ctx = List[DeviceContext]()
    for i in range(ngpus):
        list_of_ctx.append(DeviceContext(device_id=i))

    print("fused reduce-scatter + RMSNorm: TP", ngpus, "H=", num_cols)
    # Decode-shape M and a prefill tile (2048), plus the non-divisible ragged
    # edge `ngpus + ngpus//2` (rank 0 takes the remainder row: 2/2/1/1 at TP4,
    # 2/1 at TP2), exercising the extra-row / 0-row shard bookkeeping.
    for num_rows in [1, 8, 16, 32, 2048, ngpus + ngpus // 2]:
        _run_case[in_dtype, ngpus, num_cols](num_rows, list_of_ctx)

    # --- Production bit-identity sweep (fused vs standalone RS + rms_norm_gpu,
    # M3 config wo=1.0 mbc=True): locates the crossover M* that calibrates
    # RS_NORM_FUSE_THRESHOLD (fuse only where bit-identical to production). ---
    print(
        "\nfused-vs-production bit-identity sweep (wo=1.0, mbc=True, TP",
        ngpus,
        "H=",
        num_cols,
        "):",
    )
    var largest_bit_identical_m = 0
    var smallest_diverging_m = 0
    for num_rows in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        var mismatch = _run_prod_oracle_case[in_dtype, ngpus, num_cols](
            num_rows, list_of_ctx
        )
        if mismatch == 0:
            if num_rows > largest_bit_identical_m:
                largest_bit_identical_m = num_rows
        elif smallest_diverging_m == 0:
            smallest_diverging_m = num_rows

        # Calibration gate: any M the threshold would fuse MUST be bit-identical
        # to production. AMD-scoped (gfx950-calibrated); NVIDIA's `rms_norm_gpu`
        # has a different reduction geometry and may cross over elsewhere
        # (reported, not gated).
        comptime if has_amd_gpu_accelerator():
            var cfg = ReduceScatterConfig[in_dtype, ngpus](
                axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
            )
            var per_rank_bytes = (
                cfg.rank_units(0) * num_cols * size_of[in_dtype]()
            )
            if per_rank_bytes <= RS_NORM_FUSE_THRESHOLD and mismatch != 0:
                raise Error(
                    String(
                        "calibration FAILED: M=",
                        num_rows,
                        " (",
                        per_rank_bytes,
                        " B/rank) would fuse (threshold=",
                        RS_NORM_FUSE_THRESHOLD,
                        ") but is NOT bit-identical to production (mismatch=",
                        mismatch,
                        "). Lower RS_NORM_FUSE_THRESHOLD below this shape.",
                    )
                )

    print(
        "\ncrossover: largest bit-identical M =",
        largest_bit_identical_m,
        "; smallest diverging M =",
        smallest_diverging_m,
    )

    # Dispatch routing invariant: the auto dispatch (with the op's `two_launch`
    # fallback) must be bit-identical to production at EVERY M — fused below M*,
    # two-launch above. M spans the M=512 crossover.
    print("\ndispatched-op-vs-production (auto-route at real threshold):")
    for num_rows in [8, 512, 1024, 4096]:
        _ = _run_prod_oracle_case[in_dtype, ngpus, num_cols, use_dispatch=True](
            num_rows, list_of_ctx
        )

    # Collective-split-brain guard: a non-divisible M (`ngpus + ngpus//2`) with
    # a threshold (18432 B = 1.5 rows) straddling the 1-row and 2-row shards. A
    # rank-variant gate would split ranks across fused/two_launch on the same
    # rank_sigs and DEADLOCK; the rank-invariant gate (rank_units(0)=2 rows)
    # makes every rank pick two_launch. A regression hangs here.
    _run_case[in_dtype, ngpus, num_cols, use_dispatch=True](
        ngpus + ngpus // 2, list_of_ctx, dispatch_threshold=18432
    )

    print("TP", ngpus, "suite passed.")
    _ = list_of_ctx^


def main() raises:
    comptime in_dtype = DType.bfloat16
    comptime num_cols = 6144

    var num_devices = DeviceContext.number_of_devices()
    if num_devices < 2:
        print(
            "Need at least 2 GPUs but only found",
            num_devices,
            "- skipping.",
        )
        return

    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")
    if not is_p2p_enabled():
        print("P2P not enabled, skipping test.")
        return

    # TP2 is the common multi-GPU CI lane, so the suite runs below 4 GPUs; TP4
    # adds M3's real topology and H=6144 threshold calibration. The crossover is
    # per-rank-row driven, so the calibration gate holds at both.
    _run_suite[in_dtype, 2, num_cols]()
    if num_devices >= 4:
        _run_suite[in_dtype, 4, num_cols]()

    print("All fused reduce-scatter + RMSNorm tests passed!")
