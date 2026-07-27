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

"""Correctness test for the fused all-gather + RMSNorm kernel (bf16).

Runs `allgather_rmsnorm` on TP2/TP4 row-shards over several M and gates the full
replicated `[rows, H]` outputs against host references (full-H divisor,
multiply-before-cast, bf16 last):

  * NORM — vs a host RMSNorm on the gathered bf16 row. Gate: frac(ULP>1) <= 1%
    and max_ulp <= 4. (For all-gather the residual is a verbatim bf16 copy, not
    an f32 peer-sum, so the RS "tight vs bf16-ref / loose vs f32-ref" split
    collapses to one oracle: the norm input is unambiguously the gathered bf16
    value; only `block.sum`-vs-serial associativity wobbles the norm factor.)
  * RESIDUAL — `sum_out` bit-identical to a standalone `allgather` output on AMD
    (tolerance on NVIDIA). A pure gather/copy, so this is exact by construction.
  * REPLICATION — every GPU's `[rows, H]` outputs are bit-identical to GPU 0's
    (all-gather is replicated).
  * PRODUCTION crossover — fused `normed_out` vs the actual `allgather` +
    `rms_norm_gpu` KERNEL (wo=1.0, mbc=True) over the full replicated tensor;
    sweeping M locates the bit-identical crossover M* that calibrates
    `AG_NORM_FUSE_THRESHOLD`, and drives the dispatch routing-invariant check.
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
from comm.allgather_rmsnorm import (
    AG_NORM_FUSE_THRESHOLD,
    _dispatch_ag_norm,
    allgather_rmsnorm,
)
from comm.allgather import allgather
from comm.reducescatter import ReduceScatterConfig
from nn.normalization import rms_norm_gpu
from comm.sync import (
    enable_p2p,
    init_signal_buffer,
    is_p2p_enabled,
)


@always_inline
def _gathered_value[in_dtype: DType](row: Int, col: Int) -> Scalar[in_dtype]:
    """The bf16 value at global (row, col) of the gathered stream.

    Shared by the shard fill and the host oracle so they see identical bf16
    inputs. Positive (clean bf16-ULP-via-bits), varied per row and column, prime
    251 to avoid power-of-two aliasing.
    """
    return Scalar[in_dtype](1 + (row % 13)) + Scalar[in_dtype](col % 251)


def _rms_norm_full[
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
    """Standalone `rms_norm_gpu` over one full `[rows, num_cols]` tensor, M3
    config (multiply_before_cast=True). The RMSNorm half of the production
    two-launch path (`allgather` -> `rms_norm_gpu`); reads `src`, writes `dst`.
    M3's input_layernorm is `ShardingStrategy.replicate`, so the norm runs over
    all `rows` on every GPU (not a shard)."""
    comptime FullType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    var src_view = FullType(
        rebind[UnsafePointer[Scalar[in_dtype], MutAnyOrigin]](src.unsafe_ptr()),
        row_major(Coord(Index(rows, num_cols))),
    )
    var dst_view = FullType(
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


def _run_case[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
    use_dispatch: Bool = False,
](
    num_rows: Int,
    list_of_ctx: List[DeviceContext],
    dispatch_threshold: Int = AG_NORM_FUSE_THRESHOLD,
) raises:
    """Run the fused kernel (or `_dispatch_ag_norm`) and gate the norm oracle,
    the residual bit-identity, and cross-GPU replication.

    Shards are the `ReduceScatterConfig` ragged binning of `num_rows` (M3 gathers
    exactly what its reduce-scatter produced): shard `i` = `rank_units(i)` rows
    at global start `rank_unit_start(i)`, ranks past the remainder owning 0 rows
    (e.g. ranks 1-3 at M=1). Every GPU produces the full `[num_rows, H]`."""
    comptime simd_width = simd_width_of[in_dtype, target=get_gpu_target()]()
    # Crossover vs `rms_norm_gpu` was calibrated at simd==8 on AMD (block.sum
    # geometry depends on it); assert so a width change surfaces here, not
    # silently.
    comptime if has_amd_gpu_accelerator():
        comptime assert (
            simd_width == 8
        ), "fused-vs-production crossover assumes simd=8 on AMD"

    # Ragged shard layout: identical binning to a standalone reduce-scatter, so
    # the gather concatenation order matches a standalone all-gather.
    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )
    var epsilon = Float32(1e-6)
    var weight_offset = Scalar[in_dtype](0.0)
    var full_n = num_rows * num_cols

    var shard_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_full = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var ag_ref = List[DeviceBuffer[in_dtype]](capacity=ngpus)
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
        var shard_rows = config.rank_units(i)
        var shard_alloc = shard_rows * num_cols
        if shard_alloc < 1:
            shard_alloc = 1  # 0-row ranks still need a valid backing ptr.
        shard_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](shard_alloc)
        )

        # Fill shard i so its local row lr (global row start_i+lr) matches the
        # shared oracle value.
        var start_i = config.rank_unit_start(i)
        var h = List[Scalar[in_dtype]](
            length=shard_alloc, fill=Scalar[in_dtype](0)
        )
        for lr in range(shard_rows):
            var grow = start_i + lr
            for c in range(num_cols):
                h[lr * num_cols + c] = _gathered_value[in_dtype](grow, c)
        list_of_ctx[i].enqueue_copy(shard_dev[i], h)
        _ = h^

        gamma_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](num_cols)
        )
        list_of_ctx[i].enqueue_copy(gamma_dev[i], gamma_host)

        # Full replicated outputs on every GPU.
        normed.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
        sum_full.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
        ag_ref.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))

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

    # Peer-shard array (all `ngpus` shards, P2P-accessible) -- identical on every
    # GPU; full `[num_rows, num_cols]` output views.
    comptime ShardType = TileTensor[
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        ImmutAnyOrigin,
    ]
    comptime FullType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]
    var in_shards = InlineArray[ShardType, ngpus](uninitialized=True)
    comptime for i in range(ngpus):
        in_shards[i] = ShardType(
            rebind[UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin]](
                shard_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )

    # --- Run the fused kernel (or the auto dispatch) on every GPU. ---
    group_start()
    for i in range(ngpus):
        var normed_view = FullType(
            normed[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(num_rows, num_cols))),
        )
        var sum_view = FullType(
            sum_full[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(num_rows, num_cols))),
        )
        var gamma_view = GammaType(
            rebind[UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin]](
                gamma_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_cols))),
        )
        comptime if use_dispatch:
            # Production two-launch fallback (== the op's closure): all-gather
            # into the residual output `sum_full`, then `rms_norm_gpu` into
            # `normed`. `sum_out` must be the gathered stream on both branches
            # (op contract).
            @parameter
            @always_inline
            def two_launch() raises:
                _allgather_full[in_dtype, ngpus, num_cols](
                    in_shards, sum_full[i], config, rank_sigs, list_of_ctx[i], i
                )
                _rms_norm_full[in_dtype, num_cols](
                    num_rows,
                    sum_full[i],
                    normed[i],
                    gamma_dev[i],
                    epsilon,
                    weight_offset,
                    list_of_ctx[i],
                )

            _dispatch_ag_norm[two_launch=two_launch](
                in_shards,
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
            allgather_rmsnorm(
                in_shards,
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

    # --- Standalone all-gather into ag_ref, for the residual compare. ---
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()
    group_start()
    for i in range(ngpus):
        _allgather_full[in_dtype, ngpus, num_cols](
            in_shards, ag_ref[i], config, rank_sigs, list_of_ctx[i], i
        )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Host norm oracle + residual bit-identity on GPU 0. ---
    var woff = weight_offset.cast[DType.float32]()
    var total_elems = full_n
    var max_ulp = 0  # fused normed vs host ref
    var gt1_ulp = 0
    var sum_mismatch = 0  # fused sum_out vs standalone all-gather
    var sum_max_ulp = 0

    var normed_h0 = List[Scalar[in_dtype]](
        length=full_n, fill=Scalar[in_dtype](0)
    )
    var sum_h0 = List[Scalar[in_dtype]](length=full_n, fill=Scalar[in_dtype](0))
    var ag_h0 = List[Scalar[in_dtype]](length=full_n, fill=Scalar[in_dtype](0))
    list_of_ctx[0].enqueue_copy(normed_h0, normed[0])
    list_of_ctx[0].enqueue_copy(sum_h0, sum_full[0])
    list_of_ctx[0].enqueue_copy(ag_h0, ag_ref[0])
    list_of_ctx[0].synchronize()

    for r in range(num_rows):
        var base = r * num_cols
        # Pass 1: mean-square of the gathered bf16 row (in f32).
        var m2 = Float32(0)
        for c in range(num_cols):
            var x = _gathered_value[in_dtype](r, c).cast[DType.float32]()
            m2 += x * x
        var nf = rsqrt(m2 / Float32(num_cols) + epsilon)
        # Pass 2: normalize, fold gamma in f32, cast bf16 last, compare.
        for c in range(num_cols):
            var x = _gathered_value[in_dtype](r, c).cast[DType.float32]()
            var g_f = gamma_host[c].cast[DType.float32]() + woff
            var ref_v = ((x * nf) * g_f).cast[DType.bfloat16]()
            var gpu = normed_h0[base + c].cast[DType.bfloat16]()

            var ulp = abs(Int(gpu.to_bits()) - Int(ref_v.to_bits()))
            if ulp > max_ulp:
                max_ulp = ulp
            if ulp > 1:
                gt1_ulp += 1

            # Residual: fused sum_out vs standalone all-gather output.
            var s_ulp = abs(
                Int(sum_h0[base + c].to_bits()) - Int(ag_h0[base + c].to_bits())
            )
            if s_ulp != 0:
                sum_mismatch += 1
            if s_ulp > sum_max_ulp:
                sum_max_ulp = s_ulp
    _ = normed_h0^
    _ = sum_h0^
    _ = ag_h0^

    # --- Replication: every other GPU's outputs == GPU 0's (bit-for-bit). ---
    var repl_mismatch = 0
    if ngpus > 1:
        var n0 = List[Scalar[in_dtype]](length=full_n, fill=Scalar[in_dtype](0))
        var s0 = List[Scalar[in_dtype]](length=full_n, fill=Scalar[in_dtype](0))
        list_of_ctx[0].enqueue_copy(n0, normed[0])
        list_of_ctx[0].enqueue_copy(s0, sum_full[0])
        list_of_ctx[0].synchronize()
        for i in range(1, ngpus):
            var ni = List[Scalar[in_dtype]](
                length=full_n, fill=Scalar[in_dtype](0)
            )
            var si = List[Scalar[in_dtype]](
                length=full_n, fill=Scalar[in_dtype](0)
            )
            list_of_ctx[i].enqueue_copy(ni, normed[i])
            list_of_ctx[i].enqueue_copy(si, sum_full[i])
            list_of_ctx[i].synchronize()
            for e in range(full_n):
                if ni[e].to_bits() != n0[e].to_bits():
                    repl_mismatch += 1
                if si[e].to_bits() != s0[e].to_bits():
                    repl_mismatch += 1
            _ = ni^
            _ = si^
        _ = n0^
        _ = s0^

    var frac_gt1 = Float32(gt1_ulp) / Float32(total_elems)
    comptime mode_tag = "[dispatch straddle] " if use_dispatch else ""
    print(
        String(
            "  ",
            mode_tag,
            "M=",
            num_rows,
            ": NORM(vs host ref) frac>1ULP=",
            frac_gt1 * 100.0,
            "% max_ulp=",
            max_ulp,
            " | residual mismatches=",
            sum_mismatch,
            " max_ulp=",
            sum_max_ulp,
            " | replication mismatches=",
            repl_mismatch,
        )
    )

    if frac_gt1 > 0.01 or max_ulp > 4:
        raise Error(
            String(
                "norm gate failed at M=",
                num_rows,
                ": frac>1ULP=",
                frac_gt1 * 100.0,
                "% max_ulp=",
                max_ulp,
            )
        )
    # Residual: bit-identical on AMD; 1-ULP tolerance on NVIDIA (AG is a pure
    # copy either way, but leave headroom for a non-AMD path).
    comptime if has_amd_gpu_accelerator():
        if sum_mismatch != 0:
            raise Error(
                String(
                    (
                        "residual (sum_out) not bit-identical to standalone"
                        " all-gather on AMD at M="
                    ),
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
                    "residual exceeds 1-ULP tolerance at M=",
                    num_rows,
                    ": max_ulp=",
                    sum_max_ulp,
                )
            )
    if repl_mismatch != 0:
        raise Error(
            String(
                "outputs not replicated across GPUs at M=",
                num_rows,
                ": mismatches=",
                repl_mismatch,
            )
        )

    _ = shard_dev^
    _ = gamma_dev^
    _ = gamma_host^
    _ = normed^
    _ = sum_full^
    _ = ag_ref^
    _ = signal_buffers^


def _allgather_full[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
](
    in_shards: InlineArray[
        TileTensor[
            in_dtype,
            type_of(row_major(Coord(Index(0, num_cols)))),
            ImmutAnyOrigin,
        ],
        ngpus,
    ],
    out_full: DeviceBuffer[in_dtype],
    config: ReduceScatterConfig[in_dtype, ngpus],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    my_rank: Int,
) raises:
    """Standalone `allgather` of `in_shards` into the full `[rows, num_cols]`
    `out_full` on this GPU. Each source's rows land at their global offset (views
    into `out_full`), so `out_full` is the full gathered tensor -- the exact
    residual the fused kernel's `sum_out` must match."""
    comptime OutViewType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    var out_base = rebind[UnsafePointer[Scalar[in_dtype], MutAnyOrigin]](
        out_full.unsafe_ptr()
    )
    var out_views = InlineArray[OutViewType, ngpus](uninitialized=True)
    comptime for src in range(ngpus):
        var start = config.rank_unit_start(src)
        out_views[src] = OutViewType(
            out_base + start * num_cols,
            row_major(Coord(Index(config.rank_units(src), num_cols))),
        )
    allgather(in_shards, out_views, rank_sigs, ctx, my_rank)


def _run_prod_oracle_case[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
    use_dispatch: Bool = False,
](num_rows: Int, list_of_ctx: List[DeviceContext]) raises -> Int:
    """Compare the fused `normed_out` to the ACTUAL M3 production norm; return the
    fused-vs-production exact-mismatch count.

    Production is the fused op's two-launch fallback: standalone `allgather` then
    the real `rms_norm_gpu` KERNEL over the full replicated tensor at M3 config
    (weight_offset=1.0, mbc=True). Load-bearing because it runs the real kernel,
    so a 1-ULP block-reduce-geometry gap between the fused `block.sum` and
    `rms_norm_gpu` shows up as a real bf16 mismatch; sweeping M locates the
    crossover M* below which fused == production.

    With `use_dispatch`, drive the auto dispatch (real `AG_NORM_FUSE_THRESHOLD`)
    with the op's `two_launch` closure: fused below M*, two-launch above, so
    `normed_out` must be bit-identical to production at EVERY M (routing
    invariant, gated). Also asserts `sum_out` == the standalone all-gather at
    every M."""
    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )
    var epsilon = Float32(1e-6)
    # M3 input_layernorm is Gemma-style: weight_offset=1.0, mbc=True.
    var weight_offset = Scalar[in_dtype](1.0)
    var full_n = num_rows * num_cols

    var shard_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_full = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var ag_ref = List[DeviceBuffer[in_dtype]](capacity=ngpus)
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
        var shard_rows = config.rank_units(i)
        var shard_alloc = shard_rows * num_cols
        if shard_alloc < 1:
            shard_alloc = 1
        shard_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](shard_alloc)
        )
        var start_i = config.rank_unit_start(i)
        var h = List[Scalar[in_dtype]](
            length=shard_alloc, fill=Scalar[in_dtype](0)
        )
        for lr in range(shard_rows):
            var grow = start_i + lr
            for c in range(num_cols):
                h[lr * num_cols + c] = _gathered_value[in_dtype](grow, c)
        list_of_ctx[i].enqueue_copy(shard_dev[i], h)
        _ = h^

        gamma_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](num_cols)
        )
        list_of_ctx[i].enqueue_copy(gamma_dev[i], gamma_host)

        normed.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
        sum_full.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
        ag_ref.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
        prod.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))

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

    comptime ShardType = TileTensor[
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        ImmutAnyOrigin,
    ]
    comptime FullType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]
    var in_shards = InlineArray[ShardType, ngpus](uninitialized=True)
    comptime for i in range(ngpus):
        in_shards[i] = ShardType(
            rebind[UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin]](
                shard_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )

    # --- Fused kernel directly, or the op's dispatch with the production
    # two-launch fallback. ---
    group_start()
    for i in range(ngpus):
        var normed_view = FullType(
            normed[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(num_rows, num_cols))),
        )
        var sum_view = FullType(
            sum_full[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(num_rows, num_cols))),
        )
        var gamma_view = GammaType(
            rebind[UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin]](
                gamma_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_cols))),
        )
        comptime if use_dispatch:
            # Two-launch fallback writes the residual into `sum_full` (the op
            # contract), then norms it into `normed`.
            @parameter
            @always_inline
            def two_launch() raises:
                _allgather_full[in_dtype, ngpus, num_cols](
                    in_shards, sum_full[i], config, rank_sigs, list_of_ctx[i], i
                )
                _rms_norm_full[in_dtype, num_cols](
                    num_rows,
                    sum_full[i],
                    normed[i],
                    gamma_dev[i],
                    epsilon,
                    weight_offset,
                    list_of_ctx[i],
                )

            _dispatch_ag_norm[two_launch=two_launch](
                in_shards,
                normed_view,
                sum_view,
                gamma_view,
                epsilon,
                weight_offset,
                rank_sigs,
                list_of_ctx[i],
            )
        else:
            allgather_rmsnorm(
                in_shards,
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

    # --- Production two-launch on GPU 0: standalone all-gather -> rms_norm_gpu
    # over the full tensor. ---
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()
    group_start()
    for i in range(ngpus):
        _allgather_full[in_dtype, ngpus, num_cols](
            in_shards, ag_ref[i], config, rank_sigs, list_of_ctx[i], i
        )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()
    _rms_norm_full[in_dtype, num_cols](
        num_rows,
        ag_ref[0],
        prod[0],
        gamma_dev[0],
        epsilon,
        weight_offset,
        list_of_ctx[0],
    )
    list_of_ctx[0].synchronize()

    # --- Compare fused normed vs production normed on GPU 0 (bit-for-bit). ---
    var normed_h = List[Scalar[in_dtype]](
        length=full_n, fill=Scalar[in_dtype](0)
    )
    var prod_h = List[Scalar[in_dtype]](length=full_n, fill=Scalar[in_dtype](0))
    var sum_h = List[Scalar[in_dtype]](length=full_n, fill=Scalar[in_dtype](0))
    var ag_h = List[Scalar[in_dtype]](length=full_n, fill=Scalar[in_dtype](0))
    list_of_ctx[0].enqueue_copy(normed_h, normed[0])
    list_of_ctx[0].enqueue_copy(prod_h, prod[0])
    list_of_ctx[0].enqueue_copy(sum_h, sum_full[0])
    list_of_ctx[0].enqueue_copy(ag_h, ag_ref[0])
    list_of_ctx[0].synchronize()

    var normed_mismatch = 0
    var normed_max_ulp = 0
    var sum_mismatch = 0
    var sum_max_ulp = 0
    for e in range(full_n):
        var f_bits = Int(normed_h[e].to_bits())
        var p_bits = Int(prod_h[e].to_bits())
        if f_bits != p_bits:
            normed_mismatch += 1
        var ulp = abs(f_bits - p_bits)
        if ulp > normed_max_ulp:
            normed_max_ulp = ulp

        var s_ulp = abs(Int(sum_h[e].to_bits()) - Int(ag_h[e].to_bits()))
        if s_ulp != 0:
            sum_mismatch += 1
        if s_ulp > sum_max_ulp:
            sum_max_ulp = s_ulp
    _ = normed_h^
    _ = prod_h^
    _ = sum_h^
    _ = ag_h^

    var rate = Float32(normed_mismatch) / Float32(full_n) * 100.0
    comptime mode_tag = "dispatched-op-vs" if use_dispatch else "fused-vs"
    print(
        String(
            "  M=",
            num_rows,
            ": ",
            mode_tag,
            "-PRODUCTION mismatch=",
            normed_mismatch,
            "/",
            full_n,
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

    # sum_out must be bit-identical to a standalone all-gather at every M.
    comptime if has_amd_gpu_accelerator():
        if sum_mismatch != 0:
            raise Error(
                String(
                    "sum_out not bit-identical to standalone all-gather at M=",
                    num_rows,
                    ": mismatches=",
                    sum_mismatch,
                    " max_ulp=",
                    sum_max_ulp,
                )
            )

    # Routing invariant: the dispatched op must be bit-identical to production at
    # EVERY M (fused below the threshold, two-launch above). AMD-scoped.
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

    _ = shard_dev^
    _ = gamma_dev^
    _ = gamma_host^
    _ = normed^
    _ = sum_full^
    _ = ag_ref^
    _ = prod^
    _ = signal_buffers^

    return normed_mismatch


def _run_suite[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
]() raises:
    """Run the full correctness suite on `ngpus` GPUs (caller ensures >= ngpus
    are present and P2P is on). Driven at TP2 and TP4 from `main`."""
    var list_of_ctx = List[DeviceContext]()
    for i in range(ngpus):
        list_of_ctx.append(DeviceContext(device_id=i))

    print("fused all-gather + RMSNorm: TP", ngpus, "H=", num_cols)
    # Decode-shape M and a prefill tile (2048). M=1 exercises the ragged edge:
    # rank 0 owns the single row, ranks 1+ own 0 (empty shards).
    for num_rows in [1, 8, 16, 32, 2048]:
        _run_case[in_dtype, ngpus, num_cols](num_rows, list_of_ctx)

    # --- Production bit-identity sweep (fused vs allgather + rms_norm_gpu, M3
    # wo=1.0/mbc=True): locates the crossover M* that calibrates the fuse
    # threshold. ---
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
        # to production. AMD-scoped (gfx950-calibrated).
        comptime if has_amd_gpu_accelerator():
            var full_bytes = num_rows * num_cols * size_of[in_dtype]()
            if full_bytes <= AG_NORM_FUSE_THRESHOLD and mismatch != 0:
                raise Error(
                    String(
                        "calibration FAILED: M=",
                        num_rows,
                        " (",
                        full_bytes,
                        " B) would fuse (threshold=",
                        AG_NORM_FUSE_THRESHOLD,
                        ") but is NOT bit-identical to production (mismatch=",
                        mismatch,
                        "). Lower AG_NORM_FUSE_THRESHOLD below this shape.",
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
    # two-launch above. M spans the crossover.
    print("\ndispatched-op-vs-production (auto-route at real threshold):")
    for num_rows in [8, 512, 1024, 4096]:
        _ = _run_prod_oracle_case[in_dtype, ngpus, num_cols, use_dispatch=True](
            num_rows, list_of_ctx
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

    # TP2 is the common multi-GPU CI lane; TP4 adds M3's real topology and the
    # H=6144 threshold calibration.
    _run_suite[in_dtype, 2, num_cols]()
    if num_devices >= 4:
        _run_suite[in_dtype, 4, num_cols]()

    print("All fused all-gather + RMSNorm tests passed!")
