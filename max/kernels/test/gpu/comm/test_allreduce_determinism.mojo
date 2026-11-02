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
#
# Run-to-run determinism test for the P2P `allreduce` API (DeepEP-style rerun +
# hash-of-tensor). Arch-portable: the `allreduce` entry point is the same on
# NVIDIA SM90 (H100) / SM100 (B200) and AMD CDNA4 (MI355); the size-keyed
# algorithm selection lives in the shared tuning table (`comm/allreduce.mojo`).
# KERN-3129 (an AMD split-K sync race) is the class of bug this guards against:
# a fixed-order reduction should produce bit-identical output every run.
#
# Method (mirrors DeepEP's 20x rerun-and-hash-compare, and the local idiom in
# `test_allreduce.mojo`):
#   1. Fixed, deterministic per-rank inputs (a closed-form function of
#      (rank, index) -- no RNG state to thread across ranks).
#   2. Run the same `allreduce` N times back-to-back (the realistic pattern:
#      consecutive collectives reuse the same signal buffers).
#   3. Per run, hash every output buffer (bit-reinterpret each element, fold
#      with an order-sensitive FNV-1a) and assert (a) all ranks agree and
#      (b) the hash is identical to run 0's. A divergence is a real race
#      finding (KERN-3129 class), not a tolerance to loosen.
#
# WHY NOT literal "XOR across ranks" (the DeepEP hash-of-tensor): DeepEP hashes
# an all-to-all where each rank holds DIFFERENT data, so XOR-across-tensors is
# non-degenerate. `allreduce` broadcasts the SAME reduced result to every rank,
# so all output buffers are bit-identical -- an XOR across an even number of
# ranks (ngpus in {2,4,8}) cancels to exactly 0 every run, and the run-to-run
# check would pass vacuously. We instead hash each rank independently
# (order-sensitive, non-cancelling), assert the ranks agree, and compare the
# per-run hash across reruns.
#
# HARDWARE: authored arch-portable; validated on B200. A single-GPU box can only
# run the host-side hash positive control (the `allreduce` API requires
# ngpus >= 2 at comptime); the multi-GPU determinism sweep needs a >= 2 GPU
# runner (`--config=remote-b200` / the nightly multi-GPU lane). Two ranks sum to
# the same bits in either order, so only the 4- and 8-rank cases can catch an
# add-order bug, and the 2-GPU presubmit executors skip them. The 8-GPU nightly
# lane does not reliably reach this target either: it runs every multi-GPU
# target in one 180-minute `bazel test` call, which has timed out on both B200
# and MI355. A change to an allreduce kernel's add order needs a manual run on
# an 8-GPU box.

from std.memory.alloc import unsafe_alloc
from std.sys import size_of

from comm import Signal, group_end, group_start
from comm.allreduce import (
    _allreduce_naive_single,
    allreduce,
    allreduce_tuning_table,
)
from comm.allreduce_lamport_rmsnorm import lamport_allreduce_rmsnorm
from comm.allreduce_residual_rmsnorm import (
    allreduce_residual_rmsnorm,
    allreduce_rmsnorm,
)
from comm.device_query import dispatch_select_comm_config
from comm.sync import enable_p2p, init_signal_buffer
from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from std.testing import assert_equal, assert_true
from std.utils.numerics import max_finite, nan

# DeepEP reruns 20x; reruns inside one process are cheap.
comptime NUM_RERUNS = 20

# 64-bit FNV-1a constants (a standard, order-sensitive rolling hash).
comptime FNV_OFFSET_BASIS = UInt64(0xCBF29CE484222325)
comptime FNV_PRIME = UInt64(0x100000001B3)

# Element counts chosen to straddle the size-keyed algorithm switch, which
# keys on bytes (length * size_of[dtype]). On B200 at 4 and 8 GPUs, fp32
# selects LAMPORT for small and mid, ONE_STAGE for one_stage, and TWO_STAGE for
# large; the sweep prints the selected algorithm per case so the regime
# coverage is visible in the log.
comptime LEN_SMALL = 16 * 1024
# Not a multiple of any SIMD width, so the 1-stage kernel's scalar tail runs.
# Lamport needs whole 128-bit packs, so on B200 this takes 1-stage even where
# the printed tuning-table algorithm is LAMPORT.
comptime LEN_TAIL = LEN_SMALL + 1
comptime LEN_MID = 256 * 1024
comptime LEN_ONE_STAGE = 512 * 1024
comptime LEN_LARGE = 2 * 1024 * 1024


def _elem_bits_u64[dtype: DType](x: Scalar[dtype]) -> UInt64:
    """Reinterprets one float element's bits as a `UInt64` (zero-extended)."""
    return x.to_bits().cast[.uint64]()


def hash_output[
    dtype: DType
](p: MutPointer[Scalar[dtype], MutUntrackedOrigin], length: Int) -> UInt64:
    """Order-sensitive FNV-1a over a host buffer's element bit patterns.

    Sensitive to any single-bit change and to element order, and (unlike an
    XOR-across-identical-tensors fold) never cancels to a constant.
    """
    var acc = FNV_OFFSET_BASIS
    for j in range(length):
        acc = (acc ^ _elem_bits_u64(p[j])) * FNV_PRIME
    return acc


def hash_positive_control() raises -> None:
    """Host-only positive control: the hash must detect a single-bit flip.

    Proves the hash mechanism has teeth without needing multiple GPUs, so it
    runs (and gives evidence) even on a single-GPU box.
    """
    comptime n = 4096
    var buf = unsafe_alloc[Float32](n)
    for j in range(n):
        buf[j] = Float32(j) * 0.5 - 1024.0

    var h0 = hash_output(buf, n)
    # Flip the lowest bit of one element through an integer view (a true
    # single-bit change of the stored float bits), re-hash, then restore.
    var ibuf = buf.unsafe_bitcast[UInt32]()
    var saved_bits = ibuf[n // 2]
    ibuf[n // 2] = saved_bits ^ 1
    var h1 = hash_output(buf, n)
    ibuf[n // 2] = saved_bits
    var h2 = hash_output(buf, n)

    assert_true(h0 != h1, "positive control FAILED: hash blind to a bit flip")
    assert_equal(h0, h2)  # restoring the bit restores the hash
    buf.unsafe_free()
    print("hash positive control PASS (single-bit flip detected)")


def _order_sensitive_term(rank: Int) -> Float64:
    """Returns rank's term of the sequence 2^24, 1, -2^24, 1, ....

    Each term is exact in bf16, but their f32 sum depends on the add order,
    whereas inputs from the inexact formula below can sum exactly in f32.
    """
    if rank % 4 == 0:
        return 16777216.0
    if rank % 4 == 2:
        return -16777216.0
    return 1.0


def allreduce_determinism_test[
    dtype: DType, ngpus: Int, naive: Bool = False
](list_of_ctx: List[DeviceContext], length: Int) raises -> None:
    """Runs `allreduce` `NUM_RERUNS` times on fixed inputs; asserts bit-stable.

    Parameters:
        dtype: Element type (bf16 / fp32).
        ngpus: Number of participating GPUs.
        naive: Whether to call the naive (no-P2P) allreduce directly. The
            sweep enables P2P, so `allreduce` never selects it on its own.

    Args:
        list_of_ctx: One `DeviceContext` per GPU.
        length: Element count of the input/output tensors.
    """
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"

    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_in = List[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )
    var host_out = List[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )

    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var temp_buffer_num_bytes = ngpus * size_of[dtype]() * length

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](length))
        out_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](length))

        var h_in = unsafe_alloc[Scalar[dtype]](length)
        host_in.append(h_in)
        host_out.append(unsafe_alloc[Scalar[dtype]](length))

        # Per-rank inexact scale: exactly representable inputs sum to the same
        # bits in any order, so the rank-agreement check would pass vacuously.
        var scale = 1.0 + 0.1234567 * Float64(i + 1)
        for j in range(length):
            var t = (i * 1315423911 + j * 2654435761) % 2039
            h_in[j] = Scalar[dtype](Float64(t - 1019) * 0.001953125 * scale)
        # The first element goes through the SIMD loop and the last through
        # the scalar tail when `length` is not a SIMD multiple.
        h_in[0] = Scalar[dtype](_order_sensitive_term(i))
        h_in[length - 1] = Scalar[dtype](_order_sensitive_term(i))

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + temp_buffer_num_bytes
            )
        )
        list_of_ctx[i].enqueue_copy(in_dev[i], host_in[i])

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )

    comptime InTensorType = TileTensor[
        dtype, type_of(row_major(length)), ImmutAnyOrigin
    ]
    var in_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTensorType: TileTensor(
            in_dev[i].unsafe_ptr(),
            row_major(length),
        )
    )

    comptime OutTensorType = TileTensor[
        dtype, type_of(row_major(length)), MutAnyOrigin
    ]
    var out_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) {
            mut out_dev, imm
        } -> OutTensorType: TileTensor(
            out_dev[i], row_major(length)
        ).as_unsafe_any_origin()
    )

    # One-time signal-buffer init (barrier counters + Lamport sentinel), then
    # sync so every rank is initialized before the first push.
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Report which algorithm regime this (ngpus, dtype, length) selects, so the
    # log documents that both the one-stage and two-stage regimes are covered.
    comptime sm_version = DeviceContext.default_device_info.version
    var num_bytes = length * size_of[dtype]()
    var cfg = dispatch_select_comm_config[
        ngpus, sm_version, allreduce_tuning_table
    ](num_bytes)
    print(
        (
            t"  ngpus={ngpus} dtype={dtype} length={length} bytes={num_bytes}"
            t" algo="
        ),
        end="",
    )
    comptime if naive:
        print("NAIVE")
    else:
        print(cfg.algorithm)

    var ref_hash = UInt64(0)
    for run in range(NUM_RERUNS):
        comptime if naive:
            comptime for i in range(ngpus):
                var out_tensor = out_tensors[i]

                @inline(.always)
                def store_out[
                    _dtype: DType,
                    _width: SIMDLength,
                    *,
                    alignment: Int,
                ](coords: Coord, val: SIMD[_dtype, _width]) {
                    var out_tensor
                } -> None:
                    out_tensor.store[width=_width, alignment=alignment](
                        coords, rebind[SIMD[dtype, _width]](val)
                    )

                _allreduce_naive_single[dtype=dtype, ngpus=ngpus](
                    in_tensors,
                    out_tensors[i].layout,
                    store_out,
                    216,
                    list_of_ctx[i],
                )
        else:
            group_start()
            comptime for i in range(ngpus):
                allreduce[ngpus=ngpus, use_multimem=False](
                    in_tensors, out_tensors[i], rank_sigs, list_of_ctx[i]
                )
            group_end()
        for i in range(ngpus):
            list_of_ctx[i].synchronize()
        for i in range(ngpus):
            list_of_ctx[i].enqueue_copy(host_out[i], out_dev[i])
        for i in range(ngpus):
            list_of_ctx[i].synchronize()

        # Hash rank 0, and require every other rank to agree bit-for-bit (a
        # correct allreduce broadcasts the same reduction to all ranks).
        var run_hash = hash_output(host_out[0], length)
        for i in range(1, ngpus):
            var rank_hash = hash_output(host_out[i], length)
            if rank_hash != run_hash:
                print(
                    t"CROSS-RANK DIVERGENCE at run {run} rank {i} hash"
                    t" {rank_hash} != rank0 hash {run_hash}"
                )
            assert_equal(rank_hash, run_hash)

        if run == 0:
            ref_hash = run_hash
        else:
            if run_hash != ref_hash:
                print(
                    t"RUN-TO-RUN DIVERGENCE at run {run} hash {run_hash} !="
                    t" run0 hash {ref_hash}"
                )
            assert_equal(run_hash, ref_hash)

    for i in range(ngpus):
        host_in[i].unsafe_free()
        host_out[i].unsafe_free()


def _ranks_agree[
    dtype: DType
](
    host_out: List[MutPointer[Scalar[dtype], MutUntrackedOrigin]],
    length: Int,
    what: StaticString,
    run: Int,
) -> Optional[UInt64]:
    """Returns rank 0's hash when every rank agrees, or `None` if one differs.

    The hash comes back rather than a bare verdict so a caller comparing runs
    reuses it instead of hashing the buffer a second time.
    """
    var h0 = hash_output(host_out[0], length)
    for i in range(1, len(host_out)):
        var rank_hash = hash_output(host_out[i], length)
        if rank_hash != h0:
            print(
                t"CROSS-RANK DIVERGENCE in {what} at run {run} rank {i} hash"
                t" {rank_hash} != rank0 hash {h0}"
            )
            return None
    return h0


def _bit_stable[
    dtype: DType
](
    host_out: List[MutPointer[Scalar[dtype], MutUntrackedOrigin]],
    length: Int,
    what: StaticString,
    run: Int,
    mut run0_hash: Optional[UInt64],
) -> Bool:
    """Checks one buffer across ranks and against its own run-0 hash.

    Run 0 records `run0_hash` once its ranks agree, and every later run must
    reproduce it bit for bit. A divergence names the buffer, the run and both
    hashes, then returns `False` so the caller can check the rest of the run's
    buffers before failing. A run-0 cross-rank failure leaves the reference
    unset, and every later run then returns `False` without printing, so a
    lost reference fails the test instead of quietly skipping the check.
    """
    var rank0_hash = _ranks_agree(host_out, length, what, run)
    if not rank0_hash:
        return False
    if run == 0:
        run0_hash = rank0_hash
        return True
    if not run0_hash:
        return False
    if rank0_hash.value() != run0_hash.value():
        print(
            t"RUN-TO-RUN DIVERGENCE in {what} at run {run} hash"
            t" {rank0_hash.value()} != run0 hash {run0_hash.value()}"
        )
        return False
    return True


def fused_rmsnorm_determinism_test[
    ngpus: Int, rows: Int, cols: Int
](list_of_ctx: List[DeviceContext]) raises -> None:
    """Checks the fused allreduce+RMSNorm kernels are bit-stable across runs.

    Covers `lamport_allreduce_rmsnorm` and `allreduce_rmsnorm` without
    quantization, asserting per buffer that every rank agrees and that each
    rerun reproduces run 0's hash. Every launch starts from a NaN-filled output
    buffer, so a dropped write diverges instead of replaying the previous run's
    bits. For the small shapes the sweep passes, the latter takes the fused
    Lamport kernel on B200 and the warp-tiling kernel on MI355. Uses fp32 with
    an order-sensitive first column in every row.

    Parameters:
        ngpus: Number of participating GPUs.
        rows: Rows (tokens) per call.
        cols: Columns (hidden size); RMSNorm runs over this axis.

    Args:
        list_of_ctx: One `DeviceContext` per GPU.
    """
    comptime dtype = DType.float32
    comptime length = rows * cols
    print(
        "  fused rmsnorm ngpus=", ngpus, " rows=", rows, " cols=", cols, sep=""
    )

    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var scales_dev = List[DeviceBuffer[.float32]](capacity=ngpus)
    var host_out = List[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )
    var sigs_lamport = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var sigs_fused = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var sig_bytes = size_of[Signal]() + ngpus * size_of[dtype]() * length

    var h_in = unsafe_alloc[Scalar[dtype]](length)
    var h_gamma = unsafe_alloc[Scalar[dtype]](cols)
    for c in range(cols):
        h_gamma[c] = Scalar[dtype](0.75 + Float64(c % 97) * 0.00390625)
    for i in range(ngpus):
        var ctx = list_of_ctx[i]
        in_dev.append(ctx.enqueue_create_buffer[dtype](length))
        out_dev.append(ctx.enqueue_create_buffer[dtype](length))
        gamma_dev.append(ctx.enqueue_create_buffer[dtype](cols))
        scales_dev.append(ctx.enqueue_create_buffer[.float32](rows))
        host_out.append(unsafe_alloc[Scalar[dtype]](length))
        var scale = 1.0 + 0.1234567 * Float64(i + 1)
        for j in range(length):
            var t = (i * 1315423911 + j * 2654435761) % 2039
            h_in[j] = Scalar[dtype](Float64(t - 1019) * 0.001953125 * scale)
        for r in range(rows):
            h_in[r * cols] = Scalar[dtype](_order_sensitive_term(i))
        ctx.enqueue_copy(in_dev[i], h_in)
        ctx.enqueue_copy(gamma_dev[i], h_gamma)
        sigs_lamport.append(ctx.create_buffer_sync[.uint8](sig_bytes))
        sigs_fused.append(ctx.create_buffer_sync[.uint8](sig_bytes))
        init_signal_buffer(sigs_lamport[i], ctx)
        init_signal_buffer(sigs_fused[i], ctx)
        ctx.synchronize()

    var rank_sigs_lamport = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(sigs_lamport[i])
    )
    var rank_sigs_fused = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(sigs_fused[i])
    )
    comptime in_layout = row_major(Idx[rows], Idx[cols])
    comptime InTile = TileTensor[dtype, type_of(in_layout), ImmutAnyOrigin]
    var in_tiles = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTile: TileTensor(
            in_dev[i], in_layout
        ).as_imm()
    )

    # Run both kernels before failing, so one run reports every divergence.
    var lamport_ok = True
    var fused_ok = True
    var lamport_run0: Optional[UInt64] = None
    var fused_run0: Optional[UInt64] = None
    for run in range(NUM_RERUNS):
        # Poison the output before every launch below: a write a run drops would
        # otherwise keep the previous run's bits, which equal run 0's, and the
        # two kernels share `out_dev`, so one would fill the other's gaps.
        for i in range(ngpus):
            list_of_ctx[i].enqueue_memset(out_dev[i], nan[dtype]())
        group_start()
        comptime for i in range(ngpus):
            lamport_allreduce_rmsnorm[dtype, ngpus](
                i,
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    in_dev[i].unsafe_ptr()
                ),
                rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                    out_dev[i].unsafe_ptr()
                ),
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    gamma_dev[i].unsafe_ptr()
                ),
                rank_sigs_lamport,
                rows,
                cols,
                Scalar[dtype](1e-6),
                list_of_ctx[i],
            )
        group_end()
        for i in range(ngpus):
            list_of_ctx[i].enqueue_copy(host_out[i], out_dev[i])
        for i in range(ngpus):
            list_of_ctx[i].synchronize()
        var lamport_run_ok = _bit_stable(
            host_out, length, "lamport_allreduce_rmsnorm", run, lamport_run0
        )
        lamport_ok = lamport_ok and lamport_run_ok

        for i in range(ngpus):
            list_of_ctx[i].enqueue_memset(out_dev[i], nan[dtype]())
        group_start()
        comptime for i in range(ngpus):
            allreduce_rmsnorm(
                in_tiles,
                TileTensor(out_dev[i], in_layout),
                TileTensor(gamma_dev[i], row_major(Idx[cols])),
                Float32(1e-6),
                Scalar[dtype](0),
                Float32(1),
                TileTensor(scales_dev[i], row_major(Idx[rows], Idx[1])),
                rank_sigs_fused,
                list_of_ctx[i],
            )
        group_end()
        for i in range(ngpus):
            list_of_ctx[i].enqueue_copy(host_out[i], out_dev[i])
        for i in range(ngpus):
            list_of_ctx[i].synchronize()
        var fused_run_ok = _bit_stable(
            host_out, length, "allreduce_rmsnorm", run, fused_run0
        )
        fused_ok = fused_ok and fused_run_ok

    assert_true(
        lamport_ok,
        "lamport_allreduce_rmsnorm output differs across ranks or runs",
    )
    assert_true(
        fused_ok, "allreduce_rmsnorm output differs across ranks or runs"
    )

    h_in.unsafe_free()
    h_gamma.unsafe_free()
    for i in range(ngpus):
        host_out[i].unsafe_free()


def fused_fp8_residual_determinism_test[
    ngpus: Int, rows: Int, cols: Int
](list_of_ctx: List[DeviceContext]) raises -> None:
    """Checks `allreduce_residual_rmsnorm` FP8 output is bit-stable across runs.

    Small per-rank payloads take the fused warp-tiling kernel. Payloads past
    the split threshold run the plain `allreduce` with an add epilogue, then a
    separate RMSNorm + FP8 kernel, so the add order comes from the plain
    allreduce. Asserts per buffer that the residual output, the FP8 output and
    the per-row scales agree across ranks and reproduce run 0's hash. Every
    launch starts from NaN-filled outputs, so a dropped write diverges instead
    of replaying the previous run's bits.

    Parameters:
        ngpus: Number of participating GPUs.
        rows: Rows (tokens) per call.
        cols: Columns (hidden size); RMSNorm runs over this axis.

    Args:
        list_of_ctx: One `DeviceContext` per GPU.
    """
    comptime in_dtype = DType.bfloat16
    comptime out_dtype = DType.float8_e4m3fn
    comptime length = rows * cols
    print(t"  fused fp8 residual ngpus={ngpus} rows={rows} cols={cols}")

    var in_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var residual_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var residual_out_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[out_dtype]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var scales_dev = List[DeviceBuffer[.float32]](capacity=ngpus)
    var host_residual_out = List[
        MutPointer[Scalar[in_dtype], MutUntrackedOrigin]
    ](capacity=ngpus)
    var host_out = List[MutPointer[Scalar[out_dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )
    var host_scales = List[MutPointer[Scalar[.float32], MutUntrackedOrigin]](
        capacity=ngpus
    )
    var sigs = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var sig_bytes = size_of[Signal]() + ngpus * size_of[in_dtype]() * length

    var h_in = unsafe_alloc[Scalar[in_dtype]](length)
    var h_residual = unsafe_alloc[Scalar[in_dtype]](length)
    var h_gamma = unsafe_alloc[Scalar[in_dtype]](cols)
    for j in range(length):
        h_residual[j] = Scalar[in_dtype](Float64(j % 61 - 30) * 0.0078125)
    for c in range(cols):
        h_gamma[c] = Scalar[in_dtype](0.75 + Float64(c % 97) * 0.00390625)
    for i in range(ngpus):
        var ctx = list_of_ctx[i]
        in_dev.append(ctx.enqueue_create_buffer[in_dtype](length))
        residual_dev.append(ctx.enqueue_create_buffer[in_dtype](length))
        residual_out_dev.append(ctx.enqueue_create_buffer[in_dtype](length))
        out_dev.append(ctx.enqueue_create_buffer[out_dtype](length))
        gamma_dev.append(ctx.enqueue_create_buffer[in_dtype](cols))
        scales_dev.append(ctx.enqueue_create_buffer[.float32](rows))
        host_residual_out.append(unsafe_alloc[Scalar[in_dtype]](length))
        host_out.append(unsafe_alloc[Scalar[out_dtype]](length))
        host_scales.append(unsafe_alloc[Scalar[.float32]](rows))
        var scale = 1.0 + 0.1234567 * Float64(i + 1)
        for j in range(length):
            var t = (i * 1315423911 + j * 2654435761) % 2039
            h_in[j] = Scalar[in_dtype](Float64(t - 1019) * 0.001953125 * scale)
        for r in range(rows):
            h_in[r * cols] = Scalar[in_dtype](_order_sensitive_term(i))
        ctx.enqueue_copy(in_dev[i], h_in)
        ctx.enqueue_copy(residual_dev[i], h_residual)
        ctx.enqueue_copy(gamma_dev[i], h_gamma)
        sigs.append(ctx.create_buffer_sync[.uint8](sig_bytes))
        init_signal_buffer(sigs[i], ctx)
        ctx.synchronize()

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(sigs[i])
    )
    comptime in_layout = row_major(Idx[rows], Idx[cols])
    comptime InTile = TileTensor[in_dtype, type_of(in_layout), ImmutAnyOrigin]
    var in_tiles = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTile: TileTensor(
            in_dev[i], in_layout
        ).as_imm()
    )

    var ok = True
    var residual_run0: Optional[UInt64] = None
    var out_run0: Optional[UInt64] = None
    var scales_run0: Optional[UInt64] = None
    for run in range(NUM_RERUNS):
        # NaN is outside every output's range, so a write this run drops can't
        # masquerade as the previous run's identical bits.
        for i in range(ngpus):
            list_of_ctx[i].enqueue_memset(residual_out_dev[i], nan[in_dtype]())
            list_of_ctx[i].enqueue_memset(out_dev[i], nan[out_dtype]())
            list_of_ctx[i].enqueue_memset(scales_dev[i], nan[.float32]())
        group_start()
        comptime for i in range(ngpus):
            allreduce_residual_rmsnorm(
                in_tiles,
                TileTensor(residual_dev[i], in_layout).as_imm(),
                TileTensor(out_dev[i], in_layout),
                TileTensor(residual_out_dev[i], in_layout),
                TileTensor(gamma_dev[i], row_major(Idx[cols])),
                Float32(1e-6),
                Scalar[in_dtype](0),
                max_finite[out_dtype]().cast[.float32](),
                TileTensor(scales_dev[i], row_major(Idx[rows], Idx[1])),
                rank_sigs,
                list_of_ctx[i],
            )
        group_end()
        for i in range(ngpus):
            var ctx = list_of_ctx[i]
            ctx.enqueue_copy(host_residual_out[i], residual_out_dev[i])
            ctx.enqueue_copy(host_out[i], out_dev[i])
            ctx.enqueue_copy(host_scales[i], scales_dev[i])
        for i in range(ngpus):
            list_of_ctx[i].synchronize()
        # Check every buffer before failing, so one run reports each divergence.
        var residual_ok = _bit_stable(
            host_residual_out, length, "fp8 residual_output", run, residual_run0
        )
        var out_ok = _bit_stable(host_out, length, "fp8 output", run, out_run0)
        var scales_ok = _bit_stable(
            host_scales, rows, "fp8 scales", run, scales_run0
        )
        ok = ok and residual_ok and out_ok and scales_ok

    assert_true(
        ok, "allreduce_residual_rmsnorm fp8 output differs across ranks or runs"
    )

    h_in.unsafe_free()
    h_residual.unsafe_free()
    h_gamma.unsafe_free()
    for i in range(ngpus):
        host_residual_out[i].unsafe_free()
        host_out[i].unsafe_free()
        host_scales[i].unsafe_free()


def run_determinism_sweep() raises -> None:
    """Sweeps (dtype, ngpus, length) covering both algorithm regimes."""
    comptime dtypes = (DType.bfloat16, DType.float32)
    comptime gpu_counts = (2, 4, 8)
    comptime lengths = (LEN_SMALL, LEN_TAIL, LEN_MID, LEN_ONE_STAGE, LEN_LARGE)

    comptime for d in range(len(dtypes)):
        comptime for g in range(len(gpu_counts)):
            comptime dtype = dtypes[d]
            comptime ngpus = gpu_counts[g]
            if DeviceContext.number_of_devices() < ngpus:
                continue
            var ctx = List[DeviceContext]()
            for i in range(ngpus):
                ctx.append(DeviceContext(device_id=i))
            comptime for li in range(len(lengths)):
                comptime length = lengths[li]
                allreduce_determinism_test[dtype=dtype, ngpus=ngpus](
                    ctx, length
                )
            allreduce_determinism_test[dtype=dtype, ngpus=ngpus, naive=True](
                ctx, LEN_SMALL
            )
            comptime if d == 0:
                fused_rmsnorm_determinism_test[ngpus, 4, 4096](ctx)
                # At most 16 KiB per rank, below every split and 2-stage
                # threshold, so this takes the warp-tiling kernel on both
                # vendors; it never reaches Lamport because it quantizes.
                fused_fp8_residual_determinism_test[ngpus, 4, 4096](ctx)
                # The split path needs a per-rank payload at or above a
                # threshold: on B200 80 KiB at 8 ranks (1.5 MiB below), where
                # 128 and 192 rows at 4096 cols run the plain allreduce's
                # Lamport and 1-stage kernels under it; on MI355 96 KiB with
                # more than 8192 cols, where 2 rows at 57344 cols (112 KiB per
                # rank, 224 KiB in all) clear that threshold and stay within
                # CDNA4's 256 KiB 1-stage limit. The split over a 2-stage
                # allreduce is left out on purpose: 2-stage agrees across
                # ranks by construction, because one rank reduces each slice,
                # and the plain sweep still covers it.
                comptime if DeviceContext.target.is_amd_gpu():
                    fused_fp8_residual_determinism_test[ngpus, 2, 57344](ctx)
                elif ngpus == 8:
                    fused_fp8_residual_determinism_test[ngpus, 128, 4096](ctx)
                    fused_fp8_residual_determinism_test[ngpus, 192, 4096](ctx)


def main() raises:
    # Positive control runs on any GPU count (host-only): proves the hash
    # detects a flip, so a PASS below is not a placebo.
    hash_positive_control()

    var num_devices = DeviceContext.number_of_devices()
    if num_devices < 2:
        print(
            "SKIP: allreduce determinism sweep needs >= 2 GPUs; found",
            num_devices,
            "- ran the host-only hash positive control only.",
        )
        return

    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")
    run_determinism_sweep()
    print("allreduce determinism sweep PASS")
