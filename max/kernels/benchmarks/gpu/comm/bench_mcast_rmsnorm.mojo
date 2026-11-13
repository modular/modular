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
"""Benchmark for the multicast residual boundary against the shipped arms.

The same boundary shape `[rows, cols]` bf16, on every arm the fused residual
boundary has had, at the row counts a decode step sees. All of them add the
residual and write the pre-norm stream, so a comparison is equal work:

1. **mcast** -- `allreduce_mcast_rmsnorm`, the port: the team's symmetric
   window, one multicast store a pack, the peers' slots polled from this rank's
   own copy.
2. **lamport** -- `_lamport_allreduce_rmsnorm_launch`, the shipped
   barrier-free arm the port replaces.
3. **1-stage** -- `_allreduce_rmsnorm_fp8_launch`, the shipped warp-tiling
   kernel that reads every peer directly.
4. **2launch** -- the shipped Lamport norm followed by the standalone grouped
   quantizer over the rows it wrote: the unfolded-cost reference for the
   grouped epilogue. It is not the composite's path above the grouped row cap,
   which runs the standalone allreduce, then `rms_norm_residual_add`, then the
   quantizer.

`nrm` is the normed output only; `nrmq` is the 1x128-group FP8 row plus its
scales. The Lamport arm has no FP8 epilogue of its own, so its `nrmq` figure is
the two-launch pair. The 1-stage kernel's quantized form is the per-token one it
ships -- a scale a row rather than a scale a group of 128 -- so its `nrmq`
column is the lighter epilogue and reads as the distance to that shipped arm
rather than as equal work. PDL is off on every arm, so a number is the kernel
and not a consumer's overlap window.

The instrument is one CUDA-event window over 16 back-to-back calls on rank 0's
stream, divided by 16 and taken as the fastest of several rounds. Amortising the
host's enqueue over 16 calls is what makes the figure the collective's own
latency rather than the launch path's, and it is the per-call figure the port's
component is quoted at. It is a floor, not a serving number: at 7-21 us a call
the window still carries a launch's fixed cost, which is why the in-model
measurement in the change that added this bench is the one to quote for the arm
itself.

The ladder is MiMo-V2.6's tensor-parallel residual boundary on four B200s:
4096-wide bf16, one 128-bit pack a thread, which is exactly the band the window
reserves. Run with `num_gpus=4 num_cols=4096`.
"""

from internal_utils import arg_parse
from std.os import getenv
from std.sys import get_defined_int, size_of

from comm import Signal
from comm.device_collective import _launch_device_collective
from comm.allreduce_lamport_rmsnorm import _lamport_allreduce_rmsnorm_launch
from comm.allreduce_mcast_rmsnorm import allreduce_mcast_rmsnorm
from comm.allreduce_residual_rmsnorm import _allreduce_rmsnorm_fp8_launch
from comm.multicast_window import McarWindow, McarWindowRank, mcar_window
from comm.sync import enable_p2p, init_signal_buffer, is_p2p_enabled
from layout import TileTensor, row_major
from linalg.fp8_quantization import quantize_dynamic_scaled_fp8
from max.gpu.host import DeviceBuffer, DeviceContext, DeviceContextArray
from max.gpu.primitives.grid_controls import PDLLevel

comptime NG = 4
"""Ranks, and therefore the instantiations: the window needs the whole team."""

comptime DTYPE = DType.bfloat16
comptime EPSILON = Float32(1e-6)
comptime SCALE_UB = Float32(1200.0)
comptime GROUP = 128
comptime SIMD_WIDTH = 8
"""The 1-stage kernel's tile width, in bf16 lanes of a 128-bit vector."""
comptime THREADS_PER_BLOCK = 1024
comptime CALLS = 16
"""Calls per timed window; the figure reported is one call of the 16."""
comptime ROUNDS = 20
comptime WARMUP = 8
comptime BENCH_COLS = 4096
"""Row width the two-launch arm is instantiated for: the quantizer's width is
compile-time (it picks its tile from it), and 4096 is the ladder's."""


struct _Buffers:
    """Every arm's operands, one per rank, allocated once for one row count."""

    var inputs: List[DeviceBuffer[DTYPE]]
    var residuals: List[DeviceBuffer[DTYPE]]
    var normed: List[DeviceBuffer[DTYPE]]
    var pre: List[DeviceBuffer[DTYPE]]
    var fp8: List[DeviceBuffer[DType.float8_e4m3fn]]
    var scales: List[DeviceBuffer[DType.float32]]
    var gammas: List[DeviceBuffer[DTYPE]]
    var sigs: List[DeviceBuffer[DType.uint8]]

    def __init__(
        out self, rows: Int, cols: Int, ctxs: DeviceContextArray[NG]
    ) raises:
        var length = rows * cols
        var scratch = NG * length * size_of[DTYPE]()
        self.inputs = List[DeviceBuffer[DTYPE]](capacity=NG)
        self.residuals = List[DeviceBuffer[DTYPE]](capacity=NG)
        self.normed = List[DeviceBuffer[DTYPE]](capacity=NG)
        self.pre = List[DeviceBuffer[DTYPE]](capacity=NG)
        self.fp8 = List[DeviceBuffer[DType.float8_e4m3fn]](capacity=NG)
        self.scales = List[DeviceBuffer[DType.float32]](capacity=NG)
        self.gammas = List[DeviceBuffer[DTYPE]](capacity=NG)
        self.sigs = List[DeviceBuffer[DType.uint8]](capacity=NG)

        var input_host = List[Scalar[DTYPE]](
            length=length, fill=Scalar[DTYPE](0)
        )
        var residual_host = List[Scalar[DTYPE]](
            length=length, fill=Scalar[DTYPE](0)
        )
        var gamma_host = List[Scalar[DTYPE]](length=cols, fill=Scalar[DTYPE](0))
        for i in range(length):
            input_host[i] = Scalar[DTYPE]((i % 251) + 1) * Scalar[DTYPE](0.01)
            residual_host[i] = Scalar[DTYPE](1) + Scalar[DTYPE](
                (i % 97) - 48
            ) * Scalar[DTYPE](0.005)
        for i in range(cols):
            gamma_host[i] = Scalar[DTYPE](0.5) + Scalar[DTYPE](i % 7) * Scalar[
                DTYPE
            ](0.01)

        var stride = (rows + 3) // 4 * 4
        var scale_len = (cols // GROUP) * stride
        for i in range(NG):
            self.inputs.append(ctxs[i].enqueue_create_buffer[DTYPE](length))
            self.residuals.append(ctxs[i].enqueue_create_buffer[DTYPE](length))
            self.normed.append(ctxs[i].enqueue_create_buffer[DTYPE](length))
            self.pre.append(ctxs[i].enqueue_create_buffer[DTYPE](length))
            self.fp8.append(
                ctxs[i].enqueue_create_buffer[DType.float8_e4m3fn](length)
            )
            self.scales.append(
                ctxs[i].enqueue_create_buffer[DType.float32](scale_len)
            )
            self.gammas.append(ctxs[i].enqueue_create_buffer[DTYPE](cols))
            # The scratch the boundary kernels may stage through, sized as a
            # deployment's signal buffers are.
            self.sigs.append(
                ctxs[i].create_buffer_sync[DType.uint8](
                    size_of[Signal]() + scratch
                )
            )
            ctxs[i].enqueue_copy(self.inputs[i], input_host)
            ctxs[i].enqueue_copy(self.residuals[i], residual_host)
            ctxs[i].enqueue_copy(self.gammas[i], gamma_host)
            init_signal_buffer(self.sigs[i], ctxs[i])
        for i in range(NG):
            ctxs[i].synchronize()

    def rank_sigs(
        mut self,
    ) -> Array[MutPointer[Signal, MutAnyOrigin], NG]:
        """The team's signal pointers, in rank order."""
        return Array[MutPointer[Signal, MutAnyOrigin], NG](
            fill_with=lambda (i: Int) {ref} -> MutPointer[
                Signal, MutAnyOrigin
            ]: Signal.unsafe_ptr_from(self.sigs[i])
        )


def _window(
    ctxs: DeviceContextArray[NG], cols: Int
) raises -> Optional[Pointer[McarWindow[NG], MutUntrackedOrigin]]:
    """The team's symmetric window, or None where the fabric cannot map one."""
    var team = Array[DeviceContext[], NG](
        fill_with_unrolled=lambda [i: Int]() -> DeviceContext[]: ctxs[i].copy()
    )
    return mcar_window[NG, DTYPE](team, cols)


def _us_a_call[
    F: def[Int]() raises -> None
](ctxs: DeviceContextArray[NG], f: F) raises -> Float64:
    """The fastest per-call time of CALLS back-to-back calls, microseconds.

    Each rank enqueues its CALLS calls from its own worker thread, the way a
    collective is launched in a model, so the window measures the boundary and
    not one host thread's enqueue rate.
    """

    @inline(.always)
    def window() raises {imm}:
        _launch_device_collective[NG](f, ctxs.copy())

    for _ in range(WARMUP):
        window()
    for i in range(NG):
        ctxs[i].synchronize()
    var best = Float64(1e18)
    for _ in range(ROUNDS):
        var ns = ctxs[0].execution_time(window, 1)
        for i in range(NG):
            ctxs[i].synchronize()
        var per_call = Float64(ns) / Float64(CALLS)
        if per_call < best:
            best = per_call
    return best / 1000.0


def run_mcast[
    quant: Bool
](rows: Int, cols: Int, ctxs: DeviceContextArray[NG]) raises -> Float64:
    """The port: one multicast store a pack, no peer load."""
    var mcar: Pointer[McarWindow[NG], MutUntrackedOrigin]
    if case .Some(window) = _window(ctxs, cols):
        mcar = window
    else:
        print("mcast: no symmetric window on this device set; skipping")
        return Float64(-1)
    var bufs = _Buffers(rows, cols, ctxs)
    var stride = (rows + 3) // 4 * 4

    @inline(.always)
    def call[i: Int]() raises {imm}:
        for _ in range(CALLS):
            allreduce_mcast_rmsnorm[NG, quant=quant](
                rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
                    bufs.inputs[i].unsafe_ptr()
                ),
                rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
                    bufs.residuals[i].unsafe_ptr()
                ),
                rebind[MutPointer[Scalar[DTYPE], MutAnyOrigin]](
                    bufs.pre[i].unsafe_ptr()
                ),
                rebind[MutPointer[Scalar[DTYPE], MutAnyOrigin]](
                    bufs.normed[i].unsafe_ptr()
                ),
                rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
                    bufs.gammas[i].unsafe_ptr()
                ),
                rebind[MutPointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin]](
                    bufs.fp8[i].unsafe_ptr()
                ),
                rebind[MutPointer[Float32, MutAnyOrigin]](
                    bufs.scales[i].unsafe_ptr()
                ),
                stride,
                rows,
                cols,
                EPSILON,
                Float32(0),
                SCALE_UB,
                i,
                mcar[].rank(i),
                ctxs[i],
                PDLLevel.OFF,
            )

    return _us_a_call(ctxs, call)


def run_lamport(
    rows: Int, cols: Int, ctxs: DeviceContextArray[NG]
) raises -> Float64:
    """The shipped barrier-free arm, at the port's work: residual and stream."""
    var bufs = _Buffers(rows, cols, ctxs)
    var rank_sigs = bufs.rank_sigs()

    @inline(.always)
    def call[i: Int]() raises {imm}:
        for _ in range(CALLS):
            var residual = TileTensor(
                rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
                    bufs.residuals[i].unsafe_ptr()
                ),
                row_major(rows, cols),
            )
            var stream = TileTensor(
                rebind[MutPointer[Scalar[DTYPE], MutAnyOrigin]](
                    bufs.pre[i].unsafe_ptr()
                ),
                row_major(rows, cols),
            )
            _lamport_allreduce_rmsnorm_launch[DTYPE, NG, has_residual=True](
                i,
                rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
                    bufs.inputs[i].unsafe_ptr()
                ),
                rebind[MutPointer[Scalar[DTYPE], MutAnyOrigin]](
                    bufs.normed[i].unsafe_ptr()
                ),
                rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
                    bufs.gammas[i].unsafe_ptr()
                ),
                rank_sigs,
                rows,
                cols,
                Scalar[DTYPE](EPSILON),
                Scalar[DTYPE](0),
                ctxs[i],
                residual=residual.as_imm(),
                residual_output=stream,
            )

    return _us_a_call(ctxs, call)


def run_two_launch(
    rows: Int, cols: Int, ctxs: DeviceContextArray[NG]
) raises -> Float64:
    """The unfolded grouped epilogue: the Lamport norm, then the quantizer.

    This is what the composite does above the grouped row cap, and what a caller
    that misses the arm pays: the norm writes the normed rows in the message
    dtype, and the standalone grouped quantizer reads them back.
    """
    comptime assert BENCH_COLS % GROUP == 0
    var bufs = _Buffers(rows, cols, ctxs)
    var rank_sigs = bufs.rank_sigs()
    var stride = (rows + 3) // 4 * 4

    @inline(.always)
    def call[i: Int]() raises {imm}:
        for _ in range(CALLS):
            var residual = TileTensor(
                rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
                    bufs.residuals[i].unsafe_ptr()
                ),
                row_major(rows, cols),
            )
            var stream = TileTensor(
                rebind[MutPointer[Scalar[DTYPE], MutAnyOrigin]](
                    bufs.pre[i].unsafe_ptr()
                ),
                row_major(rows, cols),
            )
            _lamport_allreduce_rmsnorm_launch[DTYPE, NG, has_residual=True](
                i,
                rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
                    bufs.inputs[i].unsafe_ptr()
                ),
                rebind[MutPointer[Scalar[DTYPE], MutAnyOrigin]](
                    bufs.normed[i].unsafe_ptr()
                ),
                rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
                    bufs.gammas[i].unsafe_ptr()
                ),
                rank_sigs,
                rows,
                cols,
                Scalar[DTYPE](EPSILON),
                Scalar[DTYPE](0),
                ctxs[i],
                residual=residual.as_imm(),
                residual_output=stream,
            )
            var normed_ptr = rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
                bufs.normed[i].unsafe_ptr()
            )

            @inline(.always)
            def staged_input[
                width: Int, alignment: Int
            ](row: Int, col: Int) {var normed_ptr, var cols} -> SIMD[
                DTYPE, width
            ]:
                return normed_ptr.unsafe_load[width=width, alignment=1](
                    row * cols + col
                )

            var fp8_view = TileTensor(
                rebind[MutPointer[Scalar[DType.float8_e4m3fn], MutAnyOrigin]](
                    bufs.fp8[i].unsafe_ptr()
                ),
                row_major(rows, cols),
            )
            # The scale tensor's own row extent, padded as a consuming GEMM's is.
            var scales_view = TileTensor(
                rebind[MutPointer[Float32, MutAnyOrigin]](
                    bufs.scales[i].unsafe_ptr()
                ),
                row_major(cols // GROUP, stride),
            )
            quantize_dynamic_scaled_fp8[
                out_dtype=DType.float8_e4m3fn,
                in_dtype=DTYPE,
                scales_dtype=DType.float32,
                group_size_or_per_token=GROUP,
                num_cols=BENCH_COLS,
                pdl_level=PDLLevel.OFF,
            ](
                staged_input,
                fp8_view,
                scales_view,
                SCALE_UB,
                ctxs[i],
                num_rows=rows,
            )

    return _us_a_call(ctxs, call)


def run_one_stage[
    quant: Bool
](rows: Int, cols: Int, ctxs: DeviceContextArray[NG]) raises -> Float64:
    """The shipped warp-tiling kernel, which reads every peer directly.

    At the port's work: residual added and the pre-norm stream written.
    """
    comptime out_dtype = DType.float8_e4m3fn if quant else DTYPE
    var bufs = _Buffers(rows, cols, ctxs)
    var rank_sigs = bufs.rank_sigs()
    var src = Array[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin], NG](
        fill_with_unrolled=lambda [i: Int]() -> ImmPointer[
            Scalar[DTYPE], ImmutAnyOrigin
        ]: rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
            bufs.inputs[i].unsafe_ptr()
        )
    )

    @inline(.always)
    def call[i: Int]() raises {imm}:
        for _ in range(CALLS):
            var out_ptr: MutPointer[Scalar[out_dtype], MutAnyOrigin]
            comptime if quant:
                out_ptr = rebind[MutPointer[Scalar[out_dtype], MutAnyOrigin]](
                    bufs.fp8[i].unsafe_ptr()
                )
            else:
                out_ptr = rebind[MutPointer[Scalar[out_dtype], MutAnyOrigin]](
                    bufs.normed[i].unsafe_ptr()
                )
            var out_view = TileTensor(out_ptr, row_major(rows, cols))
            var gamma_view = TileTensor(
                rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
                    bufs.gammas[i].unsafe_ptr()
                ),
                row_major(cols),
            )
            var scale_view = TileTensor(
                rebind[MutPointer[Float32, MutAnyOrigin]](
                    bufs.scales[i].unsafe_ptr()
                ),
                row_major(rows),
            )
            var residual = TileTensor(
                rebind[ImmPointer[Scalar[DTYPE], ImmutAnyOrigin]](
                    bufs.residuals[i].unsafe_ptr()
                ),
                row_major(rows, cols),
            )
            var stream = TileTensor(
                rebind[MutPointer[Scalar[DTYPE], MutAnyOrigin]](
                    bufs.pre[i].unsafe_ptr()
                ),
                row_major(rows, cols),
            )
            _allreduce_rmsnorm_fp8_launch[
                SIMD_WIDTH,
                DTYPE,
                out_dtype,
                DType.float32,
                NG,
                THREADS_PER_BLOCK,
                has_residual=True,
                pdl_level=PDLLevel.OFF,
            ](
                rows,
                cols,
                src,
                out_view,
                gamma_view,
                EPSILON,
                Scalar[DTYPE](0),
                SCALE_UB,
                scale_view,
                rank_sigs,
                i,
                ctxs[i],
                residual=residual.as_imm(),
                residual_output=stream,
            )

    return _us_a_call(ctxs, call)


def _cell(value: Float64) -> String:
    """One decimal place, as the reference figures are quoted."""
    if value < 0.0:
        return "      -"
    var scaled = Int(value * 10.0 + 0.5)
    return String(scaled // 10, ".", scaled % 10)


def _moka_quoted(
    rows: Int, row_counts: List[Int], micros: List[Float64]
) -> Float64:
    """The component's quoted figure for `rows`, or `-1` off its ladder."""
    for i in range(len(row_counts)):
        if row_counts[i] == rows:
            return micros[i]
    return Float64(-1)


def main() raises:
    var ngpus = get_defined_int["num_gpus", 4]()
    var num_cols = get_defined_int["num_cols", 4096]()
    var num_devices = DeviceContext.number_of_devices()
    if num_devices < ngpus or ngpus != NG:
        print(
            t"this bench instantiates {NG} ranks; found {num_devices} devices"
            t" - skipping."
        )
        return
    _ = enable_p2p()
    if not is_p2p_enabled():
        print("P2P not enabled, skipping benchmark.")
        return

    var ctxs = DeviceContextArray[NG]()

    var rows_list: List[Int] = [1, 8, 16, 32, 64, 128]
    # `--rows R` runs one rung of the ladder, which is what a per-row profiler
    # capture needs: one row's kernels, under their own names.
    var only_row = arg_parse("rows", 0)
    if only_row > 0:
        rows_list = List[Int]()
        rows_list.append(only_row)
    # The Moka component's v2 protocol at these row counts, microseconds a call,
    # measured on four B200s with a per-call instrument. Keyed by row count, not
    # by position in the ladder: `--rows R` runs a single rung, so an index into
    # the ladder would quote the one-row figure for whatever R is. They are
    # quoted figures from one host, not a measurement of this one, so
    # `MCAR_MOKA_US` (comma-separated, one per row count) replaces them with a
    # same-host run's.
    var moka_rows: List[Int] = [1, 8, 16, 32, 64, 128]
    var moka_us: List[Float64] = [5.12, 5.49, 6.03, 6.85, 9.06, 14.94]
    var moka_source = "quoted from one B200 host"
    var moka_override = getenv("MCAR_MOKA_US")
    if moka_override.byte_length() > 0:
        var figures = moka_override.split(",")
        if len(figures) != len(moka_rows):
            raise Error(
                t"MCAR_MOKA_US needs {len(moka_rows)} figures, one for each of"
                t" rows 1, 8, 16, 32, 64 and 128"
            )
        for i in range(len(figures)):
            moka_us[i] = atof(figures[i].strip())
        moka_source = "from MCAR_MOKA_US"

    print(
        t"mcast residual boundary vs the shipped arms: {NG} GPUs, bf16, rows x"
        t" {num_cols}, {CALLS}-call instrument, fastest of {ROUNDS} rounds,"
        t" microseconds a call; Moka figures {moka_source}"
    )
    print(
        "rows | mcast nrm | lamport | 1stage nrm | mcast nrmq | 2launch nrmq"
        " | 1stage nrmq | Moka nrm | mcast vs Moka"
    )
    print(
        "-----+-----------+---------+------------+------------+--------------"
        "+-------------+----------+---------------"
    )
    for r in range(len(rows_list)):
        var rows = rows_list[r]
        var mcast_nrm = run_mcast[False](rows, num_cols, ctxs)
        var lamport = run_lamport(rows, num_cols, ctxs)
        var one_nrm = run_one_stage[False](rows, num_cols, ctxs)
        var mcast_nrmq = run_mcast[True](rows, num_cols, ctxs)
        var two_nrmq = run_two_launch(
            rows, num_cols, ctxs
        ) if num_cols == BENCH_COLS else Float64(-1)
        var one_nrmq = run_one_stage[True](rows, num_cols, ctxs)
        var moka = _moka_quoted(rows, moka_rows, moka_us)
        var against_moka: String
        if moka > 0.0:
            var delta = (mcast_nrm - moka) / moka * 100.0
            var sign = "+" if delta >= 0.0 else "-"
            var scaled = Int(abs(delta) * 10.0 + 0.5)
            against_moka = sign + String(scaled // 10, ".", scaled % 10, "%")
        else:
            # A row count the component does not quote: no column to compare to.
            against_moka = "- (off the quoted ladder)"
        print(
            String(rows),
            "  |   ",
            _cell(mcast_nrm),
            "   |  ",
            _cell(lamport),
            "  |    ",
            _cell(one_nrm),
            "    |   ",
            _cell(mcast_nrmq),
            "    |     ",
            _cell(two_nrmq),
            "     |    ",
            _cell(one_nrmq),
            "     |   ",
            _cell(moka),
            "    | ",
            against_moka,
        )
