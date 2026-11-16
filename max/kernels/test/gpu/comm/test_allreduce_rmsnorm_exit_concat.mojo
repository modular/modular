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

"""Tests for the fused allreduce + RMSNorm + exit SiTU-and-mul concat.

The fused kernel is checked against a host float32 reference of the same math:
the peer sum in the order the reduce-scatter sums it, rounded to the input
dtype, the norm over the whole row, then the SiTU-and-mul of the shipped
gate/up. Only the mean square's reduction order differs (a block tree on the
device, sequential here), so the comparison is a tolerance; a wrong column
shard, a dropped rank, a skipped gamma or the wrong activation moves the output
far past it. Each case relaunches the kernel on the same signal buffers many
times over, so a scratch reuse race shows up as a mismatch.

The two arms are also run forced, side by side, and compared with each other:
byte for byte where the chain's allreduce takes its 2-stage kernel (which sums
owner-first, like the fused kernel), and with the reference's tolerance where
it takes its 1-stage kernel (which sums in global rank order, so the arms
genuinely differ by the float32 sum's rounding). The partials are filled to
make the peer sum order-sensitive, so that comparison can actually fail.
"""

from std.math import ceildiv, exp, exp2, rsqrt, tanh
from std.sys import simd_width_of, size_of

from comm.allreduce import AllReduceAlgorithm, allreduce_tuning_table
from comm.allreduce_rmsnorm_exit_concat import (
    EXIT_CONCAT_CALIBRATED_COLS,
    EXIT_CONCAT_FUSE_ROWS,
    _fuse_exit_concat,
    allreduce_rmsnorm_exit_concat,
)
from comm.device_query import dispatch_select_comm_config, get_sm_version
from comm.sync import Signal, enable_p2p, init_signal_buffer, is_p2p_enabled
from layout import Idx, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.testing import assert_true

from internal_utils._testing import test_value_for_gpu_element

comptime test_gpu_counts = (2, 4, 8)
# The K3 latent is float32 today; a bfloat16 finalize would make it bfloat16.
comptime test_in_dtypes = (DType.float32, DType.bfloat16)

# Per-element relative tolerance, and the share of elements allowed past it.
# bfloat16 keeps 8 mantissa bits, so 1 ULP is a 2**-8 relative step; the only
# expected divergence is the mean square's reduction order.
comptime relative_tolerance = Float32(4.0 * 2.0**-8)
comptime max_mismatch_rate = Float32(0.02)


@inline(.always)
def _exit_concat_partial[
    in_dtype: DType, ngpus: Int
](rank: Int, elem: Int) -> Scalar[in_dtype]:
    """One partial element of the order-sensitive test data.

    Ranks pair up as `(+v, -v * factor)` at magnitudes `2**step` apart, with
    the pairs spanning tens of octaves and `v = (elem % 251 + 1) / 7.3`: the
    partials of one element then cancel to a small fraction of the largest
    while the float32 peer sum still rounds, so the sum depends on the order
    it is taken in. Data whose sums are exact in any order -- small integers,
    or plain fractions at one scale -- would leave the arm comparison below
    unable to see a sum-order change at all.

    The factor is dtype-dependent. On float32 the peer sum stays float32, and
    `1 + 2**-13` makes a wrong order move it by a visible fraction of the
    output's bfloat16 grid step. On bfloat16 the sum itself is rounded to
    bfloat16, which crushes anything smaller, so the factor is a per-element
    `1 + (elem % 16 + 1) / 256`: the pair difference stays a richly varying
    whole number of bfloat16 units, which moves enough sums across a
    bfloat16 rounding boundary to show an order change.

    Parameters:
        in_dtype: The partials' element type.
        ngpus: Co-operating devices.

    Args:
        rank: The partial's device rank.
        elem: The flat element index.

    Returns:
        The partial's value, in `in_dtype`.
    """
    # Pairs of ranks sit `2**step` apart, spanning tens of octaves: enough
    # that the float32 peer sum rounds for either input dtype. Two devices
    # have a single pair, so nothing to spread.
    comptime step = min(22, 30 // max(ngpus // 2 - 1, 1)) if ngpus > 2 else 0
    var v = (Scalar[.float32](elem % 251) + 1.0) / 7.3
    var magnitude = v * exp2(Scalar[.float32](step * (rank // 2)))
    if rank % 2 == 1:
        comptime if in_dtype == .float32:
            magnitude = -magnitude * 1.0001220703125
        else:
            magnitude = -magnitude * (
                1.0 + Scalar[.float32](elem % 16 + 1) / 256.0
            )
    return magnitude.cast[in_dtype]()


def _assert_close[
    dtype: DType,
](
    got: HostBuffer[dtype],
    want: HostBuffer[dtype],
    length: Int,
    label: String,
) raises:
    """Asserts two host buffers agree to `relative_tolerance`, mostly.

    Parameters:
        dtype: The buffers' element type.

    Args:
        got: The kernel's output.
        want: The host reference.
        length: Elements to compare.
        label: What is compared, for the failure message.
    """
    var mismatches = 0
    var worst = Float32(0)
    var worst_idx = 0
    for i in range(length):
        var got_f = got[i].cast[.float32]()
        var want_f = want[i].cast[.float32]()
        var scale = max(abs(want_f), Float32(1.0))
        var rel = abs(got_f - want_f) / scale
        if rel > worst:
            worst = rel
            worst_idx = i
        if rel > relative_tolerance:
            mismatches += 1

    var rate = Float32(mismatches) / Float32(length)
    if rate > max_mismatch_rate:
        print(
            t"    {label}: {mismatches} / {length} differ ({rate * 100.0}%)"
            t", worst {worst} at {worst_idx}: kernel"
            t" {got[worst_idx].cast[.float32]()} vs host"
            t" {want[worst_idx].cast[.float32]()}"
        )
        raise Error(t"{label}: mismatches exceed the tolerance")


def _sum_in_rotation_order[
    in_dtype: DType, //, ngpus: Int
](partials: List[HostBuffer[in_dtype]], elem: Int, start: Int,) -> Float32:
    """Sums one element's `ngpus` partials, rotating from `start`.

    The order `_load_reduce` uses: the owner first, then the peers rotating
    from it, accumulated in float32 and rounded to the input dtype, as the
    reduce-scatter stores it.

    Parameters:
        in_dtype: The partials' element type.
        ngpus: Co-operating devices.

    Args:
        partials: Every device's host copy of its partial.
        elem: The flat element index.
        start: The rank to start the sum at.

    Returns:
        The sum of that element over the ranks, rounded to `in_dtype`.
    """
    var acc = partials[start][elem].cast[.float32]()
    for i in range(1, ngpus):
        var peer = (start + i) % ngpus
        acc += partials[peer][elem].cast[.float32]()
    return acc.cast[in_dtype]().cast[.float32]()


def test_exit_concat[
    in_dtype: DType,
    gamma_dtype: DType,
    ngpus: Int,
    rows: Int,
    cols: Int,
    half: Int,
](list_of_ctx: List[DeviceContext[]]) raises:
    """Runs the fused exit collective and checks both of its halves."""
    comptime slice_cols = cols // ngpus
    comptime out_cols = slice_cols + half
    comptime gate_cols = 2 * half
    comptime in_elems = rows * cols
    comptime gate_up_elems = rows * gate_cols
    comptime out_elems = rows * out_cols
    comptime simd_width = simd_width_of[in_dtype]()
    comptime assert cols % (ngpus * simd_width) == 0
    comptime assert half % simd_width == 0

    print(
        t"  test_exit_concat[ {in_dtype} , gamma {gamma_dtype} , {ngpus} ,"
        t" {rows} x {cols} ,"
        t" half= {half} ]"
    )

    comptime InLayout = row_major(Idx[rows], Idx[cols])
    comptime GateUpLayout = row_major(Idx[rows], Idx[gate_cols])
    comptime OutLayout = row_major(Idx[rows], Idx[out_cols])

    var in_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var gate_up_dev = List[DeviceBuffer[.bfloat16]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[.bfloat16]](capacity=ngpus)
    # The two arms, forced, into their own destinations so they can be
    # compared against each other at the same shape.
    var out_fused_dev = List[DeviceBuffer[.bfloat16]](capacity=ngpus)
    var out_chain_dev = List[DeviceBuffer[.bfloat16]](capacity=ngpus)
    var partial_hosts = List[HostBuffer[in_dtype]](capacity=ngpus)
    var gate_up_hosts = List[HostBuffer[.bfloat16]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)

    # The payload behind each `Signal` header: the plain allreduce's
    # partition staging, and nothing more. Sizing it this way is what proves
    # the chain arm's full-width sum no longer lives there.
    comptime scratch_bytes = ceildiv(rows, ngpus) * cols * size_of[in_dtype]()
    for i in range(ngpus):
        var ctx = list_of_ctx[i]
        in_dev.append(ctx.enqueue_create_buffer[in_dtype](in_elems))
        gate_up_dev.append(ctx.enqueue_create_buffer[.bfloat16](gate_up_elems))
        out_dev.append(ctx.enqueue_create_buffer[.bfloat16](out_elems))
        out_fused_dev.append(ctx.enqueue_create_buffer[.bfloat16](out_elems))
        out_chain_dev.append(ctx.enqueue_create_buffer[.bfloat16](out_elems))
        var h_in = ctx.enqueue_create_host_buffer[in_dtype](in_elems)
        var h_gate_up = ctx.enqueue_create_host_buffer[.bfloat16](gate_up_elems)
        for j in range(in_elems):
            h_in[j] = _exit_concat_partial[in_dtype, ngpus](i, j)
        for j in range(gate_up_elems):
            h_gate_up[j] = test_value_for_gpu_element[.bfloat16](i, j)
        ctx.enqueue_copy(in_dev[i], h_in)
        ctx.enqueue_copy(gate_up_dev[i], h_gate_up)
        partial_hosts.append(h_in^)
        gate_up_hosts.append(h_gate_up^)
        signal_buffers.append(
            ctx.create_buffer_sync[.uint8](size_of[Signal]() + scratch_bytes)
        )
        init_signal_buffer(signal_buffers[i], ctx)

    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(signal_buffers[i])
    )
    comptime InTile = type_of(
        TileTensor(in_dev[0], InLayout).as_unsafe_any_origin().as_imm()
    )
    var in_tiles = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTile: TileTensor(in_dev[i], InLayout)
        .as_unsafe_any_origin()
        .as_imm()
    )

    # One gamma shared by every rank, far from 1.0 so a dropped gamma shows up.
    var gamma_host = list_of_ctx[0].enqueue_create_host_buffer[gamma_dtype](
        cols
    )
    for j in range(cols):
        gamma_host[j] = (Float32(j % 7) * 0.01 + 0.5).cast[gamma_dtype]()
    var gamma_dev = list_of_ctx[0].enqueue_create_buffer[gamma_dtype](cols)
    list_of_ctx[0].enqueue_copy(gamma_dev, gamma_host)
    var gamma_tile = TileTensor(gamma_dev, row_major(cols))

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var epsilon = Float32(1e-5)
    var beta = Float32(3.0)
    var linear_beta = Float32(0.5)

    # Ten launches on the same signal buffers: after the first, each reuses the
    # scratch its peers read. Both arms share that scratch layout, so a real
    # serving trace can mix them between steps.
    comptime scratch_reuse_launches = 10
    for _ in range(scratch_reuse_launches):
        for i in range(ngpus):
            allreduce_rmsnorm_exit_concat(
                in_tiles,
                gamma_tile,
                TileTensor(gate_up_dev[i], GateUpLayout),
                TileTensor(out_dev[i], OutLayout),
                epsilon,
                beta,
                linear_beta,
                rank_sigs,
                list_of_ctx[i],
                my_rank=i,
            )

    # The chain arm's allreduce picks its own stage from the payload bytes.
    # Its 2-stage kernel sums each partition from its owner first, like the
    # fused kernel, so there the arms can be compared byte for byte; its
    # 1-stage kernel sums in global rank order, so there they genuinely
    # differ by the float32 sum's rounding and take the reference's tolerance.
    comptime sm_version = get_sm_version()
    var chain_cfg = dispatch_select_comm_config[
        ngpus, sm_version, allreduce_tuning_table
    ](in_elems * size_of[in_dtype]())
    var chain_two_stage = chain_cfg.algorithm == AllReduceAlgorithm.TWO_STAGE

    # Both arms at this shape, forced. Off AMD the gate never picks the fused
    # arm, so only the chain runs there and the two are not compared.
    var compare_arms = list_of_ctx[0].target.is_amd_gpu()
    for i in range(ngpus):
        if compare_arms:
            allreduce_rmsnorm_exit_concat(
                in_tiles,
                gamma_tile,
                TileTensor(gate_up_dev[i], GateUpLayout),
                TileTensor(out_fused_dev[i], OutLayout),
                epsilon,
                beta,
                linear_beta,
                rank_sigs,
                list_of_ctx[i],
                my_rank=i,
                force_arm=1,
            )
        allreduce_rmsnorm_exit_concat(
            in_tiles,
            gamma_tile,
            TileTensor(gate_up_dev[i], GateUpLayout),
            TileTensor(out_chain_dev[i], OutLayout),
            epsilon,
            beta,
            linear_beta,
            rank_sigs,
            list_of_ctx[i],
            my_rank=i,
            force_arm=2,
        )

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Host reference ---
    var ref_host = list_of_ctx[0].enqueue_create_host_buffer[.bfloat16](
        out_elems
    )
    var ref_tile = TileTensor(ref_host, OutLayout)

    for rank in range(ngpus):
        var gate_up_tile = TileTensor(gate_up_hosts[rank], GateUpLayout)
        for row in range(rows):
            # Mean square over the whole row, from the rotated sums.
            var m2 = Float32(0)
            for c in range(cols):
                var col_slice = c // slice_cols
                var owner = (row * ngpus + col_slice) // rows
                var v = _sum_in_rotation_order[ngpus](
                    partial_hosts, row * cols + c, owner
                )
                m2 += v * v
            var norm_factor = rsqrt(m2 / Float32(cols) + epsilon)
            # This rank's slice is column shard `rank`, so `owner` does not
            # depend on `j`.
            var owner = (row * ngpus + rank) // rows
            for j in range(slice_cols):
                var c = rank * slice_cols + j
                var v = _sum_in_rotation_order[ngpus](
                    partial_hosts, row * cols + c, owner
                )
                var normed = (
                    (v * norm_factor).cast[in_dtype]()
                    * gamma_host[c].cast[in_dtype]()
                ).cast[.bfloat16]()
                ref_tile[row, j] = normed
            for j in range(half):
                var gate = gate_up_tile[row, j].cast[.float32]()
                var up = gate_up_tile[row, half + j].cast[.float32]()
                var situ = beta * tanh(gate / beta)
                situ = situ * (1.0 / (1.0 + exp(-gate)))
                if linear_beta != 0.0:
                    up = linear_beta * tanh(up / linear_beta)
                ref_tile[row, slice_cols + j] = (situ * up).cast[.bfloat16]()

        # This rank's kernel output, against the reference built above.
        var got_host = list_of_ctx[rank].enqueue_create_host_buffer[.bfloat16](
            out_elems
        )
        list_of_ctx[rank].enqueue_copy(got_host, out_dev[rank])
        list_of_ctx[rank].synchronize()
        _assert_close(
            got_host,
            ref_host,
            out_elems,
            String(t"rank {rank}"),
        )

        if not compare_arms:
            continue
        # The forced arms against each other.
        var fused_host = list_of_ctx[rank].enqueue_create_host_buffer[
            .bfloat16
        ](out_elems)
        var chain_host = list_of_ctx[rank].enqueue_create_host_buffer[
            .bfloat16
        ](out_elems)
        list_of_ctx[rank].enqueue_copy(fused_host, out_fused_dev[rank])
        list_of_ctx[rank].enqueue_copy(chain_host, out_chain_dev[rank])
        list_of_ctx[rank].synchronize()
        if not chain_two_stage:
            # The chain's 1-stage allreduce sums in global rank order, so the
            # arms differ by the float32 sum's rounding: ULP-level, well
            # under the tolerance the host reference is held to.
            _assert_close(
                fused_host,
                chain_host,
                out_elems,
                String(t"rank {rank} arms"),
            )
            continue
        # Where the chain is 2-stage the arms must agree bit for bit.
        var differing = 0
        for j in range(out_elems):
            if fused_host[j] != chain_host[j]:
                differing += 1
                if differing <= 4:
                    print(
                        t"    rank ",
                        rank,
                        t" element ",
                        j,
                        t": fused ",
                        fused_host[j].cast[.float32](),
                        t" vs chain ",
                        chain_host[j].cast[.float32](),
                    )
        if differing != 0:
            raise Error(
                String(
                    "the fused and chain arms disagree on ",
                    differing,
                    " of ",
                    out_elems,
                    " elements on rank ",
                    rank,
                )
            )

    print("    PASS")


def test_size_gate[
    in_dtype: DType,
    ngpus: Int,
](list_of_ctx: List[DeviceContext[]]) raises:
    """Checks the size gate's byte cut and how it scales with the dtype."""
    comptime cols = EXIT_CONCAT_CALIBRATED_COLS
    # The cut is in bytes, so a bf16 call (half the bytes an element) reaches
    # it at twice the rows of a float32 call: same traffic, same decision.
    comptime at_cut = (
        EXIT_CONCAT_FUSE_ROWS if in_dtype
        == .float32 else 2 * EXIT_CONCAT_FUSE_ROWS
    )
    # Nothing is calibrated off AMD, so the chain always serves there.
    var fuses = list_of_ctx[0].target.is_amd_gpu()
    print(t"  test_size_gate[ {in_dtype} , {ngpus} , fuses= {fuses} ]")
    # Decode (K3's 32 rows) stays fused on AMD; a prefill chunk does not.
    assert_true(
        _fuse_exit_concat[in_dtype, ngpus](32, cols, list_of_ctx[0]) == fuses,
        "the gate must fuse a 32-row call on AMD and never fuse off it",
    )
    assert_true(
        _fuse_exit_concat[in_dtype, ngpus](at_cut, cols, list_of_ctx[0])
        == fuses,
        "the gate must fuse at its row cut on AMD and never fuse off it",
    )
    assert_true(
        not _fuse_exit_concat[in_dtype, ngpus](
            at_cut + 1, cols, list_of_ctx[0]
        ),
        "the gate must not fuse above its row cut",
    )
    assert_true(
        not _fuse_exit_concat[in_dtype, ngpus](4096, cols, list_of_ctx[0]),
        "the gate must not fuse a prefill chunk",
    )
    # A world size the threshold was not calibrated for takes the chain.
    assert_true(
        not _fuse_exit_concat[in_dtype, 3](32, cols, list_of_ctx[0]),
        "an uncalibrated world size must take the chain",
    )
    print("    PASS")


def main() raises:
    var num_devices = DeviceContext.number_of_devices()
    assert_true(num_devices >= 2, "need at least 2 GPUs")
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")
    assert_true(is_p2p_enabled(), "peer access should be on")

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime num_gpus = test_gpu_counts[gpu_idx]
        if num_devices < num_gpus:
            continue
        var list_of_ctx = List[DeviceContext[]]()
        for i in range(num_gpus):
            list_of_ctx.append(DeviceContext(device_id=i))

        print("\n=== allreduce_rmsnorm_exit_concat (", num_gpus, "GPUs) ===")
        comptime for dtype_idx in range(len(test_in_dtypes)):
            comptime in_dtype = test_in_dtypes[dtype_idx]
            test_exit_concat[in_dtype, in_dtype, num_gpus, 1, 512, 64](
                list_of_ctx
            )
            test_exit_concat[in_dtype, in_dtype, num_gpus, 7, 512, 128](
                list_of_ctx
            )
            # K3 decode: 32 rows of the 3584-wide latent, and the 768-wide
            # shared half TP8 gives K3's shared experts' intermediate width
            # -- the same per-device exit widths the K3 wiring tests trace
            # (a 448-wide latent shard beside the 768-wide half).
            test_exit_concat[in_dtype, in_dtype, num_gpus, 32, 3584, 768](
                list_of_ctx
            )
            # 256 rows at these cols: past the cut for float32 (chain arm),
            # exactly at it for bfloat16 (fused).
            test_exit_concat[in_dtype, in_dtype, num_gpus, 256, 3584, 256](
                list_of_ctx
            )
            test_exit_concat[in_dtype, in_dtype, num_gpus, 1024, 512, 128](
                list_of_ctx
            )
            test_size_gate[in_dtype, num_gpus](list_of_ctx)
        # Kimi K3's production form: a bfloat16 norm weight on the float32
        # latent, widened in the kernel.
        test_exit_concat[.float32, .bfloat16, num_gpus, 7, 512, 128](
            list_of_ctx
        )
        test_exit_concat[.float32, .bfloat16, num_gpus, 32, 3584, 768](
            list_of_ctx
        )

    print("\nAll tests passed!")
