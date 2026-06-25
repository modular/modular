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
# NON-PERTURBING ITERATION TRACE of the REAL search loop.
#
# This kernel is a BYTE-FAITHFUL copy of the current Fix#1 search loop in
# TopKSamplingFromProbKernel (topk_fi.mojo lines 659-789), importing the REAL
# `device_sampling_from_prob` and `_block_reduce_value_counts`, driven by the
# SAME Random(seed, offset=bx) stream, the SAME fill_random_for_test
# distribution, and grid_dim=16 (matching the real launch occupancy). The ONLY
# addition is per-iteration global-buffer stores from tx==0 (non-perturbing:
# they read already-computed scalars; no extra collectives, no in-kernel print).
#
# For the chosen batch it records, per iteration:
#   low, high, q, u, sampled_id (post-fallback), raw sampled_id (pre-fallback),
#   last_valid_id, count_0, count_1, value_0, accept-flag.
# Plus the final emitted sampled_id and an accepted/collapsed verdict.
#
# Run: source utils/start-modular.sh; mojo <thisfile>
# ===----------------------------------------------------------------------=== #

from std.math import ceildiv
from std.gpu import thread_idx, block_idx, barrier
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.random import Random, random_float64, seed
from std.testing import TestSuite
from layout import Idx, TileTensor, row_major

from nn.topk_fi import device_sampling_from_prob, _block_reduce_value_counts

comptime N = 1024
comptime K = 50
comptime BATCH = 16
comptime BLOCK = 1024
comptime VEC = 8
comptime MAX_ITERS = 40
comptime TRACE_FIELDS = 16  # per-iter scalars stored


def trace_kernel[
    block_size: Int, vec_size: Int
](
    probs_ptr: UnsafePointer[Float32, MutAnyOrigin],
    d: Int,
    k: Int,
    rng_seed: UInt64,
    target_batch: Int,
    # trace[0] = niter; trace[1] = final sampled_id; trace[2] = accepted(1)/collapsed(0)
    # trace[3 + it*TRACE_FIELDS + f] = per-iter field f
    trace: UnsafePointer[Float32, MutAnyOrigin],
):
    var bx = block_idx.x
    var tx = thread_idx.x

    var sampled_id_sram = stack_allocation[
        1, Int, address_space=AddressSpace.SHARED
    ]()
    var last_valid_id_sram = stack_allocation[
        1, Int, address_space=AddressSpace.SHARED
    ]()

    var generator = Random(seed=rng_seed, offset=UInt64(bx))
    var probs_row = TileTensor(probs_ptr + bx * d, row_major(Idx[1], d))

    var probs_vec: SIMD[DType.float32, vec_size]
    var aggregate: Float32
    var sampled_id = 0
    var q: Float32 = 1.0
    var low: Float32 = 0.0
    var high: Float32 = 1.0

    var do_trace = bx == target_batch
    var it = 0
    var accepted = 0

    while low < high:
        if tx == 0:
            sampled_id_sram[0] = d
        barrier()

        var u = generator.step_uniform()[0] * q
        aggregate = 0.0
        var thread_max_valid = -1

        for i in range(ceildiv(d, block_size * vec_size)):
            probs_vec = 0
            if (i * block_size + tx) * vec_size < d:
                probs_vec = probs_row.load[width=vec_size](
                    (Idx[0], ((i * block_size + tx) * vec_size))
                ).cast[DType.float32]()

            var result = device_sampling_from_prob[
                vec_size, block_size, DType.float32, False
            ](i, d, low, u, probs_vec, aggregate, sampled_id_sram)
            aggregate = result[0]
            thread_max_valid = max(thread_max_valid, result[1])
            if aggregate > u:
                break

        var block_max_valid = block.max[block_size=block_size, broadcast=False](
            Int32(thread_max_valid)
        )

        if tx == 0 and block_max_valid != -1:
            last_valid_id_sram[0] = Int(block_max_valid)

        barrier()

        var raw_sampled = sampled_id_sram[0]
        sampled_id = raw_sampled
        if sampled_id == d:
            sampled_id = last_valid_id_sram[0]

        var pivot_0 = Float32(probs_row.load[width=1]((Idx[0], sampled_id)))
        var pivot_1 = (pivot_0 + high) / 2.0

        var thread_count_0 = Float32(0)
        var thread_value_0 = Float32(0)
        var thread_count_1 = Float32(0)
        var thread_value_1 = Float32(0)

        for i in range(ceildiv(d, block_size * vec_size)):
            probs_vec = 0
            if (i * block_size + tx) * vec_size < d:
                probs_vec = probs_row.load[width=vec_size](
                    (Idx[0], ((i * block_size + tx) * vec_size))
                ).cast[DType.float32]()

            comptime for j in range(vec_size):
                var idx = (i * block_size + tx) * vec_size + j
                var is_valid = idx < d
                var gt0 = probs_vec[j] > pivot_0
                thread_value_0 += probs_vec[j] if gt0 else 0.0
                thread_count_0 += Float32(1) if (gt0 and is_valid) else Float32(
                    0
                )
                var gt1 = probs_vec[j] > pivot_1
                thread_value_1 += probs_vec[j] if gt1 else 0.0
                thread_count_1 += Float32(1) if (gt1 and is_valid) else Float32(
                    0
                )

        var counts = _block_reduce_value_counts[block_size, broadcast=True](
            Int32(thread_count_0),
            thread_value_0,
            Int32(thread_count_1),
            thread_value_1,
        )
        var count_0 = counts[0]
        var value_0 = counts[1]
        var count_1 = counts[2]
        var value_1 = counts[3]

        if do_trace and tx == 0 and it < MAX_ITERS:
            # tx==0 SEQUENTIAL ground-truth count of #{prob > pivot_0}, the
            # independent check that the reduced count_0 belongs to pivot_0.
            var seq_c0 = Int32(0)
            for sidx in range(d):
                if Float32(probs_row.load[width=1]((Idx[0], sidx))) > pivot_0:
                    seq_c0 += 1

            var base = 3 + it * TRACE_FIELDS
            trace[base + 0] = low
            trace[base + 1] = high
            trace[base + 2] = q
            trace[base + 3] = u
            trace[base + 4] = Float32(sampled_id)
            trace[base + 5] = Float32(raw_sampled)
            trace[base + 6] = Float32(last_valid_id_sram[0])
            trace[base + 7] = Float32(count_0)
            trace[base + 8] = Float32(count_1)
            trace[base + 9] = value_0
            trace[base + 10] = Float32(1) if count_0 < Int32(k) else Float32(0)
            trace[base + 11] = pivot_0
            trace[base + 12] = Float32(seq_c0)

        # Parallel store from the LAST thread (tx==1023): does it see the SAME
        # sampled_id / raw_sampled / pivot_0 as tx==0? If not, the per-thread
        # pivot is NON-UNIFORM => the fused count_0 mixes pivots (the desync).
        if do_trace and tx == block_size - 1 and it < MAX_ITERS:
            var base = 3 + it * TRACE_FIELDS
            trace[base + 13] = Float32(sampled_id)
            trace[base + 14] = Float32(raw_sampled)
            trace[base + 15] = pivot_0

        it += 1

        if count_0 < Int32(k):
            accepted = 1
            break

        if count_1 < Int32(k):
            low = pivot_0
            high = pivot_1
            q = value_0
        else:
            low = pivot_1
            q = value_1

    if do_trace and tx == 0:
        trace[0] = Float32(it)
        trace[1] = Float32(sampled_id)
        trace[2] = Float32(accepted)

    # Loop-divergence check: record the iteration count seen by tx==0, a
    # mid-warp thread (tx==512), and the last thread (tx==1023). If they differ,
    # the `while low < high` loop is GENUINELY DIVERGENT across threads => the
    # in-loop barriers run on a PARTIAL threadgroup (UB) and NO per-iteration
    # publish can fix it. Stored at the very end of the trace buffer.
    var div_base = 3 + MAX_ITERS * TRACE_FIELDS
    if do_trace and tx == 0:
        trace[div_base + 0] = Float32(it)
    if do_trace and tx == block_size // 2:
        trace[div_base + 1] = Float32(it)
    if do_trace and tx == block_size - 1:
        trace[div_base + 2] = Float32(it)


def run(ctx: DeviceContext, target_batch: Int) raises:
    var inp = ctx.enqueue_create_buffer[DType.float32](BATCH * N)
    var trace = ctx.enqueue_create_buffer[DType.float32](
        6 + MAX_ITERS * TRACE_FIELDS
    )

    # Replicate fill_random_for_test[normalized] over BATCH rows.
    seed(42)
    var rows = List[List[Float32]]()
    var host_in = InlineArray[Float32, BATCH * N](fill=0)
    for b in range(BATCH):
        var row = List[Float32]()
        var s = Float32(0)
        for i in range(N):
            var val = random_float64(0.01, 10.0).cast[DType.float32]()
            row.append(val)
            s += val
        for i in range(N):
            row[i] = row[i] / s
            host_in[b * N + i] = row[i]
        rows.append(row^)

    inp.enqueue_copy_from(Span(host_in))
    trace.enqueue_fill(Float32(-1.0e30))

    ctx.enqueue_function[trace_kernel[BLOCK, VEC]](
        inp,
        N,
        K,
        UInt64(42),
        target_batch,
        trace,
        grid_dim=BATCH,
        block_dim=BLOCK,
    )

    var got = InlineArray[Float32, 6 + MAX_ITERS * TRACE_FIELDS](fill=0)
    trace.enqueue_copy_to(Span(got))
    ctx.synchronize()

    var niter = Int(got[0])
    var final_sid = Int(got[1])
    var accepted = Int(got[2])

    # Host rank of the final sid.
    var prob = rows[target_batch][final_sid]
    var rank = 0
    for i in range(N):
        if rows[target_batch][i] > prob:
            rank += 1

    var dbase = 3 + MAX_ITERS * TRACE_FIELDS
    var niter_tx0 = Int(got[dbase + 0])
    var niter_tx512 = Int(got[dbase + 1])
    var niter_tx1023 = Int(got[dbase + 2])

    print("=== TRACE batch", target_batch, "===")
    print(
        "  niter=",
        niter,
        " final_sid=",
        final_sid,
        " host_rank=",
        rank,
        " (k=",
        K,
        ") verdict=",
        "ACCEPTED" if accepted == 1 else "COLLAPSED",
    )
    print(
        "  LOOP-DIVERGENCE: niter tx0=",
        niter_tx0,
        " tx512=",
        niter_tx512,
        " tx1023=",
        niter_tx1023,
        "->",
        "DIVERGENT" if (
            niter_tx0 != niter_tx512 or niter_tx0 != niter_tx1023
        ) else "uniform",
    )
    for it in range(min(niter, MAX_ITERS)):
        var base = 3 + it * TRACE_FIELDS
        print(
            "  it",
            it,
            ": low=",
            got[base + 0],
            " high=",
            got[base + 1],
            " q=",
            got[base + 2],
            " u=",
            got[base + 3],
            " sid=",
            Int(got[base + 4]),
            " raw=",
            Int(got[base + 5]),
            " lastvalid=",
            Int(got[base + 6]),
            " c0=",
            Int(got[base + 7]),
            " c1=",
            Int(got[base + 8]),
            " v0=",
            got[base + 9],
            " accept=",
            Int(got[base + 10]),
            " pivot0_um=",
            Int(got[base + 11] * 1.0e6),
            " seq_c0=",
            Int(got[base + 12]),
            " | tx1023: sid=",
            Int(got[base + 13]),
            " raw=",
            Int(got[base + 14]),
            " pivot0_um=",
            Int(got[base + 15] * 1.0e6),
        )
    print()


def test_iter_trace() raises:
    with DeviceContext() as ctx:
        run(ctx, 4)  # far-miss / fallback (sid=1023 in real run, rank 178)
        run(ctx, 1)  # near-miss (rank 62)
        run(ctx, 11)  # far-miss (rank 129)
        run(ctx, 0)  # passes (rank 41) -- control


def main() raises:
    var suite = TestSuite()
    suite.test[test_iter_trace]()
    suite^.run()
