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
"""The fused EP combine over many rounds on one buffer set, on 4 GPUs.

`ep_fused_combine_kernel_api` runs at DeepSeek-V4.1-Flash decode shapes
(hidden 5120, top-6, 384 experts, bf16 messages) for many dispatch-then-combine
rounds on ONE buffer set. The host never resets it after the first memset, so
every counter and flag that dispatch and combine restore is reused round after
round. Each round's output must equal a host reference exactly.

- Rows 1, 7, 13, 71, 72 and 96 a rank, the sizes decode batches send, under
  three routings: uniform; skewed (75% of picks from 8 hot experts on rank 0,
  so hot pairs and many empty pairs); and every pick on rank 0 (ranks 1-3
  receive nothing and publish only empty pairs).
- Rounds go out back to back, with short per-rank delay kernels in a third of
  them, so the ranks drift and interleave differently from round to round.
- An expert kernel scales each received row by a per-expert power of two, and
  the router weights are distinct powers of two per top-k slot, so the reduced
  output is exact in fp32. A row delivered to the wrong (token, k) slot, a
  stale row from an earlier round or a missing row changes the bf16 result.
- A planted fault must fail: one case swaps the routing of two received rows
  in one round, and the test fails if the check misses it.
"""

from std.sys import size_of
from std.memory.alloc import unsafe_alloc
from std.time import global_perf_counter_ns

from comm.sync import enable_p2p
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Coord, TileTensor, Idx
from layout.tile_layout import row_major
from shmem.ep import ep_fused_combine_kernel_api
from shmem.ep_comm import (
    EP_NO_INPUT_SCALES,
    BF16TokenFormat,
    EPLocalSyncCounters,
    dispatch_async_kernel,
    dispatch_wait_kernel,
)

comptime HIDDEN = 5120
comptime TOP_K = 6
comptime N_EXPERTS = 384
comptime N_RANKS = 4
comptime MAX_TOK = 96
comptime N_LOCAL = N_EXPERTS // N_RANKS
comptime MAX_RECV = N_EXPERTS * MAX_TOK
comptime ROUNDS = 100
"""Rounds per case, all on the same buffer set."""
comptime input_type = DType.bfloat16


@always_inline
def mix64(x: UInt64) -> UInt64:
    """Splitmix64 finalizer: a deterministic hash for test data."""
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@always_inline
def token_value(rank: Int, rnd: Int, case_id: Int, t: Int, h: Int) -> Float32:
    """A multiple of 1/64 in [-127/64, 127/64]: exact in bf16."""
    var key = (
        UInt64(case_id) << 52
        ^ UInt64(rnd) << 36
        ^ UInt64(rank) << 32
        ^ UInt64(t) << 16
        ^ UInt64(h)
    )
    var v = Int(mix64(key) % 255) - 127
    return Float32(v) / 64.0


@always_inline
def expert_scale(e: Int) -> Float32:
    """+-2^(e%5 - 2), negative for odd experts: exact scaling of bf16 values."""
    var mags = SIMD[DType.float32, 8](0.25, 0.5, 1.0, 2.0, 4.0, 0, 0, 0)
    var m = mags[e % 5]
    return -m if (e % 2) == 1 else m


@always_inline
def router_weight(t: Int, k: Int, rnd: Int) -> Float32:
    """Distinct powers of two over the six top-k slots of a token."""
    var p = (t + k + rnd) % 6 + 1
    var w = Float32(1.0)
    for _ in range(p):
        w = w * 0.5
    return w


def reuse_expert_scale_kernel(
    rows_p: MutPointer[Scalar[input_type], MutAnyOrigin],
    row_offsets_p: MutPointer[UInt32, MutAnyOrigin],
    my_rank: Int32,
):
    """One block per local expert scales that expert's received rows in place.
    """
    var e_l = block_idx.x
    var lo = Int(row_offsets_p[e_l])
    var hi = Int(row_offsets_p[e_l + 1])
    var s = expert_scale(Int(my_rank) * N_LOCAL + e_l)
    for r in range(lo, hi):
        for h in range(thread_idx.x, HIDDEN, 256):
            var p = rows_p.unsafe_offset(r * HIDDEN + h)
            p[] = (p[].cast[DType.float32]() * s).cast[input_type]()


def reuse_swap_rows_kernel(src_info_p: MutPointer[Int32, MutAnyOrigin]):
    """Planted fault: swaps the routing of the first two received rows, so the
    combine delivers each to the other's (token, k) slot."""
    var a0 = src_info_p[0]
    var a1 = src_info_p[1]
    src_info_p[0] = src_info_p[2]
    src_info_p[1] = src_info_p[3]
    src_info_p[2] = a0
    src_info_p[3] = a1


def reuse_delay_kernel(ns: UInt64):
    """Spins one warp for `ns` nanoseconds to shift this rank's schedule."""
    var t0 = global_perf_counter_ns()
    while global_perf_counter_ns() - t0 < ns:
        pass


def pick_experts(
    topk: MutPointer[Int32, MutUntrackedOrigin],
    rows: Int,
    mode: Int,
    rank: Int,
    rnd: Int,
    case_id: Int,
):
    """Picks top-k experts per token, distinct within a token.

    Mode 0 is uniform, 1 skewed (75% of picks from 8 hot experts on rank 0),
    and 2 puts every pick on rank 0.
    """
    var ctr = UInt64(0)
    for t in range(rows):
        for k in range(TOP_K):
            var clash = True
            var pick = 0
            while clash:
                ctr += 1
                var r = mix64(
                    UInt64(case_id) << 48
                    ^ UInt64(rnd) << 32
                    ^ UInt64(rank) << 28
                    ^ UInt64(t) << 12
                    ^ ctr
                )
                if mode == 0:
                    pick = Int(r % UInt64(N_EXPERTS))
                elif mode == 1:
                    if (r >> 40) % 4 != 0:
                        pick = Int(r % 8)
                    else:
                        pick = Int(r % UInt64(N_EXPERTS))
                else:
                    pick = Int(r % UInt64(N_LOCAL))
                clash = False
                for j in range(k):
                    if Int(topk[t * TOP_K + j]) == pick:
                        clash = True
            topk[t * TOP_K + k] = Int32(pick)


def main() raises:
    if not enable_p2p():
        raise Error("Cannot enable P2P Mem Access!")
    if DeviceContext.number_of_devices() != N_RANKS:
        print("This test requires exactly", N_RANKS, "GPUs; skipping.")
        return

    var ctxs = List[DeviceContext]()
    for i in range(N_RANKS):
        ctxs.append(DeviceContext(device_id=i))

    comptime output_layout = row_major((Idx[MAX_RECV], Idx[HIDDEN]))
    comptime token_fmt_type = BF16TokenFormat[
        output_layout=type_of(output_layout), HIDDEN, TOP_K
    ]
    comptime msg_bytes = token_fmt_type.msg_size()
    comptime combine_msg_bytes = size_of[input_type]() * HIDDEN
    comptime counters_size = EPLocalSyncCounters[N_EXPERTS].total_size()
    comptime hw_info = type_of(ctxs[0]).default_device_info

    # One communication buffer set per rank, set once and never reset by the
    # host again.
    var d_send = List[DeviceBuffer[.uint8]](capacity=N_RANKS)
    var d_recv = List[DeviceBuffer[.uint8]](capacity=N_RANKS)
    var d_recv_count = List[DeviceBuffer[.uint64]](capacity=N_RANKS)
    var c_send = List[DeviceBuffer[.uint8]](capacity=N_RANKS)
    var c_recv = List[DeviceBuffer[.uint8]](capacity=N_RANKS)
    var c_recv_count = List[DeviceBuffer[.uint64]](capacity=N_RANKS)
    var counters = List[DeviceBuffer[.int32]](capacity=N_RANKS)
    var d_out = List[DeviceBuffer[input_type]](capacity=N_RANKS)
    var row_offsets = List[DeviceBuffer[.uint32]](capacity=N_RANKS)
    var expert_ids = List[DeviceBuffer[.int32]](capacity=N_RANKS)
    var src_info = List[DeviceBuffer[.int32]](capacity=N_RANKS)
    # Per-round inputs and outputs (not communication state).
    var r_topk = List[DeviceBuffer[.int32]](capacity=N_RANKS)
    var r_tok = List[DeviceBuffer[input_type]](capacity=N_RANKS)
    var r_w = List[DeviceBuffer[.float32]](capacity=N_RANKS)
    var r_out = List[DeviceBuffer[input_type]](capacity=N_RANKS)
    for i in range(N_RANKS):
        var ctx = ctxs[i]
        d_send.append(ctx.enqueue_create_buffer[.uint8](MAX_TOK * msg_bytes))
        d_recv.append(ctx.enqueue_create_buffer[.uint8](MAX_RECV * msg_bytes))
        d_recv_count.append(ctx.enqueue_create_buffer[.uint64](N_EXPERTS))
        ctx.enqueue_memset(d_recv_count[i], UInt64.MAX_FINITE)
        c_send.append(ctx.enqueue_create_buffer[.uint8](4096))
        c_recv.append(
            ctx.enqueue_create_buffer[.uint8](
                MAX_TOK * TOP_K * combine_msg_bytes
            )
        )
        ctx.enqueue_memset(c_recv[i], UInt8(0))
        c_recv_count.append(ctx.enqueue_create_buffer[.uint64](N_EXPERTS))
        ctx.enqueue_memset(c_recv_count[i], UInt64.MAX_FINITE)
        counters.append(ctx.enqueue_create_buffer[.int32](counters_size))
        ctx.enqueue_memset(counters[i], Int32(0))
        d_out.append(ctx.enqueue_create_buffer[input_type](MAX_RECV * HIDDEN))
        row_offsets.append(ctx.enqueue_create_buffer[.uint32](N_LOCAL + 1))
        expert_ids.append(ctx.enqueue_create_buffer[.int32](N_LOCAL))
        src_info.append(ctx.enqueue_create_buffer[.int32](MAX_RECV * 2))
        r_topk.append(
            ctx.enqueue_create_buffer[.int32](ROUNDS * MAX_TOK * TOP_K)
        )
        r_tok.append(
            ctx.enqueue_create_buffer[input_type](ROUNDS * MAX_TOK * HIDDEN)
        )
        r_w.append(
            ctx.enqueue_create_buffer[.float32](ROUNDS * MAX_TOK * TOP_K)
        )
        r_out.append(
            ctx.enqueue_create_buffer[input_type](ROUNDS * MAX_TOK * HIDDEN)
        )
    for i in range(N_RANKS):
        ctxs[i].synchronize()

    var d_recv_ptrs = Array[MutPointer[UInt8, MutAnyOrigin], N_RANKS](
        uninitialized=True
    )
    var d_recv_count_ptrs = Array[MutPointer[UInt64, MutAnyOrigin], N_RANKS](
        uninitialized=True
    )
    var c_send_ptrs = unsafe_alloc[UInt64](N_RANKS)
    var c_recv_ptrs = unsafe_alloc[UInt64](N_RANKS)
    var c_recv_count_ptrs = unsafe_alloc[UInt64](N_RANKS)
    for i in range(N_RANKS):
        d_recv_ptrs[i] = d_recv[i].unsafe_ptr().as_unsafe_any_origin()
        d_recv_count_ptrs[i] = (
            d_recv_count[i].unsafe_ptr().as_unsafe_any_origin()
        )
        c_send_ptrs[i] = UInt64(Int(c_send[i].unsafe_ptr()))
        c_recv_ptrs[i] = UInt64(Int(c_recv[i].unsafe_ptr()))
        c_recv_count_ptrs[i] = UInt64(Int(c_recv_count[i].unsafe_ptr()))
    var c_ptrs_layout = row_major[N_RANKS]()
    var c_send_tt = TileTensor[.uint64, type_of(c_ptrs_layout), ImmutAnyOrigin](
        ptr=c_send_ptrs.as_unsafe_any_origin(), layout=c_ptrs_layout
    )
    var c_recv_tt = TileTensor[.uint64, type_of(c_ptrs_layout), ImmutAnyOrigin](
        ptr=c_recv_ptrs.as_unsafe_any_origin(), layout=c_ptrs_layout
    )
    var c_recv_count_tt = TileTensor[
        .uint64, type_of(c_ptrs_layout), ImmutAnyOrigin
    ](ptr=c_recv_count_ptrs.as_unsafe_any_origin(), layout=c_ptrs_layout)
    var counters_layout = row_major[counters_size]()
    var out_tt_layout = row_major((Idx[MAX_RECV], Idx[HIDDEN]))
    var row_offsets_layout = row_major[N_LOCAL + 1]()
    var expert_ids_layout = row_major[N_LOCAL]()
    var src_info_layout = row_major((Idx[MAX_RECV], Idx[2]))

    # The kernels are instantiated for the maximum row count; each case passes
    # its row count at run time.
    var probe_topk_layout = row_major(MAX_TOK, Idx[TOP_K])
    var probe_tok_layout = row_major((MAX_TOK, Idx[HIDDEN]))
    var format_handler = token_fmt_type(
        TileTensor[input_type, type_of(out_tt_layout), MutAnyOrigin](
            ptr=d_out[0].unsafe_ptr().as_unsafe_any_origin(),
            layout=out_tt_layout,
        )
    )
    comptime dispatch_async = dispatch_async_kernel[
        input_type,
        hw_info.max_thread_block_size,
        type_of(probe_tok_layout),
        type_of(probe_topk_layout),
        hw_info.sm_count,
        N_EXPERTS,
        N_RANKS,
        MAX_TOK,
        N_RANKS,
        token_fmt_type,
        use_shmem=False,
        InputScalesFnType=type_of(EP_NO_INPUT_SCALES),
    ]
    comptime dispatch_wait = dispatch_wait_kernel[
        hw_info.max_thread_block_size,
        type_of(row_offsets_layout),
        type_of(expert_ids_layout),
        type_of(src_info_layout),
        hw_info.sm_count,
        N_EXPERTS,
        N_RANKS,
        MAX_TOK,
        type_of(format_handler),
    ]

    @inline(.always)
    def launch_dispatch_async(
        d: Int, rnd: Int, rows: Int
    ) raises {mut r_tok, mut r_topk, mut d_send, mut counters, imm}:
        var tok_layout = row_major((rows, Idx[HIDDEN]))
        var topk_layout = row_major(rows, Idx[TOP_K])
        ctxs[d].enqueue_function[dispatch_async](
            TileTensor[input_type, type_of(tok_layout), ImmutAnyOrigin](
                ptr=r_tok[d]
                .unsafe_ptr()
                .unsafe_offset(rnd * MAX_TOK * HIDDEN)
                .as_unsafe_any_origin(),
                layout=tok_layout,
            ),
            TileTensor[.int32, type_of(topk_layout), ImmutAnyOrigin](
                ptr=r_topk[d]
                .unsafe_ptr()
                .unsafe_offset(rnd * MAX_TOK * TOP_K)
                .as_unsafe_any_origin(),
                layout=topk_layout,
            ),
            d_send[d].unsafe_ptr().as_unsafe_any_origin(),
            d_recv_ptrs,
            d_recv_count_ptrs,
            EPLocalSyncCounters[N_EXPERTS](
                counters[d].unsafe_ptr().as_unsafe_any_origin()
            ),
            Int32(d),
            host_args=(EP_NO_INPUT_SCALES,),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @inline(.always)
    def launch_dispatch_wait(
        d: Int,
    ) raises {
        mut d_out,
        mut row_offsets,
        mut expert_ids,
        mut src_info,
        mut counters,
        imm,
    }:
        ctxs[d].enqueue_function[dispatch_wait](
            type_of(format_handler)(
                TileTensor[input_type, type_of(out_tt_layout), MutAnyOrigin](
                    ptr=d_out[d].unsafe_ptr().as_unsafe_any_origin(),
                    layout=out_tt_layout,
                )
            ),
            TileTensor[.uint32, type_of(row_offsets_layout), MutAnyOrigin](
                ptr=row_offsets[d].unsafe_ptr().as_unsafe_any_origin(),
                layout=row_offsets_layout,
            ),
            TileTensor[.int32, type_of(expert_ids_layout), MutAnyOrigin](
                ptr=expert_ids[d].unsafe_ptr().as_unsafe_any_origin(),
                layout=expert_ids_layout,
            ),
            TileTensor[.int32, type_of(src_info_layout), MutAnyOrigin](
                ptr=src_info[d].unsafe_ptr().as_unsafe_any_origin(),
                layout=src_info_layout,
            ),
            d_recv_ptrs[d],
            d_recv_count_ptrs[d],
            EPLocalSyncCounters[N_EXPERTS](
                counters[d].unsafe_ptr().as_unsafe_any_origin()
            ),
            Int32(d),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @inline(.always)
    def launch_combine(
        d: Int, rnd: Int, rows: Int
    ) raises {mut r_w, mut r_out, mut counters, mut d_out, mut src_info, imm}:
        var w_layout = row_major((rows, Idx[TOP_K]))
        var o_layout = row_major((rows, Idx[HIDDEN]))
        var w_tt = TileTensor[.float32, type_of(w_layout), ImmutAnyOrigin](
            ptr=r_w[d]
            .unsafe_ptr()
            .unsafe_offset(rnd * MAX_TOK * TOP_K)
            .as_unsafe_any_origin(),
            layout=w_layout,
        )
        var o_tt = TileTensor[input_type, type_of(o_layout), MutAnyOrigin](
            ptr=r_out[d]
            .unsafe_ptr()
            .unsafe_offset(rnd * MAX_TOK * HIDDEN)
            .as_unsafe_any_origin(),
            layout=o_layout,
        )

        @inline(.always)
        def store_out[
            dtype: DType, width: SIMDLength, *, alignment: Int = 1
        ](coords: Coord, val: SIMD[dtype, width]) {var o_tt}:
            o_tt.store[width=width, alignment=alignment * size_of[dtype]()](
                coords, rebind[SIMD[input_type, width]](val)
            )

        @inline(.always)
        def router_w[
            width: Int
        ](token_idx: Int, topk_id: Int) {var w_tt} -> SIMD[.float32, width]:
            return w_tt.load[width=width]((token_idx, topk_id))

        ep_fused_combine_kernel_api[
            HIDDEN,
            TOP_K,
            N_EXPERTS,
            MAX_TOK,
            N_RANKS,
            1,
            "gpu",
            use_shmem=False,
        ](
            rows,
            TileTensor[.int32, type_of(counters_layout), MutAnyOrigin](
                ptr=counters[d].unsafe_ptr().as_unsafe_any_origin(),
                layout=counters_layout,
            ),
            TileTensor[input_type, type_of(out_tt_layout), MutAnyOrigin](
                ptr=d_out[d].unsafe_ptr().as_unsafe_any_origin(),
                layout=out_tt_layout,
            ).as_imm(),
            TileTensor[.int32, type_of(src_info_layout), MutAnyOrigin](
                ptr=src_info[d].unsafe_ptr().as_unsafe_any_origin(),
                layout=src_info_layout,
            ).as_imm(),
            c_send_tt,
            c_recv_tt,
            c_recv_count_tt,
            ctxs[d],
            store_out,
            router_w,
        )

    var rows_list = [1, 7, 13, 71, 72, 96]
    # Routings 0-2 are checked; the last pass is the planted fault (uniform
    # routing, 13 rows a rank), which must fail the check.
    var mode_names = ["uniform", "skewed", "rank0", "planted fault"]
    var host_topk = unsafe_alloc[Int32](ROUNDS * MAX_TOK * TOP_K)
    var host_tok = unsafe_alloc[Scalar[input_type]](ROUNDS * MAX_TOK * HIDDEN)
    var host_w = unsafe_alloc[Float32](ROUNDS * MAX_TOK * TOP_K)
    # Expected and returned outputs per rank, for the check.
    var host_exp = List[MutPointer[Scalar[input_type], MutUntrackedOrigin]]()
    var host_got = List[MutPointer[Scalar[input_type], MutUntrackedOrigin]]()
    for _ in range(N_RANKS):
        host_exp.append(
            unsafe_alloc[Scalar[input_type]](ROUNDS * MAX_TOK * HIDDEN)
        )
        host_got.append(
            unsafe_alloc[Scalar[input_type]](ROUNDS * MAX_TOK * HIDDEN)
        )

    var total_bad = 0
    var case_id = 0
    for mode in range(4):
        for ri in range(len(rows_list)):
            var rows = rows_list[ri]
            var planted = mode == 3
            if planted and rows != 13:
                continue
            case_id += 1
            # Host inputs and expected outputs for every round of this case.
            for d in range(N_RANKS):
                for rnd in range(ROUNDS):
                    var tk = host_topk.unsafe_offset(rnd * MAX_TOK * TOP_K)
                    pick_experts(
                        tk, rows, 0 if planted else mode, d, rnd, case_id
                    )
                    for t in range(rows):
                        # Every product is exact in fp32 (a token value has 8
                        # significant bits and each scale and weight is a power
                        # of two), so a row reduces to x * sum_k(w_k * s_k).
                        var c_t = Float32(0)
                        for k in range(TOP_K):
                            var w = router_weight(t, k, rnd)
                            host_w[(rnd * MAX_TOK + t) * TOP_K + k] = w
                            c_t += w * expert_scale(Int(tk[t * TOP_K + k]))
                        for h in range(HIDDEN):
                            var x = token_value(d, rnd, case_id, t, h)
                            var i = (rnd * MAX_TOK + t) * HIDDEN + h
                            host_tok[i] = x.cast[input_type]()
                            host_exp[d][i] = (x * c_t).cast[input_type]()
                ctxs[d].enqueue_copy(r_topk[d], host_topk)
                ctxs[d].enqueue_copy(r_tok[d], host_tok)
                ctxs[d].enqueue_copy(r_w[d], host_w)
                ctxs[d].enqueue_memset(r_out[d], Scalar[input_type](0))
                ctxs[d].synchronize()

            # Enqueue every round back to back on all ranks.
            for rnd in range(ROUNDS):
                for d in range(N_RANKS):
                    if rnd % 3 == 2 and d == (rnd * 5 + 1) % N_RANKS:
                        ctxs[d].enqueue_function[reuse_delay_kernel](
                            UInt64(((rnd * 13) % 5) * 5000),
                            grid_dim=1,
                            block_dim=32,
                        )
                    launch_dispatch_async(d, rnd, rows)
                for d in range(N_RANKS):
                    launch_dispatch_wait(d)
                for d in range(N_RANKS):
                    ctxs[d].enqueue_function[reuse_expert_scale_kernel](
                        d_out[d].unsafe_ptr().as_unsafe_any_origin(),
                        row_offsets[d].unsafe_ptr().as_unsafe_any_origin(),
                        Int32(d),
                        grid_dim=N_LOCAL,
                        block_dim=256,
                    )
                if planted and rnd == ROUNDS // 2:
                    ctxs[0].enqueue_function[reuse_swap_rows_kernel](
                        src_info[0].unsafe_ptr().as_unsafe_any_origin(),
                        grid_dim=1,
                        block_dim=1,
                    )
                for d in range(N_RANKS):
                    if rnd % 3 == 1 and d == (rnd * 7 + 3) % N_RANKS:
                        ctxs[d].enqueue_function[reuse_delay_kernel](
                            UInt64(((rnd * 11) % 5) * 5000),
                            grid_dim=1,
                            block_dim=32,
                        )
                    launch_combine(d, rnd, rows)
            for d in range(N_RANKS):
                ctxs[d].synchronize()

            # Check every round on every rank.
            var case_bad = 0
            for d in range(N_RANKS):
                ctxs[d].enqueue_copy(host_got[d], r_out[d])
                ctxs[d].synchronize()
            for d in range(N_RANKS):
                var reported = False
                for rnd in range(ROUNDS):
                    for t in range(rows):
                        var base = (rnd * MAX_TOK + t) * HIDDEN
                        for h in range(HIDDEN):
                            var g = host_got[d][base + h]
                            var e = host_exp[d][base + h]
                            if (
                                g.cast[DType.float32]()
                                != e.cast[DType.float32]()
                            ):
                                case_bad += 1
                                if not reported:
                                    reported = True
                                    print(
                                        "  mismatch: rank",
                                        d,
                                        "round",
                                        rnd,
                                        "token",
                                        t,
                                        "h",
                                        h,
                                        "got",
                                        g,
                                        "expected",
                                        e,
                                    )
            if planted:
                print("planted fault: mismatches", case_bad)
                if case_bad == 0:
                    raise Error("the planted fault went undetected")
                continue
            total_bad += case_bad
            print(
                "routing",
                mode_names[mode],
                "rows",
                rows,
                "rounds",
                ROUNDS,
                "mismatches",
                case_bad,
                "PASS" if case_bad == 0 else "FAIL",
            )

    host_topk.unsafe_free()
    host_tok.unsafe_free()
    host_w.unsafe_free()
    for d in range(N_RANKS):
        host_exp[d].unsafe_free()
        host_got[d].unsafe_free()
    c_send_ptrs.unsafe_free()
    c_recv_ptrs.unsafe_free()
    c_recv_count_ptrs.unsafe_free()
    if total_bad != 0:
        raise Error("EP combine reuse test failed: ", total_bad, " mismatches")
    print("EP combine reuse test passed:", case_id, "cases")
