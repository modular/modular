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

"""Checks the all-reduce mode's local EP kernels against the protocol kernels.

Under `skip_a2a`, `ep_fused_dispatch_kernel_api` and
`ep_fused_combine_kernel_api` run `ep_local_dispatch` and `ep_local_combine`
in place of the protocol kernels, which `_ep_fused_dispatch_protocol` and
`_ep_fused_combine_protocol` launch here, so the two must give the same
results bit for bit: every (token, top-k) slot's row holds the same bytes and
scales, the row offsets, expert ids and the counter buffer's row ranges match,
and the combine stores the same output. A row-local stand-in for the expert
FFN runs between dispatch and combine, so the check holds whatever order each
dispatch puts an expert's rows in. NVFP4 rows also get the same scales in the
grouped matmul's 128-row scale blocks, the same per-group block offsets, and
zeros past each group's rows in its last block. The local combine also runs
after the protocol dispatch, which the all-reduce mode keeps for MXFP8 and
MXFP4 rows, and the protocol combine after the local dispatch.
"""

from std.collections import Dict
from std.math import align_up, max, min
from std.memory.alloc import unsafe_alloc
from std.memory.unsafe import bitcast
from std.random import randn, random_float64, seed
from std.sys import size_of
from std.testing import assert_equal, assert_true

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from shmem.ep import (
    _ep_fused_combine_protocol,
    _ep_fused_dispatch_protocol,
    ep_fused_combine_kernel_api,
    ep_fused_dispatch_kernel_api,
)
from shmem.ep_comm import (
    BF16TokenFormat,
    BlockwiseFP8TokenFormat,
    EP_BLOCKWISE_FP8_GROUP,
    EP_DATA_READY_FLAG,
    EPLocalSyncCounters,
    NVBlockScaledTokenFormat,
    NullExpertLoad,
    TokenFormat,
    ep_dispatch_recv_size,
)
from linalg.fp4_utils import (
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)

comptime H = 1024
comptime TOP_K = 8
comptime WORLD = 4
comptime N_TOTAL = 288
"""GLM-5.3-Flash's experts: 72 a device, so the group scans take three
rounds of 32."""
comptime N_LOCAL = N_TOTAL // WORLD
comptime MAX_TPR = 1024
comptime MAX_RECV = MAX_TPR * min(N_LOCAL, TOP_K) + MAX_TPR
comptime GROUP = EP_BLOCKWISE_FP8_GROUP
comptime FP8 = DType.float8_e4m3fn
comptime COUNTERS = EPLocalSyncCounters[N_LOCAL].total_size()
comptime NV = NVFP4_SF_VECTOR_SIZE
comptime SF_ATOMS = H // NV // SF_ATOM_K
comptime SF_BLOCK_BYTES = SF_ATOMS * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
# Every group can open one block past the rows' own.
comptime SF_BLOCKS = MAX_RECV // SF_MN_GROUP_SIZE + N_LOCAL + 2
comptime INPUT_SCALE = Float32(512.0)
comptime SF_SENTINEL = UInt8(0x7F)
"""An E4M3 NaN, so a scale byte the dispatch should write and did not is
never mistaken for one it wrote."""

comptime BF16_ROWS = 0
comptime FP8_ROWS = 1
comptime NVFP4_ROWS = 2


def expert_stand_in[
    dtype: DType
](
    rows: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    scales: MutPointer[Float32, MutUntrackedOrigin],
    dst: MutPointer[BFloat16, MutUntrackedOrigin],
):
    """Maps each dispatched row to an expert output that depends only on the
    row's bytes and scales."""
    var r = block_idx.x
    for i in range(thread_idx.x, H, block_dim.x):
        var v: Float32
        comptime if dtype == DType.uint8:
            # Two E2M1 codes a byte; the scales are checked per row instead.
            var byte = rows[r * (H // 2) + i // 2].cast[DType.uint32]()
            v = ((byte >> UInt32(4 * (i % 2))) & 0xF).cast[DType.float32]()
        else:
            v = rows[r * H + i].cast[DType.float32]()
        comptime if dtype == FP8:
            v *= scales[(i // GROUP) * MAX_RECV + r]
        dst[r * H + i] = (v * 0.5 + 0.25).cast[DType.bfloat16]()


def make_routing(topk: MutPointer[Int32, _], n: Int, skew: Bool):
    """Draws distinct experts per token, optionally with a few hot ones."""
    for t in range(n):
        var k = 0
        while k < TOP_K:
            var u = random_float64()
            var e = (Int(Float64(N_TOTAL) * u * u * u) * 37 + 5) % N_TOTAL
            if not skew:
                e = Int(u * Float64(N_TOTAL)) % N_TOTAL
            var dup = False
            for j in range(k):
                dup = dup or Int(topk[t * TOP_K + j]) == e
            if not dup:
                topk[t * TOP_K + k] = Int32(e)
                k += 1


struct Outputs(Movable):
    """One dispatch's outputs and one combine's output, on the host."""

    var rows: List[UInt8]
    var scales: List[Float32]
    var sf: List[UInt8]
    var sf_offsets: List[UInt32]
    var row_offsets: List[UInt32]
    var expert_ids: List[Int32]
    var src_info: List[Int32]
    var region_a: List[Int32]
    var combined: List[UInt16]

    def __init__(out self):
        self.rows = List[UInt8]()
        self.scales = List[Float32]()
        self.sf = List[UInt8]()
        self.sf_offsets = List[UInt32]()
        self.row_offsets = List[UInt32]()
        self.expert_ids = List[Int32]()
        self.src_info = List[Int32]()
        self.region_a = List[Int32]()
        self.combined = List[UInt16]()


def _to_list[
    dtype: DType
](ctx: DeviceContext, buf: DeviceBuffer[dtype], count: Int) raises -> List[
    Scalar[dtype]
]:
    var host = unsafe_alloc[Scalar[dtype]](count)
    ctx.enqueue_copy(host, buf.create_sub_buffer[dtype](0, count))
    ctx.synchronize()
    var values = List[Scalar[dtype]](capacity=count)
    for i in range(count):
        values.append(host[i])
    host.unsafe_free()
    return values^


def slot_rows(o: Outputs, n_local: Int) -> Dict[Int, Int]:
    """Maps each routed `token * TOP_K + k` slot to its dispatched row."""
    var rows = Dict[Int, Int]()
    for e in range(n_local):
        var count = Int(o.region_a[2 * e + 1])
        var start = Int(o.region_a[2 * e]) - EP_DATA_READY_FLAG - count
        for r in range(start, start + count):
            rows[
                Int(o.src_info[2 * r]) * TOP_K + Int(o.src_info[2 * r + 1])
            ] = r
    return rows^


def writes_local_rows[Fmt: TokenFormat]() -> Bool:
    """Returns whether the all-reduce mode dispatches `Fmt` on one device."""
    return Fmt.local_rows


def sf_row_of(o: Outputs, row: Int, n_groups: Int) -> Int:
    """Returns a dispatched row's place in the NVFP4 scale blocks."""
    var g = 0
    while g + 1 < n_groups and Int(o.row_offsets[g + 1]) <= row:
        g += 1
    var start = Int(o.row_offsets[g])
    var first_block = start // SF_MN_GROUP_SIZE + Int(o.sf_offsets[g])
    return first_block * SF_MN_GROUP_SIZE + row - start


def sf_byte(sf_row: Int, s: Int) -> Int:
    """Returns the offset of scale `s` of a row in the 5D scale layout."""
    var block, u = divmod(sf_row, SF_MN_GROUP_SIZE)
    var u_hi, u_lo = divmod(u, SF_ATOM_M[0])
    var atom, j = divmod(s, SF_ATOM_K)
    return (
        block * SF_BLOCK_BYTES
        + atom * SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
        + u_lo * SF_ATOM_M[1] * SF_ATOM_K
        + u_hi * SF_ATOM_K
        + j
    )


def check_sf_tails(o: Outputs, n_groups: Int, what: String) raises:
    """Checks that each group's last scale block is zero past its rows."""
    var bad = 0
    for g in range(n_groups):
        var rows = Int(o.row_offsets[g + 1]) - Int(o.row_offsets[g])
        if rows % SF_MN_GROUP_SIZE == 0:
            continue
        var first = sf_row_of(o, Int(o.row_offsets[g]), n_groups)
        for r in range(first + rows, first + align_up(rows, SF_MN_GROUP_SIZE)):
            for s in range(H // NV):
                bad += Int(o.sf[sf_byte(r, s)] != 0)
    assert_equal(bad, 0, String(t"scale tails not zero: {what}"))


def run_case[
    kind: Int, fused: Bool
](ctx: DeviceContext, n: Int, skew: Bool) raises:
    comptime fp8 = kind == FP8_ROWS
    comptime nvfp4 = kind == NVFP4_ROWS
    comptime SE = 1 if fused else 0
    comptime out_layout = row_major((MAX_RECV, Idx[H]))
    comptime fp4_layout = row_major((MAX_RECV, Idx[H // 2]))
    comptime sf_layout = row_major(
        (
            Idx[SF_BLOCKS],
            Idx[SF_ATOMS],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    comptime sf_offsets_layout = row_major[N_LOCAL + SE]()
    comptime Nvfp4Fmt = NVBlockScaledTokenFormat[
        quant_dtype=DType.uint8,
        scales_dtype=FP8,
        output_layout=type_of(fp4_layout),
        scales_offset_layout=type_of(sf_offsets_layout),
        H,
        TOP_K,
    ]
    comptime scales_layout = row_major((H // GROUP, Idx[MAX_RECV]))
    comptime Fp8Fmt = BlockwiseFP8TokenFormat[
        fp8_dtype=FP8,
        scales_dtype=DType.float32,
        output_layout=type_of(out_layout),
        scales_layout=type_of(scales_layout),
        H,
        TOP_K,
    ]
    comptime Bf16Fmt = BF16TokenFormat[
        output_layout=type_of(out_layout), H, TOP_K
    ]
    comptime msg_bytes = Nvfp4Fmt.msg_size() if nvfp4 else (
        Fp8Fmt.msg_size() if fp8 else Bf16Fmt.msg_size()
    )
    comptime row_bytes = H // 2 if nvfp4 else H * (1 if fp8 else 2)
    comptime assert writes_local_rows[
        Nvfp4Fmt
    ](), "the all-reduce mode dispatches NVFP4 rows on their own device"
    var my_rank = Int(ctx.device().id)
    print(t"case kind={kind} fused={fused} n={n} skew={skew}")

    # fmt: off
    var d_send = ctx.enqueue_create_buffer[DType.uint8](MAX_TPR * msg_bytes)
    var d_recv = ctx.enqueue_create_buffer[DType.uint8](ep_dispatch_recv_size(n_experts=N_LOCAL, n_ranks=1, max_tokens_per_rank=MAX_TPR, msg_bytes=msg_bytes, token_recv=True))
    var d_recv_count = ctx.enqueue_create_buffer[DType.uint64](N_LOCAL)
    ctx.enqueue_memset(d_recv_count, UInt64.MAX_FINITE)
    var c_recv = ctx.enqueue_create_buffer[DType.uint8](TOP_K * MAX_TPR * H * 2)
    var c_recv_count = ctx.enqueue_create_buffer[DType.uint64](N_LOCAL)
    ctx.enqueue_memset(c_recv_count, UInt64.MAX_FINITE)
    var counters = ctx.enqueue_create_buffer[DType.int32](COUNTERS)
    ctx.enqueue_memset(counters, Int32(0))
    var rows_fp8 = ctx.enqueue_create_buffer[FP8](MAX_RECV * H)
    var rows_bf16 = ctx.enqueue_create_buffer[DType.bfloat16](MAX_RECV * H)
    var scales = ctx.enqueue_create_buffer[DType.float32](MAX_RECV * (H // GROUP))
    var rows_fp4 = ctx.enqueue_create_buffer[DType.uint8](MAX_RECV * (H // 2))
    var sf = ctx.enqueue_create_buffer[FP8](SF_BLOCKS * SF_BLOCK_BYTES)
    var sf_offsets = ctx.enqueue_create_buffer[DType.uint32](N_LOCAL + SE)
    var row_offsets = ctx.enqueue_create_buffer[DType.uint32](N_LOCAL + 1 + SE)
    var expert_ids = ctx.enqueue_create_buffer[DType.int32](N_LOCAL + SE)
    var src_info = ctx.enqueue_create_buffer[DType.int32](MAX_RECV * 2)
    var expert_out = ctx.enqueue_create_buffer[DType.bfloat16](MAX_RECV * H)
    var tokens = ctx.enqueue_create_buffer[DType.bfloat16](max(n, 1) * H)
    var topk = ctx.enqueue_create_buffer[DType.int32](max(n, 1) * TOP_K)
    var weights = ctx.enqueue_create_buffer[DType.float32](max(n, 1) * TOP_K)
    var combined = ctx.enqueue_create_buffer[DType.bfloat16](max(n, 1) * H)
    # fmt: on

    var h_topk = unsafe_alloc[Int32](max(n, 1) * TOP_K)
    var h_tokens = unsafe_alloc[BFloat16](max(n, 1) * H)
    var h_weights = unsafe_alloc[Float32](max(n, 1) * TOP_K)
    seed(n * 7 + Int(skew))
    make_routing(h_topk, n, skew)
    randn(h_tokens, n * H)
    randn(h_weights, n * TOP_K)
    ctx.enqueue_copy(topk, h_topk)
    ctx.enqueue_copy(tokens, h_tokens)
    ctx.enqueue_copy(weights, h_weights)

    comptime N_PTR = 8
    var ptrs = unsafe_alloc[UInt64](4 * N_PTR)
    ptrs[my_rank] = UInt64(Int(d_send.unsafe_ptr()))
    ptrs[N_PTR + my_rank] = UInt64(Int(d_recv.unsafe_ptr()))
    ptrs[2 * N_PTR + my_rank] = UInt64(Int(d_recv_count.unsafe_ptr()))
    ptrs[3 * N_PTR + my_rank] = UInt64(Int(c_recv.unsafe_ptr()))
    var ptrs_layout = row_major[N_PTR]()
    var send_tt = TileTensor(ptrs, ptrs_layout)
    var recv_tt = TileTensor(ptrs.unsafe_offset(N_PTR), ptrs_layout)
    var count_tt = TileTensor(ptrs.unsafe_offset(2 * N_PTR), ptrs_layout)
    var c_recv_tt = TileTensor(ptrs.unsafe_offset(3 * N_PTR), ptrs_layout)
    var c_count_ptrs = unsafe_alloc[UInt64](N_PTR)
    c_count_ptrs[my_rank] = UInt64(Int(c_recv_count.unsafe_ptr()))
    var c_count_tt = TileTensor(c_count_ptrs, ptrs_layout)

    var tokens_tt = TileTensor(tokens, row_major((n, Idx[H])))
    var topk_tt = TileTensor(topk, row_major((n, Idx[TOP_K])))
    var row_offsets_tt = TileTensor(row_offsets, row_major[N_LOCAL + 1 + SE]())
    var expert_ids_tt = TileTensor(expert_ids, row_major[N_LOCAL + SE]())
    var src_info_tt = TileTensor(src_info, row_major((Idx[MAX_RECV], Idx[2])))
    var counters_tt = TileTensor(counters, row_major[COUNTERS]())
    var rows_fp8_tt = TileTensor(rows_fp8, out_layout)
    var rows_bf16_tt = TileTensor(rows_bf16, out_layout)
    var scales_tt = TileTensor(scales, scales_layout)
    var rows_fp4_tt = TileTensor(rows_fp4, fp4_layout)
    var sf_tt = TileTensor(sf, sf_layout)
    var sf_offsets_tt = TileTensor(sf_offsets, sf_offsets_layout)
    var expert_out_tt = TileTensor(expert_out, row_major((MAX_RECV, Idx[H])))
    var combined_tt = TileTensor(combined, row_major((n, Idx[H])))
    var weights_tt = TileTensor(weights, row_major((n, Idx[TOP_K])))
    var topk_p = (
        topk.unsafe_ptr().as_imm().unsafe_origin_cast[ImmUntrackedOrigin]()
    )

    @inline(.always)
    def router_weights_fn[
        width: Int
    ](token: Int, k: Int) {var weights_tt} -> SIMD[.float32, width]:
        return SIMD[.float32, width](weights_tt[token, k])

    @inline(.always)
    def store_fn[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](coords: Coord, val: SIMD[dtype, width]) {var combined_tt}:
        combined_tt.store[alignment=alignment * size_of[dtype]()](
            coords, rebind[SIMD[.bfloat16, width]](val)
        )

    # Only NVFP4 rows read the input scale.
    @inline(.always)
    def input_scales_fn[dtype: DType](expert_id: Int) -> Scalar[dtype]:
        return INPUT_SCALE.cast[dtype]()

    def dispatch[
        Fmt: TokenFormat, //, local: Bool
    ](token_handler: Fmt) raises {imm}:
        comptime if local:
            ep_fused_dispatch_kernel_api[
                N_LOCAL,
                MAX_TPR,
                1,
                1,
                fused,
                skip_a2a=True,
                allreduce_world_size=WORLD,
                has_input_scales=nvfp4,
            ](
                token_handler,
                row_offsets_tt,
                expert_ids_tt,
                src_info_tt,
                counters_tt,
                tokens_tt,
                topk_tt,
                send_tt,
                recv_tt,
                count_tt,
                ctx,
                input_scales_fn,
            )
        else:
            _ep_fused_dispatch_protocol[
                N_LOCAL,
                MAX_TPR,
                1,
                1,
                fused,
                skip_a2a=True,
                use_shmem=False,
                allreduce_world_size=WORLD,
                has_input_scales=nvfp4,
                dispatch_warp_min_tokens=-1,
                p2p_small_counter_words=0,
            ](
                token_handler,
                row_offsets_tt,
                expert_ids_tt,
                src_info_tt,
                counters_tt,
                tokens_tt,
                topk_tt,
                send_tt,
                recv_tt,
                count_tt,
                ctx,
                Int32(my_rank),
                input_scales_fn,
                NullExpertLoad(),
            )

    def combine[local: Bool]() raises {imm}:
        comptime if local:
            ep_fused_combine_kernel_api[
                H,
                TOP_K,
                N_LOCAL,
                MAX_TPR,
                1,
                1,
                fused_shared_expert=fused,
                skip_a2a=True,
                allreduce_world_size=WORLD,
            ](
                n,
                counters_tt,
                expert_out_tt.as_imm(),
                src_info_tt.as_imm(),
                send_tt,
                c_recv_tt,
                c_count_tt,
                ctx,
                store_fn,
                router_weights_fn,
                topk_ids_p=topk_p,
            )
        else:
            _ep_fused_combine_protocol[
                H,
                TOP_K,
                N_LOCAL,
                MAX_TPR,
                1,
                1,
                fused_shared_expert=fused,
                skip_a2a=True,
                use_shmem=False,
                allreduce_world_size=WORLD,
                has_router_weights=True,
                p2p_small_counter_words=0,
            ](
                n,
                counters_tt,
                expert_out_tt.as_imm(),
                src_info_tt.as_imm(),
                send_tt,
                c_recv_tt,
                c_count_tt,
                ctx,
                Int32(my_rank),
                store_fn,
                router_weights_fn,
                topk_p,
            )

    def run[
        local_dispatch: Bool, local_combine: Bool
    ]() raises {imm} -> Outputs:
        ctx.enqueue_memset(rows_fp8, Scalar[FP8](0))
        ctx.enqueue_memset(rows_bf16, BFloat16(0))
        ctx.enqueue_memset(rows_fp4, UInt8(0))
        ctx.enqueue_memset(sf, bitcast[FP8, 1](SF_SENTINEL))
        ctx.enqueue_memset(sf_offsets, UInt32.MAX_FINITE)
        ctx.enqueue_memset(expert_out, BFloat16(0))
        ctx.enqueue_memset(combined, BFloat16(0))
        comptime if nvfp4:
            dispatch[local_dispatch](
                Nvfp4Fmt(rows_fp4_tt, sf_tt, sf_offsets_tt, ctx)
            )
        elif fp8:
            dispatch[local_dispatch](Fp8Fmt(rows_fp8_tt, scales_tt))
        else:
            dispatch[local_dispatch](Bf16Fmt(rows_bf16_tt))
        var o = Outputs()
        # The protocol combine clears the row ranges, so read them first.
        o.region_a = _to_list(
            ctx,
            counters.create_sub_buffer[DType.int32](
                EPLocalSyncCounters[N_LOCAL].dispatch_async_size(), 2 * N_LOCAL
            ),
            2 * N_LOCAL,
        )
        comptime if nvfp4:
            o.rows = _to_list(ctx, rows_fp4, MAX_RECV * (H // 2))
            for v in _to_list(ctx, sf, SF_BLOCKS * SF_BLOCK_BYTES):
                o.sf.append(bitcast[DType.uint8, 1](v))
            o.sf_offsets = _to_list(ctx, sf_offsets, N_LOCAL + SE)
        elif fp8:
            for v in _to_list(ctx, rows_fp8, MAX_RECV * H):
                o.rows.append(bitcast[DType.uint8, 1](v))
        else:
            for v in _to_list(ctx, rows_bf16, MAX_RECV * H):
                var two = bitcast[DType.uint8, 2](v)
                o.rows.append(two[0])
                o.rows.append(two[1])
        o.scales = _to_list(ctx, scales, MAX_RECV * (H // GROUP))
        o.row_offsets = _to_list(ctx, row_offsets, N_LOCAL + 1 + SE)
        o.expert_ids = _to_list(ctx, expert_ids, N_LOCAL + SE)
        o.src_info = _to_list(ctx, src_info, MAX_RECV * 2)
        var scales_p = scales.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        var expert_out_p = expert_out.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        comptime if nvfp4:
            ctx.enqueue_function[expert_stand_in[DType.uint8]](
                rows_fp4.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                scales_p,
                expert_out_p,
                grid_dim=MAX_RECV,
                block_dim=256,
            )
        elif fp8:
            ctx.enqueue_function[expert_stand_in[FP8]](
                rows_fp8.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                scales_p,
                expert_out_p,
                grid_dim=MAX_RECV,
                block_dim=256,
            )
        else:
            ctx.enqueue_function[expert_stand_in[DType.bfloat16]](
                rows_bf16.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                scales_p,
                expert_out_p,
                grid_dim=MAX_RECV,
                block_dim=256,
            )
        combine[local_combine]()
        for v in _to_list(ctx, combined, max(n, 1) * H):
            o.combined.append(bitcast[DType.uint16, 1](v))
        return o^

    var reference = run[False, False]()
    var reference_rows = slot_rows(reference, N_LOCAL)
    comptime n_groups = N_LOCAL + SE
    comptime if nvfp4:
        check_sf_tails(reference, n_groups, "protocol dispatch")

    def check(got: Outputs, what: String, local_dispatch: Bool) raises {imm}:
        var bad = 0
        for i in range(n * H):
            bad += Int(got.combined[i] != reference.combined[i])
        assert_equal(bad, 0, String(t"combine outputs differ: {what}"))
        if not local_dispatch:
            return
        for i in range(N_LOCAL + 1 + SE):
            assert_equal(got.row_offsets[i], reference.row_offsets[i], what)
        for i in range(N_LOCAL + SE):
            assert_equal(got.expert_ids[i], reference.expert_ids[i], what)
        for i in range(2 * N_LOCAL):
            assert_equal(got.region_a[i], reference.region_a[i], what)
        comptime if nvfp4:
            for i in range(n_groups):
                assert_equal(got.sf_offsets[i], reference.sf_offsets[i], what)
            check_sf_tails(got, n_groups, what)
        var got_rows = slot_rows(got, N_LOCAL)
        assert_equal(len(got_rows), len(reference_rows), what)
        # Each slot's row, then the shared expert's rows, which both
        # dispatches put at the same place.
        var pairs = List[Tuple[Int, Int]]()
        for entry in reference_rows.items():
            pairs.append((entry.value, got_rows[entry.key]))
        var shared_rows = Int(reference.row_offsets[1]) if fused else 0
        for r in range(shared_rows):
            pairs.append((r, r))
        bad = 0
        for rows_pair in pairs:
            var a = rows_pair[0]
            var b = rows_pair[1]
            for byte in range(row_bytes):
                bad += Int(
                    got.rows[b * row_bytes + byte]
                    != reference.rows[a * row_bytes + byte]
                )
            comptime if nvfp4:
                var sa = sf_row_of(reference, a, n_groups)
                var sb = sf_row_of(got, b, n_groups)
                for s in range(H // NV):
                    bad += Int(
                        got.sf[sf_byte(sb, s)] != reference.sf[sf_byte(sa, s)]
                    )
                    bad += Int(got.sf[sf_byte(sb, s)] == SF_SENTINEL)
            elif fp8:
                for g in range(H // GROUP):
                    bad += Int(
                        bitcast[DType.uint32, 1](got.scales[g * MAX_RECV + b])
                        != bitcast[DType.uint32, 1](
                            reference.scales[g * MAX_RECV + a]
                        )
                    )
        assert_equal(bad, 0, String(t"dispatched rows differ: {what}"))
        assert_true(len(pairs) > 0, what)

    check(run[True, True](), "local dispatch and combine", True)
    check(run[False, True](), "local combine", False)
    check(run[True, False](), "local dispatch", True)

    h_topk.unsafe_free()
    h_tokens.unsafe_free()
    h_weights.unsafe_free()
    ptrs.unsafe_free()
    c_count_ptrs.unsafe_free()
    _ = d_send^
    _ = d_recv^
    _ = d_recv_count^
    _ = c_recv^
    _ = c_recv_count^
    _ = counters^
    _ = rows_fp8^
    _ = rows_bf16^
    _ = scales^
    _ = rows_fp4^
    _ = sf^
    _ = sf_offsets^
    _ = row_offsets^
    _ = expert_ids^
    _ = src_info^
    _ = expert_out^
    _ = tokens^
    _ = topk^
    _ = weights^
    _ = combined^


def main() raises:
    var ctx = DeviceContext(device_id=0)
    # One token, a decode step, a full and a ragged chunk, and steps that
    # take the count and index launches.
    for n in [1, 5, 96, 256, 300, 1000]:
        run_case[FP8_ROWS, True](ctx, n, False)
    run_case[FP8_ROWS, True](ctx, 300, True)
    run_case[FP8_ROWS, False](ctx, 96, False)
    run_case[FP8_ROWS, False](ctx, 777, False)
    run_case[BF16_ROWS, True](ctx, 96, False)
    run_case[BF16_ROWS, True](ctx, 777, True)
    # NVFP4 at decode steps (1-32 tokens times the verify rows), a full and
    # a ragged chunk, a skewed step, and the count launch's sizes.
    for n in [1, 6, 7, 48, 96, 192, 256, 300, 1000]:
        run_case[NVFP4_ROWS, False](ctx, n, False)
    run_case[NVFP4_ROWS, False](ctx, 300, True)
    run_case[NVFP4_ROWS, True](ctx, 6, False)
    run_case[NVFP4_ROWS, True](ctx, 777, True)
