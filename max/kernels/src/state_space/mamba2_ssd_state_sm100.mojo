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

"""The chunked Mamba-2 scan's chunk state on SM100 (B200).

  S[slot, h, p, n] = sum_t exp2(cs_last - cs_t) dt_t x[t, h, p] B[t, g, n]

over each chunk slot's tokens, on the chunked scan's varlen contract
(`mamba2_ssd_chunked`): x and B are rows of the packed conv output read by
TMA at the slot's first token, dt and cs its fp32 `[H, slots, CHUNK]`
vectors, and S fp32 `[slots, H, P, N]` for the state passing. A slot's dt is
0 past its sequence's end and its cs is flat there, so cs_last is
cs[CHUNK - 1] and rows past the end weigh 0.

A persistent CTA walks work items of one slot and two heads of one group.
Per 128-token row block the load warp brings B's two 64-column N spans and
the item's x spans; the scale warps write w_t x into a second slot, row t
scaled in place in its swizzle (a 128-byte row stays one row); the MMA warp
issues S^T[n, p] += B^T (w x) with both operands MN-major (M 128 = N, N 64
= P a head, K 16 tokens an instruction); the epilogue reads S^T by TMEM
lane n and stores each p's 32 lanes as one 128-byte run.

This is the Moka component `Kernels/test/gpu/Moka/kernels/mamba2/
ssd_state_varlen_plan.mojo`, statement for statement, with no Moka in it.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.math import ceildiv, exp2
from std.utils import StaticTuple
from std.utils.coord import Idx

from layout.tma_async import SharedMemBarrier
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptor,
    UMMAInsDescriptor,
    UMMAKind,
    mma,
    mma_arrive,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
)
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import (
    TMADescriptor,
    TensorMapSwizzle,
    create_tma_descriptor,
    prefetch_tma_descriptor,
)
from max.gpu.memory import (
    AddressSpace,
    cp_async_bulk_tensor_shared_cluster_global,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.sync import barrier, named_barrier, named_barrier_arrive, syncwarp

from .mamba2_ssd_out_sm100 import (
    Bars,
    Cursor,
    HP,
    Item,
    L,
    N,
    NRB,
    P,
    ROWS,
    SMS,
    SPtr,
    VEC_BOX,
    VEC_ROW,
    VEC_SLOT,
    _arrive_warp,
    _bf16_buf,
    _f32_buf,
    _rows_desc,
    _zero_rows,
)

comptime THREADS = 384
comptime SLOT = 32768
comptime OFF_B = 0
comptime OFF_X = 65536
comptime OFF_W = 131072
comptime OFF_DT = 196608
comptime OFF_CS = 201728
comptime OFF_BARS = 206848
comptime OFF_TMEM = 207040
comptime SMEM_BYTES = 207056

comptime B_B = 0
comptime B_X = 4
comptime B_W = 8
comptime B_DC = 12
comptime B_S = 16
comptime B_TFREE = 20
comptime B_BF = 21
"""B tiles land here; the fixup then publishes them on B_B."""

comptime NB_TPTR = 1
comptime NB_FIX = 2
comptime TMEM_COLS = 256


@fieldwise_init
struct SsdStateParams(DevicePassable, ImplicitlyCopyable):
    """TMA descriptors and uniform scalars, passed as one grid constant."""

    var b_desc: TMADescriptor
    var x_desc: TMADescriptor
    var dt_desc: TMADescriptor
    var cs_desc: TMADescriptor
    var slots: Int32
    var batch: Int32
    var heads: Int32
    var heads_per_group: Int32
    var head_pairs: Int32
    var info_addr: Int64
    var states_addr: Int64

    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        return "SsdStateParams"


comptime StPtr = Pointer[SsdStateParams, ImmUntrackedOrigin]


@inline(.always)
def _desc[k: Int](p: StPtr) -> OpaquePointer[ImmUntrackedOrigin]:
    comptime if k == 0:
        return Pointer(to=p[].b_desc).unsafe_bitcast[NoneType]()
    elif k == 1:
        return Pointer(to=p[].x_desc).unsafe_bitcast[NoneType]()
    elif k == 2:
        return Pointer(to=p[].dt_desc).unsafe_bitcast[NoneType]()
    return Pointer(to=p[].cs_desc).unsafe_bitcast[NoneType]()


@inline(.always)
def _info(p: StPtr, k: Int) -> Int:
    return Int(
        Pointer[Int32, ImmUntrackedOrigin](
            unsafe_from_address=Int(p[].info_addr)
        )[unsafe_offset=k]
    )


@inline(.always)
def _item(p: StPtr, it: Int) -> Item:
    var nhp = UInt32(p[].head_pairs)
    var q = UInt32(it) // nhp
    return Item(Int(q), Int(UInt32(it) - q * nhp))


@inline(.always)
def _items_of_cta(p: StPtr) -> Int:
    var slots = Int(p[].slots)
    var nb = Int(p[].batch)
    var used = _info(p, 3 * slots + nb - 1) + _info(p, 3 * slots + 2 * nb - 1)
    var n = UInt32(min(used, slots) * Int(p[].head_pairs))
    var g = UInt32(grid_dim.x)
    var c = UInt32(block_idx.x)
    return Int((n - c + g - 1) // g) if c < n else 0


@inline(.always)
def _vec_start(p: StPtr, w: Item, hh: Int) -> Int:
    return (w.hp * HP + hh) * (Int(p[].slots) * L) + w.slot * L


@inline(.always)
def _first_token(p: StPtr, slot: Int) -> Int:
    if _info(p, Int(p[].slots) + slot) <= 0:
        return 0
    return _info(p, slot)


@inline(.always)
def _load(p: StPtr, smem: SPtr[UInt8], bars: Bars, T: Int):
    var c_b = Cursor.producer()
    var c_x = Cursor.producer()
    var c_dc = Cursor.producer()
    var cta = Int(block_idx.x)
    var grid = Int(grid_dim.x)
    for k in range(T):
        var w = _item(p, cta + k * grid)
        var t0 = _first_token(p, w.slot)
        var g = Int(UInt32(w.hp * HP) // UInt32(p[].heads_per_group))
        var d = bars.wait_empty[B_DC, 2](c_dc)
        fence_async_view_proxy()
        if elect_one_sync():
            var bar = bars.full[B_DC](d)
            bar[].expect_bytes(Int32(2 * HP * VEC_ROW * 4))
            comptime for v in range(2):
                var dst = smem.unsafe_offset(
                    (OFF_DT if v == 0 else OFF_CS) + Int(d) * VEC_SLOT
                )
                comptime for hh in range(HP):
                    var e = _vec_start(p, w, hh)
                    var e0 = e - Int(UInt32(e) % 4)
                    comptime for kk in range(VEC_ROW // VEC_BOX):
                        cp_async_bulk_tensor_shared_cluster_global(
                            dst.unsafe_offset(
                                (hh * VEC_ROW + kk * VEC_BOX) * 4
                            ),
                            _desc[2 + v](p),
                            bar,
                            (e0 + kk * VEC_BOX,),
                        )
        for rb in range(NRB):
            var row = t0 + rb * ROWS
            var xs = bars.wait_empty[B_X, 2](c_x)
            fence_async_view_proxy()
            if elect_one_sync():
                var bar = bars.full[B_X](xs)
                bar[].expect_bytes(Int32(SLOT))
                var dst = smem.unsafe_offset(OFF_X + Int(xs) * SLOT)
                comptime for z in range(HP):
                    cp_async_bulk_tensor_shared_cluster_global(
                        dst.unsafe_offset(16384 * z),
                        _desc[1](p),
                        bar,
                        ((w.hp * HP + z) * P, row, 0),
                    )
            var bs = bars.wait_empty[B_B, 2](c_b)
            if elect_one_sync():
                # B lands on B_BF for the fixup, which publishes it on B_B
                var bar = bars.full[B_BF](bs)
                bar[].expect_bytes(Int32(SLOT))
                var dst = smem.unsafe_offset(OFF_B + Int(bs) * SLOT)
                comptime for z in range(2):
                    cp_async_bulk_tensor_shared_cluster_global(
                        dst.unsafe_offset(16384 * z),
                        _desc[0](p),
                        bar,
                        (g * N + z * 64, row, 0),
                    )
    comptime for _ in range(2):
        _ = bars.wait_empty[B_B, 2](c_b)
    comptime for _ in range(2):
        _ = bars.wait_empty[B_X, 2](c_x)
    comptime for _ in range(2):
        _ = bars.wait_empty[B_DC, 2](c_dc)


@inline(.always)
def _scale(p: StPtr, smem: SPtr[UInt8], bars: Bars, T: Int, t: Int):
    """Scales x by w_t for each row block: row t of each head's 64-column
    span, times exp2(cs[L - 1] - cs_t) dt_t, into the same row of the scaled
    slot. Rows past the sequence's end are written as zeros (selected, not
    multiplied by a zero weight, so a non-finite x there stays out)."""
    var c_dc = Cursor.consumer()
    var c_x = Cursor.consumer()
    var c_w = Cursor.producer()
    var cta = Int(block_idx.x)
    var grid = Int(grid_dim.x)
    for k in range(T):
        var w = _item(p, cta + k * grid)
        var slen = _info(p, Int(p[].slots) + w.slot)
        var d = bars.wait_full[B_DC, 2](c_dc)
        var dt0 = smem.unsafe_offset(OFF_DT + Int(d) * VEC_SLOT).unsafe_bitcast[
            Float32
        ]()
        var cs0 = smem.unsafe_offset(OFF_CS + Int(d) * VEC_SLOT).unsafe_bitcast[
            Float32
        ]()
        for rb in range(NRB):
            var xs = bars.wait_full[B_X, 2](c_x)
            var ws = bars.wait_empty[B_W, 2](c_w)
            comptime for hh in range(HP):
                var e = _vec_start(p, w, hh)
                var sk = Int(UInt32(e) % 4)
                var cs = cs0.unsafe_offset(hh * VEC_ROW + sk)
                var dt = dt0.unsafe_offset(hh * VEC_ROW + sk)
                var j = rb * ROWS + t
                var f = (
                    exp2(cs[unsafe_offset=L - 1] - cs[unsafe_offset=j])
                    * dt[unsafe_offset=j]
                )
                var src = (
                    smem.unsafe_offset(OFF_X + Int(xs) * SLOT + 16384 * hh)
                    .unsafe_bitcast[BFloat16]()
                    .unsafe_offset(t * 64)
                )
                var dst = (
                    smem.unsafe_offset(OFF_W + Int(ws) * SLOT + 16384 * hh)
                    .unsafe_bitcast[BFloat16]()
                    .unsafe_offset(t * 64)
                )
                comptime for q in range(8):
                    var v = src.unsafe_load[width=8, alignment=16](8 * q)
                    var o = (v.cast[DType.float32]() * f).cast[DType.bfloat16]()
                    if j >= slen:
                        o = SIMD[DType.bfloat16, 8](0)
                    dst.unsafe_store[alignment=16](8 * q, o)
            _arrive_warp(bars.empty[B_X, 2](xs))
            fence_async_view_proxy()
            _arrive_warp(bars.full[B_W](ws))
        _arrive_warp(bars.empty[B_DC, 2](d))


@inline(.always)
def _fix(p: StPtr, smem: SPtr[UInt8], bars: Bars, T: Int, ft: Int):
    """Publishes each B tile on B_B, after zeroing its rows past the
    sequence's end inside the last 16-token step the MMA reads (it skips the
    steps past that).

    Those rows belong to the next sequence. Their w x is 0, but 0 * a
    non-finite B is NaN. The tiles land on B_BF, which only these two warps
    wait on, in order.
    """
    var c_b = Cursor.consumer()
    var cta = Int(block_idx.x)
    var grid = Int(grid_dim.x)
    for k in range(T):
        var slen = _info(p, Int(p[].slots) + _item(p, cta + k * grid).slot)
        for rb in range(NRB):
            var bs = bars.wait_full[B_BF, 2](c_b)
            var live = slen - rb * ROWS
            if live > 0 and live < ROWS and live % 16 != 0:
                _zero_rows[64](
                    smem.unsafe_offset(OFF_B + Int(bs) * SLOT),
                    live,
                    ceildiv(live, 16) * 16,
                    ft,
                )
                fence_async_view_proxy()
            named_barrier[Int32(64)](Int32(NB_FIX))
            if ft == 0:
                _ = bars.full[B_B](bs)[].arrive()


@inline(.always)
def _umma_st(
    d: UInt32, b_lo: SPtr[UInt8], w_lo: SPtr[UInt8], rb: Int, ksteps: Int
):
    """Accumulates S^T += B^T (w x) over 128 tokens, each head into its 64
    columns: M 128 (n over B's two spans), N 64 (p), K 16 tokens an
    instruction, both operands MN-major; row block 0 overwrites. Only the
    first `ksteps` steps run: the later ones hold only tokens past the
    sequence's end, whose w x is 0."""
    comptime idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        DType.float32,
        DType.bfloat16,
        DType.bfloat16,
        (UInt32(N), UInt32(P)),
        transpose_a=True,
        transpose_b=False,
    ]()
    var bd = MMASmemDescriptor.create[
        1024, 16384, TensorMapSwizzle.SWIZZLE_128B
    ](b_lo)
    comptime for hh in range(HP):
        var xd = MMASmemDescriptor.create[
            1024, 16384, TensorMapSwizzle.SWIZZLE_128B
        ](w_lo.unsafe_offset(16384 * hh))
        comptime for j in range(ROWS // 16):
            if j < ksteps:
                var acc = UInt32(1)
                comptime if j == 0:
                    acc = UInt32(0) if rb == 0 else UInt32(1)
                mma(
                    bd + j * 16 * 128,
                    xd + j * 16 * 128,
                    d + UInt32(P * hh),
                    idesc,
                    acc,
                )


@inline(.always)
def _mma(p: StPtr, smem: SPtr[UInt8], bars: Bars, T: Int):
    var slot_ptr = smem.unsafe_offset(OFF_TMEM).unsafe_bitcast[UInt32]()
    tcgen05_alloc[1](slot_ptr, UInt32(TMEM_COLS))
    syncwarp()
    var tmem = slot_ptr[]
    named_barrier_arrive[Int32(32 + 128)](Int32(NB_TPTR))
    var c_s = Cursor.producer()
    var c_b = Cursor.consumer()
    var c_w = Cursor.consumer()
    var cta = Int(block_idx.x)
    var grid = Int(grid_dim.x)
    for k in range(T):
        var slen = _info(p, Int(p[].slots) + _item(p, cta + k * grid).slot)
        var s = bars.wait_empty[B_S, 2](c_s)
        tcgen05_fence_after()
        for rb in range(NRB):
            # row block 0 always holds a live token, so it still overwrites
            var ks = min(ceildiv(max(slen - rb * ROWS, 0), 16), ROWS // 16)
            var bs = bars.wait_full[B_B, 2](c_b)
            var ws = bars.wait_full[B_W, 2](c_w)
            if elect_one_sync():
                _umma_st(
                    tmem + UInt32(64 * HP * Int(s)),
                    smem.unsafe_offset(OFF_B + Int(bs) * SLOT),
                    smem.unsafe_offset(OFF_W + Int(ws) * SLOT),
                    rb,
                    ks,
                )
                mma_arrive[1](bars.empty[B_B, 2](bs))
                mma_arrive[1](bars.empty[B_W, 2](ws))
        if elect_one_sync():
            mma_arrive[1](bars.full[B_S](s))
    tcgen05_release_allocation_lock[1]()
    bars.full[B_TFREE](0)[].wait(0)
    tcgen05_fence_after()
    tcgen05_dealloc[1](tmem, UInt32(TMEM_COLS))


@inline(.always)
def _epi(p: StPtr, smem: SPtr[UInt8], bars: Bars, T: Int, n: Int):
    """Stores S^T (TMEM lane n, column 64 h + p) to S[slot, h, p, n] fp32."""
    named_barrier[Int32(32 + 128)](Int32(NB_TPTR))
    var tmem = smem.unsafe_offset(OFF_TMEM).unsafe_bitcast[UInt32]()[]
    var c_s = Cursor.consumer()
    var cta = Int(block_idx.x)
    var grid = Int(grid_dim.x)
    var nh = Int(p[].heads)
    var sp = Pointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(p[].states_addr)
    )
    for k in range(T):
        var w = _item(p, cta + k * grid)
        var s = bars.wait_full[B_S, 2](c_s)
        tcgen05_fence_after()
        var acc = tmem + UInt32(64 * HP * Int(s))
        comptime for hh in range(HP):
            comptime for half in range(2):
                var r = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=32,
                    dtype=DType.float32,
                    pack=False,
                ](acc + UInt32(P * hh + 32 * half))
                tcgen05_load_wait()
                var h = w.hp * HP + hh
                var base = ((w.slot * nh + h) * P + 32 * half) * N + n
                comptime for q in range(32):
                    sp[unsafe_offset=base + q * N] = r[q]
        tcgen05_fence_before()
        _arrive_warp(bars.empty[B_S, 2](s))
    tcgen05_fence_before()
    _ = bars.full[B_TFREE](0)[].arrive()


@__llvm_arg_metadata(params, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(THREADS))
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
def ssd_state_sm100_kernel(params: SsdStateParams):
    """Grid min(SMS, items), block THREADS: warps 0-3 scale, 4-7 epilogue, 8
    mma, 9 load, 10-11 B fixup."""
    var smem = external_memory[
        UInt8, address_space=AddressSpace.SHARED, alignment=1024
    ]()
    var p = Pointer(to=params).unsafe_origin_cast[ImmUntrackedOrigin]()
    var bars = Bars(smem, OFF_BARS)
    var tid = Int(thread_idx.x)
    if tid == 0:
        comptime for g in range(4):
            prefetch_tma_descriptor(_desc[g](p))
        bars.init_ring[B_B, 2, 1, 1]()
        bars.init_ring[B_X, 2, 1, 4]()
        bars.init_ring[B_W, 2, 4, 1]()
        bars.init_ring[B_DC, 2, 1, 4]()
        bars.init_ring[B_S, 2, 1, 4]()
        bars.init_ring[B_TFREE, 1, 128, 0]()
        bars.init_ring[B_BF, 2, 1, 0]()
        fence_mbarrier_init()
    var T = _items_of_cta(p)
    barrier()
    var w = tid // WARP_SIZE
    if w < 4:
        _scale(p, smem, bars, T, tid)
    elif w < 8:
        _epi(p, smem, bars, T, tid - 128)
    elif w == 8:
        _mma(p, smem, bars, T)
    elif w == 9:
        _load(p, smem, bars, T)
    else:
        _fix(p, smem, bars, T, tid - 320)


def ssd_state_sm100(
    ctx: DeviceContext,
    total_len: Int,
    batch: Int,
    slots: Int,
    nheads: Int,
    ngroups: Int,
    x_addr: Int,
    b_addr: Int,
    x_ld: Int,
    bc_ld: Int,
    dt_addr: Int,
    cs_addr: Int,
    states_addr: Int,
    info_addr: Int,
) raises:
    """Computes the chunk state of every used chunk slot and head.

    Args:
        ctx: The device context.
        total_len: Tokens of the packed batch.
        batch: Sequences.
        slots: Chunk slots of the table (`chunked_slots`).
        nheads: Heads.
        ngroups: B groups.
        x_addr: The first element of x, [T, H, P] bf16 with rows `x_ld`
            apart.
        b_addr: The first element of B, [T, G, N] bf16 with rows `bc_ld`
            apart.
        x_ld: Row stride of x in elements (a multiple of 8).
        bc_ld: Row stride of B in elements (a multiple of 8).
        dt_addr: The softplus dt, fp32 [H, slots, CHUNK].
        cs_addr: The log2 decay cumsum, fp32 [H, slots, CHUNK].
        states_addr: The output, fp32 [slots, H, P, N].
        info_addr: The chunk table, int32 [3 slots + 2 batch].
    """
    var nv = nheads * slots * L
    var params = SsdStateParams(
        _rows_desc(
            _bf16_buf(ctx, b_addr, total_len * bc_ld),
            total_len,
            ngroups * N,
            bc_ld,
        ),
        _rows_desc(
            _bf16_buf(ctx, x_addr, total_len * x_ld),
            total_len,
            nheads * P,
            x_ld,
        ),
        create_tma_descriptor[DType.float32](
            _f32_buf(ctx, dt_addr, nv), (nv,), (Idx[1],), (Idx[VEC_BOX],)
        ),
        create_tma_descriptor[DType.float32](
            _f32_buf(ctx, cs_addr, nv), (nv,), (Idx[1],), (Idx[VEC_BOX],)
        ),
        Int32(slots),
        Int32(batch),
        Int32(nheads),
        Int32(nheads // ngroups),
        Int32(nheads // HP),
        Int64(info_addr),
        Int64(states_addr),
    )
    var items = slots * (nheads // HP)
    ctx.enqueue_function[ssd_state_sm100_kernel](
        params,
        grid_dim=min(SMS, items),
        block_dim=THREADS,
        shared_mem_bytes=SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(SMEM_BYTES)
        ),
    )
