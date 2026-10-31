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

"""The chunked Mamba-2 scan's chunk state on MI355 (gfx950).

  S[slot, h, p, n] = sum_t exp2(cs_last - cs_t) dt_t x[t, h, p] B[t, g, n]

over each chunk slot's tokens, on the chunked scan's varlen contract
(`mamba2_ssd_chunked`): x and B rows of the packed conv output at the
slot's first token, fp32 dt and log2 cumsum `[H, slots, CHUNK]`, and S fp32
`[slots, H, P, N]` for the state passing. A slot's dt is 0 past its
sequence's end and its cs is flat there, so cs_last is cs[CHUNK - 1].

A persistent workgroup walks work items of the used slots, a slot and two
heads of one group. Two load waves write each head's weights w_t into LDS
and stream 32-token key blocks (B's rows and both heads' x rows; rows past
the slot's end are zeros) through a 4-deep LDS ring. Eight MFMA waves each
take one head and two 16-wide N tiles over all of P, accumulating
S^T[n, p] += B^T (w x) with 16x16x32 bf16 MFMAs whose operands are read
token-major with `ds_read_tr16_b64`; w scales x's fragments in fp32 before
they round to bf16. A lane stores 16 contiguous bytes of S.

This is the Moka component `Kernels/test/gpu/Moka/amd/moka_amd/kernels/
ssd_varlen.mojo` (`ssd_state_amd_plan`), statement for statement, with no
Moka in it.
"""

from std.math import exp2
from std.utils import StaticTuple

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from max.gpu.compute.mma import mma
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace, external_memory

from .mamba2_ssd_out_mi355 import (
    GROUP,
    KB,
    KEY_DEPTH,
    L,
    LOAD_THREADS,
    LOAD_WAVES,
    MMA_WAVES,
    N,
    NKB,
    OX_PITCH,
    P,
    THREADS,
    Acc,
    F4,
    Frag,
    GMutPtr,
    GPtr,
    Item,
    Lds,
    Ring,
    SsdMi355Params,
    Table,
    lds_at,
    load_keys,
    resident_workgroups,
    tr_frag,
    zero_words,
)

# B and x are both read transposed: 288 = 72 words keeps B's 4 rows 8
# banks apart.
comptime SB_PITCH = 2 * N + 32
comptime SX_PITCH = OX_PITCH
comptime SX_OFF = KB * SB_PITCH
comptime SKEY_BYTES = SX_OFF + 2 * KB * SX_PITCH
comptime W_BYTES = 2 * L * 4
comptime W_DEPTH = 2

comptime OFF_KEYS = 0
comptime OFF_W = OFF_KEYS + KEY_DEPTH * SKEY_BYTES
comptime OFF_END = OFF_W + W_DEPTH * W_BYTES
comptime W_KEYS = 0
comptime W_W = W_KEYS + 2 * KEY_DEPTH
comptime NWORDS = W_W + 2 * W_DEPTH
comptime STATE_LDS = OFF_END + 4 * NWORDS


@always_inline
def _st_vec(p: SsdMi355Params, it: Item, sw_t: Lds):
    """Each head's per-token weight w_t = exp2(cs_last - cs_t) dt_t, [2][L]
    fp32."""
    var sw = sw_t.unsafe_bitcast[Float32]()
    var dt = GPtr[Float32](unsafe_from_address=Int(p.dt))
    var cs = GPtr[Float32](unsafe_from_address=Int(p.cs))
    var slots = Int(p.slots)
    comptime for k in range(2 * L // LOAD_THREADS):
        var e = Int(thread_idx.x) + k * LOAD_THREADS
        var hh, j = divmod(e, L)
        var v = ((it.h0 + hh) * slots + it.slot) * L
        var w = exp2(cs[unsafe_offset=v + L - 1] - cs[unsafe_offset=v + j])
        w *= dt[unsafe_offset=v + j]
        sw[unsafe_offset=hh * L + j] = w


@always_inline
def _st_load(p: SsdMi355Params, lds: Lds, mut tab: Table):
    var words = lds_at[UInt32](lds, OFF_END)
    var wts = Ring[W_DEPTH, LOAD_WAVES, MMA_WAVES](words, W_W)
    var keys = Ring[KEY_DEPTH, LOAD_WAVES, MMA_WAVES](words, W_KEYS)
    var T = tab.items(Int(p.batch))
    for k in range(T):
        var it = tab.fetch(Int(block_idx.x) + k * Int(grid_dim.x))
        var v = wts.acquire()
        if it.len > 0:
            _st_vec(p, it, lds.unsafe_offset(OFF_W + v * W_BYTES))
        wts.commit()
        for kb in range(NKB):
            var s = keys.acquire()
            var t0 = kb * KB
            if t0 < it.len:
                load_keys[SB_PITCH, SX_PITCH](
                    p,
                    it,
                    t0,
                    it.len - t0,
                    lds.unsafe_offset(OFF_KEYS + s * SKEY_BYTES),
                )
            keys.commit()


@always_inline
def _st_step(
    wave: Int, kb: Int, slot: Lds, sw: Lds, mut acc: StaticTuple[Acc, 8]
):
    """S^T[n, p] += B^T (w X) over key block `kb`, the wave's two N tiles of
    its head, w applied to X's fragments in fp32 and rounded to bf16."""
    var q = lane_id() // 16
    var hh = wave // 4
    var nt0 = 2 * (wave % 4)
    comptime for ks in range(KB // GROUP):
        comptime t = GROUP * ks
        var a = StaticTuple[Frag, 2]()
        comptime for nt in range(2):
            a[nt] = tr_frag(slot, t, t + 16, SB_PITCH, 16 * (nt0 + nt))
        # the weights of the lane's 8 tokens, in the fragments' order
        var wo = hh * L + kb * KB + t + 4 * q
        var w = lds_at[F4](sw, 4 * wo)[].join(lds_at[F4](sw, 4 * (wo + 16))[])
        comptime for pt in range(4):
            var b = (
                tr_frag(
                    slot.unsafe_offset(SX_OFF + hh * KB * SX_PITCH),
                    t,
                    t + 16,
                    SX_PITCH,
                    16 * pt,
                ).cast[DType.float32]()
                * w
            ).cast[DType.bfloat16]()
            comptime for nt in range(2):
                mma(acc[4 * nt + pt], a[nt], b, acc[4 * nt + pt])


@always_inline
def _st_store(p: SsdMi355Params, it: Item, wave: Int, acc: StaticTuple[Acc, 8]):
    """The wave's part of S[slot, h, p, n] in fp32: a lane holds four
    consecutive n of one p."""
    var l = lane_id()
    var s, q = l % 16, l // 16
    var hh = wave // 4
    var nt0 = 2 * (wave % 4)
    var out = GMutPtr[F4](unsafe_from_address=Int(p.out))
    var base = (it.slot * Int(p.heads) + it.h0 + hh) * P
    comptime for nt in range(2):
        comptime for pt in range(4):
            var pp = 16 * pt + s
            var n = 16 * (nt0 + nt) + 4 * q
            out[unsafe_offset=((base + pp) * N + n) // 4] = acc[4 * nt + pt]


@always_inline
def _st_mma(p: SsdMi355Params, lds: Lds, mut tab: Table, wave: Int):
    var words = lds_at[UInt32](lds, OFF_END)
    var wts = Ring[W_DEPTH, LOAD_WAVES, MMA_WAVES](words, W_W)
    var keys = Ring[KEY_DEPTH, LOAD_WAVES, MMA_WAVES](words, W_KEYS)
    var T = tab.items(Int(p.batch))
    for k in range(T):
        var it = tab.fetch(Int(block_idx.x) + k * Int(grid_dim.x))
        var v = wts.wait()
        var acc = StaticTuple[Acc, 8](fill=Acc(0))
        for kb in range(NKB):
            var s = keys.wait()
            if kb * KB < it.len:
                _st_step(
                    wave,
                    kb,
                    lds.unsafe_offset(OFF_KEYS + s * SKEY_BYTES),
                    lds.unsafe_offset(OFF_W + v * W_BYTES),
                    acc,
                )
            keys.release()
        wts.release()
        if it.len > 0:
            _st_store(p, it, wave, acc)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(THREADS))
)
def ssd_state_mi355_kernel(params: SsdMi355Params):
    """Persistent: workgroup c takes items c, c + grid, ... Waves 0-1 load,
    2-9 compute."""
    var lds = external_memory[
        UInt8, address_space=AddressSpace.SHARED, alignment=128
    ]()
    zero_words(lds_at[UInt32](lds, OFF_END), NWORDS)
    var tab = Table(params, 1)
    var wave = Int(thread_idx.x) // WARP_SIZE
    if wave < LOAD_WAVES:
        _st_load(params, lds, tab)
    else:
        _st_mma(params, lds, tab, wave - LOAD_WAVES)


def ssd_state_mi355(
    ctx: DeviceContext,
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
    """S for every used chunk slot and head into `states` ([slots, H, P, N]
    fp32). x and B start at their first element, rows `x_ld` and `bc_ld`
    apart."""
    var params = SsdMi355Params(
        Int64(slots),
        Int64(nheads),
        Int64(nheads // ngroups),
        Int64(batch),
        Int64(x_ld),
        Int64(bc_ld),
        Int64(x_addr),
        Int64(b_addr),
        0,
        0,
        Int64(dt_addr),
        Int64(cs_addr),
        0,
        Int64(states_addr),
        Int64(info_addr),
    )
    var grid = min(
        slots * (nheads // 2),
        resident_workgroups[ssd_state_mi355_kernel](ctx, STATE_LDS),
    )
    ctx.enqueue_function[ssd_state_mi355_kernel](
        params, grid_dim=grid, block_dim=THREADS, shared_mem_bytes=STATE_LDS
    )
