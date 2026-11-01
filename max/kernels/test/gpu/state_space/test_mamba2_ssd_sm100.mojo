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

"""The sm100 chunk state and output kernels against fp32 references.

Each case packs ragged sequences the way the chunked scan does: x, B and C
side by side in one [T, ld] tensor (the conv output), a chunk table with
every slot past the last used one empty, and per-slot dt and log2 decay
cumsums that restart at each slot. Sequences of one token, of a partial
chunk, of exact chunks, of a chunk plus one, and long ones; x/B/C rows wider
than their columns; D present and absent. Both the TileGym decay range (a
row sees only its last few keys) and a mild one (every key block, the carry
and the factored decay move the output) run. Outputs start as NaN, so an
unwritten row fails, and a guard past y's end must stay NaN.
"""

from std.math import ceildiv, cos, exp, exp2, log, sqrt
from std.utils.numerics import nan

from max.gpu import global_idx
from std.testing import assert_equal, assert_true
from max.gpu.host import DeviceBuffer, DeviceContext
from state_space.mamba2_ssd_out_sm100 import ssd_out_sm100
from state_space.mamba2_ssd_state_sm100 import ssd_state_sm100

comptime L = 256
comptime P = 64
comptime N = 128
comptime TOL = Float64(0.01)
"""Linf-relative bound: bf16 rounding of the output, of the decayed scores
and of the states."""
comptime LOG2E = Float32(1.4426950408889634)
comptime GUARD = 64


struct Rng:
    var s: UInt64

    def __init__(out self, seed: Int):
        self.s = UInt64(seed) * 0x9E3779B97F4A7C15 + 1

    def next(mut self) -> UInt64:
        self.s += 0x9E3779B97F4A7C15
        var z = self.s
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def uniform(mut self) -> Float32:
        return Float32(self.next() >> 40) * Float32(1.0 / 16777216.0)

    def normal(mut self) -> Float32:
        var u1 = max(self.uniform(), Float32(1e-7))
        var u2 = self.uniform()
        return sqrt(-2.0 * log(u1)) * cos(Float32(6.283185307179586) * u2)


def silu(x: Float32) -> Float32:
    return x / (1.0 + exp(-x))


def softplus(x: Float32) -> Float32:
    return log(1.0 + exp(x)) if x < 20.0 else x


def ref_state_kernel(
    xbc: Pointer[BFloat16, MutAnyOrigin],
    dt: Pointer[Float32, MutAnyOrigin],
    cs: Pointer[Float32, MutAnyOrigin],
    info: Pointer[Int32, MutAnyOrigin],
    st: Pointer[Float32, MutAnyOrigin],
    nh: Int32,
    g: Int32,
    slots: Int32,
    ld: Int32,
):
    """S[slot, h, p, n] = sum_t exp2(cs_last - cs_t) dt_t x[t, p] B[t, n]."""
    var n = Int(global_idx.x) % N
    var p = Int(global_idx.x) // N
    var hi = Int(global_idx.y)
    var slot = Int(global_idx.z)
    var H = Int(nh)
    var S = Int(slots)
    if p >= P or hi >= H or slot >= S:
        return
    var length = Int(info[unsafe_offset=S + slot])
    var start = Int(info[unsafe_offset=slot])
    var gi = hi // (H // Int(g))
    var vrow = (hi * S + slot) * L
    var acc = Float32(0)
    if length > 0:
        var last = cs[unsafe_offset=vrow + length - 1]
        for t in range(length):
            var w = (
                exp2(last - cs[unsafe_offset=vrow + t])
                * dt[unsafe_offset=vrow + t]
            )
            var row = (start + t) * Int(ld)
            acc += (
                w
                * xbc[unsafe_offset=row + hi * P + p].cast[DType.float32]()
                * xbc[unsafe_offset=row + H * P + gi * N + n].cast[
                    DType.float32
                ]()
            )
    st[unsafe_offset=((slot * H + hi) * P + p) * N + n] = acc


def ref_out_kernel(
    xbc: Pointer[BFloat16, MutAnyOrigin],
    d: Pointer[BFloat16, MutAnyOrigin],
    dt: Pointer[Float32, MutAnyOrigin],
    cs: Pointer[Float32, MutAnyOrigin],
    h: Pointer[BFloat16, MutAnyOrigin],
    info: Pointer[Int32, MutAnyOrigin],
    tok_slot: Pointer[Int32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    t: Int32,
    nh: Int32,
    g: Int32,
    slots: Int32,
    ld: Int32,
    has_d: Int32,
):
    """Y[t, h, p] = exp2(cs_i) C_i h_p + sum_{j <= i} (C_i . B_j)
    exp2(cs_i - cs_j) dt_j x_j + D x_i over the slot holding token t."""
    var p = Int(global_idx.x) % P
    var hi = Int(global_idx.x) // P
    var ti = Int(global_idx.y)
    if hi >= Int(nh) or ti >= Int(t):
        return
    var H = Int(nh)
    var S = Int(slots)
    var LD = Int(ld)
    var gi = hi // (H // Int(g))
    var slot = Int(tok_slot[unsafe_offset=ti])
    var start = Int(info[unsafe_offset=slot])
    var i = ti - start
    var vrow = (hi * S + slot) * L
    var csi = cs[unsafe_offset=vrow + i]
    var cr = xbc.unsafe_offset(ti * LD + H * P + Int(g) * N + gi * N)
    var hr = h.unsafe_offset(((slot * H + hi) * P + p) * N)
    var carry = Float32(0)
    for n in range(N):
        carry += (
            cr[unsafe_offset=n].cast[DType.float32]()
            * hr[unsafe_offset=n].cast[DType.float32]()
        )
    var acc = carry * exp2(csi)
    for j in range(i + 1):
        var tj = start + j
        var br = xbc.unsafe_offset(tj * LD + H * P + gi * N)
        var cb = Float32(0)
        for n in range(N):
            cb += (
                cr[unsafe_offset=n].cast[DType.float32]()
                * br[unsafe_offset=n].cast[DType.float32]()
            )
        var w = (
            exp2(csi - cs[unsafe_offset=vrow + j]) * dt[unsafe_offset=vrow + j]
        )
        acc += (
            cb
            * w
            * xbc[unsafe_offset=tj * LD + hi * P + p].cast[DType.float32]()
        )
    if has_d != 0:
        acc += (
            d[unsafe_offset=hi].cast[DType.float32]()
            * xbc[unsafe_offset=ti * LD + hi * P + p].cast[DType.float32]()
        )
    y[unsafe_offset=(ti * H + hi) * P + p] = acc


@fieldwise_init
struct Case(Movable):
    var t: Int
    var batch: Int
    var slots: Int
    var ld: Int
    var xbc: DeviceBuffer[DType.bfloat16]
    var d: DeviceBuffer[DType.bfloat16]
    var dt: DeviceBuffer[DType.float32]
    var cs: DeviceBuffer[DType.float32]
    var h: DeviceBuffer[DType.bfloat16]
    var info: DeviceBuffer[DType.int32]
    var tok_slot: DeviceBuffer[DType.int32]


comptime ROWS_PAD = 256


def make_case(
    ctx: DeviceContext,
    lens: List[Int],
    nh: Int,
    g: Int,
    mild: Bool,
    seed: Int,
    pad_cols: Int,
    poison_rows: Int = 0,
) raises -> Case:
    var r = Rng(seed)
    var t = 0
    for n in lens:
        t += n
    var batch = len(lens)
    var slots = ceildiv(t, L) + batch
    var ld = nh * P + 2 * g * N + pad_cols
    # `poison_rows` NaN rows follow the last sequence, inside the rows the
    # kernels are told about: a partial last row block must not read them.
    var nx = (t + poison_rows) * ld
    var nv = nh * slots * L
    var nst = slots * nh * P * N
    var ninfo = 3 * slots + 2 * batch
    var hx = ctx.enqueue_create_host_buffer[DType.bfloat16](max(nx, 1))
    var hd = ctx.enqueue_create_host_buffer[DType.bfloat16](nh)
    var hdt = ctx.enqueue_create_host_buffer[DType.float32](nv)
    var hcs = ctx.enqueue_create_host_buffer[DType.float32](nv)
    var hh = ctx.enqueue_create_host_buffer[DType.bfloat16](nst)
    var hinfo = ctx.enqueue_create_host_buffer[DType.int32](ninfo)
    var hts = ctx.enqueue_create_host_buffer[DType.int32](max(t, 1))
    ctx.synchronize()
    for i in range(nx):
        hx[i] = silu(r.normal()).cast[DType.bfloat16]()
        if i >= t * ld:
            hx[i] = nan[DType.bfloat16]()
    for i in range(nh):
        hd[i] = r.normal().cast[DType.bfloat16]()
    var a = List[Float32]()
    for _ in range(nh):
        var u = r.uniform()
        a.append(-(0.001 + 0.019 * u) if mild else -(1.0 + 15.0 * u))
    for i in range(nst):
        hh[i] = (0.5 * r.normal()).cast[DType.bfloat16]()
    for i in range(nv):
        hdt[i] = 0
        hcs[i] = 0
    for i in range(ninfo):
        hinfo[i] = 0
    var next = 0
    var tok = 0
    for b in range(batch):
        var n = lens[b]
        var nc = ceildiv(n, L)
        hinfo[3 * slots + b] = Int32(next)
        hinfo[3 * slots + batch + b] = Int32(nc)
        for c in range(nc):
            var st = tok + c * L
            var cl = min(L, n - c * L)
            hinfo[next] = Int32(st)
            hinfo[slots + next] = Int32(cl)
            hinfo[2 * slots + next] = Int32(b)
            for i in range(cl):
                hts[st + i] = Int32(next)
            for hi in range(nh):
                var acc = Float32(0)
                for i in range(L):
                    var v = Float32(0)
                    if i < cl:
                        v = softplus(r.normal())
                    var o = (hi * slots + next) * L + i
                    hdt[o] = v
                    acc += a[hi] * v
                    hcs[o] = acc * LOG2E
            next += 1
        tok += n
    var xbc = ctx.enqueue_create_buffer[DType.bfloat16](max(nx, 1))
    var d = ctx.enqueue_create_buffer[DType.bfloat16](nh)
    var dt = ctx.enqueue_create_buffer[DType.float32](nv)
    var cs = ctx.enqueue_create_buffer[DType.float32](nv)
    var h = ctx.enqueue_create_buffer[DType.bfloat16](nst)
    var info = ctx.enqueue_create_buffer[DType.int32](ninfo)
    var tok_slot = ctx.enqueue_create_buffer[DType.int32](max(t, 1))
    ctx.enqueue_copy(xbc, hx)
    ctx.enqueue_copy(d, hd)
    ctx.enqueue_copy(dt, hdt)
    ctx.enqueue_copy(cs, hcs)
    ctx.enqueue_copy(h, hh)
    ctx.enqueue_copy(info, hinfo)
    ctx.enqueue_copy(tok_slot, hts)
    ctx.synchronize()
    return Case(t, batch, slots, ld, xbc^, d^, dt^, cs^, h^, info^, tok_slot^)


def linf_rel(
    ctx: DeviceContext,
    got: DeviceBuffer[DType.float32],
    want: DeviceBuffer[DType.float32],
    n: Int,
    skip_per: Int = 0,
    used: List[Bool] = List[Bool](),
) raises -> Tuple[Float64, Int]:
    """The Linf-relative error and the non-finite count over the first `n`
    values; with `skip_per`, block k of that many values is checked only if
    `used[k]`."""
    var hg = ctx.enqueue_create_host_buffer[DType.float32](max(n, 1))
    var hw = ctx.enqueue_create_host_buffer[DType.float32](max(n, 1))
    ctx.enqueue_copy(hg, got.create_sub_buffer[DType.float32](0, max(n, 1)))
    ctx.enqueue_copy(hw, want.create_sub_buffer[DType.float32](0, max(n, 1)))
    ctx.synchronize()
    var err = Float64(0)
    var top = Float64(0)
    var bad = 0
    for i in range(n):
        if skip_per > 0 and not used[i // skip_per]:
            continue
        var a = Float64(hg[i])
        var b = Float64(hw[i])
        if not (abs(a) < 1e30):
            bad += 1
            continue
        err = max(err, abs(a - b))
        top = max(top, abs(b))
    return (err / max(top, 1e-30), bad)


def run(
    ctx: DeviceContext,
    lens: List[Int],
    nh: Int,
    g: Int,
    mild: Bool,
    seed: Int,
    has_d: Bool = True,
    pad_cols: Int = 0,
    poison_rows: Int = 0,
) raises:
    var k = make_case(ctx, lens, nh, g, mild, seed, pad_cols, poison_rows)
    var rows = k.t + poison_rows
    var base = Int(k.xbc.unsafe_ptr())
    var bo = 2 * nh * P
    var co = bo + 2 * g * N

    # the chunk state
    var nst = k.slots * nh * P * N
    var st = ctx.enqueue_create_buffer[DType.float32](nst)
    var st_ref = ctx.enqueue_create_buffer[DType.float32](nst)
    st.enqueue_fill(nan[DType.float32]())
    ctx.enqueue_function[ref_state_kernel](
        k.xbc.unsafe_ptr(),
        k.dt.unsafe_ptr(),
        k.cs.unsafe_ptr(),
        k.info.unsafe_ptr(),
        st_ref.unsafe_ptr(),
        Int32(nh),
        Int32(g),
        Int32(k.slots),
        Int32(k.ld),
        grid_dim=(ceildiv(P * N, 128), nh, k.slots),
        block_dim=(128, 1, 1),
    )
    ssd_state_sm100(
        ctx,
        rows,
        k.batch,
        k.slots,
        nh,
        g,
        base,
        base + bo,
        k.ld,
        k.ld,
        Int(k.dt.unsafe_ptr()),
        Int(k.cs.unsafe_ptr()),
        Int(st.unsafe_ptr()),
        Int(k.info.unsafe_ptr()),
    )
    var hinfo = ctx.enqueue_create_host_buffer[DType.int32](
        3 * k.slots + 2 * k.batch
    )
    ctx.enqueue_copy(hinfo, k.info)
    ctx.synchronize()
    var used = List[Bool]()
    for s in range(k.slots):
        used.append(hinfo[k.slots + s] > 0)
    var rs = linf_rel(ctx, st, st_ref, nst, nh * P * N, used)

    # the output stage, into y with a NaN guard past its end
    var ny = k.t * nh * P
    var y = ctx.enqueue_create_buffer[DType.bfloat16](ny + GUARD)
    var y32 = ctx.enqueue_create_buffer[DType.float32](max(ny, 1))
    var y_ref = ctx.enqueue_create_buffer[DType.float32](max(ny, 1))
    y.enqueue_fill(nan[DType.bfloat16]())
    if k.t > 0:
        ctx.enqueue_function[ref_out_kernel](
            k.xbc.unsafe_ptr(),
            k.d.unsafe_ptr(),
            k.dt.unsafe_ptr(),
            k.cs.unsafe_ptr(),
            k.h.unsafe_ptr(),
            k.info.unsafe_ptr(),
            k.tok_slot.unsafe_ptr(),
            y_ref.unsafe_ptr(),
            Int32(k.t),
            Int32(nh),
            Int32(g),
            Int32(k.slots),
            Int32(k.ld),
            Int32(1 if has_d else 0),
            grid_dim=(ceildiv(nh * P, 128), k.t, 1),
            block_dim=(128, 1, 1),
        )
    ssd_out_sm100(
        ctx,
        rows,
        k.batch,
        k.slots,
        nh,
        g,
        base,
        base + bo,
        base + co,
        k.ld,
        k.ld,
        Int(k.d.unsafe_ptr()) if has_d else 0,
        Int(k.dt.unsafe_ptr()),
        Int(k.cs.unsafe_ptr()),
        Int(k.h.unsafe_ptr()),
        Int(y.unsafe_ptr()),
        Int(k.info.unsafe_ptr()),
    )
    var hy = ctx.enqueue_create_host_buffer[DType.bfloat16](ny + GUARD)
    ctx.enqueue_copy(hy, y)
    ctx.synchronize()
    var hy32 = ctx.enqueue_create_host_buffer[DType.float32](max(ny, 1))
    for i in range(ny):
        hy32[i] = hy[i].cast[DType.float32]()
    ctx.enqueue_copy(y32, hy32)
    var ro = linf_rel(ctx, y32, y_ref, ny)
    var guard_ok = True
    for i in range(GUARD):
        if hy[ny + i] == hy[ny + i]:
            guard_ok = False

    var ok = (
        rs[1] == 0 and rs[0] < TOL and ro[1] == 0 and ro[0] < TOL and guard_ok
    )
    print(
        "case lens=",
        String(lens).replace(" ", ""),
        "H=",
        nh,
        "G=",
        g,
        "mild" if mild else "tilegym",
        "D" if has_d else "noD",
        "pad",
        pad_cols,
        "| state linf_rel",
        rs[0],
        "nonfinite",
        rs[1],
        "| out linf_rel",
        ro[0],
        "nonfinite",
        ro[1],
        "guard",
        "OK" if guard_ok else "WRITTEN",
        "->",
        "OK" if ok else "FAIL",
    )
    _ = st^
    _ = st_ref^
    _ = y^
    _ = y32^
    _ = y_ref^
    _ = k^
    assert_equal(rs[1], 0, "non-finite chunk states")
    assert_true(rs[0] < TOL, String("chunk state linf_rel ", rs[0]))
    assert_equal(ro[1], 0, "non-finite outputs")
    assert_true(ro[0] < TOL, String("output linf_rel ", ro[0]))
    assert_true(guard_ok, "a write past the end of y")


def seqs(*xs: Int) -> List[Int]:
    var l = List[Int]()
    for x in xs:
        l.append(x)
    return l^


def rep(n: Int, k: Int) -> List[Int]:
    var l = List[Int]()
    for _ in range(k):
        l.append(n)
    return l^


def main() raises:
    var ctx = DeviceContext()
    if ctx.device().get_name() != "NVIDIA B200":
        print("skipped: the sm100 SSD kernels run on B200 only")
        return
    var cases = List[Tuple[List[Int], Int, Int]]()
    cases.append((seqs(256), 4, 2))
    cases.append((seqs(1), 2, 1))
    cases.append((seqs(255, 1, 257), 8, 2))
    cases.append((seqs(129, 128, 127, 383), 4, 1))
    cases.append((seqs(3001, 17, 512, 900), 16, 4))
    cases.append((seqs(4096), 32, 4))
    cases.append((rep(700, 7), 64, 8))
    cases.append((seqs(2048, 1, 1, 1, 1023), 32, 4))
    for i in range(len(cases)):
        ref c = cases[i]
        run(ctx, c[0], c[1], c[2], True, 100 + i)
        run(ctx, c[0], c[1], c[2], False, 200 + i)
    run(ctx, seqs(300, 1000), 8, 2, True, 400, has_d=False)
    run(ctx, seqs(300, 1000), 8, 2, True, 401, pad_cols=64)
    # non-finite rows just past a partial last row block (and a whole one)
    run(ctx, seqs(300, 77), 8, 2, True, 402, poison_rows=ROWS_PAD)
    run(ctx, seqs(513, 200), 8, 2, False, 403, poison_rows=ROWS_PAD)
    print("MAMBA2_SSD_SM100 ALL OK")
