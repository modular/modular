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
"""Isolation unit test for the Apple M5 FA2-prefill `AppleSoftmax` struct.

Mirrors `test/gpu/structured_kernels/test_mha_softmax_unit.mojo` (the AMD
Softmax-struct unit test): drive the register-resident online-softmax state
object through a TWO-tile sequence in isolation and compare against a host fp32
flash-attention online-softmax reference. Two tiles exercise the running-max
correction (`alpha`), the `l` accumulation, and the final `normalize` -- a
single tile would not move `m`/`l` off their identity seeds.

The "output" the softmax rescales here is a synthetic per-(row, col) value (not
a real P.V product); the test asserts that running an attention output through
`update` x2 + `normalize` reproduces the mathematically-correct softmax-weighted
combination, which is the contract the prefill kernel relies on.

PASS on M5 gates increment 2 of the prefill bring-up (DESIGN.md test plan).
"""

from std.gpu import WARP_SIZE, lane_id
from std.gpu.host import DeviceContext
from std.math import exp
from std.sys.info import _accelerator_arch

from linalg.arch.apple.mma import MmaOpApple

from nn.attention.gpu.apple.fa_prefill import AppleSoftmax

comptime NUM_M_MMAS = 2
comptime NUM_N_MMAS = 2
comptime SQ = NUM_M_MMAS * 16  # query rows in the tile
comptime SK = NUM_N_MMAS * 16  # KV cols per score tile
comptime NUM_TILES = 2
comptime DEPTH_MMAS = 2  # output (P.V) column fragments; depth = 32 here
comptime OUT_N = DEPTH_MMAS * 16


def _softmax_unit_kernel(
    # Two score tiles (SQ x SK each), row-major, fp32.
    s0_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    s1_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    # Per-tile synthetic "attention output" contribution (SQ x OUT_N), the
    # value PV would add for that tile (here injected directly so the test is
    # softmax-only). out_contribution[tile] is what `O += P@V` would produce
    # for tile `tile` if P were all-ones; we instead weight it by the tile's
    # row-sum to emulate the real recurrence (see host reference).
    o0_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    o1_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
):
    """One simdgroup: load 2 score tiles + 2 PV-output tiles into the MMA accum
    layout, run AppleSoftmax.update twice + normalize, write the final SQ x OUT_N
    output back."""
    var lane = Int(lane_id())
    var rb = ((lane & 7) >> 1) + ((lane & 16) >> 2)
    var cb = ((lane & 1) << 2) + (lane & 8)

    comptime ScoreMma = MmaOpApple[
        DType.float32, DType.float32, NUM_M_MMAS, NUM_N_MMAS
    ]
    comptime OutMma = MmaOpApple[
        DType.float32, DType.float32, NUM_M_MMAS, DEPTH_MMAS
    ]

    var softmax = AppleSoftmax[NUM_M_MMAS, NUM_N_MMAS]()

    # Running output accumulator (what O would hold).
    var output = OutMma.zero_accum()

    # --- helper: load a row-major SQ x SK score matrix into accum layout. ---
    @parameter
    def load_scores(
        src: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ) -> ScoreMma.AccumType:
        var acc = ScoreMma.zero_accum()
        comptime for mi in range(NUM_M_MMAS):
            comptime for ni in range(NUM_N_MMAS):
                var frag = SIMD[DType.float32, 8](0)
                comptime for el in range(8):
                    var row = mi * 16 + rb + (8 if el > 3 else 0)
                    var col = ni * 16 + cb + (el & 3)
                    frag[el] = src[row * SK + col]
                acc[mi * NUM_N_MMAS + ni] = frag
        return acc

    # --- helper: load a row-major SQ x OUT_N tile into the output accum. ---
    @parameter
    def add_output(
        mut acc: OutMma.AccumType,
        src: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ):
        comptime for mi in range(NUM_M_MMAS):
            comptime for ni in range(DEPTH_MMAS):
                var idx = mi * DEPTH_MMAS + ni
                var frag = acc[idx]
                comptime for el in range(8):
                    var row = mi * 16 + rb + (8 if el > 3 else 0)
                    var col = ni * 16 + cb + (el & 3)
                    frag[el] = frag[el] + src[row * OUT_N + col]
                acc[idx] = frag

    # Tile 0: O starts at 0; update rescales it (no-op on zero) then we add the
    # tile-0 PV contribution AFTER the rescale -- mirrors the kernel order
    # (rescale old O, then mma_pv adds new P@V into O).
    var s0 = load_scores(s0_ptr)
    softmax.update[DEPTH_MMAS](s0, output)
    add_output(output, o0_ptr)

    # Tile 1: update rescales the accumulated O by the new correction, then we
    # add tile-1's PV contribution.
    var s1 = load_scores(s1_ptr)
    softmax.update[DEPTH_MMAS](s1, output)
    add_output(output, o1_ptr)

    softmax.normalize[DEPTH_MMAS](output)

    # Write the final output back (row-major SQ x OUT_N). Only the cb==0 lanes
    # plus the col walk cover every element once -- but each lane owns distinct
    # (row, col) pairs, so all lanes write without races.
    comptime for mi in range(NUM_M_MMAS):
        comptime for ni in range(DEPTH_MMAS):
            var frag = output[mi * DEPTH_MMAS + ni]
            comptime for el in range(8):
                var row = mi * 16 + rb + (8 if el > 3 else 0)
                var col = ni * 16 + cb + (el & 3)
                out_ptr[row * OUT_N + col] = frag[el]


def test_apple_fa_softmax_unit(ctx: DeviceContext) raises:
    print("== test_apple_fa_softmax_unit (2-tile online softmax + normalize)")

    comptime n_s = SQ * SK
    comptime n_o = SQ * OUT_N

    var s0_h = ctx.enqueue_create_host_buffer[DType.float32](n_s)
    var s1_h = ctx.enqueue_create_host_buffer[DType.float32](n_s)
    var o0_h = ctx.enqueue_create_host_buffer[DType.float32](n_o)
    var o1_h = ctx.enqueue_create_host_buffer[DType.float32](n_o)

    # Deterministic, non-monotonic scores so the max is not trivially the last
    # column, and so tile-1 can have a larger max than tile-0 for some rows (to
    # exercise the correction) and smaller for others.
    for r in range(SQ):
        for c in range(SK):
            var v0 = Float32(((r * 37 + c * 101) % 211) - 105) * 0.05
            var v1 = Float32(((r * 71 + c * 53) % 197) - 98) * 0.05
            s0_h[r * SK + c] = v0
            s1_h[r * SK + c] = v1
    for r in range(SQ):
        for c in range(OUT_N):
            o0_h[r * OUT_N + c] = Float32(((r * 13 + c * 7) % 100)) * 0.1
            o1_h[r * OUT_N + c] = Float32(((r * 29 + c * 3) % 100)) * 0.1 - 5.0

    var s0_d = ctx.enqueue_create_buffer[DType.float32](n_s)
    var s1_d = ctx.enqueue_create_buffer[DType.float32](n_s)
    var o0_d = ctx.enqueue_create_buffer[DType.float32](n_o)
    var o1_d = ctx.enqueue_create_buffer[DType.float32](n_o)
    var out_d = ctx.enqueue_create_buffer[DType.float32](n_o)
    ctx.enqueue_copy(s0_d, s0_h)
    ctx.enqueue_copy(s1_d, s1_h)
    ctx.enqueue_copy(o0_d, o0_h)
    ctx.enqueue_copy(o1_d, o1_h)

    ctx.enqueue_function[_softmax_unit_kernel](
        s0_d.unsafe_ptr(),
        s1_d.unsafe_ptr(),
        o0_d.unsafe_ptr(),
        o1_d.unsafe_ptr(),
        out_d.unsafe_ptr(),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )

    var out_h = ctx.enqueue_create_host_buffer[DType.float32](n_o)
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    # DRIV-199: keep device buffers alive past synchronize.
    _ = s0_d^
    _ = s1_d^
    _ = o0_d^
    _ = o1_d^
    _ = out_d^

    # Host reference: full flash-attention online softmax over the two score
    # tiles, with the kernel's "add PV after rescale" recurrence. For row r:
    #   m=-inf,l=0,O=0
    #   tile t: m_t=rowmax(s_t); m_new=max(m,m_t); a=exp(m-m_new)
    #           P=exp(s_t - m_new); l=l*a + sum(P); O=O*a + o_t (PV contribution)
    #           m=m_new
    #   final: O /= l
    var pass_ = True
    for r in range(SQ):
        var m = Float32(-3.0e38)
        var l = Float32(0)
        var o_ref = [Float32(0)] * OUT_N

        # tile 0
        var m0 = Float32(-3.0e38)
        for c in range(SK):
            m0 = max(m0, s0_h[r * SK + c])
        var m_new = max(m, m0)
        var a = exp(m - m_new)
        var psum = Float32(0)
        for c in range(SK):
            psum += exp(s0_h[r * SK + c] - m_new)
        l = l * a + psum
        for c in range(OUT_N):
            o_ref[c] = o_ref[c] * a + o0_h[r * OUT_N + c]
        m = m_new

        # tile 1
        var m1 = Float32(-3.0e38)
        for c in range(SK):
            m1 = max(m1, s1_h[r * SK + c])
        m_new = max(m, m1)
        a = exp(m - m_new)
        psum = Float32(0)
        for c in range(SK):
            psum += exp(s1_h[r * SK + c] - m_new)
        l = l * a + psum
        for c in range(OUT_N):
            o_ref[c] = o_ref[c] * a + o1_h[r * OUT_N + c]

        for c in range(OUT_N):
            var expected = o_ref[c] / l
            var got = out_h[r * OUT_N + c]
            if abs(got - expected) > Float32(1e-3) * (1.0 + abs(expected)):
                print("FAIL row", r, "col", c, "got", got, "exp", expected)
                pass_ = False

    if not pass_:
        raise Error("FAILED (see FAIL lines above)")
    print("PASS")


def main() raises:
    comptime if "metal" not in _accelerator_arch():
        print("SKIP: Apple GPU required")
        return
    var ctx = DeviceContext()
    if ctx.compute_capability() != 5:
        print("SKIP: Apple M5 required (16x16 simdgroup MMA fragment)")
        return
    test_apple_fa_softmax_unit(ctx)
