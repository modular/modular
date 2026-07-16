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
"""KS224: `ep_wait` direct `scale_4d` emission.

The MXFP4 up/gate-projection grouped matmul consumes the activation E8M0 scale
in the per-expert fixed-stride `scale_4d` slot layout
(`Shuffler.scale_4d_byte_off`). Today that layout is produced by a standalone
kernel (`preshuffle_grouped_scale_4d_gpu`, traced
`mxfp4_preshuffle_grouped_scale_4d_kernel_KS224`) that runs *serially* after
`ep_wait` (dispatch_wait) writes the per-token scale row-major. The fusion
(KS224/up-proj) has `MXFP4TokenFormat.copy_msg_to_output_tensor`
write the scale DIRECTLY in the slot layout behind a
`comptime fuse_a_scale_preshuffle` flag, deleting the separate kernel from the
decode critical path.

This test drives the REAL `MXFP4TokenFormat.copy_msg_to_output_tensor` two ways
and asserts the produced slot buffers are byte-for-byte equal:

  reference (fuse_a_scale_preshuffle=False): copy writes raw [tokens, K/32]
      --preshuffle_grouped_scale_4d_gpu-->  per-expert scale_4d slots
  fused     (fuse_a_scale_preshuffle=True):  copy writes the slot layout directly.

A tiny launcher kernel replays `ep_wait`'s per-token copy granularity: ONE WARP
PER TOKEN, with the warps spread across blocks so that rows `r` and `r+16`
(which pack into the same `scale_4d` i32 cell) are written by DIFFERENT
warps/SMs. That is precisely the layout in which the byte-store race-safety
claim must hold: each store is an independent single E8M0 byte into the shared
i32 cell (no read-modify-write), so concurrent writers of byte 0 (row `r`) and
byte 1 (row `r+16`) of the same cell do not clobber each other. If the store
ever lowered to an i32 RMW, this multi-warp/multi-SM layout would surface the
corruption as a byte mismatch.

The expert slot index passed in mirrors the real caller: the dispatch-wait tile
loop passes `expert_id + shared_expert_offset` as `expert_slot` and
`expert_start_pos` as `expert_start`. Here `shared_expert_offset=0` (dispatch
wait does not fuse the shared expert), so `expert_slot == expert_id`.

Single-GPU is sufficient: the scale write reads a local recv buffer and writes
a local output; nothing the fusion changes crosses ranks.

Currently MI355X-only.

Usage (after editing ep_comm.mojo, plain `mojo` is shadowed by the installed
shmem package — use bazel mojo):
  ./bazelw run //KGEN/tools/mojo -- \\
      max/kernels/test/gpu/shmem/test_mxfp4_ep_wait_scale_fusion.mojo
"""

from std.gpu import block_idx
from std.gpu.host import DeviceContext, HostBuffer
from std.gpu.host.info import MI355X
from std.gpu.primitives import warp_id
from std.math import align_up
from std.random import random_ui64, seed

from layout import Coord, Idx, TileTensor, row_major
from layout.tile_layout import TensorLayout
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import Shuffler

from shmem.ep_comm import MXFP4TokenFormat

from std.testing import assert_equal

comptime WARP_SIZE = 64  # gfx950


# ===----------------------------------------------------------------------=== #
# Input helpers
# ===----------------------------------------------------------------------=== #


def _build_routing(
    a_offsets_host: HostBuffer[DType.uint32],
    num_tokens_by_expert: List[Int],
):
    """Ragged per-expert prefix sums: row_offsets[0]=0, row_offsets[e+1]=sum."""
    a_offsets_host[0] = UInt32(0)
    for i in range(len(num_tokens_by_expert)):
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )


# ===----------------------------------------------------------------------=== #
# Launcher kernel: one warp per token, calling the REAL format copy.
#
# Grid is sized so warps spread across many blocks (one token per warp, a few
# warps per block) — this puts rows r and r+16 on different warps/SMs, which is
# the race-stress layout for the shared scale_4d i32 cell.
# ===----------------------------------------------------------------------=== #


def _ep_wait_copy_kernel[
    fp4_dtype: DType,
    scales_dtype: DType,
    out_layout: TensorLayout,
    scales_layout: TensorLayout,
    aoff_layout: TensorLayout,
    hidden_size: Int,
    top_k: Int,
    fuse_a_scale_preshuffle: Bool,
    warps_per_block: Int,
](
    output_tokens: TileTensor[fp4_dtype, out_layout, MutUntrackedOrigin],
    output_scales: TileTensor[scales_dtype, scales_layout, MutUntrackedOrigin],
    recv_buf: UnsafePointer[UInt8, MutUntrackedOrigin],
    a_offsets: TileTensor[DType.uint32, aoff_layout, ImmutAnyOrigin],
    msg_bytes: Int,
    num_active: Int,
    total_tokens: Int,
    max_padded_M: Int,
):
    var fmt = MXFP4TokenFormat[
        hidden_size, top_k, fuse_a_scale_preshuffle=fuse_a_scale_preshuffle
    ](output_tokens, output_scales, max_padded_M)

    # Global warp index → token. One warp processes one token (matches ep_wait).
    var warp_global = Int(block_idx.x) * warps_per_block + Int(warp_id())
    var token = warp_global
    if token >= total_tokens:
        return

    # Locate the expert slot owning this token (linear scan over the small
    # active-expert count; mirrors how each dispatch-wait comm SM knows its
    # expert). expert_start is the expert's first global output row.
    var expert_slot = 0
    while (
        expert_slot < num_active - 1
        and Int(a_offsets[Coord(expert_slot + 1)]) <= token
    ):
        expert_slot += 1
    var expert_start = Int(a_offsets[Coord(expert_slot)])

    var msg_ptr = recv_buf + token * msg_bytes
    fmt.copy_msg_to_output_tensor(msg_ptr, token, expert_slot, expert_start)


# ===----------------------------------------------------------------------=== #
# The check: fused slots == reference (separate-kernel) slots, byte for byte.
# ===----------------------------------------------------------------------=== #


def _run_fusion_check[
    hidden_size: Int,
    NUM_ACTIVE: Int,
    top_k: Int = 8,
](
    name: String,
    num_tokens_by_expert: List[Int],
    inflate_padded_M: Int,
    ctx: DeviceContext,
) raises:
    comptime assert hidden_size % MXFP4_SF_VECTOR_SIZE == 0
    comptime output_dim = hidden_size // 2  # FP4 packs 2 elems/byte
    comptime scale_K = hidden_size // MXFP4_SF_VECTOR_SIZE
    comptime n_off = NUM_ACTIVE + 1
    comptime warps_per_block = 4

    var num_active = NUM_ACTIVE
    debug_assert(len(num_tokens_by_expert) == NUM_ACTIVE)
    var total_tokens = 0
    var max_tokens = 0
    for ne in num_tokens_by_expert:
        total_tokens += ne
        max_tokens = max(max_tokens, ne)
    # The slot stride the matmul reads with. `inflate_padded_M` lets a test
    # drive a build-time bound strictly larger than the runtime max (the common
    # decode case) to guard the producer/consumer single-source-of-truth.
    var max_padded_M = max(align_up(max_tokens, 32), inflate_padded_M)
    var slot_bytes = num_active * max_padded_M * scale_K

    print(
        "  ",
        name,
        " active=",
        num_active,
        " total_tokens=",
        total_tokens,
        " hidden=",
        hidden_size,
        " scale_K=",
        scale_K,
        " max_padded_M=",
        max_padded_M,
    )

    comptime hw = ctx.default_device_info

    # Dummy FP4 token output (the copy writes it; we don't compare it here).
    var tok_out_d = ctx.enqueue_create_buffer[DType.uint8](
        total_tokens * output_dim
    )
    var out_tt = TileTensor[origin=MutAnyOrigin](
        tok_out_d, row_major(Coord(total_tokens, Idx[output_dim]))
    )

    # The MXFP4 message per token: [FP4 quants | E8M0 scales | ...]. We only
    # exercise the quant + scale region; the copy reads scales at
    # `scales_offset()`. Use the format's own static sizing for fidelity — the
    # dtype/layout params are inferred from the construction args (mirroring
    # `ep.mojo`), so build a throwaway instance to read the static offsets.
    var dummy_scales_d = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        scale_K
    )
    var dummy_fmt = MXFP4TokenFormat[hidden_size, top_k](
        out_tt,
        TileTensor[origin=MutAnyOrigin](
            dummy_scales_d, row_major(Coord(1, Idx[scale_K]))
        ),
    )
    comptime FmtType = type_of(dummy_fmt)
    var scales_off = FmtType.scales_offset()
    var msg_bytes = FmtType.token_size()

    # --- Host inputs: synthesized recv buffer (scales region) + routing. ---
    var recv_h = ctx.enqueue_create_host_buffer[DType.uint8](
        total_tokens * msg_bytes
    )
    var a_off_h = ctx.enqueue_create_host_buffer[DType.uint32](n_off)
    ctx.synchronize()
    # Zero the whole message, then place random E8M0 bytes in the scale region.
    for i in range(total_tokens * msg_bytes):
        recv_h[i] = UInt8(0)
    for t in range(total_tokens):
        for k in range(scale_K):
            recv_h[t * msg_bytes + scales_off + k] = UInt8(random_ui64(0, 255))
    _build_routing(a_off_h, num_tokens_by_expert)

    var recv_d = ctx.enqueue_create_buffer[DType.uint8](
        total_tokens * msg_bytes
    )
    var a_off_d = ctx.enqueue_create_buffer[DType.uint32](n_off)
    ctx.enqueue_copy(recv_d, recv_h)
    ctx.enqueue_copy(a_off_d, a_off_h)
    var a_off_tt = TileTensor[origin=ImmutAnyOrigin](
        a_off_d, row_major[n_off]()
    )

    var grid_blocks = (total_tokens + warps_per_block - 1) // warps_per_block
    if grid_blocks == 0:
        grid_blocks = 1

    # ---- Path A (reference): copy writes raw [tokens, scale_K], then the
    #      standalone preshuffle kernel rearranges into slots. ----
    var raw_scales_d = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var ref_d = ctx.enqueue_create_buffer[DType.uint8](slot_bytes)
    ref_d.enqueue_fill(UInt8(0))

    var raw_scales_tt = TileTensor[origin=MutAnyOrigin](
        raw_scales_d, row_major(Coord(total_tokens, Idx[scale_K]))
    )

    # Bind the launcher kernel fully: the layout params after `//,` are
    # inferred, so `enqueue_function` needs every param pinned (incl. the two
    # scales layouts, which differ between the raw and slot buffers).
    comptime kernel_ref = _ep_wait_copy_kernel[
        DType.uint8,
        DType.float8_e8m0fnu,
        out_tt.LayoutType,
        raw_scales_tt.LayoutType,
        a_off_tt.LayoutType,
        hidden_size=hidden_size,
        top_k=top_k,
        fuse_a_scale_preshuffle=False,
        warps_per_block=warps_per_block,
    ]
    ctx.enqueue_function[kernel_ref](
        out_tt,
        raw_scales_tt,
        recv_d.unsafe_ptr(),
        a_off_tt,
        msg_bytes,
        num_active,
        total_tokens,
        max_padded_M,
        grid_dim=grid_blocks,
        block_dim=warps_per_block * WARP_SIZE,
    )

    var raw_scales_bytes = TileTensor[origin=ImmutAnyOrigin](
        raw_scales_d.unsafe_ptr().bitcast[UInt8]().as_unsafe_any_origin(),
        row_major(Coord(total_tokens, Idx[scale_K])),
    )
    var ref_slots_tt = TileTensor[origin=MutAnyOrigin](
        ref_d, row_major(Coord(num_active * max_padded_M, Idx[scale_K]))
    )
    Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=scale_K](
        raw_scales_bytes,
        ref_slots_tt,
        a_off_tt,
        num_active,
        # IMPORTANT: the reference standalone preshuffle must write with the
        # SAME slot stride the fused path uses (`max_padded_M`), not the runtime
        # `max_tokens`. Passing the inflated bound keeps both paths' slot
        # strides identical so the whole-buffer byte compare is valid
        # (producer/consumer single-source-of-truth, in test form).
        max_padded_M,
        hw.sm_count * 2,
        ctx,
    )

    # ---- Path B (fused): copy writes the slot layout directly. ----
    var fused_scales_d = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        slot_bytes
    )
    fused_scales_d.enqueue_fill(Scalar[DType.float8_e8m0fnu](0))
    var fused_slots_tt = TileTensor[origin=MutAnyOrigin](
        fused_scales_d,
        row_major(Coord(num_active * max_padded_M, Idx[scale_K])),
    )
    comptime kernel_fused = _ep_wait_copy_kernel[
        DType.uint8,
        DType.float8_e8m0fnu,
        out_tt.LayoutType,
        fused_slots_tt.LayoutType,
        a_off_tt.LayoutType,
        hidden_size=hidden_size,
        top_k=top_k,
        fuse_a_scale_preshuffle=True,
        warps_per_block=warps_per_block,
    ]
    ctx.enqueue_function[kernel_fused](
        out_tt,
        fused_slots_tt,
        recv_d.unsafe_ptr(),
        a_off_tt,
        msg_bytes,
        num_active,
        total_tokens,
        max_padded_M,
        grid_dim=grid_blocks,
        block_dim=warps_per_block * WARP_SIZE,
    )

    # --- Compare byte for byte over the full slot region. ---
    var ref_host = ctx.enqueue_create_host_buffer[DType.uint8](slot_bytes)
    var fused_host = ctx.enqueue_create_host_buffer[DType.uint8](slot_bytes)
    ctx.enqueue_copy(ref_host, ref_d)
    ctx.enqueue_copy(fused_host, fused_scales_d.unsafe_ptr().bitcast[UInt8]())
    ctx.synchronize()

    var mismatches = 0
    for i in range(slot_bytes):
        if ref_host[i] != fused_host[i]:
            mismatches += 1
    assert_equal(mismatches, 0)
    print("    OK: ", slot_bytes, " bytes match")


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp4_ep_wait_scale_fusion currently requires MI355X"

    # KS224 = up/gate proj: hidden = 7168 → scale_K = 224.
    # Decode-ish (few tokens/expert), prefill-ish (many), empty experts, and
    # cells split across r / r+16 (multi-tile m_blocks).
    _run_fusion_check[hidden_size=7168, NUM_ACTIVE=1](
        "up-proj-single-tiny", [1], 0, ctx
    )
    _run_fusion_check[hidden_size=7168, NUM_ACTIVE=5](
        "up-proj-decode", [1, 3, 0, 2, 5], 0, ctx
    )
    # >32 tokens/expert → multiple m_blocks → rows r and r+16 are real tokens
    # in the same cell, written by different warps (race-stress).
    _run_fusion_check[hidden_size=7168, NUM_ACTIVE=4](
        "up-proj-multi-tile", [37, 64, 100, 5], 0, ctx
    )
    # Inflated max_padded_M (> runtime max) with >1 expert: guards that the
    # reference and fused slot strides are the SAME build-time constant, so
    # experts >= 1 land in the right slot.
    _run_fusion_check[hidden_size=7168, NUM_ACTIVE=3](
        "up-proj-inflated-stride", [5, 20, 12], 128, ctx
    )
    # A down-proj-sized check too (KS64) to confirm the same code path is
    # K_SCALES-agnostic.
    _run_fusion_check[hidden_size=2048, NUM_ACTIVE=4](
        "down-proj-decode", [1, 2, 0, 4], 0, ctx
    )
    print("All ep_wait MXFP4 scale-fusion checks passed.")
