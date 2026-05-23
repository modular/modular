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
"""Exhaustive kernel-level tests for the preshuffled-B grouped MXFP4 kernels.

Bypasses the public `mxfp4_grouped_matmul_amd_preb` dispatcher and exercises
each preb kernel directly against a per-expert ungrouped GPU reference
(`mxfp4_block_scaled_matmul_amd`):

  - `PreShuffledBGroupedGEMM.launch[persistent=True]`   — persistent 1D grid
                                                          + XCD swizzle
  - `PreShuffledBGroupedGEMM.launch[persistent=False]`  — direct 3D grid

Dense kernel coverage lives in `test_mxfp4_grouped_matmul_amd.mojo`.

Coverage matrix per kernel:
  - Single expert tiny  (most WGs idle for persistent)
  - Multi-expert mixed sizes
  - Inactive slot: M=0 in the middle
  - Inactive slot: expert_id=-1
  - More tiles than total_wg (persistent multi-wave per WG)
  - Kimi-decode / Kimi-prefill scale at L2 dimensions
  - Both BK_ELEMS=512 (K_BYTES % 256 == 0) and BK_ELEMS=128 paths

Currently MI355X-only.

Usage:
  br test_mxfp4_grouped_matmul_amd_kernels.mojo.test
"""

from std.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu.host.info import MI355X
from std.math import ceildiv
from std.memory import bitcast
from std.random import random_ui64, seed

from internal_utils import assert_almost_equal
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import (
    PreShuffledBGroupedGEMM,
    Shuffler,
    mxfp4_block_scaled_matmul_amd,
)


# ===----------------------------------------------------------------------=== #
# Input helpers
# ===----------------------------------------------------------------------=== #


def _fill_random_bytes(buf: HostBuffer[DType.uint8], n: Int):
    for i in range(n):
        buf[i] = UInt8(random_ui64(0, 255))


def _fill_random_e8m0(buf: HostBuffer[DType.float8_e8m0fnu], n: Int):
    """Scales clamped to E8M0 byte range [125..129] = magnitudes [0.25..4],
    keeping f32 accumulators in range while still exercising scale-dequant."""
    for i in range(n):
        buf[i] = bitcast[DType.float8_e8m0fnu](UInt8(random_ui64(125, 129)))


def _build_routing(
    a_offsets_host: HostBuffer[DType.uint32],
    expert_ids_host: HostBuffer[DType.int32],
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
):
    a_offsets_host[0] = UInt32(0)
    for i in range(len(num_tokens_by_expert)):
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host[i] = Int32(expert_ids_list[i])


# ===----------------------------------------------------------------------=== #
# GPU reference — per-expert ungrouped `mxfp4_block_scaled_matmul_amd`.
# Shares the AMD MFMA code path, so a bug common to all MFMA paths would not
# be caught; but it's fast enough to support large shapes.
# ===----------------------------------------------------------------------=== #


def _gpu_per_expert_reference[
    num_experts: Int, N: Int, K: Int
](
    ctx: DeviceContext,
    num_active: Int,
    a_offsets_host: HostBuffer[DType.uint32],
    expert_ids_host: HostBuffer[DType.int32],
    a_dev: DeviceBuffer[DType.uint8],
    b_dev: DeviceBuffer[DType.uint8],
    a_scales_dev: DeviceBuffer[DType.float8_e8m0fnu],
    b_scales_dev: DeviceBuffer[DType.float8_e8m0fnu],
    c_ref_dev: DeviceBuffer[DType.float32],
) raises:
    comptime packed_K = K // 2
    comptime scale_K = K // MXFP4_SF_VECTOR_SIZE

    for i in range(num_active):
        var token_start = Int(a_offsets_host[i])
        var token_end = Int(a_offsets_host[i + 1])
        var num_tokens = token_end - token_start
        var expert_id = Int(expert_ids_host[i])

        if num_tokens <= 0 or expert_id < 0:
            continue

        var a_expert = TileTensor(
            a_dev.unsafe_ptr() + token_start * packed_K,
            row_major(Coord(num_tokens, Idx[packed_K])),
        )
        var b_expert = TileTensor[mut=False](
            b_dev.unsafe_ptr() + expert_id * N * packed_K,
            row_major[N, packed_K](),
        )
        var sfa_expert = TileTensor[mut=False](
            a_scales_dev.unsafe_ptr() + token_start * scale_K,
            row_major(Coord(num_tokens, Idx[scale_K])),
        )
        var sfb_expert = TileTensor[mut=False](
            b_scales_dev.unsafe_ptr() + expert_id * N * scale_K,
            row_major[N, scale_K](),
        )
        var c_expert = TileTensor[mut=True](
            c_ref_dev.unsafe_ptr() + token_start * N,
            row_major(Coord(num_tokens, Idx[N])),
        )

        mxfp4_block_scaled_matmul_amd(
            c_expert, a_expert, b_expert, sfa_expert, sfb_expert, ctx
        )


# ===----------------------------------------------------------------------=== #
# Shared test body — only the `persistent` and `cu_count` comptime values
# differ between `test_persistent` and `test_direct`.
# ===----------------------------------------------------------------------=== #


def _run_preb[
    num_experts: Int,
    N: Int,
    K: Int,
    BK_ELEMS: Int,
    persistent: Bool,
    cu_count: Int,  # struct param — `total_wg = cu_count * 2`
](
    name: String,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises:
    comptime assert K % 128 == 0, "K must be a multiple of 128"
    comptime packed_K = K // 2
    comptime scale_K = K // MXFP4_SF_VECTOR_SIZE

    var total_tokens = 0
    var max_tokens = 0
    var num_active = len(num_tokens_by_expert)
    for ne in num_tokens_by_expert:
        total_tokens += ne
        max_tokens = max(max_tokens, ne)

    comptime label = "[persistent]" if persistent else "[direct]    "
    print(
        "  ",
        label,
        name,
        " E=",
        num_experts,
        " N=",
        N,
        " K=",
        K,
        " active=",
        num_active,
        " total_tokens=",
        total_tokens,
        " BK_ELEMS=",
        BK_ELEMS,
    )

    # Host buffers + random init.
    var a_h = ctx.enqueue_create_host_buffer[DType.uint8](
        total_tokens * packed_K
    )
    var b_h = ctx.enqueue_create_host_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    var a_sc_h = ctx.enqueue_create_host_buffer[DType.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var b_sc_h = ctx.enqueue_create_host_buffer[DType.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var a_off_h = ctx.enqueue_create_host_buffer[DType.uint32](num_active + 1)
    var eid_h = ctx.enqueue_create_host_buffer[DType.int32](num_active)
    ctx.synchronize()

    _fill_random_bytes(a_h, total_tokens * packed_K)
    _fill_random_bytes(b_h, num_experts * N * packed_K)
    _fill_random_e8m0(a_sc_h, total_tokens * scale_K)
    _fill_random_e8m0(b_sc_h, num_experts * N * scale_K)
    _build_routing(a_off_h, eid_h, num_tokens_by_expert, expert_ids_list)

    # Device buffers + upload.
    var a_d = ctx.enqueue_create_buffer[DType.uint8](total_tokens * packed_K)
    var b_d = ctx.enqueue_create_buffer[DType.uint8](num_experts * N * packed_K)
    var b_pre_d = ctx.enqueue_create_buffer[DType.uint8](
        num_experts * N * packed_K
    )
    var a_sc_d = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var b_sc_d = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var a_off_d = ctx.enqueue_create_buffer[DType.uint32](num_active + 1)
    var eid_d = ctx.enqueue_create_buffer[DType.int32](num_active)
    var c_d = ctx.enqueue_create_buffer[DType.float32](total_tokens * N)
    var c_ref_d = ctx.enqueue_create_buffer[DType.float32](total_tokens * N)

    # Zero c_d and c_ref_d. Inactive slots (M=0 or expert_id=-1) leave their
    # output range unwritten by both the kernel and the reference, so the
    # comparison against fresh-allocated garbage would otherwise mismatch.
    c_d.enqueue_fill(Float32(0.0))
    c_ref_d.enqueue_fill(Float32(0.0))

    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(a_sc_d, a_sc_h)
    ctx.enqueue_copy(b_sc_d, b_sc_h)
    ctx.enqueue_copy(a_off_d, a_off_h)
    ctx.enqueue_copy(eid_d, eid_h)

    # GPU preshuffle b_d → b_pre_d.
    var b_raw_tt = TileTensor[mut=False](
        b_d, row_major[num_experts, N, packed_K]()
    )
    var b_pre_dst_tt = TileTensor[mut=True](
        b_pre_d,
        Shuffler[num_experts].b_5d_grouped_layout[N=N, K_BYTES=packed_K],
    )
    Shuffler[num_experts].preshuffle_b_5d[N=N, K_BYTES=packed_K](
        b_raw_tt, b_pre_dst_tt, ctx
    )

    # Reference: per-expert ungrouped matmul against raw B.
    _gpu_per_expert_reference[num_experts, N, K](
        ctx,
        num_active,
        a_off_h,
        eid_h,
        a_d,
        b_d,
        a_sc_d,
        b_sc_d,
        c_ref_d,
    )

    # Run the preb kernel under test.
    var a_tt = TileTensor[mut=False](
        a_d, row_major(Coord(total_tokens, Idx[packed_K]))
    )
    var b_pre_tt = TileTensor[mut=False](
        b_pre_d, row_major[num_experts, N * packed_K]()
    )
    var a_sc_tt = TileTensor[mut=False](
        a_sc_d, row_major(Coord(total_tokens, Idx[scale_K]))
    )
    var b_sc_tt = TileTensor[mut=False](
        b_sc_d, row_major[num_experts, N, scale_K]()
    )
    var a_off_tt = TileTensor(a_off_d, row_major(Coord(num_active + 1)))
    var eid_tt = TileTensor(eid_d, row_major(Coord(num_active)))
    var c_tt = TileTensor[mut=True](c_d, row_major(Coord(total_tokens, Idx[N])))

    PreShuffledBGroupedGEMM[cu_count=cu_count].launch[
        BM=64,
        BN=128,
        BK_ELEMS=BK_ELEMS,
        WN=64,
        persistent=persistent,
    ](
        c_tt,
        a_tt,
        b_pre_tt,
        a_sc_tt,
        b_sc_tt,
        a_off_tt,
        eid_tt,
        max_tokens,
        num_active,
        ctx,
    )
    ctx.synchronize()

    # Compare.
    var c_h = ctx.enqueue_create_host_buffer[DType.float32](total_tokens * N)
    var c_ref_h = ctx.enqueue_create_host_buffer[DType.float32](
        total_tokens * N
    )
    ctx.enqueue_copy(c_h, c_d)
    ctx.enqueue_copy(c_ref_h, c_ref_d)
    ctx.synchronize()

    assert_almost_equal(
        c_h.unsafe_ptr(),
        c_ref_h.unsafe_ptr(),
        total_tokens * N,
        atol=0.05,
        rtol=0.05,
    )
    print("    PASS")

    _ = a_d^
    _ = b_d^
    _ = b_pre_d^
    _ = a_sc_d^
    _ = b_sc_d^
    _ = a_off_d^
    _ = eid_d^
    _ = c_d^
    _ = c_ref_d^


def test_persistent[
    num_experts: Int,
    N: Int,
    K: Int,
    BK_ELEMS: Int = 512,
    cu_count: Int = 256,  # default = MI355X
](
    name: String,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises:
    """`PreShuffledBGroupedGEMM.launch[persistent=True]` — 1D `total_wg` grid
    walking a global tile counter; XCD swizzle on `block_idx.x`.

    `cu_count` overrides the launch grid size; default 256 = MI355X. Lower
    values shrink `total_wg = cu_count * 2`, which lets the multi-wave test
    trigger the per-WG `while target_tile < expert_end: target_tile +=
    total_wg` path with a CPU-tractable workload.
    """
    _run_preb[
        num_experts,
        N,
        K,
        BK_ELEMS=BK_ELEMS,
        persistent=True,
        cu_count=cu_count,
    ](name, num_tokens_by_expert, expert_ids_list, ctx)


def test_direct[
    num_experts: Int,
    N: Int,
    K: Int,
    BK_ELEMS: Int = 512,
    cu_count: Int = 256,
](
    name: String,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises:
    """`PreShuffledBGroupedGEMM.launch[persistent=False]` — 3D workload-sized
    grid: one WG per (n_tile, m_tile, expert)."""
    _run_preb[
        num_experts,
        N,
        K,
        BK_ELEMS=BK_ELEMS,
        persistent=False,
        cu_count=cu_count,
    ](name, num_tokens_by_expert, expert_ids_list, ctx)


# ===----------------------------------------------------------------------=== #
# Test matrix
# ===----------------------------------------------------------------------=== #


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp4_grouped_matmul_amd_kernels currently requires MI355X"

    print("===> preshuffled-B grouped MXFP4 — exhaustive kernel-level tests")

    # ----------------------------------------------------------------- #
    # Shape conventions (matching test_mxfp4_moe_matmul_amd_routed.mojo):
    #   L2 decode/prefill shape:  N=512,   K=2048
    #   L2 gate+up aspect:        N=1024,  K=512
    #   L2 down aspect:           N=512,   K=1024
    #   L3 kimi gate+up (real):   N=14336, K=4096
    #   L3 kimi down (real):      N=4096,  K=7168
    # ----------------------------------------------------------------- #

    # ----------------------------------------------------------------- #
    # Preb persistent kernel — launch[persistent=True]
    # ----------------------------------------------------------------- #
    print("---- preb persistent kernel ----")

    # Structural edge cases (L2 decode shape: N=512, K=2048).
    test_persistent[1, 512, 2048]("single-tiny", [16], [0], ctx)
    test_persistent[1, 512, 2048]("single-mid", [128], [0], ctx)
    test_persistent[4, 512, 2048](
        "multi-mixed", [32, 64, 128, 256], [0, 1, 2, 3], ctx
    )
    test_persistent[4, 512, 2048](
        "inactive-M0", [64, 0, 128, 32], [0, 1, 2, 3], ctx
    )
    test_persistent[4, 512, 2048](
        "inactive-eid-1", [64, 64, 128, 32], [0, -1, 2, 3], ctx
    )

    # More tiles than total_wg (= 512 on MI355X) — exercise multi-wave per WG.
    # 9 experts × m_count=4 × gx_n(N=1024)=8 = 288 tiles? need >512.
    # 9 × m_count=8 × gx_n=8 = 576 → M=512 per expert.
    test_persistent[9, 1024, 512](
        "multi-wave",
        [512, 512, 512, 512, 512, 512, 512, 512, 512],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        ctx,
    )

    # Kimi-decode-like: 49 active experts, very few tokens each, mixed
    # inactive slots (slot 40: M=0; slot 47: expert_id=-1) — matches the
    # L2.2 / L2.3 pattern from test_mxfp4_moe_matmul_amd_routed.
    test_persistent[49, 512, 2048](
        "kimi-decode-49experts",
        [
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            0,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            -1,
            48,
        ],
        ctx,
    )

    # Kimi-prefill scaled (L2.4): 49 active experts, ~40 tokens each.
    test_persistent[49, 512, 2048](
        "kimi-prefill-49experts",
        [
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
        ],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
        ],
        ctx,
    )

    # BK_ELEMS=128 path (K=256 → packed_K=128, not divisible by 256).
    test_persistent[4, 128, 256, BK_ELEMS=128](
        "bk128-multi", [32, 64, 128, 32], [0, 1, 2, 3], ctx
    )

    # ----------------------------------------------------------------- #
    # Preb direct kernel — launch[persistent=False]
    # ----------------------------------------------------------------- #
    print("---- preb direct kernel ----")
    test_direct[1, 512, 2048]("single-tiny", [16], [0], ctx)
    test_direct[1, 512, 2048]("single-mid", [128], [0], ctx)
    test_direct[4, 512, 2048](
        "multi-mixed", [32, 64, 128, 256], [0, 1, 2, 3], ctx
    )
    test_direct[4, 512, 2048](
        "inactive-M0", [64, 0, 128, 32], [0, 1, 2, 3], ctx
    )
    test_direct[4, 512, 2048](
        "inactive-eid-1", [64, 64, 128, 32], [0, -1, 2, 3], ctx
    )

    # Large single-expert prefill (where direct typically wins over persistent).
    test_direct[1, 1024, 512]("single-large", [8192], [0], ctx)

    # Kimi-prefill scaled.
    test_direct[49, 512, 2048](
        "kimi-prefill-49experts",
        [
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
            40,
        ],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
        ],
        ctx,
    )

    # BK_ELEMS=128 path.
    test_direct[4, 128, 256, BK_ELEMS=128](
        "bk128-multi", [32, 64, 128, 32], [0, 1, 2, 3], ctx
    )

    print("==== all preb grouped MXFP4 kernel tests passed ====")
