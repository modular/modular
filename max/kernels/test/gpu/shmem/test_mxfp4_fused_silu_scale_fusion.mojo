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
"""KS64: `fused_silu_mx` direct `scale_4d` emission.

`fused_silu_mx_kernel` (MXFP4) stores each E8M0 activation scale in the
down-proj grouped matmul's per-expert slot layout
(`Shuffler.scale_4d_slot_byte_off`). A synthesized ragged activation and its
`row_offsets` drive it on one MI355X. Two gates check it:

1. `_run_fusion_check`: every real row's scales are written and nothing else,
   in the standalone preshuffle's layout. Attribution is the probe's job.
2. `_run_activation_probe`: each FP4 element, dequantized with the scale at the
   slot this test's own routing assigns it, matches a host fp32 reference.

Usage (plain `mojo` is shadowed by the installed shmem package; use bazel):
  ./bazelw run //Mojo/tools/mojo -- \\
      max/kernels/test/gpu/shmem/test_mxfp4_fused_silu_scale_fusion.mojo
"""

from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.info import MI355X
from std.math import align_up, exp, exp2, isfinite
from std.random import random_float64, seed

from layout import Idx, TileTensor, row_major
from linalg.swiglu_activation import get_swiglu_activation
from linalg.fp4_utils import E2M1_TO_FLOAT32, MXFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import Shuffler

from shmem.ep_comm import fused_silu_mx_kernel

from std.testing import assert_equal


# ===----------------------------------------------------------------------=== #
# Input helpers
# ===----------------------------------------------------------------------=== #


def _fill_random_bf16(
    buf: HostBuffer[.bfloat16],
    n: Int,
    lo: Float64 = -4.0,
    hi: Float64 = 4.0,
):
    # Deterministic activations in [lo, hi]. Default [-4,4] stays under the
    # clamp (limit=7); the probe widens it so the OAI clamps fire.
    for i in range(n):
        buf[i] = BFloat16(random_float64(lo, hi))


def _build_routing(
    a_offsets_host: HostBuffer[.uint32],
    num_tokens_by_expert: List[Int],
):
    """Ragged per-expert prefix sums: row_offsets[0]=0, row_offsets[e+1]=sum."""
    a_offsets_host[0] = UInt32(0)
    for i in range(len(num_tokens_by_expert)):
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )


# ===----------------------------------------------------------------------=== #
# Gate 1: every real row's scales are written and nothing else.
# ===----------------------------------------------------------------------=== #


def _run_fusion_check[
    hidden_size: Int,
    NUM_ACTIVE: Int,
    clamp_activation: Bool = False,
    alpha: Float32 = 0.0,
    limit: Float32 = 0.0,
](
    name: String,
    num_tokens_by_expert: List[Int],
    inflate_padded_M: Int,
    ctx: DeviceContext,
) raises:
    comptime assert hidden_size % MXFP4_SF_VECTOR_SIZE == 0
    # fused_silu input is gate+up concatenated: input_dim == hidden_size * 2.
    comptime input_dim = hidden_size * 2
    # FP4 packs 2 elements/byte → packed output dim is hidden_size // 2.
    comptime output_dim = hidden_size // 2
    comptime scale_K = hidden_size // MXFP4_SF_VECTOR_SIZE

    # `row_offsets` MUST have a STATIC outer dim: the kernel reads
    # `row_offsets[static_shape[0] - 1]` for num_tokens (and the scale store
    # scans `static_shape[0] - 1` experts). A runtime dim makes static_shape[0]
    # == -1, so the kernel reads OOB garbage. Hence NUM_ACTIVE is comptime.
    comptime n_off = NUM_ACTIVE + 1
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

    # --- Host inputs: synthesized ragged BF16 activation + ragged routing. ---
    var input_h = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * input_dim
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](n_off)
    ctx.synchronize()
    _fill_random_bf16(input_h, total_tokens * input_dim)
    _build_routing(a_off_h, num_tokens_by_expert)

    # --- Device buffers. ---
    var input_d = ctx.enqueue_create_buffer[.bfloat16](total_tokens * input_dim)
    var a_off_d = ctx.enqueue_create_buffer[.uint32](n_off)
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(a_off_d, a_off_h)

    var input_tt = TileTensor[origin=ImmutAnyOrigin](
        input_d, row_major(total_tokens, Idx[input_dim])
    )
    var a_off_tt = TileTensor[origin=ImmutAnyOrigin](
        a_off_d, row_major[n_off]()
    )

    # The fused_silu_mx_kernel is monomorphic over layout types; bind the
    # comptime params (dtypes, layouts, threads/SMs).
    comptime out_layout = type_of(
        TileTensor[origin=MutAnyOrigin](
            input_d, row_major(total_tokens, Idx[output_dim])
        )
    ).LayoutType
    comptime slot_scales_layout = type_of(
        TileTensor[origin=MutAnyOrigin](
            input_d, row_major(num_active * max_padded_M, Idx[scale_K])
        )
    ).LayoutType

    comptime kernel = fused_silu_mx_kernel[
        DType.uint8,
        DType.float8_e8m0fnu,
        DType.bfloat16,
        out_layout,
        slot_scales_layout,
        input_tt.LayoutType,
        a_off_tt.LayoutType,
        hw.max_thread_block_size,
        hw.sm_count,
        activation=get_swiglu_activation[
            "swigluoai" if clamp_activation else "silu",
            alpha,
            limit if clamp_activation else 0.0,
        ],
    ]

    # ---- fused_silu writes the slot layout directly (S). ----
    var out_d = ctx.enqueue_create_buffer[.uint8](total_tokens * output_dim)
    var fused_scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slot_bytes)
    fused_scales_d.enqueue_fill(Float8_e8m0fnu(0))

    ctx.enqueue_function[kernel](
        TileTensor[origin=MutAnyOrigin](
            out_d, row_major(total_tokens, Idx[output_dim])
        ),
        # The kernel treats `scales` as a flat slot byte buffer via
        # `scale_4d_slot_byte_off`; the [slot_rows, scale_K] 2D wrapper just
        # supplies the base pointer and total element count.
        TileTensor[origin=MutAnyOrigin](
            fused_scales_d,
            row_major(num_active * max_padded_M, Idx[scale_K]),
        ),
        input_tt,
        a_off_tt,
        Int32(max_padded_M),
        grid_dim=hw.sm_count,
        block_dim=hw.max_thread_block_size,
    )

    var fused_host = ctx.enqueue_create_host_buffer[.uint8](slot_bytes)
    var raw_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * scale_K)
    ctx.enqueue_copy(
        fused_host, fused_scales_d.unsafe_ptr().unsafe_bitcast[UInt8]()
    )
    ctx.synchronize()

    # ---- Recover row-major [tokens, scale_K] scales (R) from S. ----
    var unwritten = 0
    for e in range(num_active):
        var token_start = Int(a_off_h[e])
        for r in range(num_tokens_by_expert[e]):
            for k in range(scale_K):
                var byte = fused_host[
                    Shuffler[1].scale_4d_slot_byte_off[K_SCALES=scale_K](
                        e, r, k, max_padded_M
                    )
                ]
                # These activations never quantize to E8M0 0x00 (2^-127), so
                # a zero is a scale the fold never wrote.
                if byte == 0:
                    unwritten += 1
                raw_h[(token_start + r) * scale_K + k] = byte
    assert_equal(unwritten, 0)

    # ---- Reference (S2): the standalone preshuffle of R. ----
    var raw_d = ctx.enqueue_create_buffer[.uint8](total_tokens * scale_K)
    var ref_d = ctx.enqueue_create_buffer[.uint8](slot_bytes)
    ctx.enqueue_copy(raw_d, raw_h)
    ref_d.enqueue_fill(UInt8(0))
    Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=scale_K](
        TileTensor[origin=ImmutAnyOrigin](
            raw_d, row_major(total_tokens, Idx[scale_K])
        ),
        TileTensor[origin=MutAnyOrigin](
            ref_d, row_major(num_active * max_padded_M, Idx[scale_K])
        ),
        a_off_tt,
        num_active,
        # The kernel's (possibly inflated) stride, not max_tokens.
        max_padded_M,
        hw.sm_count * 2,
        ctx,
    )

    # --- Compare byte for byte over the full slot region. ---
    var ref_host = ctx.enqueue_create_host_buffer[.uint8](slot_bytes)
    ctx.enqueue_copy(ref_host, ref_d)
    ctx.synchronize()

    var mismatches = 0
    for i in range(slot_bytes):
        if ref_host[i] != fused_host[i]:
            mismatches += 1
    assert_equal(mismatches, 0)
    print("    OK: ", slot_bytes, " bytes match")


# ===----------------------------------------------------------------------=== #
# Gate 2: which scale went where, and the activation math.
# ===----------------------------------------------------------------------=== #


def _run_activation_probe[
    hidden_size: Int,
    NUM_ACTIVE: Int,
    clamp_activation: Bool = False,
    alpha: Float32 = 0.0,
    limit: Float32 = 0.0,
](
    name: String,
    num_tokens_by_expert: List[Int],
    inflate_padded_M: Int,
    ctx: DeviceContext,
) raises:
    """Dequantizes the FP4 output and checks it against a host fp32 reference.

    Each element's scale is read at `scale_4d_slot_byte_off(e, r, h // 32)`,
    with expert `e` and local row `r` from this test's own routing. Random
    per-block input gains spread the scales, so the check fails if the kernel
    stores a scale at the wrong (expert, row, block) or gets the activation
    wrong. The reference is `silu(g) * u`, or with `clamp_activation` the
    OAI-clamped SwiGLU:

        g' = min(g, L);  u' = clamp(u, -L, L);  z = (u'+1)*g'*sigmoid(g'*alpha)

    Tolerance is ~1 MXFP4 quant step. Inputs reach magnitude 12 (> limit), so
    both clamps fire.
    """
    comptime input_dim = hidden_size * 2
    comptime output_dim = hidden_size // 2
    comptime scale_K = hidden_size // MXFP4_SF_VECTOR_SIZE
    comptime n_off = NUM_ACTIVE + 1
    debug_assert(len(num_tokens_by_expert) == NUM_ACTIVE)
    var total_tokens = 0
    var max_tokens = 0
    for ne in num_tokens_by_expert:
        total_tokens += ne
        max_tokens = max(max_tokens, ne)
    var max_padded_M = max(align_up(max_tokens, 32), inflate_padded_M)
    var slot_bytes = NUM_ACTIVE * max_padded_M * scale_K

    print(
        "  ",
        name,
        " active=",
        NUM_ACTIVE,
        " total_tokens=",
        total_tokens,
        " hidden=",
        hidden_size,
        " scale_K=",
        scale_K,
        " max_padded_M=",
        max_padded_M,
        " alpha=",
        alpha,
        " limit=",
        limit,
    )

    comptime hw = ctx.default_device_info

    var input_h = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * input_dim
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](n_off)
    ctx.synchronize()
    # > limit so both clamps fire.
    _fill_random_bf16(input_h, total_tokens * input_dim, -12.0, 12.0)
    # Equal scales would hide a misplaced one, so vary them per block.
    for b in range(0, total_tokens * input_dim, MXFP4_SF_VECTOR_SIZE):
        var gain = BFloat16(exp2(random_float64(-4.0, 0.0)))
        for i in range(b, b + MXFP4_SF_VECTOR_SIZE):
            input_h[i] = input_h[i] * gain
    # Token 0's first block: a huge negative gate must give 0, not NaN.
    for h in range(MXFP4_SF_VECTOR_SIZE):
        input_h[h] = BFloat16(-3.0e38)
        input_h[h + hidden_size] = BFloat16(7.0)
    _build_routing(a_off_h, num_tokens_by_expert)

    var input_d = ctx.enqueue_create_buffer[.bfloat16](total_tokens * input_dim)
    var a_off_d = ctx.enqueue_create_buffer[.uint32](n_off)
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(a_off_d, a_off_h)

    var input_tt = TileTensor[origin=ImmutAnyOrigin](
        input_d, row_major(total_tokens, Idx[input_dim])
    )
    var a_off_tt = TileTensor[origin=ImmutAnyOrigin](
        a_off_d, row_major[n_off]()
    )

    comptime out_layout = type_of(
        TileTensor[origin=MutAnyOrigin](
            input_d, row_major(total_tokens, Idx[output_dim])
        )
    ).LayoutType
    comptime scales_layout = type_of(
        TileTensor[origin=MutAnyOrigin](
            input_d, row_major(NUM_ACTIVE * max_padded_M, Idx[scale_K])
        )
    ).LayoutType

    comptime kernel = fused_silu_mx_kernel[
        DType.uint8,
        DType.float8_e8m0fnu,
        DType.bfloat16,
        out_layout,
        scales_layout,
        input_tt.LayoutType,
        a_off_tt.LayoutType,
        hw.max_thread_block_size,
        hw.sm_count,
        activation=get_swiglu_activation[
            "swigluoai" if clamp_activation else "silu",
            alpha,
            limit if clamp_activation else 0.0,
        ],
    ]

    var out_d = ctx.enqueue_create_buffer[.uint8](total_tokens * output_dim)
    var scales_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slot_bytes)
    out_d.enqueue_fill(UInt8(0))
    scales_d.enqueue_fill(Float8_e8m0fnu(0))

    ctx.enqueue_function[kernel](
        TileTensor[origin=MutAnyOrigin](
            out_d, row_major(total_tokens, Idx[output_dim])
        ),
        TileTensor[origin=MutAnyOrigin](
            scales_d, row_major(NUM_ACTIVE * max_padded_M, Idx[scale_K])
        ),
        input_tt,
        a_off_tt,
        Int32(max_padded_M),
        grid_dim=hw.sm_count,
        block_dim=hw.max_thread_block_size,
    )

    var out_host = ctx.enqueue_create_host_buffer[.uint8](
        total_tokens * output_dim
    )
    var scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        slot_bytes
    )
    ctx.enqueue_copy(out_host, out_d)
    ctx.enqueue_copy(scales_host, scales_d)
    ctx.synchronize()

    var violations = 0
    var max_abs_err = Float32(0.0)
    var max_ratio = Float32(0.0)
    var e = 0
    for m in range(total_tokens):
        while Int(a_off_h[e + 1]) <= m:
            e += 1
        var r = m - Int(a_off_h[e])
        for h in range(hidden_size):
            var gate = input_h[m * input_dim + h].cast[.float32]()
            var up = input_h[m * input_dim + h + hidden_size].cast[.float32]()
            var expected: Float32
            comptime if clamp_activation:
                var g_c = min(gate, limit)
                var u_c = min(max(up, -limit), limit)
                var sig = 1.0 / (1.0 + exp(-(g_c * alpha)))
                expected = g_c * sig * (u_c + 1.0)
            else:
                expected = gate / (1.0 + exp(-gate)) * up

            # Dequant: low nibble=even elem, high=odd; E2M1 value * block scale
            # (E8M0->f32 = 2^(byte-127), the scale fed to cvt.scalef32.pk.fp4).
            var byte = out_host[m * output_dim + (h // 2)]
            var nibble = (byte & 0xF) if (h % 2 == 0) else ((byte >> 4) & 0xF)
            var fp4 = E2M1_TO_FLOAT32[Int(nibble)]
            var s = scales_host[
                Shuffler[1].scale_4d_slot_byte_off[K_SCALES=scale_K](
                    e, r, h // MXFP4_SF_VECTOR_SIZE, max_padded_M
                )
            ].cast[.float32]()
            var dequant = fp4 * s

            var err = abs(dequant - expected)
            if not isfinite(dequant):
                violations += 1
            # ~1 quant step (= block scale). 1.10x covers the 4->6 midpoint
            # (err==1.0*S) + transcendental jitter, still fails a wrong clamp.
            var tol = 1.10 * s
            if err > tol:
                violations += 1
            if err > max_abs_err:
                max_abs_err = err
            if s > 0.0 and (err / s) > max_ratio:
                max_ratio = err / s

    print(
        "    max_abs_err=",
        max_abs_err,
        " worst err/S=",
        max_ratio,
        " violations=",
        violations,
    )
    assert_equal(violations, 0)
    print(
        "    OK: activation probe passed (",
        total_tokens * hidden_size,
        " elems)",
    )


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp4_fused_silu_scale_fusion currently requires MI355X"

    # KS64 = down proj: intermediate (hidden) = 2048 → scale_K = 64.
    _run_fusion_check[hidden_size=2048, NUM_ACTIVE=1](
        "down-proj-single-tiny", [1], 0, ctx
    )
    _run_fusion_check[hidden_size=2048, NUM_ACTIVE=4](
        "down-proj-decode", [1, 2, 0, 4], 0, ctx
    )
    _run_fusion_check[hidden_size=2048, NUM_ACTIVE=5](
        "down-proj-mixed", [3, 33, 0, 17, 64], 0, ctx
    )
    _run_fusion_check[hidden_size=2048, NUM_ACTIVE=3](
        "down-proj-prefill", [128, 33, 96], 0, ctx
    )
    # Inflated max_padded_M (> runtime max): experts sit at the passed stride.
    _run_fusion_check[hidden_size=2048, NUM_ACTIVE=3](
        "down-proj-inflated-stride", [5, 20, 12], 128, ctx
    )

    # KS96 = M3 down proj (hidden 3072 -> scale_K 96) with clamped SwiGLU.
    _run_fusion_check[
        hidden_size=3072,
        NUM_ACTIVE=1,
        clamp_activation=True,
        alpha=1.702,
        limit=7.0,
    ]("ks96-clamp-single-tiny", [1], 0, ctx)
    _run_fusion_check[
        hidden_size=3072,
        NUM_ACTIVE=4,
        clamp_activation=True,
        alpha=1.702,
        limit=7.0,
    ]("ks96-clamp-decode", [1, 2, 0, 4], 0, ctx)
    _run_fusion_check[
        hidden_size=3072,
        NUM_ACTIVE=3,
        clamp_activation=True,
        alpha=1.702,
        limit=7.0,
    ]("ks96-clamp-prefill", [128, 33, 96], 0, ctx)
    # Inflated max_padded_M (> runtime max), clamp on: slot-stride guard.
    _run_fusion_check[
        hidden_size=3072,
        NUM_ACTIVE=3,
        clamp_activation=True,
        alpha=1.702,
        limit=7.0,
    ]("ks96-clamp-inflated-stride", [5, 20, 12], 128, ctx)
    # Cross-coverage: KS64 with the clamp on, and KS96 with plain SiLU.
    _run_fusion_check[
        hidden_size=2048,
        NUM_ACTIVE=4,
        clamp_activation=True,
        alpha=1.702,
        limit=7.0,
    ]("ks64-clamp-decode", [1, 2, 0, 4], 0, ctx)
    _run_fusion_check[hidden_size=3072, NUM_ACTIVE=5](
        "ks96-plainsilu-mixed", [3, 33, 0, 17, 64], 0, ctx
    )
    print("All fused_silu_mx scale-fusion checks passed.")

    # M3 clamp constants: alpha=1.702, limit=7.0.
    _run_activation_probe[
        hidden_size=3072,
        NUM_ACTIVE=1,
        clamp_activation=True,
        alpha=1.702,
        limit=7.0,
    ]("ks96-activation-probe", [128], 0, ctx)
    _run_activation_probe[
        hidden_size=2048,
        NUM_ACTIVE=1,
        clamp_activation=True,
        alpha=1.702,
        limit=7.0,
    ]("ks64-activation-probe", [96], 0, ctx)
    # Several experts, an empty one, and an inflated max_padded_M.
    _run_activation_probe[hidden_size=2048, NUM_ACTIVE=3](
        "ks64-probe-inflated-stride", [3, 0, 5], 64, ctx
    )
    _run_activation_probe[
        hidden_size=3072,
        NUM_ACTIVE=3,
        clamp_activation=True,
        alpha=1.702,
        limit=7.0,
    ]("ks96-clamp-probe-inflated-stride", [3, 0, 5], 64, ctx)
    _run_activation_probe[hidden_size=2048, NUM_ACTIVE=4](
        "ks64-probe-decode", [1, 2, 0, 4], 256, ctx
    )
    _run_activation_probe[
        hidden_size=2048,
        NUM_ACTIVE=4,
        clamp_activation=True,
        alpha=1.702,
        limit=7.0,
    ]("ks64-clamp-probe-decode", [1, 2, 0, 4], 256, ctx)
    # Later experts with rows past 16 and 32.
    _run_activation_probe[hidden_size=3072, NUM_ACTIVE=5](
        "ks96-probe-mixed", [3, 33, 0, 17, 64], 0, ctx
    )
    print("All fused_silu_mx activation probes passed.")
