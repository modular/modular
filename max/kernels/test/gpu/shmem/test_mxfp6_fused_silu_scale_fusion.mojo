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
"""`fused_silu_mxfp6` direct `scale_4d` emission — FP6 sibling of
`test_mxfp8_fused_silu_scale_fusion.mojo`.

Each case runs the kernel once into a zero-filled slot buffer, then:

1. No real-row scale is left zero, and the standalone
   `preshuffle_grouped_scale_4d_gpu` of the scales read back from the slots
   rebuilds them byte for byte. Both sides go through `scale_4d_slot_byte_off`,
   so this only proves the kernel writes the real-row bytes and no others.
2. Each FP6 output, dequantized with the scale at the slot the test computes
   from its own routing (expert, local row, `k // 32`), matches a host fp32
   reference of the activation. Only this gate checks which scale lands where;
   the inputs vary in magnitude per block, so a misplaced scale differs ~90%
   of the time.

FP6 packs four codes per three bytes (`output_dim == hidden_size * 3 // 4`), so
the likely slip is a `k_scale` taken from the packed byte column instead of the
element index. That leaves real-row scales unwritten and pairs elements with
the wrong scale, so both gates catch it.

Usage (plain `mojo` is shadowed by the installed shmem package — use bazel):
  ./bazelw run //Mojo/tools/mojo -- \
      max/kernels/test/gpu/shmem/test_mxfp6_fused_silu_scale_fusion.mojo
"""

from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.info import MI355X
from std.math import align_up, exp, isfinite, ldexp
from std.memory import bitcast
from std.random import random_float64, random_ui64, seed
from std.testing import assert_equal, assert_true

from layout import Idx, TileTensor, row_major
from linalg.swiglu_activation import get_swiglu_activation
from linalg.fp6_utils import FP6Format, MXFP6_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import Shuffler
from shmem.ep_comm import fused_silu_mxfp6_kernel


def _fill_random_bf16(mut buf: HostBuffer[.bfloat16], n: Int, span: Float64):
    # Each 32-value run spans `span / 4^j`, j random in 0..3, so neighboring
    # blocks rarely share a scale and gate 2 sees a misplaced one.
    for run in range(0, n, MXFP6_SF_VECTOR_SIZE):
        var s = ldexp(span, -2 * Int32(random_ui64(0, 3)))
        for i in range(run, run + MXFP6_SF_VECTOR_SIZE):
            buf[i] = BFloat16(random_float64(-s, s))


def _build_routing(
    mut a_off: HostBuffer[.uint32], num_tokens_by_expert: List[Int]
):
    a_off[0] = UInt32(0)
    for i in range(len(num_tokens_by_expert)):
        a_off[i + 1] = a_off[i] + UInt32(num_tokens_by_expert[i])


def _decode_fp6_e2m3(code: UInt8) -> Float32:
    """Decodes one 6-bit E2M3 code to fp32 (sign, 2-bit exp, 3-bit mantissa)."""
    var sign = Float32(-1.0) if (code & 0x20) != 0 else Float32(1.0)
    var exp_bits = Int((code >> 3) & 0x3)
    var mant = Int(code & 0x7)
    if exp_bits == 0:
        return sign * (Float32(mant) / Float32(8.0))
    var m = Float32(1.0) + Float32(mant) / Float32(8.0)
    return ldexp(sign * m, Int32(exp_bits - 1))


def _unpack_fp6_code(bytes: HostBuffer[.uint8], base: Int, idx: Int) -> UInt8:
    """Reads code `idx` from a 4-codes-per-3-bytes packed run at `base`."""
    var group, lane = divmod(idx, 4)
    var b0 = Int(bytes[base + group * 3 + 0])
    var b1 = Int(bytes[base + group * 3 + 1])
    var b2 = Int(bytes[base + group * 3 + 2])
    var bits = b0 | (b1 << 8) | (b2 << 16)
    return UInt8((bits >> (lane * 6)) & 0x3F)


def _run_check[
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
    comptime assert hidden_size % MXFP6_SF_VECTOR_SIZE == 0
    comptime input_dim = hidden_size * 2
    comptime output_dim = hidden_size * 3 // 4
    comptime scale_K = hidden_size // MXFP6_SF_VECTOR_SIZE

    comptime n_off = NUM_ACTIVE + 1
    var num_active = NUM_ACTIVE
    debug_assert(len(num_tokens_by_expert) == NUM_ACTIVE)
    var total_tokens = 0
    var max_tokens = 0
    for ne in num_tokens_by_expert:
        total_tokens += ne
        max_tokens = max(max_tokens, ne)
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

    var input_h = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * input_dim
    )
    var a_off_h = ctx.enqueue_create_host_buffer[.uint32](n_off)
    ctx.synchronize()
    # Past `limit` when clamped, so both clamps fire.
    comptime span = 12.0 if clamp_activation else 4.0
    _fill_random_bf16(input_h, total_tokens * input_dim, span)
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
    comptime slot_scales_layout = type_of(
        TileTensor[origin=MutAnyOrigin](
            input_d, row_major(num_active * max_padded_M, Idx[scale_K])
        )
    ).LayoutType

    comptime kernel = fused_silu_mxfp6_kernel[
        DType.float8_e8m0fnu,
        DType.bfloat16,
        out_layout,
        slot_scales_layout,
        input_tt.LayoutType,
        a_off_tt.LayoutType,
        hw.max_thread_block_size,
        hw.sm_count,
        fp6_format=FP6Format.E2M3,
        activation=get_swiglu_activation[
            "swigluoai" if clamp_activation else "silu",
            alpha,
            limit if clamp_activation else 0.0,
        ],
    ]

    var out_d = ctx.enqueue_create_buffer[.uint8](total_tokens * output_dim)
    var slots_d = ctx.enqueue_create_buffer[.float8_e8m0fnu](slot_bytes)
    slots_d.enqueue_fill(Float8_e8m0fnu(0))

    ctx.enqueue_function[kernel](
        TileTensor[origin=MutAnyOrigin](
            out_d, row_major(total_tokens, Idx[output_dim])
        ),
        TileTensor[origin=MutAnyOrigin](
            slots_d,
            row_major(num_active * max_padded_M, Idx[scale_K]),
        ),
        input_tt,
        a_off_tt,
        Int32(max_padded_M),
        grid_dim=hw.sm_count,
        block_dim=hw.max_thread_block_size,
    )

    var slots_h = ctx.enqueue_create_host_buffer[.uint8](slot_bytes)
    var out_h = ctx.enqueue_create_host_buffer[.uint8](
        total_tokens * output_dim
    )
    var raw_h = ctx.enqueue_create_host_buffer[.uint8](total_tokens * scale_K)
    ctx.enqueue_copy(slots_h, slots_d.unsafe_ptr().unsafe_bitcast[UInt8]())
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    # Gate 1: real-row bytes written, all others still zero. Circular at real
    # rows: the preshuffle writes back through the offsets they were read from.
    var unwritten = 0
    for e in range(num_active):
        var start = Int(a_off_h[e])
        for r in range(num_tokens_by_expert[e]):
            for k in range(scale_K):
                var byte = slots_h[
                    Shuffler[1].scale_4d_slot_byte_off[K_SCALES=scale_K](
                        e, r, k, max_padded_M
                    )
                ]
                # No block here quantizes to E8M0 0x00 (2^-127).
                if byte == 0:
                    unwritten += 1
                raw_h[(start + r) * scale_K + k] = byte
    assert_equal(unwritten, 0)

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
        max_padded_M,
        hw.sm_count * 2,
        ctx,
    )

    var ref_h = ctx.enqueue_create_host_buffer[.uint8](slot_bytes)
    ctx.enqueue_copy(ref_h, ref_d)
    ctx.synchronize()

    var mismatches = 0
    for i in range(slot_bytes):
        if ref_h[i] != slots_h[i]:
            mismatches += 1
    assert_equal(mismatches, 0)

    # Gate 2: dequantize with the scale at the test's own (expert, row, k // 32)
    # slot against a host fp32 reference of the kernel's activation.
    for e in range(num_active):
        var start = Int(a_off_h[e])
        for r in range(num_tokens_by_expert[e]):
            var m = start + r
            for k in range(hidden_size):
                var g = input_h[m * input_dim + k].cast[.float32]()
                var u = input_h[m * input_dim + hidden_size + k].cast[
                    .float32
                ]()
                var want: Float32
                comptime if clamp_activation:
                    var g_c = min(g, limit)
                    var u_c = u.clamp(-limit, limit)
                    want = g_c / (1.0 + exp(-(g_c * alpha))) * (u_c + 1.0)
                else:
                    want = g / (1.0 + exp(-g)) * u
                var scale = bitcast[.float8_e8m0fnu](
                    slots_h[
                        Shuffler[1].scale_4d_slot_byte_off[K_SCALES=scale_K](
                            e, r, k // MXFP6_SF_VECTOR_SIZE, max_padded_M
                        )
                    ]
                ).cast[.float32]()
                var code = _unpack_fp6_code(out_h, m * output_dim, k)
                var got = _decode_fp6_e2m3(code) * scale
                assert_true(isfinite(got), "dequantized output must be finite")
                # Admits E2M3 rounding, not a power-of-two scale error (>= 50%).
                var tol = max(Float32(0.35) * abs(want), scale * Float32(0.75))
                assert_true(
                    abs(got - want) <= tol,
                    String(
                        "dequant mismatch at (",
                        m,
                        ",",
                        k,
                        "): got ",
                        got,
                        " want ",
                        want,
                        " scale ",
                        scale,
                    ),
                )

    print("    OK: ", slot_bytes, " scale bytes match; dequant verified")


def main() raises:
    seed(0)
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "test_mxfp6_fused_silu_scale_fusion currently requires MI355X"

    print("===> fused_silu_mxfp6: slot round trip + dequant")
    _run_check[hidden_size=3072, NUM_ACTIVE=1](
        "down-proj-single-tiny", [1], 0, ctx
    )
    _run_check[hidden_size=3072, NUM_ACTIVE=4](
        "down-proj-decode", [1, 2, 0, 4], 0, ctx
    )
    _run_check[hidden_size=3072, NUM_ACTIVE=5](
        "down-proj-mixed", [3, 33, 0, 17, 64], 0, ctx
    )
    _run_check[hidden_size=3072, NUM_ACTIVE=3](
        "down-proj-prefill", [128, 33, 96], 0, ctx
    )
    _run_check[hidden_size=3072, NUM_ACTIVE=3](
        "down-proj-inflated-stride", [5, 20, 12], 128, ctx
    )
    _run_check[hidden_size=2048, NUM_ACTIVE=2](
        "down-proj-ks64", [9, 40], 0, ctx
    )
    _run_check[
        hidden_size=3072,
        NUM_ACTIVE=2,
        clamp_activation=True,
        alpha=1.702,
        limit=7.0,
    ]("down-proj-clamped", [7, 40], 0, ctx)
