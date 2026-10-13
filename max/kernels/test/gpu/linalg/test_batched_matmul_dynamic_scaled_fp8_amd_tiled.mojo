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
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Route + numerics test for the AMD tiled batched blockwise-FP8 dispatch.

The GLM-5.3 MLA decode absorb GEMMs reach the blockwise-FP8 kernel through the
*batched* `batched_matmul_dynamic_scaled_fp8` path, not the 2D
`matmul_dynamic_scaled_fp8` path. This test pins that batched path at the two
real absorb shapes:

  * V/output absorb  ``[B, M, v_head_dim=256, kv_lora_rank=512]`` (K=512)
  * QK absorb        ``[B, M, kv_lora_rank=512, qk_nope_head_dim=192]`` (K=192)

at the production scale granularities ``(1, 128, 128)`` and ``(1, 64, 64)``.

The route gate proves the batched call takes the tiled
`blockwise_scaled_fp8_matmul_amd` kernel, not the naive per-batch loop: the
tiled kernel's fp32 accumulation order differs from the naive kernel, so on
random inputs a correct tiled route yields bf16 outputs that differ from the
naive reference. A numerics-only compare would pass either way, so the
launch-count gate additionally asserts the per-batch loop collapsed to one
launch.

Scales are drawn LOG-UNIFORM in ``[2^-4, 2^4)`` per block: a constant scale
hides a scale-indexing bug, so non-constant scales are load-bearing here.

The QK absorb also runs with an e4m3 OUTPUT: over an FP8 latent cache the
absorbed Q is staged in the cache dtype, so `c` is fp8 and the kernel casts the
fp32 accumulator on store. Those cases shrink the scale range to ``[2^-4, 2^0)``
so the K=192 sums stay inside e4m3's finite range, and compare against the
un-rounded fp32 host sum within an e4m3 half-ULP tolerance.
"""

from std.math import ceildiv, exp2, isnan

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.random import rand
from linalg.bmm import (
    _batched_matmul_dynamic_scaled_fp8_impl,
    batched_matmul_dynamic_scaled_fp8,
    batched_matmul_dynamic_scaled_fp8_naive,
)
from std.utils.index import Index


def test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    batch_size: Int,
    M: Int,
    N: Int,
    K: Int,
    check_route: Bool,
    gran_nk: Int = 128,
    expect_dispatches: Int = 1,
](ctx: DeviceContext) raises:
    comptime m_scale = 1
    comptime n_scale = gran_nk
    comptime k_scale = gran_nk
    comptime BLOCK_SCALE_K = k_scale
    comptime BLOCK_SCALE_N = n_scale
    comptime transpose_b = True
    comptime gran = Index(m_scale, n_scale, k_scale)
    # Scale-tensor block counts use ceildiv (K=192 -> 2 K-blocks); the host ref
    # must stride scales by these, not floor division, or it reads the wrong tail.
    comptime K_BLOCKS = ceildiv(K, k_scale)
    comptime N_BLOCKS = ceildiv(N, n_scale)

    var a_size = batch_size * M * K
    var b_size = batch_size * N * K
    var c_size = batch_size * M * N
    var a_scales_size = batch_size * ceildiv(K, BLOCK_SCALE_K) * M
    var b_scales_size = (
        batch_size * ceildiv(N, BLOCK_SCALE_N) * ceildiv(K, BLOCK_SCALE_K)
    )

    print(
        "== test_batched_matmul_dynamic_scaled_fp8_amd_tiled",
        " B=",
        batch_size,
        " M=",
        M,
        " N=",
        N,
        " K=",
        K,
        " scale=(",
        m_scale,
        ",",
        n_scale,
        ",",
        k_scale,
        ") check_route=",
        check_route,
        sep="",
    )

    var a_host = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[b_type](b_size)
    var c_disp_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_naive_host = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_ref_host = ctx.enqueue_create_host_buffer[.float32](c_size)
    var a_scales_host = ctx.enqueue_create_host_buffer[.float32](a_scales_size)
    var b_scales_host = ctx.enqueue_create_host_buffer[.float32](b_scales_size)

    rand(a_host.unsafe_ptr(), a_size)
    rand(b_host.unsafe_ptr(), b_size)
    rand(a_scales_host.unsafe_ptr(), a_scales_size)
    rand(b_scales_host.unsafe_ptr(), b_scales_size)
    # Log-uniform scales in [2^-4, 2^4): non-constant scales are load-bearing.
    # An e4m3 output caps the range at 2^0 so no sum reaches the 448 max.
    comptime scale_log2_span = Float32(4.0 if c_type.is_float8() else 8.0)
    for i in range(a_scales_size):
        a_scales_host[i] = exp2(scale_log2_span * a_scales_host[i] - 4.0)
    for i in range(b_scales_size):
        b_scales_host[i] = exp2(scale_log2_span * b_scales_host[i] - 4.0)

    var a_dev = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev = ctx.enqueue_create_buffer[b_type](b_size)
    var c_disp_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var c_naive_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var a_scales_dev = ctx.enqueue_create_buffer[.float32](a_scales_size)
    var b_scales_dev = ctx.enqueue_create_buffer[.float32](b_scales_size)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    ctx.enqueue_copy(b_scales_dev, b_scales_host)

    # Contiguous rank-3 row-major tensors.
    var a_tt = TileTensor(a_dev, row_major[batch_size, M, K]())
    var b_tt = TileTensor(b_dev, row_major[batch_size, N, K]())
    var c_disp_tt = TileTensor(c_disp_dev, row_major[batch_size, M, N]())
    var c_naive_tt = TileTensor(c_naive_dev, row_major[batch_size, M, N]())
    var a_scales_tt = TileTensor(
        a_scales_dev, row_major[batch_size, ceildiv(K, BLOCK_SCALE_K), M]()
    )
    var b_scales_tt = TileTensor(
        b_scales_dev,
        row_major[
            batch_size, ceildiv(N, BLOCK_SCALE_N), ceildiv(K, BLOCK_SCALE_K)
        ](),
    )

    # Dispatched batched path under test. Returns the dispatch count: 1 for the
    # single batched launch (grid.z=batch), `batch_size` on the per-batch fallback.
    var dispatched_count = _batched_matmul_dynamic_scaled_fp8_impl[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=m_scale,
        n_scale_granularity=n_scale,
        k_scale_granularity=k_scale,
        transpose_b=transpose_b,
        target="gpu",
    ](c_disp_tt, a_tt, b_tt, a_scales_tt, b_scales_tt, ctx)

    # Explicit naive per-batch reference (always the naive kernel); returns the
    # per-batch launch count (`batch_size`).
    var naive_count = batched_matmul_dynamic_scaled_fp8_naive[
        scales_granularity_mnk=gran, transpose_b=transpose_b
    ](c_naive_tt, a_tt, b_tt, a_scales_tt, b_scales_tt, ctx)

    ctx.enqueue_copy(c_disp_host, c_disp_dev)
    ctx.enqueue_copy(c_naive_host, c_naive_dev)
    ctx.synchronize()

    # Independent host fp32 reference (the final correctness guard).
    for batch in range(batch_size):
        for m in range(M):
            for n in range(N):
                var accum = Scalar[DType.float32](0)
                for k in range(K):
                    var a_val = a_host[batch * M * K + m * K + k].cast[
                        DType.float32
                    ]()
                    var b_val = b_host[batch * N * K + n * K + k].cast[
                        DType.float32
                    ]()
                    var a_scale = a_scales_host[
                        batch * K_BLOCKS * M + (k // k_scale) * M + m
                    ]
                    var b_scale = b_scales_host[
                        batch * N_BLOCKS * K_BLOCKS
                        + (n // n_scale) * K_BLOCKS
                        + (k // k_scale)
                    ]
                    accum += (
                        a_val
                        * b_val
                        * a_scale.cast[DType.float32]()
                        * b_scale.cast[DType.float32]()
                    )
                # bf16/f32 compare against the sum rounded through `c_type`;
                # e4m3 compares against the un-rounded sum (see module doc).
                comptime if c_type.is_float8():
                    c_ref_host[batch * M * N + m * N + n] = accum
                else:
                    c_ref_host[batch * M * N + m * N + n] = accum.cast[
                        c_type
                    ]().cast[DType.float32]()

    # e4m3 has 3 mantissa bits: half an ULP is at most 2^-4 of the value, so
    # a correctly rounded neighbour of the host sum sits within 1/16 of it.
    comptime rtol = Float32(
        1.0 / 16.0 + 1e-3
    ) if c_type.is_float8() else Float32(1e-2)
    comptime atol = Float32(1e-2)
    var ndiff_ref = 0
    for i in range(c_size):
        var got = c_disp_host[i].cast[DType.float32]()
        var ref_val = c_ref_host[i]
        var diff = got - ref_val
        var abs_diff = diff if diff >= 0 else -diff
        var abs_ref = ref_val if ref_val >= 0 else -ref_val
        if isnan(got) or abs_diff > atol + rtol * abs_ref:
            ndiff_ref += 1
            if ndiff_ref <= 5:
                print("  ref diff @", i, "got=", got, "ref=", ref_val)
    if ndiff_ref > 0:
        raise Error(
            "batched absorb GEMM diverged from host ref at "
            + String(ndiff_ref)
            + " of "
            + String(c_size)
            + " positions"
        )

    if check_route:
        # Route gate: the dispatched call must NOT be bitwise-identical to the
        # naive loop, else the K=512 shape never reached the tiled kernel.
        var ndiff_route = 0
        for i in range(c_size):
            # Compared in fp32: the host has no fcmp for an e4m3 scalar.
            if (
                c_disp_host[i].cast[DType.float32]()
                != c_naive_host[i].cast[DType.float32]()
            ):
                ndiff_route += 1
                if ndiff_route <= 5:
                    print(
                        "  route diff @",
                        i,
                        "dispatched=",
                        c_disp_host[i],
                        "naive=",
                        c_naive_host[i],
                    )
        if ndiff_route == 0:
            raise Error(
                "route gate failed: the dispatched batched K=512 absorb GEMM"
                " is bitwise-identical to the naive per-batch loop, so it did"
                " NOT take the tiled AMD kernel"
            )
        print(
            "  route-gate: dispatched != naive at",
            ndiff_route,
            "of",
            c_size,
            "positions (tiled route taken)",
        )

    # Launch-count gate: numerics pass whether or not the path batches, so only
    # the dispatch count proves the per-batch loop collapsed to grid.z=batch.
    if naive_count != batch_size:
        raise Error(
            "launch-count gate: naive per-batch reference should issue "
            + String(batch_size)
            + " dispatches, got "
            + String(naive_count)
        )
    if dispatched_count != expect_dispatches:
        raise Error(
            "launch-count gate: expected "
            + String(expect_dispatches)
            + " dispatch(es) for the whole batch, got "
            + String(dispatched_count)
            + " (per-batch loop not collapsed)"
        )
    print(
        "  launch-count gate: naive(per-batch ref)="
        + String(naive_count)
        + " dispatches, dispatched(batched)="
        + String(dispatched_count)
        + (
            " -- collapsed the per-batch loop into 1 launch" if expect_dispatches
            == 1 else " -- per-batch loop (this granularity is not tiled)"
        ),
    )

    # Different-data-per-batch airtightness: inputs are i.i.d. per batch, so a
    # base-pointer bug collapsing every batch onto batch 0 shows as equal outputs.
    if batch_size > 1:
        var cross_batch_diff = 0
        for batch in range(1, batch_size):
            var base_off = batch * M * N
            for i in range(M * N):
                if (
                    c_disp_host[base_off + i].cast[DType.float32]()
                    != c_disp_host[i].cast[DType.float32]()
                ):
                    cross_batch_diff += 1
                    break
        if cross_batch_diff != batch_size - 1:
            raise Error(
                "different-data-per-batch airtightness: "
                + String(cross_batch_diff)
                + " of "
                + String(batch_size - 1)
                + " non-zero batches differ from batch 0 -- per-batch outputs "
                "are identical, a base-pointer bug hidden by identical data"
            )

    print("  PASSED")


def main() raises:
    with DeviceContext() as ctx:
        # K=512, N=256 with a float32 accumulator output: the AMD tile gate
        # holds, so the dispatched batched call must take the tiled kernel.
        # float32 (not bf16) output is used for the route gate because the
        # tiled and naive kernels only differ in fp32 accumulation *order*
        # (reassociation); in bf16 that difference is below the 8-bit mantissa
        # and the two are bitwise identical, so a bf16 route gate would prove
        # nothing. In float32 the reassociation survives, so dispatched(tiled)
        # != naive on random data -> fails before (both naive, identical),
        # passes after (tiled vs naive, differ).
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float32,
            batch_size=4,
            M=1,
            N=256,
            K=512,
            check_route=True,
        ](ctx)

        # Same K=512 shape in production bf16 output: numerics vs the host ref
        # only (route already pinned above; bf16 cannot distinguish the kernels).
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            batch_size=4,
            M=1,
            N=256,
            K=512,
            check_route=False,
        ](ctx)

        # K=192, N=512 (QK-absorb shape), float32 output. Route gate proves the
        # partial-K-tail path admits K%128 != 0 to the tiled kernel.
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float32,
            batch_size=4,
            M=1,
            N=512,
            K=192,
            check_route=True,
        ](ctx)

        # Same K=192 shape in production bf16 output: numerics vs the host ref
        # only (bf16 cannot distinguish the kernels).
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            batch_size=4,
            M=1,
            N=512,
            K=192,
            check_route=False,
        ](ctx)

        # Non-power-of-two batch: grid.z=batch_size must not assume a power of
        # two, so B=3 exercises the tail (same K=512 float32 gates as above).
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float32,
            batch_size=3,
            M=1,
            N=256,
            K=512,
            check_route=True,
        ](ctx)

        # M>1 at the partial-K-tail shape: only M>1 reads an adjacent row into
        # the 128-K MFMA, so the K-tail zeroing is dead at M=1 (production warms M=8).
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float32,
            batch_size=4,
            M=8,
            N=512,
            K=192,
            check_route=True,
        ](ctx)

        # M>1 with K a whole multiple of BK (no K-tail path): pairs with the
        # case above to separate general M>1 indexing from tail-zeroing bugs.
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float32,
            batch_size=4,
            M=8,
            N=256,
            K=512,
            check_route=True,
        ](ctx)

        # K<BK (K=64) at M=8: the single slab is partial, so the K-tail zeroing
        # fires on it; non-blind (disabling it diverges at 7140/8192 positions).
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float32,
            batch_size=4,
            M=8,
            N=256,
            K=64,
            check_route=True,
        ](ctx)

        # Production granularity: GLM-5.3 hits 448 = 192+256 (not a 128
        # multiple), so `_b_scale_granularity` collapses to gcd(448%128,128)=64.
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float32,
            batch_size=4,
            M=8,
            N=512,
            K=192,
            check_route=False,
            gran_nk=64,
            expect_dispatches=1,
        ](ctx)

        # ---- granularity 64: the tiled kernel promotes per 64-K block ------
        # Production absorb shapes at granularity 64, M=8, float32. K=192 is
        # three 64-K scale blocks over two slabs; the tail slab's 2nd k-tile is skipped.
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float32,
            batch_size=4,
            M=8,
            N=512,
            K=192,
            check_route=True,
            gran_nk=64,
        ](ctx)

        # V-absorb shape at granularity 64: K=512 is eight 64-K scale blocks
        # over four slabs, two promotions per slab, no K tail.
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float32,
            batch_size=4,
            M=8,
            N=256,
            K=512,
            check_route=True,
            gran_nk=64,
        ](ctx)

        # Both production shapes at granularity 64 in the production bfloat16
        # output dtype (numerics vs the host reference only; bf16 cannot
        # distinguish the kernels).
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            batch_size=4,
            M=8,
            N=512,
            K=192,
            check_route=False,
            gran_nk=64,
        ](ctx)
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            batch_size=4,
            M=8,
            N=256,
            K=512,
            check_route=False,
            gran_nk=64,
        ](ctx)

        # M=1 (the first decode step) at granularity 64 with the K tail.
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float32,
            batch_size=4,
            M=1,
            N=512,
            K=192,
            check_route=True,
            gran_nk=64,
        ](ctx)

        # K not on a 64-K boundary (neither production shape is): K=160/K=224
        # end mid tail-slab, so the K-tail zeroing must hit a non-zero block offset.
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float32,
            batch_size=2,
            M=8,
            N=128,
            K=160,
            check_route=True,
            gran_nk=64,
        ](ctx)
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float32,
            batch_size=2,
            M=8,
            N=128,
            K=224,
            check_route=True,
            gran_nk=64,
        ](ctx)

        # ---- e4m3 OUTPUT: the dominant GLM-5.3 decode absorb GEMM ----------
        # Q@w_uk over an FP8 latent cache writes absorbed Q in e4m3, not bf16.
        # Launch-count gate is non-blind: reverting e4m3 admission = 16 dispatches.
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float8_e4m3fn,
            batch_size=16,
            M=1,
            N=512,
            K=192,
            check_route=False,
            gran_nk=64,
            expect_dispatches=1,
        ](ctx)
        test_batched_matmul_dynamic_scaled_fp8_amd_tiled[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .float8_e4m3fn,
            batch_size=16,
            M=8,
            N=512,
            K=192,
            check_route=False,
            gran_nk=64,
            expect_dispatches=1,
        ](ctx)
