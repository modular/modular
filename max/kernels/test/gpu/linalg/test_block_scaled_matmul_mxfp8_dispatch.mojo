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
"""Dispatch coverage test for the high-level MXFP8 `block_scaled_matmul`.

`block_scaled_matmul` (the entry point behind the
`mo.matmul.dynamic.block.scaled` op) routes every MXFP8 shape through the Mojo
`heuristic_and_outliers_dispatch`. That heuristic only enumerates configs for
M up to 8192, so a large-M shape such as the 30k-token prefill in KERN-3024
(M=30000, N=128, K=6144) produced a config the dispatcher never built and used
to hard-error with "heuristic dispatch found no config for this shape (the
vendor fallback is NVFP4-only)."

This test exercises both regimes against a `naive_block_scaled_matmul`
reference:
  * a heuristic-covered shape (M <= 8192) that hits the Mojo kernel, and
  * the KERN-3024 large-M shape that must now fall back to the vendor
    (cuBLASLt) MXFP8 block-scaled matmul.

It also runs every decode shape with tuning-table rows at M values inside
and at the edges of each row, and checks that a compute closure that needs
column vectors gets them from every kind of config.
"""

from std.math import ceildiv, align_up
from std.os import abort
from std.random import rand, random_ui64
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind

from internal_utils import assert_almost_equal
from layout import (
    Coord,
    CoordLike,
    Idx,
    TileTensor,
    row_major,
)
from std.testing import assert_equal, assert_raises

from linalg.block_scaled_quantization import (
    block_scaled_matmul,
    naive_block_scaled_matmul,
)
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
    set_scale_factor,
)
from linalg.matmul.gpu.sm100.block_scaled_dispatch import (
    DISPATCH_HIT,
    heuristic_and_outliers_dispatch,
)
from linalg.utils import _as_compute_fn


def test_mxfp8_dispatch[
    NType: CoordLike,
    KType: CoordLike,
    //,
    input_type: DType,
    output_type: DType,
    pdl_level: PDLLevel = PDLLevel.OFF,
    check_vector_compute: Bool = False,
    col_period: Int = 0,
    col_start: Int = 0,
    pass_col_start: Int = col_start,
](
    ctx: DeviceContext,
    m: Int,
    n: NType,
    k: KType,
    expect_heuristic_hit: Bool,
) raises:
    """Run the high-level `block_scaled_matmul` for an MXFP8 shape and compare
    against the naive reference. N and K are static; M is dynamic (matching the
    dynamic token dimension of the failing op).

    When `expect_heuristic_hit` is set, also runs
    `heuristic_and_outliers_dispatch` with an epilogue closure and with a
    compute closure, each against the same reference. With
    `check_vector_compute`, also runs a closure that swaps each pair of
    columns, which only works on column vectors. With `col_period`, the
    closure swaps only the columns at or past `col_start` in each period,
    and the matmul is told that the changed columns start at
    `pass_col_start`."""
    var M = m
    var N = Int(n)
    var K = Int(k)

    comptime scales_type = MXFP8_SF_DTYPE
    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE

    print(
        t"in/out dtypes=({input_type}, {output_type}, {scales_type})  problem"
        t" shape=(M={M}, N={N}, K={K})"
    )

    var a_shape = row_major(M, Idx[KType.static_value])
    var b_shape = row_major(Idx[NType.static_value], Idx[KType.static_value])
    var c_shape = row_major(M, Idx[NType.static_value])

    var a_scales_shape = row_major(
        ceildiv(M, SF_MN_GROUP_SIZE),
        Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var b_scales_shape = row_major(
        Idx[ceildiv(NType.static_value, SF_MN_GROUP_SIZE)],
        Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var a_size = M * K
    var b_size = N * K
    var c_size = M * N
    var a_scales_size = a_scales_shape.product()
    var b_scales_size = b_scales_shape.product()

    var a_host_ptr = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[input_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[output_type](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[output_type](c_size)

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        a_scales_size
    )
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        b_scales_size
    )

    var a_device = ctx.enqueue_create_buffer[input_type](a_size)
    var b_device = ctx.enqueue_create_buffer[input_type](b_size)
    var c_device = ctx.enqueue_create_buffer[output_type](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[output_type](c_size)
    var a_scales_device = ctx.enqueue_create_buffer[scales_type](a_scales_size)
    var b_scales_device = ctx.enqueue_create_buffer[scales_type](b_scales_size)

    rand(a_host_ptr.unsafe_ptr(), a_size)
    rand(b_host_ptr.unsafe_ptr(), b_size)

    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    # NOTE: unused scales must be 0.0 or we hit accuracy issues.
    for idx0 in range(align_up(M, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 < M and idx1 < K:
                var scale_value = (
                    (1 << random_ui64(0, 3))
                    .cast[.float32]()
                    .cast[scales_type]()
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host, idx0, idx1, scale_value
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host, idx0, idx1, Scalar[scales_type](0.0)
                )

    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)
    for idx0 in range(align_up(N, SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0, align_up(K, SF_VECTOR_SIZE * SF_ATOM_K), SF_VECTOR_SIZE
        ):
            if idx0 < N and idx1 < K:
                var scale_value = (
                    (1 << random_ui64(0, 3))
                    .cast[.float32]()
                    .cast[scales_type]()
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_host, idx0, idx1, scale_value
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_host, idx0, idx1, Scalar[scales_type](0.0)
                )

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a = TileTensor(a_device, a_shape)
    var b = TileTensor(b_device, b_shape)
    var c = TileTensor(c_device, c_shape)
    var c_ref = TileTensor(c_device_ref, c_shape)
    var a_scales = TileTensor(a_scales_device, a_scales_shape)
    var b_scales = TileTensor(b_scales_device, b_scales_shape)

    # High-level dispatch: this is the path that hard-errored for large M.
    block_scaled_matmul[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=True,
        pdl_level=pdl_level,
        target="gpu",
    ](c, a, b, a_scales, b_scales, Float32(1.0), ctx)

    naive_block_scaled_matmul[
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=True,
    ](
        c_ref,
        a,
        b,
        a_scales,
        b_scales,
        ctx,
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_almost_equal(
        c_host_ptr.unsafe_ptr(),
        c_host_ref_ptr.unsafe_ptr(),
        c_size,
        atol=1e-2,
        rtol=1e-2,
    )
    print("  -> PASSED")

    if expect_heuristic_hit:

        def store_fn[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[_dtype, width]) {var c}:
            comptime assert (
                idx.rank == 2
            ), "the matmul epilogue contract is rank 2"
            c.store(idx, rebind[SIMD[output_type, width]](val))

        def identity_fn[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
            comptime assert (
                idx.rank == 2
            ), "the matmul epilogue contract is rank 2"
            return val

        comptime for use_compute in [False, True]:
            c_device.enqueue_fill(Scalar[output_type](0))
            var status: Int
            comptime if use_compute:
                status = heuristic_and_outliers_dispatch[
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE
                ](
                    c,
                    a,
                    b,
                    a_scales,
                    b_scales,
                    Float32(1.0),
                    _as_compute_fn(identity_fn),
                    ctx,
                )
            else:
                # `store_fn` owns `c`; the reference is already on the host, so
                # `c_ref` carries the output shape.
                status = heuristic_and_outliers_dispatch[
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE
                ](c_ref, a, b, a_scales, b_scales, Float32(1.0), store_fn, ctx)
            assert_equal(status, DISPATCH_HIT)
            ctx.enqueue_copy(c_host_ptr, c_device)
            ctx.synchronize()
            assert_almost_equal(
                c_host_ptr.unsafe_ptr(),
                c_host_ref_ptr.unsafe_ptr(),
                c_size,
                atol=1e-2,
                rtol=1e-2,
            )
            print("  -> PASSED (heuristic dispatch, compute=", use_compute, ")")

    comptime if check_vector_compute:

        def swap_pairs[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
            comptime assert (
                idx.rank == 2
            ), "the matmul epilogue contract is rank 2"
            comptime if width == 1:
                abort()
            else:
                comptime if col_period > 0:
                    if Int(idx[1]) % col_period < col_start:
                        return val
                var even, odd = val.deinterleave()
                return rebind[SIMD[_dtype, width]](odd.interleave(even))

        c_device.enqueue_fill(Scalar[output_type](0))
        block_scaled_matmul[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=True,
            has_compute_fn=True,
            compute_fn_needs_vectors=True,
            compute_fn_col_period=col_period,
            compute_fn_col_start=pass_col_start,
            pdl_level=pdl_level,
            target="gpu",
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            Float32(1.0),
            ctx,
            compute_fn=_as_compute_fn(swap_pairs),
        )
        ctx.enqueue_copy(c_host_ptr, c_device)
        ctx.synchronize()
        var want = c_host_ref_ptr.unsafe_ptr()
        for i in range(0, c_size, 2):
            comptime if col_period > 0:
                if i % N % col_period < col_start:
                    continue
            var even = want[unsafe_offset=i]
            want[unsafe_offset=i] = want[unsafe_offset=i + 1]
            want[unsafe_offset=i + 1] = even
        assert_almost_equal(
            c_host_ptr.unsafe_ptr(),
            want,
            c_size,
            atol=1e-2,
            rtol=1e-2,
        )
        print("  -> PASSED (column-vector compute)")

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^


def test_tuned_decode_shape[N: Int, K: Int](ctx: DeviceContext) raises:
    # M inside and at the edges of the shape's tuning-table rows, plus one
    # past the last row. PDL on turns on the rows' weight prefetch.
    for m in [17, 61, 96, 127, 129, 151, 191, 193]:
        test_mxfp8_dispatch[.float8_e4m3fn, .bfloat16](
            ctx, m, Idx[N], Idx[K], expect_heuristic_hit=True
        )
        test_mxfp8_dispatch[.float8_e4m3fn, .bfloat16, PDLLevel.ON](
            ctx, m, Idx[N], Idx[K], expect_heuristic_hit=False
        )


def main() raises:
    with DeviceContext() as ctx:
        test_tuned_decode_shape[1792, 5120](ctx)
        test_tuned_decode_shape[4096, 1280](ctx)
        test_tuned_decode_shape[4608, 5120](ctx)
        test_tuned_decode_shape[5120, 2304](ctx)
        test_tuned_decode_shape[5120, 8192](ctx)
        test_tuned_decode_shape[32768, 1280](ctx)

        # Heuristic-covered shape (M <= 8192): exercises the Mojo block-scaled
        # kernel path and guards against a regression there.
        test_mxfp8_dispatch[.float8_e4m3fn, .bfloat16](
            ctx, 4096, Idx[128], Idx[6144], expect_heuristic_hit=True
        )

        # Small-M decode regime: the heuristic routes to the small-BN kernel.
        test_mxfp8_dispatch[.float8_e4m3fn, .bfloat16](
            ctx, 16, Idx[128], Idx[6144], expect_heuristic_hit=True
        )

        # Decode-regime dense shapes (M3.1 shared expert and fused QKV). At
        # these M compute closures run separately except on small-BN; store
        # redirects stay fused. Check both against the same reference.
        for m in [8, 16, 36, 130]:
            test_mxfp8_dispatch[.float8_e4m3fn, .bfloat16](
                ctx, m, Idx[3072], Idx[6144], expect_heuristic_hit=True
            )
        test_mxfp8_dispatch[.float8_e4m3fn, .bfloat16](
            ctx, 36, Idx[6144], Idx[3072], expect_heuristic_hit=True
        )
        # m in (8, 16] with N past one wave of SM tiles skips the small-BN
        # kernel.
        test_mxfp8_dispatch[.float8_e4m3fn, .bfloat16](
            ctx, 12, Idx[9856], Idx[6144], expect_heuristic_hit=True
        )

        # A closure that needs column vectors, on a small-BN config, below
        # and above the 512-row split, and at a size where the heuristic swaps
        # A and B.
        for m in [8, 300, 513, 2053]:
            test_mxfp8_dispatch[
                .float8_e4m3fn, .bfloat16, check_vector_compute=True
            ](ctx, m, Idx[2048], Idx[1280], expect_heuristic_hit=False)

        # The same closure changing only the last 64 columns of each
        # 512-column head, as the RoPE tail does, so a separate pass visits
        # only those. Decode and prefill rows at the q projection's width,
        # with prime and ragged counts.
        for m in [1, 7, 48, 144, 300, 513, 577, 2053]:
            test_mxfp8_dispatch[
                .float8_e4m3fn,
                .bfloat16,
                check_vector_compute=True,
                col_period=512,
                col_start=448,
            ](ctx, m, Idx[32768], Idx[1280], expect_heuristic_hit=False)
        # A pass told the changed columns start 16 late must fail, both on a
        # config that swaps A and B and on one below the 512-row split.
        for m in [48, 300]:
            with assert_raises():
                test_mxfp8_dispatch[
                    .float8_e4m3fn,
                    .bfloat16,
                    check_vector_compute=True,
                    col_period=512,
                    col_start=448,
                    pass_col_start=464,
                ](ctx, m, Idx[32768], Idx[1280], expect_heuristic_hit=False)

        # KERN-3024: 30k-token prefill. The heuristic has no config for this
        # M, so the dispatcher must fall back to the vendor (cuBLASLt) MXFP8
        # block-scaled matmul instead of raising.
        test_mxfp8_dispatch[.float8_e4m3fn, .bfloat16](
            ctx, 30000, Idx[128], Idx[6144], expect_heuristic_hit=False
        )

    print("\n=== ALL TESTS PASSED ===\n")
