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
#
# SM100 (B200) correctness test for `fused_bias_residual_matmul_dispatch_sm100`
# (`mo.composite.matmul_add`). Exercises every dispatch path of the bias/residual
# fused matmul and confirms the residual is applied exactly once on each:
#
#   1. Native blackwell TMA-epilogue GEMM  (bf16, transpose_b, static aligned
#      N/K, runtime M>1).
#   2. GEMV fallback                        (M==1; gemv_gpu split-k).
#   3. cuBLAS fallback                      (bf16, M>1, N misaligned to 16B).
#   4. small-M native TMA path             (M>1 but small; small-M config).
#   5. dynamic M                            (runtime M; M>1 native, M==1 GEMV).
#   6. dynamic N / dynamic K                (defeats `has_static_NK` -> cuBLAS).
#   7. SM100-GEMV shapes                    (M==1, (N,K) in SM100_GEMV_SHAPES ->
#      SM100 GEMM instead of gemv_gpu).
#   8. N==1                                 (always GEMV; GEMV_KERNEL_VECTOR).
#   9. transpose_b=False                    (no SM100 GEMM gate -> cuBLAS).
#
# The residual is applied as a TMA-epilogue load on the native path (cases 1, 4)
# and as an elementwise (store) epilogue on every fallback.
#
# `run_pdl_case` repeats representative cases of each branch with programmatic
# dependent launch on, directly behind a producer grid that writes A and the
# residual, with no host sync in between (see its docstring).

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.primitives.grid_controls import PDLLevel, launch_dependent_grids
from std.memory import alloc
from std.memory.alloc import unsafe_alloc
from internal_utils import Table, assert_almost_equal
from std.math import ceildiv, nan
from std.sys.intrinsics import llvm_intrinsic
from std.random import rand
from layout import (
    Coord,
    RowMajorLayout,
    TileTensor,
    row_major,
    Idx,
)
from linalg.matmul.gpu.sm100_structured.default.dispatch_fused_bias_residual import (
    fused_bias_residual_matmul_dispatch_sm100,
)
from linalg.matmul.gpu.sm100_structured.default.dispatch import (
    matmul_dispatch_sm100,
)
from linalg.gemv import GEMVAlgorithm
from linalg.matmul.gpu.sm100_structured.default.tuning_configs import (
    TuningConfigSmallMNGemms,
    _get_tuning_list_small_MN_gemms_bf16,
)
from linalg.matmul.vendor.matmul import matmul as matmul_vendor

comptime EpilogueTile[c_type: DType] = TileTensor[
    c_type, RowMajorLayout[Int64, Int64], ImmutAnyOrigin
]


@inline(.always)
def _make_epilogue[
    c_type: DType, //
](resid_dev_buf: DeviceBuffer[c_type], epi_m: Int, epi_n: Int) -> EpilogueTile[
    c_type
]:
    """Wrap the residual/bias buffer as the dispatcher's immutable epilogue tile.

    Mirrors `FusedMatmulAdd.execute`: a 2D residual has `epi_m=M, epi_n=N`; a 1D
    bias has `epi_m=1, epi_n=N`. The layout uses `Int64` extents to match
    `RowMajorLayout[Int64, Int64]`.
    """
    return rebind[EpilogueTile[c_type]](
        TileTensor(
            resid_dev_buf.unsafe_ptr(),
            row_major(Int64(epi_m), Int64(epi_n)),
        ).as_imm()
    )


@inline(.always)
def _add_residual_and_assert[
    c_type: DType, //
](
    c_host_ptr: MutPointer[Scalar[c_type], MutAnyOrigin],
    c_ref_host_ptr: MutPointer[Scalar[c_type], MutAnyOrigin],
    resid_host_ptr: MutPointer[Scalar[c_type], MutAnyOrigin],
    M: Int,
    N: Int,
    epilogue_is_1d: Bool,
    atol: Float64,
    rtol: Float64,
) raises:
    """Add the residual to the reference on host, then compare against kernel.

    2D: `c_ref[i, j] += resid[i, j]`. 1D bias: `c_ref[i, j] += bias[j]` (row 0,
    broadcast over all M rows).
    """
    for i in range(M):
        for j in range(N):
            var ref_idx = i * N + j
            var resid_idx = j if epilogue_is_1d else (i * N + j)
            c_ref_host_ptr[ref_idx] += resid_host_ptr[resid_idx]

    assert_almost_equal(
        c_host_ptr,
        c_ref_host_ptr,
        M * N,
        atol=atol,
        rtol=rtol,
    )
    print("    PASS")


def run_case[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    static_n: Bool,
    static_k: Bool,
    epilogue_is_1d: Bool,
    transpose_b: Bool = True,
    vendor_reference: Bool = False,
](
    ctx: DeviceContext,
    M: Int,
    label: String,
    atol: Float64 = 1e-2,
    rtol: Float64 = 1e-2,
) raises:
    # The reference runs the same kernel without the residual (see below), so
    # the matmul is identical on both sides and only the residual application is
    # compared -- 1e-2 holds for every shape, no per-case loosening. A
    # dropped/double residual still shows as a ~residual-magnitude diff.
    print(
        t"[case] {label} :: dtypes=( {a_type} {b_type} {c_type} ) MNK=("
        t" {M} {N} {K} ) static(N,K)=( {static_n} {static_k} ) 1d_bias="
        t" {epilogue_is_1d}  atol= {atol}"
    )

    var a_size = M * K
    var b_size = N * K  # transpose_b: B is [N, K]
    var c_size = M * N
    var resid_size = N if epilogue_is_1d else (M * N)

    var a_host_ptr = unsafe_alloc[Scalar[a_type]](a_size)
    var b_host_ptr = unsafe_alloc[Scalar[b_type]](b_size)
    var c_host_ptr = unsafe_alloc[Scalar[c_type]](c_size)
    var c_ref_host_ptr = unsafe_alloc[Scalar[c_type]](c_size)
    var resid_host_ptr = unsafe_alloc[Scalar[c_type]](resid_size)

    var a_dev = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var resid_dev = ctx.enqueue_create_buffer[c_type](resid_size)

    rand(a_host_ptr, a_size, min=-1.0, max=1.0)
    rand(b_host_ptr, b_size, min=-1.0, max=1.0)
    rand(resid_host_ptr, resid_size, min=-2.0, max=2.0)

    ctx.enqueue_copy(a_dev, a_host_ptr)
    ctx.enqueue_copy(b_dev, b_host_ptr)
    ctx.enqueue_copy(resid_dev, resid_host_ptr)

    # Reference = the SAME kernel run WITHOUT the residual, into `c_ref`.
    # `matmul_dispatch_sm100` and the fused dispatcher route identically for a
    # given shape (except shapes only the fused dispatcher sends to vendor
    # BLAS; see `vendor_reference`), so the matmul is bit-identical on both
    # sides and the
    # comparison isolates the residual application -- every case holds at a
    # tight 1e-2 regardless of K. (A cuBLAS reference drifts by a few bf16 ulps
    # vs the Mojo kernels on long reductions, which would force loose tols.)
    # The residual is added on host in `_add_residual_and_assert`.
    var epi_m = 1 if epilogue_is_1d else M
    var epilogue = _make_epilogue(resid_dev, epi_m, N)

    comptime if static_n and static_k:
        var a_tt = TileTensor(a_dev, row_major(Int64(M), Idx[K]))
        var c_tt = TileTensor(c_dev, row_major(Int64(M), Idx[N]))
        var c_ref_tt = TileTensor(c_ref_dev, row_major(Int64(M), Idx[N]))
        # b_tt is [N, K] for transpose_b, else [K, N]; dims static either way.
        comptime if transpose_b:
            var b_tt = TileTensor(b_dev, row_major(Idx[N], Idx[K]))
            # Shapes the fused dispatcher hands to vendor BLAS while the plain
            # dispatcher keeps them on Mojo take the reference from the same
            # vendor matmul.
            comptime if vendor_reference:
                matmul_vendor[transpose_b=transpose_b](
                    c_ref_tt, a_tt.as_imm(), b_tt.as_imm(), ctx
                )
            else:
                matmul_dispatch_sm100[transpose_b=transpose_b](
                    c_ref_tt, a_tt.as_imm(), b_tt.as_imm(), ctx
                )
            fused_bias_residual_matmul_dispatch_sm100[
                transpose_b=transpose_b,
                has_epilogue_tensor=True,
                epilogue_is_1d=epilogue_is_1d,
            ](c_tt, a_tt.as_imm(), b_tt.as_imm(), epilogue, ctx)
        else:
            var b_tt = TileTensor(b_dev, row_major(Idx[K], Idx[N]))
            matmul_dispatch_sm100[transpose_b=transpose_b](
                c_ref_tt, a_tt.as_imm(), b_tt.as_imm(), ctx
            )
            fused_bias_residual_matmul_dispatch_sm100[
                transpose_b=transpose_b,
                has_epilogue_tensor=True,
                epilogue_is_1d=epilogue_is_1d,
            ](c_tt, a_tt.as_imm(), b_tt.as_imm(), epilogue, ctx)
    elif static_k and not static_n:
        # Dynamic N: static_N == -1 -> fallback. (Assumes transpose_b: B [N, K].)
        comptime assert transpose_b, "dynamic-shape cases assume transpose_b"
        var a_tt = TileTensor(a_dev, row_major(Int64(M), Idx[K]))
        var b_tt = TileTensor(b_dev, row_major(Int64(N), Idx[K]))
        var c_tt = TileTensor(c_dev, row_major(Int64(M), Int64(N)))
        var c_ref_tt = TileTensor(c_ref_dev, row_major(Int64(M), Int64(N)))
        matmul_dispatch_sm100[transpose_b=transpose_b](
            c_ref_tt, a_tt.as_imm(), b_tt.as_imm(), ctx
        )
        fused_bias_residual_matmul_dispatch_sm100[
            transpose_b=transpose_b,
            has_epilogue_tensor=True,
            epilogue_is_1d=epilogue_is_1d,
        ](c_tt, a_tt.as_imm(), b_tt.as_imm(), epilogue, ctx)
    elif static_n and not static_k:
        # Dynamic K: static_K == -1 -> fallback. (Assumes transpose_b: B [N, K].)
        comptime assert transpose_b, "dynamic-shape cases assume transpose_b"
        var a_tt = TileTensor(a_dev, row_major(Int64(M), Int64(K)))
        var b_tt = TileTensor(b_dev, row_major(Idx[N], Int64(K)))
        var c_tt = TileTensor(c_dev, row_major(Int64(M), Idx[N]))
        var c_ref_tt = TileTensor(c_ref_dev, row_major(Int64(M), Idx[N]))
        matmul_dispatch_sm100[transpose_b=transpose_b](
            c_ref_tt, a_tt.as_imm(), b_tt.as_imm(), ctx
        )
        fused_bias_residual_matmul_dispatch_sm100[
            transpose_b=transpose_b,
            has_epilogue_tensor=True,
            epilogue_is_1d=epilogue_is_1d,
        ](c_tt, a_tt.as_imm(), b_tt.as_imm(), epilogue, ctx)
    else:
        # Fully dynamic N and K. (Assumes transpose_b: B [N, K].)
        comptime assert transpose_b, "dynamic-shape cases assume transpose_b"
        var a_tt = TileTensor(a_dev, row_major(Int64(M), Int64(K)))
        var b_tt = TileTensor(b_dev, row_major(Int64(N), Int64(K)))
        var c_tt = TileTensor(c_dev, row_major(Int64(M), Int64(N)))
        var c_ref_tt = TileTensor(c_ref_dev, row_major(Int64(M), Int64(N)))
        matmul_dispatch_sm100[transpose_b=transpose_b](
            c_ref_tt, a_tt.as_imm(), b_tt.as_imm(), ctx
        )
        fused_bias_residual_matmul_dispatch_sm100[
            transpose_b=transpose_b,
            has_epilogue_tensor=True,
            epilogue_is_1d=epilogue_is_1d,
        ](c_tt, a_tt.as_imm(), b_tt.as_imm(), epilogue, ctx)

    ctx.synchronize()
    ctx.enqueue_copy(c_host_ptr, c_dev)
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_dev)
    ctx.synchronize()

    _add_residual_and_assert(
        c_host_ptr.unsafe_origin_cast[MutAnyOrigin](),
        c_ref_host_ptr.unsafe_origin_cast[MutAnyOrigin](),
        resid_host_ptr.unsafe_origin_cast[MutAnyOrigin](),
        M,
        N,
        epilogue_is_1d,
        atol,
        rtol,
    )

    a_host_ptr.unsafe_free()
    b_host_ptr.unsafe_free()
    c_host_ptr.unsafe_free()
    c_ref_host_ptr.unsafe_free()
    resid_host_ptr.unsafe_free()
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = c_ref_dev^
    _ = resid_dev^


@always_inline
def _pdl_val(idx: Int, salt: Int, seed: Int) -> Float32:
    var z = UInt64(idx) * 0x9E3779B97F4A7C15 + UInt64(seed * 4 + salt)
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float32(Int(z >> 40) % 2048 - 1024) / 1024.0


def _pdl_producer_kernel[
    dtype: DType
](
    a: MutPointer[Scalar[dtype], MutAnyOrigin],
    resid: MutPointer[Scalar[dtype], MutAnyOrigin],
    a_len: Int32,
    resid_len: Int32,
    seed: Int32,
    delay_us: Int32,
):
    """Writes A and the residual late, after releasing the dependent grid.

    The early `launch_dependent_grids` makes the dependent GEMM's CTAs
    resident while this grid sleeps for `delay_us` of wall clock (as
    `slow_producer` in `test_gemv_pdl_race.mojo` does), so the only thing
    standing between the GEMM and a stale A / residual read is its own
    `wait_on_dependent_grids`.
    """
    launch_dependent_grids()
    var i = global_idx.x
    # nanosleep takes nanoseconds and caps at about 1 ms per call.
    var remaining = Int(delay_us)
    while remaining > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
        remaining -= 1000

    if i < Int(a_len):
        a[i] = _pdl_val(i, 0, Int(seed)).cast[dtype]()
    if i < Int(resid_len):
        resid[i] = (2.0 * _pdl_val(i, 1, Int(seed))).cast[dtype]()


def _run_fused[
    N: Int, K: Int, epilogue_is_1d: Bool, level: PDLLevel
](
    ctx: DeviceContext,
    M: Int,
    mut c_dev: DeviceBuffer[DType.bfloat16],
    a_dev: DeviceBuffer[DType.bfloat16],
    b_dev: DeviceBuffer[DType.bfloat16],
    resid_dev: DeviceBuffer[DType.bfloat16],
) raises:
    var c_tt = TileTensor(c_dev, row_major(Int64(M), Idx[N]))
    var a_tt = TileTensor(a_dev, row_major(Int64(M), Idx[K]))
    var b_tt = TileTensor(b_dev, row_major(Idx[N], Idx[K]))
    var epi_m = 1 if epilogue_is_1d else M
    fused_bias_residual_matmul_dispatch_sm100[
        transpose_b=True,
        pdl_level=level,
        has_epilogue_tensor=True,
        epilogue_is_1d=epilogue_is_1d,
    ](
        c_tt,
        a_tt.as_imm(),
        b_tt.as_imm(),
        _make_epilogue(resid_dev, epi_m, N),
        ctx,
    )


def run_pdl_case[
    N: Int,
    K: Int,
    epilogue_is_1d: Bool,
    pdl_level: PDLLevel = PDLLevel.OVERLAP_AT_BEGINNING,
](ctx: DeviceContext, M: Int, label: String) raises:
    """Checks the fused dispatch behind a producer grid with PDL on.

    Every iteration enqueues a producer grid that writes a fresh A and
    residual, then the fused matmul with `pdl_level`, with no host sync
    between them, so the GEMM's wait is the only ordering. The output must
    match a `PDLLevel.OFF` run of the same dispatch on the same inputs bit
    for bit. A `ctx.synchronize()` between launches (as `run_case` does)
    would hide a missing or misplaced wait.
    """
    comptime bf16 = DType.bfloat16
    comptime iters = 3
    print("[pdl case]", label, ":: MNK=(", M, N, K, ")")

    var resid_size = N if epilogue_is_1d else M * N

    var b_host_ptr = unsafe_alloc[Scalar[bf16]](N * K)
    rand(b_host_ptr, N * K, min=-1.0, max=1.0)
    var b_dev = ctx.enqueue_create_buffer[bf16](N * K)
    ctx.enqueue_copy(b_dev, b_host_ptr)

    var a_devs = List[DeviceBuffer[bf16]]()
    var resid_devs = List[DeviceBuffer[bf16]]()
    var c_pdl_devs = List[DeviceBuffer[bf16]]()
    var c_ref_devs = List[DeviceBuffer[bf16]]()
    for _ in range(iters):
        a_devs.append(ctx.enqueue_create_buffer[bf16](M * K))
        resid_devs.append(ctx.enqueue_create_buffer[bf16](resid_size))
        c_pdl_devs.append(ctx.enqueue_create_buffer[bf16](M * N))
        c_ref_devs.append(ctx.enqueue_create_buffer[bf16](M * N))
        # Poison so a skipped or partial store cannot match the reference.
        c_pdl_devs[len(c_pdl_devs) - 1].enqueue_fill(nan[bf16]())
        c_ref_devs[len(c_ref_devs) - 1].enqueue_fill(nan[bf16]())
    ctx.synchronize()

    for i in range(iters):
        ctx.enqueue_function[_pdl_producer_kernel[bf16]](
            a_devs[i],
            resid_devs[i],
            Int32(M * K),
            Int32(resid_size),
            Int32(7 * i + 1),
            Int32(1000),
            grid_dim=ceildiv(max(M * K, resid_size), 256),
            block_dim=256,
        )
        _run_fused[N, K, epilogue_is_1d, pdl_level](
            ctx, M, c_pdl_devs[i], a_devs[i], b_dev, resid_devs[i]
        )
    ctx.synchronize()

    for i in range(iters):
        _run_fused[N, K, epilogue_is_1d, PDLLevel.OFF](
            ctx, M, c_ref_devs[i], a_devs[i], b_dev, resid_devs[i]
        )
    ctx.synchronize()

    var pdl_host = unsafe_alloc[Scalar[bf16]](M * N)
    var ref_host = unsafe_alloc[Scalar[bf16]](M * N)
    for i in range(iters):
        ctx.enqueue_copy(pdl_host, c_pdl_devs[i])
        ctx.enqueue_copy(ref_host, c_ref_devs[i])
        ctx.synchronize()
        var bad = 0
        for j in range(M * N):
            if pdl_host[j] != ref_host[j]:
                bad += 1
        if bad != 0:
            raise Error(
                label,
                ": PDL output differs from the PDL-off output in ",
                bad,
                " of ",
                M * N,
                " elements (iteration ",
                i,
                ")",
            )
    print("    PASS")

    pdl_host.unsafe_free()
    ref_host.unsafe_free()
    b_host_ptr.unsafe_free()


comptime PLAIN = 0
comptime STORE_CLOSURE = 1
comptime FUSED_RESIDUAL = 2


def _run_small_mn[
    N: Int, K: Int, variant: Int, level: PDLLevel
](
    ctx: DeviceContext,
    M: Int,
    mut out_dev: DeviceBuffer[DType.bfloat16],
    mut scratch_dev: DeviceBuffer[DType.bfloat16],
    a_dev: DeviceBuffer[DType.bfloat16],
    b_dev: DeviceBuffer[DType.bfloat16],
    resid_dev: DeviceBuffer[DType.bfloat16],
) raises:
    """Runs one of the three small-MN split-K GEMV instantiations: plain,
    store epilogue (`matmul_dispatch_sm100` with a store closure), or compute
    epilogue (the fused dispatcher with a 2D residual)."""
    var a_tt = TileTensor(a_dev, row_major(M, Idx[K]))
    var b_tt = TileTensor(b_dev, row_major(Idx[N], Idx[K]))
    comptime if variant == STORE_CLOSURE:
        # The kernel hands every value to the closure and leaves `c` alone.
        var c_tt = TileTensor(scratch_dev, row_major(M, Idx[N]))
        var d = TileTensor(out_dev, row_major(M, Idx[N]))

        def store[
            dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[dtype, width]) {var d}:
            comptime assert idx.rank == 2, "the epilogue contract is rank 2"
            d.store(idx, val.cast[DType.bfloat16]())

        matmul_dispatch_sm100[transpose_b=True, pdl_level=level](
            c_tt, a_tt.as_imm(), b_tt.as_imm(), store, ctx
        )
    elif variant == FUSED_RESIDUAL:
        var c_tt = TileTensor(out_dev, row_major(M, Idx[N]))
        fused_bias_residual_matmul_dispatch_sm100[
            transpose_b=True,
            pdl_level=level,
            has_epilogue_tensor=True,
            epilogue_is_1d=False,
        ](
            c_tt,
            a_tt.as_imm(),
            b_tt.as_imm(),
            _make_epilogue(resid_dev, M, N),
            ctx,
        )
    else:
        var c_tt = TileTensor(out_dev, row_major(M, Idx[N]))
        matmul_dispatch_sm100[transpose_b=True, pdl_level=level](
            c_tt, a_tt.as_imm(), b_tt.as_imm(), ctx
        )


def run_pdl_small_mn_case[
    N: Int, K: Int, variant: Int
](ctx: DeviceContext, M: Int, label: String) raises:
    """`run_pdl_case` for each small-MN split-K GEMV instantiation.

    A PDL-off run on the same inputs is the reference; a producer grid writes
    A (and the residual) late and there is no host sync before the matmul.
    B stays a constant weight, as the split-K GEMV's pre-wait weight prefetch
    requires.
    """
    comptime bf16 = DType.bfloat16
    comptime iters = 3
    print("[pdl small-MN case]", label, ":: MNK=(", M, N, K, ")")

    # Guard the routing: a retuned table that moves this shape off the split-K
    # GEMV would leave the cases below exercising a kernel that already waited.
    comptime small_mn_table = Table(
        _get_tuning_list_small_MN_gemms_bf16(), "small_MN_gemms_configs"
    )
    comptime small_mn_configs = small_mn_table.find(
        rule=lambda (x: TuningConfigSmallMNGemms) -> Bool: x.K == K and x.N == N
    )
    var routed_to_split_k = False
    comptime for config in small_mn_configs:
        if (
            M >= config.M
            and M < config.M_end
            and config.kernel_kind == GEMVAlgorithm.GemvSplitK
        ):
            routed_to_split_k = True
    if not routed_to_split_k:
        raise Error(
            label,
            (
                ": (M, N, K) no longer dispatches to the small-MN split-K GEMV;"
                " retarget this case"
            ),
        )

    var b_host_ptr = unsafe_alloc[Scalar[bf16]](N * K)
    rand(b_host_ptr, N * K, min=-1.0, max=1.0)
    var scratch = ctx.enqueue_create_buffer[bf16](M * N)

    var a_devs = List[DeviceBuffer[bf16]]()
    var b_devs = List[DeviceBuffer[bf16]]()
    var resid_devs = List[DeviceBuffer[bf16]]()
    var out_pdl = List[DeviceBuffer[bf16]]()
    var out_ref = List[DeviceBuffer[bf16]]()
    for _ in range(iters):
        a_devs.append(ctx.enqueue_create_buffer[bf16](M * K))
        b_devs.append(ctx.enqueue_create_buffer[bf16](N * K))
        resid_devs.append(ctx.enqueue_create_buffer[bf16](M * N))
        out_pdl.append(ctx.enqueue_create_buffer[bf16](M * N))
        out_ref.append(ctx.enqueue_create_buffer[bf16](M * N))
        ctx.enqueue_copy(b_devs[len(b_devs) - 1], b_host_ptr)
        out_pdl[len(out_pdl) - 1].enqueue_fill(nan[bf16]())
        out_ref[len(out_ref) - 1].enqueue_fill(nan[bf16]())
    ctx.synchronize()

    for i in range(iters):
        ctx.enqueue_function[_pdl_producer_kernel[bf16]](
            a_devs[i],
            resid_devs[i],
            Int32(M * K),
            Int32(M * N),
            Int32(7 * i + 1),
            Int32(1000),
            grid_dim=ceildiv(M * K, 256),
            block_dim=256,
        )
        _run_small_mn[N, K, variant, PDLLevel.OVERLAP_AT_BEGINNING](
            ctx, M, out_pdl[i], scratch, a_devs[i], b_devs[i], resid_devs[i]
        )
    ctx.synchronize()
    for i in range(iters):
        _run_small_mn[N, K, variant, PDLLevel.OFF](
            ctx, M, out_ref[i], scratch, a_devs[i], b_devs[i], resid_devs[i]
        )
    ctx.synchronize()

    var pdl_host = unsafe_alloc[Scalar[bf16]](M * N)
    var ref_host = unsafe_alloc[Scalar[bf16]](M * N)
    for i in range(iters):
        ctx.enqueue_copy(pdl_host, out_pdl[i])
        ctx.enqueue_copy(ref_host, out_ref[i])
        ctx.synchronize()
        var bad = 0
        for j in range(M * N):
            if pdl_host[j] != ref_host[j]:
                bad += 1
        if bad != 0:
            raise Error(
                label,
                ": PDL output differs from the PDL-off output in ",
                bad,
                " of ",
                M * N,
                " elements (iteration ",
                i,
                ")",
            )
    print("    PASS")

    pdl_host.unsafe_free()
    ref_host.unsafe_free()
    b_host_ptr.unsafe_free()


def main() raises:
    with DeviceContext() as ctx:
        comptime bf16 = DType.bfloat16

        # --- Case 1: native TMA epilogue (bf16, static aligned N/K, M>1) -----
        run_case[
            bf16,
            bf16,
            bf16,
            N=2048,
            K=1024,
            static_n=True,
            static_k=True,
            epilogue_is_1d=False,
        ](ctx, 512, "1-native-2D")
        run_case[
            bf16,
            bf16,
            bf16,
            N=2048,
            K=1024,
            static_n=True,
            static_k=True,
            epilogue_is_1d=True,
        ](ctx, 512, "1-native-1Dbias")

        # --- Case 2: GEMV fallback (M==1) ------------------------------------
        run_case[
            bf16,
            bf16,
            bf16,
            N=2048,
            K=1024,
            static_n=True,
            static_k=True,
            epilogue_is_1d=False,
        ](ctx, 1, "2-gemv-2D")
        run_case[
            bf16,
            bf16,
            bf16,
            N=2048,
            K=1024,
            static_n=True,
            static_k=True,
            epilogue_is_1d=True,
        ](ctx, 1, "2-gemv-1Dbias")

        # --- Case 3: cuBLAS fallback via unaligned N -------------------------
        # N=2052 -> 2052 * size_of(bf16=2) = 4104; 4104 % 16 == 8 != 0, so the
        # native gate AND the generic SM100 GEMM both decline -> cuBLAS.
        run_case[
            bf16,
            bf16,
            bf16,
            N=2052,
            K=1024,
            static_n=True,
            static_k=True,
            epilogue_is_1d=False,
        ](ctx, 256, "3-cublas-unalignedN-2D")

        # --- Case 4: small-M native TMA path (M>1 but small; aligned N/K) ----
        # M=8 still satisfies the native gate (m != 1), so it takes the TMA
        # epilogue with a small-M config (cta1 mma64x8) -- not a fallback. The
        # small_MN_gemms kernels are unreachable here: the native heuristic
        # hits first and shadows them.
        run_case[
            bf16,
            bf16,
            bf16,
            N=2048,
            K=1024,
            static_n=True,
            static_k=True,
            epilogue_is_1d=False,
        ](ctx, 8, "4-smallM-native-2D")

        # --- Case 5: dynamic M (runtime M>1 native, M==1 GEMV) ---------------
        # M is always a runtime extent in run_case, so the native-vs-GEMV
        # routing here exercises the dynamic-M decode/prefill split.
        run_case[
            bf16,
            bf16,
            bf16,
            N=2048,
            K=1024,
            static_n=True,
            static_k=True,
            epilogue_is_1d=False,
        ](ctx, 384, "5-dynM-native-2D")
        run_case[
            bf16,
            bf16,
            bf16,
            N=2048,
            K=1024,
            static_n=True,
            static_k=True,
            epilogue_is_1d=True,
        ](ctx, 1, "5-dynM-gemv-1Dbias")

        # --- Case 6: dynamic N and dynamic K -> fallback ---------------------
        run_case[
            bf16,
            bf16,
            bf16,
            N=2048,
            K=1024,
            static_n=False,
            static_k=True,
            epilogue_is_1d=False,
        ](ctx, 256, "6-dynN-2D")
        run_case[
            bf16,
            bf16,
            bf16,
            N=2048,
            K=1024,
            static_n=True,
            static_k=False,
            epilogue_is_1d=False,
        ](ctx, 256, "6-dynK-2D")

        # --- Case 7: SM100-GEMV shapes (M==1 -> SM100 GEMM, not gemv_gpu) -----
        # (N,K)=(12288,1536) is in dispatch_gemv's SM100_GEMV_SHAPES, so M==1
        # routes to sm100_heuristic_and_outliers_dispatch (SM100 GEMM) with the
        # residual as an elementwise epilogue -- NOT the plain gemv_gpu kernel.
        run_case[
            bf16,
            bf16,
            bf16,
            N=12288,
            K=1536,
            static_n=True,
            static_k=True,
            epilogue_is_1d=False,
        ](ctx, 1, "7-sm100gemv-2D")
        run_case[
            bf16,
            bf16,
            bf16,
            N=12288,
            K=1536,
            static_n=True,
            static_k=True,
            epilogue_is_1d=True,
        ](ctx, 1, "7-sm100gemv-1Dbias")

        # --- Case 8: N==1 (always routes to GEMV regardless of M) ------------
        # static_N==1 -> dispatch_gemv even at M>1: TMA needs N*size%16==0, and
        # 1*2 % 16 != 0, so the native/SM100 GEMM gates can never fire.
        run_case[
            bf16,
            bf16,
            bf16,
            N=1,
            K=1024,
            static_n=True,
            static_k=True,
            epilogue_is_1d=False,
        ](ctx, 512, "8-n1-gemv-2D")

        # --- Case 9: transpose_b=False (-> cuBLAS final fallback) ------------
        # Every SM100 GEMM gate (native + generic) requires transpose_b, so a
        # non-transposed B routes M>1 to the cuBLAS final fallback, applying the
        # residual via the elementwise wrapper lambda.
        run_case[
            bf16,
            bf16,
            bf16,
            N=2048,
            K=1024,
            static_n=True,
            static_k=True,
            epilogue_is_1d=False,
            transpose_b=False,
        ](ctx, 512, "9-transposeBfalse-2D")

        # --- Case 10: small_MN_gemms GEMV_SPLIT_K (now reached, not shadowed) -
        # (N,K)=(384,7168) is tuned in the small_MN table; the dispatcher now
        # checks it BEFORE the native gate, so M in [5,9) selects a GEMV_SPLIT_K
        # config and applies the residual via the elementwise lambda. Long K
        # reduction -> looser tolerance.
        run_case[
            bf16,
            bf16,
            bf16,
            N=384,
            K=7168,
            static_n=True,
            static_k=True,
            epilogue_is_1d=False,
        ](ctx, 8, "10-smallMN-splitk-2D")
        # 1D bias through GEMV_SPLIT_K: regression guard. This used to read past
        # the [1, N] bias buffer for output rows > 0 because gemv_split_k passed
        # the epilogue a flat (0, base_idx) coord; it now passes proper (row,
        # col), so the row-0 broadcast indexes the bias by column correctly.
        run_case[
            bf16,
            bf16,
            bf16,
            N=384,
            K=7168,
            static_n=True,
            static_k=True,
            epilogue_is_1d=True,
        ](ctx, 8, "10-smallMN-splitk-1Dbias")

        # --- Case 11: small_MN_gemms GEMM_MMA_CPASYNC ------------------------
        # Same tuned shape; M in [25,33) selects the GEMM_MMA_CPASYNC config.
        run_case[
            bf16,
            bf16,
            bf16,
            N=384,
            K=7168,
            static_n=True,
            static_k=True,
            epilogue_is_1d=False,
        ](ctx, 28, "11-smallMN-cpasync-2D")
        run_case[
            bf16,
            bf16,
            bf16,
            N=384,
            K=7168,
            static_n=True,
            static_k=True,
            epilogue_is_1d=True,
        ](ctx, 28, "11-smallMN-cpasync-1Dbias")

        # --- Case 12: low_perf_shapes -> cuBLAS ------------------------------
        # (N,K)=(2112,14336) is in low_perf_shapes; the dispatcher hands it to
        # cuBLAS (residual via the elementwise wrapper) before the native gate.
        # The plain dispatcher runs it on Mojo, so the reference is cuBLAS too.
        run_case[
            bf16,
            bf16,
            bf16,
            N=2112,
            K=14336,
            static_n=True,
            static_k=True,
            epilogue_is_1d=False,
            vendor_reference=True,
        ](ctx, 256, "12-lowperf-cublas-2D")

        # --- PDL on, behind a producer grid, no host sync -------------------
        # One case per branch of the dispatch: native TMA epilogue (2D and 1D,
        # large / small / ragged M), GEMV, small-MN split-K and cpasync, the
        # SM100-GEMV shape and N==1 GEMV. Vendor BLAS takes no PDL level, so it is
        # not repeated here.
        run_pdl_case[2048, 1024, False](ctx, 512, "pdl-native-2D")
        run_pdl_case[2048, 1024, True](ctx, 512, "pdl-native-1Dbias")
        run_pdl_case[2048, 1024, False](ctx, 8, "pdl-smallM-native-2D")
        run_pdl_case[2048, 1024, False](ctx, 37, "pdl-ragged-native-2D")
        run_pdl_case[2048, 1024, False, PDLLevel.ON](ctx, 512, "pdl-on-native")
        run_pdl_case[2048, 1024, False](ctx, 1, "pdl-gemv-2D")
        run_pdl_case[2048, 1024, True](ctx, 1, "pdl-gemv-1Dbias")
        run_pdl_case[12288, 1536, False](ctx, 1, "pdl-sm100gemv-2D")
        run_pdl_case[1, 1024, False](ctx, 512, "pdl-n1-gemv-2D")
        run_pdl_case[384, 7168, False](ctx, 8, "pdl-smallMN-splitk-2D")
        run_pdl_case[384, 7168, True](ctx, 8, "pdl-smallMN-splitk-1Dbias")
        run_pdl_small_mn_case[384, 7168, PLAIN](ctx, 3, "pdl-smallMN-plain-m3")
        run_pdl_small_mn_case[384, 7168, PLAIN](ctx, 8, "pdl-smallMN-plain-m8")
        run_pdl_small_mn_case[384, 7168, STORE_CLOSURE](
            ctx, 3, "pdl-smallMN-store-m3"
        )
        run_pdl_small_mn_case[384, 7168, STORE_CLOSURE](
            ctx, 8, "pdl-smallMN-store-m8"
        )
        run_pdl_small_mn_case[384, 7168, FUSED_RESIDUAL](
            ctx, 3, "pdl-smallMN-compute-m3"
        )
        run_pdl_small_mn_case[384, 7168, FUSED_RESIDUAL](
            ctx, 8, "pdl-smallMN-compute-m8"
        )
        run_pdl_case[384, 7168, False](ctx, 28, "pdl-smallMN-cpasync-2D")
        run_pdl_case[384, 7168, True](ctx, 28, "pdl-smallMN-cpasync-1Dbias")

        print("\n=== ALL CASES PASSED ===\n")
