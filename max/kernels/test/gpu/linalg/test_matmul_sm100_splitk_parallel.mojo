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
"""SM100 split-K: the parallel reduction (one accumulator row per epilogue
thread), the tuned (2112, 14336) dispatch, the tuned fp32-output (384, 5120) and
(512, 5120) dispatches, and the k-group tail of the split ranges."""

from std.random import rand
from std.sys import size_of
from std.time import global_perf_counter_ns

import linalg.matmul.vendor.blas as vendor_blas
from layout import Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from linalg.matmul.gpu.sm100_structured.default.dispatch import (
    DISPATCH_HIT,
    DISPATCH_MISS,
    sm100_heuristic_and_outliers_dispatch,
)
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    _get_splitk_scratch,
    _splitk_context,
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_scheduler_splitk import (
    SPLITK_COUNTER_SLOTS,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from linalg.utils import identity_compute_fn, no_epilogue_fn
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from std.utils.index import Index

comptime dtype = DType.bfloat16


def _run[
    N: Int,
    K: Int,
    config: MatmulConfig[dtype, dtype, dtype, True],
    pdl: Bool = False,
](
    ctx: DeviceContext,
    m: Int,
    a: DeviceBuffer[dtype],
    b: DeviceBuffer[dtype],
    c: DeviceBuffer[dtype],
) raises:
    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=True,
        config=config,
        pdl_level=PDLLevel.ON if pdl else PDLLevel(),
    ](
        TileTensor(c, row_major(m, Idx[N])),
        TileTensor(a, row_major(m, Idx[K])),
        TileTensor(b, row_major(Idx[N], Idx[K])),
        ctx,
    )


def _vendor[
    N: Int, K: Int
](
    ctx: DeviceContext,
    m: Int,
    a: DeviceBuffer[dtype],
    b: DeviceBuffer[dtype],
    c: DeviceBuffer[dtype],
) raises:
    vendor_blas.matmul(
        ctx,
        TileTensor(_untracked(c), row_major(m, Idx[N])),
        TileTensor(_untracked(a), row_major(m, Idx[K])),
        TileTensor(_untracked(b), row_major(Idx[N], Idx[K])),
        c_row_major=True,
        transpose_b=True,
    )


def _untracked[
    t: DType
](buf: DeviceBuffer[t]) -> MutPointer[Scalar[t], MutUntrackedOrigin]:
    # vendor BLAS takes a mutable output tensor; the caller keeps every
    # buffer alive past the call.
    return (
        buf.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


def parallel_config[
    mma_n: Int, num_split_k: Int, swap_ab: Bool, stages: Int
]() -> MatmulConfig[dtype, dtype, dtype, True]:
    return MatmulConfig[dtype, dtype, dtype, True](
        cluster_shape=(Int(1), Int(1), Int(1)),
        mma_shape=(Int(128), Int(mma_n), Int(16)),
        cta_group=1,
        AB_swapped=swap_ab,
        k_group_size=2,
        num_split_k=num_split_k,
        num_pipeline_stages=stages,
        num_accum_pipeline_stages=1,
        num_clc_pipeline_stages=0,
    )


def test_parallel_exact[
    N: Int,
    K: Int,
    config: MatmulConfig[dtype, dtype, dtype, True],
](ctx: DeviceContext, m: Int) raises:
    # Small integers keep every partial and every sum exact in fp32, so the
    # split-K output must equal vendor BLAS bit for bit whatever the
    # summation order. Each of three launches starts from a poisoned
    # workspace and C, so a launch whose counters were not left at zero
    # reads NaN partials instead of the previous launch's identical ones.
    var a_host = ctx.enqueue_create_host_buffer[dtype](m * K)
    var b_host = ctx.enqueue_create_host_buffer[dtype](N * K)
    var c_host = ctx.enqueue_create_host_buffer[dtype](m * N)
    var r_host = ctx.enqueue_create_host_buffer[dtype](m * N)
    ctx.synchronize()
    for i in range(m * K):
        a_host[i] = Scalar[dtype](Float32((i * 7 + 3) % 5 - 2))
    for i in range(N * K):
        b_host[i] = Scalar[dtype](Float32((i * 11 + 1) % 5 - 2))
    var a = ctx.enqueue_create_buffer[dtype](m * K)
    var b = ctx.enqueue_create_buffer[dtype](N * K)
    var c = ctx.enqueue_create_buffer[dtype](m * N)
    var r = ctx.enqueue_create_buffer[dtype](m * N)
    ctx.enqueue_copy(a, a_host)
    ctx.enqueue_copy(b, b_host)
    _vendor[N, K](ctx, m, a, b, r)
    ctx.enqueue_copy(r_host, r)
    for launch in range(3):
        c.enqueue_fill(Scalar[dtype](-7))
        _get_splitk_scratch(ctx).enqueue_fill(UInt8(0xFF))
        _run[N, K, config](ctx, m, a, b, c)
        ctx.enqueue_copy(c_host, c)
        ctx.synchronize()
        for i in range(m * N):
            if c_host[i] != r_host[i]:
                raise Error(
                    t"split-K parallel reduction, launch {launch}: c[{i // N},"
                    t" {i % N}] = {c_host[i]}, expected {r_host[i]} (M={m}"
                    t" N={N} K={K} MMA_N={config.mma_shape[1]}"
                    t" S={config.num_split_k} swap={config.AB_swapped})"
                )
    print(
        t"exact OK: M={m} N={N} K={K} MMA_N={config.mma_shape[1]}"
        t" S={config.num_split_k} swap={config.AB_swapped}"
    )


def test_parallel_repeatable[
    N: Int,
    K: Int,
    config: MatmulConfig[dtype, dtype, dtype, True],
    pdl: Bool,
](ctx: DeviceContext, m: Int) raises:
    # Every split sums its slice of the tile in split order, so the output is
    # bitwise identical from launch to launch whichever split finished last.
    var a_host = ctx.enqueue_create_host_buffer[dtype](m * K)
    var b_host = ctx.enqueue_create_host_buffer[dtype](N * K)
    var c0_host = ctx.enqueue_create_host_buffer[dtype](m * N)
    var c1_host = ctx.enqueue_create_host_buffer[dtype](m * N)
    var r_host = ctx.enqueue_create_host_buffer[dtype](m * N)
    ctx.synchronize()
    rand(a_host.unsafe_ptr(), m * K)
    rand(b_host.unsafe_ptr(), N * K)
    var a = ctx.enqueue_create_buffer[dtype](m * K)
    var b = ctx.enqueue_create_buffer[dtype](N * K)
    var c0 = ctx.enqueue_create_buffer[dtype](m * N)
    var c1 = ctx.enqueue_create_buffer[dtype](m * N)
    var r = ctx.enqueue_create_buffer[dtype](m * N)
    ctx.enqueue_copy(a, a_host)
    ctx.enqueue_copy(b, b_host)
    _run[N, K, config, pdl](ctx, m, a, b, c0)
    for _ in range(3):
        _run[N, K, config, pdl](ctx, m, a, b, c1)
    _vendor[N, K](ctx, m, a, b, r)
    ctx.enqueue_copy(c0_host, c0)
    ctx.enqueue_copy(c1_host, c1)
    ctx.enqueue_copy(r_host, r)
    ctx.synchronize()
    for i in range(m * N):
        if c0_host[i].to_bits() != c1_host[i].to_bits():
            raise Error(
                t"split-K output not bitwise repeatable at [{i // N},"
                t" {i % N}]: {c0_host[i]} vs {c1_host[i]}"
            )
        var got = c1_host[i].cast[DType.float32]()
        var want = r_host[i].cast[DType.float32]()
        # Negated so that a NaN counts as a mismatch.
        if not (abs(got - want) <= 1e-2 * abs(want) + 1e-1):
            raise Error(
                t"split-K output far from vendor BLAS at [{i // N},"
                t" {i % N}]: {got} vs {want}"
            )
    print(
        t"repeatable OK: M={m} N={N} K={K} MMA_N={config.mma_shape[1]}"
        t" S={config.num_split_k} pdl={pdl}"
    )


def _exact_inputs[
    N: Int, K: Int
](ctx: DeviceContext, m: Int, seed: Int) raises -> Tuple[
    DeviceBuffer[dtype], DeviceBuffer[dtype]
]:
    # Small integers keep every partial and sum exact in fp32, so any correct
    # kernel matches vendor BLAS bit for bit.
    var a_host = ctx.enqueue_create_host_buffer[dtype](m * K)
    var b_host = ctx.enqueue_create_host_buffer[dtype](N * K)
    ctx.synchronize()
    for i in range(m * K):
        a_host[i] = Scalar[dtype](Float32((i * 7 + 3 + seed) % 5 - 2))
    for i in range(N * K):
        b_host[i] = Scalar[dtype](Float32((i * 11 + 1) % 5 - 2))
    var a = ctx.enqueue_create_buffer[dtype](m * K)
    var b = ctx.enqueue_create_buffer[dtype](N * K)
    ctx.enqueue_copy(a, a_host)
    ctx.enqueue_copy(b, b_host)
    return (a^, b^)


def _expect_equal[
    N: Int
](
    ctx: DeviceContext,
    m: Int,
    c: DeviceBuffer[dtype],
    r: DeviceBuffer[dtype],
    what: String,
) raises:
    var c_host = ctx.enqueue_create_host_buffer[dtype](m * N)
    var r_host = ctx.enqueue_create_host_buffer[dtype](m * N)
    ctx.enqueue_copy(c_host, c)
    ctx.enqueue_copy(r_host, r)
    ctx.synchronize()
    for i in range(m * N):
        if c_host[i] != r_host[i]:
            raise Error(
                t"{what}: c[{i // N}, {i % N}] = {c_host[i]}, expected"
                t" {r_host[i]} (M={m} N={N})"
            )


def _expect_equal_f32[
    N: Int
](
    ctx: DeviceContext,
    m: Int,
    c: DeviceBuffer[DType.float32],
    r: DeviceBuffer[DType.float32],
    what: String,
) raises:
    var c_host = ctx.enqueue_create_host_buffer[DType.float32](m * N)
    var r_host = ctx.enqueue_create_host_buffer[DType.float32](m * N)
    ctx.enqueue_copy(c_host, c)
    ctx.enqueue_copy(r_host, r)
    ctx.synchronize()
    for i in range(m * N):
        if c_host[i] != r_host[i]:
            raise Error(
                t"{what}: c[{i // N}, {i % N}] = {c_host[i]}, expected"
                t" {r_host[i]} (M={m} N={N})"
            )


def test_back_to_back[
    N: Int,
    K: Int,
    config: MatmulConfig[dtype, dtype, dtype, True],
](ctx: DeviceContext, m: Int) raises:
    # Two PDL launches on different inputs, the second overlapping the first:
    # one into its own C, then both into a shared C. A launch that reads the
    # other's partials, or a counter not back at zero, changes the result.
    var a0b = _exact_inputs[N, K](ctx, m, 0)
    var a1b = _exact_inputs[N, K](ctx, m, 1)
    ref a0 = a0b[0]
    ref b = a0b[1]
    ref a1 = a1b[0]
    var c0 = ctx.enqueue_create_buffer[dtype](m * N)
    var c1 = ctx.enqueue_create_buffer[dtype](m * N)
    var r0 = ctx.enqueue_create_buffer[dtype](m * N)
    var r1 = ctx.enqueue_create_buffer[dtype](m * N)
    _vendor[N, K](ctx, m, a0, b, r0)
    _vendor[N, K](ctx, m, a1, b, r1)
    _get_splitk_scratch(ctx).enqueue_fill(UInt8(0xFF))
    _run[N, K, config, True](ctx, m, a0, b, c0)
    _run[N, K, config, True](ctx, m, a1, b, c1)
    _expect_equal[N](ctx, m, c0, r0, "back-to-back, first launch")
    _expect_equal[N](ctx, m, c1, r1, "back-to-back, second launch")
    _run[N, K, config, True](ctx, m, a0, b, c0)
    _run[N, K, config, True](ctx, m, a1, b, c0)
    _expect_equal[N](ctx, m, c0, r1, "back-to-back into one C")
    print(
        t"back-to-back OK: M={m} N={N} K={K} MMA_N={config.mma_shape[1]}"
        t" S={config.num_split_k}"
    )


def test_two_streams[
    N: Int,
    K: Int,
    config: MatmulConfig[dtype, dtype, dtype, True],
](ctx: DeviceContext, m: Int) raises:
    # A second context on the device launches on its own stream, so its
    # launches overlap the first context's. Each context has its own counter
    # slot and workspace, so the interleaved launches on different inputs
    # must not see each other's partials or arrivals.
    var ctx2 = DeviceContext()
    var a0b = _exact_inputs[N, K](ctx, m, 0)
    var a1b = _exact_inputs[N, K](ctx2, m, 1)
    var c0 = ctx.enqueue_create_buffer[dtype](m * N)
    var c1 = ctx2.enqueue_create_buffer[dtype](m * N)
    var r0 = ctx.enqueue_create_buffer[dtype](m * N)
    var r1 = ctx2.enqueue_create_buffer[dtype](m * N)
    _vendor[N, K](ctx, m, a0b[0], a0b[1], r0)
    _vendor[N, K](ctx2, m, a1b[0], a1b[1], r1)
    ctx.synchronize()
    ctx2.synchronize()
    for _ in range(8):
        _run[N, K, config, True](ctx, m, a0b[0], a0b[1], c0)
        _run[N, K, config, True](ctx2, m, a1b[0], a1b[1], c1)
    _expect_equal[N](ctx, m, c0, r0, "two streams, first context")
    _expect_equal[N](ctx2, m, c1, r1, "two streams, second context")
    print(
        t"two streams OK: M={m} MMA_N={config.mma_shape[1]}"
        t" slots={_splitk_context(ctx).slot},{_splitk_context(ctx2).slot}"
    )


def test_past_last_slot[
    N: Int,
    K: Int,
    config: MatmulConfig[dtype, dtype, dtype, True],
](m: Int) raises:
    # Contexts past the last counter slot run on per-call buffers whose
    # counters are cleared in stream order. The contexts stay alive together:
    # a context created where a destroyed one lived takes over its slot.
    var contexts = List[DeviceContext]()
    for _ in range(SPLITK_COUNTER_SLOTS + 1):
        contexts.append(DeviceContext())
    var fallback = 0
    for c in contexts:
        var ab = _exact_inputs[N, K](c, m, 2)
        var out = c.enqueue_create_buffer[dtype](m * N)
        var r = c.enqueue_create_buffer[dtype](m * N)
        _vendor[N, K](c, m, ab[0], ab[1], r)
        for _ in range(3):
            out.enqueue_fill(Scalar[dtype](-7))
            _run[N, K, config, True](c, m, ab[0], ab[1], out)
        _expect_equal[N](c, m, out, r, "context past the last slot")
        if _splitk_context(c).slot < 0:
            fallback += 1
    if fallback == 0:
        raise Error("no context ran past the last counter slot")
    print(t"past last slot OK: {fallback} context(s) on per-call counters")


def test_dispatch_exact[N: Int, K: Int](ctx: DeviceContext, m: Int) raises:
    # The production entry point (PDL on), whichever tuned row, heuristic
    # config or split-K reduction it takes for this M.
    var ab = _exact_inputs[N, K](ctx, m, 0)
    ref a = ab[0]
    ref b = ab[1]
    var c = ctx.enqueue_create_buffer[dtype](m * N)
    var r = ctx.enqueue_create_buffer[dtype](m * N)
    _vendor[N, K](ctx, m, a, b, r)
    _get_splitk_scratch(ctx).enqueue_fill(UInt8(0xFF))
    for _ in range(2):
        c.enqueue_fill(Scalar[dtype](-7))
        _matmul_gpu[use_tensor_core=True, transpose_b=True](
            TileTensor(c, row_major(m, Idx[N])),
            TileTensor(a, row_major(m, Idx[K])).as_imm(),
            TileTensor(b, row_major(Idx[N], Idx[K])).as_imm(),
            ctx,
        )
        _expect_equal[N](ctx, m, c, r, "dispatch")
    print(t"dispatch exact OK: M={m} N={N} K={K}")


def test_dispatch_f32_out[
    N: Int, K: Int
](ctx: DeviceContext, m: Int, expect_tuned: Bool) raises:
    # bf16 inputs with fp32 output take a tuned row or vendor BLAS, never the
    # heuristic configs.
    comptime c_type = DType.float32
    var ab = _exact_inputs[N, K](ctx, m, 1)
    ref a = ab[0]
    ref b = ab[1]
    var c = ctx.enqueue_create_buffer[c_type](m * N)
    var r = ctx.enqueue_create_buffer[c_type](m * N)
    var c_tile = TileTensor(c, row_major(m, Idx[N]))
    var a_tile = TileTensor(a, row_major(m, Idx[K])).as_imm()
    var b_tile = TileTensor(b, row_major(Idx[N], Idx[K])).as_imm()
    vendor_blas.matmul(
        ctx,
        TileTensor(_untracked(r), row_major(m, Idx[N])),
        TileTensor(_untracked(a), row_major(m, Idx[K])),
        TileTensor(_untracked(b), row_major(Idx[N], Idx[K])),
        c_row_major=True,
        transpose_b=True,
    )
    var status = sm100_heuristic_and_outliers_dispatch[
        transpose_b=True, has_compute_fn=False, outliers_only=True
    ](c_tile, a_tile, b_tile, no_epilogue_fn, identity_compute_fn, ctx)
    var want = DISPATCH_HIT if expect_tuned else DISPATCH_MISS
    if status != want:
        raise Error(
            t"fp32 out: dispatch status {status}, expected {want} (M={m})"
        )
    _get_splitk_scratch(ctx).enqueue_fill(UInt8(0xFF))
    var c_host = ctx.enqueue_create_host_buffer[c_type](m * N)
    var r_host = ctx.enqueue_create_host_buffer[c_type](m * N)
    ctx.enqueue_copy(r_host, r)
    for _ in range(2):
        c.enqueue_fill(Scalar[c_type](-7))
        _matmul_gpu[use_tensor_core=True, transpose_b=True](
            c_tile, a_tile, b_tile, ctx
        )
        ctx.enqueue_copy(c_host, c)
        ctx.synchronize()
        for i in range(m * N):
            if c_host[i] != r_host[i]:
                raise Error(
                    t"fp32 out: c[{i // N}, {i % N}] = {c_host[i]}, expected"
                    t" {r_host[i]} (M={m} N={N})"
                )
    print(t"fp32 out OK: M={m} N={N} K={K} tuned={expect_tuned}")


def _late_fill_kernel(
    a: MutPointer[Scalar[dtype], MutAnyOrigin],
    n: Int32,
    spin_ns: Int64,
):
    # Releases the dependent grid at entry, then publishes the activations
    # late: a GEMM that reads A before its wait sees the poison instead.
    launch_dependent_grids()
    var t0 = global_perf_counter_ns()
    while Int64(global_perf_counter_ns() - t0) < spin_ns:
        pass
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(n), stride):
        a[i] = Scalar[dtype](Float32((i * 7 + 3) % 5 - 2))


def _slow_copy_kernel(
    src: MutPointer[Scalar[DType.float32], MutAnyOrigin],
    dst: MutPointer[Scalar[DType.float32], MutAnyOrigin],
    n: Int32,
    spin_ns: Int64,
):
    # Releases the dependent grid at entry, waits for the GEMM before it, then
    # reads C late: a GEMM after it that writes C before its own wait would be
    # seen here.
    launch_dependent_grids()
    wait_on_dependent_grids()
    var t0 = global_perf_counter_ns()
    while Int64(global_perf_counter_ns() - t0) < spin_ns:
        pass
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(n), stride):
        dst[i] = src[i]


def test_pdl_chain[N: Int, K: Int](ctx: DeviceContext, m: Int) raises:
    # The production chain through the dispatch (PDL at OVERLAP_AT_BEGINNING).
    # The producer and consumer use 32 blocks: with 512 blocks the dependent
    # grid was observed not to launch early at all, which would leave a missing
    # wait undetected.
    #   late producer of A -> GEMM -> slow consumer of C -> GEMM into the same C
    # The producer and the consumer release their dependents at entry, so both
    # GEMMs launch while the grid before them is still running. The first GEMM
    # must wait before reading A, the second before writing C (which the
    # consumer is still reading), and the consumer must see the first GEMM's
    # whole output.
    comptime c_type = DType.float32
    comptime spin_ns = 200000
    var a_buf = ctx.enqueue_create_buffer[dtype](m * K)
    var a2b = _exact_inputs[N, K](ctx, m, 1)
    ref a2 = a2b[0]
    ref b = a2b[1]
    var c = ctx.enqueue_create_buffer[c_type](m * N)
    var o = ctx.enqueue_create_buffer[c_type](m * N)
    var r1 = ctx.enqueue_create_buffer[c_type](m * N)
    var r2 = ctx.enqueue_create_buffer[c_type](m * N)
    var a_ptr = rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
        a_buf.unsafe_ptr()
    )
    var c_ptr = rebind[MutPointer[Scalar[c_type], MutAnyOrigin]](c.unsafe_ptr())
    var o_ptr = rebind[MutPointer[Scalar[c_type], MutAnyOrigin]](o.unsafe_ptr())
    var c_tile = TileTensor(c, row_major(m, Idx[N]))
    var b_tile = TileTensor(b, row_major(Idx[N], Idx[K])).as_imm()
    # The chain only tests the split-K kernel if the dispatch picks the row: on
    # a device that cannot hold its grid, `_matmul_gpu` falls back to vendor
    # BLAS and every check below would pass without exercising it.
    var status = sm100_heuristic_and_outliers_dispatch[
        transpose_b=True, has_compute_fn=False, outliers_only=True
    ](
        c_tile,
        TileTensor(a2, row_major(m, Idx[K])).as_imm(),
        b_tile,
        no_epilogue_fn,
        identity_compute_fn,
        ctx,
    )
    if status != DISPATCH_HIT:
        raise Error(t"pdl chain: the tuned row was not selected (M={m})")
    _get_splitk_scratch(ctx).enqueue_fill(UInt8(0xFF))
    for _ in range(3):
        a_buf.enqueue_fill(Scalar[dtype](-3))
        c.enqueue_fill(Scalar[c_type](-7))
        o.enqueue_fill(Scalar[c_type](-9))
        ctx.enqueue_function[_late_fill_kernel](
            a_ptr,
            Int32(m * K),
            Int64(spin_ns),
            grid_dim=32,
            block_dim=256,
        )
        _matmul_gpu[use_tensor_core=True, transpose_b=True](
            c_tile,
            TileTensor(a_buf, row_major(m, Idx[K])).as_imm(),
            b_tile,
            ctx,
        )
        ctx.enqueue_function[_slow_copy_kernel](
            c_ptr,
            o_ptr,
            Int32(m * N),
            Int64(spin_ns),
            grid_dim=32,
            block_dim=256,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        _matmul_gpu[use_tensor_core=True, transpose_b=True](
            c_tile,
            TileTensor(a2, row_major(m, Idx[K])).as_imm(),
            b_tile,
            ctx,
        )
        ctx.synchronize()
        # `a_buf` now holds the producer's final data.
        vendor_blas.matmul(
            ctx,
            TileTensor(_untracked(r1), row_major(m, Idx[N])),
            TileTensor(_untracked(a_buf), row_major(m, Idx[K])),
            TileTensor(_untracked(b), row_major(Idx[N], Idx[K])),
            c_row_major=True,
            transpose_b=True,
        )
        vendor_blas.matmul(
            ctx,
            TileTensor(_untracked(r2), row_major(m, Idx[N])),
            TileTensor(_untracked(a2), row_major(m, Idx[K])),
            TileTensor(_untracked(b), row_major(Idx[N], Idx[K])),
            c_row_major=True,
            transpose_b=True,
        )
        _expect_equal_f32[N](ctx, m, o, r1, "pdl chain, consumer saw C")
        _expect_equal_f32[N](ctx, m, c, r2, "pdl chain, second GEMM")
    print(t"pdl chain OK: M={m} N={N} K={K}")


def test_dispatch_out_random[
    c_type: DType, N: Int, K: Int
](ctx: DeviceContext, m: Int, rel_tol: Float64) raises:
    # Non-integer inputs, so split-K accumulation and reduction order show:
    # the output against an fp64 host reference of the bf16 operands.
    var a_host = ctx.enqueue_create_host_buffer[dtype](m * K)
    var b_host = ctx.enqueue_create_host_buffer[dtype](N * K)
    ctx.synchronize()
    rand(a_host.unsafe_ptr(), m * K)
    rand(b_host.unsafe_ptr(), N * K)
    for i in range(m * K):
        a_host[i] = a_host[i] - 0.5
    for i in range(N * K):
        b_host[i] = b_host[i] - 0.5
    var a = ctx.enqueue_create_buffer[dtype](m * K)
    var b = ctx.enqueue_create_buffer[dtype](N * K)
    var c = ctx.enqueue_create_buffer[c_type](m * N)
    ctx.enqueue_copy(a, a_host)
    ctx.enqueue_copy(b, b_host)
    var c_host = ctx.enqueue_create_host_buffer[c_type](m * N)
    _get_splitk_scratch(ctx).enqueue_fill(UInt8(0xFF))
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        TileTensor(c, row_major(m, Idx[N])),
        TileTensor(a, row_major(m, Idx[K])).as_imm(),
        TileTensor(b, row_major(Idx[N], Idx[K])).as_imm(),
        ctx,
    )
    ctx.enqueue_copy(c_host, c)
    ctx.synchronize()
    var ref_host = List[Float64](length=m * N, fill=0.0)
    var peak = Float64(0)
    for i in range(m):
        for j in range(N):
            var acc = Float64(0)
            for k in range(K):
                acc += (
                    a_host[i * K + k].cast[DType.float64]()
                    * b_host[j * K + k].cast[DType.float64]()
                )
            ref_host[i * N + j] = acc
            peak = max(peak, abs(acc))
    var tol = rel_tol * peak
    for i in range(m * N):
        var got = c_host[i].cast[DType.float64]()
        # Negated so that a NaN counts as a mismatch.
        if not (abs(got - ref_host[i]) <= tol):
            raise Error(
                t"{c_type} out random: c[{i // N}, {i % N}] = {got}, expected"
                t" {ref_host[i]} +- {tol} (M={m} N={N} K={K})"
            )
    print(t"{c_type} out random OK: M={m} N={N} K={K} tol={tol}")


def test_k_group_tail[
    N: Int, K: Int, num_split_k: Int, k_group_size: Int
](ctx: DeviceContext) raises:
    # A is all ones and B[n, k] is 1 only inside K tile (n % num_k_tiles), so
    # every output is exactly BK; a K tile consumed by two splits reads
    # 2 * BK. Runs the serial reduction's 2-SM path.
    comptime M = 128
    comptime BK = TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[dtype]()
    comptime num_k_tiles = K // BK
    comptime assert K % BK == 0 and num_k_tiles % k_group_size == 0
    comptime config = MatmulConfig[dtype, dtype, dtype, True](
        cluster_shape=(Int(2), Int(1), Int(1)),
        mma_shape=(Int(128), Int(96), Int(16)),
        cta_group=2,
        num_split_k=num_split_k,
        k_group_size=k_group_size,
    )
    var a_host = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_host = ctx.enqueue_create_host_buffer[dtype](N * K)
    var c_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    ctx.synchronize()
    for i in range(M * K):
        a_host[i] = Scalar[dtype](1)
    for n in range(N):
        for k in range(K):
            b_host[n * K + k] = Scalar[dtype](
                1 if k // BK == n % num_k_tiles else 0
            )
    var a = ctx.enqueue_create_buffer[dtype](M * K)
    var b = ctx.enqueue_create_buffer[dtype](N * K)
    var c = ctx.enqueue_create_buffer[dtype](M * N)
    ctx.enqueue_copy(a, a_host)
    ctx.enqueue_copy(b, b_host)
    _run[N, K, config](ctx, M, a, b, c)
    ctx.enqueue_copy(c_host, c)
    ctx.synchronize()
    var expected = Scalar[dtype](BK)
    for i in range(M * N):
        if c_host[i] != expected:
            raise Error(
                t"split-K k-group tail: c[{i // N}, {i % N}] = {c_host[i]},"
                t" expected {expected} (S={num_split_k}, kg={k_group_size},"
                t" K={K})"
            )
    print(t"k-group tail OK: N={N} K={K} S={num_split_k} kg={k_group_size}")


def main() raises:
    with DeviceContext() as ctx:
        # KERN-2796: N=2112, K=14336 at the M bands the tuned split-K rows
        # serve, ragged M included (2112 = 16.5 weight tiles of 128).
        comptime N = 2112
        comptime K = 14336
        comptime c16 = parallel_config[16, 8, True, 8]()
        comptime c32 = parallel_config[32, 8, True, 8]()
        comptime c64 = parallel_config[64, 8, True, 6]()
        comptime c96 = parallel_config[96, 8, True, 6]()
        comptime c128 = parallel_config[128, 8, True, 6]()
        # Two 80-wide X tiles split 4 ways: the only tuned row whose second
        # tile starts past column 0 of the transposed output.
        comptime c80 = parallel_config[80, 4, True, 8]()
        for m in [1, 2, 13, 16]:
            test_parallel_exact[N, K, c16](ctx, m)
        for m in [17, 32]:
            test_parallel_exact[N, K, c32](ctx, m)
        for m in [33, 64]:
            test_parallel_exact[N, K, c64](ctx, m)
        for m in [65, 96]:
            test_parallel_exact[N, K, c96](ctx, m)
        for m in [97, 128]:
            test_parallel_exact[N, K, c128](ctx, m)
        for m in [129, 150, 160]:
            test_parallel_exact[N, K, c80](ctx, m)
        test_back_to_back[N, K, c16](ctx, 8)
        test_back_to_back[N, K, c80](ctx, 150)
        test_back_to_back[N, K, c128](ctx, 100)
        test_two_streams[N, K, c16](ctx, 8)
        test_two_streams[N, K, c128](ctx, 100)
        test_past_last_slot[N, K, c16](8)
        test_parallel_repeatable[N, K, c16, False](ctx, 8)
        test_parallel_repeatable[N, K, c16, True](ctx, 8)
        test_parallel_repeatable[N, K, c128, True](ctx, 100)
        # Not swapped: accumulator rows are output rows.
        test_parallel_exact[1024, 4096, parallel_config[64, 4, False, 6]()](
            ctx, 256
        )

        # One M per band of the tuned rows, the heuristic between them, and
        # the 2-SM rows at 385-512.
        for m in [2, 16, 32, 64, 96, 128, 129, 160, 176, 385, 448, 512]:
            test_dispatch_exact[N, K](ctx, m)

        # The tuned row covers M 1-192 (three 64-wide X tiles, 144 CTAs at
        # most); M 193 is the vendor control. M 1 takes the GEMV.
        for m in [2, 47, 64, 65, 96, 144, 191, 192]:
            test_dispatch_f32_out[384, 5120](ctx, m, True)
        test_dispatch_f32_out[384, 5120](ctx, 193, False)

        # The (512, 5120) row covers M 17-128 (up to two 64-wide X tiles, 128
        # CTAs); M 16 and 129 are the vendor controls. The row is keyed on (N, K)
        # alone, so test_dispatch_exact covers its bf16-output use too.
        for m in [17, 33, 64, 65, 100, 127, 128]:
            test_dispatch_f32_out[512, 5120](ctx, m, True)
        for m in [16, 129]:
            test_dispatch_f32_out[512, 5120](ctx, m, False)
        for m in [17, 64, 128]:
            # fp32 output: every product is exact, so the error is fp32
            # accumulation over K terms; 1e-4 of the peak is two orders above
            # it. bf16 output adds one rounding, at most 2^-9 of the value.
            test_dispatch_out_random[DType.float32, 512, 5120](ctx, m, 1e-4)
            test_dispatch_out_random[DType.bfloat16, 512, 5120](ctx, m, 4e-3)
            test_dispatch_exact[512, 5120](ctx, m)
            test_pdl_chain[512, 5120](ctx, m)

        # Splits must be whole k-groups of 2: 6 K tiles split 4 + 2 (3 + 3
        # would end mid group); 224 K tiles split 76/76/72 (S=3) and
        # 46 x 4 + 40 (S=5); S=4 (56 each) is the aligned control.
        test_k_group_tail[2112, 384, 2, 2](ctx)
        test_k_group_tail[2112, 14336, 3, 2](ctx)
        test_k_group_tail[2112, 14336, 5, 2](ctx)
        test_k_group_tail[2112, 14336, 4, 2](ctx)
