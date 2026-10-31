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


from std.sys import simd_width_of
from std.time import sleep

import linalg.matmul.vendor.blas as vendor_blas
from max.algorithm.functional import elementwise
from max.gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
)
from layout import Coord, Idx, DefaultEngine, TileTensor, row_major
from layout.tile_layout import Layout
from linalg.bmm import (
    _batched_matmul_gpu,
    batched_matmul,
    _reshape_tile_tensor_with_batch_to_3d,
)

from std.random import rand
from std.testing import assert_almost_equal, assert_equal, assert_true

comptime epilogue_func_type = def[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](SIMD[dtype, width]) thin -> SIMD[dtype, width]


@inline(.always)
def elementwise_epilogue_fn[
    dtype: DType,
    width: SIMDLength,
    *,
    alignment: Int = 1,
](val: SIMD[dtype, width],) -> SIMD[dtype, width]:
    return val + 2


def run_bmm_and_check_result[
    dtype: DType,
    //,
    transpose_b: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
    check_against_naive_kernel: Bool = False,
    pdl_level: PDLLevel = PDLLevel(),
](
    a_host: TileTensor[
        mut=True,
        dtype,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    b_host: TileTensor[
        mut=True,
        dtype,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    c_host: TileTensor[
        mut=True,
        dtype,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    c_host_ref: TileTensor[
        mut=True,
        dtype,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    ctx: DeviceContext,
    rtol: Float64 = 1e-3 if dtype == .float32 else 1e-2,
) raises:
    comptime assert c_host.flat_rank >= 3, "c_device must have rank >= 3"
    comptime assert (
        c_host_ref.flat_rank >= 3
    ), "c_device_ref must have rank >= 3"
    var a_size = a_host.num_elements()
    var b_size = b_host.num_elements()
    var c_size = c_host.num_elements()

    # allocate device buffers
    var a_device_buffer = ctx.enqueue_create_buffer[dtype](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[dtype](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[dtype](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[dtype](c_size)

    var a_device = TileTensor(a_device_buffer, a_host.layout)
    var b_device = TileTensor(b_device_buffer, b_host.layout)
    var c_device = TileTensor(c_device_buffer, c_host.layout)
    var c_device_ref = TileTensor(c_device_ref_buffer, c_host_ref.layout)

    var a_device_3d = _reshape_tile_tensor_with_batch_to_3d(a_device)
    var b_device_3d = _reshape_tile_tensor_with_batch_to_3d(b_device)
    var c_device_ref_3d = _reshape_tile_tensor_with_batch_to_3d(c_device_ref)
    var c_host_3d = _reshape_tile_tensor_with_batch_to_3d(c_host)
    var c_host_ref_3d = _reshape_tile_tensor_with_batch_to_3d(c_host_ref)

    rand(a_host._storage, a_size)
    rand(b_host._storage, b_size)
    c_device_buffer.enqueue_fill(0)
    c_device_ref_buffer.enqueue_fill(0)

    # Copy operands to the Device
    ctx.enqueue_copy(a_device_buffer, a_host._storage)
    ctx.enqueue_copy(b_device_buffer, b_host._storage)

    # Run BMM
    comptime c_rank = c_device.rank

    @inline(.always)
    def compute_fn[
        dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int,
    ](coord: Coord, val: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert coord.rank == c_rank
        comptime func = lambda_fn.value()
        return func(val)

    comptime if lambda_fn:
        _batched_matmul_gpu[transpose_b=transpose_b, pdl_level=pdl_level](
            c_device, a_device, b_device, compute_fn, ctx
        )
    else:
        _batched_matmul_gpu[transpose_b=transpose_b, pdl_level=pdl_level](
            c_device, a_device, b_device, ctx
        )

    ctx.synchronize()

    var b = Int(c_device_ref_3d.dim(0))
    var m = Int(c_device_ref_3d.dim(1))
    var n = Int(c_device_ref_3d.dim(2))
    var k = Int(a_device_3d.dim(2))

    # Skip equality check if N or K are 0 (causes error in vendor_blas).
    if n == 0 or k == 0:
        return
    if not ctx.target.is_nvidia_gpu() and m == 0:
        # AMD doesn't support matmul with M=0
        return

    comptime if check_against_naive_kernel:
        # erase static dimensions so that the naive kernel can be used
        _batched_matmul_gpu[transpose_b=transpose_b](
            c_device_ref.make_dynamic[.int64](),
            a_device.make_dynamic[.int64](),
            b_device.make_dynamic[.int64](),
            ctx,
        )
    else:
        var c_batch_stride = Int(c_device_ref_3d.dynamic_stride(0))
        var a_batch_stride = Int(a_device_3d.dynamic_stride(0))
        var b_batch_stride = Int(b_device_3d.dynamic_stride(0))

        for i in range(b):
            var c_ptr = c_device_ref_3d._storage.unsafe_offset(
                Scalar[a_host.linear_idx_type](i * c_batch_stride)
            )
            var a_ptr = a_device_3d._storage.unsafe_offset(
                Scalar[a_host.linear_idx_type](i * a_batch_stride)
            )
            var b_ptr = b_device_3d._storage.unsafe_offset(
                Scalar[a_host.linear_idx_type](i * b_batch_stride)
            )

            var c_buffer = TileTensor(c_ptr, row_major(m, n))
            var a_buffer = TileTensor(a_ptr, row_major(m, k))
            var b_buffer = TileTensor(
                b_ptr,
                row_major((n if transpose_b else k, k if transpose_b else n)),
            )

            vendor_blas.matmul(
                ctx,
                c_buffer,
                a_buffer,
                b_buffer,
                c_row_major=True,
                transpose_b=transpose_b,
            )

    comptime pack_size = simd_width_of[dtype, target=ctx.target]()

    @inline(.always)
    def func[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var val = c_device_ref_3d.load[width=simd_width](coord)
        comptime element_lambda = lambda_fn.value()
        var update_val = element_lambda(val)

        c_device_ref_3d.store(
            coord,
            update_val,
        )

    comptime if lambda_fn:
        elementwise[pack_size, target="gpu"](
            func,
            (b, m, n),
            ctx,
        )

    ctx.enqueue_copy(c_host._storage, c_device_buffer)
    ctx.enqueue_copy(c_host_ref._storage, c_device_ref_buffer)
    ctx.synchronize()

    for batch_idx in range(b):
        for m_idx in range(m):
            for n_idx in range(n):
                var expect = c_host_ref_3d[batch_idx, m_idx, n_idx][0]
                var actual = c_host_3d[batch_idx, m_idx, n_idx][0]

                assert_almost_equal(actual, expect, rtol=rtol)


def test_dynamic_shapes[
    dtype: DType,
    /,
    *,
    transpose_b: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    ctx: DeviceContext,
    b: Int,
    m: Int,
    n: Int,
    k: Int,
    rtol: Float64 = 1e-3 if dtype == .float32 else 1e-2,
) raises:
    # fmt: off
    print(t"test_dynamic_shapes {b} x {m} x {n} x {k} transpose_b {transpose_b}")
    # fmt: on

    var a_size = b * m * k
    var b_size = b * n * k
    var c_size = b * m * n

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)

    var a_host = TileTensor(a_host_ptr, row_major(b, m, k))
    var c_host = TileTensor(c_host_ptr, row_major(b, m, n))
    var c_host_ref = TileTensor(c_host_ref_ptr, row_major(b, m, n))

    var b_host = TileTensor(
        b_host_ptr, row_major(b, n, k)
    ) if transpose_b else TileTensor(b_host_ptr, row_major(b, k, n))
    run_bmm_and_check_result[
        transpose_b=transpose_b, lambda_fn=lambda_fn, pdl_level=pdl_level
    ](a_host, b_host, c_host, c_host_ref, ctx, rtol)


def test_static_NK[
    dtype: DType,
    /,
    *,
    N: Int,
    K: Int,
    transpose_b: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
    pdl_level: PDLLevel = PDLLevel(),
](
    ctx: DeviceContext,
    b: Int,
    m: Int,
    rtol: Float64 = 1e-3 if dtype == .float32 else 1e-2,
) raises:
    print(t"test_static_NK {b} x {m} x {N} x {K} transpose_b {transpose_b}")

    var a_size = b * m * K
    var b_size = b * N * K
    var c_size = b * m * N

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)

    var a_host = TileTensor(a_host_ptr, row_major(b, m, Idx[K]))
    var c_host = TileTensor(c_host_ptr, row_major(b, m, Idx[N]))
    var c_host_ref = TileTensor(c_host_ref_ptr, row_major(b, m, Idx[N]))

    comptime if transpose_b:
        var b_host = TileTensor(b_host_ptr, row_major(b, Idx[N], Idx[K]))
        run_bmm_and_check_result[
            transpose_b=transpose_b, lambda_fn=lambda_fn, pdl_level=pdl_level
        ](a_host, b_host, c_host, c_host_ref, ctx, rtol)

    else:
        var b_host = TileTensor(b_host_ptr, row_major(b, Idx[K], Idx[N]))
        run_bmm_and_check_result[
            transpose_b=transpose_b, lambda_fn=lambda_fn, pdl_level=pdl_level
        ](a_host, b_host, c_host, c_host_ref, ctx, rtol)


def test_static_NK_rank4[
    dtype: DType,
    /,
    *,
    N: Int,
    K: Int,
    transpose_b: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
](
    ctx: DeviceContext,
    b0: Int,
    b1: Int,
    m: Int,
    rtol: Float64 = 1e-3 if dtype == .float32 else 1e-2,
) raises:
    """
    Tests a rank-4 batched matmul, whose two leading batch dimensions the
    kernel collapses into one. With a fused epilogue this is the shape that
    regressed in GEX-4200: the tiled dispatch handed the epilogue a rank-3
    coordinate for a rank-4 output.
    """
    # fmt: off
    print(t"test_static_NK_rank4 {b0} x {b1} x {m} x {N} x {K} transpose_b {transpose_b}")
    # fmt: on

    var batch = b0 * b1
    var a_size = batch * m * K
    var b_size = batch * N * K
    var c_size = batch * m * N

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)

    var a_host = TileTensor(a_host_ptr, row_major(b0, b1, m, Idx[K]))
    var c_host = TileTensor(c_host_ptr, row_major(b0, b1, m, Idx[N]))
    var c_host_ref = TileTensor(c_host_ref_ptr, row_major(b0, b1, m, Idx[N]))

    comptime if transpose_b:
        var b_host = TileTensor(b_host_ptr, row_major(b0, b1, Idx[N], Idx[K]))
        run_bmm_and_check_result[transpose_b=transpose_b, lambda_fn=lambda_fn](
            a_host, b_host, c_host, c_host_ref, ctx, rtol
        )
    else:
        var b_host = TileTensor(b_host_ptr, row_major(b0, b1, Idx[K], Idx[N]))
        run_bmm_and_check_result[transpose_b=transpose_b, lambda_fn=lambda_fn](
            a_host, b_host, c_host, c_host_ref, ctx, rtol
        )


def test_non_row_major_layout[
    dtype: DType,
    /,
    *,
    B: Int,
    N: Int,
    K: Int,
    lambda_fn: Optional[epilogue_func_type] = None,
](
    ctx: DeviceContext,
    m: Int,
    rtol: Float64 = 1e-3 if dtype == .float32 else 1e-2,
) raises:
    """
    This function tests bacthed matmul with non-row major inputs.
    For example, A's layout could be (B, m, K):(K, B*K, 1).
    """
    print(t"test_non_row_major_layout {B} x {m} x {N} x {K}")

    var a_size = B * m * K
    var b_size = B * N * K
    var c_size = B * m * N

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)

    var a_layout = Layout((Idx[B], m, Idx[K]), (Idx[K], Idx[B * K], Idx[1]))
    var c_layout = Layout((Idx[B], m, Idx[N]), (Idx[N], Idx[B * N], Idx[1]))

    var a_host = TileTensor(a_host_ptr, a_layout)
    var c_host = TileTensor(c_host_ptr, c_layout)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_layout)

    var b_host = TileTensor(b_host_ptr, row_major(Idx[B], Idx[N], Idx[K]))
    run_bmm_and_check_result[
        transpose_b=True, lambda_fn=lambda_fn, check_against_naive_kernel=True
    ](a_host, b_host, c_host, c_host_ref, ctx, rtol)


def _mut_ptr(
    buf: DeviceBuffer[DType.bfloat16],
) -> MutPointer[Scalar[DType.bfloat16], MutAnyOrigin]:
    # The intermediates live in a `List`, so indexing them yields an
    # immutable view; the launch needs a mutable output tensor.
    return (
        buf.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutAnyOrigin]()
    )


def _launch_pdl_chain[
    pdl_level: PDLLevel, *, N: Int, K: Int
](
    ctx: DeviceContext,
    q: DeviceBuffer[DType.bfloat16],
    w1: DeviceBuffer[DType.bfloat16],
    w2: DeviceBuffer[DType.bfloat16],
    xs: List[DeviceBuffer[DType.bfloat16]],
    ys: List[DeviceBuffer[DType.bfloat16]],
    b: Int,
    m: Int,
) raises:
    for i in range(len(xs)):
        # (m, K) -> (m, N), then (m, N) -> (m, K): the second launch reads
        # what the first writes, so a wait placed too late shows up as a
        # mismatch against the PDL-off run.
        _batched_matmul_gpu[transpose_b=True, pdl_level=pdl_level](
            TileTensor(_mut_ptr(xs[i]), row_major(b, m, Idx[N])),
            TileTensor(q, row_major(b, m, Idx[K])),
            TileTensor(w1, row_major(b, Idx[N], Idx[K])),
            ctx,
        )
        _batched_matmul_gpu[transpose_b=True, pdl_level=pdl_level](
            TileTensor(_mut_ptr(ys[i]), row_major(b, m, Idx[K])),
            TileTensor(xs[i], row_major(b, m, Idx[N])),
            TileTensor(w2, row_major(b, Idx[K], Idx[N])),
            ctx,
        )


def test_pdl_chain[
    pdl_level: PDLLevel, *, N: Int, K: Int
](ctx: DeviceContext, b: Int, m: Int) raises:
    """Checks that dependent batched matmuls launched back to back with PDL
    produce bit-identical results to the same chain with PDL off.

    This is a regression guard for the whole chain, over several waves. It
    does not by itself prove the dependency wait: the grids may simply not
    overlap. `test_late_producer` is the ordering test.
    """
    comptime dtype = DType.bfloat16
    comptime reps = 6
    print(t"test_pdl_chain {b} x {m} x {N} x {K}")

    var q_size = b * m * K
    var w1_size = b * N * K
    var x_size = b * m * N
    var y_size = b * m * K

    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    var w1_host = ctx.enqueue_create_host_buffer[dtype](w1_size)
    var w2_host = ctx.enqueue_create_host_buffer[dtype](w1_size)
    rand(q_host.unsafe_ptr(), q_size)
    rand(w1_host.unsafe_ptr(), w1_size)
    rand(w2_host.unsafe_ptr(), w1_size)

    var q = ctx.enqueue_create_buffer[dtype](q_size)
    var w1 = ctx.enqueue_create_buffer[dtype](w1_size)
    var w2 = ctx.enqueue_create_buffer[dtype](w1_size)
    ctx.enqueue_copy(q, q_host)
    ctx.enqueue_copy(w1, w1_host)
    ctx.enqueue_copy(w2, w2_host)

    # Fresh zeroed intermediates per repetition, so a consumer that runs
    # ahead of its producer reads zeros instead of a stale copy of the
    # right answer.
    var xs_ref = List[DeviceBuffer[dtype]]()
    var ys_ref = List[DeviceBuffer[dtype]]()
    var xs = List[DeviceBuffer[dtype]]()
    var ys = List[DeviceBuffer[dtype]]()
    for _ in range(reps):
        xs_ref.append(ctx.enqueue_create_buffer[dtype](x_size))
        ys_ref.append(ctx.enqueue_create_buffer[dtype](y_size))
        xs.append(ctx.enqueue_create_buffer[dtype](x_size))
        ys.append(ctx.enqueue_create_buffer[dtype](y_size))
        xs_ref[len(xs_ref) - 1].enqueue_fill(0)
        ys_ref[len(ys_ref) - 1].enqueue_fill(0)
        xs[len(xs) - 1].enqueue_fill(0)
        ys[len(ys) - 1].enqueue_fill(0)

    _launch_pdl_chain[PDLLevel(), N=N, K=K](
        ctx, q, w1, w2, xs_ref, ys_ref, b, m
    )
    _launch_pdl_chain[pdl_level, N=N, K=K](ctx, q, w1, w2, xs, ys, b, m)

    var y_ref_host = ctx.enqueue_create_host_buffer[dtype](y_size)
    var y_host = ctx.enqueue_create_host_buffer[dtype](y_size)
    for i in range(reps):
        ctx.enqueue_copy(y_ref_host, ys_ref[i])
        ctx.enqueue_copy(y_host, ys[i])
        ctx.synchronize()
        var nonzero = 0
        for j in range(y_size):
            if y_ref_host[j] != 0:
                nonzero += 1
            assert_equal(Float32(y_host[j]), Float32(y_ref_host[j]))
        assert_true(nonzero > y_size // 2, "reference chain output is empty")


def late_fill_kernel(
    dst: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    n: Int32,
    delay_seconds: Float64,
):
    """Releases its dependents at entry like `OVERLAP_AT_BEGINNING`, then
    publishes `dst` only after a delay, so a consumer that reads before
    waiting for this grid sees the stale contents.

    The release comes from `launch_dependent_grids()`; the launch itself
    carries no PDL attribute, since that would let it start before the fills
    that precede it.
    """
    launch_dependent_grids()
    if thread_idx.x == 0:
        sleep(delay_seconds)
    barrier()
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(n), stride):
        dst[i] = Scalar[DType.bfloat16](1)


def copy_without_wait_kernel(
    src: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    dst: MutPointer[Scalar[DType.bfloat16], MutAnyOrigin],
    n: Int32,
):
    """Negative control: a PDL consumer that never calls
    `wait_on_dependent_grids`."""
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(n), stride):
        dst[i] = src[i]


# Long enough to dwarf the host-side cost of enqueueing the consumer, so the
# consumer is queued behind the producer while it is still running. Each test
# runs once with no delay first to load its kernels: a first launch compiles
# on the host and would otherwise outlast a short delay.
comptime _LATE_DELAY_S = 0.02


def _negative_control_once(
    ctx: DeviceContext, delay_seconds: Float64
) raises -> Int:
    comptime n = 1 << 16
    var src = ctx.enqueue_create_buffer[DType.bfloat16](n)
    var dst = ctx.enqueue_create_buffer[DType.bfloat16](n)
    src.enqueue_fill(0)
    dst.enqueue_fill(7)
    ctx.enqueue_function[late_fill_kernel](
        src,
        Int32(n),
        delay_seconds,
        grid_dim=1,
        block_dim=256,
    )
    ctx.enqueue_function[copy_without_wait_kernel](
        src,
        dst,
        Int32(n),
        grid_dim=64,
        block_dim=256,
        attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_BEGINNING),
    )
    var host = ctx.enqueue_create_host_buffer[DType.bfloat16](n)
    ctx.enqueue_copy(host, dst)
    ctx.synchronize()
    var stale = 0
    for i in range(n):
        if host[i] == 0:
            stale += 1
    return stale


def test_pdl_negative_control(ctx: DeviceContext) raises -> Bool:
    """Checks that the late-producer setup exposes a missing wait: a consumer
    launched with the PDL attribute that skips the wait should read the stale
    (zero) data while the producer sleeps.

    PDL allows early scheduling but does not guarantee it, so a consumer that
    never overlaps is an inconclusive setup, not a failure: this returns
    False and the late-producer cases then only check results. A few
    attempts are made before giving up.

    Returns:
        Whether the setup was seen to let an early reader see stale data.
    """
    print("test_pdl_negative_control")
    _ = _negative_control_once(ctx, 0.0)
    for _ in range(3):
        if _negative_control_once(ctx, _LATE_DELAY_S) > 0:
            print(
                "test_pdl_negative_control: overlap observed for the copy"
                " kernel control; whether the batched matmul cases below"
                " overlap is not established"
            )
            return True
    print(
        "WARNING: the consumer that skips the wait never overlapped the"
        " delayed producer; test_late_producer cannot detect a missing wait"
        " on this run (inconclusive, not a failure)"
    )
    return False


def _late_producer_once[
    pdl_level: PDLLevel, *, N: Int, K: Int, late_b: Bool, via_public: Bool
](
    ctx: DeviceContext, b: Int, m: Int, delay_seconds: Float64, check: Bool
) raises:
    comptime dtype = DType.bfloat16
    var a_size = b * m * K
    var b_size = b * N * K
    var c_size = b * m * N

    var a = ctx.enqueue_create_buffer[dtype](a_size)
    # In a batched matmul both operands are activations, not weights: only a
    # plain GEMM's B is a static weight that may be read before the PDL wait.
    # That is why B must be read after the wait here.
    var w = ctx.enqueue_create_buffer[dtype](b_size)
    var c = ctx.enqueue_create_buffer[dtype](c_size)
    comptime if late_b:
        a.enqueue_fill(1)
        w.enqueue_fill(0)
    else:
        a.enqueue_fill(0)
        w.enqueue_fill(1)
    c.enqueue_fill(-1)

    var late_size = b_size if late_b else a_size
    ctx.enqueue_function[late_fill_kernel](
        w if late_b else a,
        Int32(late_size),
        delay_seconds,
        grid_dim=1,
        block_dim=256,
    )

    var c_t = TileTensor(c, row_major(b, m, Idx[N]))
    var a_t = TileTensor(a, row_major(b, m, Idx[K]))
    var w_t = TileTensor(w, row_major(b, Idx[N], Idx[K]))
    comptime if via_public:
        batched_matmul[transpose_b=True, target="gpu", pdl_level=pdl_level](
            c_t, a_t.as_imm(), w_t.as_imm(), context=ctx
        )
    else:
        _batched_matmul_gpu[transpose_b=True, pdl_level=pdl_level](
            c_t, a_t.as_imm(), w_t.as_imm(), ctx
        )

    var host = ctx.enqueue_create_host_buffer[dtype](c_size)
    ctx.enqueue_copy(host, c)
    ctx.synchronize()
    if not check:
        return
    for i in range(c_size):
        assert_equal(Float32(host[i]), Float32(K))


def test_late_producer[
    pdl_level: PDLLevel, *, N: Int, K: Int, late_b: Bool, via_public: Bool
](ctx: DeviceContext, b: Int, m: Int) raises:
    """Publishes one operand of a batched matmul late, from a grid that has
    already released its dependents, and checks the matmul waited for it.

    With a ones-filled A and B every output element is exactly `K`; a matmul
    that read the late operand early would produce zeros instead.
    `test_pdl_negative_control` shows whether this setup makes an early reader
    see the stale data on the current run.
    """
    print(
        "test_late_producer",
        b,
        "x",
        m,
        "x",
        N,
        "x",
        K,
        "late_b",
        late_b,
        "public",
        via_public,
    )
    _late_producer_once[
        pdl_level, N=N, K=K, late_b=late_b, via_public=via_public
    ](ctx, b, m, 0.0, False)
    _late_producer_once[
        pdl_level, N=N, K=K, late_b=late_b, via_public=via_public
    ](ctx, b, m, _LATE_DELAY_S, True)


def test_late_producer_cases[pdl_level: PDLLevel](ctx: DeviceContext) raises:
    # A is the late operand of the first two, B of the last two. `public`
    # goes through the `batched_matmul` entry the graph op calls.
    test_late_producer[
        pdl_level, N=Int(512), K=Int(128), late_b=False, via_public=False
    ](ctx, 16, 61)
    test_late_producer[
        pdl_level, N=Int(512), K=Int(128), late_b=False, via_public=True
    ](ctx, 16, 61)
    test_late_producer[
        pdl_level, N=Int(128), K=Int(512), late_b=True, via_public=False
    ](ctx, 3, 129)
    test_late_producer[
        pdl_level, N=Int(128), K=Int(512), late_b=True, via_public=True
    ](ctx, 3, 129)


def test_pdl_cases[pdl_level: PDLLevel](ctx: DeviceContext) raises:
    test_static_NK[
        .bfloat16,
        transpose_b=True,
        pdl_level=pdl_level,
        N=Int(512),
        K=Int(128),
    ](ctx, 16, 61)

    test_static_NK[
        .bfloat16,
        transpose_b=True,
        lambda_fn=elementwise_epilogue_fn,
        pdl_level=pdl_level,
        N=Int(128),
        K=Int(512),
    ](ctx, 3, 129)

    # batch_size == 1 reroutes through the plain matmul dispatch.
    test_static_NK[
        .bfloat16,
        transpose_b=True,
        pdl_level=pdl_level,
        N=Int(512),
        K=Int(128),
    ](ctx, 1, 13)

    test_pdl_chain[pdl_level, N=Int(512), K=Int(128)](ctx, 16, 61)
    test_pdl_chain[pdl_level, N=Int(512), K=Int(128)](ctx, 16, 1)
    test_pdl_chain[pdl_level, N=Int(512), K=Int(128)](ctx, 3, 2049)
    # Several waves of CTAs, so a dependent grid's CTAs queue behind the
    # predecessor's; a hang here is caught by the test timeout.
    test_pdl_chain[pdl_level, N=Int(512), K=Int(128)](ctx, 16, 1024)
    comptime if ctx.target._has_blackwell_tcgen05():
        test_late_producer_cases[pdl_level](ctx)


def main() raises:
    with DeviceContext() as ctx:
        # Test zero-dimension edge cases
        test_dynamic_shapes[
            .bfloat16,
            transpose_b=False,
        ](ctx, 0, 2, 2, 2)

        # Test non-batch dispatch logic
        test_dynamic_shapes[
            .bfloat16,
            transpose_b=False,
        ](ctx, 1, 2, 2, 2)

        test_dynamic_shapes[
            .bfloat16,
            transpose_b=False,
        ](ctx, 2, 0, 2, 2)

        test_dynamic_shapes[
            .bfloat16,
            transpose_b=False,
        ](ctx, 2, 2, 0, 2)

        test_dynamic_shapes[
            .bfloat16,
            transpose_b=False,
        ](ctx, 2, 2, 2, 0)

        # tests naive kernels
        test_dynamic_shapes[
            .bfloat16,
            transpose_b=False,
        ](ctx, 2, 2, 2, 2)

        test_dynamic_shapes[
            .float32,
            transpose_b=False,
            lambda_fn=elementwise_epilogue_fn,
        ](ctx, 2, 2, 2, 2)

        test_dynamic_shapes[
            .float32,
            transpose_b=False,
            lambda_fn=elementwise_epilogue_fn,
        ](ctx, 64, 256, 512, 128)

        # Rank-4 output on the tiled dispatch with a fused epilogue
        # (GEX-4200). fp32/transpose_b=False takes the A100 multistage path.
        test_static_NK_rank4[
            .float32,
            transpose_b=False,
            lambda_fn=elementwise_epilogue_fn,
            N=Int(128),
            K=Int(128),
        ](ctx, 2, 3, 64)

        # transpose_b=True takes the SM100 path on Blackwell and the AMD
        # path on CDNA.
        test_static_NK_rank4[
            .bfloat16,
            transpose_b=True,
            lambda_fn=elementwise_epilogue_fn,
            N=Int(128),
            K=Int(128),
        ](ctx, 2, 3, 64)

        comptime if ctx.target.is_nvidia_gpu():
            # NOTE: these tests should be run on a100 and above

            # tests kernels.ampere_128x128_4
            test_static_NK[
                .bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                N=Int(128256),
                K=Int(4096),
            ](ctx, 2, 600)

            # tests kernels.ampere_256x64_4
            test_static_NK[
                .bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                N=Int(3072),
                K=Int(12288),
            ](ctx, 4, 14, rtol=2e-2)

            # tests DeepSeek Case
            test_static_NK[
                .bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                N=Int(128),
                K=Int(512),
            ](ctx, 128, 256)

            test_static_NK[
                .bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                N=Int(512),
                K=Int(128),
            ](ctx, 128, 256)

            test_static_NK[
                .bfloat16,
                transpose_b=False,
                lambda_fn=elementwise_epilogue_fn,
                N=Int(3072),
                K=Int(12288),
            ](ctx, 4, 14, rtol=2e-2)

            # test non-row major layout
            test_non_row_major_layout[
                .bfloat16,
                B=Int(128),
                N=Int(128),
                K=Int(512),
            ](ctx, 22)

            test_non_row_major_layout[
                .bfloat16,
                B=Int(128),
                N=Int(512),
                K=Int(128),
            ](ctx, 22)

        # Runs first so the log says whether the late-producer cases below
        # could have caught a missing wait.
        comptime if ctx.target._has_blackwell_tcgen05():
            _ = test_pdl_negative_control(ctx)

        # PDL on the SM100 batched path (MLA absorbed projections), ragged
        # and prime M. Other targets ignore the level.
        test_pdl_cases[PDLLevel.ON](ctx)
        test_pdl_cases[PDLLevel.OVERLAP_AT_BEGINNING](ctx)
