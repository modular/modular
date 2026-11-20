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
"""Checks `lamport_allreduce_post_norm_residual_rmsnorm` against a reference.

The reference sums the ranks' inputs in rank order in float32, norms the sum,
rounds it to bf16, weighs it and the residual each with its own bf16 rounding,
adds them in bf16 and norms the join, as the unfused ops would. Both outputs,
the join and its norm, are checked on every rank. The residual is not 1.0 and
the two norm weights differ, so swapping them, dropping a mix weight or adding
the residual before the norm fails. A row wider than one block is refused even
where the plain fused norm would spread it over a cluster.
"""

from std.math import rsqrt
from std.memory.alloc import unsafe_alloc
from std.random import random_float64
from std.sys import size_of

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, row_major

from comm import Signal, group_end, group_start
from comm.allreduce_lamport_rmsnorm import (
    lamport_allreduce_post_norm_residual_rmsnorm,
)
from comm.sync import enable_p2p, init_signal_buffer

from internal_utils import assert_almost_equal
from std.testing import assert_raises, assert_true

comptime dtype = DType.bfloat16
comptime EPS = Float32(1e-6)
# More calls than the three Lamport generations, so the rotation wraps.
comptime NUM_ITERS = 7


def _bf16(x: Float32) -> Float32:
    return x.cast[dtype]().cast[.float32]()


def post_norm_residual_test[
    ngpus: Int
](
    list_of_ctx: List[DeviceContext[]],
    M: Int,
    K: Int,
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
) raises:
    print(t"==== Lamport post-norm residual  ngpus= {ngpus}  M= {M}  K= {K}")
    var size = M * K
    var mix_normed = Float32(0.37)
    var mix_residual = Float32(1.61)

    var act = List[DeviceBuffer[dtype]](capacity=ngpus)
    var residual = List[DeviceBuffer[dtype]](capacity=ngpus)
    var gamma = List[DeviceBuffer[dtype]](capacity=ngpus)
    var gamma2 = List[DeviceBuffer[dtype]](capacity=ngpus)
    var mix = List[DeviceBuffer[.float32]](capacity=ngpus)
    var joined = List[DeviceBuffer[dtype]](capacity=ngpus)
    var normed = List[DeviceBuffer[dtype]](capacity=ngpus)

    var gamma_host = unsafe_alloc[Scalar[dtype]](K)
    var gamma2_host = unsafe_alloc[Scalar[dtype]](K)
    for i in range(K):
        gamma_host[i] = random_float64(min=0.5, max=1.5).cast[dtype]()
        gamma2_host[i] = random_float64(min=0.5, max=1.5).cast[dtype]()
    var residual_host = unsafe_alloc[Scalar[dtype]](size)
    for i in range(size):
        residual_host[i] = random_float64(min=-2.0, max=2.0).cast[dtype]()
    var mix_host = unsafe_alloc[Float32](2)
    mix_host[0] = mix_normed
    mix_host[1] = mix_residual
    var act_host = List[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )

    for g in range(ngpus):
        var a = unsafe_alloc[Scalar[dtype]](size)
        for i in range(size):
            a[i] = random_float64(min=-0.5, max=0.5).cast[dtype]()
        act_host.append(a)
        var ctx = list_of_ctx[g]
        act.append(ctx.enqueue_create_buffer[dtype](size))
        residual.append(ctx.enqueue_create_buffer[dtype](size))
        gamma.append(ctx.enqueue_create_buffer[dtype](K))
        gamma2.append(ctx.enqueue_create_buffer[dtype](K))
        mix.append(ctx.enqueue_create_buffer[.float32](2))
        joined.append(ctx.enqueue_create_buffer[dtype](size))
        normed.append(ctx.enqueue_create_buffer[dtype](size))
        ctx.enqueue_copy(act[g], act_host[g])
        ctx.enqueue_copy(residual[g], residual_host)
        ctx.enqueue_copy(gamma[g], gamma_host)
        ctx.enqueue_copy(gamma2[g], gamma2_host)
        ctx.enqueue_copy(mix[g], mix_host)

    for g in range(ngpus):
        list_of_ctx[g].synchronize()

    # Host reference.
    var want_joined = unsafe_alloc[Float32](size)
    var want_normed = unsafe_alloc[Float32](size)
    var sum_row = unsafe_alloc[Float32](K)
    for r in range(M):
        var sumsq = Float32(0)
        for k in range(K):
            var s = Float32(0)
            for g in range(ngpus):
                s += act_host[g][r * K + k].cast[.float32]()
            sum_row[k] = s
            sumsq += s * s
        var n1 = rsqrt(sumsq / Float32(K) + EPS)
        var sumsq2 = Float32(0)
        for k in range(K):
            var normed1 = _bf16(
                sum_row[k] * n1 * gamma_host[k].cast[.float32]()
            )
            var h = _bf16(
                _bf16(normed1 * mix_normed)
                + _bf16(
                    residual_host[r * K + k].cast[.float32]() * mix_residual
                )
            )
            want_joined[r * K + k] = h
            sumsq2 += h * h
        var n2 = rsqrt(sumsq2 / Float32(K) + EPS)
        for k in range(K):
            want_normed[r * K + k] = _bf16(
                want_joined[r * K + k] * n2 * gamma2_host[k].cast[.float32]()
            )

    var host = unsafe_alloc[Scalar[dtype]](size)
    var got = unsafe_alloc[Float32](size)
    for it in range(NUM_ITERS):
        group_start()
        comptime for g in range(ngpus):
            lamport_allreduce_post_norm_residual_rmsnorm(
                g,
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    act[g].unsafe_ptr()
                ),
                TileTensor(residual[g], row_major(M, K)),
                TileTensor(joined[g], row_major(M, K)),
                rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                    normed[g].unsafe_ptr()
                ),
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    gamma[g].unsafe_ptr()
                ),
                TileTensor(gamma2[g], row_major(K)),
                TileTensor(mix[g], row_major[2]()),
                rank_sigs,
                M,
                K,
                EPS.cast[dtype](),
                list_of_ctx[g],
            )
        group_end()
        for g in range(ngpus):
            list_of_ctx[g].synchronize()
        for g in range(ngpus):
            list_of_ctx[g].enqueue_copy(host, joined[g])
            list_of_ctx[g].synchronize()
            for i in range(size):
                got[i] = host[i].cast[.float32]()
            assert_almost_equal(
                got, want_joined, num_elements=size, atol=2e-2, rtol=1e-2
            )
            list_of_ctx[g].enqueue_copy(host, normed[g])
            list_of_ctx[g].synchronize()
            for i in range(size):
                got[i] = host[i].cast[.float32]()
            assert_almost_equal(
                got, want_normed, num_elements=size, atol=2e-2, rtol=1e-2
            )
        print("  [ok] iter", it)

    for g in range(ngpus):
        act_host[g].unsafe_free()
    gamma_host.unsafe_free()
    gamma2_host.unsafe_free()
    residual_host.unsafe_free()
    mix_host.unsafe_free()
    want_joined.unsafe_free()
    want_normed.unsafe_free()
    sum_row.unsafe_free()
    host.unsafe_free()
    got.unsafe_free()
    print("PASSED\n")


def wide_row_test(
    ctx: DeviceContext, sig: MutPointer[Signal, MutAnyOrigin]
) raises:
    """Checks that a row past one block's packs is refused, not cut short.

    On 8 GPUs of H100 or newer the plain fused norm spreads a 1536-pack row
    over a cluster, but the residual join's second norm runs in one block. The
    shape check raises before anything is enqueued, so one device and one
    signal buffer stand in for all 8 ranks.
    """
    comptime K = 12288
    var act = ctx.enqueue_create_buffer[dtype](K)
    var residual = ctx.enqueue_create_buffer[dtype](K)
    var joined = ctx.enqueue_create_buffer[dtype](K)
    var normed = ctx.enqueue_create_buffer[dtype](K)
    var gamma = ctx.enqueue_create_buffer[dtype](K)
    var mix = ctx.enqueue_create_buffer[.float32](2)
    with assert_raises(contains="cols too large"):
        lamport_allreduce_post_norm_residual_rmsnorm(
            0,
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](act.unsafe_ptr()),
            TileTensor(residual, row_major(1, K)),
            TileTensor(joined, row_major(1, K)),
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                normed.unsafe_ptr()
            ),
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                gamma.unsafe_ptr()
            ),
            TileTensor(gamma, row_major(K)),
            TileTensor(mix, row_major[2]()),
            Array[MutPointer[Signal, MutAnyOrigin], 8](fill=sig),
            1,
            K,
            EPS.cast[dtype](),
            ctx,
        )
    print("  [ok] a 1536-pack row is refused")


def run_all[ngpus: Int]() raises:
    var ctx = List[DeviceContext[]]()
    for i in range(ngpus):
        ctx.append(DeviceContext(device_id=i))
    # One signal buffer set for every shape, as a model reuses it across
    # steps: a call after a larger one must clear that call's leftovers.
    var sigs = List[DeviceBuffer[.uint8]](capacity=ngpus)
    for g in range(ngpus):
        sigs.append(ctx[g].create_buffer_sync[.uint8](size_of[Signal]()))
        init_signal_buffer(sigs[g], ctx[g])
    var rank_sigs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> MutPointer[
            Signal, MutAnyOrigin
        ]: Signal.unsafe_ptr_from(sigs[i])
    )
    for g in range(ngpus):
        ctx[g].synchronize()

    # One decode row, a decode batch, the most rows a 5120-wide bf16 message
    # takes on Lamport (102) and one past it, so a call runs as two and then
    # three row chunks, shrinking and growing between calls; then a
    # narrower width.
    for m in [1, 16, 150, 7, 256, 96, 103, 1, 200, 102, 128]:
        post_norm_residual_test[ngpus](ctx, m, 5120, rank_sigs)
    post_norm_residual_test[ngpus](ctx, 7, 4096, rank_sigs)


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    run_all[2]()
    var ctx = DeviceContext(device_id=0)
    var sig = ctx.create_buffer_sync[.uint8](size_of[Signal]())
    wide_row_test(ctx, Signal.unsafe_ptr_from(sig))
    if DeviceContext.number_of_devices() >= 4:
        run_all[4]()
    print("All Lamport post-norm residual checks passed.")
