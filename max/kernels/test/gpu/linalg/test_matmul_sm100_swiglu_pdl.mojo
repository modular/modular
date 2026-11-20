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
"""SM100 fused GEMM+SwiGLU under programmatic dependent launch (PDL).

Each case runs the kernel in a launch chain, as serving does, and holds its
output bit for bit to the same chain with the SwiGLU kernel's PDL off:

  producer -> SwiGLU GEMM -> consumer

The producer releases its dependents on entry, spins, then writes the
activations over a poisoned buffer, so a SwiGLU CTA that read them before its
`wait_on_dependent_grids` would load poison. The consumer waits and copies the
output, so a release that fired before the epilogue stores landed would show
as poison or stale values. Cases cover the PDL levels, 1-SM and 2-SM tiles,
swapped and unswapped operands, both epilogues, a bias, K shorter and longer
than the weight prefetch depth, and ragged M.

The weight and bias are constant model weights by contract, so the producer
never writes them.

Three more chains put SwiGLU CTAs, which allocate TMEM and shared memory
before their wait, behind predecessors that hold TMEM: a SwiGLU GEMM chained to
itself, a bf16 GEMM feeding a SwiGLU GEMM, and a synthetic predecessor that
releases its dependents on entry while its CTAs hold or queue for all of TMEM.
A dependent launches only once every predecessor CTA has started, so none can
be left pending behind it. Each chain checks that SwiGLU launched behind such a
predecessor completes without a hang and produces the right output; a hang
shows as a test timeout. The last chain also requires, from device timestamps,
that a lightweight PDL dependent became resident before the predecessor
finished in at least one timed pass. That is the stamp, not a SwiGLU CTA: the
kernel has no device-side hook, so the test cannot observe the SwiGLU CTAs
reaching `tcgen05.alloc`.
"""

from std.collections import OptionalReg
from std.math import ceildiv
from std.testing import assert_true
from std.time import global_perf_counter_ns

from layout import Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from linalg.matmul.gpu.sm100_structured.fused_swiglu.config import (
    FusedSwiGLUMatmulConfig,
)
from linalg.matmul.gpu.sm100_structured.fused_swiglu.dispatch import (
    matmul_swiglu_dispatch_sm100,
)
from max.gpu import block_dim, block_idx, grid_dim, thread_idx, warp_id
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_release_allocation_lock,
)
from max.gpu.memory import external_memory
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)

comptime dtype = DType.bfloat16
comptime ITERS = 8
comptime POISON = Scalar[dtype](-7)


def slow_producer(
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    length: Int64,
    spins: Int32,
):
    """Releases its dependents on entry, spins, then writes `dst`."""
    launch_dependent_grids()
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var acc = Float32(tid)
    for _ in range(Int(spins)):
        acc = acc * 1.0000001 + 0.5
    var i = tid
    while i < Int(length):
        var value = src[i]
        if acc < Float32(-1):
            value = Scalar[dtype](acc)
        dst[i] = value
        i += stride


def tmem_hog(
    ends: MutPointer[UInt64, MutAnyOrigin],
    hold_ns: UInt64,
):
    """Releases its dependents on entry, then holds all of TMEM for `hold_ns`.

    Several CTAs fit on one SM, so all but one queue in `tcgen05.alloc` behind
    it. Any dependent CTA that allocates TMEM before its wait joins that queue.
    """
    launch_dependent_grids()
    if warp_id() == 0:
        var tmem_addr = external_memory[
            UInt32, address_space=.SHARED, alignment=16
        ]().unsafe_bitcast[UInt32]()
        tcgen05_alloc[1](tmem_addr, 512)
        tcgen05_release_allocation_lock[1]()
        var addr = tmem_addr[0]
        var start = global_perf_counter_ns()
        while global_perf_counter_ns() - start < hold_ns:
            pass
        tcgen05_dealloc[1](addr, 512)
        if thread_idx.x == 0:
            ends[block_idx.x] = global_perf_counter_ns()


def launch_stamp(
    stamp: MutPointer[UInt64, MutAnyOrigin],
):
    """Releases its dependents on entry and records when it became resident."""
    launch_dependent_grids()
    if thread_idx.x == 0:
        stamp[0] = global_perf_counter_ns()
    wait_on_dependent_grids()


def consumer(
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    length: Int64,
):
    """Waits for its predecessor, then copies `src` to `dst`."""
    wait_on_dependent_grids()
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    while i < Int(length):
        dst[i] = src[i]
        i += stride


def _mut(
    buf: DeviceBuffer[dtype],
) -> MutPointer[Scalar[dtype], MutUntrackedOrigin]:
    # TileTensor needs a mutable view of an output buffer the caller owns.
    return (
        buf.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


def _fill_pattern(
    mut host: HostBuffer[dtype], seed: Int, modulus: Int, scale: Float32
):
    for i in range(len(host)):
        host[i] = Scalar[dtype](
            Float32((i * seed + 3) % modulus - modulus // 2) * scale
        )


def _launch_producer(
    ctx: DeviceContext,
    src: DeviceBuffer[dtype],
    dst: DeviceBuffer[dtype],
    length: Int,
) raises:
    ctx.enqueue_function[slow_producer](
        src,
        dst,
        Int64(length),
        Int32(4000),
        grid_dim=ceildiv(length, 256),
        block_dim=256,
    )


def _launch_consumer(
    ctx: DeviceContext,
    src: DeviceBuffer[dtype],
    dst: DeviceBuffer[dtype],
    length: Int,
    pdl_level: PDLLevel,
) raises:
    ctx.enqueue_function[consumer](
        src,
        dst,
        Int64(length),
        grid_dim=min(ceildiv(length, 256), 1024),
        block_dim=256,
        attributes=pdl_launch_attributes(pdl_level),
    )


def _check_equal(
    ctx: DeviceContext,
    got: DeviceBuffer[dtype],
    want: List[Scalar[dtype]],
    what: String,
) raises:
    var host = ctx.enqueue_create_host_buffer[dtype](len(want))
    ctx.enqueue_copy(host, got)
    ctx.synchronize()
    for i in range(len(want)):
        if host[i] != want[i]:
            raise Error(t"{what}: element {i} is {host[i]}, expected {want[i]}")


def _to_list(
    ctx: DeviceContext, buf: DeviceBuffer[dtype], length: Int
) raises -> List[Scalar[dtype]]:
    var host = ctx.enqueue_create_host_buffer[dtype](length)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    var result = List[Scalar[dtype]](capacity=length)
    for i in range(length):
        result.append(host[i])
    return result^


def _launch_swiglu[
    config: FusedSwiGLUMatmulConfig[dtype, dtype, dtype, True],
    pdl_level: PDLLevel,
    N: Int,
    K: Int,
](
    ctx: DeviceContext,
    c: DeviceBuffer[dtype],
    a: DeviceBuffer[dtype],
    w: DeviceBuffer[dtype],
    bias: DeviceBuffer[dtype],
    m: Int,
) raises:
    var c_t = TileTensor(_mut(c), row_major(m, Idx[N // 2]))
    var a_t = TileTensor(a, row_major(m, Idx[K])).as_imm()
    var w_t = TileTensor(w, row_major(Idx[N], Idx[K])).as_imm()
    var bias_ptr = ImmPointer[BFloat16, ImmutAnyOrigin](
        unsafe_from_address=Int(bias.unsafe_ptr())
    )
    matmul_swiglu_dispatch_sm100[config, pdl_level](
        c_t, a_t, w_t, ctx, OptionalReg(bias_ptr)
    )


def test_chain[
    config: FusedSwiGLUMatmulConfig[dtype, dtype, dtype, True],
    pdl_level: PDLLevel,
    N: Int,
    K: Int,
](ctx: DeviceContext, m: Int) raises:
    comptime H = N // 2
    print(
        t"  pdl_level={pdl_level._level} m={m} n={N} k={K}"
        t" mma={config.mma_shape[0]}x{config.mma_shape[1]}"
        t" cta_group={config.cta_group} swap_ab={config.AB_swapped}"
        t" k_group={config.k_group_size}"
        t" bias={config.use_bias} reg={config.register_swiglu}"
    )

    var a_host = ctx.enqueue_create_host_buffer[dtype](m * K)
    var w_host = ctx.enqueue_create_host_buffer[dtype](N * K)
    var bias_host = ctx.enqueue_create_host_buffer[dtype](N)
    ctx.synchronize()
    _fill_pattern(a_host, 7, 17, 0.125)
    _fill_pattern(w_host, 11, 15, 0.0625)
    _fill_pattern(bias_host, 5, 9, 0.125)

    var a_src = ctx.enqueue_create_buffer[dtype](m * K)
    var a = ctx.enqueue_create_buffer[dtype](m * K)
    var w = ctx.enqueue_create_buffer[dtype](N * K)
    var bias = ctx.enqueue_create_buffer[dtype](N)
    var c = ctx.enqueue_create_buffer[dtype](m * H)
    var out = ctx.enqueue_create_buffer[dtype](m * H)
    ctx.enqueue_copy(a_src, a_host)
    ctx.enqueue_copy(w, w_host)
    ctx.enqueue_copy(bias, bias_host)

    # The reference: the same kernel with PDL off, on the finished input.
    ctx.enqueue_copy(a, a_src)
    c.enqueue_fill(POISON)
    _launch_swiglu[config, PDLLevel.OFF, N, K](ctx, c, a, w, bias, m)
    var want = _to_list(ctx, c, m * H)

    for _ in range(ITERS):
        a.enqueue_fill(POISON)
        c.enqueue_fill(POISON)
        out.enqueue_fill(POISON)
        _launch_producer(ctx, a_src, a, m * K)
        _launch_swiglu[config, pdl_level, N, K](ctx, c, a, w, bias, m)
        _launch_consumer(ctx, c, out, m * H, pdl_level)
        _check_equal(ctx, out, want, "chain output")


def test_self_chain[
    config: FusedSwiGLUMatmulConfig[dtype, dtype, dtype, True],
    pdl_level: PDLLevel,
    N1: Int,
    K1: Int,
    N2: Int,
](ctx: DeviceContext, m: Int) raises:
    """SwiGLU -> SwiGLU: the second GEMM's A is the first one's output."""
    comptime H1 = N1 // 2
    comptime H2 = N2 // 2
    print(t"  self chain pdl_level={pdl_level._level} m={m}")
    var a_host = ctx.enqueue_create_host_buffer[dtype](m * K1)
    var w1_host = ctx.enqueue_create_host_buffer[dtype](N1 * K1)
    var w2_host = ctx.enqueue_create_host_buffer[dtype](N2 * H1)
    var bias_host = ctx.enqueue_create_host_buffer[dtype](max(N1, N2))
    ctx.synchronize()
    _fill_pattern(a_host, 7, 17, 0.125)
    _fill_pattern(w1_host, 11, 15, 0.0625)
    _fill_pattern(w2_host, 13, 15, 0.0625)
    _fill_pattern(bias_host, 5, 9, 0.125)

    var a_src = ctx.enqueue_create_buffer[dtype](m * K1)
    var a = ctx.enqueue_create_buffer[dtype](m * K1)
    var w1 = ctx.enqueue_create_buffer[dtype](N1 * K1)
    var w2 = ctx.enqueue_create_buffer[dtype](N2 * H1)
    var bias = ctx.enqueue_create_buffer[dtype](max(N1, N2))
    var c1 = ctx.enqueue_create_buffer[dtype](m * H1)
    var c2 = ctx.enqueue_create_buffer[dtype](m * H2)
    var out = ctx.enqueue_create_buffer[dtype](m * H2)
    ctx.enqueue_copy(a_src, a_host)
    ctx.enqueue_copy(w1, w1_host)
    ctx.enqueue_copy(w2, w2_host)
    ctx.enqueue_copy(bias, bias_host)

    ctx.enqueue_copy(a, a_src)
    _launch_swiglu[config, PDLLevel.OFF, N1, K1](ctx, c1, a, w1, bias, m)
    _launch_swiglu[config, PDLLevel.OFF, N2, H1](ctx, c2, c1, w2, bias, m)
    var want = _to_list(ctx, c2, m * H2)

    for _ in range(ITERS):
        a.enqueue_fill(POISON)
        c1.enqueue_fill(POISON)
        c2.enqueue_fill(POISON)
        out.enqueue_fill(POISON)
        _launch_producer(ctx, a_src, a, m * K1)
        _launch_swiglu[config, pdl_level, N1, K1](ctx, c1, a, w1, bias, m)
        _launch_swiglu[config, pdl_level, N2, H1](ctx, c2, c1, w2, bias, m)
        _launch_consumer(ctx, c2, out, m * H2, pdl_level)
        _check_equal(ctx, out, want, "self chain output")


def _tmem_chain[
    config: FusedSwiGLUMatmulConfig[dtype, dtype, dtype, True],
    pdl_level: PDLLevel,
    N0: Int,
    K0: Int,
    N: Int,
](
    ctx: DeviceContext,
    a: DeviceBuffer[dtype],
    w0: DeviceBuffer[dtype],
    w: DeviceBuffer[dtype],
    bias: DeviceBuffer[dtype],
    mid: DeviceBuffer[dtype],
    c: DeviceBuffer[dtype],
    m: Int,
) raises:
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        TileTensor(_mut(mid), row_major(m, Idx[N0])),
        TileTensor(a, row_major(m, Idx[K0])).as_imm(),
        TileTensor(w0, row_major(Idx[N0], Idx[K0])).as_imm(),
        ctx,
    )
    _launch_swiglu[config, pdl_level, N, N0](ctx, c, mid, w, bias, m)


def test_tmem_predecessor[
    config: FusedSwiGLUMatmulConfig[dtype, dtype, dtype, True],
    pdl_level: PDLLevel,
    N0: Int,
    K0: Int,
    N: Int,
](ctx: DeviceContext, m: Int) raises:
    """Runs a bf16 GEMM into SwiGLU: both hold TMEM, the first in two waves.

    On SM100 `_matmul_gpu` launches its kernel at `OVERLAP_AT_BEGINNING`
    whatever `pdl_level` it is given (`linalg/matmul/gpu/__init__.mojo`), so
    the GEMM releases the SwiGLU grid on entry.
    """
    comptime H = N // 2
    print(t"  TMEM predecessor pdl_level={pdl_level._level} m={m}")
    var a_host = ctx.enqueue_create_host_buffer[dtype](m * K0)
    var w0_host = ctx.enqueue_create_host_buffer[dtype](N0 * K0)
    var w_host = ctx.enqueue_create_host_buffer[dtype](N * N0)
    var bias_host = ctx.enqueue_create_host_buffer[dtype](N)
    ctx.synchronize()
    _fill_pattern(a_host, 7, 17, 0.125)
    _fill_pattern(w0_host, 11, 15, 0.0625)
    _fill_pattern(w_host, 13, 15, 0.0625)
    _fill_pattern(bias_host, 5, 9, 0.125)

    var a = ctx.enqueue_create_buffer[dtype](m * K0)
    var w0 = ctx.enqueue_create_buffer[dtype](N0 * K0)
    var w = ctx.enqueue_create_buffer[dtype](N * N0)
    var bias = ctx.enqueue_create_buffer[dtype](N)
    var mid = ctx.enqueue_create_buffer[dtype](m * N0)
    var c = ctx.enqueue_create_buffer[dtype](m * H)
    var out = ctx.enqueue_create_buffer[dtype](m * H)
    ctx.enqueue_copy(a, a_host)
    ctx.enqueue_copy(w0, w0_host)
    ctx.enqueue_copy(w, w_host)
    ctx.enqueue_copy(bias, bias_host)

    c.enqueue_fill(POISON)
    _tmem_chain[config, PDLLevel.OFF, N0, K0, N](ctx, a, w0, w, bias, mid, c, m)
    var want = _to_list(ctx, c, m * H)

    for _ in range(ITERS):
        c.enqueue_fill(POISON)
        out.enqueue_fill(POISON)
        _tmem_chain[config, pdl_level, N0, K0, N](
            ctx, a, w0, w, bias, mid, c, m
        )
        _launch_consumer(ctx, c, out, m * H, pdl_level)
        _check_equal(ctx, out, want, "TMEM predecessor chain output")


def test_early_release_predecessor[
    config: FusedSwiGLUMatmulConfig[dtype, dtype, dtype, True],
    N: Int,
    K: Int,
](ctx: DeviceContext, m: Int) raises:
    """Runs hog -> stamp -> SwiGLU -> consumer behind a TMEM-holding hog.

    The hog launches four CTAs per SM, each holding all of TMEM for 200 us, and
    releases its dependents on entry. The stamp is a PDL dependent that records
    when it became resident, and the SwiGLU grid launches behind it. The test
    requires the stamp to predate the hog's last exit in at least one timed
    pass, and the output to match PDL off. The stamp is not a SwiGLU CTA, so
    this shows a PDL dependent becoming resident behind the hog, not that the
    SwiGLU CTAs reached `tcgen05.alloc`.
    """
    comptime H = N // 2
    comptime hold_ns = UInt64(200_000)
    var hog_ctas = 4 * ctx.default_device_info.sm_count
    print(t"  early-release TMEM predecessor m={m}")

    var a_host = ctx.enqueue_create_host_buffer[dtype](m * K)
    var w_host = ctx.enqueue_create_host_buffer[dtype](N * K)
    var bias_host = ctx.enqueue_create_host_buffer[dtype](N)
    ctx.synchronize()
    _fill_pattern(a_host, 7, 17, 0.125)
    _fill_pattern(w_host, 11, 15, 0.0625)
    _fill_pattern(bias_host, 5, 9, 0.125)
    var a = ctx.enqueue_create_buffer[dtype](m * K)
    var w = ctx.enqueue_create_buffer[dtype](N * K)
    var bias = ctx.enqueue_create_buffer[dtype](N)
    var c = ctx.enqueue_create_buffer[dtype](m * H)
    var out = ctx.enqueue_create_buffer[dtype](m * H)
    var ends = ctx.enqueue_create_buffer[DType.uint64](hog_ctas)
    var stamp = ctx.enqueue_create_buffer[DType.uint64](1)
    ctx.enqueue_copy(a, a_host)
    ctx.enqueue_copy(w, w_host)
    ctx.enqueue_copy(bias, bias_host)

    c.enqueue_fill(POISON)
    _launch_swiglu[config, PDLLevel.OFF, N, K](ctx, c, a, w, bias, m)
    var want = _to_list(ctx, c, m * H)

    var overlapped = 0
    for iteration in range(ITERS):
        c.enqueue_fill(POISON)
        out.enqueue_fill(POISON)
        ends.enqueue_fill(0)
        stamp.enqueue_fill(0)
        ctx.enqueue_function[tmem_hog](
            ends,
            hold_ns,
            grid_dim=hog_ctas,
            block_dim=128,
            shared_mem_bytes=16,
        )
        ctx.enqueue_function[launch_stamp](
            stamp,
            grid_dim=1,
            block_dim=32,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        _launch_swiglu[config, PDLLevel.ON, N, K](ctx, c, a, w, bias, m)
        _launch_consumer(ctx, c, out, m * H, PDLLevel.ON)
        _check_equal(ctx, out, want, "early-release predecessor output")

        var ends_host = ctx.enqueue_create_host_buffer[DType.uint64](hog_ctas)
        var stamp_host = ctx.enqueue_create_host_buffer[DType.uint64](1)
        ctx.enqueue_copy(ends_host, ends)
        ctx.enqueue_copy(stamp_host, stamp)
        ctx.synchronize()
        var last_exit = UInt64(0)
        for i in range(hog_ctas):
            last_exit = max(last_exit, ends_host[i])
        assert_true(
            stamp_host[0] != 0 and last_exit != 0,
            "the hog or the stamp did not run",
        )
        # The first pass loads each kernel on the host, which can stall the
        # stamp's launch past the hog's exit; it only warms them up.
        if iteration > 0 and stamp_host[0] < last_exit:
            overlapped += 1

    print(
        t"    dependent resident before the predecessor's last exit in"
        t" {overlapped} of {ITERS - 1} timed passes"
    )
    assert_true(
        overlapped > 0,
        "the stamp never became resident before the hog's last exit",
    )


def test_levels[
    config: FusedSwiGLUMatmulConfig[dtype, dtype, dtype, True],
    N: Int,
    K: Int,
](ctx: DeviceContext, m: Int) raises:
    test_chain[config, PDLLevel.ON, N, K](ctx, m)
    test_chain[config, PDLLevel.OVERLAP_AT_BEGINNING, N, K](ctx, m)


def main() raises:
    with DeviceContext() as ctx:
        # 2-SM, unswapped, a CLC-scheduled grid with several waves: the
        # weight prefetch runs on the first tile of each CTA only.
        comptime two_sm = FusedSwiGLUMatmulConfig[dtype, dtype, dtype, True](
            mma_shape=(256, 128, 16),
            cta_group=2,
            cluster_shape=(2, 1, 1),
            register_swiglu=True,
        )
        comptime two_sm_smem = FusedSwiGLUMatmulConfig[
            dtype, dtype, dtype, True
        ](
            mma_shape=(256, 128, 16),
            cta_group=2,
            cluster_shape=(2, 1, 1),
        )
        comptime two_sm_bias = FusedSwiGLUMatmulConfig[
            dtype, dtype, dtype, True
        ](
            mma_shape=(256, 128, 16),
            cta_group=2,
            cluster_shape=(2, 1, 1),
            register_swiglu=True,
            use_bias=True,
        )
        # 2-SM swapped decode tile: no CLC stages, k_group 2.
        comptime two_sm_swap = FusedSwiGLUMatmulConfig[
            dtype, dtype, dtype, True
        ](
            mma_shape=(256, 32, 16),
            cta_group=2,
            cluster_shape=(2, 1, 1),
            AB_swapped=True,
            k_group_size=2,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            register_swiglu=True,
        )
        comptime one_sm_swap = FusedSwiGLUMatmulConfig[
            dtype, dtype, dtype, True
        ](
            mma_shape=(128, 16, 16),
            cta_group=1,
            cluster_shape=(1, 1, 1),
            AB_swapped=True,
            k_group_size=4,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            register_swiglu=True,
        )
        comptime one_sm = FusedSwiGLUMatmulConfig[dtype, dtype, dtype, True](
            mma_shape=(128, 64, 16),
            cta_group=1,
            cluster_shape=(1, 1, 1),
            register_swiglu=True,
        )

        print("producer -> SwiGLU -> consumer")
        # K=1536 is 24 K-tiles, past any prefetch depth; K=64 is one.
        test_levels[two_sm, 2048, 1536](ctx, 4099)
        test_levels[two_sm, 1024, 64](ctx, 333)
        test_levels[two_sm_smem, 1024, 640](ctx, 333)
        test_levels[two_sm_bias, 1024, 640](ctx, 129)
        test_levels[two_sm_swap, 4096, 1536](ctx, 7)
        test_levels[two_sm_swap, 1024, 192](ctx, 1)
        test_levels[one_sm_swap, 2048, 4096](ctx, 5)
        test_levels[one_sm, 1024, 1536](ctx, 257)

        print("SwiGLU -> SwiGLU, more CTAs than SMs")
        test_self_chain[two_sm, PDLLevel.ON, 8192, 1024, 8192](ctx, 4096)
        test_self_chain[
            two_sm, PDLLevel.OVERLAP_AT_BEGINNING, 8192, 1024, 8192
        ](ctx, 4096)
        test_self_chain[
            two_sm_swap, PDLLevel.OVERLAP_AT_BEGINNING, 32768, 1024, 8192
        ](ctx, 16)

        print("bf16 GEMM -> SwiGLU, more CTAs than SMs")
        test_tmem_predecessor[two_sm, PDLLevel.ON, 8192, 1024, 8192](ctx, 4096)
        test_tmem_predecessor[
            two_sm, PDLLevel.OVERLAP_AT_BEGINNING, 8192, 1024, 8192
        ](ctx, 4096)

        print("early-releasing TMEM predecessor")
        test_early_release_predecessor[two_sm, 2048, 1536](ctx, 1024)
        test_early_release_predecessor[two_sm_bias, 1024, 640](ctx, 333)
        test_early_release_predecessor[two_sm_swap, 4096, 1536](ctx, 7)
        print("PASS")
