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
"""PDL race test for the MHA decode and naive-attention fallback kernels.

`mha_decoding`, `_bmm0_bs` and `_bmm1_bs` are launched with the programmatic
stream serialization attribute, so they may become resident while their
predecessor grid is still running. Each must therefore call
`wait_on_dependent_grids()` before it reads anything the predecessor wrote
(lengths, Q, K, V, the softmax output).

Every case runs a real PDL chain with no host sync between the launches:

1. `_late_publish` releases its dependents on entry, spins for `delay_us`
   microseconds, and only then writes the real operands over a stale copy.
2. The kernel under test launches next with the PDL attribute. A kernel
   without a wait reads the stale operands and produces a different result.
3. The result is compared bit for bit with a run over the real operands that
   had no predecessor in flight.

`_bmm0_bs` and `_bmm1_bs` are launched directly because their production
predecessors (the Q projection, the softmax kernel) are not something a test
can delay. `mha_decoding` and the full naive pipeline go through
`flash_attention` (dense operands) and `flash_attention_dispatch` (ragged
operands), as production does; the ragged cases publish the row offsets and the
cache lengths the kernels read right after the wait. The result must not
depend on `delay_us`.

PDL only permits the dependent to start early. Before the cases at each delay,
an unwaited dependent is run behind the same producer; it must observe the
stale operand, which shows the GPU does overlap them and the cases can fail.
"""

from std.collections import OptionalReg
from std.math import ceildiv
from std.random import rand, seed
from std.sys import get_defined_int
from std.sys.intrinsics import llvm_intrinsic
from std.time import global_perf_counter_ns

from layout import Idx, TileTensor, row_major
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
)
from nn.attention.gpu.mha import (
    _NAIVE_BMM_BLOCK_DIM,
    _bmm0_bs,
    _bmm1_bs,
    flash_attention,
    flash_attention_dispatch,
)
from nn.attention.gpu.nvidia.common import (
    ImmutTileTensor1D,
    immut_tile_tensor_1d,
)
from nn.attention.mha_mask import NullMask
from nn.attention.mha_operand import RaggedMHAOperand, TileTensorMHAOperand

comptime _DT = DType.float32
comptime _SENTINEL = Float32(7.0)

# Overlap is expected on nearly every attempt; the canary fails only if none of
# these overlap.
comptime _CANARY_ATTEMPTS = 20


# `nanosleep` can sleep for less than requested (PTX allows 0 to 2x), so the
# delay below is a `globaltimer` deadline and `nanosleep` only paces the poll.
def _late_publish[
    t0: DType, t1: DType, t2: DType
](
    dst0: MutPointer[Scalar[t0], MutAnyOrigin],
    src0: MutPointer[Scalar[t0], MutAnyOrigin],
    n0: Int32,
    dst1: MutPointer[Scalar[t1], MutAnyOrigin],
    src1: MutPointer[Scalar[t1], MutAnyOrigin],
    n1: Int32,
    dst2: MutPointer[Scalar[t2], MutAnyOrigin],
    src2: MutPointer[Scalar[t2], MutAnyOrigin],
    n2: Int32,
    delay_us: Int32,
):
    """Releases the dependent grid on entry, spins, then publishes operands.

    A dependent that reads any destination before its own PDL wait sees the
    stale contents.
    """
    launch_dependent_grids()
    var deadline = global_perf_counter_ns() + UInt64(delay_us) * 1000
    while global_perf_counter_ns() < deadline:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(50_000))
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(n0), stride):
        dst0[i] = src0[i]
    for i in range(tid, Int(n1), stride):
        dst1[i] = src1[i]
    for i in range(tid, Int(n2), stride):
        dst2[i] = src2[i]


struct _Operand[dtype: DType](Movable):
    """One predecessor-produced input: the real values and a stale copy."""

    var real_host: HostBuffer[Self.dtype]
    var stale_host: HostBuffer[Self.dtype]
    var real: DeviceBuffer[Self.dtype]
    var live: DeviceBuffer[Self.dtype]
    var size: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        size: Int,
        stale: Scalar[Self.dtype],
    ) raises:
        self.size = size
        self.real_host = ctx.enqueue_create_host_buffer[Self.dtype](size)
        self.stale_host = ctx.enqueue_create_host_buffer[Self.dtype](size)
        self.real = ctx.enqueue_create_buffer[Self.dtype](size)
        self.live = ctx.enqueue_create_buffer[Self.dtype](size)
        for i in range(size):
            self.stale_host[i] = stale
        ctx.synchronize()

    def fill_random(mut self, ctx: DeviceContext) raises:
        rand[Self.dtype](self.real_host.as_span())
        ctx.enqueue_copy(self.real, self.real_host)

    def fill(self, ctx: DeviceContext, values: List[Scalar[Self.dtype]]) raises:
        for i in range(self.size):
            self.real_host[i] = values[i]
        ctx.enqueue_copy(self.real, self.real_host)

    def reset(self, ctx: DeviceContext, late: Bool) raises:
        """Makes `live` stale if the producer will publish it, else real."""
        if late:
            ctx.enqueue_copy(self.live, self.stale_host)
        else:
            ctx.enqueue_copy(self.live, self.real_host)

    def count(self, late: Bool) -> Int32:
        return Int32(self.size) if late else Int32(0)


def _publish[
    t0: DType, t1: DType, t2: DType
](
    ctx: DeviceContext,
    op0: _Operand[t0],
    late0: Bool,
    op1: _Operand[t1],
    late1: Bool,
    op2: _Operand[t2],
    late2: Bool,
    delay_us: Int,
) raises:
    comptime producer = _late_publish[t0, t1, t2]
    ctx.enqueue_function[producer](
        op0.live,
        op0.real,
        op0.count(late0),
        op1.live,
        op1.real,
        op1.count(late1),
        op2.live,
        op2.real,
        op2.count(late2),
        Int32(delay_us),
        grid_dim=4,
        block_dim=64,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


def _unwaited_read(
    dst: MutPointer[Scalar[_DT], MutAnyOrigin],
    src: MutPointer[Scalar[_DT], MutAnyOrigin],
    n: Int32,
):
    """Copies `src` to `dst` with no PDL wait, as a kernel missing one would."""
    var i = Int(thread_idx.x + block_idx.x * block_dim.x)
    if i < Int(n):
        dst[i] = src[i]


def _dependent_overlaps(ctx: DeviceContext, delay_us: Int) raises:
    """Raises unless a dependent without a wait runs inside the producer.

    PDL lets a released dependent start early but does not promise it does. The
    race cases above only detect a missing wait when it does, so run a
    dependent that has no wait behind the same producer: it must read the stale
    operand, or the producer's delay hides nothing and those cases prove
    nothing on this GPU.
    """
    var n = 64
    var src = _Operand[_DT](ctx, n, _SENTINEL)
    var other = _Operand[_DT](ctx, n, 0)
    src.fill_random(ctx)
    var out = ctx.enqueue_create_buffer[_DT](n)
    var out_host = ctx.enqueue_create_host_buffer[_DT](n)
    ctx.synchronize()

    # The first launch of `_unwaited_read` loads its module on the host while
    # the producer is already running, so it always starts after the producer
    # has finished. Launch it once, uncounted, so the counted attempts measure
    # overlap and not module load time.
    var overlapped = 0
    for attempt in range(-1, _CANARY_ATTEMPTS):
        src.reset(ctx, True)
        _publish(ctx, src, True, other, False, other, False, delay_us)
        ctx.enqueue_function[_unwaited_read](
            out,
            src.live,
            Int32(n),
            grid_dim=1,
            block_dim=n,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        ctx.enqueue_copy(out_host, out)
        ctx.synchronize()
        if attempt < 0:
            continue
        var stale = True
        for i in range(n):
            stale = stale and out_host[i] == _SENTINEL
        if stale:
            overlapped += 1
    if overlapped == 0:
        raise Error(
            "an unwaited PDL dependent never overlapped its producer at"
            " delay_us = "
            + String(delay_us)
            + "; the race cases cannot detect a missing wait on this GPU"
        )
    print(
        "overlap:",
        overlapped,
        "/",
        _CANARY_ATTEMPTS,
        "at delay_us =",
        delay_us,
    )


def _check(
    label: String,
    got: HostBuffer[_DT],
    expected: HostBuffer[_DT],
    n: Int,
) -> Bool:
    var bad = 0
    var first = -1
    for i in range(n):
        if got[i].to_bits[.uint32]() != expected[i].to_bits[.uint32]():
            if first < 0:
                first = i
            bad += 1
    if bad > 0:
        print(
            "FAIL:",
            label,
            "-",
            bad,
            "/",
            n,
            "elements differ from the serialized run; first at",
            first,
            "got",
            got[first],
            "expected",
            expected[first],
        )
        return False
    return True


# ===-----------------------------------------------------------------------===#
# _bmm0_bs: S = Q.K^T
# ===-----------------------------------------------------------------------===#

comptime _B = 2
comptime _H = 4
comptime _KVH = 2
comptime _GROUP = _H // _KVH
comptime _S = 32
comptime _D = 64


def _launch_bmm0(
    ctx: DeviceContext,
    p: DeviceBuffer[_DT],
    q: DeviceBuffer[_DT],
    k: DeviceBuffer[_DT],
    vl: DeviceBuffer[.uint32],
) raises:
    var k_op = TileTensorMHAOperand(
        TileTensor(k.unsafe_ptr(), row_major[_B, _S, _KVH, _D]()).as_imm()
    )
    var valid_length = OptionalReg[ImmutTileTensor1D[.uint32]](
        immut_tile_tensor_1d(vl.unsafe_ptr(), _B)
    )
    comptime kernel = _bmm0_bs[
        _DT,
        type_of(k_op),
        NullMask,
        _DT,
        ragged=False,
        _use_valid_length=True,
        _is_cache_length_accurate=True,
    ]
    ctx.enqueue_function[kernel](
        p,
        q,
        k_op,
        valid_length,
        Float32(0.125),
        Int32(_B),
        Int32(_S),
        Int32(_S),
        Int32(_H),
        Int32(_D),
        Int32(_GROUP),
        NullMask(),
        grid_dim=(ceildiv(_S, 32), ceildiv(_S, 16), _H * _B),
        block_dim=_NAIVE_BMM_BLOCK_DIM,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


def _run_bmm0(
    ctx: DeviceContext,
    label: String,
    late_q: Bool,
    late_k: Bool,
    late_vl: Bool,
    delay_us: Int,
    iters: Int,
) raises -> Bool:
    var q = _Operand[_DT](ctx, _B * _S * _H * _D, 0)
    var k = _Operand[_DT](ctx, _B * _S * _KVH * _D, 0)
    var vl = _Operand[.uint32](ctx, _B, 0)
    q.fill_random(ctx)
    k.fill_random(ctx)
    vl.fill(ctx, [UInt32(_S), UInt32(20)])
    var p_size = _B * _H * _S * _S
    var p_ref = ctx.enqueue_create_buffer[_DT](p_size)
    var p_live = ctx.enqueue_create_buffer[_DT](p_size)
    var p_ref_host = ctx.enqueue_create_host_buffer[_DT](p_size)
    var p_host = ctx.enqueue_create_host_buffer[_DT](p_size)
    var sentinel = ctx.enqueue_create_host_buffer[_DT](p_size)
    for i in range(p_size):
        sentinel[i] = _SENTINEL
    ctx.synchronize()

    ctx.enqueue_copy(p_ref, sentinel)
    _launch_bmm0(ctx, p_ref, q.real, k.real, vl.real)
    ctx.enqueue_copy(p_ref_host, p_ref)
    ctx.synchronize()

    for it in range(iters):
        q.reset(ctx, late_q)
        k.reset(ctx, late_k)
        vl.reset(ctx, late_vl)
        ctx.enqueue_copy(p_live, sentinel)
        _publish(ctx, q, late_q, k, late_k, vl, late_vl, delay_us)
        _launch_bmm0(ctx, p_live, q.live, k.live, vl.live)
        ctx.enqueue_copy(p_host, p_live)
        ctx.synchronize()
        if not _check(
            label + " iter " + String(it), p_host, p_ref_host, p_size
        ):
            return False
    print("PASS:", label, "delay_us =", delay_us, "x", iters)
    return True


# ===-----------------------------------------------------------------------===#
# _bmm1_bs: O = P.V
# ===-----------------------------------------------------------------------===#


def _launch_bmm1(
    ctx: DeviceContext,
    o: DeviceBuffer[_DT],
    p: DeviceBuffer[_DT],
    v: DeviceBuffer[_DT],
    vl: DeviceBuffer[.uint32],
) raises:
    var v_op = TileTensorMHAOperand(
        TileTensor(v.unsafe_ptr(), row_major[_B, _S, _KVH, _D]()).as_imm()
    )
    var valid_length = OptionalReg[ImmutTileTensor1D[.uint32]](
        immut_tile_tensor_1d(vl.unsafe_ptr(), _B)
    )
    comptime kernel = _bmm1_bs[
        _DT,
        _DT,
        type_of(v_op),
        ragged=False,
        _use_valid_length=True,
        _is_cache_length_accurate=True,
    ]
    ctx.enqueue_function[kernel](
        o,
        p,
        v_op,
        valid_length,
        Int32(_S),
        Int32(_S),
        Int32(_H),
        Int32(_D),
        Int32(_GROUP),
        grid_dim=(ceildiv(_D, 32), ceildiv(_S, 16), _H * _B),
        block_dim=_NAIVE_BMM_BLOCK_DIM,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


def _run_bmm1(
    ctx: DeviceContext,
    label: String,
    late_p: Bool,
    late_v: Bool,
    late_vl: Bool,
    delay_us: Int,
    iters: Int,
) raises -> Bool:
    var p = _Operand[_DT](ctx, _B * _H * _S * _S, 0)
    var v = _Operand[_DT](ctx, _B * _S * _KVH * _D, 0)
    var vl = _Operand[.uint32](ctx, _B, 0)
    p.fill_random(ctx)
    v.fill_random(ctx)
    vl.fill(ctx, [UInt32(_S), UInt32(20)])
    var o_size = _B * _S * _H * _D
    var o_ref = ctx.enqueue_create_buffer[_DT](o_size)
    var o_live = ctx.enqueue_create_buffer[_DT](o_size)
    var o_ref_host = ctx.enqueue_create_host_buffer[_DT](o_size)
    var o_host = ctx.enqueue_create_host_buffer[_DT](o_size)
    var sentinel = ctx.enqueue_create_host_buffer[_DT](o_size)
    for i in range(o_size):
        sentinel[i] = _SENTINEL
    ctx.synchronize()

    ctx.enqueue_copy(o_ref, sentinel)
    _launch_bmm1(ctx, o_ref, p.real, v.real, vl.real)
    ctx.enqueue_copy(o_ref_host, o_ref)
    ctx.synchronize()

    for it in range(iters):
        p.reset(ctx, late_p)
        v.reset(ctx, late_v)
        vl.reset(ctx, late_vl)
        ctx.enqueue_copy(o_live, sentinel)
        _publish(ctx, p, late_p, v, late_v, vl, late_vl, delay_us)
        _launch_bmm1(ctx, o_live, p.live, v.live, vl.live)
        ctx.enqueue_copy(o_host, o_live)
        ctx.synchronize()
        if not _check(
            label + " iter " + String(it), o_host, o_ref_host, o_size
        ):
            return False
    print("PASS:", label, "delay_us =", delay_us, "x", iters)
    return True


# ===-----------------------------------------------------------------------===#
# flash_attention: fp32 decode (`mha_decoding`) and the naive pipeline
# ===-----------------------------------------------------------------------===#


def _launch_flash[
    batch: Int,
    seq: Int,
    keys: Int,
    heads: Int,
    kv_heads: Int,
    depth: Int,
    naive: Bool,
](
    ctx: DeviceContext,
    mut o: DeviceBuffer[_DT],
    q: DeviceBuffer[_DT],
    k: DeviceBuffer[_DT],
    v: DeviceBuffer[_DT],
) raises:
    var q_t = TileTensor(
        q.unsafe_ptr(), row_major((batch, seq, Idx[heads], Idx[depth]))
    )
    var k_t = TileTensor(
        k.unsafe_ptr(), row_major((batch, keys, Idx[kv_heads], Idx[depth]))
    )
    var v_t = TileTensor(
        v.unsafe_ptr(), row_major((batch, keys, Idx[kv_heads], Idx[depth]))
    )
    var o_t = TileTensor(
        o.unsafe_ptr(), row_major((batch, seq, Idx[heads], Idx[depth]))
    )
    flash_attention[naive_kernel=naive](
        o_t, q_t, k_t, v_t, NullMask(), Float32(0.125), ctx
    )


def _run_flash[
    batch: Int,
    seq: Int,
    keys: Int,
    heads: Int,
    kv_heads: Int,
    depth: Int,
    naive: Bool,
](
    ctx: DeviceContext,
    label: String,
    late_q: Bool,
    late_k: Bool,
    late_v: Bool,
    delay_us: Int,
    iters: Int,
) raises -> Bool:
    var q = _Operand[_DT](ctx, batch * seq * heads * depth, 0)
    var k = _Operand[_DT](ctx, batch * keys * kv_heads * depth, 0)
    var v = _Operand[_DT](ctx, batch * keys * kv_heads * depth, 0)
    q.fill_random(ctx)
    k.fill_random(ctx)
    v.fill_random(ctx)
    var o_size = batch * seq * heads * depth
    var o_ref = ctx.enqueue_create_buffer[_DT](o_size)
    var o_live = ctx.enqueue_create_buffer[_DT](o_size)
    var o_ref_host = ctx.enqueue_create_host_buffer[_DT](o_size)
    var o_host = ctx.enqueue_create_host_buffer[_DT](o_size)
    var sentinel = ctx.enqueue_create_host_buffer[_DT](o_size)
    for i in range(o_size):
        sentinel[i] = _SENTINEL
    ctx.synchronize()

    ctx.enqueue_copy(o_ref, sentinel)
    _launch_flash[batch, seq, keys, heads, kv_heads, depth, naive](
        ctx, o_ref, q.real, k.real, v.real
    )
    ctx.enqueue_copy(o_ref_host, o_ref)
    ctx.synchronize()

    for it in range(iters):
        q.reset(ctx, late_q)
        k.reset(ctx, late_k)
        v.reset(ctx, late_v)
        ctx.enqueue_copy(o_live, sentinel)
        _publish(ctx, q, late_q, k, late_k, v, late_v, delay_us)
        _launch_flash[batch, seq, keys, heads, kv_heads, depth, naive](
            ctx, o_live, q.live, k.live, v.live
        )
        ctx.enqueue_copy(o_host, o_live)
        ctx.synchronize()
        if not _check(
            label + " iter " + String(it), o_host, o_ref_host, o_size
        ):
            return False
    print("PASS:", label, "delay_us =", delay_us, "x", iters)
    return True


# ===-----------------------------------------------------------------------===#
# Ragged operands: row offsets and cache lengths published by the predecessor
# ===-----------------------------------------------------------------------===#

# Two sequences of 8 and 5 new tokens on top of 20 and 12 cached ones. The K/V
# buffer is padded past the last cache offset because a non-accurate cache
# length makes the kernels read `cache_length + new_tokens` rows per sequence.
comptime _RQ = 13
comptime _RK = 40
comptime _RMAX_PROMPT = 8
comptime _RMAX_CACHE = 32
comptime _RP_SIZE = _B * _H * _RMAX_PROMPT * _RMAX_CACHE


def _launch_bmm0_ragged(
    ctx: DeviceContext,
    p: DeviceBuffer[_DT],
    q: DeviceBuffer[_DT],
    k: DeviceBuffer[_DT],
    q_offsets: DeviceBuffer[.uint32],
    k_offsets: DeviceBuffer[.uint32],
) raises:
    var k_op = RaggedMHAOperand(
        TileTensor[mut=False](k.unsafe_ptr(), row_major[_RK, _KVH, _D]()),
        TileTensor[mut=False](k_offsets.unsafe_ptr(), row_major[_B + 1]()),
    )
    var valid_length = OptionalReg[ImmutTileTensor1D[.uint32]](
        immut_tile_tensor_1d(q_offsets.unsafe_ptr(), _B + 1)
    )
    comptime kernel = _bmm0_bs[
        _DT,
        type_of(k_op),
        NullMask,
        _DT,
        ragged=True,
        _use_valid_length=True,
        _is_cache_length_accurate=False,
    ]
    ctx.enqueue_function[kernel](
        p,
        q,
        k_op,
        valid_length,
        Float32(0.125),
        Int32(_B),
        Int32(_RMAX_PROMPT),
        Int32(_RMAX_CACHE),
        Int32(_H),
        Int32(_D),
        Int32(_GROUP),
        NullMask(),
        grid_dim=(
            ceildiv(_RMAX_CACHE, 32),
            ceildiv(_RMAX_PROMPT, 16),
            _H * _B,
        ),
        block_dim=_NAIVE_BMM_BLOCK_DIM,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


def _launch_bmm1_ragged(
    ctx: DeviceContext,
    o: DeviceBuffer[_DT],
    p: DeviceBuffer[_DT],
    v: DeviceBuffer[_DT],
    q_offsets: DeviceBuffer[.uint32],
    v_offsets: DeviceBuffer[.uint32],
) raises:
    var v_op = RaggedMHAOperand(
        TileTensor[mut=False](v.unsafe_ptr(), row_major[_RK, _KVH, _D]()),
        TileTensor[mut=False](v_offsets.unsafe_ptr(), row_major[_B + 1]()),
    )
    var valid_length = OptionalReg[ImmutTileTensor1D[.uint32]](
        immut_tile_tensor_1d(q_offsets.unsafe_ptr(), _B + 1)
    )
    comptime kernel = _bmm1_bs[
        _DT,
        _DT,
        type_of(v_op),
        ragged=True,
        _use_valid_length=True,
        _is_cache_length_accurate=False,
    ]
    ctx.enqueue_function[kernel](
        o,
        p,
        v_op,
        valid_length,
        Int32(_RMAX_PROMPT),
        Int32(_RMAX_CACHE),
        Int32(_H),
        Int32(_D),
        Int32(_GROUP),
        grid_dim=(ceildiv(_D, 32), ceildiv(_RMAX_PROMPT, 16), _H * _B),
        block_dim=_NAIVE_BMM_BLOCK_DIM,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


def _run_bmm_ragged(
    ctx: DeviceContext,
    label: String,
    is_bmm1: Bool,
    late_offsets: Bool,
    late_cache_offsets: Bool,
    delay_us: Int,
    iters: Int,
) raises -> Bool:
    """Late-publishes the ragged row offsets or the cache offsets that
    `cache_length()` is derived from, for either BMM kernel."""
    var in_size = _RP_SIZE if is_bmm1 else _RQ * _H * _D
    var out_size = _RQ * _H * _D if is_bmm1 else _RP_SIZE
    var data = _Operand[_DT](ctx, in_size, 0)
    var kv = _Operand[_DT](ctx, _RK * _KVH * _D, 0)
    var offsets = _Operand[.uint32](ctx, _B + 1, 0)
    var cache_offsets = _Operand[.uint32](ctx, _B + 1, 0)
    data.fill_random(ctx)
    kv.fill_random(ctx)
    offsets.fill(ctx, [UInt32(0), UInt32(8), UInt32(13)])
    cache_offsets.fill(ctx, [UInt32(0), UInt32(20), UInt32(32)])
    var ref_dev = ctx.enqueue_create_buffer[_DT](out_size)
    var live_dev = ctx.enqueue_create_buffer[_DT](out_size)
    var ref_host = ctx.enqueue_create_host_buffer[_DT](out_size)
    var host = ctx.enqueue_create_host_buffer[_DT](out_size)
    var sentinel = ctx.enqueue_create_host_buffer[_DT](out_size)
    for i in range(out_size):
        sentinel[i] = _SENTINEL
    ctx.synchronize()

    ctx.enqueue_copy(ref_dev, sentinel)
    if is_bmm1:
        _launch_bmm1_ragged(
            ctx, ref_dev, data.real, kv.real, offsets.real, cache_offsets.real
        )
    else:
        _launch_bmm0_ragged(
            ctx, ref_dev, data.real, kv.real, offsets.real, cache_offsets.real
        )
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.synchronize()

    for it in range(iters):
        data.reset(ctx, False)
        kv.reset(ctx, False)
        offsets.reset(ctx, late_offsets)
        cache_offsets.reset(ctx, late_cache_offsets)
        ctx.enqueue_copy(live_dev, sentinel)
        _publish(
            ctx,
            offsets,
            late_offsets,
            cache_offsets,
            late_cache_offsets,
            data,
            False,
            delay_us,
        )
        if is_bmm1:
            _launch_bmm1_ragged(
                ctx,
                live_dev,
                data.live,
                kv.live,
                offsets.live,
                cache_offsets.live,
            )
        else:
            _launch_bmm0_ragged(
                ctx,
                live_dev,
                data.live,
                kv.live,
                offsets.live,
                cache_offsets.live,
            )
        ctx.enqueue_copy(host, live_dev)
        ctx.synchronize()
        if not _check(label + " iter " + String(it), host, ref_host, out_size):
            return False
    print("PASS:", label, "delay_us =", delay_us, "x", iters)
    return True


def _launch_ragged_flash[
    heads: Int,
    kv_heads: Int,
    depth: Int,
    naive: Bool,
](
    ctx: DeviceContext,
    mut o: DeviceBuffer[_DT],
    q: DeviceBuffer[_DT],
    k: DeviceBuffer[_DT],
    v: DeviceBuffer[_DT],
    q_offsets: DeviceBuffer[.uint32],
    cache_offsets: DeviceBuffer[.uint32],
    total_q: Int,
    total_k: Int,
    max_prompt_len: Int,
    max_cache_len: Int,
) raises:
    """Launches `flash_attention_dispatch` the way the KV-cache entry point
    does (ragged, `_is_cache_length_accurate=False`), over ragged operands."""
    var q_t = TileTensor[mut=False](
        q.unsafe_ptr(), row_major((total_q, Idx[heads], Idx[depth]))
    )
    var o_t = TileTensor(
        o.unsafe_ptr(), row_major((total_q, Idx[heads], Idx[depth]))
    )
    var cro_t = TileTensor[mut=False](
        cache_offsets.unsafe_ptr(), row_major(_B + 1)
    )
    var k_op = RaggedMHAOperand(
        TileTensor[mut=False](
            k.unsafe_ptr(), row_major((total_k, Idx[kv_heads], Idx[depth]))
        ),
        cro_t,
    )
    var v_op = RaggedMHAOperand(
        TileTensor[mut=False](
            v.unsafe_ptr(), row_major((total_k, Idx[kv_heads], Idx[depth]))
        ),
        cro_t,
    )
    var valid_length = immut_tile_tensor_1d(q_offsets.unsafe_ptr(), _B + 1)
    flash_attention_dispatch[
        kv_num_heads=kv_heads,
        ragged=True,
        _is_flash_attention_applicable=not naive,
    ](
        o_t,
        q_t,
        k_op,
        v_op,
        NullMask(),
        max_prompt_len,
        max_cache_len,
        Float32(0.125),
        max_prompt_len == 1,
        ctx,
        OptionalReg[ImmutTileTensor1D[.uint32]](valid_length),
    )


def _run_ragged_flash[
    heads: Int,
    kv_heads: Int,
    depth: Int,
    naive: Bool,
](
    ctx: DeviceContext,
    label: String,
    late_offsets: Bool,
    late_cache_offsets: Bool,
    row_offsets: List[UInt32],
    cache_row_offsets: List[UInt32],
    max_prompt_len: Int,
    max_cache_len: Int,
    delay_us: Int,
    iters: Int,
) raises -> Bool:
    """Late-publishes the offsets the ragged branches of `mha_decoding` and
    the naive pipeline read: `valid_length` and the K cache lengths."""
    var total_q = Int(row_offsets[_B])
    var total_k = Int(cache_row_offsets[_B]) + max_prompt_len
    var q = _Operand[_DT](ctx, total_q * heads * depth, 0)
    var k = _Operand[_DT](ctx, total_k * kv_heads * depth, 0)
    var v = _Operand[_DT](ctx, total_k * kv_heads * depth, 0)
    var offsets = _Operand[.uint32](ctx, _B + 1, 0)
    var cache_offsets = _Operand[.uint32](ctx, _B + 1, 0)
    q.fill_random(ctx)
    k.fill_random(ctx)
    v.fill_random(ctx)
    offsets.fill(ctx, row_offsets)
    cache_offsets.fill(ctx, cache_row_offsets)
    var o_size = total_q * heads * depth
    var o_ref = ctx.enqueue_create_buffer[_DT](o_size)
    var o_live = ctx.enqueue_create_buffer[_DT](o_size)
    var o_ref_host = ctx.enqueue_create_host_buffer[_DT](o_size)
    var o_host = ctx.enqueue_create_host_buffer[_DT](o_size)
    var sentinel = ctx.enqueue_create_host_buffer[_DT](o_size)
    for i in range(o_size):
        sentinel[i] = _SENTINEL
    ctx.synchronize()

    ctx.enqueue_copy(o_ref, sentinel)
    _launch_ragged_flash[heads, kv_heads, depth, naive](
        ctx,
        o_ref,
        q.real,
        k.real,
        v.real,
        offsets.real,
        cache_offsets.real,
        total_q,
        total_k,
        max_prompt_len,
        max_cache_len,
    )
    ctx.enqueue_copy(o_ref_host, o_ref)
    ctx.synchronize()

    for it in range(iters):
        q.reset(ctx, False)
        k.reset(ctx, False)
        v.reset(ctx, False)
        offsets.reset(ctx, late_offsets)
        cache_offsets.reset(ctx, late_cache_offsets)
        ctx.enqueue_copy(o_live, sentinel)
        _publish(
            ctx,
            offsets,
            late_offsets,
            cache_offsets,
            late_cache_offsets,
            q,
            False,
            delay_us,
        )
        _launch_ragged_flash[heads, kv_heads, depth, naive](
            ctx,
            o_live,
            q.live,
            k.live,
            v.live,
            offsets.live,
            cache_offsets.live,
            total_q,
            total_k,
            max_prompt_len,
            max_cache_len,
        )
        ctx.enqueue_copy(o_host, o_live)
        ctx.synchronize()
        if not _check(
            label + " iter " + String(it), o_host, o_ref_host, o_size
        ):
            return False
    print("PASS:", label, "delay_us =", delay_us, "x", iters)
    return True


# ===-----------------------------------------------------------------------===#
# Non-ragged mha_decoding reading a `valid_length` tensor
# ===-----------------------------------------------------------------------===#


def _launch_decode_valid_length[
    batch: Int,
    keys: Int,
    heads: Int,
    kv_heads: Int,
    depth: Int,
](
    ctx: DeviceContext,
    mut o: DeviceBuffer[_DT],
    q: DeviceBuffer[_DT],
    k: DeviceBuffer[_DT],
    v: DeviceBuffer[_DT],
    vl: DeviceBuffer[.uint32],
) raises:
    """Dense decode with `_use_valid_length=True` and a non-accurate cache
    length, so `mha_decoding` reads `valid_length[b]` into `num_keys`."""
    var q_t = TileTensor[mut=False](
        q.unsafe_ptr(), row_major((batch, 1, Idx[heads], Idx[depth]))
    )
    var o_t = TileTensor(
        o.unsafe_ptr(), row_major((batch, 1, Idx[heads], Idx[depth]))
    )
    var k_op = TileTensorMHAOperand(
        TileTensor[mut=False](
            k.unsafe_ptr(), row_major((batch, keys, Idx[kv_heads], Idx[depth]))
        )
    )
    var v_op = TileTensorMHAOperand(
        TileTensor[mut=False](
            v.unsafe_ptr(), row_major((batch, keys, Idx[kv_heads], Idx[depth]))
        )
    )
    flash_attention_dispatch[
        kv_num_heads=kv_heads,
        ragged=False,
        _use_valid_length=True,
        _is_cache_length_accurate=False,
    ](
        o_t,
        q_t,
        k_op,
        v_op,
        NullMask(),
        1,
        keys,
        Float32(0.125),
        True,
        ctx,
        OptionalReg[ImmutTileTensor1D[.uint32]](
            immut_tile_tensor_1d(vl.unsafe_ptr(), batch)
        ),
    )


def _run_decode_valid_length[
    batch: Int,
    keys: Int,
    heads: Int,
    kv_heads: Int,
    depth: Int,
](ctx: DeviceContext, label: String, delay_us: Int, iters: Int) raises -> Bool:
    """Late-publishes the `valid_length` tensor `mha_decoding` reads."""
    # One row of slack: each sequence attends to `keys + valid_length` rows.
    var kv_size = (batch * keys + 1) * kv_heads * depth
    var q = _Operand[_DT](ctx, batch * heads * depth, 0)
    var k = _Operand[_DT](ctx, kv_size, 0)
    var v = _Operand[_DT](ctx, kv_size, 0)
    var vl = _Operand[.uint32](ctx, batch, 0)
    q.fill_random(ctx)
    k.fill_random(ctx)
    v.fill_random(ctx)
    var ones = List[UInt32]()
    for _ in range(batch):
        ones.append(UInt32(1))
    vl.fill(ctx, ones)
    var o_size = batch * heads * depth
    var o_ref = ctx.enqueue_create_buffer[_DT](o_size)
    var o_live = ctx.enqueue_create_buffer[_DT](o_size)
    var o_ref_host = ctx.enqueue_create_host_buffer[_DT](o_size)
    var o_host = ctx.enqueue_create_host_buffer[_DT](o_size)
    var sentinel = ctx.enqueue_create_host_buffer[_DT](o_size)
    for i in range(o_size):
        sentinel[i] = _SENTINEL
    ctx.synchronize()

    ctx.enqueue_copy(o_ref, sentinel)
    _launch_decode_valid_length[batch, keys, heads, kv_heads, depth](
        ctx, o_ref, q.real, k.real, v.real, vl.real
    )
    ctx.enqueue_copy(o_ref_host, o_ref)
    ctx.synchronize()

    for it in range(iters):
        q.reset(ctx, False)
        k.reset(ctx, False)
        v.reset(ctx, False)
        vl.reset(ctx, True)
        ctx.enqueue_copy(o_live, sentinel)
        _publish(ctx, vl, True, q, False, k, False, delay_us)
        _launch_decode_valid_length[batch, keys, heads, kv_heads, depth](
            ctx, o_live, q.live, k.live, v.live, vl.live
        )
        ctx.enqueue_copy(o_host, o_live)
        ctx.synchronize()
        if not _check(
            label + " iter " + String(it), o_host, o_ref_host, o_size
        ):
            return False
    print("PASS:", label, "delay_us =", delay_us, "x", iters)
    return True


def main() raises:
    var iters = get_defined_int["ITERS", 3]()
    seed(1234567890)
    var ok = True

    with DeviceContext() as ctx:
        # The result must not depend on how long the predecessor keeps its
        # operands unpublished: a kernel that waits is correct at any delay,
        # one that does not is wrong at every delay past its first read.
        for delay_us in [1000, 4000]:
            _dependent_overlaps(ctx, delay_us)
            ok = (
                _run_bmm0(
                    ctx, "bmm0 late Q", True, False, False, delay_us, iters
                )
                and ok
            )
            ok = (
                _run_bmm0(
                    ctx, "bmm0 late K", False, True, False, delay_us, iters
                )
                and ok
            )
            ok = (
                _run_bmm0(
                    ctx,
                    "bmm0 late valid_length",
                    False,
                    False,
                    True,
                    delay_us,
                    iters,
                )
                and ok
            )
            ok = (
                _run_bmm1(
                    ctx, "bmm1 late P", True, False, False, delay_us, iters
                )
                and ok
            )
            ok = (
                _run_bmm1(
                    ctx, "bmm1 late V", False, True, False, delay_us, iters
                )
                and ok
            )
            ok = (
                _run_bmm1(
                    ctx,
                    "bmm1 late valid_length",
                    False,
                    False,
                    True,
                    delay_us,
                    iters,
                )
                and ok
            )

            # fp32 decode: `mha_decoding` on every NVIDIA target, since FA4
            # and FA3 only take half-precision and fp8.
            ok = (
                _run_flash[2, 1, 1024, 8, 8, 128, False](
                    ctx,
                    "mha_decoding late Q",
                    True,
                    False,
                    False,
                    delay_us,
                    iters,
                )
                and ok
            )
            ok = (
                _run_flash[2, 1, 1024, 8, 8, 128, False](
                    ctx,
                    "mha_decoding late K",
                    False,
                    True,
                    False,
                    delay_us,
                    iters,
                )
                and ok
            )
            ok = (
                _run_flash[2, 1, 1024, 8, 8, 128, False](
                    ctx,
                    "mha_decoding late V",
                    False,
                    False,
                    True,
                    delay_us,
                    iters,
                )
                and ok
            )

            # The whole bmm0 -> softmax -> bmm1 pipeline, as dispatched.
            ok = (
                _run_flash[2, 32, 32, 4, 2, 64, True](
                    ctx, "naive late Q", True, False, False, delay_us, iters
                )
                and ok
            )

            # Ragged operands: the row offsets and the cache lengths derived
            # from the K cache offsets are read right after the wait.
            ok = (
                _run_bmm_ragged(
                    ctx,
                    "bmm0 ragged late row offsets",
                    False,
                    True,
                    False,
                    delay_us,
                    iters,
                )
                and ok
            )
            ok = (
                _run_bmm_ragged(
                    ctx,
                    "bmm0 ragged late cache_length",
                    False,
                    False,
                    True,
                    delay_us,
                    iters,
                )
                and ok
            )
            ok = (
                _run_bmm_ragged(
                    ctx,
                    "bmm1 ragged late row offsets",
                    True,
                    True,
                    False,
                    delay_us,
                    iters,
                )
                and ok
            )
            ok = (
                _run_bmm_ragged(
                    ctx,
                    "bmm1 ragged late cache_length",
                    True,
                    False,
                    True,
                    delay_us,
                    iters,
                )
                and ok
            )

            # Production ragged decode: one token per sequence on top of 700
            # and 1000 cached ones.
            ok = (
                _run_ragged_flash[8, 8, 128, False](
                    ctx,
                    "mha_decoding ragged late row offsets",
                    True,
                    False,
                    [UInt32(0), UInt32(1), UInt32(2)],
                    [UInt32(0), UInt32(700), UInt32(1700)],
                    1,
                    1024,
                    delay_us,
                    iters,
                )
                and ok
            )
            ok = (
                _run_ragged_flash[8, 8, 128, False](
                    ctx,
                    "mha_decoding ragged late cache_length",
                    False,
                    True,
                    [UInt32(0), UInt32(1), UInt32(2)],
                    [UInt32(0), UInt32(700), UInt32(1700)],
                    1,
                    1024,
                    delay_us,
                    iters,
                )
                and ok
            )

            # Non-ragged decode: `valid_length` is a tensor, not row offsets.
            ok = (
                _run_decode_valid_length[2, 1024, 8, 8, 128](
                    ctx, "mha_decoding late valid_length", delay_us, iters
                )
                and ok
            )

            # The same ragged inputs through the naive fallback.
            ok = (
                _run_ragged_flash[4, 2, 64, True](
                    ctx,
                    "naive ragged late row offsets",
                    True,
                    False,
                    [UInt32(0), UInt32(8), UInt32(13)],
                    [UInt32(0), UInt32(20), UInt32(32)],
                    8,
                    32,
                    delay_us,
                    iters,
                )
                and ok
            )
            ok = (
                _run_ragged_flash[4, 2, 64, True](
                    ctx,
                    "naive ragged late cache_length",
                    False,
                    True,
                    [UInt32(0), UInt32(8), UInt32(13)],
                    [UInt32(0), UInt32(20), UInt32(32)],
                    8,
                    32,
                    delay_us,
                    iters,
                )
                and ok
            )

    if not ok:
        raise Error("PDL race detected; see FAIL lines above")
    print("All MHA fallback PDL race tests passed!")
