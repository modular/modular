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
"""PDL correctness for the Mamba-2 chain kernels (NVIDIA SM90+).

The varlen causal conv1d, the dstate-split SSD scan and the gated group RMSNorm
each read activations, the sequence table and the cache-slot table (and the
scan its `A`, `D` and `dt_bias`, the conv its taps and bias, the norm its
weight, which the graph slices or casts per call) written by an earlier grid,
and their outputs are read by a later one. This test builds both ends
adversarially with no host synchronisation in between:

- `late_producer` releases its dependents immediately, then sleeps before it
  copies the kernel's inputs into the buffers the kernel reads (which start
  zeroed, and the has-initial-state flags start at the opposite of their final
  value). A kernel that reads an input before its PDL wait sees zeros or a
  stale flag and produces a different result.
- `consumer` waits on the kernel and copies its outputs, which checks that the
  kernel's stores are visible after a dependent's wait. It cannot observe an
  early release: the dependent's wait blocks until the kernel has completed
  either way.

The PDL chain must be bit-identical to the same kernel run with PDL off on the
same inputs, since PDL reorders loads and never arithmetic. Both enabled
levels run: `ON` and `OVERLAP_AT_BEGINNING`, which also releases dependents
right after the kernel's own wait. The conv's multi-token (tile) launch takes
no part in PDL by design, so its cases only check that the level is inert there.

Removing the wait from any of the three kernels fails this test. Removing the
CTA barrier that `wait_on_producers` issues after the wait does not: the
barrier guards against the assembler hoisting `LDG.E.CONSTANT` loads above
`griddepcontrol.wait`, which an earlier layout of the kernels showed. The
current kernels' SASS, dumped at the tile each case launches (the scan's tile
comes from `ssd_scan_tile_split` and `ssd_scan_tile_channels`, as in
production), has no global load before the first `ACQBULK` either way, so no
case here exercises it. Re-dump the SASS when a kernel's tile or load order
changes.
"""

from std.math import ceildiv
from std.random import rand
from std.sys import size_of
from std.sys.intrinsics import llvm_intrinsic

from layout import TileTensor, row_major
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from state_space.gated_group_rmsnorm import gated_group_rmsnorm_gpu
from state_space.mamba2_ssd_scan import (
    mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split,
    ssd_scan_tile_channels,
    ssd_scan_tile_split,
)
from state_space.varlen_causal_conv1d import causal_conv1d_varlen_fwd_gpu

comptime Bytes = MutPointer[UInt8, MutUntrackedOrigin]
comptime NUM_ITERS = 12


def bytes_of[T: DType](buf: DeviceBuffer[T]) -> Bytes:
    """Views a device buffer as raw bytes."""
    return (
        buf.unsafe_ptr()
        .unsafe_bitcast[UInt8]()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


def _copy_bytes(src: Bytes, dst: Bytes, n: Int32):
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(tid, Int(n), stride):
        dst[i] = src[i]


def late_producer(
    s0: Bytes,
    d0: Bytes,
    n0: Int32,
    s1: Bytes,
    d1: Bytes,
    n1: Int32,
    s2: Bytes,
    d2: Bytes,
    n2: Int32,
    s3: Bytes,
    d3: Bytes,
    n3: Int32,
    s4: Bytes,
    d4: Bytes,
    n4: Int32,
    s5: Bytes,
    d5: Bytes,
    n5: Int32,
    s6: Bytes,
    d6: Bytes,
    n6: Int32,
    s7: Bytes,
    d7: Bytes,
    n7: Int32,
):
    """Releases dependents at once, then publishes its buffers late."""
    launch_dependent_grids()
    # nanosleep takes nanoseconds; hold the data back for ~100 us so a
    # dependent that skips its wait is certain to read the zeroed buffers.
    llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(50_000))
    llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(50_000))
    _copy_bytes(s0, d0, n0)
    _copy_bytes(s1, d1, n1)
    _copy_bytes(s2, d2, n2)
    _copy_bytes(s3, d3, n3)
    _copy_bytes(s4, d4, n4)
    _copy_bytes(s5, d5, n5)
    _copy_bytes(s6, d6, n6)
    _copy_bytes(s7, d7, n7)


def consumer(s0: Bytes, d0: Bytes, n0: Int32, s1: Bytes, d1: Bytes, n1: Int32):
    """Copies the kernel's outputs after waiting on it."""
    wait_on_dependent_grids()
    _copy_bytes(s0, d0, n0)
    _copy_bytes(s1, d1, n1)


struct Region(Copyable):
    """A source buffer, the zeroed buffer a kernel reads, and its byte size."""

    var src: Bytes
    var dst: Bytes
    var n: Int32

    def __init__(out self, src: Bytes, dst: Bytes, n: Int):
        self.src = src
        self.dst = dst
        self.n = Int32(n)


def launch_producer(
    ctx: DeviceContext,
    r0: Region,
    r1: Region,
    r2: Region,
    r3: Region,
    r4: Region,
    r5: Region,
    r6: Region,
    r7: Region,
) raises:
    ctx.enqueue_function[late_producer](
        r0.src,
        r0.dst,
        r0.n,
        r1.src,
        r1.dst,
        r1.n,
        r2.src,
        r2.dst,
        r2.n,
        r3.src,
        r3.dst,
        r3.n,
        r4.src,
        r4.dst,
        r4.n,
        r5.src,
        r5.dst,
        r5.n,
        r6.src,
        r6.dst,
        r6.n,
        r7.src,
        r7.dst,
        r7.n,
        grid_dim=2,
        block_dim=128,
    )


def launch_consumer[
    level: PDLLevel
](
    ctx: DeviceContext,
    s0: Bytes,
    d0: Bytes,
    n0: Int,
    s1: Bytes,
    d1: Bytes,
    n1: Int,
) raises:
    ctx.enqueue_function[consumer](
        s0,
        d0,
        Int32(n0),
        s1,
        d1,
        Int32(n1),
        grid_dim=ceildiv(max(n0, n1), 256),
        block_dim=256,
        attributes=pdl_launch_attributes(level),
    )


def bf16_rand(n: Int, scale: Float32) -> List[Scalar[DType.bfloat16]]:
    var f = List[Float32](unsafe_uninit_length=n)
    rand(f.unsafe_ptr(), n)
    var out = List[Scalar[DType.bfloat16]](unsafe_uninit_length=n)
    for i in range(n):
        out[i] = ((f[i] - 0.5) * scale).cast[DType.bfloat16]()
    return out^


def check_equal(
    name: StringLiteral, it: Int, want: List[UInt8], got: List[UInt8]
) raises:
    var mismatch = 0
    for j in range(len(want)):
        if want[j] != got[j]:
            mismatch += 1
    if mismatch != 0:
        raise Error(
            String(
                name,
                ": PDL race: iter ",
                it,
                " has ",
                mismatch,
                " of ",
                len(want),
                " output bytes differing from the PDL-off kernel",
            )
        )


# =============================================================================
# Causal conv1d
# =============================================================================


def conv_launch[
    WIDTH: Int, pdl: PDLLevel
](
    ctx: DeviceContext,
    w_buf: DeviceBuffer[DType.bfloat16],
    bias_buf: DeviceBuffer[DType.bfloat16],
    x_buf: DeviceBuffer[DType.bfloat16],
    qsl_buf: DeviceBuffer[DType.int32],
    slot_buf: DeviceBuffer[DType.uint32],
    his_buf: DeviceBuffer[DType.bool],
    upd_buf: DeviceBuffer[DType.bool],
    mut st_buf: DeviceBuffer[DType.bfloat16],
    mut out_buf: DeviceBuffer[DType.bfloat16],
    dim: Int,
    total: Int,
    batch: Int,
    nslots: Int,
) raises:
    comptime dtype = DType.bfloat16
    var zero = 0
    var w_t = TileTensor(w_buf, row_major(dim, WIDTH))
    var bias_t = TileTensor(bias_buf, row_major(dim))
    var x_t = TileTensor(x_buf, row_major(total, dim))
    var qsl_t = TileTensor(qsl_buf, row_major(batch + 1))
    var slot_t = TileTensor(slot_buf, row_major(batch))
    var his_t = TileTensor(his_buf, row_major(batch))
    var upd_t = TileTensor(upd_buf, row_major(zero))
    var st_t = TileTensor(st_buf, row_major(nslots, dim, WIDTH - 1))
    var out_t = TileTensor(out_buf, row_major(total, dim))

    def x_fn[
        width: Int, alignment: Int
    ](i: Int, j: Int) {var x_t} -> SIMD[dtype, width]:
        return x_t.load[width=width]((i, j))

    def slot_fn[
        width: Int, alignment: Int
    ](b: Int) {var slot_t} -> SIMD[DType.uint32, width]:
        return slot_t.load[width=width]((b,))

    causal_conv1d_varlen_fwd_gpu[WIDTH, True, False, True, pdl](
        w_t,
        bias_t,
        qsl_t,
        his_t,
        upd_t,
        st_t,
        out_t,
        0,
        0,
        x_fn,
        slot_fn,
        ctx,
        x_vector_loads=False,
    )


def conv_case[
    level: PDLLevel, WIDTH: Int
](ctx: DeviceContext, seq_lens: List[Int], dim: Int) raises:
    comptime dtype = DType.bfloat16
    var batch = len(seq_lens)
    var total = 0
    var qsl_h = List[Int32](unsafe_uninit_length=batch + 1)
    qsl_h[0] = 0
    for b in range(batch):
        total += seq_lens[b]
        qsl_h[b + 1] = Int32(total)
    var state_len = WIDTH - 1
    var nslots = batch + 3
    print("conv1d PDL: batch=", batch, " tokens=", total, " dim=", dim, sep="")

    # Slots are a permutation with spare rows, and odd sequences continue
    # from a stored state.
    var slot_h = List[UInt32](unsafe_uninit_length=batch)
    var his_h = List[Scalar[DType.bool]](unsafe_uninit_length=batch)
    # What the PDL arm's flags hold until the producer writes them: the
    # opposite of every flag, so an early read changes the output.
    var his_stale_h = List[Scalar[DType.bool]](unsafe_uninit_length=batch)
    for b in range(batch):
        slot_h[b] = UInt32(nslots - 1 - b)
        his_h[b] = Scalar[DType.bool](b % 2 == 1)
        his_stale_h[b] = Scalar[DType.bool](b % 2 == 0)
    var x_h = bf16_rand(total * dim, 4.0)
    var w_h = bf16_rand(dim * WIDTH, 1.0)
    var bias_h = bf16_rand(dim, 1.0)
    var st_h = bf16_rand(nslots * dim * state_len, 4.0)

    var x_src = ctx.enqueue_create_buffer[dtype](total * dim)
    var x_ref = ctx.enqueue_create_buffer[dtype](total * dim)
    var x_dst = ctx.enqueue_create_buffer[dtype](total * dim)
    var qsl_src = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var qsl_dst = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var slot_src = ctx.enqueue_create_buffer[DType.uint32](batch)
    var slot_dst = ctx.enqueue_create_buffer[DType.uint32](batch)
    var his_src = ctx.enqueue_create_buffer[DType.bool](batch)
    var his_dst = ctx.enqueue_create_buffer[DType.bool](batch)
    var w_d = ctx.enqueue_create_buffer[dtype](dim * WIDTH)
    var bias_d = ctx.enqueue_create_buffer[dtype](dim)
    var w_dst = ctx.enqueue_create_buffer[dtype](dim * WIDTH)
    var bias_dst = ctx.enqueue_create_buffer[dtype](dim)
    var upd_d = ctx.enqueue_create_buffer[DType.bool](1)
    var out_ref = ctx.enqueue_create_buffer[dtype](total * dim)
    var out_pdl = ctx.enqueue_create_buffer[dtype](total * dim)
    var st_ref = ctx.enqueue_create_buffer[dtype](nslots * dim * state_len)
    var st_pdl = ctx.enqueue_create_buffer[dtype](nslots * dim * state_len)
    var out_bytes = total * dim * 2
    var st_bytes = nslots * dim * state_len * 2
    var want_out = ctx.enqueue_create_buffer[DType.uint8](out_bytes)
    var want_st = ctx.enqueue_create_buffer[DType.uint8](st_bytes)
    var got_out = ctx.enqueue_create_buffer[DType.uint8](out_bytes)
    var got_st = ctx.enqueue_create_buffer[DType.uint8](st_bytes)
    ctx.enqueue_copy(x_src, x_h)
    ctx.enqueue_copy(x_ref, x_h)
    ctx.enqueue_copy(qsl_src, qsl_h)
    ctx.enqueue_copy(slot_src, slot_h)
    ctx.enqueue_copy(his_src, his_h)
    ctx.enqueue_copy(w_d, w_h)
    ctx.enqueue_copy(bias_d, bias_h)
    ctx.enqueue_copy(st_ref, st_h)
    ctx.enqueue_copy(st_pdl, st_h)

    # Reference: PDL off, straight from the sources.
    conv_launch[WIDTH, PDLLevel.OFF](
        ctx,
        w_d,
        bias_d,
        x_ref,
        qsl_src,
        slot_src,
        his_src,
        upd_d,
        st_ref,
        out_ref,
        dim,
        total,
        batch,
        nslots,
    )
    launch_consumer[PDLLevel.OFF](
        ctx,
        bytes_of(out_ref),
        bytes_of(want_out),
        out_bytes,
        bytes_of(st_ref),
        bytes_of(want_st),
        st_bytes,
    )
    var want_o = List[UInt8](unsafe_uninit_length=out_bytes)
    var want_s = List[UInt8](unsafe_uninit_length=st_bytes)
    ctx.enqueue_copy(want_o, want_out)
    ctx.enqueue_copy(want_s, want_st)
    ctx.synchronize()

    var got_o = List[UInt8](unsafe_uninit_length=out_bytes)
    var got_s = List[UInt8](unsafe_uninit_length=st_bytes)
    for it in range(NUM_ITERS):
        ctx.enqueue_memset[dtype](x_dst, 0)
        ctx.enqueue_memset[DType.int32](qsl_dst, 0)
        ctx.enqueue_memset[DType.uint32](slot_dst, 0)
        ctx.enqueue_memset[dtype](w_dst, 0)
        ctx.enqueue_memset[dtype](bias_dst, 0)
        ctx.enqueue_copy(his_dst, his_stale_h)
        ctx.enqueue_memset[dtype](out_pdl, 0)
        ctx.enqueue_copy(st_pdl, st_h)
        var empty = Region(
            bytes_of(x_src),
            bytes_of(x_dst),
            0,
        )
        launch_producer(
            ctx,
            Region(
                bytes_of(slot_src),
                bytes_of(slot_dst),
                batch * 4,
            ),
            Region(
                bytes_of(qsl_src),
                bytes_of(qsl_dst),
                (batch + 1) * 4,
            ),
            Region(
                bytes_of(his_src),
                bytes_of(his_dst),
                batch,
            ),
            Region(
                bytes_of(x_src),
                bytes_of(x_dst),
                total * dim * 2,
            ),
            Region(
                bytes_of(w_d),
                bytes_of(w_dst),
                dim * WIDTH * 2,
            ),
            Region(
                bytes_of(bias_d),
                bytes_of(bias_dst),
                dim * 2,
            ),
            empty,
            empty,
        )
        conv_launch[WIDTH, level](
            ctx,
            w_dst,
            bias_dst,
            x_dst,
            qsl_dst,
            slot_dst,
            his_dst,
            upd_d,
            st_pdl,
            out_pdl,
            dim,
            total,
            batch,
            nslots,
        )
        launch_consumer[level](
            ctx,
            bytes_of(out_pdl),
            bytes_of(got_out),
            out_bytes,
            bytes_of(st_pdl),
            bytes_of(got_st),
            st_bytes,
        )
        ctx.enqueue_copy(got_o, got_out)
        ctx.enqueue_copy(got_s, got_st)
        ctx.synchronize()
        check_equal("conv1d state", it, want_s, got_s)
        check_equal("conv1d output", it, want_o, got_o)


# =============================================================================
# SSD scan (dstate split)
# =============================================================================


def scan_launch[
    prefill: Bool, pdl: PDLLevel
](
    ctx: DeviceContext,
    a_buf: DeviceBuffer[DType.bfloat16],
    d_buf: DeviceBuffer[DType.bfloat16],
    dtb_buf: DeviceBuffer[DType.bfloat16],
    slot_buf: DeviceBuffer[DType.uint32],
    his_buf: DeviceBuffer[DType.bool],
    upd_buf: DeviceBuffer[DType.bool],
    x_buf: DeviceBuffer[DType.bfloat16],
    dt_buf: DeviceBuffer[DType.bfloat16],
    b_buf: DeviceBuffer[DType.bfloat16],
    c_buf: DeviceBuffer[DType.bfloat16],
    qsl_buf: DeviceBuffer[DType.int32],
    mut pool_buf: DeviceBuffer[DType.float32],
    mut y_buf: DeviceBuffer[DType.bfloat16],
    total: Int,
    batch: Int,
    nslots: Int,
) raises:
    comptime dtype = DType.bfloat16
    comptime state_dtype = DType.float32
    comptime DSTATE = 128
    comptime DSTATE_SPLIT = ssd_scan_tile_split(DSTATE, prefill, True, False)
    comptime CH_PER_THREAD = ssd_scan_tile_channels(
        DSTATE, prefill, True, False
    )
    comptime ROWS = 128 // DSTATE_SPLIT
    comptime CH_PER_BLOCK = ROWS * CH_PER_THREAD
    comptime nheads = 12
    comptime head_dim = 80
    comptime ngroups = 4
    var zero = 0
    var a_t = TileTensor(a_buf, row_major(nheads))
    var d_t = TileTensor(d_buf, row_major(nheads))
    var dtb_t = TileTensor(dtb_buf, row_major(nheads))
    var slot_t = TileTensor(slot_buf, row_major(batch))
    var his_t = TileTensor(his_buf, row_major(batch))
    var upd_t = TileTensor(upd_buf, row_major(zero))
    var x_t = TileTensor(x_buf, row_major(total, nheads, head_dim))
    var dt_t = TileTensor(dt_buf, row_major(total, nheads))
    var b_t = TileTensor(b_buf, row_major(total, ngroups, DSTATE))
    var c_t = TileTensor(c_buf, row_major(total, ngroups, DSTATE))
    var qsl_t = TileTensor(qsl_buf, row_major(batch + 1))
    var y_t = TileTensor(y_buf, row_major(total, nheads, head_dim))
    var pool_t = TileTensor(
        pool_buf, row_major(nslots, nheads, head_dim, DSTATE)
    )

    def x_fn[
        width: Int, alignment: Int
    ](t: Int, h: Int, p: Int) {var x_t} -> SIMD[dtype, width]:
        return x_t.load[width=width, alignment=alignment * 2]((t, h, p))

    def dt_fn[
        width: Int, alignment: Int
    ](t: Int, h: Int) {var dt_t} -> SIMD[dtype, width]:
        return dt_t.load[width=width, alignment=alignment * 2]((t, h))

    def b_fn[
        width: Int, alignment: Int
    ](t: Int, g: Int, n: Int) {var b_t} -> SIMD[dtype, width]:
        return b_t.load[width=width, alignment=alignment * 2]((t, g, n))

    def c_fn[
        width: Int, alignment: Int
    ](t: Int, g: Int, n: Int) {var c_t} -> SIMD[dtype, width]:
        return c_t.load[width=width, alignment=alignment * 2]((t, g, n))

    def slot_fn[
        width: Int, alignment: Int
    ](b: Int) {var slot_t} -> SIMD[DType.uint32, width]:
        return slot_t.load[width=width]((b,))

    comptime kernel = mamba2_ssd_chunk_scan_varlen_fwd_inplace_gpu_dstate_split[
        dtype,
        state_dtype,
        DSTATE,
        DSTATE_SPLIT,
        a_t.LayoutType,
        d_t.LayoutType,
        dtb_t.LayoutType,
        y_t.LayoutType,
        pool_t.LayoutType,
        qsl_t.LayoutType,
        his_t.LayoutType,
        upd_t.LayoutType,
        y_t.Engine,
        type_of(x_fn),
        type_of(dt_fn),
        type_of(b_fn),
        type_of(c_fn),
        type_of(slot_fn),
        True,
        prefill,
        CH_PER_THREAD=CH_PER_THREAD,
        pdl_level=pdl,
    ]
    ctx.enqueue_function[kernel](
        Int32(ngroups),
        a_t,
        d_t,
        dtb_t,
        y_t,
        pool_t,
        qsl_t,
        his_t,
        upd_t,
        host_args=(x_fn, dt_fn, b_fn, c_fn, slot_fn),
        grid_dim=(ceildiv(head_dim, CH_PER_BLOCK), nheads, batch),
        block_dim=(DSTATE_SPLIT, ROWS, 1),
        attributes=pdl_launch_attributes(pdl),
    )


def scan_case[
    level: PDLLevel, prefill: Bool
](ctx: DeviceContext, seq_lens: List[Int]) raises:
    comptime dtype = DType.bfloat16
    comptime state_dtype = DType.float32
    comptime DSTATE = 128
    comptime nheads = 12
    comptime head_dim = 80
    comptime ngroups = 4
    var batch = len(seq_lens)
    var total = 0
    var qsl_h = List[Int32](unsafe_uninit_length=batch + 1)
    qsl_h[0] = 0
    for b in range(batch):
        total += seq_lens[b]
        qsl_h[b + 1] = Int32(total)
    var nslots = batch + 2
    var pool_n = nslots * nheads * head_dim * DSTATE
    print("ssd scan PDL: batch=", batch, " tokens=", total, sep="")

    var slot_h = List[UInt32](unsafe_uninit_length=batch)
    var his_h = List[Scalar[DType.bool]](unsafe_uninit_length=batch)
    # What the PDL arm's flags hold until the producer writes them: the
    # opposite of every flag, so an early read changes the output.
    var his_stale_h = List[Scalar[DType.bool]](unsafe_uninit_length=batch)
    for b in range(batch):
        slot_h[b] = UInt32(nslots - 1 - b)
        his_h[b] = Scalar[DType.bool](b % 2 == 0)
        his_stale_h[b] = Scalar[DType.bool](b % 2 == 1)
    var x_h = bf16_rand(total * nheads * head_dim, 2.0)
    var dt_h = bf16_rand(total * nheads, 2.0)
    var b_h = bf16_rand(total * ngroups * DSTATE, 2.0)
    var c_h = bf16_rand(total * ngroups * DSTATE, 2.0)
    var a_h = bf16_rand(nheads, 1.0)
    for i in range(nheads):
        a_h[i] = -abs(a_h[i]) - 0.1
    var d_h = bf16_rand(nheads, 1.0)
    var dtb_h = bf16_rand(nheads, 1.0)
    var pool_h = List[Float32](unsafe_uninit_length=pool_n)
    rand(pool_h.unsafe_ptr(), pool_n)

    var x_src = ctx.enqueue_create_buffer[dtype](len(x_h))
    var x_dst = ctx.enqueue_create_buffer[dtype](len(x_h))
    var dt_src = ctx.enqueue_create_buffer[dtype](len(dt_h))
    var dt_dst = ctx.enqueue_create_buffer[dtype](len(dt_h))
    var b_src = ctx.enqueue_create_buffer[dtype](len(b_h))
    var b_dst = ctx.enqueue_create_buffer[dtype](len(b_h))
    var c_src = ctx.enqueue_create_buffer[dtype](len(c_h))
    var c_dst = ctx.enqueue_create_buffer[dtype](len(c_h))
    var qsl_src = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var qsl_dst = ctx.enqueue_create_buffer[DType.int32](batch + 1)
    var a_d = ctx.enqueue_create_buffer[dtype](nheads)
    var d_d = ctx.enqueue_create_buffer[dtype](nheads)
    var dtb_d = ctx.enqueue_create_buffer[dtype](nheads)
    var slot_src = ctx.enqueue_create_buffer[DType.uint32](batch)
    var slot_dst = ctx.enqueue_create_buffer[DType.uint32](batch)
    var his_d = ctx.enqueue_create_buffer[DType.bool](batch)
    var his_src = ctx.enqueue_create_buffer[DType.bool](batch)
    var his_dst = ctx.enqueue_create_buffer[DType.bool](batch)
    var upd_d = ctx.enqueue_create_buffer[DType.bool](1)
    var y_n = total * nheads * head_dim
    var y_ref = ctx.enqueue_create_buffer[dtype](y_n)
    var y_pdl = ctx.enqueue_create_buffer[dtype](y_n)
    var pool_ref = ctx.enqueue_create_buffer[state_dtype](pool_n)
    var pool_pdl = ctx.enqueue_create_buffer[state_dtype](pool_n)
    var y_bytes = y_n * 2
    var pool_bytes = pool_n * 4
    var want_y = ctx.enqueue_create_buffer[DType.uint8](y_bytes)
    var want_p = ctx.enqueue_create_buffer[DType.uint8](pool_bytes)
    var got_y = ctx.enqueue_create_buffer[DType.uint8](y_bytes)
    var got_p = ctx.enqueue_create_buffer[DType.uint8](pool_bytes)
    ctx.enqueue_copy(x_src, x_h)
    ctx.enqueue_copy(dt_src, dt_h)
    ctx.enqueue_copy(b_src, b_h)
    ctx.enqueue_copy(c_src, c_h)
    ctx.enqueue_copy(qsl_src, qsl_h)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(d_d, d_h)
    ctx.enqueue_copy(dtb_d, dtb_h)
    # `A`, `D` and `dt_bias` are cast or derived in the graph, so an earlier
    # grid can write them just before the scan; the chain publishes them late
    # as well.
    var par_h = List[Scalar[dtype]](unsafe_uninit_length=3 * nheads)
    for i in range(nheads):
        par_h[i] = a_h[i]
        par_h[nheads + i] = d_h[i]
        par_h[2 * nheads + i] = dtb_h[i]
    var par_src = ctx.enqueue_create_buffer[dtype](3 * nheads)
    var par_dst = ctx.enqueue_create_buffer[dtype](3 * nheads)
    ctx.enqueue_copy(par_src, par_h)
    var a_dst = par_dst.create_sub_buffer[dtype](0, nheads)
    var d_dst = par_dst.create_sub_buffer[dtype](nheads, nheads)
    var dtb_dst = par_dst.create_sub_buffer[dtype](2 * nheads, nheads)
    ctx.enqueue_copy(slot_src, slot_h)
    ctx.enqueue_copy(his_d, his_h)
    ctx.enqueue_copy(his_src, his_h)
    ctx.enqueue_copy(pool_ref, pool_h)

    scan_launch[prefill, PDLLevel.OFF](
        ctx,
        a_d,
        d_d,
        dtb_d,
        slot_src,
        his_d,
        upd_d,
        x_src,
        dt_src,
        b_src,
        c_src,
        qsl_src,
        pool_ref,
        y_ref,
        total,
        batch,
        nslots,
    )
    launch_consumer[PDLLevel.OFF](
        ctx,
        bytes_of(y_ref),
        bytes_of(want_y),
        y_bytes,
        bytes_of(pool_ref),
        bytes_of(want_p),
        pool_bytes,
    )
    var want_yh = List[UInt8](unsafe_uninit_length=y_bytes)
    var want_ph = List[UInt8](unsafe_uninit_length=pool_bytes)
    ctx.enqueue_copy(want_yh, want_y)
    ctx.enqueue_copy(want_ph, want_p)
    ctx.synchronize()

    var got_yh = List[UInt8](unsafe_uninit_length=y_bytes)
    var got_ph = List[UInt8](unsafe_uninit_length=pool_bytes)
    for it in range(NUM_ITERS):
        ctx.enqueue_memset[dtype](x_dst, 0)
        ctx.enqueue_memset[dtype](dt_dst, 0)
        ctx.enqueue_memset[dtype](b_dst, 0)
        ctx.enqueue_memset[dtype](c_dst, 0)
        ctx.enqueue_memset[DType.int32](qsl_dst, 0)
        ctx.enqueue_memset[DType.uint32](slot_dst, 0)
        ctx.enqueue_memset[dtype](par_dst, 0)
        ctx.enqueue_copy(his_dst, his_stale_h)
        ctx.enqueue_memset[dtype](y_pdl, 0)
        ctx.enqueue_copy(pool_pdl, pool_h)
        launch_producer(
            ctx,
            Region(
                bytes_of(x_src),
                bytes_of(x_dst),
                len(x_h) * 2,
            ),
            Region(
                bytes_of(dt_src),
                bytes_of(dt_dst),
                len(dt_h) * 2,
            ),
            Region(
                bytes_of(b_src),
                bytes_of(b_dst),
                len(b_h) * 2,
            ),
            Region(
                bytes_of(c_src),
                bytes_of(c_dst),
                len(c_h) * 2,
            ),
            Region(
                bytes_of(qsl_src),
                bytes_of(qsl_dst),
                (batch + 1) * 4,
            ),
            Region(
                bytes_of(slot_src),
                bytes_of(slot_dst),
                batch * 4,
            ),
            Region(
                bytes_of(par_src),
                bytes_of(par_dst),
                3 * nheads * 2,
            ),
            Region(
                bytes_of(his_src),
                bytes_of(his_dst),
                batch,
            ),
        )
        scan_launch[prefill, level](
            ctx,
            a_dst,
            d_dst,
            dtb_dst,
            slot_dst,
            his_dst,
            upd_d,
            x_dst,
            dt_dst,
            b_dst,
            c_dst,
            qsl_dst,
            pool_pdl,
            y_pdl,
            total,
            batch,
            nslots,
        )
        launch_consumer[level](
            ctx,
            bytes_of(y_pdl),
            bytes_of(got_y),
            y_bytes,
            bytes_of(pool_pdl),
            bytes_of(got_p),
            pool_bytes,
        )
        ctx.enqueue_copy(got_yh, got_y)
        ctx.enqueue_copy(got_ph, got_p)
        ctx.synchronize()
        check_equal("ssd scan y", it, want_yh, got_yh)
        check_equal("ssd scan state", it, want_ph, got_ph)


# =============================================================================
# Gated group RMSNorm
# =============================================================================


def norm_launch[
    group_size: Int, gate_dtype: DType, pdl: PDLLevel
](
    ctx: DeviceContext,
    w_buf: DeviceBuffer[DType.float32],
    y_buf: DeviceBuffer[DType.bfloat16],
    gate_buf: DeviceBuffer[gate_dtype],
    mut out_buf: DeviceBuffer[DType.bfloat16],
    n_rows: Int,
    num_groups: Int,
) raises:
    var width = num_groups * group_size
    var w_t = TileTensor(w_buf, row_major(width))
    var y_t = TileTensor(y_buf, row_major(n_rows, width))
    var gate_t = TileTensor(gate_buf, row_major(n_rows, width))
    var out_t = TileTensor(out_buf, row_major(n_rows, width))

    def gate_fn[
        w: Int, alignment: Int
    ](r: Int, col: Int) {var gate_t} -> SIMD[gate_dtype, w]:
        return gate_t.load[
            width=w, alignment=alignment * size_of[gate_dtype]()
        ]((r, col))

    gated_group_rmsnorm_gpu[
        DType.bfloat16, gate_dtype, group_size, type_of(gate_fn), pdl
    ](out_t, y_t, gate_fn, w_t, n_rows, num_groups, Float32(1e-5), ctx)


def norm_case[
    level: PDLLevel, group_size: Int, gate_dtype: DType
](ctx: DeviceContext, n_rows: Int, num_groups: Int) raises:
    comptime dtype = DType.bfloat16
    var width = num_groups * group_size
    var n = n_rows * width
    print(
        "gated group rmsnorm PDL: rows=",
        n_rows,
        " groups=",
        num_groups,
        " group_size=",
        group_size,
        sep="",
    )
    var y_h = bf16_rand(n, 8.0)
    var gate_f = List[Float32](unsafe_uninit_length=n)
    rand(gate_f.unsafe_ptr(), n)
    var gate_h = List[Scalar[gate_dtype]](unsafe_uninit_length=n)
    for i in range(n):
        gate_h[i] = ((gate_f[i] - 0.5) * 8.0).cast[gate_dtype]()
    var w_h = List[Float32](unsafe_uninit_length=width)
    rand(w_h.unsafe_ptr(), width)

    var y_src = ctx.enqueue_create_buffer[dtype](n)
    var y_dst = ctx.enqueue_create_buffer[dtype](n)
    var gate_src = ctx.enqueue_create_buffer[gate_dtype](n)
    var gate_dst = ctx.enqueue_create_buffer[gate_dtype](n)
    var w_d = ctx.enqueue_create_buffer[DType.float32](width)
    var w_dst = ctx.enqueue_create_buffer[DType.float32](width)
    var out_ref = ctx.enqueue_create_buffer[dtype](n)
    var out_pdl = ctx.enqueue_create_buffer[dtype](n)
    var out_bytes = n * 2
    var want_out = ctx.enqueue_create_buffer[DType.uint8](out_bytes)
    var got_out = ctx.enqueue_create_buffer[DType.uint8](out_bytes)
    ctx.enqueue_copy(y_src, y_h)
    ctx.enqueue_copy(gate_src, gate_h)
    ctx.enqueue_copy(w_d, w_h)
    norm_launch[group_size, gate_dtype, PDLLevel.OFF](
        ctx, w_d, y_src, gate_src, out_ref, n_rows, num_groups
    )
    launch_consumer[PDLLevel.OFF](
        ctx,
        bytes_of(out_ref),
        bytes_of(want_out),
        out_bytes,
        bytes_of(out_ref),
        bytes_of(want_out),
        0,
    )
    var want_h = List[UInt8](unsafe_uninit_length=out_bytes)
    ctx.enqueue_copy(want_h, want_out)
    ctx.synchronize()

    var got_h = List[UInt8](unsafe_uninit_length=out_bytes)
    for it in range(NUM_ITERS):
        ctx.enqueue_memset[dtype](y_dst, 0)
        ctx.enqueue_memset[gate_dtype](gate_dst, 0)
        ctx.enqueue_memset[DType.float32](w_dst, 0)
        ctx.enqueue_memset[dtype](out_pdl, 0)
        var empty = Region(
            bytes_of(y_src),
            bytes_of(y_dst),
            0,
        )
        launch_producer(
            ctx,
            Region(
                bytes_of(y_src),
                bytes_of(y_dst),
                out_bytes,
            ),
            Region(
                bytes_of(gate_src),
                bytes_of(gate_dst),
                n * size_of[gate_dtype](),
            ),
            Region(
                bytes_of(w_d),
                bytes_of(w_dst),
                width * 4,
            ),
            empty,
            empty,
            empty,
            empty,
            empty,
        )
        norm_launch[group_size, gate_dtype, level](
            ctx, w_dst, y_dst, gate_dst, out_pdl, n_rows, num_groups
        )
        launch_consumer[level](
            ctx,
            bytes_of(out_pdl),
            bytes_of(got_out),
            out_bytes,
            bytes_of(out_pdl),
            bytes_of(got_out),
            0,
        )
        ctx.enqueue_copy(got_h, got_out)
        ctx.synchronize()
        check_equal("gated group rmsnorm", it, want_h, got_h)


def run_level[level: PDLLevel](ctx: DeviceContext) raises:
    # Single-token decode takes the one-row-per-sequence conv path; the ragged
    # prefill lengths take the tile path, with a sequence that is empty of
    # history and one spanning several tiles.
    var decode = List[Int]()
    for _ in range(7):
        decode.append(1)
    conv_case[level, 4](ctx, decode, 9728)
    var prefill = List[Int]()
    prefill.append(70)
    prefill.append(1)
    prefill.append(133)
    prefill.append(5)
    conv_case[level, 4](ctx, prefill, 9728)
    conv_case[level, 4](ctx, prefill, 9731)

    scan_case[level, False](ctx, decode)
    scan_case[level, True](ctx, prefill)

    norm_case[level, 960, DType.float32](ctx, 1, 8)
    norm_case[level, 960, DType.float32](ctx, 37, 8)
    norm_case[level, 100, DType.bfloat16](ctx, 5, 3)


def main() raises:
    with DeviceContext() as ctx:
        if not ctx.is_compatible():
            return
        # PDL is an NVIDIA launch attribute; the kernels ignore it elsewhere.
        comptime if ctx.target.is_nvidia_gpu():
            run_level[PDLLevel.ON](ctx)
            run_level[PDLLevel.OVERLAP_AT_BEGINNING](ctx)
            print("PASSED")
