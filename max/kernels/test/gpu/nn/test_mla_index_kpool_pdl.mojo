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
"""PDL race test for the k-pool kernels.

Each k-pool kernel is launched with the programmatic-dependent-launch
attribute straight behind a chain of producer grids, with no host sync in
between. Every producer releases its dependent grid on entry, sleeps, and only
then publishes one input buffer, so a k-pool kernel that read an input before
its own `wait_on_dependent_grids()` would see the zeros the buffer was
poisoned with. The producers wait on each other, so the last one completing
means they all have.

The result of every PDL launch must be bit-identical to the same kernel run
with PDL off on host-synchronised inputs. A consumer grid launched behind each
k-pool kernel then copies the kernel's output and a witness buffer published
last by the producer chain. The consumer waits only on the k-pool grid, so it
sees the witness only if that grid ordered itself after the producers, which
the early-return cases exercise with grids whose blocks all exit at once.
"""

from std.sys import get_defined_int
from std.sys.intrinsics import llvm_intrinsic
from std.random import rand
from std.testing import assert_equal, assert_true

from layout import TileTensor, row_major
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from nn.attention.gpu.mla_index_kpool import (
    KPOOL_EXPAND_BLOCK_SIZE,
    kpool_compress_kernel,
    kpool_expand_topk_kernel,
    kpool_ring_close_kernel,
    kpool_seed_tail_kernel,
)


def slow_copy[
    dtype: DType
](
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    length: Int32,
    delay_us: Int32,
):
    """Releases its dependent on entry, sleeps, then copies `src` to `dst`."""
    launch_dependent_grids()
    wait_on_dependent_grids()
    # nanosleep takes nanoseconds and caps at about 1 ms per call.
    var spins = Int(delay_us)
    while spins > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
        spins -= 1000
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(length), stride):
        dst[i] = src[i]


def _publish[
    dtype: DType
](
    ctx: DeviceContext,
    dev: DeviceBuffer[dtype],
    stage: DeviceBuffer[dtype],
    n: Int,
    delay_us: Int,
) raises:
    ctx.enqueue_function[slow_copy[dtype]](
        stage,
        dev,
        Int32(n),
        Int32(delay_us),
        grid_dim=4,
        block_dim=64,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


def _assert_same[
    dtype: DType
](a: HostBuffer[dtype], b: HostBuffer[dtype], n: Int, what: String) raises:
    for i in range(n):
        if a[i] != b[i]:
            assert_equal(a[i], b[i], String(what, " differs at element ", i))


def copy_after_wait[
    dtype: DType
](
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    length: Int32,
):
    """Waits on the grid before it, then copies `src` to `dst`."""
    wait_on_dependent_grids()
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(length), stride):
        dst[i] = src[i]


def _consume[
    dtype: DType
](
    ctx: DeviceContext,
    src: DeviceBuffer[dtype],
    dst: DeviceBuffer[dtype],
    n: Int,
) raises:
    ctx.enqueue_function[copy_after_wait[dtype]](
        src,
        dst,
        Int32(n),
        grid_dim=4,
        block_dim=64,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


comptime WITNESS_N = 512


struct Witness(Movable):
    """A buffer the producer chain publishes last, read by the consumer."""

    var host: HostBuffer[DType.uint32]
    var stage: DeviceBuffer[DType.uint32]
    var dev: DeviceBuffer[DType.uint32]
    var out: DeviceBuffer[DType.uint32]

    def __init__(out self, ctx: DeviceContext) raises:
        self.host = ctx.enqueue_create_host_buffer[.uint32](WITNESS_N)
        self.stage = ctx.enqueue_create_buffer[.uint32](WITNESS_N)
        self.dev = ctx.enqueue_create_buffer[.uint32](WITNESS_N)
        self.out = ctx.enqueue_create_buffer[.uint32](WITNESS_N)
        ctx.synchronize()
        for i in range(WITNESS_N):
            self.host[i] = UInt32(i + 1)
        ctx.enqueue_copy(self.stage, self.host)
        ctx.synchronize()

    def poison(self, ctx: DeviceContext) raises:
        ctx.enqueue_memset(self.dev, 0)
        ctx.enqueue_memset(self.out, 0)

    def publish(self, ctx: DeviceContext, delay_us: Int) raises:
        _publish(ctx, self.dev, self.stage, WITNESS_N, delay_us)

    def consume(self, ctx: DeviceContext) raises:
        _consume(ctx, self.dev, self.out, WITNESS_N)

    def check(self, ctx: DeviceContext, what: String) raises:
        var got = ctx.enqueue_create_host_buffer[.uint32](WITNESS_N)
        ctx.synchronize()
        ctx.enqueue_copy(got, self.out)
        ctx.synchronize()
        _assert_same(got, self.host, WITNESS_N, what)


struct Case[head_dim: Int, kpool: Int](Movable):
    """Ragged k-pool inputs, each held on host, on device, and as a staged copy.

    The staged copy is what a producer grid publishes into the device buffer.
    """

    var batch: Int
    var tokens: Int
    var total_pools: Int
    var k_h: HostBuffer[DType.bfloat16]
    var gate_h: HostBuffer[DType.bfloat16]
    var ape_h: HostBuffer[DType.float32]
    var tail_h: HostBuffer[DType.bfloat16]
    var iro_h: HostBuffer[DType.uint32]
    var pro_h: HostBuffer[DType.uint32]
    var clen_h: HostBuffer[DType.uint32]
    var slot_h: HostBuffer[DType.uint32]
    var k_d: DeviceBuffer[DType.bfloat16]
    var gate_d: DeviceBuffer[DType.bfloat16]
    var ape_d: DeviceBuffer[DType.float32]
    var tail_d: DeviceBuffer[DType.bfloat16]
    var iro_d: DeviceBuffer[DType.uint32]
    var pro_d: DeviceBuffer[DType.uint32]
    var clen_d: DeviceBuffer[DType.uint32]
    var slot_d: DeviceBuffer[DType.uint32]
    var k_s: DeviceBuffer[DType.bfloat16]
    var gate_s: DeviceBuffer[DType.bfloat16]
    var ape_s: DeviceBuffer[DType.float32]
    var tail_s: DeviceBuffer[DType.bfloat16]
    var iro_s: DeviceBuffer[DType.uint32]
    var pro_s: DeviceBuffer[DType.uint32]
    var clen_s: DeviceBuffer[DType.uint32]
    var slot_s: DeviceBuffer[DType.uint32]

    def __init__(
        out self,
        seq_lens: List[Int],
        cache_lens: List[Int],
        ctx: DeviceContext,
    ) raises:
        self.batch = len(seq_lens)
        self.tokens = 0
        self.total_pools = 0
        for i in range(self.batch):
            self.tokens += seq_lens[i]
        var ring_n = self.batch * 2 * Self.kpool * Self.head_dim

        self.k_h = ctx.enqueue_create_host_buffer[.bfloat16](
            self.tokens * Self.head_dim
        )
        self.gate_h = ctx.enqueue_create_host_buffer[.bfloat16](
            self.tokens * Self.head_dim
        )
        self.ape_h = ctx.enqueue_create_host_buffer[.float32](
            Self.kpool * Self.head_dim
        )
        self.tail_h = ctx.enqueue_create_host_buffer[.bfloat16](ring_n)
        self.iro_h = ctx.enqueue_create_host_buffer[.uint32](self.batch + 1)
        self.pro_h = ctx.enqueue_create_host_buffer[.uint32](self.batch + 1)
        self.clen_h = ctx.enqueue_create_host_buffer[.uint32](self.batch)
        self.slot_h = ctx.enqueue_create_host_buffer[.uint32](self.batch)
        self.k_d = ctx.enqueue_create_buffer[.bfloat16](
            self.tokens * Self.head_dim
        )
        self.gate_d = ctx.enqueue_create_buffer[.bfloat16](
            self.tokens * Self.head_dim
        )
        self.ape_d = ctx.enqueue_create_buffer[.float32](
            Self.kpool * Self.head_dim
        )
        self.tail_d = ctx.enqueue_create_buffer[.bfloat16](ring_n)
        self.iro_d = ctx.enqueue_create_buffer[.uint32](self.batch + 1)
        self.pro_d = ctx.enqueue_create_buffer[.uint32](self.batch + 1)
        self.clen_d = ctx.enqueue_create_buffer[.uint32](self.batch)
        self.slot_d = ctx.enqueue_create_buffer[.uint32](self.batch)
        self.k_s = ctx.enqueue_create_buffer[.bfloat16](
            self.tokens * Self.head_dim
        )
        self.gate_s = ctx.enqueue_create_buffer[.bfloat16](
            self.tokens * Self.head_dim
        )
        self.ape_s = ctx.enqueue_create_buffer[.float32](
            Self.kpool * Self.head_dim
        )
        self.tail_s = ctx.enqueue_create_buffer[.bfloat16](ring_n)
        self.iro_s = ctx.enqueue_create_buffer[.uint32](self.batch + 1)
        self.pro_s = ctx.enqueue_create_buffer[.uint32](self.batch + 1)
        self.clen_s = ctx.enqueue_create_buffer[.uint32](self.batch)
        self.slot_s = ctx.enqueue_create_buffer[.uint32](self.batch)
        ctx.synchronize()

        rand(self.k_h.unsafe_ptr(), self.tokens * Self.head_dim)
        rand(self.gate_h.unsafe_ptr(), self.tokens * Self.head_dim)
        rand(self.ape_h.unsafe_ptr(), Self.kpool * Self.head_dim)
        rand(self.tail_h.unsafe_ptr(), ring_n)
        # Spread the scores, or the member weights come out nearly equal and
        # a wrong member mapping would not show.
        for i in range(self.tokens * Self.head_dim):
            self.gate_h[i] = (self.gate_h[i] - 0.5) * 8.0
        for i in range(Self.kpool * Self.head_dim):
            self.ape_h[i] = (self.ape_h[i] - 0.5) * 4.0

        var tok_off = 0
        var pool_off = 0
        for b in range(self.batch):
            self.iro_h[b] = UInt32(tok_off)
            self.pro_h[b] = UInt32(pool_off)
            self.clen_h[b] = UInt32(cache_lens[b])
            # Reversed, so a kernel that indexed the ring by row is caught.
            self.slot_h[b] = UInt32(self.batch - 1 - b)
            var align = (Self.kpool - cache_lens[b] % Self.kpool) % Self.kpool
            var n = seq_lens[b] - align
            tok_off += seq_lens[b]
            pool_off += n // Self.kpool if n > 0 else 0
        self.iro_h[self.batch] = UInt32(tok_off)
        self.pro_h[self.batch] = UInt32(pool_off)
        self.total_pools = pool_off

        ctx.enqueue_copy(self.k_s, self.k_h)
        ctx.enqueue_copy(self.gate_s, self.gate_h)
        ctx.enqueue_copy(self.ape_s, self.ape_h)
        ctx.enqueue_copy(self.tail_s, self.tail_h)
        ctx.enqueue_copy(self.iro_s, self.iro_h)
        ctx.enqueue_copy(self.pro_s, self.pro_h)
        ctx.enqueue_copy(self.clen_s, self.clen_h)
        ctx.enqueue_copy(self.slot_s, self.slot_h)
        ctx.synchronize()

    def load_synced(self, ctx: DeviceContext) raises:
        """Fills every input buffer and waits, so nothing is still in flight."""
        ctx.enqueue_copy(self.k_d, self.k_s)
        ctx.enqueue_copy(self.gate_d, self.gate_s)
        ctx.enqueue_copy(self.ape_d, self.ape_s)
        ctx.enqueue_copy(self.tail_d, self.tail_s)
        ctx.enqueue_copy(self.iro_d, self.iro_s)
        ctx.enqueue_copy(self.pro_d, self.pro_s)
        ctx.enqueue_copy(self.clen_d, self.clen_s)
        ctx.enqueue_copy(self.slot_d, self.slot_s)
        ctx.synchronize()

    def poison(self, ctx: DeviceContext) raises:
        """Zeroes every input buffer; a premature read sees these zeros."""
        ctx.enqueue_memset(self.k_d, 0)
        ctx.enqueue_memset(self.gate_d, 0)
        ctx.enqueue_memset(self.ape_d, 0)
        ctx.enqueue_memset(self.tail_d, 0)
        ctx.enqueue_memset(self.iro_d, 0)
        ctx.enqueue_memset(self.pro_d, 0)
        ctx.enqueue_memset(self.clen_d, 0)
        ctx.enqueue_memset(self.slot_d, 0)

    def publish(self, ctx: DeviceContext, delay_us: Int) raises:
        """Queues one slow producer grid per input, back to back."""
        var k_n = self.tokens * Self.head_dim
        _publish(ctx, self.k_d, self.k_s, k_n, delay_us)
        _publish(ctx, self.gate_d, self.gate_s, k_n, delay_us)
        _publish(
            ctx, self.ape_d, self.ape_s, Self.kpool * Self.head_dim, delay_us
        )
        _publish(
            ctx,
            self.tail_d,
            self.tail_s,
            self.batch * 2 * Self.kpool * Self.head_dim,
            delay_us,
        )
        _publish(ctx, self.iro_d, self.iro_s, self.batch + 1, delay_us)
        _publish(ctx, self.pro_d, self.pro_s, self.batch + 1, delay_us)
        _publish(ctx, self.clen_d, self.clen_s, self.batch, delay_us)
        _publish(ctx, self.slot_d, self.slot_s, self.batch, delay_us)


def _launch_compress[
    head_dim: Int, kpool: Int, pdl_level: PDLLevel
](
    ctx: DeviceContext,
    c: Case[head_dim, kpool],
    mut out_d: DeviceBuffer[.bfloat16],
) raises:
    var out_t = TileTensor(out_d, row_major(c.total_pools, head_dim))
    var k_t = TileTensor(c.k_d, row_major(c.tokens, head_dim))
    var gate_t = TileTensor(c.gate_d, row_major(c.tokens, head_dim))
    var ape_t = TileTensor(c.ape_d, row_major(kpool, head_dim))
    var iro_t = TileTensor(c.iro_d, row_major(c.batch + 1))
    var pro_t = TileTensor(c.pro_d, row_major(c.batch + 1))
    var clen_t = TileTensor(c.clen_d, row_major(c.batch))
    comptime kernel = kpool_compress_kernel[
        .bfloat16,
        type_of(k_t.as_imm()).LayoutType,
        ImmOrigin(k_t.origin),
        type_of(gate_t.as_imm()).LayoutType,
        ImmOrigin(gate_t.origin),
        type_of(ape_t.as_imm()).LayoutType,
        ImmOrigin(ape_t.origin),
        type_of(iro_t.as_imm()).LayoutType,
        ImmOrigin(iro_t.origin),
        type_of(pro_t.as_imm()).LayoutType,
        ImmOrigin(pro_t.origin),
        type_of(clen_t.as_imm()).LayoutType,
        out_t.LayoutType,
        out_t.origin,
        type_of(k_t.as_imm()).Engine,
        type_of(gate_t.as_imm()).Engine,
        type_of(ape_t.as_imm()).Engine,
        type_of(iro_t.as_imm()).Engine,
        type_of(pro_t.as_imm()).Engine,
        type_of(clen_t.as_imm()).Engine,
        out_t.Engine,
        head_dim,
        kpool,
        pdl_level,
    ]
    ctx.enqueue_function[kernel](
        out_t,
        k_t.as_imm(),
        gate_t.as_imm(),
        ape_t.as_imm(),
        iro_t.as_imm(),
        pro_t.as_imm(),
        clen_t.as_imm(),
        # No pool at all leaves one block, which must exit at once.
        grid_dim=(max(c.total_pools, 1), 1, 1),
        block_dim=(head_dim, 1, 1),
        attributes=pdl_launch_attributes(pdl_level),
    )


def _launch_seed_tail[
    head_dim: Int, kpool: Int, pdl_level: PDLLevel
](ctx: DeviceContext, mut c: Case[head_dim, kpool], live: Bool = True) raises:
    var tail_t = TileTensor(c.tail_d, row_major(c.batch, 2, kpool, head_dim))
    var k_t = TileTensor(c.k_d, row_major(c.tokens, head_dim))
    var gate_t = TileTensor(c.gate_d, row_major(c.tokens, head_dim))
    var iro_t = TileTensor(c.iro_d, row_major(c.batch + 1))
    var clen_t = TileTensor(c.clen_d, row_major(c.batch))
    var slot_t = TileTensor(c.slot_d, row_major(c.batch))
    comptime kernel = kpool_seed_tail_kernel[
        .bfloat16,
        tail_t.LayoutType,
        tail_t.origin,
        type_of(k_t.as_imm()).LayoutType,
        ImmOrigin(k_t.origin),
        type_of(gate_t.as_imm()).LayoutType,
        ImmOrigin(gate_t.origin),
        type_of(iro_t.as_imm()).LayoutType,
        ImmOrigin(iro_t.origin),
        type_of(clen_t.as_imm()).LayoutType,
        type_of(slot_t.as_imm()).LayoutType,
        ImmOrigin(slot_t.origin),
        tail_t.Engine,
        type_of(k_t.as_imm()).Engine,
        type_of(gate_t.as_imm()).Engine,
        type_of(iro_t.as_imm()).Engine,
        type_of(clen_t.as_imm()).Engine,
        type_of(slot_t.as_imm()).Engine,
        head_dim,
        kpool,
        pdl_level,
    ]
    ctx.enqueue_function[kernel](
        tail_t,
        k_t.as_imm(),
        gate_t.as_imm(),
        iro_t.as_imm(),
        clen_t.as_imm(),
        slot_t.as_imm(),
        # Zero requests: every block exits at its first check.
        Int32(c.batch) if live else Int32(0),
        grid_dim=(c.batch, 1, 1),
        block_dim=(head_dim, 1, 1),
        attributes=pdl_launch_attributes(pdl_level),
    )


def _launch_ring_close[
    head_dim: Int, kpool: Int, pdl_level: PDLLevel
](
    ctx: DeviceContext,
    mut c: Case[head_dim, kpool],
    mut out_d: DeviceBuffer[.bfloat16],
    mut closed_d: DeviceBuffer[.int32],
    live: Bool = True,
) raises:
    var out_t = TileTensor(out_d, row_major(c.batch, head_dim))
    var closed_t = TileTensor(closed_d, row_major(c.batch))
    var tail_t = TileTensor(c.tail_d, row_major(c.batch, 2, kpool, head_dim))
    var k_t = TileTensor(c.k_d, row_major(c.tokens, head_dim))
    var gate_t = TileTensor(c.gate_d, row_major(c.tokens, head_dim))
    var ape_t = TileTensor(c.ape_d, row_major(kpool, head_dim))
    var iro_t = TileTensor(c.iro_d, row_major(c.batch + 1))
    var clen_t = TileTensor(c.clen_d, row_major(c.batch))
    var slot_t = TileTensor(c.slot_d, row_major(c.batch))
    comptime kernel = kpool_ring_close_kernel[
        .bfloat16,
        out_t.LayoutType,
        out_t.origin,
        closed_t.LayoutType,
        closed_t.origin,
        type_of(tail_t.as_imm()).LayoutType,
        ImmOrigin(tail_t.origin),
        type_of(k_t.as_imm()).LayoutType,
        ImmOrigin(k_t.origin),
        type_of(gate_t.as_imm()).LayoutType,
        ImmOrigin(gate_t.origin),
        type_of(ape_t.as_imm()).LayoutType,
        ImmOrigin(ape_t.origin),
        type_of(iro_t.as_imm()).LayoutType,
        ImmOrigin(iro_t.origin),
        type_of(clen_t.as_imm()).LayoutType,
        type_of(slot_t.as_imm()).LayoutType,
        ImmOrigin(slot_t.origin),
        out_t.Engine,
        closed_t.Engine,
        type_of(tail_t.as_imm()).Engine,
        type_of(k_t.as_imm()).Engine,
        type_of(gate_t.as_imm()).Engine,
        type_of(ape_t.as_imm()).Engine,
        type_of(iro_t.as_imm()).Engine,
        type_of(clen_t.as_imm()).Engine,
        type_of(slot_t.as_imm()).Engine,
        head_dim,
        kpool,
        pdl_level,
    ]
    ctx.enqueue_function[kernel](
        out_t,
        closed_t,
        tail_t.as_imm(),
        k_t.as_imm(),
        gate_t.as_imm(),
        ape_t.as_imm(),
        iro_t.as_imm(),
        clen_t.as_imm(),
        slot_t.as_imm(),
        Int32(c.batch) if live else Int32(0),
        grid_dim=(c.batch, 1, 1),
        block_dim=(head_dim, 1, 1),
        attributes=pdl_launch_attributes(pdl_level),
    )


def _launch_expand[
    kpool: Int, pool_topk: Int, always_select_tail: Bool, pdl_level: PDLLevel
](
    ctx: DeviceContext,
    mut out_d: DeviceBuffer[.int32],
    pool_d: DeviceBuffer[.int32],
    iro_d: DeviceBuffer[.uint32],
    clen_d: DeviceBuffer[.uint32],
    tokens: Int,
    batch: Int,
    live: Bool = True,
) raises:
    comptime out_width = pool_topk * kpool + (
        (kpool - 1) if always_select_tail else 0
    )
    var out_t = TileTensor(out_d, row_major(tokens, out_width))
    var pool_t = TileTensor(pool_d, row_major(tokens, pool_topk))
    var iro_t = TileTensor(iro_d, row_major(batch + 1))
    var clen_t = TileTensor(clen_d, row_major(batch))
    comptime kernel = kpool_expand_topk_kernel[
        out_t.LayoutType,
        out_t.origin,
        type_of(pool_t.as_imm()).LayoutType,
        ImmOrigin(pool_t.origin),
        type_of(iro_t.as_imm()).LayoutType,
        ImmOrigin(iro_t.origin),
        type_of(clen_t.as_imm()).LayoutType,
        out_t.Engine,
        type_of(pool_t.as_imm()).Engine,
        type_of(iro_t.as_imm()).Engine,
        type_of(clen_t.as_imm()).Engine,
        kpool,
        pool_topk,
        always_select_tail,
        pdl_level,
    ]
    ctx.enqueue_function[kernel](
        out_t,
        pool_t.as_imm(),
        iro_t.as_imm(),
        clen_t.as_imm(),
        Int32(tokens) if live else Int32(0),
        grid_dim=(tokens, 1, 1),
        block_dim=(KPOOL_EXPAND_BLOCK_SIZE, 1, 1),
        attributes=pdl_launch_attributes(pdl_level),
    )


def test_compress[
    head_dim: Int, kpool: Int, pdl_level: PDLLevel
](
    seq_lens: List[Int],
    cache_lens: List[Int],
    iters: Int,
    delay_us: Int,
    ctx: DeviceContext,
) raises:
    var c = Case[head_dim, kpool](seq_lens, cache_lens, ctx)
    assert_true(c.total_pools > 0, "the case closes no pool")
    var n = c.total_pools * head_dim
    var out_d = ctx.enqueue_create_buffer[.bfloat16](n)
    var want_h = ctx.enqueue_create_host_buffer[.bfloat16](n)
    var got_h = ctx.enqueue_create_host_buffer[.bfloat16](n)
    var copy_d = ctx.enqueue_create_buffer[.bfloat16](n)
    var w = Witness(ctx)

    c.load_synced(ctx)
    ctx.enqueue_memset(out_d, 0)
    _launch_compress[head_dim, kpool, PDLLevel.OFF](ctx, c, out_d)
    ctx.synchronize()
    ctx.enqueue_copy(want_h, out_d)
    ctx.synchronize()

    for _ in range(iters):
        ctx.enqueue_memset(out_d, 0)
        ctx.enqueue_memset(copy_d, 0)
        c.poison(ctx)
        w.poison(ctx)
        c.publish(ctx, delay_us)
        w.publish(ctx, delay_us)
        _launch_compress[head_dim, kpool, pdl_level](ctx, c, out_d)
        _consume(ctx, out_d, copy_d, n)
        w.consume(ctx)
        ctx.synchronize()
        ctx.enqueue_copy(got_h, copy_d)
        ctx.synchronize()
        _assert_same(got_h, want_h, n, "compress")
        w.check(ctx, "compress witness")
    print("  compress ok")


def test_seed_tail[
    head_dim: Int, kpool: Int, pdl_level: PDLLevel
](
    seq_lens: List[Int],
    cache_lens: List[Int],
    iters: Int,
    delay_us: Int,
    ctx: DeviceContext,
) raises:
    var c = Case[head_dim, kpool](seq_lens, cache_lens, ctx)
    var n = c.batch * 2 * kpool * head_dim
    var want_h = ctx.enqueue_create_host_buffer[.bfloat16](n)
    var got_h = ctx.enqueue_create_host_buffer[.bfloat16](n)
    var copy_d = ctx.enqueue_create_buffer[.bfloat16](n)
    var w = Witness(ctx)

    c.load_synced(ctx)
    _launch_seed_tail[head_dim, kpool, PDLLevel.OFF](ctx, c)
    ctx.synchronize()
    ctx.enqueue_copy(want_h, c.tail_d)
    ctx.synchronize()
    # The ring must actually change, or the comparison proves nothing.
    var changed = False
    for i in range(n):
        if want_h[i] != c.tail_h[i]:
            changed = True
    assert_true(changed, "the case seeds nothing")

    for _ in range(iters):
        ctx.enqueue_memset(copy_d, 0)
        c.poison(ctx)
        w.poison(ctx)
        c.publish(ctx, delay_us)
        w.publish(ctx, delay_us)
        _launch_seed_tail[head_dim, kpool, pdl_level](ctx, c)
        _consume(ctx, c.tail_d, copy_d, n)
        w.consume(ctx)
        ctx.synchronize()
        ctx.enqueue_copy(got_h, copy_d)
        ctx.synchronize()
        _assert_same(got_h, want_h, n, "seed_tail")
        w.check(ctx, "seed_tail witness")
    print("  seed_tail ok")


def test_ring_close[
    head_dim: Int, kpool: Int, pdl_level: PDLLevel
](
    seq_lens: List[Int],
    cache_lens: List[Int],
    iters: Int,
    delay_us: Int,
    ctx: DeviceContext,
) raises:
    var c = Case[head_dim, kpool](seq_lens, cache_lens, ctx)
    var n = c.batch * head_dim
    var out_d = ctx.enqueue_create_buffer[.bfloat16](n)
    var closed_d = ctx.enqueue_create_buffer[.int32](c.batch)
    var want_h = ctx.enqueue_create_host_buffer[.bfloat16](n)
    var got_h = ctx.enqueue_create_host_buffer[.bfloat16](n)
    var want_closed_h = ctx.enqueue_create_host_buffer[.int32](c.batch)
    var got_closed_h = ctx.enqueue_create_host_buffer[.int32](c.batch)
    var out_copy_d = ctx.enqueue_create_buffer[.bfloat16](n)
    var closed_copy_d = ctx.enqueue_create_buffer[.int32](c.batch)
    var w = Witness(ctx)

    c.load_synced(ctx)
    ctx.enqueue_memset(out_d, 0)
    ctx.enqueue_memset(closed_d, 0)
    _launch_ring_close[head_dim, kpool, PDLLevel.OFF](ctx, c, out_d, closed_d)
    ctx.synchronize()
    ctx.enqueue_copy(want_h, out_d)
    ctx.enqueue_copy(want_closed_h, closed_d)
    ctx.synchronize()
    var closed_any = False
    for r in range(c.batch):
        if want_closed_h[r] >= 0:
            closed_any = True
    assert_true(closed_any, "the case closes no pool")

    for _ in range(iters):
        ctx.enqueue_memset(out_d, 0)
        ctx.enqueue_memset(closed_d, 0)
        ctx.enqueue_memset(out_copy_d, 0)
        ctx.enqueue_memset(closed_copy_d, 0)
        c.poison(ctx)
        w.poison(ctx)
        c.publish(ctx, delay_us)
        w.publish(ctx, delay_us)
        _launch_ring_close[head_dim, kpool, pdl_level](ctx, c, out_d, closed_d)
        _consume(ctx, out_d, out_copy_d, n)
        _consume(ctx, closed_d, closed_copy_d, c.batch)
        w.consume(ctx)
        ctx.synchronize()
        ctx.enqueue_copy(got_h, out_copy_d)
        ctx.enqueue_copy(got_closed_h, closed_copy_d)
        ctx.synchronize()
        _assert_same(got_h, want_h, n, "ring_close pooled")
        _assert_same(got_closed_h, want_closed_h, c.batch, "ring_close id")
        w.check(ctx, "ring_close witness")
    print("  ring_close ok")


def test_expand_topk[
    kpool: Int, pool_topk: Int, always_select_tail: Bool, pdl_level: PDLLevel
](
    seq_lens: List[Int],
    cache_lens: List[Int],
    iters: Int,
    delay_us: Int,
    ctx: DeviceContext,
) raises:
    comptime out_width = pool_topk * kpool + (
        (kpool - 1) if always_select_tail else 0
    )
    var batch = len(seq_lens)
    var tokens = 0
    for i in range(batch):
        tokens += seq_lens[i]

    var pool_h = ctx.enqueue_create_host_buffer[.int32](tokens * pool_topk)
    var iro_h = ctx.enqueue_create_host_buffer[.uint32](batch + 1)
    var clen_h = ctx.enqueue_create_host_buffer[.uint32](batch)
    var want_h = ctx.enqueue_create_host_buffer[.int32](tokens * out_width)
    var got_h = ctx.enqueue_create_host_buffer[.int32](tokens * out_width)
    ctx.synchronize()
    var off = 0
    for b in range(batch):
        iro_h[b] = UInt32(off)
        clen_h[b] = UInt32(cache_lens[b])
        off += seq_lens[b]
    iro_h[batch] = UInt32(off)
    for t in range(tokens):
        for j in range(pool_topk):
            if j == pool_topk - 1 or (t % 2 == 1 and j % 5 == 2):
                pool_h[t * pool_topk + j] = Int32(-1)
            else:
                pool_h[t * pool_topk + j] = Int32((t * 7 + j * 3) % 32)

    var pool_d = ctx.enqueue_create_buffer[.int32](tokens * pool_topk)
    var iro_d = ctx.enqueue_create_buffer[.uint32](batch + 1)
    var clen_d = ctx.enqueue_create_buffer[.uint32](batch)
    var pool_s = ctx.enqueue_create_buffer[.int32](tokens * pool_topk)
    var iro_s = ctx.enqueue_create_buffer[.uint32](batch + 1)
    var clen_s = ctx.enqueue_create_buffer[.uint32](batch)
    var out_d = ctx.enqueue_create_buffer[.int32](tokens * out_width)
    var copy_d = ctx.enqueue_create_buffer[.int32](tokens * out_width)
    var w = Witness(ctx)
    ctx.enqueue_copy(pool_s, pool_h)
    ctx.enqueue_copy(iro_s, iro_h)
    ctx.enqueue_copy(clen_s, clen_h)
    ctx.synchronize()

    ctx.enqueue_copy(pool_d, pool_s)
    ctx.enqueue_copy(iro_d, iro_s)
    ctx.enqueue_copy(clen_d, clen_s)
    ctx.enqueue_memset(out_d, 0)
    _launch_expand[kpool, pool_topk, always_select_tail, PDLLevel.OFF](
        ctx, out_d, pool_d, iro_d, clen_d, tokens, batch
    )
    ctx.synchronize()
    ctx.enqueue_copy(want_h, out_d)
    ctx.synchronize()

    for _ in range(iters):
        ctx.enqueue_memset(out_d, 0)
        ctx.enqueue_memset(pool_d, 0)
        ctx.enqueue_memset(iro_d, 0)
        ctx.enqueue_memset(clen_d, 0)
        ctx.enqueue_memset(copy_d, 0)
        w.poison(ctx)
        _publish(ctx, pool_d, pool_s, tokens * pool_topk, delay_us)
        _publish(ctx, iro_d, iro_s, batch + 1, delay_us)
        _publish(ctx, clen_d, clen_s, batch, delay_us)
        w.publish(ctx, delay_us)
        _launch_expand[kpool, pool_topk, always_select_tail, pdl_level](
            ctx, out_d, pool_d, iro_d, clen_d, tokens, batch
        )
        _consume(ctx, out_d, copy_d, tokens * out_width)
        w.consume(ctx)
        ctx.synchronize()
        ctx.enqueue_copy(got_h, copy_d)
        ctx.synchronize()
        _assert_same(got_h, want_h, tokens * out_width, "expand_topk")
        w.check(ctx, "expand_topk witness")
    print("  expand_topk ok, always_select_tail =", always_select_tail)


def test_early_return_chain[
    pdl_level: PDLLevel
](iters: Int, delay_us: Int, ctx: DeviceContext) raises:
    """Runs each kernel with every block exiting at its first check.

    The consumer behind the kernel waits only on that grid, so it reads the
    witness published last by the producers only if the kernel ordered itself
    after them before exiting.
    """
    comptime head_dim = 128
    comptime kpool = 4
    comptime expand_topk = 16
    comptime expand_width = expand_topk * kpool + kpool - 1
    var w = Witness(ctx)
    # Neither request completes a pool, so `compress` has no work.
    var short_lens: List[Int] = [1, 2]
    var short_cache: List[Int] = [0, 0]
    var c = Case[head_dim, kpool](short_lens, short_cache, ctx)
    assert_true(c.total_pools == 0, "the case must close no pool")
    var pooled_d = ctx.enqueue_create_buffer[.bfloat16](c.batch * head_dim)
    var closed_d = ctx.enqueue_create_buffer[.int32](c.batch)
    var expand_d = ctx.enqueue_create_buffer[.int32](c.batch * expand_width)
    var pool_d = ctx.enqueue_create_buffer[.int32](c.batch * expand_topk)
    ctx.synchronize()

    for kernel in range(4):
        for _ in range(iters):
            c.poison(ctx)
            w.poison(ctx)
            c.publish(ctx, delay_us)
            w.publish(ctx, delay_us)
            if kernel == 0:
                _launch_compress[head_dim, kpool, pdl_level](ctx, c, pooled_d)
            elif kernel == 1:
                _launch_seed_tail[head_dim, kpool, pdl_level](ctx, c, False)
            elif kernel == 2:
                _launch_ring_close[head_dim, kpool, pdl_level](
                    ctx, c, pooled_d, closed_d, False
                )
            else:
                _launch_expand[kpool, expand_topk, True, pdl_level](
                    ctx,
                    expand_d,
                    pool_d,
                    c.iro_d,
                    c.clen_d,
                    c.batch,
                    c.batch,
                    False,
                )
            w.consume(ctx)
            ctx.synchronize()
            w.check(ctx, String("early-return kernel ", kernel, " witness"))
    print("  early-return chain ok")


def run_level[
    pdl_level: PDLLevel
](iters: Int, delay_us: Int, ctx: DeviceContext) raises:
    # Decode-like and ragged prefill-like requests mixed, with cached prefixes
    # ending at every phase of the pool, as a production batch has.
    var seq_lens: List[Int] = [9, 1, 5, 13, 1, 4, 2, 17]
    var cache_lens: List[Int] = [0, 3, 6, 2, 11, 0, 1, 5]
    test_compress[128, 4, pdl_level](seq_lens, cache_lens, iters, delay_us, ctx)
    test_seed_tail[128, 4, pdl_level](
        seq_lens, cache_lens, iters, delay_us, ctx
    )
    test_ring_close[128, 4, pdl_level](
        seq_lens, cache_lens, iters, delay_us, ctx
    )
    test_expand_topk[4, 512, True, pdl_level](
        seq_lens, cache_lens, iters, delay_us, ctx
    )
    test_expand_topk[4, 512, False, pdl_level](
        seq_lens, cache_lens, iters, delay_us, ctx
    )
    test_early_return_chain[pdl_level](iters, delay_us, ctx)
    # Prefill-sized grids: thousands of blocks, so several waves of
    # dependents are resident or queued behind the producers.
    var prefill_lens: List[Int] = [3001, 2999]
    var prefill_cache: List[Int] = [0, 5]
    test_compress[128, 4, pdl_level](
        prefill_lens, prefill_cache, iters, delay_us, ctx
    )
    test_expand_topk[4, 512, True, pdl_level](
        prefill_lens, prefill_cache, iters, delay_us, ctx
    )
    # An odd pool size, and a batch of single decode tokens.
    var decode_lens: List[Int] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    var decode_cache: List[Int] = [3, 7, 6, 10, 11, 0, 1, 5, 2, 8, 9, 4]
    test_ring_close[128, 8, pdl_level](
        decode_lens, decode_cache, iters, delay_us, ctx
    )
    test_seed_tail[128, 8, pdl_level](
        decode_lens, decode_cache, iters, delay_us, ctx
    )


def main() raises:
    var iters = get_defined_int["ITERS", 10]()
    # Spent in every producer after it releases its dependent: the k-pool
    # grid is resident and running its prologue long before the input lands.
    var delay_us = get_defined_int["DELAY_US", 2000]()
    with DeviceContext() as ctx:
        run_level[PDLLevel.ON](iters, delay_us, ctx)
    print("All k-pool PDL race tests passed!")
