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
"""Tests for the DSA indexer's k-pool compression kernel."""

from std.math import exp
from std.random import rand
from std.testing import assert_almost_equal, assert_equal, assert_true

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext

from nn.attention.gpu.mla_index_kpool import kpool_compress_kernel


def test_kpool_compress[
    head_dim: Int, kpool: Int
](
    seq_lens: List[Int],
    ctx: DeviceContext,
    cache_lens: List[Int] = List[Int](),
    extra_blocks: Int = 0,
) raises:
    """Compress `seq_lens` requests' keys and check every pooled channel.

    Parameters:
        head_dim: Channels per key.
        kpool: Tokens per pool.

    Args:
        seq_lens: Tokens per request in this call.
        ctx: Device context.
        cache_lens: Cached-prefix length per request; empty means all zero.
        extra_blocks: Surplus blocks to launch beyond the real pool count.
    """
    var batch_size = len(seq_lens)
    var clens = List[Int]()
    for i in range(batch_size):
        clens.append(cache_lens[i] if len(cache_lens) > 0 else 0)
    var total_tokens = 0
    for i in range(batch_size):
        total_tokens += seq_lens[i]

    # A cached prefix ending mid-pool costs the call the pool it opens in.
    var aligns = List[Int]()
    var pools_per = List[Int]()
    var total_pools = 0
    for i in range(batch_size):
        var align = (kpool - clens[i] % kpool) % kpool
        var n = (seq_lens[i] - align) // kpool if seq_lens[i] > align else 0
        aligns.append(align)
        pools_per.append(n)
        total_pools += n

    print(
        "kpool_compress: head_dim=",
        head_dim,
        " kpool=",
        kpool,
        " tokens=",
        total_tokens,
        " pools=",
        total_pools,
    )
    assert_true(total_pools > 0, "test shape selects no complete pool")

    var k_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * head_dim
    )
    var gate_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * head_dim
    )
    var ape_host = ctx.enqueue_create_host_buffer[.float32](kpool * head_dim)
    var iro_host = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var pro_host = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var clen_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](
        (total_pools + extra_blocks) * head_dim
    )
    ctx.synchronize()

    rand(k_host.unsafe_ptr(), total_tokens * head_dim)
    rand(gate_host.unsafe_ptr(), total_tokens * head_dim)
    rand(ape_host.unsafe_ptr(), kpool * head_dim)

    # Spread the gate scores, or the member weights come out nearly equal and
    # a wrong member mapping would not show.
    for i in range(total_tokens * head_dim):
        gate_host[i] = (gate_host[i] - 0.5) * 8.0
    for i in range(kpool * head_dim):
        ape_host[i] = (ape_host[i] - 0.5) * 4.0

    var tok_off = 0
    var pool_off = 0
    for b in range(batch_size):
        iro_host[b] = UInt32(tok_off)
        pro_host[b] = UInt32(pool_off)
        tok_off += seq_lens[b]
        pool_off += pools_per[b]
    iro_host[batch_size] = UInt32(tok_off)
    pro_host[batch_size] = UInt32(pool_off)

    for b in range(batch_size):
        clen_host[b] = UInt32(clens[b])
    var clen_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(clen_dev, clen_host)
    var k_dev = ctx.enqueue_create_buffer[.bfloat16](total_tokens * head_dim)
    var gate_dev = ctx.enqueue_create_buffer[.bfloat16](total_tokens * head_dim)
    var ape_dev = ctx.enqueue_create_buffer[.float32](kpool * head_dim)
    var iro_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var pro_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var out_dev = ctx.enqueue_create_buffer[.bfloat16](
        (total_pools + extra_blocks) * head_dim
    )
    ctx.enqueue_memset(out_dev, 0)

    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(gate_dev, gate_host)
    ctx.enqueue_copy(ape_dev, ape_host)
    ctx.enqueue_copy(iro_dev, iro_host)
    ctx.enqueue_copy(pro_dev, pro_host)
    ctx.synchronize()

    var out_tile = TileTensor(
        out_dev, row_major(total_pools + extra_blocks, head_dim)
    )
    var k_tile = TileTensor(k_dev, row_major(total_tokens, head_dim))
    var gate_tile = TileTensor(gate_dev, row_major(total_tokens, head_dim))
    var ape_tile = TileTensor(ape_dev, row_major(kpool, head_dim))
    var iro_tile = TileTensor(iro_dev, row_major(batch_size + 1))
    var pro_tile = TileTensor(pro_dev, row_major(batch_size + 1))
    var clen_tile = TileTensor(clen_dev, row_major(batch_size))

    comptime kernel = kpool_compress_kernel[
        .bfloat16,
        type_of(k_tile.as_immut()).LayoutType,
        ImmOrigin(k_tile.origin),
        type_of(gate_tile.as_immut()).LayoutType,
        ImmOrigin(gate_tile.origin),
        type_of(ape_tile.as_immut()).LayoutType,
        ImmOrigin(ape_tile.origin),
        type_of(iro_tile.as_immut()).LayoutType,
        ImmOrigin(iro_tile.origin),
        type_of(pro_tile.as_immut()).LayoutType,
        ImmOrigin(pro_tile.origin),
        type_of(clen_tile.as_immut()).LayoutType,
        out_tile.LayoutType,
        out_tile.origin,
        head_dim,
        kpool,
    ]
    ctx.enqueue_function[kernel](
        out_tile,
        k_tile.as_immut(),
        gate_tile.as_immut(),
        ape_tile.as_immut(),
        iro_tile.as_immut(),
        pro_tile.as_immut(),
        clen_tile.as_immut(),
        # Over-sized on purpose: the surplus blocks must write nothing.
        grid_dim=(total_pools + extra_blocks, 1, 1),
        block_dim=(head_dim, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for b in range(batch_size):
        for p in range(pools_per[b]):
            var pool_row = Int(pro_host[b]) + p
            var first_row = Int(iro_host[b]) + aligns[b] + p * kpool
            for c in range(head_dim):
                # No max-subtraction here, so a match also says the kernel's
                # stabilization was neutral.
                var denom = Float64(0)
                var acc = Float64(0)
                for m in range(kpool):
                    var logit = Float64(
                        gate_host[(first_row + m) * head_dim + c].cast[
                            .float32
                        ]()
                    ) + Float64(ape_host[m * head_dim + c])
                    var w = exp(logit)
                    denom += w
                    acc += w * Float64(
                        k_host[(first_row + m) * head_dim + c].cast[.float32]()
                    )
                var want = acc / denom
                var got = Float64(
                    out_host[pool_row * head_dim + c].cast[.float32]()
                )
                # The tolerance comes from bf16's 8-bit mantissa.
                assert_almost_equal(
                    got,
                    want,
                    atol=4e-3,
                    rtol=1e-2,
                    msg=String("pool ", pool_row, " channel ", c, " mismatch"),
                )

    # Surplus blocks must have written nothing.
    for i in range(
        total_pools * head_dim, (total_pools + extra_blocks) * head_dim
    ):
        assert_equal(
            out_host[i],
            Scalar[.bfloat16](0),
            String("a surplus block wrote output element ", i),
        )

    _ = k_dev
    _ = gate_dev
    _ = ape_dev
    _ = iro_dev
    _ = pro_dev
    _ = clen_dev
    _ = out_dev


def test_kpool_one_is_identity[head_dim: Int](ctx: DeviceContext) raises:
    """At kpool=1 a pooled key must be its single member, bit for bit.

    A one-member softmax is exactly 1.0 whatever the gate and position
    embedding hold, so this pins that neither reaches the output.
    """
    comptime kpool = 1
    var seq_lens = [5, 3]
    var batch_size = len(seq_lens)
    var total_tokens = 8

    var k_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * head_dim
    )
    var gate_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * head_dim
    )
    var ape_host = ctx.enqueue_create_host_buffer[.float32](kpool * head_dim)
    var iro_host = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var pro_host = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var clen_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * head_dim
    )
    ctx.synchronize()

    rand(k_host.unsafe_ptr(), total_tokens * head_dim)
    rand(gate_host.unsafe_ptr(), total_tokens * head_dim)
    rand(ape_host.unsafe_ptr(), kpool * head_dim)
    # At kpool=1 neither the gate nor the position embedding may reach the
    # output, so make both large and uneven.
    for i in range(total_tokens * head_dim):
        gate_host[i] = (gate_host[i] - 0.5) * 20.0
    for i in range(kpool * head_dim):
        ape_host[i] = (ape_host[i] - 0.5) * 20.0

    iro_host[0] = 0
    iro_host[1] = UInt32(seq_lens[0])
    iro_host[2] = UInt32(total_tokens)
    pro_host[0] = 0
    pro_host[1] = UInt32(seq_lens[0])
    pro_host[2] = UInt32(total_tokens)

    for b in range(batch_size):
        clen_host[b] = 0
    var clen_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(clen_dev, clen_host)
    var k_dev = ctx.enqueue_create_buffer[.bfloat16](total_tokens * head_dim)
    var gate_dev = ctx.enqueue_create_buffer[.bfloat16](total_tokens * head_dim)
    var ape_dev = ctx.enqueue_create_buffer[.float32](kpool * head_dim)
    var iro_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var pro_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var out_dev = ctx.enqueue_create_buffer[.bfloat16](total_tokens * head_dim)

    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(gate_dev, gate_host)
    ctx.enqueue_copy(ape_dev, ape_host)
    ctx.enqueue_copy(iro_dev, iro_host)
    ctx.enqueue_copy(pro_dev, pro_host)
    ctx.synchronize()

    var out_tile = TileTensor(out_dev, row_major(total_tokens, head_dim))
    var k_tile = TileTensor(k_dev, row_major(total_tokens, head_dim))
    var gate_tile = TileTensor(gate_dev, row_major(total_tokens, head_dim))
    var ape_tile = TileTensor(ape_dev, row_major(kpool, head_dim))
    var iro_tile = TileTensor(iro_dev, row_major(batch_size + 1))
    var pro_tile = TileTensor(pro_dev, row_major(batch_size + 1))
    var clen_tile = TileTensor(clen_dev, row_major(batch_size))

    comptime kernel = kpool_compress_kernel[
        .bfloat16,
        type_of(k_tile.as_immut()).LayoutType,
        ImmOrigin(k_tile.origin),
        type_of(gate_tile.as_immut()).LayoutType,
        ImmOrigin(gate_tile.origin),
        type_of(ape_tile.as_immut()).LayoutType,
        ImmOrigin(ape_tile.origin),
        type_of(iro_tile.as_immut()).LayoutType,
        ImmOrigin(iro_tile.origin),
        type_of(pro_tile.as_immut()).LayoutType,
        ImmOrigin(pro_tile.origin),
        type_of(clen_tile.as_immut()).LayoutType,
        out_tile.LayoutType,
        out_tile.origin,
        head_dim,
        kpool,
    ]
    ctx.enqueue_function[kernel](
        out_tile,
        k_tile.as_immut(),
        gate_tile.as_immut(),
        ape_tile.as_immut(),
        iro_tile.as_immut(),
        pro_tile.as_immut(),
        clen_tile.as_immut(),
        grid_dim=(total_tokens, 1, 1),
        block_dim=(head_dim, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(total_tokens * head_dim):
        assert_equal(
            out_host[i],
            k_host[i],
            String("kpool=1 changed element ", i),
        )
    print("kpool=1 identity holds over", total_tokens * head_dim, "elements")

    _ = k_dev
    _ = gate_dev
    _ = ape_dev
    _ = iro_dev
    _ = pro_dev
    _ = clen_dev
    _ = out_dev


def main() raises:
    with DeviceContext() as ctx:
        test_kpool_one_is_identity[head_dim=128](ctx)

        # GLM-5.3-Flash geometry.
        test_kpool_compress[head_dim=128, kpool=4](seq_lens=[8, 4, 12], ctx=ctx)
        # Lengths not divisible by kpool: the trailing tokens must be ignored
        # rather than folded into a short pool.
        test_kpool_compress[head_dim=128, kpool=4](
            seq_lens=[7, 5, 3, 1], ctx=ctx
        )
        # Cached prefixes at every phase: a request whose prefix ends
        # mid-pool opens this call partway through a pool it cannot build.
        test_kpool_compress[head_dim=128, kpool=4](
            seq_lens=[9, 9, 9, 9], ctx=ctx, cache_lens=[0, 1, 2, 3]
        )
        # An over-sized grid: the kernel bounds itself from the offsets, so
        # the surplus blocks must not write.
        test_kpool_compress[head_dim=128, kpool=4](
            seq_lens=[8, 4], ctx=ctx, extra_blocks=3
        )
        # A single request, and a pool count above one block per request.
        test_kpool_compress[head_dim=128, kpool=4](seq_lens=[64], ctx=ctx)

        print("\nAll tests passed!")
