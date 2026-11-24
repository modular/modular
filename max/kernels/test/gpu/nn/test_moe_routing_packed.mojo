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
"""The packed routing rows: the router writes them, the index build reads
them."""

from std.math import align_up
from std.os import getenv
from std.random import random_float64, random_ui64, seed
from std.testing import assert_equal, assert_true
from std.utils.coord import Coord

from layout import Idx, TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext
from nn.moe import single_group_router
from nn.moe_indices_packed import (
    POISONS,
    Out,
    PackedIndicesArgs,
    moe_indices_packed,
)


def test_router_packed[
    n_experts: Int, top_k: Int
](tokens: Int, ctx: DeviceContext) raises:
    """Each packed row is the token's ids, as f32, then its weights, bit for
    bit the router's separate outputs."""
    var scores = ctx.enqueue_create_buffer[.float32](tokens * n_experts)
    var bias = ctx.enqueue_create_buffer[.float32](n_experts)
    var ids = ctx.enqueue_create_buffer[.int32](tokens * top_k)
    var weights = ctx.enqueue_create_buffer[.float32](tokens * top_k)
    var packed = ctx.enqueue_create_buffer[.float32](tokens * 2 * top_k)
    with scores.map_to_host() as s:
        for i in range(tokens * n_experts):
            s[i] = Float32(random_float64())
    with bias.map_to_host() as b:
        for i in range(n_experts):
            b[i] = Float32(random_float64(-0.1, 0.1))

    var scores_t = TileTensor(scores, row_major((tokens, Idx[n_experts])))
    var bias_t = TileTensor(bias, row_major[n_experts]())

    def load_scores[
        width: Int
    ](coords: Coord) {var scores_t} -> SIMD[DType.float32, width]:
        return scores_t.load[width=width](coords)

    single_group_router[n_experts, top_k, True, packed=True](
        TileTensor(ids, row_major((tokens, Idx[top_k]))),
        TileTensor(weights, row_major((tokens, Idx[top_k]))),
        scores_t.layout,
        load_scores,
        bias_t.as_imm(),
        2.5,
        ctx,
        packed.unsafe_ptr().as_unsafe_any_origin(),
    )
    ctx.synchronize()

    with ids.map_to_host() as i_h, weights.map_to_host() as w_h, packed.map_to_host() as p_h:
        for t in range(tokens):
            for j in range(top_k):
                var row = t * 2 * top_k
                assert_equal(
                    p_h[row + j].to_bits(),
                    Float32(i_h[t * top_k + j]).to_bits(),
                    "packed id",
                )
                assert_equal(
                    p_h[row + top_k + j].to_bits(),
                    w_h[t * top_k + j].to_bits(),
                    "packed weight",
                )


def out[T: DType](b: DeviceBuffer[T]) -> Out[T]:
    return (
        b.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


def indices_mismatches[
    top_k: Int, local_experts: Int, poison: Int = 0
](
    tokens: Int,
    global_experts: Int,
    expert_offset: Int,
    skew: Int,
    ctx: DeviceContext,
) raises -> Int:
    """Elements of every output that differ from a host model of the stable
    counting sort, unwritten outputs included. `skew` > 0 routes every slot
    to expert `skew - 1`."""
    comptime groups = local_experts + 1
    comptime align = 128
    var slots = tokens * top_k
    var routing = ctx.enqueue_create_buffer[.float32](tokens * 2 * top_k)
    var order = ctx.enqueue_create_buffer[.uint32](slots)
    var start = ctx.enqueue_create_buffer[.uint32](groups + 1)
    var restore = ctx.enqueue_create_buffer[.uint32](slots)
    var eids = ctx.enqueue_create_buffer[.int32](groups)
    var usage = ctx.enqueue_create_buffer[.uint32](2)
    var so = ctx.enqueue_create_buffer[.uint32](groups)
    var weights = ctx.enqueue_create_buffer[.float32](slots)
    for b in [order, start, restore, usage, so]:
        b.enqueue_fill(UInt32(0xDEADBEEF))
    eids.enqueue_fill(Int32(-7))
    weights.enqueue_fill(Float32(-1e30))

    var ids = List[Int](capacity=slots)
    var w = List[Float32](capacity=slots)
    with routing.map_to_host() as r:
        for t in range(tokens):
            for j in range(top_k):
                var e = skew - 1 if skew > 0 else Int(
                    random_ui64(0, UInt64(global_experts - 1))
                )
                var x = Float32(random_float64())
                ids.append(e)
                w.append(x)
                r[t * 2 * top_k + j] = Float32(e)
                r[t * 2 * top_k + top_k + j] = x

    moe_indices_packed[top_k, local_experts, poison](
        PackedIndicesArgs(
            routing.unsafe_ptr()
            .unsafe_mut_cast[False]()
            .unsafe_origin_cast[ImmUntrackedOrigin](),
            out(order),
            out(start),
            out(restore),
            out(eids),
            out(usage),
            out(so),
            out(weights),
            Int64(tokens),
            Int64(expert_offset),
        ),
        ctx,
    )
    ctx.synchronize()

    # The model: groups in id order, slots ascending within a group.
    var csr = List[Int](length=groups + 1, fill=0)
    var aligned = List[Int](length=groups + 1, fill=0)
    var counts = List[Int](length=groups, fill=0)
    var group = List[Int](capacity=slots)
    for i in range(slots):
        var g = ids[i] - expert_offset
        if g < 0 or g >= local_experts:
            g = local_experts
        group.append(g)
        counts[g] += 1
    var most = 0
    for g in range(groups):
        csr[g + 1] = csr[g] + counts[g]
        aligned[g + 1] = aligned[g] + align_up(counts[g], align)
        most = max(most, counts[g])
    var want_order = List[Int](length=slots, fill=0)
    var cursor = csr.copy()
    for i in range(slots):
        want_order[cursor[group[i]]] = i
        cursor[group[i]] += 1

    var bad = 0
    with start.map_to_host() as s_h, eids.map_to_host() as e_h, so.map_to_host() as so_h, usage.map_to_host() as u_h:
        for g in range(groups + 1):
            bad += Int(Int(s_h[g]) != csr[g])
        for g in range(groups):
            bad += Int(Int(e_h[g]) != g)
            bad += Int(Int(so_h[g]) != aligned[g] // align - csr[g] // align)
        bad += Int(Int(u_h[0]) != most) + Int(Int(u_h[1]) != groups)
    with order.map_to_host() as o_h, restore.map_to_host() as r_h, weights.map_to_host() as w_h:
        for p in range(slots):
            bad += Int(Int(o_h[p]) != want_order[p])
            bad += Int(Int(r_h[want_order[p]]) != p)
        for i in range(slots):
            bad += Int(w_h[i].to_bits() != w[i].to_bits())
    return bad


def test_indices_packed[
    top_k: Int, local_experts: Int
](
    tokens: Int,
    global_experts: Int,
    expert_offset: Int,
    ctx: DeviceContext,
    skew: Int = 0,
) raises:
    var bad = indices_mismatches[top_k, local_experts](
        tokens, global_experts, expert_offset, skew, ctx
    )
    assert_equal(bad, 0, "packed indices differ from the model")


def _redzone_allocator() -> Bool:
    """Whether `MODULAR_DEBUG_DEVICE_ALLOCATOR=out-of-bounds` is set.

    It aborts on any write past a buffer, including the one planted fault 2
    makes on purpose.
    """
    return "out-of-bounds" in getenv("MODULAR_DEBUG_DEVICE_ALLOCATOR")


def test_poison[poison: Int](ctx: DeviceContext) raises:
    """A planted fault of the kernel must make some case differ."""
    var bad = indices_mismatches[8, 32, poison](67, 256, 64, 0, ctx)
    assert_true(bad > 0, String("planted fault ", poison, " went unnoticed"))


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        # Muse Spark: 256 experts, top 8; tokens prime and ragged.
        test_router_packed[256, 8](1, ctx)
        test_router_packed[256, 8](7, ctx)
        test_router_packed[256, 8](131, ctx)
        # EP8 (32 local experts) and EP2 (128); first, middle and last
        # device; a span ending mid-chunk; every slot remote; every slot on
        # one local expert; more slots than one pass of the block.
        test_indices_packed[8, 32](1, 256, 0, ctx)
        test_indices_packed[8, 32](13, 256, 96, ctx)
        test_indices_packed[8, 32](64, 256, 224, ctx)
        test_indices_packed[8, 32](257, 256, 32, ctx)
        test_indices_packed[8, 128](131, 256, 128, ctx)
        test_indices_packed[8, 32](37, 256, 32, ctx, skew=1)
        test_indices_packed[8, 32](53, 256, 32, ctx, skew=40)
        test_indices_packed[8, 32](1031, 256, 0, ctx)
        comptime for p in range(1, POISONS + 1):
            if p == 2 and _redzone_allocator():
                print("skipping planted fault 2: it overruns `order`")
            else:
                test_poison[p](ctx)
