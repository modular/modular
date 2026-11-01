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


from max.gpu.host import DeviceContext, HostBuffer
from std.collections import Set
from std.math import align_up, ceildiv
from layout import Idx, TileTensor, row_major
from layout._fillers import random
from nn.moe import moe_create_indices, moe_create_indices_with_slots
from std.random import random_float64, random_ui64, seed
from std.testing import assert_equal, assert_true


def get_expert_dictionary(
    topk_ids: HostBuffer[.uint32], num_tokens: Int
) -> Dict[UInt32, UInt32]:
    var expert_dictionary = Dict[UInt32, UInt32]()

    for i in range(num_tokens):
        var expert_id = topk_ids[i]
        var current_value = expert_dictionary.get(expert_id, 0)
        current_value += 1
        expert_dictionary[expert_id] = current_value

    return expert_dictionary^


def check_token_expert_order(
    token_expert_order: HostBuffer[.uint32],
    topk_ids: HostBuffer[.uint32],
    num_tokens: Int,
) raises:
    """
    This function asserts all tokens of the same expert are together in the token_expert_order.
    """

    var expert_dictionary = get_expert_dictionary(topk_ids, num_tokens)
    var current_expert_id = topk_ids[Int(token_expert_order[0])]
    var token_count = expert_dictionary.get(current_expert_id, 0) - 1

    for i in range(1, num_tokens):
        var expert_id = topk_ids[Int(token_expert_order[i])]

        if expert_id != current_expert_id:
            assert_equal(token_count, 0, "tokens are grouped incorrectly")
            expert_dictionary[current_expert_id] = 0
            current_expert_id = expert_id

            token_count = expert_dictionary.get(current_expert_id, 0) - 1
        else:
            token_count -= 1

    assert_equal(token_count, 0, "tokens are grouped incorrectly")
    expert_dictionary[current_expert_id] = 0

    for k_v in expert_dictionary.take_items():
        assert_equal(k_v.value, 0, "tokens are grouped incorrectly")


def check_expert_stats(
    expert_usage_stats: HostBuffer[.uint32],
    topk_ids: HostBuffer[.uint32],
    num_tokens: Int,
    num_experts: Int,
) raises:
    """
    Checks if the most frequent expert is accurate, and if the number of experts is correct.
    """

    var expert_dictionary = get_expert_dictionary(topk_ids, num_tokens)

    var mx_value: UInt32 = 0

    for k_v in expert_dictionary.take_items():
        if k_v.value > mx_value:
            mx_value = k_v.value

    assert_equal(
        expert_usage_stats[0], mx_value, "most frequent expert is incorrect"
    )
    assert_equal(
        expert_usage_stats[1],
        UInt32(num_experts),
        "expert count is incorrect",
    )


def check_expert_indices(
    expert_start_indices: HostBuffer[.uint32],
    expert_ids: HostBuffer[.int32],
    token_expert_order: HostBuffer[.uint32],
    expert_usage_stats: HostBuffer[.uint32],
    topk_ids: HostBuffer[.uint32],
    num_tokens: Int,
) raises:
    """
    Checks if the provided start indices are correct.
    """

    for i in range(expert_usage_stats[1]):
        var start_idx = expert_start_indices[Int(i)]
        var end_idx = expert_start_indices[Int(i + 1)]
        var expert_id = expert_ids[Int(i)]

        for j in range(start_idx, end_idx):
            var current_expert_id = topk_ids[Int(token_expert_order[Int(j)])]
            assert_equal(
                expert_id,
                Int32(current_expert_id),
                "expert range in start indices is incorrect",
            )


def check_total_token_count(
    expert_start_indices: HostBuffer[.uint32],
    num_experts: Int,
    num_tokens: Int,
) raises:
    """
    Checks that the final CSR boundary accounts for every token.
    """

    assert_equal(
        Int(expert_start_indices[num_experts]),
        num_tokens,
        "expert_start_indices[num_experts] does not cover every token",
    )


def check_expert_ids_permutation(
    expert_ids: HostBuffer[.int32],
    num_experts: Int,
) raises:
    """
    Checks that expert_ids holds every expert in [0, num_experts) exactly
    once, regardless of which order a kernel assigns them to slots.
    """

    var seen = Set[Int]()
    for i in range(num_experts):
        var expert_id = Int(expert_ids[i])
        assert_true(
            expert_id >= 0 and expert_id < num_experts,
            "expert_ids["
            + String(i)
            + "] is out of range: "
            + String(expert_id),
        )
        assert_true(
            not (expert_id in seen),
            "expert id appears more than once: " + String(expert_id),
        )
        seen.add(expert_id)


def check_restore_token_order(
    restore_token_order: HostBuffer[.uint32],
    token_expert_order: HostBuffer[.uint32],
    num_tokens: Int,
) raises:
    """
    Checks if original export order can be restored.
    """

    for i in range(num_tokens):
        assert_equal(
            i,
            Int(token_expert_order[Int(restore_token_order[i])]),
            "restore token order is incorrect",
        )


def check_scales_offset[
    scale_alignment: Int = 128,
](
    scales_offset: HostBuffer[.uint32],
    expert_start_indices: HostBuffer[.uint32],
    expert_usage_stats: HostBuffer[.uint32],
) raises:
    """Validates scales_offset values against expert_start_indices.

    For each expert i (in ascending expert id), scales_offset[i] should
    equal the difference between the cumulative aligned block count and the
    cumulative actual block count up to that expert.
    """
    var num_experts = Int(expert_usage_stats[1])
    var cumulative_actual: UInt32 = 0
    var cumulative_aligned: UInt32 = 0

    for i in range(num_experts):
        var expected = cumulative_aligned // UInt32(
            scale_alignment
        ) - cumulative_actual // UInt32(scale_alignment)
        assert_equal(
            scales_offset[i],
            expected,
            "scales_offset mismatch at expert index " + String(i),
        )
        var token_count = expert_start_indices[i + 1] - expert_start_indices[i]
        cumulative_actual += token_count
        cumulative_aligned += align_up(token_count, UInt32(scale_alignment))


def fill_skewed(top_k_buffer_host: HostBuffer[.uint32], num_tokens: Int):
    """
    Routes every token to expert 0, the extreme case of a skewed expert
    distribution the uniform random fill never produces.
    """

    for i in range(num_tokens):
        top_k_buffer_host[i] = 0


def test_moe_create_indices[
    num_experts: Int = 256,
    test_scales_offset: Bool = False,
    skewed: Bool = False,
](token_expert_order_length: Int, ctx: DeviceContext) raises:
    var token_expert_order_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](token_expert_order_length)
    var top_k_buffer_host = ctx.enqueue_create_host_buffer[.uint32](
        token_expert_order_length
    )
    var restore_token_order_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](token_expert_order_length)
    var expert_usage_stats_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](2)
    var expert_start_indices_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](num_experts + 1)
    var expert_ids_buffer_host = ctx.enqueue_create_host_buffer[.int32](
        num_experts
    )

    var token_expert_order_buffer_device = ctx.enqueue_create_buffer[
        DType.uint32
    ](token_expert_order_length)
    var expert_start_indices_buffer = ctx.enqueue_create_buffer[.uint32](
        num_experts + 1
    )
    var restore_token_order_buffer = ctx.enqueue_create_buffer[.uint32](
        token_expert_order_length
    )
    var expert_ids_buffer = ctx.enqueue_create_buffer[.int32](num_experts)
    var expert_usage_stats_buffer = ctx.enqueue_create_buffer[.uint32](2)
    var top_k_buffer_device = ctx.enqueue_create_buffer[.uint32](
        token_expert_order_length
    )

    var token_expert_order = TileTensor(
        token_expert_order_buffer_device,
        row_major(token_expert_order_length),
    )

    var expert_start_indices = TileTensor(
        expert_start_indices_buffer,
        row_major(num_experts + 1),
    )

    var restore_token_order = TileTensor(
        restore_token_order_buffer,
        row_major(token_expert_order_length),
    )

    var expert_ids = TileTensor(
        expert_ids_buffer,
        row_major(num_experts),
    )

    var expert_usage_stats = TileTensor(
        expert_usage_stats_buffer,
        row_major(Idx[2]),
    )

    var top_k = TileTensor(
        top_k_buffer_device,
        row_major(token_expert_order_length),
    )

    var top_k_host = TileTensor(
        top_k_buffer_host,
        row_major(token_expert_order_length),
    )

    ctx.synchronize()

    # Fill top_k_host with random expert IDs, or route every token to one
    # expert to exercise a skewed (non-uniform) distribution.
    comptime if skewed:
        fill_skewed(top_k_buffer_host, token_expert_order_length)
    else:
        random(top_k_host, min=0, max=UInt32(num_experts))
    ctx.enqueue_copy(top_k_buffer_device, top_k_buffer_host)

    var scales_offset_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](num_experts)
    var scales_offset_buffer = ctx.enqueue_create_buffer[.uint32](num_experts)

    comptime if test_scales_offset:
        moe_create_indices["gpu"](
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
            top_k,
            ctx,
            scales_offset_p=scales_offset_buffer.unsafe_ptr().as_unsafe_any_origin(),
        )
    else:
        moe_create_indices["gpu"](
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
            top_k,
            ctx,
        )

    ctx.enqueue_copy(
        token_expert_order_buffer_host, token_expert_order_buffer_device
    )
    ctx.enqueue_copy(
        restore_token_order_buffer_host, restore_token_order_buffer
    )
    ctx.enqueue_copy(expert_usage_stats_buffer_host, expert_usage_stats_buffer)
    ctx.enqueue_copy(expert_ids_buffer_host, expert_ids_buffer)
    ctx.enqueue_copy(
        expert_start_indices_buffer_host, expert_start_indices_buffer
    )
    ctx.enqueue_copy(scales_offset_buffer_host, scales_offset_buffer)
    ctx.synchronize()

    comptime if test_scales_offset:
        check_scales_offset(
            scales_offset_buffer_host,
            expert_start_indices_buffer_host,
            expert_usage_stats_buffer_host,
        )

    check_token_expert_order(
        token_expert_order_buffer_host,
        top_k_buffer_host,
        token_expert_order_length,
    )
    check_expert_stats(
        expert_usage_stats_buffer_host,
        top_k_buffer_host,
        token_expert_order_length,
        num_experts,
    )
    check_expert_indices(
        expert_start_indices_buffer_host,
        expert_ids_buffer_host,
        token_expert_order_buffer_host,
        expert_usage_stats_buffer_host,
        top_k_buffer_host,
        token_expert_order_length,
    )
    check_total_token_count(
        expert_start_indices_buffer_host,
        num_experts,
        token_expert_order_length,
    )
    check_expert_ids_permutation(expert_ids_buffer_host, num_experts)

    check_restore_token_order(
        restore_token_order_buffer_host,
        token_expert_order_buffer_host,
        token_expert_order_length,
    )


def test_moe_create_indices_with_slots[
    num_groups: Int,
    local_experts: Int,
    slot_stride: Int = 6,
    skew: Int = 0,
](
    num_slots: Int, global_experts: Int, expert_offset: Int, ctx: DeviceContext
) raises:
    """Checks every output of the slot-layout variant against a host model.

    Ids are drawn from `[0, global_experts)` and mapped as the kernel does
    (`id - expert_offset` if local, else group `local_experts`). The CSR,
    usage and scale offsets are fully determined, as are the scale-row and
    per-slot maps given the kernel's order; the order inside a group follows
    the atomic cursor and is only checked for consistency. `skew` > 0 routes
    every slot to global expert `skew - 1`.
    """
    comptime assert (
        num_groups == local_experts or num_groups == local_experts + 1
    )
    comptime align = 128
    # The bound `DeepseekV4RoutedExperts` allocates: only non-empty groups
    # take a tile, and there are at most min(groups, slots) of them.
    var scale_rows = align * (
        ceildiv(num_slots, align) + min(num_groups, num_slots)
    )

    var ids_h = ctx.enqueue_create_host_buffer[.int32](num_slots)
    var w_h = ctx.enqueue_create_host_buffer[.float32](num_slots)
    var order_h = ctx.enqueue_create_host_buffer[.uint32](num_slots)
    var start_h = ctx.enqueue_create_host_buffer[.uint32](num_groups + 1)
    var restore_h = ctx.enqueue_create_host_buffer[.int32](num_slots)
    var eids_h = ctx.enqueue_create_host_buffer[.int32](num_groups)
    var usage_h = ctx.enqueue_create_host_buffer[.uint32](2)
    var so_h = ctx.enqueue_create_host_buffer[.uint32](num_groups)
    var st_h = ctx.enqueue_create_host_buffer[.int32](num_slots)
    var sw_h = ctx.enqueue_create_host_buffer[.float32](num_slots)
    var ss_h = ctx.enqueue_create_host_buffer[.int32](scale_rows)

    var ids_d = ctx.enqueue_create_buffer[.int32](num_slots)
    var w_d = ctx.enqueue_create_buffer[.float32](num_slots)
    var order_d = ctx.enqueue_create_buffer[.uint32](num_slots)
    var start_d = ctx.enqueue_create_buffer[.uint32](num_groups + 1)
    var restore_d = ctx.enqueue_create_buffer[.int32](num_slots)
    var eids_d = ctx.enqueue_create_buffer[.int32](num_groups)
    var usage_d = ctx.enqueue_create_buffer[.uint32](2)
    var so_d = ctx.enqueue_create_buffer[.uint32](num_groups)
    var st_d = ctx.enqueue_create_buffer[.int32](num_slots)
    var sw_d = ctx.enqueue_create_buffer[.float32](num_slots)
    var ss_d = ctx.enqueue_create_buffer[.int32](scale_rows)
    ctx.synchronize()

    for i in range(num_slots):
        if skew > 0:
            ids_h[i] = Int32(skew - 1)
        else:
            ids_h[i] = Int32(random_ui64(0, UInt64(global_experts - 1)))
        w_h[i] = Float32(random_float64())
    # The scale-row pass must overwrite every row, so start from garbage.
    for r in range(scale_rows):
        ss_h[r] = -7
    ctx.enqueue_copy(ids_d, ids_h)
    ctx.enqueue_copy(w_d, w_h)
    ctx.enqueue_copy(ss_d, ss_h)

    moe_create_indices_with_slots[slot_stride, "gpu"](
        TileTensor(order_d, row_major(num_slots)),
        TileTensor(start_d, row_major(num_groups + 1)),
        TileTensor(restore_d, row_major(num_slots)),
        TileTensor(eids_d, row_major(num_groups)),
        TileTensor(usage_d, row_major(Idx[2])),
        so_d.unsafe_ptr().as_unsafe_any_origin(),
        st_d.unsafe_ptr().as_unsafe_any_origin(),
        sw_d.unsafe_ptr().as_unsafe_any_origin(),
        ss_d.unsafe_ptr().as_unsafe_any_origin(),
        scale_rows,
        TileTensor(ids_d, row_major(num_slots)).as_imm(),
        w_d.unsafe_ptr().as_unsafe_any_origin(),
        expert_offset,
        local_experts,
        ctx,
    )

    ctx.enqueue_copy(order_h, order_d)
    ctx.enqueue_copy(start_h, start_d)
    ctx.enqueue_copy(restore_h, restore_d)
    ctx.enqueue_copy(eids_h, eids_d)
    ctx.enqueue_copy(usage_h, usage_d)
    ctx.enqueue_copy(so_h, so_d)
    ctx.enqueue_copy(st_h, st_d)
    ctx.enqueue_copy(sw_h, sw_d)
    ctx.enqueue_copy(ss_h, ss_d)
    ctx.synchronize()

    # Host model of the group each slot lands in, and the CSR it implies.
    var group = List[Int](capacity=num_slots)
    var counts = List[Int](length=num_groups, fill=0)
    for i in range(num_slots):
        var g = Int(ids_h[i]) - expert_offset
        if g < 0 or g >= local_experts:
            g = local_experts
        assert_true(g < num_groups, "slot mapped past the last group")
        group.append(g)
        counts[g] += 1
    var start = List[Int](length=num_groups + 1, fill=0)
    var aligned = List[Int](length=num_groups + 1, fill=0)
    var max_count = 0
    for g in range(num_groups):
        start[g + 1] = start[g] + counts[g]
        aligned[g + 1] = aligned[g] + align_up(counts[g], align)
        max_count = max(max_count, counts[g])

    for g in range(num_groups + 1):
        assert_equal(Int(start_h[g]), start[g], "expert_start_indices")
    for g in range(num_groups):
        assert_equal(Int(eids_h[g]), g, "expert_ids")
        assert_equal(
            Int(so_h[g]),
            aligned[g] // align - start[g] // align,
            "scales_offset",
        )
    assert_equal(Int(usage_h[0]), max_count, "usage max count")
    assert_equal(Int(usage_h[1]), num_groups, "usage expert count")

    # Order: a permutation of the slots, each in its own group's range, with
    # restore its inverse; the per-slot maps follow it exactly.
    for p in range(num_slots):
        var i = Int(order_h[p])
        assert_true(i < num_slots, "order out of range")
        assert_true(
            start[group[i]] <= p and p < start[group[i] + 1],
            "slot sorted outside its group",
        )
        assert_equal(Int(restore_h[i]), p, "restore is not order's inverse")
        assert_equal(Int(st_h[p]), i // slot_stride, "slot_token")
        assert_equal(
            sw_h[p].to_bits(), w_h[i].to_bits(), "slot_weight not bitwise"
        )

    # Scale rows, as the graph formula: the group whose aligned range holds
    # the row (clamped to the last), slot start + offset while in the count.
    for r in range(scale_rows):
        var g = 0
        for e in range(num_groups):
            if aligned[e + 1] <= r:
                g += 1
        g = min(g, num_groups - 1)
        var in_group = r - aligned[g]
        var want = start[g] + in_group if in_group < counts[g] else 0
        assert_equal(Int(ss_h[r]), want, "scale_slot at row " + String(r))


def main() raises:
    seed(0)
    with DeviceContext() as ctx:
        test_moe_create_indices(197, ctx)
        test_moe_create_indices(2500, ctx)
        test_moe_create_indices(11, ctx)
        test_moe_create_indices(1, ctx)
        test_moe_create_indices(20660, ctx)
        test_moe_create_indices(100_000, ctx)

        test_moe_create_indices[test_scales_offset=True](197, ctx)
        test_moe_create_indices[test_scales_offset=True](2500, ctx)
        test_moe_create_indices[test_scales_offset=True](11, ctx)
        test_moe_create_indices[test_scales_offset=True](1, ctx)
        test_moe_create_indices[test_scales_offset=True](20660, ctx)
        test_moe_create_indices[test_scales_offset=True](100_000, ctx)

        # The kernel walks the experts in chunks of one block width (512,
        # 1024 on Apple), carrying the running offsets across chunks. These
        # cover exact and partial chunks at both widths; 1025 ends in a
        # single-expert chunk, and 2049 gives Apple a third chunk.
        test_moe_create_indices[num_experts=512](100, ctx)
        test_moe_create_indices[num_experts=513](2500, ctx)
        test_moe_create_indices[num_experts=896](2500, ctx)
        test_moe_create_indices[num_experts=896, test_scales_offset=True](
            2500, ctx
        )
        test_moe_create_indices[num_experts=896, skewed=True](16, ctx)
        test_moe_create_indices[num_experts=1024](100_000, ctx)
        test_moe_create_indices[num_experts=513, test_scales_offset=True](
            2500, ctx
        )
        test_moe_create_indices[num_experts=1025](2500, ctx)
        test_moe_create_indices[num_experts=2049](20660, ctx)

        # Skewed distribution: every token on expert 0, so one group holds
        # them all and the rest are empty.
        test_moe_create_indices[skewed=True](1024, ctx)
        test_moe_create_indices[num_experts=513, skewed=True](100_000, ctx)

        # Slot layout. DeepSeek-V4 decode/prefill shapes, expert parallel on
        # two devices (128 local + 1 remote group) and on one (256 groups).
        test_moe_create_indices_with_slots[129, 128](6, 256, 0, ctx)
        test_moe_create_indices_with_slots[129, 128](6, 256, 128, ctx)
        test_moe_create_indices_with_slots[129, 128](6 * 197, 256, 128, ctx)
        test_moe_create_indices_with_slots[129, 128](6 * 4096, 256, 0, ctx)
        test_moe_create_indices_with_slots[256, 256](6, 256, 0, ctx)
        test_moe_create_indices_with_slots[256, 256](6 * 2500, 256, 0, ctx)
        # Power-of-two stride; three expert chunks with the remote group
        # alone in the last one.
        test_moe_create_indices_with_slots[1025, 1024, slot_stride=8](
            8 * 300, 2048, 1024, ctx
        )
        # Every slot remote, and every slot on one local expert.
        test_moe_create_indices_with_slots[129, 128, skew=1](600, 256, 128, ctx)
        test_moe_create_indices_with_slots[129, 128, skew=200](
            6 * 1000, 256, 128, ctx
        )
