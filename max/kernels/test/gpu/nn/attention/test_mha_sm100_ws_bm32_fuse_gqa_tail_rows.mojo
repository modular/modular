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

"""Tail-row canary for `fuse_gqa` with a group that does not divide MMA_M.

Target hardware family: NVIDIA SM100 (B200).

The warp-specialized BM=32 depth-128 route fuses GQA groups 9 and 6 even though
they do not divide the 32-row MMA tile. The Q TMA box then delivers
`BM_eff * group` rows (27 at group 9, 30 at group 6) and the remaining 5 / 2
rows of the Q shared-memory tile are not written by TMA. `q_async_copy` in
`load_warp.mojo` zero-fills those `config.gqa_pad_rows()` rows and leaves them
out of the Q barrier's byte count; this test proves that matters.

Shared memory is not cleared between kernel launches, so a kernel that leaves
those rows alone reads whatever the previous grid left there. Each case
launches a grid that fills the whole dynamic shared-memory carveout of every SM
with a 16-bit pattern (bf16 NaN, +huge, plain garbage, or zero as the control)
immediately before the attention launch, then compares the fused output against
`mha_gpu_naive` over the full key range. The pad rows are real tile rows that
the MMA scores, so stale shared memory in them can produce NaN/Inf that
propagates into the softmax statistics and the output; masking the output rows
alone is not enough.

`_run_group` keeps `valid_length` within `BM_eff` (3 at group 9, 5 at group 6)
so the dispatch single-tiles into the fused BM=32 WS config. `_run_boundary`
covers the prompt lengths where the dispatch changes config: `BM_eff` is the
last fused length, `BM_eff + 1` the first one on the unfused BM=32 config (kept
there up to 32), and 33 the first Layout-E one. All of them are checked
against the same naive oracle.
"""

from std.collections import Set
from std.math import ceildiv, isnan, rsqrt
from std.random import random_ui64, seed

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import B200
from max.gpu.memory import external_memory
from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import (
    CausalMask,
    MHAMask,
    SlidingWindowCausalMask,
)

comptime _LUT_TAIL_PAD = 16
comptime NO_POISON = -1

# bf16 bit patterns. The last two are what a previous Q tile could plausibly
# leave behind: a huge finite value and ordinary-looking garbage.
comptime POISON_NAN = 0x7FC0
comptime POISON_HUGE = 0x7F7F
comptime POISON_LARGE = 0x5F80
comptime POISON_ZERO = 0x0000

comptime SMEM_FILL_BYTES = B200.shared_memory_per_multiprocessor - 1024


def padded_lut_cols(cols: Int) -> Int:
    return ((cols + 7) // 8) * 8 + _LUT_TAIL_PAD


def _fill_smem_kernel(pattern: UInt16):
    """Fills this CTA's whole dynamic shared memory with `pattern`."""
    var smem = external_memory[
        UInt16, address_space=.SHARED, alignment=16, name="poison_smem"
    ]()
    for i in range(thread_idx.x, SMEM_FILL_BYTES // 2, block_dim.x):
        smem[i] = pattern


def poison_all_sms(pattern: Int, ctx: DeviceContext) raises:
    """One CTA per SM (the footprint forces it); 2x oversubscribed so every SM
    gets a CTA even if the scheduler doubles up early on."""
    ctx.enqueue_function[_fill_smem_kernel](
        UInt16(pattern),
        grid_dim=2 * ctx.default_device_info.sm_count,
        block_dim=1024,
        shared_mem_bytes=SMEM_FILL_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(SMEM_FILL_BYTES)
        ),
    )


def execute_tail_row_test[
    mask_t: MHAMask,
    num_q_heads: Int,
    poison: Int,
    page_size: Int = 128,
    num_partitions: Int = 0,
](
    mask: mask_t, cache_length: Int, valid_length: Int, ctx: DeviceContext
) raises:
    comptime head_size = 128
    comptime kv_params = KVCacheStaticParams(num_heads=8, head_size=head_size)
    comptime dtype = DType.bfloat16
    comptime group = num_q_heads // kv_params.num_heads
    comptime num_layers = 1
    comptime layer_idx = 0

    var num_keys = cache_length + valid_length
    var total_length = valid_length
    var batch_size = 1

    print(
        "test_mha_sm100_ws_bm32_fuse_gqa_tail_rows: mask=",
        mask_t.name(),
        " group=",
        group,
        " poison=",
        poison,
        " page_size=",
        page_size,
        " P=",
        num_partitions,
        " valid_length=",
        valid_length,
        " num_keys=",
        num_keys,
        sep="",
    )

    comptime Collection = PagedKVCacheCollection[
        dtype,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]

    var scale = rsqrt(Float32(head_size))
    var np_opt = (
        Optional[Int](num_partitions) if num_partitions > 0 else Optional[Int]()
    )

    seed(0x7A11)

    var input_row_offsets = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    input_row_offsets[0] = 0
    input_row_offsets[1] = UInt32(valid_length)
    var input_row_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_dev, input_row_offsets)
    var input_row_offsets_lt = (
        TileTensor(
            input_row_offsets_dev,
            row_major(Int64(len(input_row_offsets_dev))),
        )
        .as_imm()
        .as_unsafe_any_origin()
    )

    var q_size = total_length * num_q_heads * head_size
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_size)
    random(
        TileTensor(
            q_host,
            row_major(Int64(total_length), Idx[num_q_heads], Idx[head_size]),
        )
    )
    var q_dev = ctx.enqueue_create_buffer[dtype](q_size)
    ctx.enqueue_copy(q_dev, q_host)
    var q_lt = TileTensor(
        q_dev,
        row_major(Int64(total_length), Idx[num_q_heads], Idx[head_size]),
    ).as_imm()

    var num_paged_blocks = ceildiv(num_keys, page_size) * batch_size + 4
    comptime BlocksLayout = Collection.blocks_tt_layout
    var blocks_shape = Coord[*BlocksLayout.shape_types]()
    blocks_shape[0] = Int64(num_paged_blocks)
    blocks_shape[2] = Int64(num_layers)
    var blocks_strides = Coord[*BlocksLayout.stride_types]()
    blocks_strides[0] = Int64(
        2 * num_layers * page_size * kv_params.num_heads * head_size
    )
    blocks_strides[1] = Int64(
        num_layers * page_size * kv_params.num_heads * head_size
    )
    var kv_blocks_layout = BlocksLayout(blocks_shape, blocks_strides)
    var kv_block_size = (
        num_paged_blocks
        * 2
        * num_layers
        * page_size
        * kv_params.num_heads
        * head_size
    )
    var kv_block_host = ctx.enqueue_create_host_buffer[dtype](kv_block_size)
    random(TileTensor(kv_block_host, kv_blocks_layout))
    var kv_block_dev = ctx.enqueue_create_buffer[dtype](kv_block_size)
    ctx.enqueue_copy(kv_block_dev, kv_block_host)
    var kv_block_paged_lt = TileTensor(kv_block_dev, kv_blocks_layout)

    # Unique physical block per logical page; the padded LUT tail repeats the
    # last real page so a SIMD populate past the end reads an in-range block.
    var full_pages = ceildiv(num_keys, page_size)
    var lut_cols = padded_lut_cols(full_pages)
    var paged_lut_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size * lut_cols
    )
    var lut_set = Set[Int]()
    for blk in range(full_pages):
        var randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
        while randval in lut_set:
            randval = Int(random_ui64(0, UInt64(num_paged_blocks - 1)))
        lut_set.add(randval)
        paged_lut_host[blk] = UInt32(randval)
    for blk in range(full_pages, lut_cols):
        paged_lut_host[blk] = paged_lut_host[full_pages - 1]

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    cache_lengths_host[0] = UInt32(cache_length)
    var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    var cache_lengths_lt = (
        TileTensor(cache_lengths_dev, row_major(Int64(len(cache_lengths_dev))))
        .as_imm()
        .as_unsafe_any_origin()
    )

    var paged_lut_dev = ctx.enqueue_create_buffer[.uint32](
        batch_size * lut_cols
    )
    ctx.enqueue_copy(paged_lut_dev, paged_lut_host)
    var paged_lut_lt = (
        TileTensor(paged_lut_dev, row_major(Int64(batch_size), Int64(lut_cols)))
        .as_imm()
        .as_unsafe_any_origin()
    )

    var kv_collection = Collection(
        kv_block_paged_lt.as_unsafe_any_origin(),
        cache_lengths_lt,
        paged_lut_lt,
        UInt32(valid_length),
        UInt32(num_keys),
    )
    var k_cache = kv_collection.get_key_cache(layer_idx)
    var v_cache = kv_collection.get_value_cache(layer_idx)

    var test_out_size = total_length * num_q_heads * head_size
    var test_out_dev = ctx.enqueue_create_buffer[dtype](test_out_size)
    var test_out_lt = TileTensor(
        test_out_dev,
        row_major(Int64(total_length), Idx[num_q_heads], Idx[head_size]),
    )
    var ctrl_out_dev = ctx.enqueue_create_buffer[dtype](test_out_size)
    var ctrl_out_lt = TileTensor(
        ctrl_out_dev,
        row_major(Int64(total_length), Idx[num_q_heads], Idx[head_size]),
    )
    var ref_out_dev = ctx.enqueue_create_buffer[dtype](test_out_size)
    var ref_out_lt = TileTensor(
        ref_out_dev,
        row_major(Int64(total_length), Idx[num_q_heads], Idx[head_size]),
    )

    # Everything above is staged on the stream; the poison grid is the last
    # launch before each kernel run. The control run leaves zeros in shared
    # memory, which is what the tail rows must look like to the MMA.
    ctx.synchronize()
    comptime if poison != NO_POISON:
        poison_all_sms(POISON_ZERO, ctx)
        flash_attention[ragged=True](
            ctrl_out_lt,
            q_lt,
            k_cache,
            v_cache,
            mask,
            input_row_offsets_lt,
            scale,
            ctx,
            num_partitions=np_opt,
        )
        ctx.synchronize()
        poison_all_sms(poison, ctx)
    flash_attention[ragged=True](
        test_out_lt,
        q_lt,
        k_cache,
        v_cache,
        mask,
        input_row_offsets_lt,
        scale,
        ctx,
        num_partitions=np_opt,
    )
    ctx.synchronize()

    mha_gpu_naive[ragged=True](
        q_lt,
        k_cache,
        v_cache,
        mask,
        ref_out_lt,
        input_row_offsets_lt,
        scale,
        batch_size,
        valid_length,
        num_keys,
        num_q_heads,
        head_size,
        group,
        ctx,
    )
    ctx.synchronize()

    var test_out_host = ctx.enqueue_create_host_buffer[dtype](test_out_size)
    var ctrl_out_host = ctx.enqueue_create_host_buffer[dtype](test_out_size)
    var ref_out_host = ctx.enqueue_create_host_buffer[dtype](test_out_size)
    ctx.enqueue_copy(test_out_host, test_out_dev)
    ctx.enqueue_copy(ctrl_out_host, ctrl_out_dev)
    ctx.enqueue_copy(ref_out_host, ref_out_dev)
    ctx.synchronize()

    var max_abs_diff: Float32 = 0.0
    var argmax_idx = 0
    var nan_count = 0
    var moved_count = 0
    # FNV-1a over the output bits, so two builds (e.g. fused vs unfused
    # routing) can be compared for bit-identity from their logs.
    var out_hash = UInt64(0xCBF29CE484222325)
    for flat in range(test_out_size):
        var a = test_out_host[flat].cast[.float32]()
        out_hash = (
            out_hash ^ UInt64(test_out_host[flat].cast[.float32]().to_bits())
        ) * UInt64(0x100000001B3)
        comptime if poison != NO_POISON:
            if test_out_host[flat] != ctrl_out_host[flat]:
                moved_count += 1
        var expected = ref_out_host[flat].cast[.float32]()
        if isnan(a):
            nan_count += 1
        var diff = abs(a - expected)
        if diff > max_abs_diff:
            max_abs_diff = diff
            argmax_idx = flat
    print(
        "  max-abs diff(fused O, naive) =",
        max_abs_diff,
        " at flat idx",
        argmax_idx,
        " nan_count=",
        nan_count,
        " moved_vs_zero_control=",
        moved_count,
        " out_hash=",
        out_hash,
    )

    _ = q_dev^
    _ = kv_block_dev^
    _ = paged_lut_dev^
    _ = input_row_offsets_dev^
    _ = cache_lengths_dev^
    _ = test_out_dev^
    _ = ctrl_out_dev^
    _ = ref_out_dev^

    # bf16-MMA + f32-softmax vs f32 naive over the same bf16 inputs, matching
    # `test_mha_sm100_ws_bm32`.
    comptime atol = Float32(0.04)
    # The poisoned run must be bit-identical to the zero-filled control: a stale
    # tail row that only perturbs the lazy-rescale vote still moves rounding.
    if nan_count > 0 or max_abs_diff > atol or moved_count > 0:
        raise Error(
            "test_mha_sm100_ws_bm32_fuse_gqa_tail_rows: FAILED group="
            + String(group)
            + " poison="
            + String(poison)
            + " num_keys="
            + String(num_keys)
            + " nan_count="
            + String(nan_count)
            + " max_abs_diff="
            + String(max_abs_diff)
            + " moved_vs_zero_control="
            + String(moved_count)
        )
    print("test_mha_sm100_ws_bm32_fuse_gqa_tail_rows: PASSED")


def _run_group[num_q_heads: Int](ctx: DeviceContext, max_valid: Int) raises:
    comptime for poison in [
        NO_POISON,
        POISON_ZERO,
        POISON_LARGE,
        POISON_HUGE,
        POISON_NAN,
    ]:
        # Decode and short prefill, single tile (T=1) and multi-tile (T=3).
        for vl in range(1, max_valid + 1):
            execute_tail_row_test[CausalMask, num_q_heads, poison](
                CausalMask(), cache_length=357, valid_length=vl, ctx=ctx
            )
        execute_tail_row_test[CausalMask, num_q_heads, poison](
            CausalMask(), cache_length=640, valid_length=max_valid, ctx=ctx
        )
        # Laguna's sliding layers.
        execute_tail_row_test[
            SlidingWindowCausalMask[512], num_q_heads, poison
        ](
            SlidingWindowCausalMask[512](),
            cache_length=640,
            valid_length=1,
            ctx=ctx,
        )
        execute_tail_row_test[
            SlidingWindowCausalMask[512], num_q_heads, poison
        ](
            SlidingWindowCausalMask[512](),
            cache_length=357,
            valid_length=max_valid,
            ctx=ctx,
        )
        # Cluster split-K (P=2) and page_size < BN.
        execute_tail_row_test[
            CausalMask, num_q_heads, poison, num_partitions=2
        ](CausalMask(), cache_length=1024, valid_length=1, ctx=ctx)
        execute_tail_row_test[CausalMask, num_q_heads, poison, page_size=64](
            CausalMask(), cache_length=536, valid_length=1, ctx=ctx
        )


def _run_boundary[num_q_heads: Int](ctx: DeviceContext, bm_eff: Int) raises:
    comptime for poison in [NO_POISON, POISON_NAN]:
        for vl in [bm_eff, bm_eff + 1, 32, 33]:
            execute_tail_row_test[CausalMask, num_q_heads, poison](
                CausalMask(), cache_length=357, valid_length=vl, ctx=ctx
            )
            execute_tail_row_test[
                SlidingWindowCausalMask[512], num_q_heads, poison
            ](
                SlidingWindowCausalMask[512](),
                cache_length=640,
                valid_length=vl,
                ctx=ctx,
            )
        # Cluster split-K (P=2) on both sides of the fused/unfused switch.
        for vl in [bm_eff, bm_eff + 1]:
            execute_tail_row_test[
                CausalMask, num_q_heads, poison, num_partitions=2
            ](CausalMask(), cache_length=1024, valid_length=vl, ctx=ctx)


def main() raises:
    with DeviceContext() as ctx:
        # group 9 (72 q heads): BM_eff = 3. group 6 (48 q heads): BM_eff = 5.
        _run_group[72](ctx, 3)
        _run_group[48](ctx, 5)
        _run_boundary[72](ctx, 3)
        _run_boundary[48](ctx, 5)
