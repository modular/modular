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
"""PDL ordering tests for the grouped NVFP4 quantize kernels.

The grouped quantize and its per-row-scaled variant launch with programmatic
dependent launch, so they can become resident while the kernel before them is
still running, and they release the kernel after them right after their own
wait. Two orderings matter:

1. Their routing operands (`row_offsets`, `scales_offsets`, `expert_ids`,
   `sf_tensor`) are often written by the kernel before them: a MoE layer
   derives the offsets from `moe_create_indices` and, with per-expert
   activation scales, computes `sf_tensor = 1 / input_scale` right before the
   quantize. The kernels must not read any of them before
   `wait_on_dependent_grids()`. Here a producer kernel releases its dependents
   on entry, sleeps, and only then publishes the real operands over zeroed
   ones. A quantize that reads before its wait sees every expert empty and
   leaves the output at its sentinel; on the real model the stale table
   instead sent blocks past the input rows (CUDA_ERROR_ILLEGAL_ADDRESS) or
   quantized against a stale scale. The PDL output must match a run whose
   operands were published first.
2. A kernel after the quantize must see every finished store. Two quantizes
   with different inputs write the same buffers and a PDL-launched copy then
   reads them back to front, so a copy that ran ahead of the second quantize
   would pick up the first one's values and not match the reference.

3. The quantize releases its dependents right after its own wait, not when
   its blocks exit. A PDL-launched probe grid reads the clock before and after
   its own wait: it becomes resident while the multi-wave quantize is still
   running, so the gap between the two reads is the quantize's remaining
   run time (about 30 us here), where a release at block exit leaves a gap
   under 5 us. A quarter of the launches must show a long gap.

Orderings 1 and 2 run over ragged and empty experts and check that every scale slot of an
allocated tile is written (padding rows included) and that the spare tiles
past the last expert are left alone.
"""

from std.math import ceildiv
from std.memory.alloc import unsafe_alloc
from std.sys._assembly import inlined_assembly
from std.time import sleep

from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from linalg.block_scaled_quantization import (
    grouped_quantize_dynamic_scaled_fp4_async,
    grouped_quantize_nvfp4_row_amax,
)
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from std.testing import assert_equal, assert_true

comptime dtype = DType.bfloat16
comptime scales_dtype = NVFP4_SF_DTYPE
comptime out_dtype = DType.uint8
comptime SENTINEL = 0xEE
# Tiles allocated past the last expert's, which the kernels must not touch.
comptime SPARE_TILES = 2


def _mut[
    dt: DType
](buf: DeviceBuffer[dt]) -> MutPointer[Scalar[dt], MutAnyOrigin]:
    """Gives the launch a writable view of a buffer passed by reference."""
    return (
        buf.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutAnyOrigin]()
    )


def publish_after_delay[
    num_experts: Int
](
    row_offsets: MutPointer[UInt32, MutUntrackedOrigin],
    scales_offsets: MutPointer[UInt32, MutUntrackedOrigin],
    expert_ids: MutPointer[Int32, MutUntrackedOrigin],
    sf_tensor: MutPointer[Float32, MutUntrackedOrigin],
    src_row_offsets: MutPointer[UInt32, MutUntrackedOrigin],
    src_scales_offsets: MutPointer[UInt32, MutUntrackedOrigin],
    src_expert_ids: MutPointer[Int32, MutUntrackedOrigin],
    src_sf_tensor: MutPointer[Float32, MutUntrackedOrigin],
):
    """Releases dependents at once, then publishes the operands late."""
    launch_dependent_grids()
    sleep(0.05)
    for i in range(num_experts + 1):
        row_offsets[i] = src_row_offsets[i]
    for i in range(num_experts):
        scales_offsets[i] = src_scales_offsets[i]
        expert_ids[i] = src_expert_ids[i]
        sf_tensor[i] = src_sf_tensor[i]


def copy_back_to_front_after_wait(
    out_src: MutPointer[Scalar[out_dtype], MutUntrackedOrigin],
    out_dst: MutPointer[Scalar[out_dtype], MutUntrackedOrigin],
    out_n: Int32,
    scales_src: MutPointer[Scalar[scales_dtype], MutUntrackedOrigin],
    scales_dst: MutPointer[Scalar[scales_dtype], MutUntrackedOrigin],
    scales_n: Int32,
    rs_src: MutPointer[Scalar[DType.bfloat16], MutUntrackedOrigin],
    rs_dst: MutPointer[Scalar[DType.bfloat16], MutUntrackedOrigin],
    rs_n: Int32,
):
    """A dependent grid: copies all three outputs, last element first.

    The quantize blocks that run last write the highest columns of the last
    rows, so reading from the end reaches the data most likely to still be in
    flight if this grid ran ahead of its predecessor. One grid copies every
    output behind one wait, so each is read directly behind the quantize and
    none is ordered after the quantize by an earlier copy.
    """
    wait_on_dependent_grids()
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    var i = Int(out_n) - 1 - tid
    while i >= 0:
        out_dst[i] = out_src[i]
        i -= stride
    i = Int(scales_n) - 1 - tid
    while i >= 0:
        scales_dst[i] = scales_src[i]
        i -= stride
    i = Int(rs_n) - 1 - tid
    while i >= 0:
        rs_dst[i] = rs_src[i]
        i -= stride


struct Routing(Movable):
    """Host-side routing tables of one ragged expert layout."""

    var row_offsets: MutPointer[UInt32, MutUntrackedOrigin]
    var scales_offsets: MutPointer[UInt32, MutUntrackedOrigin]
    var expert_ids: MutPointer[Int32, MutUntrackedOrigin]
    var sf_tensor: MutPointer[Float32, MutUntrackedOrigin]
    var total_tokens: Int
    var total_m_tiles: Int

    def __init__(out self, counts: List[Int]):
        var n = len(counts)
        self.row_offsets = unsafe_alloc[UInt32](n + 1)
        self.scales_offsets = unsafe_alloc[UInt32](n)
        self.expert_ids = unsafe_alloc[Int32](n)
        self.sf_tensor = unsafe_alloc[Float32](n)
        var tile_start = 0
        self.row_offsets[0] = 0
        for i in range(n):
            self.row_offsets[i + 1] = self.row_offsets[i] + UInt32(counts[i])
            self.scales_offsets[i] = UInt32(
                tile_start - Int(self.row_offsets[i]) // SF_MN_GROUP_SIZE
            )
            tile_start += ceildiv(counts[i], SF_MN_GROUP_SIZE)
            self.expert_ids[i] = Int32(i)
            self.sf_tensor[i] = Float32(1.5 + 0.5 * Float64(i))
        self.total_tokens = Int(self.row_offsets[n])
        self.total_m_tiles = tile_start

    def free(self):
        self.row_offsets.unsafe_free()
        self.scales_offsets.unsafe_free()
        self.expert_ids.unsafe_free()
        self.sf_tensor.unsafe_free()


def _quantize[
    row_amax: Bool, N: Int, num_experts: Int
](
    ctx: DeviceContext,
    out_buf: DeviceBuffer[out_dtype],
    scales_buf: DeviceBuffer[scales_dtype],
    row_scales_buf: DeviceBuffer[DType.bfloat16],
    input_buf: DeviceBuffer[dtype],
    row_offsets: DeviceBuffer[DType.uint32],
    scales_offsets: DeviceBuffer[DType.uint32],
    expert_ids: DeviceBuffer[DType.int32],
    sf_tensor: DeviceBuffer[DType.float32],
    total_tokens: Int,
    total_m_tiles: Int,
) raises:
    """Launches the grouped quantize, or the per-row-scaled one."""
    comptime K_tiles = ceildiv(N, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    var scales_shape = Coord(
        total_m_tiles + SPARE_TILES,
        Idx[K_tiles],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var out_t = TileTensor(_mut(out_buf), row_major(total_tokens, Idx[N // 2]))
    var scales_t = TileTensor(_mut(scales_buf), row_major(scales_shape))
    var input_t = TileTensor(input_buf, row_major(total_tokens, Idx[N]))
    var row_offsets_t = TileTensor(row_offsets, row_major(Idx[num_experts + 1]))
    var scales_offsets_t = TileTensor(
        scales_offsets, row_major(Idx[num_experts])
    )
    comptime if row_amax:
        grouped_quantize_nvfp4_row_amax(
            out_t,
            scales_t,
            TileTensor(_mut(row_scales_buf), row_major(total_tokens)),
            input_t,
            row_offsets_t,
            scales_offsets_t,
            ctx,
        )
    else:
        grouped_quantize_dynamic_scaled_fp4_async(
            out_t,
            scales_t,
            input_t,
            row_offsets_t,
            scales_offsets_t,
            TileTensor(expert_ids, row_major(Idx[num_experts])),
            TileTensor(sf_tensor, row_major(Idx[num_experts])),
            ctx,
        )


def _random_input(
    ctx: DeviceContext, rows: Int, N: Int
) raises -> DeviceBuffer[dtype]:
    var host = unsafe_alloc[Scalar[dtype]](rows * N)
    random(TileTensor(host, row_major(rows, N)), min=-1.0, max=1.0)
    var dev = ctx.enqueue_create_buffer[dtype](rows * N)
    ctx.enqueue_copy(dev, host)
    ctx.synchronize()
    host.unsafe_free()
    return dev^


def _fill_sentinels(
    ctx: DeviceContext,
    out_buf: DeviceBuffer[out_dtype],
    scales_buf: DeviceBuffer[scales_dtype],
    row_scales_buf: DeviceBuffer[DType.bfloat16],
    out_bytes: Int,
    total_scales: Int,
    total_tokens: Int,
) raises:
    var out_host = unsafe_alloc[Scalar[out_dtype]](out_bytes)
    for i in range(out_bytes):
        out_host[i] = SENTINEL
    var scales_host = unsafe_alloc[Scalar[scales_dtype]](total_scales)
    for i in range(total_scales):
        scales_host[i] = Scalar[scales_dtype](-1.0)
    var row_scales_host = unsafe_alloc[Scalar[DType.bfloat16]](total_tokens)
    for i in range(total_tokens):
        row_scales_host[i] = Scalar[DType.bfloat16](-3.0)
    ctx.enqueue_copy(out_buf, out_host)
    ctx.enqueue_copy(scales_buf, scales_host)
    ctx.enqueue_copy(row_scales_buf, row_scales_host)
    ctx.synchronize()
    out_host.unsafe_free()
    scales_host.unsafe_free()
    row_scales_host.unsafe_free()


def _check_against_reference[
    row_amax: Bool
](
    ctx: DeviceContext,
    r: Routing,
    N: Int,
    out_buf: DeviceBuffer[out_dtype],
    scales_buf: DeviceBuffer[scales_dtype],
    row_scales_buf: DeviceBuffer[DType.bfloat16],
    ref_out: DeviceBuffer[out_dtype],
    ref_scales: DeviceBuffer[scales_dtype],
    ref_row_scales: DeviceBuffer[DType.bfloat16],
    msg: String,
) raises:
    """Compares a run with its reference and checks which slots were written."""
    var K_tiles = ceildiv(N, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
    var per_tile = K_tiles * SF_MN_GROUP_SIZE * SF_ATOM_K
    var out_bytes = r.total_tokens * (N // 2)
    var total_scales = (r.total_m_tiles + SPARE_TILES) * per_tile
    var h_out = unsafe_alloc[Scalar[out_dtype]](out_bytes)
    var h_ref_out = unsafe_alloc[Scalar[out_dtype]](out_bytes)
    var h_scales = unsafe_alloc[Scalar[scales_dtype]](total_scales)
    var h_ref_scales = unsafe_alloc[Scalar[scales_dtype]](total_scales)
    var h_rs = unsafe_alloc[Scalar[DType.bfloat16]](r.total_tokens)
    var h_ref_rs = unsafe_alloc[Scalar[DType.bfloat16]](r.total_tokens)
    ctx.enqueue_copy(h_out, out_buf)
    ctx.enqueue_copy(h_ref_out, ref_out)
    ctx.enqueue_copy(h_scales, scales_buf)
    ctx.enqueue_copy(h_ref_scales, ref_scales)
    ctx.enqueue_copy(h_rs, row_scales_buf)
    ctx.enqueue_copy(h_ref_rs, ref_row_scales)
    ctx.synchronize()

    var mismatched = 0
    for i in range(out_bytes):
        if h_out[i] != h_ref_out[i]:
            mismatched += 1
    for i in range(total_scales):
        if (
            h_scales[i].cast[DType.float32]()
            != h_ref_scales[i].cast[DType.float32]()
        ):
            mismatched += 1
    comptime if row_amax:
        for i in range(r.total_tokens):
            if (
                h_rs[i].cast[DType.float32]()
                != h_ref_rs[i].cast[DType.float32]()
            ):
                mismatched += 1
    assert_equal(mismatched, 0, msg=msg)

    # Every slot of an allocated tile is written, padding rows included, and
    # the spare tiles past the last expert's stay at their sentinel.
    var unwritten = 0
    var touched_spare = 0
    for i in range(total_scales):
        var is_sentinel = h_scales[i].cast[DType.float32]() == -1.0
        if i < r.total_m_tiles * per_tile:
            if is_sentinel:
                unwritten += 1
        elif not is_sentinel:
            touched_spare += 1
    assert_equal(unwritten, 0, msg="a scale slot of an allocated tile is unset")
    assert_equal(touched_spare, 0, msg="a spare scale tile was written")

    h_out.unsafe_free()
    h_ref_out.unsafe_free()
    h_scales.unsafe_free()
    h_ref_scales.unsafe_free()
    h_rs.unsafe_free()
    h_ref_rs.unsafe_free()


def test_operands_from_predecessor_kernel[
    row_amax: Bool
](ctx: DeviceContext) raises:
    comptime N = 256
    comptime num_experts = 4

    # An empty expert and a two-tile expert, as a decode/prefill mix routes.
    var counts: List[Int] = [3, 0, 5, 130]
    var r = Routing(counts)
    var out_bytes = r.total_tokens * (N // 2)
    var per_tile = (
        ceildiv(N, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_MN_GROUP_SIZE
        * SF_ATOM_K
    )
    var total_scales = (r.total_m_tiles + SPARE_TILES) * per_tile
    var dev_input = _random_input(ctx, r.total_tokens, N)

    # The published operands: staging copies (the producer's source) and
    # the buffers the quantize reads, zeroed until the producer publishes.
    var src_row_offsets = ctx.enqueue_create_buffer[DType.uint32](
        num_experts + 1
    )
    var src_scales_offsets = ctx.enqueue_create_buffer[DType.uint32](
        num_experts
    )
    var src_expert_ids = ctx.enqueue_create_buffer[DType.int32](num_experts)
    var src_sf_tensor = ctx.enqueue_create_buffer[DType.float32](num_experts)
    ctx.enqueue_copy(src_row_offsets, r.row_offsets)
    ctx.enqueue_copy(src_scales_offsets, r.scales_offsets)
    ctx.enqueue_copy(src_expert_ids, r.expert_ids)
    ctx.enqueue_copy(src_sf_tensor, r.sf_tensor)

    var ref_out = ctx.enqueue_create_buffer[out_dtype](out_bytes)
    var ref_scales = ctx.enqueue_create_buffer[scales_dtype](total_scales)
    var ref_rs = ctx.enqueue_create_buffer[DType.bfloat16](r.total_tokens)
    _fill_sentinels(
        ctx,
        ref_out,
        ref_scales,
        ref_rs,
        out_bytes,
        total_scales,
        r.total_tokens,
    )
    # Reference: the operands are already in place when the quantize runs.
    _quantize[row_amax, N, num_experts](
        ctx,
        ref_out,
        ref_scales,
        ref_rs,
        dev_input,
        src_row_offsets,
        src_scales_offsets,
        src_expert_ids,
        src_sf_tensor,
        r.total_tokens,
        r.total_m_tiles,
    )
    ctx.synchronize()

    var row_offsets = ctx.enqueue_create_buffer[DType.uint32](num_experts + 1)
    var scales_offsets = ctx.enqueue_create_buffer[DType.uint32](num_experts)
    var expert_ids = ctx.enqueue_create_buffer[DType.int32](num_experts)
    var sf_tensor = ctx.enqueue_create_buffer[DType.float32](num_experts)
    ctx.enqueue_memset(row_offsets, 0)
    ctx.enqueue_memset(scales_offsets, 0)
    ctx.enqueue_memset(expert_ids, 0)
    ctx.enqueue_memset(sf_tensor, 0)
    var output = ctx.enqueue_create_buffer[out_dtype](out_bytes)
    var scales = ctx.enqueue_create_buffer[scales_dtype](total_scales)
    var rs = ctx.enqueue_create_buffer[DType.bfloat16](r.total_tokens)
    _fill_sentinels(
        ctx, output, scales, rs, out_bytes, total_scales, r.total_tokens
    )

    ctx.enqueue_function[publish_after_delay[num_experts]](
        row_offsets,
        scales_offsets,
        expert_ids,
        sf_tensor,
        src_row_offsets,
        src_scales_offsets,
        src_expert_ids,
        src_sf_tensor,
        grid_dim=1,
        block_dim=1,
    )
    _quantize[row_amax, N, num_experts](
        ctx,
        output,
        scales,
        rs,
        dev_input,
        row_offsets,
        scales_offsets,
        expert_ids,
        sf_tensor,
        r.total_tokens,
        r.total_m_tiles,
    )
    ctx.synchronize()

    _check_against_reference[row_amax](
        ctx,
        r,
        N,
        output,
        scales,
        rs,
        ref_out,
        ref_scales,
        ref_rs,
        (
            "quantize read its routing operands before the producer kernel"
            " published them"
        ),
    )
    r.free()


def test_dependent_sees_finished_quantize[
    row_amax: Bool
](ctx: DeviceContext) raises:
    """Two quantizes with different inputs, then a PDL copy of their output."""
    comptime N = 7168
    comptime num_experts = 32
    comptime iterations = 8

    # Ragged and empty experts, several waves of blocks in all.
    var counts = List[Int]()
    for i in range(num_experts):
        counts.append(0 if i % 9 == 4 else 70 + (i * 53) % 330)
    var r = Routing(counts)
    var out_bytes = r.total_tokens * (N // 2)
    var per_tile = (
        ceildiv(N, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_MN_GROUP_SIZE
        * SF_ATOM_K
    )
    var total_scales = (r.total_m_tiles + SPARE_TILES) * per_tile

    var input_a = _random_input(ctx, r.total_tokens, N)
    var input_b = _random_input(ctx, r.total_tokens, N)
    var row_offsets = ctx.enqueue_create_buffer[DType.uint32](num_experts + 1)
    var scales_offsets = ctx.enqueue_create_buffer[DType.uint32](num_experts)
    var expert_ids = ctx.enqueue_create_buffer[DType.int32](num_experts)
    var sf_tensor = ctx.enqueue_create_buffer[DType.float32](num_experts)
    ctx.enqueue_copy(row_offsets, r.row_offsets)
    ctx.enqueue_copy(scales_offsets, r.scales_offsets)
    ctx.enqueue_copy(expert_ids, r.expert_ids)
    ctx.enqueue_copy(sf_tensor, r.sf_tensor)

    var ref_out = ctx.enqueue_create_buffer[out_dtype](out_bytes)
    var ref_scales = ctx.enqueue_create_buffer[scales_dtype](total_scales)
    var ref_rs = ctx.enqueue_create_buffer[DType.bfloat16](r.total_tokens)
    _fill_sentinels(
        ctx,
        ref_out,
        ref_scales,
        ref_rs,
        out_bytes,
        total_scales,
        r.total_tokens,
    )
    _quantize[row_amax, N, num_experts](
        ctx,
        ref_out,
        ref_scales,
        ref_rs,
        input_b,
        row_offsets,
        scales_offsets,
        expert_ids,
        sf_tensor,
        r.total_tokens,
        r.total_m_tiles,
    )
    ctx.synchronize()

    var output = ctx.enqueue_create_buffer[out_dtype](out_bytes)
    var scales = ctx.enqueue_create_buffer[scales_dtype](total_scales)
    var rs = ctx.enqueue_create_buffer[DType.bfloat16](r.total_tokens)
    var out_copy = ctx.enqueue_create_buffer[out_dtype](out_bytes)
    var scales_copy = ctx.enqueue_create_buffer[scales_dtype](total_scales)
    var rs_copy = ctx.enqueue_create_buffer[DType.bfloat16](r.total_tokens)

    for it in range(iterations):
        _fill_sentinels(
            ctx, output, scales, rs, out_bytes, total_scales, r.total_tokens
        )
        _fill_sentinels(
            ctx,
            out_copy,
            scales_copy,
            rs_copy,
            out_bytes,
            total_scales,
            r.total_tokens,
        )
        _quantize[row_amax, N, num_experts](
            ctx,
            output,
            scales,
            rs,
            input_a,
            row_offsets,
            scales_offsets,
            expert_ids,
            sf_tensor,
            r.total_tokens,
            r.total_m_tiles,
        )
        _quantize[row_amax, N, num_experts](
            ctx,
            output,
            scales,
            rs,
            input_b,
            row_offsets,
            scales_offsets,
            expert_ids,
            sf_tensor,
            r.total_tokens,
            r.total_m_tiles,
        )
        ctx.enqueue_function[copy_back_to_front_after_wait](
            output,
            out_copy,
            Int32(out_bytes),
            scales,
            scales_copy,
            Int32(total_scales),
            rs,
            rs_copy,
            Int32(r.total_tokens),
            grid_dim=512,
            block_dim=256,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        ctx.synchronize()
        _check_against_reference[row_amax](
            ctx,
            r,
            N,
            out_copy,
            scales_copy,
            rs_copy,
            ref_out,
            ref_scales,
            ref_rs,
            String(
                (
                    "a dependent grid read the quantize output before it"
                    " finished (iteration "
                ),
                it,
                ")",
            ),
        )
    r.free()


@always_inline
def _globaltimer() -> UInt64:
    """Reads the nanosecond global timer; volatile, so it cannot be moved past
    the wait."""
    return inlined_assembly[
        "mov.u64 $0, %globaltimer;", UInt64, constraints="=l"
    ]()


def stamp_wait_gap(gaps: MutPointer[UInt64, MutUntrackedOrigin], slot: Int32):
    """A one-thread dependent grid: times how long its own wait blocks.

    A single thread, so no other lane of its warp waits ahead of the first
    clock read.
    """
    var before = _globaltimer()
    wait_on_dependent_grids()
    gaps[Int(slot)] = _globaltimer() - before


def test_dependents_released_after_the_wait[
    row_amax: Bool
](ctx: DeviceContext) raises:
    """A dependent grid is resident while the quantize is still running."""
    comptime N = 7168
    comptime num_experts = 32
    comptime iterations = 24
    # The gap is the last blocks' run time with a release after the wait (about
    # 30 us on a B200) and a few us at most when the release is at block exit.
    # An early release permits the dependent to become resident but does not
    # force it to, so a busy GPU can delay any one launch past the quantize.
    # The test therefore asks for a quarter of the launches in a round to show
    # a long gap, and only fails if every round misses: the release at block
    # exit shows no long gap in any round, since a launch cannot be resident
    # before the quantize is done.
    comptime long_gap_ns = 10_000
    comptime min_long_gaps = (iterations - 1) // 4
    comptime max_rounds = 5

    var counts = List[Int]()
    for i in range(num_experts):
        counts.append(0 if i % 9 == 4 else 70 + (i * 53) % 330)
    var r = Routing(counts)
    var out_bytes = r.total_tokens * (N // 2)
    var per_tile = (
        ceildiv(N, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_MN_GROUP_SIZE
        * SF_ATOM_K
    )
    var total_scales = (r.total_m_tiles + SPARE_TILES) * per_tile
    var input = _random_input(ctx, r.total_tokens, N)
    var row_offsets = ctx.enqueue_create_buffer[DType.uint32](num_experts + 1)
    var scales_offsets = ctx.enqueue_create_buffer[DType.uint32](num_experts)
    var expert_ids = ctx.enqueue_create_buffer[DType.int32](num_experts)
    var sf_tensor = ctx.enqueue_create_buffer[DType.float32](num_experts)
    ctx.enqueue_copy(row_offsets, r.row_offsets)
    ctx.enqueue_copy(scales_offsets, r.scales_offsets)
    ctx.enqueue_copy(expert_ids, r.expert_ids)
    ctx.enqueue_copy(sf_tensor, r.sf_tensor)
    var output = ctx.enqueue_create_buffer[out_dtype](out_bytes)
    var scales = ctx.enqueue_create_buffer[scales_dtype](total_scales)
    var rs = ctx.enqueue_create_buffer[DType.bfloat16](r.total_tokens)
    var gaps = ctx.enqueue_create_buffer[DType.uint64](iterations)
    ctx.synchronize()

    var long_gaps = 0
    for _ in range(max_rounds):
        for it in range(iterations):
            _quantize[row_amax, N, num_experts](
                ctx,
                output,
                scales,
                rs,
                input,
                row_offsets,
                scales_offsets,
                expert_ids,
                sf_tensor,
                r.total_tokens,
                r.total_m_tiles,
            )
            ctx.enqueue_function[stamp_wait_gap](
                gaps,
                Int32(it),
                grid_dim=1,
                block_dim=1,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
        var host_gaps = ctx.enqueue_create_host_buffer[DType.uint64](iterations)
        ctx.enqueue_copy(host_gaps, gaps)
        ctx.synchronize()

        # Skip the first launch, which pays for module load.
        long_gaps = 0
        for it in range(1, iterations):
            if Int(host_gaps[it]) > long_gap_ns:
                long_gaps += 1
        if long_gaps >= min_long_gaps:
            break
    assert_true(
        long_gaps >= min_long_gaps,
        msg=String(
            "the quantize released its dependents late: in each of ",
            max_rounds,
            " rounds, fewer than ",
            min_long_gaps,
            " of ",
            iterations - 1,
            " dependent grids were resident for over ",
            long_gap_ns,
            " ns before their wait returned (last round: ",
            long_gaps,
            ", row_amax=",
            row_amax,
            ")",
        ),
    )
    r.free()


def main() raises:
    with DeviceContext() as ctx:
        test_operands_from_predecessor_kernel[False](ctx)
        test_operands_from_predecessor_kernel[True](ctx)
        test_dependent_sees_finished_quantize[False](ctx)
        test_dependent_sees_finished_quantize[True](ctx)
        test_dependents_released_after_the_wait[False](ctx)
        test_dependents_released_after_the_wait[True](ctx)
