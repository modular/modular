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
"""DeepSeek-V4's compressor over the windows a ragged chunk touches.

A compressed stream pools every ``ratio`` consecutive positions into one
entry. A chunk of ``s`` tokens after ``P`` cached ones touches
``ceil(s / ratio)`` windows, the first starting at ``P - P % ratio``; the
batch's windows are laid end to end, request ``b`` owning output rows
``window_offsets[b] .. window_offsets[b + 1]``.

Window ``w`` pools ``coff * ratio`` positions: its own ``ratio`` and, when
overlapping (``coff == 2``), its predecessor's. A position before ``P`` is read
from the state leaf, which keeps every position's raw ``wkv`` (K) and
``wgate`` (V) rows; one at or after it from the chunk's projections. A window
still open after the chunk is written as zeros without being pooled, and its
zone slot is left unwritten: it is unread until the window closes and a later
chunk writes it. With ``m`` running over the pooled positions, output channel
``c`` is

    entry[c] = sum_m softmax_m(wgate[m, c'] + ape[m % ratio, c']) * wkv[m, c']

where ``c' = c + (m // ratio) * head_dim`` when overlapping (the predecessor's
low half, then the window's own high half) and ``c' = c`` otherwise. The entry
is then finished as the reference finishes it: rounded to ``dtype``,
RMS-normed, its last ``rope_dim`` channels rotated at the window's first
position, and snapped to the grid the model was trained against. The
attention's entries take FP8 e4m3 on the non-rotary channels in groups of 64;
the indexer's are rotated by the Hadamard matrix and take FP4 e2m1 in groups
of 32. A group's scale is its absolute max over the format's max, rounded up
to a power of two (UE8M0).

The same launch does the stream's two cache writes. Each chunk row's raw
``wkv`` and ``wgate`` go into the state leaf at its position, and each entry
into the zone leaf at entry ``P // ratio + j`` of its request, beside its output
row. A window reads the state leaf only below ``P`` and the chunk's rows land
at or after it, so the row stores and the window reads never meet.

The window count is a sum of per-request ceilings, which the graph cannot
size an output by without reading it back. The output's row count is an upper
bound derived from shapes instead: the kernel reads the true count from
``window_offsets`` and leaves the rows past it, and their zone slots,
unwritten.

The first ``total_rows`` blocks store one chunk row each; the rest take one
window each, ``vec`` channels per thread. A window's positions are split over
``groups`` sets of threads so that more state-leaf loads are in flight, and
the sets' partial softmax sums are merged through shared memory. A rotary pair
stays in one thread and a quantization group in one warp, and the indexer's
entry fits one warp for the Hadamard butterflies, so only the norm reduces
across the block.
"""

from std.math import align_down, exp
from std.memory import unsafe_stack_allocation
from std.sys.info import align_of, simd_width_of
from std.utils.numerics import min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from max.gpu.sync import barrier

from internal_utils.fp8_utils import cast_saturating
from kv_cache.types import KVCacheT
from layout import TileTensor
from nn._ragged_utils import get_batch_from_row_offsets
from nn.attention.dsv4_quant import (
    FP4_GROUP,
    FP8_GROUP,
    fake_quantize,
    hadamard,
)
from nn.normalization import _rms_norm_warp_tiling_subkernel


def compress_windows_ragged_paged[
    dtype: DType,
    cache_t: KVCacheT,
    zone_t: KVCacheT,
    //,
    rotate: Bool,
    groups: Int,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    kv: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    score: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    ape: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    norm_weight: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    freqs: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    window_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    starts: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    kv_cache: cache_t,
    score_cache: cache_t,
    zone_cache: zone_t,
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    """Stores a ragged chunk's rows and pools every window it touches.

    Parameters:
        dtype: Element type of the entries and the norm weight (inferred).
        cache_t: The state leaf's cache type (inferred): one head of
            ``coff * head_dim`` float32 channels, paged by position.
        zone_t: The zone leaf's cache type (inferred): one head of
            ``head_dim`` channels, paged by entry.
        rotate: The indexer's entries: Hadamard-rotated and snapped to FP4,
            rather than FP8 on the non-rotary channels.
        groups: Thread sets a window's positions are split over; divides
            ``ratio``.

    Args:
        output: ``[rows, head_dim]`` entries. ``rows`` bounds the window
            count; the rows past it are left unwritten.
        kv: ``[total_rows, coff * head_dim]`` the chunk's raw ``wkv`` rows.
        score: Same shape, the chunk's raw ``wgate`` rows.
        ape: ``[ratio, coff * head_dim]`` per-slot gate bias.
        norm_weight: ``[head_dim]`` RMSNorm weight.
        freqs: ``[positions, rope_dim // 2, 2]`` rotary cos/sin table.
        input_row_offsets: ``[batch + 1]`` row offsets of ``kv`` and
            ``score``.
        window_offsets: ``[batch + 1]`` output row offsets, request ``b``
            owning ``ceil(len_b / ratio)`` rows.
        starts: ``[batch]`` tokens each request had cached before the chunk.
        kv_cache: The state leaf's K (``wkv``) at this layer.
        score_cache: The state leaf's V (``wgate``) at this layer.
        zone_cache: The zone leaf at this layer; window ``j`` of request
            ``b`` lands on entry ``starts[b] // ratio + j``.
        epsilon: RMSNorm epsilon.
        ctx: Device context used to enqueue the kernel.
    """
    comptime assert output.flat_rank == 2 and kv.flat_rank == 2
    comptime assert score.flat_rank == 2 and ape.flat_rank == 2
    comptime assert norm_weight.flat_rank == 1 and freqs.flat_rank == 3
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert window_offsets.flat_rank == 1 and starts.flat_rank == 1
    comptime ratio = ape.static_shape[0]
    comptime head_dim = output.static_shape[1]
    comptime rope_dim = 2 * freqs.static_shape[1]
    comptime assert (
        ratio > 0 and head_dim > 0 and rope_dim > 0
    ), "ape, output and freqs need static widths"
    comptime coff = ape.static_shape[1] // head_dim
    comptime proj_dim = coff * head_dim
    comptime assert (coff == 1 or coff == 2) and ape.static_shape[
        1
    ] == proj_dim, "ape must be [ratio, coff * head_dim], coff 1 or 2"
    comptime assert (
        cache_t.kv_params.num_heads == 1
        and cache_t.kv_params.head_size == proj_dim
    ), "the state leaf holds one head of the projection width"
    comptime assert (
        zone_t.kv_params.num_heads == 1
        and zone_t.kv_params.head_size == head_dim
    ), "the zone leaf holds one head of the entry width"

    comptime vec = min(
        simd_width_of[DType.float32, target=get_gpu_target()](),
        head_dim // WARP_SIZE,
    )
    comptime cols = head_dim // vec
    comptime threads = cols * groups
    comptime rope_start = head_dim - rope_dim
    comptime group = FP4_GROUP if rotate else FP8_GROUP
    comptime assert (
        vec % 2 == 0 and cols % WARP_SIZE == 0
    ), "a thread holds whole rotary pairs and a set whole warps"
    comptime assert rope_start % vec == 0, "a thread is rotary or not"
    comptime assert (
        group % vec == 0 and group // vec <= WARP_SIZE
    ), "a quantization group spans one warp at most"
    comptime assert (
        rotate or rope_start % group == 0
    ), "FP8 groups cover the non-rotary channels"
    comptime assert (
        not rotate or cols == WARP_SIZE
    ), "the Hadamard rotation runs within one warp"
    comptime assert groups > 0 and ratio % groups == 0
    comptime per_group = ratio // groups
    comptime chunk = min(per_group, 8)
    comptime assert per_group % chunk == 0

    # Kernel captures must be fixed-width.
    var total_rows = Int64(kv.dim[0]())
    var blocks = Int(total_rows) + Int(output.dim[0]())
    if blocks == 0:
        return

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(threads))
    )
    def kernel() {
        var output,
        var kv,
        var score,
        var ape,
        var norm_weight,
        var freqs,
        var input_row_offsets,
        var window_offsets,
        var starts,
        var kv_cache,
        var score_cache,
        var zone_cache,
        var epsilon,
        var total_rows,
    }:
        var tid = thread_idx.x
        var g = tid // cols
        var c = (tid % cols) * vec

        with PDL():
            var rows = Int(total_rows)
            if block_idx.x < rows:
                var t = block_idx.x
                var b = get_batch_from_row_offsets(input_row_offsets, t)
                var pos = (
                    Int(starts.load((b,)))
                    + t
                    - Int(input_row_offsets.load((b,)))
                )
                for col in range(tid * vec, proj_dim, threads * vec):
                    kv_cache.store(
                        b,
                        0,
                        pos,
                        col,
                        cast_saturating[cache_t.dtype](
                            kv.load[width=vec]((t, col))
                        ),
                    )
                    score_cache.store(
                        b,
                        0,
                        pos,
                        col,
                        cast_saturating[cache_t.dtype](
                            score.load[width=vec]((t, col))
                        ),
                    )
                return

            var w = block_idx.x - rows
            var batch = Int(window_offsets.dim[0]()) - 1
            if w >= Int(window_offsets.load((batch,))):
                return
            var b = get_batch_from_row_offsets(window_offsets, w)
            var first_row = Int(input_row_offsets.load((b,)))
            var last_row = Int(input_row_offsets.load((b + 1,))) - 1
            var cached = Int(starts.load((b,)))
            var start = align_down(cached, ratio) + ratio * (
                w - Int(window_offsets.load((b,)))
            )
            # A window still open after the chunk skips pooling and its zone
            # store: no query reads that slot until the window closes and a
            # later chunk writes it.
            if start + ratio > cached + last_row - first_row + 1:
                if g == 0:
                    output.store((w, c), SIMD[dtype, vec](0))
                return

            # Online softmax over this set's share of each half's positions,
            # `chunk` at a time.
            var run_max = SIMD[.float32, vec](min_or_neg_inf[DType.float32]())
            var den = SIMD[.float32, vec](0)
            var num = SIMD[.float32, vec](0)
            comptime for half in range(coff):
                var col = half * head_dim + c
                var first = start + (half + 1 - coff) * ratio
                # Windows are ratio-aligned, so one lies wholly before
                # position 0 (the first window's predecessor) or wholly after
                # it.
                if first >= 0:
                    for k0 in range(g * per_group, (g + 1) * per_group, chunk):
                        var values = Array[SIMD[.float32, vec], chunk](fill=0)
                        var logits = Array[SIMD[.float32, vec], chunk](fill=0)
                        comptime for i in range(chunk):
                            var pos = first + k0 + i
                            if pos < cached:
                                values[i] = kv_cache.load[
                                    width=vec, output_dtype=.float32
                                ](b, 0, pos, col)
                                logits[i] = score_cache.load[
                                    width=vec, output_dtype=.float32
                                ](b, 0, pos, col)
                            else:
                                var row = min(
                                    first_row + pos - cached, last_row
                                )
                                values[i] = kv.load[width=vec]((row, col))
                                logits[i] = score.load[width=vec]((row, col))
                            logits[i] += ape.load[width=vec]((k0 + i, col))
                        var chunk_max = logits[0]
                        comptime for i in range(1, chunk):
                            chunk_max = max(chunk_max, logits[i])
                        var new_max = max(run_max, chunk_max)
                        var rescale = exp(run_max - new_max)
                        den *= rescale
                        num *= rescale
                        comptime for i in range(chunk):
                            var e = exp(logits[i] - new_max)
                            den += e
                            num += e * values[i]
                        run_max = new_max

            comptime if groups > 1:
                # Every set saw its share of the window's own half, so every
                # partial max is finite.
                comptime align = align_of[SIMD[.float32, vec]]()
                var maxes = unsafe_stack_allocation[
                    groups * head_dim,
                    Float32,
                    alignment=align,
                    address_space=.SHARED,
                ]()
                var dens = unsafe_stack_allocation[
                    groups * head_dim,
                    Float32,
                    alignment=align,
                    address_space=.SHARED,
                ]()
                var nums = unsafe_stack_allocation[
                    groups * head_dim,
                    Float32,
                    alignment=align,
                    address_space=.SHARED,
                ]()
                maxes.unsafe_store(g * head_dim + c, run_max)
                dens.unsafe_store(g * head_dim + c, den)
                nums.unsafe_store(g * head_dim + c, num)
                barrier()
                if g == 0:
                    comptime for h in range(1, groups):
                        run_max = max(
                            run_max,
                            maxes.unsafe_load[width=vec](h * head_dim + c),
                        )
                    den = 0
                    num = 0
                    comptime for h in range(groups):
                        var f = exp(
                            maxes.unsafe_load[width=vec](h * head_dim + c)
                            - run_max
                        )
                        den += dens.unsafe_load[width=vec](h * head_dim + c) * f
                        num += nums.unsafe_load[width=vec](h * head_dim + c) * f

            # The reference norms the entry after rounding it to `dtype`. Every
            # set joins the block reduction; only the first contributes.
            var live = g == 0
            var entry = SIMD[.float32, vec](0)
            if live:
                entry = (num / den).cast[dtype]().cast[.float32]()
            var normed = _rms_norm_warp_tiling_subkernel[
                threads // WARP_SIZE, multiply_before_cast=True
            ](
                w,
                c + g * head_dim,
                entry,
                norm_weight.load[width=vec]((c,)),
                epsilon,
                Float32(0),
                head_dim,
            )
            if not live:
                return
            var y = normed.cast[.float32]()

            if c >= rope_start:
                comptime for p in range(0, vec, 2):
                    var cos_sin = freqs.load[width=2](
                        (start, (c - rope_start + p) // 2, 0)
                    )
                    var re = y[p]
                    var im = y[p + 1]
                    y[p] = re * cos_sin[0] - im * cos_sin[1]
                    y[p + 1] = re * cos_sin[1] + im * cos_sin[0]
                y = y.cast[dtype]().cast[.float32]()

            comptime if rotate:
                y = hadamard[head_dim](y).cast[dtype]().cast[.float32]()
                y = fake_quantize[group, fp4=True](y)
            else:
                # Every lane joins the group reduction; the rotary ones drop it.
                var q = fake_quantize[group, fp4=False](y)
                if c < rope_start:
                    y = q
            var out = y.cast[dtype]()
            output.store((w, c), out)
            zone_cache.store(
                b, 0, start // ratio, c, cast_saturating[zone_t.dtype](out)
            )

    ctx.enqueue_closure(
        kernel,
        grid_dim=blocks,
        block_dim=threads,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
