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
``wgate`` (V) rows; one at or after it from the chunk's projections, clamped to
the chunk's last row so a window still open reads finite rows. With ``m``
running over the pooled positions, output channel ``c`` is

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

The window count is a sum of per-request ceilings, which the graph cannot
size an output by without reading it back. The output's row count is an upper
bound derived from shapes instead: the kernel reads the true count from
``window_offsets`` and leaves the rows past it unwritten, and the ragged store
that consumes the entries skips them.

One block per window, ``vec`` channels per thread. A rotary pair stays in one
thread and a quantization group in one warp, and the indexer's entry fits one
warp for the Hadamard butterflies, so only the norm reduces across the block.
"""

from std.bit import log2_floor
from std.math import align_down, exp, sqrt
from std.sys.info import simd_width_of
from std.utils.numerics import min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
)
from max.gpu.host import DeviceContext, get_gpu_target
import max.gpu.primitives.warp as warp

from kv_cache.types import KVCacheT
from layout import TileTensor
from linalg.fp4_utils import cast_fp_to_fp4e2m1
from nn._ragged_utils import get_batch_from_row_offsets
from nn.normalization import _rms_norm_warp_tiling_subkernel

# The reference's activation quantizers (`act_quant` and `fp4_act_quant` in
# `inference/kernel.py`): group width, format max and the floor under a
# group's absolute max.
comptime _FP8_GROUP = 64
comptime _FP8_MAX: Float32 = 448.0
comptime _FP8_AMAX_FLOOR: Float32 = 1e-4
comptime _FP4_GROUP = 32
comptime _FP4_MAX: Float32 = 6.0
comptime _FP4_AMAX_FLOOR = Float32(6.0) * Float32(1.1754943508222875e-38)
"""Six times the smallest normal float32."""


@inline(.always)
def _fake_quantize[
    vec: Int, //, group: Int, fp4: Bool
](x: SIMD[.float32, vec]) -> type_of(x):
    """Snaps ``x`` to the FP4 or FP8 grid and back, a scale per group.

    A group of ``group`` channels spans ``group // vec`` consecutive lanes,
    every one of which must call this.
    """
    comptime value_max = _FP4_MAX if fp4 else _FP8_MAX
    comptime amax_floor = _FP4_AMAX_FLOOR if fp4 else _FP8_AMAX_FLOOR
    # The reference multiplies by the reciprocal rounded to float32.
    comptime inv_max = Float32(1.0 / Float64(value_max))
    var amax = warp.lane_group_max[num_lanes=group // vec](abs(x).reduce_max())
    var scale = (
        (max(amax, amax_floor) * inv_max)
        .cast[.float8_e8m0fnu]()
        .cast[.float32]()
    )
    var q = (x / scale).clamp(-value_max, value_max)
    comptime if fp4:
        q = cast_fp_to_fp4e2m1(q)
    else:
        q = q.cast[.float8_e4m3fn]().cast[.float32]()
    return q * scale


@inline(.always)
def _hadamard[vec: Int, //, width: Int](x: SIMD[.float32, vec]) -> type_of(x):
    """``x`` times the orthonormal Sylvester Hadamard matrix of order ``width``.

    The row is spread over one warp, ``vec`` consecutive channels per lane.
    """
    var y = x
    comptime for level in range(log2_floor(width)):
        comptime stride = 1 << level
        comptime if stride < vec:
            var z = y
            comptime for i in range(vec):
                comptime if i & stride == 0:
                    z[i] = y[i] + y[i + stride]
                else:
                    z[i] = y[i - stride] - y[i]
            y = z
        else:
            comptime lanes = stride // vec
            var other = type_of(y)()
            comptime for i in range(vec):
                other[i] = warp.shuffle_xor(y[i], UInt32(lanes))
            y = other - y if (lane_id() & lanes) != 0 else y + other
    comptime scale = Float32(1.0 / sqrt(Float64(width)))
    return y * scale


def compress_windows_ragged_paged[
    dtype: DType,
    cache_t: KVCacheT,
    //,
    rotate: Bool,
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
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    """Pools every window of a ragged chunk into its compressed entry.

    Parameters:
        dtype: Element type of the entries and the norm weight (inferred).
        cache_t: The state leaf's cache type (inferred): one head of
            ``coff * head_dim`` float32 channels, paged by position.
        rotate: The indexer's entries: Hadamard-rotated and snapped to FP4,
            rather than FP8 on the non-rotary channels.

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
    comptime assert (coff == 1 or coff == 2) and ape.static_shape[
        1
    ] == coff * head_dim, "ape must be [ratio, coff * head_dim], coff 1 or 2"
    comptime assert (
        cache_t.kv_params.num_heads == 1
        and cache_t.kv_params.head_size == coff * head_dim
    ), "the state leaf holds one head of the projection width"

    comptime vec = min(
        simd_width_of[DType.float32, target=get_gpu_target()](),
        head_dim // WARP_SIZE,
    )
    comptime threads = head_dim // vec
    comptime rope_start = head_dim - rope_dim
    comptime group = _FP4_GROUP if rotate else _FP8_GROUP
    comptime assert (
        vec % 2 == 0 and threads % WARP_SIZE == 0
    ), "a thread holds whole rotary pairs and the block whole warps"
    comptime assert rope_start % vec == 0, "a thread is rotary or not"
    comptime assert (
        group % vec == 0 and group // vec <= WARP_SIZE
    ), "a quantization group spans one warp at most"
    comptime assert (
        rotate or rope_start % group == 0
    ), "FP8 groups cover the non-rotary channels"
    comptime assert (
        not rotate or threads == WARP_SIZE
    ), "the Hadamard rotation runs within one warp"
    comptime chunk = min(ratio, 8)
    comptime assert ratio % chunk == 0

    var rows = Int(output.dim[0]())
    if rows == 0:
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
        var epsilon,
    }:
        var w = block_idx.x
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
        var c = thread_idx.x * vec

        # Online softmax over the window's positions, `chunk` at a time.
        var run_max = SIMD[.float32, vec](min_or_neg_inf[DType.float32]())
        var den = SIMD[.float32, vec](0)
        var num = SIMD[.float32, vec](0)
        comptime for half in range(coff):
            var col = half * head_dim + c
            var first = start + (half + 1 - coff) * ratio
            # Windows are ratio-aligned, so one lies wholly before position 0
            # (the first window's predecessor) or wholly after it.
            if first >= 0:
                for k0 in range(0, ratio, chunk):
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
                            var row = min(first_row + pos - cached, last_row)
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

        # The reference norms the entry after rounding it to `dtype`.
        var normed = _rms_norm_warp_tiling_subkernel[
            threads // WARP_SIZE, multiply_before_cast=True
        ](
            w,
            c,
            (num / den).cast[dtype]().cast[.float32](),
            norm_weight.load[width=vec]((c,)),
            epsilon,
            Float32(0),
            head_dim,
        )
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
            y = _hadamard[head_dim](y).cast[dtype]().cast[.float32]()
            y = _fake_quantize[group, fp4=True](y)
        else:
            # Every lane joins the group reduction; the rotary ones drop it.
            var q = _fake_quantize[group, fp4=False](y)
            if c < rope_start:
                y = q
        output.store((w, c), y.cast[dtype]())

    ctx.enqueue_function(kernel, grid_dim=rows, block_dim=threads)
