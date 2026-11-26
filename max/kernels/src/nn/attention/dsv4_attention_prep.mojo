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
"""DeepSeek-V4 query preparation in one launch.

``dsv4_qkv_prep_ragged_paged`` replaces the run of small graph kernels in
front of the latent sparse attention: the query's weightless per-head RMS
normalize and tail RoPE, and the KV latent's RMSNorm, tail RoPE, FP8 QAT
round trip on the non-RoPE dims and store into the sliding-window leaf. The
compressed streams run in ``nn.attention.compress_windows``.

The op keeps every rounding point of the graph composition, rotates with the
complex multiply of ``nn.rope`` (the kernel the graph's RoPE tail runs) and
sums each row in the graph's reduction order, so it is bit-identical to the
composition; that pins the graph's current kernels, not the model's math, and
a change there is a test update.

The RoPE is interleaved: column pair ``(2i, 2i + 1)`` of the rotated tail is
the real and imaginary part of pair ``i``, and ``freqs`` row ``p`` holds
``(cos, sin)`` interleaved the same way, ``rope_dim`` floats per position.
"""

from std.math import rsqrt
from std.sys import inlined_assembly
from std.sys.info import is_nvidia_gpu, simd_width_of

from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.host import DeviceContext, get_gpu_target
import max.gpu.primitives.warp as warp

from internal_utils.fp8_utils import cast_saturating
from kv_cache.types import KVCacheT
from layout import TileTensor
from nn._ragged_utils import get_batch_from_row_offsets
from nn.attention.dsv4_quant import FP8_GROUP, fake_quantize
from nn.rope import _rope


@inline(.always)
def _square_rn[width: Int](x: SIMD[.float32, width]) -> type_of(x):
    """``x * x`` rounded on its own, never contracted into a following add.

    The graph's weightless normalize materializes ``x32 * x32`` before its
    mean reduction, so each square is rounded before it is summed; a plain
    multiply here would be fused with the running sum into an FMA.
    """
    comptime if is_nvidia_gpu():
        var r = type_of(x)(0)
        comptime for i in range(width):
            r[i] = inlined_assembly[
                "mul.rn.f32 $0, $1, $1;",
                Float32,
                constraints="=f,f",
                has_side_effect=False,
            ](x[i])
        return r
    else:
        return x * x


def dsv4_qkv_prep_ragged_paged[
    cache_t: KVCacheT,
    act_type: DType,
    //,
    rope_dim: Int,
](
    q_output: TileTensor[mut=True, act_type, address_space=.GENERIC, ...],
    q: TileTensor[mut=False, act_type, address_space=.GENERIC, ...],
    kv: TileTensor[mut=False, act_type, address_space=.GENERIC, ...],
    kv_gamma: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    freqs: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    swa_cache: cache_t,
    kv_col: Int,
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    """Normalizes and rotates the queries; builds and stores the KV latents.

    One warp per query (row, head) and one per KV row after them. Each row is
    summed exactly as the graph's warp-tier rowwise kernels sum it: lane
    ``l`` owns the ``width``-wide chunks at ``l * width + k * WARP_SIZE *
    width``, folds them lane-wise in ``k`` order, collapses its vector with
    ``reduce_add`` and the warp combines with ``warp.sum``. The query uses
    ``mo.reduce.mean``'s float32 width over materialized squares; the latent
    uses ``mo.reduce.rms_norm``'s width with the square folded into the sum
    and that op's epsilon added in float32.

    Parameters:
        cache_t: The sliding-window leaf's key cache type (inferred); one
            head of ``head_dim``.
        act_type: The activation dtype every intermediate rounds to.
        rope_dim: Width of the rotated tail.

    Args:
        q_output: ``[rows, heads, head_dim]`` normalized, rotated queries.
        q: ``[rows, heads, head_dim]`` raw ``wq_b`` output.
        kv: ``[rows, >= kv_col + head_dim]`` holding the raw ``wkv`` output
            at columns ``kv_col ..``, e.g. the stacked ``[wq_a; wkv]``
            projection.
        kv_gamma: ``[head_dim]`` the ``kv_norm`` weight.
        freqs: ``[max_seq_len, rope_dim]`` interleaved cos/sin per position.
        input_row_offsets: ``[batch + 1]`` ragged row offsets.
        swa_cache: This layer's sliding-window leaf; row ``t`` lands at its
            absolute position.
        kv_col: First column of the ``wkv`` output within ``kv``.
        epsilon: The RMS epsilon shared by both norms.
        ctx: Device context used to enqueue the kernel.
    """
    comptime head_dim = cache_t.kv_params.head_size
    comptime head_w = head_dim - rope_dim
    comptime assert cache_t.kv_params.num_heads == 1
    comptime assert rope_dim % 2 == 0 and rope_dim <= head_dim
    comptime assert head_w % FP8_GROUP == 0
    comptime assert q_output.flat_rank == 3 and q.flat_rank == 3
    comptime assert kv.flat_rank == 2 and freqs.flat_rank == 2
    comptime assert kv_gamma.flat_rank == 1
    comptime assert input_row_offsets.flat_rank == 1
    # `rowwise.pick_simd_width`'s choice for the graph's two reductions: the
    # widest natural width of the dtypes involved, which the per-thread
    # budget never caps for a single float32 accumulator.
    comptime q_width = simd_width_of[DType.float32, target=get_gpu_target()]()
    comptime kv_width = max(
        q_width, simd_width_of[act_type, target=get_gpu_target()]()
    )
    comptime assert head_dim % (WARP_SIZE * q_width) == 0
    comptime assert head_dim % (WARP_SIZE * kv_width) == 0
    # The rows fit the warp tier (at most 8 chunks a lane), whose summation
    # order the kernel reproduces.
    comptime assert head_dim <= WARP_SIZE * q_width * 8
    comptime assert FP8_GROUP % kv_width == 0 and rope_dim % kv_width == 0

    var num_rows = Int(q.dim[0]())
    var num_heads = Int(q.dim[1]())
    debug_assert(
        Int(q.dim[2]()) == head_dim and Int(kv.dim[1]()) >= kv_col + head_dim,
        "q must be head_dim wide and kv hold head_dim columns from kv_col",
    )
    debug_assert(Int(freqs.dim[1]()) == rope_dim, "freqs must be rope_dim wide")
    if num_rows == 0:
        return
    # Kernel captures must be fixed-width.
    var rows64 = Int64(num_rows)
    var heads64 = Int64(num_heads)
    var kv_col64 = Int64(kv_col)

    def kernel() {
        var q_output,
        var q,
        var kv,
        var kv_gamma,
        var freqs,
        var input_row_offsets,
        var swa_cache,
        var kv_col64,
        var epsilon,
        var rows64,
        var heads64,
    }:
        var num_heads = Int(heads64)
        var kv_col = Int(kv_col64)
        var row = block_idx.x
        var lane = thread_idx.x
        var q_rows = Int(rows64) * num_heads
        var t: Int
        var h = 0
        if row >= q_rows:
            t = row - q_rows
        else:
            t, h = divmod(row, num_heads)
        var b = get_batch_from_row_offsets(input_row_offsets, t)
        var pos = (
            swa_cache.cache_length(b) + t - Int(input_row_offsets.load((b,)))
        )

        if row < q_rows:
            comptime span = WARP_SIZE * q_width
            comptime chunks = head_dim // span
            var x = Array[SIMD[.float32, q_width], chunks](fill=0)
            var acc = SIMD[.float32, q_width](0)
            comptime for k in range(chunks):
                x[k] = q.load[width=q_width](
                    (t, h, lane * q_width + k * span)
                ).cast[.float32]()
                acc += _square_rn(x[k])
            var total = warp.sum(acc.reduce_add())
            # ``mean`` scales by the float64 reciprocal of the row length.
            var scale = rsqrt(
                total * (Float64(1) / Float64(head_dim)).cast[.float32]()
                + epsilon
            )
            comptime for k in range(chunks):
                var col = lane * q_width + k * span
                var y = (x[k] * scale).cast[act_type]().cast[.float32]()
                if col >= head_w:
                    y = (
                        _rope(y, freqs.load[width=q_width]((pos, col - head_w)))
                        .cast[act_type]()
                        .cast[.float32]()
                    )
                q_output.store((t, h, col), y.cast[act_type]())
        else:
            comptime span = WARP_SIZE * kv_width
            comptime chunks = head_dim // span
            var x = Array[SIMD[.float32, kv_width], chunks](fill=0)
            var acc = SIMD[.float32, kv_width](0)
            comptime for k in range(chunks):
                x[k] = kv.load[width=kv_width](
                    (t, kv_col + lane * kv_width + k * span)
                ).cast[.float32]()
                acc += x[k] * x[k]
            var total = warp.sum(acc.reduce_add())
            var inv_rms = rsqrt(total / Float32(head_dim) + epsilon)
            comptime for k in range(chunks):
                var col = lane * kv_width + k * span
                # ``RMSNorm`` casts its float32 weight to the input dtype first.
                var gamma = (
                    kv_gamma.load[width=kv_width]((col,))
                    .cast[act_type]()
                    .cast[.float32]()
                )
                var y = (
                    (x[k] * inv_rms * gamma).cast[act_type]().cast[.float32]()
                )
                if col >= head_w:
                    y = (
                        _rope(
                            y, freqs.load[width=kv_width]((pos, col - head_w))
                        )
                        .cast[act_type]()
                        .cast[.float32]()
                    )
                # Every lane joins the group reduction; the rotary ones drop
                # it.
                var quantized = fake_quantize[FP8_GROUP, fp4=False](y)
                if col < head_w:
                    y = quantized
                swa_cache.store(
                    b,
                    0,
                    pos,
                    col,
                    cast_saturating[cache_t.dtype](y.cast[act_type]()),
                )

    ctx.enqueue_closure(
        kernel, grid_dim=num_rows * (num_heads + 1), block_dim=WARP_SIZE
    )
