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
"""Input projection for a multi-token-prediction draft layer.

An MTP draft layer predicts token `t+2` from the embedding of the token the
target just produced and the target's hidden state at `t`:

    eh_proj( concat( enorm(embedding), hnorm(hidden_state) ) )

This kernel does everything before the projection. Both halves reduce over the
same `hidden_size`, so one block computes both row sums and writes the whole
`[tokens, 2 * hidden_size]` row.

Normalization matches `ops.rms_norm` in its Llama-style configuration, which is
what every MTP draft in the tree uses:

    out = cast(x * rsqrt(mean(x^2) + eps)) * weight

The cast comes before the multiply (`multiply_before_cast=False`). Swapping that
order changes the low bits, so the kernel would no longer match
`ops.rms_norm`.

Every tensor must be contiguous and row-major with a last dimension of exactly
`hidden_size`. The kernel addresses elements by linear offset through flat
views of the storage, which bypass the layout, so a transposed or strided view
would read the wrong elements.

Callers own position handling. MAX shifts a draft's token stream left per
request and appends a bonus token (`eagle_prefill_shift_tokens`), so every row
arriving here already holds the embedding it needs. Masking the first position
here as well would blank a correct row.
"""

from std.math import rsqrt
from std.utils import StaticTuple
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_dim,
    block_idx,
    thread_idx,
)
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from layout import TensorEngine, TensorLayout, TileTensor, row_major

from nn.normalization import block_reduce_dual_sum


@__name(t"mtp_eh_norm_{hidden_size}_{block_threads}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_threads))
)
def mtp_eh_norm_kernel[
    dtype: DType,
    OutLayoutType: TensorLayout,
    out_origin: MutOrigin,
    EmbedLayoutType: TensorLayout,
    embed_origin: ImmOrigin,
    PrevLayoutType: TensorLayout,
    prev_origin: ImmOrigin,
    EWLayoutType: TensorLayout,
    ew_origin: ImmOrigin,
    HWLayoutType: TensorLayout,
    hw_origin: ImmOrigin,
    hidden_size: Int,
    block_threads: Int,
    max_warps_per_block: Int,
    OutEngine: TensorEngine,
    EmbedEngine: TensorEngine,
    PrevEngine: TensorEngine,
    EWEngine: TensorEngine,
    HWEngine: TensorEngine,
    *,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    out_buf: TileTensor[dtype, OutLayoutType, out_origin, Engine=OutEngine],
    embed: TileTensor[
        mut=False, dtype, EmbedLayoutType, embed_origin, Engine=EmbedEngine
    ],
    prev: TileTensor[
        mut=False, dtype, PrevLayoutType, prev_origin, Engine=PrevEngine
    ],
    enorm_weight: TileTensor[
        mut=False, dtype, EWLayoutType, ew_origin, Engine=EWEngine
    ],
    hnorm_weight: TileTensor[
        mut=False, dtype, HWLayoutType, hw_origin, Engine=HWEngine
    ],
    epsilon: Float32,
    num_tokens: Int32,
):
    """Normalizes both inputs for one token and writes them side by side.

    One pass, 16 bytes at a time: each thread holds two 8-channel pieces of
    both input rows in registers (four 16-byte loads), reduces their squares
    through `block_reduce_dual_sum`, and stores both halves from those
    registers, so each element of a row is read once and written once. The
    arithmetic matches `ops.rms_norm` in its Llama-style configuration --
    f32 accumulation, `rsqrt(mean + eps)`, cast before the weight multiply --
    with one bf16 rounding per step.

    The register path needs `hidden_size == block_threads * 16` at compile
    time. Any other shape takes a strided body that walks the row with 2-byte
    accesses, so the ragged sizes the tests use keep working and every shape
    gets the same arithmetic either way.

    Parameters:
        dtype: Element type of the inputs, the weights and the output.
        OutLayoutType: Layout of `out_buf`.
        out_origin: Origin of `out_buf`.
        EmbedLayoutType: Layout of `embed`.
        embed_origin: Origin of `embed`.
        PrevLayoutType: Layout of `prev`.
        prev_origin: Origin of `prev`.
        EWLayoutType: Layout of `enorm_weight`.
        ew_origin: Origin of `enorm_weight`.
        HWLayoutType: Layout of `hnorm_weight`.
        hw_origin: Origin of `hnorm_weight`.
        hidden_size: Channels per input; the output row is twice this.
        block_threads: The launch's block width. The register path runs when
            `hidden_size == block_threads * 16`, and asserts that the launch
            matches it; every other shape takes the strided body, which needs
            a whole number of warps.
        max_warps_per_block: Power-of-two ceiling on the block's warp count,
            not the launch's actual warp count. `block_reduce_dual_sum`
            reduces over a power-of-two lane group; slots past the real warp
            count hold zero.
        OutEngine: Engine policy of `out_buf`.
        EmbedEngine: Engine policy of `embed`.
        PrevEngine: Engine policy of `prev`.
        EWEngine: Engine policy of `enorm_weight`.
        HWEngine: Engine policy of `hnorm_weight`.
        pdl_level: Whether the launch may overlap the grid before it in the
            stream. The op launches with that attribute and passes
            `PDLLevel.ON`; a caller launching without it must pass
            `PDLLevel.OFF`, or this kernel would read its inputs before the
            grid that wrote them finished.

    Args:
        out_buf: Output `[num_tokens, 2 * hidden_size]`.
        embed: Token embeddings `[num_tokens, hidden_size]`.
        prev: Target hidden states `[num_tokens, hidden_size]`.
        enorm_weight: Embedding norm weight `[hidden_size]`.
        hnorm_weight: Hidden-state norm weight `[hidden_size]`.
        epsilon: Added inside the square root, matching `ops.rms_norm`.
        num_tokens: Rows to process.
    """
    var token = block_idx.x
    if token >= Int(num_tokens):
        return
    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    var tid = thread_idx.x
    var row = token * hidden_size
    var embed_flat = embed.reshape(row_major(embed.num_elements()))
    var prev_flat = prev.reshape(row_major(prev.num_elements()))
    var out_row = token * 2 * hidden_size
    var out_flat = out_buf.reshape(row_major(out_buf.num_elements()))
    var enorm_flat = enorm_weight.reshape(row_major[hidden_size]())
    var hnorm_flat = hnorm_weight.reshape(row_major[hidden_size]())

    comptime wide = hidden_size == block_threads * 16
    comptime if wide:
        # Two 8-channel pieces per row half: [tid * 8, +8) and
        # [block_threads * 8 + tid * 8, +8). Together they cover the row.
        # The register path derives its second piece's offset from the
        # compile-time width, so the launch has to be that width.
        debug_assert(
            block_dim.x == block_threads,
            "mtp_eh_norm register path needs a launch at block_threads",
        )
        var lo = tid * 8
        var hi = block_threads * 8 + tid * 8
        var e_lo = embed_flat.load[width=8, alignment=16]((row + lo,)).cast[
            .float32
        ]()
        var e_hi = embed_flat.load[width=8, alignment=16]((row + hi,)).cast[
            .float32
        ]()
        var p_lo = prev_flat.load[width=8, alignment=16]((row + lo,)).cast[
            .float32
        ]()
        var p_hi = prev_flat.load[width=8, alignment=16]((row + hi,)).cast[
            .float32
        ]()
        var acc_e = (e_lo * e_lo + e_hi * e_hi).reduce_add()
        var acc_h = (p_lo * p_lo + p_hi * p_hi).reduce_add()
        var sums = block_reduce_dual_sum[.float32, max_warps_per_block](
            acc_e, acc_h
        )
        var rrms_e = rsqrt(sums[0] / Float32(hidden_size) + epsilon)
        var rrms_h = rsqrt(sums[1] / Float32(hidden_size) + epsilon)
        var w_lo = enorm_flat.load[width=8, alignment=16]((lo,))
        var w_hi = enorm_flat.load[width=8, alignment=16]((hi,))
        var v_lo = hnorm_flat.load[width=8, alignment=16]((lo,))
        var v_hi = hnorm_flat.load[width=8, alignment=16]((hi,))
        # Cast before the multiply, one rounding to bf16 after the weight.
        out_flat.store[width=8, alignment=16](
            (out_row + lo,),
            (e_lo * SIMD[DType.float32, 8](rrms_e)).cast[dtype]() * w_lo,
        )
        out_flat.store[width=8, alignment=16](
            (out_row + hi,),
            (e_hi * SIMD[DType.float32, 8](rrms_e)).cast[dtype]() * w_hi,
        )
        out_flat.store[width=8, alignment=16](
            (out_row + hidden_size + lo,),
            (p_lo * SIMD[DType.float32, 8](rrms_h)).cast[dtype]() * v_lo,
        )
        out_flat.store[width=8, alignment=16](
            (out_row + hidden_size + hi,),
            (p_hi * SIMD[DType.float32, 8](rrms_h)).cast[dtype]() * v_hi,
        )
        comptime if pdl_level > PDLLevel.OFF:
            launch_dependent_grids()
        return

    # Ragged shapes: the strided scalar body. `block_reduce_dual_sum` reads
    # one partial sum per whole warp, so a width that is not a multiple of
    # `WARP_SIZE` would drop its last partial warp.
    debug_assert(
        block_dim.x % WARP_SIZE == 0,
        "mtp_eh_norm block width must be a whole number of warps",
    )
    var stride = block_dim.x
    var acc_e = Float32(0)
    var acc_h = Float32(0)
    var col = tid
    while col < hidden_size:
        var e = embed_flat.load[width=1]((row + col,)).cast[.float32]()
        var p = prev_flat.load[width=1]((row + col,)).cast[.float32]()
        acc_e += e * e
        acc_h += p * p
        col += stride
    var sums = block_reduce_dual_sum[.float32, max_warps_per_block](
        acc_e, acc_h
    )
    var rrms_e = rsqrt(sums[0] / Float32(hidden_size) + epsilon)
    var rrms_h = rsqrt(sums[1] / Float32(hidden_size) + epsilon)
    col = tid
    while col < hidden_size:
        var e = embed_flat.load[width=1]((row + col,)).cast[.float32]() * rrms_e
        var p = prev_flat.load[width=1]((row + col,)).cast[.float32]() * rrms_h
        out_flat.store(
            (out_row + col,),
            e.cast[dtype]() * enorm_flat.load[width=1]((col,)),
        )
        out_flat.store(
            (out_row + hidden_size + col,),
            p.cast[dtype]() * hnorm_flat.load[width=1]((col,)),
        )
        col += stride
    comptime if pdl_level > PDLLevel.OFF:
        launch_dependent_grids()
