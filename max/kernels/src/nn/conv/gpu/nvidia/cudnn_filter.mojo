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
"""Filter layout change for the cuDNN convolution wrapper.

The wrapper hands cuDNN an NHWC activation and an FCRS (`[F, C, R, S]`)
filter. cuDNN's NHWC engines read a KRSC (`[F, R, S, C]`) filter, so with an
NCHW filter descriptor it runs its own `nchwToNhwcKernel` before every
convolution, which is slow: 21 us for the 5 MB 512 -> 1024 filter of the
1x5 Conv1D shape, 11 us for the 677 KB patch-embed filter, against 3 us for
the transpose below. `fcrs_to_krsc` does that change with a bandwidth-bound
kernel so the filter can be described to cuDNN as NHWC.
"""

from std.math import ceildiv
from std.sys import size_of

from layout import TensorEngine, TensorLayout, TileTensor
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import external_memory
from max.gpu.sync import barrier


comptime _BLOCK = 256
"""Threads per block of the transpose."""
comptime _VEC = 8
"""Elements one thread moves with a single vector store when `C % 8 == 0`."""


@__name(t"cudnn_filter_fcrs_to_krsc_{dtype}_{width}")
def _fcrs_to_krsc[
    dtype: DType,
    width: Int,
    SrcLayoutType: TensorLayout,
    src_origin: ImmOrigin,
    SrcEngine: TensorEngine,
    DstLayoutType: TensorLayout,
    dst_origin: MutOrigin,
    DstEngine: TensorEngine,
](
    src: TileTensor[dtype, SrcLayoutType, src_origin, Engine=SrcEngine],
    dst: TileTensor[dtype, DstLayoutType, dst_origin, Engine=DstEngine],
    channels: UInt32,
    taps: UInt32,
    vectors: UInt32,
):
    """Thread `t` writes `width` consecutive output channels of one tap of one
    output filter: the store is contiguous, and the `width` loads come from the
    same `src[f]` row (`taps` apart), which the block keeps in L1."""
    var t = UInt32(block_idx.x) * UInt32(_BLOCK) + UInt32(thread_idx.x)
    if t >= vectors:
        return
    var row = channels * taps
    var out_index = t * UInt32(width)
    var f, rem = divmod(out_index, row)
    var tap, c = divmod(rem, channels)
    var from_offset = Int(f * row + c * taps + tap)
    var v = SIMD[dtype, width]()
    comptime for i in range(width):
        v[i] = src.raw_load(from_offset + i * Int(taps))
    dst.raw_store(Int(out_index), v)


comptime _ROW_SMEM_LIMIT = 48 * 1024
"""Largest filter row, in bytes, the one-block-per-row kernel stages in
shared memory without opting in to a larger allocation."""


@__name(t"cudnn_filter_fcrs_to_krsc_row_{dtype}")
def _fcrs_to_krsc_row[
    dtype: DType,
    SrcLayoutType: TensorLayout,
    src_origin: ImmOrigin,
    SrcEngine: TensorEngine,
    DstLayoutType: TensorLayout,
    dst_origin: MutOrigin,
    DstEngine: TensorEngine,
](
    src: TileTensor[dtype, SrcLayoutType, src_origin, Engine=SrcEngine],
    dst: TileTensor[dtype, DstLayoutType, dst_origin, Engine=DstEngine],
    channels: UInt32,
    taps: UInt32,
):
    """One block per output filter `f`: the row `src[f]` (`C * R * S`
    contiguous elements) is read coalesced into shared memory and written
    back coalesced in KRSC order."""
    var smem = external_memory[
        Scalar[dtype], address_space=.SHARED, alignment=16
    ]()
    var n = channels * taps
    var base = Int(UInt32(block_idx.x) * n)
    var t = UInt32(thread_idx.x)
    var i = t
    while i < n:
        smem.unsafe_offset(Int(i)).unsafe_store(src.raw_load(base + Int(i)))
        i += UInt32(_BLOCK)
    barrier()
    i = t
    while i < n:
        var tap, c = divmod(i, channels)
        dst.raw_store(
            base + Int(i), smem.unsafe_offset(Int(c * taps + tap)).unsafe_load()
        )
        i += UInt32(_BLOCK)


def fcrs_to_krsc[
    dtype: DType
](
    filter: TileTensor[dtype, ...],
    dst: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """Writes the FCRS `filter` to `dst` as KRSC, `[F, R, S, C]`.

    Parameters:
        dtype: Element type of the filter.

    Args:
        filter: Dense row-major filter `[F, C, R, S]`.
        dst: Row-major destination of `filter.num_elements()` elements.
        ctx: Device context whose stream gets the launch.

    Raises:
        If the filter has more elements than 32-bit indices address, or the
        launch fails.
    """
    comptime assert filter.flat_rank == 4, "filter must be rank 4 (FCRS)"
    comptime assert filter.is_row_major, "the transpose indexes a dense filter"
    var total = filter.num_elements()
    if total >= (1 << 31):
        raise Error("cuDNN filter too large for the 32-bit layout change")
    var channels = Int(filter.dim[1]())
    var taps = Int(filter.dim[2]()) * Int(filter.dim[3]())
    var row_bytes = channels * taps * size_of[dtype]()
    if row_bytes <= _ROW_SMEM_LIMIT:
        comptime kernel = _fcrs_to_krsc_row[
            dtype,
            filter.LayoutType,
            ImmOrigin(filter.origin),
            filter.Engine,
            dst.LayoutType,
            dst.origin,
            dst.Engine,
        ]
        ctx.enqueue_function[kernel](
            filter.as_imm(),
            dst,
            UInt32(channels),
            UInt32(taps),
            grid_dim=Int(filter.dim[0]()),
            block_dim=_BLOCK,
            shared_mem_bytes=row_bytes,
        )
    elif channels % _VEC == 0:
        comptime kernel = _fcrs_to_krsc[
            dtype,
            _VEC,
            filter.LayoutType,
            ImmOrigin(filter.origin),
            filter.Engine,
            dst.LayoutType,
            dst.origin,
            dst.Engine,
        ]
        ctx.enqueue_function[kernel](
            filter.as_imm(),
            dst,
            UInt32(channels),
            UInt32(taps),
            UInt32(total // _VEC),
            grid_dim=ceildiv(total // _VEC, _BLOCK),
            block_dim=_BLOCK,
        )
    else:
        comptime kernel = _fcrs_to_krsc[
            dtype,
            1,
            filter.LayoutType,
            ImmOrigin(filter.origin),
            filter.Engine,
            dst.LayoutType,
            dst.origin,
            dst.Engine,
        ]
        ctx.enqueue_function[kernel](
            filter.as_imm(),
            dst,
            UInt32(channels),
            UInt32(taps),
            UInt32(total),
            grid_dim=ceildiv(total, _BLOCK),
            block_dim=_BLOCK,
        )
