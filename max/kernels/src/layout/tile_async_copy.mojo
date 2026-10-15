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
"""Tile copies into shared memory laid out for `ldmatrix` readers."""

from std.bit import log2_floor
from std.sys import is_nvidia_gpu, size_of

from .swizzle import Swizzle, make_ldmatrix_swizzle
from .tile_io import copy_dram_to_sram, copy_dram_to_sram_async
from .tile_layout import Layout
from .tile_tensor import TileTensor


@always_inline
def copy_dram_to_sram_async_ldmatrix[
    thread_layout: Layout,
    *,
    simd_size: Int,
    swizzle: Bool,
    masked: Bool = False,
    num_threads: Int = thread_layout.size(),
](
    dst: TileTensor[mut=True, address_space=.SHARED, ...],
    src: TileTensor,
    num_valid_rows: Int = 0,
):
    """Copies a tile from global to shared memory, one `simd_size` vector at a
    time, optionally swizzled for `ldmatrix`.

    On NVIDIA this is an asynchronous `cp.async` copy; the caller commits and
    waits. Elsewhere it falls back to the synchronous copy, where `swizzle` and
    `masked` do not apply.

    With `swizzle`, the destination is swizzled by the pattern
    `make_ldmatrix_swizzle` derives from its row stride, so the tile can be read
    by `TensorCore.load_a`/`load_b` with that same row stride. A `masked` copy
    zero-fills the vectors of rows at or beyond `num_valid_rows`.

    Parameters:
        thread_layout: Layout of the threads over the copy.
        simd_size: Elements per copied vector.
        swizzle: Whether to swizzle the destination for `ldmatrix`.
        masked: Whether to zero-fill rows past `num_valid_rows`.
        num_threads: Total threads in the block; those beyond
            `thread_layout.size()` do not copy.

    Args:
        dst: Destination tile in shared memory, with a static row stride.
        src: Source tile in global memory.
        num_valid_rows: Valid rows of `src`; only consulted when `masked`.
    """
    comptime if is_nvidia_gpu():
        comptime row_size = dst.static_stride[0]
        # See make_ldmatrix_swizzle for `conflict_ways`.
        comptime bytes_32_banks = 128
        comptime conflict_ways = min(
            8 * row_size * size_of[dst.dtype]() // bytes_32_banks, 8
        )
        comptime assert (
            swizzle and (conflict_ways in (4, 8))
        ) or not swizzle, "Only support swizzle for 4 or 8 ways conflict."
        comptime assert (
            swizzle and row_size in (16, 32, 64, 128, 256, 512)
        ) or not swizzle, (
            "Only support 2^4-2^9 elements per row in shared memory tile"
            " for async copy with swizzling."
        )
        comptime swizzle_option = Optional[Swizzle](
            make_ldmatrix_swizzle[dst.dtype, row_size, log2_floor(simd_size)]()
        ) if swizzle else Optional[Swizzle]()
        copy_dram_to_sram_async[
            thread_layout=thread_layout,
            swizzle=swizzle_option,
            masked=masked,
            num_threads=num_threads,
        ](
            dst.vectorize[1, simd_size](),
            src.address_space_cast[.GENERIC]().vectorize[1, simd_size](),
            num_valid_rows,
        )
    else:
        copy_dram_to_sram[thread_layout=thread_layout](
            dst.vectorize[1, simd_size](),
            src.address_space_cast[.GENERIC]().vectorize[1, simd_size](),
        )
