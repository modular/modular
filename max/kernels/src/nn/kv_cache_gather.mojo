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
"""Gathers rows of a paged KV cache by an explicit per-row slot list.

The read side of ``kv_cache_store_ragged``: row ``r`` of ``slots`` belongs to
the request ``b`` with ``row_offsets[b] <= r < row_offsets[b + 1]``, and
``output[r, j, :]`` is the cache row at slot ``slots[r, j]`` of that request,
resolved through the request's lookup table exactly as ``load`` resolves a
token index. A leaf whose buffer holds ``slots_per_page`` rows per page is
paged by that count, because ``load`` reads the page size off the block shape,
so its slot is an entry index rather than a token index.

This is a pure copy in the cache's own dtype, so the result is bit-identical
to indexing the block buffer by hand, on every target.
"""

from std.sys.info import CompilationTarget, simd_width_of

from max.algorithm import elementwise
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import is_cpu

from kv_cache.types import KVCacheT
from layout import Coord, TileTensor
from std.utils import IndexList

from nn._ragged_utils import get_batch_from_row_offsets


def kv_cache_gather_rows_ragged[
    cache_t: KVCacheT,
    dtype: DType,
    //,
    target: StaticString,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    slots: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    row_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    cache: cache_t,
    ctx: DeviceContext,
) raises:
    """Copies ``cache`` rows at ``slots`` into ``output``.

    Parameters:
        cache_t: The key or value cache type (inferred); one head per slot.
        dtype: The output element type (inferred); must be the cache's.
        target: Compilation target string, selects the CPU or GPU path.

    Args:
        output: ``[num_rows, num_slots, head_dim]``, the last axis contiguous.
        slots: ``[num_rows, num_slots]`` slot indices. Every slot must be
            non-negative and inside its request's allocated pages.
        row_offsets: ``[batch + 1]`` ragged offsets mapping each row of
            ``slots`` to a request.
        cache: The key or value cache of one layer.
        ctx: Device context used to enqueue the GPU kernel.
    """
    comptime head_dim = cache_t.kv_params.head_size
    comptime assert (
        dtype == cache_t.dtype
    ), "gather_rows copies rows in the cache's own dtype"
    comptime assert (
        cache_t.kv_params.num_heads == 1
    ), "gather_rows reads caches that hold one head per slot"
    comptime assert output.flat_rank == 3 and slots.flat_rank == 2
    comptime assert row_offsets.flat_rank == 1

    var num_rows = Int(slots.dim[0]())
    var num_slots = Int(slots.dim[1]())
    debug_assert(
        Int(output.dynamic_stride(2)) == 1,
        "output must be contiguous along head_dim",
    )
    if num_rows == 0 or num_slots == 0:
        return

    # Same closure form as `kv_cache_store_ragged`: a KV cache view cannot be
    # captured by a unified closure yet.
    @__parameter
    @__copy_capture(cache, row_offsets, output, slots)
    def copy_row[width: Int, alignment: Int = 1](idx: Coord) capturing:
        var r = Int(idx[0].value())
        var j = Int(idx[1].value())
        var d = Int(idx[2].value())
        var b = get_batch_from_row_offsets(row_offsets, r)
        var slot = Int(slots[r, j])
        # `dtype == cache_t.dtype` is asserted above; the two are still
        # distinct symbols to the type checker.
        output.store[width=width, alignment=alignment](
            (r, j, d),
            rebind[SIMD[dtype, width]](cache.load[width=width](b, 0, slot, d)),
        )

    comptime simd_width = (
        simd_width_of[
            cache_t.dtype, target=CompilationTarget.current()
        ]() if is_cpu[target]() else simd_width_of[
            cache_t.dtype, target=get_gpu_target()
        ]()
    )

    elementwise[
        copy_row,
        simd_width,
        target=target,
        _trace_description="kv_cache_gather_rows_ragged",
    ](Coord(num_rows, num_slots, head_dim), ctx)
