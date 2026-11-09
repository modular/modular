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
"""CPU implementations of elementwise functions."""

from std.math import align_down, ceildiv

from std.utils.coord import Coord
from std.utils.static_tuple import StaticTuple

from max.gpu.host import DeviceContext
from std.algorithm.backend.vectorize import vectorize

from .parallelize import _get_num_workers, sync_parallelize
from max.algorithm.functional import _index_coord, _start_indices_of_nth_element


# ===-----------------------------------------------------------------------===#
# Elementwise CPU implementations
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _elementwise_impl_cpu[
    simd_width: Int,
    FuncType: def[width: Int, alignment: Int = 1](Coord) -> None,
    *,
    trace_description: StaticString,
](func: FuncType, *, shape: Coord, ctx: Optional[DeviceContext.CPU] = None,):
    """Dispatches elementwise execution on CPU to the 1D or ND implementation
    based on the rank of the input shape.

    Parameters:
        simd_width: The SIMD vector width to use.
        FuncType: The body function type.
        trace_description: Description of the trace.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: Optional CPU DeviceContext to execute the tasks on.
    """

    comptime impl = _elementwise_impl_cpu_1d if shape.rank == 1 else _elementwise_impl_cpu_nd
    impl[simd_width](func, shape, ctx)


@inline(.always)
def _elementwise_impl_cpu_1d[
    simd_width: Int,
    FuncType: def[width: Int, alignment: Int = 1](Coord) -> None,
](func: FuncType, shape: Coord, ctx: Optional[DeviceContext.CPU] = None,):
    """Executes `func[width, rank](indices)`, possibly using sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        simd_width: The SIMD vector width to use.
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: Optional CPU DeviceContext to execute the tasks on.
    """
    comptime assert shape.rank == 1, "Specialization for 1D"

    comptime unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    var problem_size = shape.product()

    var num_workers = _get_num_workers(problem_size, ctx=ctx)
    var chunk_size = ceildiv(problem_size, num_workers)

    @inline(.always)
    def task_func(i: Int) {imm}:
        var start_offset = i * chunk_size
        var end_offset = min((i + 1) * chunk_size, problem_size)
        var len = end_offset - start_offset

        @inline(.always)
        def func_wrapper[
            simd_width: Int
        ](idx: Int) {imm start_offset, imm func,}:
            var offset = start_offset + idx
            func[simd_width](Coord(offset))

        vectorize[simd_width, unroll_factor=unroll_factor](len, func_wrapper)

    sync_parallelize(task_func, num_workers, ctx)


@inline(.always)
def _elementwise_impl_cpu_nd[
    simd_width: Int,
    FuncType: def[width: Int, alignment: Int = 1](Coord) -> None,
](func: FuncType, shape: Coord, ctx: Optional[DeviceContext.CPU] = None,):
    """Executes `func[width, rank](indices)`, possibly using sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns
    when all sub-tasks have completed.

    Parameters:
        simd_width: The SIMD vector width to use.
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: Optional CPU DeviceContext to execute the tasks on.
    """
    comptime assert shape.rank > 1, "Specialization for ND where N > 1"
    comptime rank = shape.rank

    # If we know we won't do any work, return early
    if shape[rank - 1].value() == 0:
        return

    comptime unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    # Strategy: we parallelize over all dimensions except the innermost and
    # vectorize over the innermost dimension. We unroll the innermost dimension
    # by a factor of unroll_factor.

    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.
    var total_size = shape.product()

    var num_workers = _get_num_workers(total_size, ctx=ctx)
    var parallelism_size = total_size // SIMDLength(shape[rank - 1].value())
    var chunk_size = ceildiv(parallelism_size, num_workers)

    @inline(.always)
    def task_func(i: Int) {imm}:
        var start_row = i * chunk_size
        var end_row = min((i + 1) * chunk_size, parallelism_size)
        if end_row <= start_row:
            return
        _elementwise_nd_rows[simd_width, unroll_factor=unroll_factor](
            func, shape, start_row, end_row - start_row
        )

    sync_parallelize(task_func, num_workers, ctx)


@inline(.always)
def _elementwise_nd_rows[
    simd_width: Int,
    FuncType: def[width: Int, alignment: Int = 1](Coord) -> None,
    *,
    unroll_factor: Int,
](func: FuncType, shape: Coord, start_row: Int, num_rows: Int):
    """Runs `func` over `num_rows` innermost rows of `shape`, starting at row
    `start_row` of the row-major order over all but the innermost dim.

    Parameters:
        simd_width: The SIMD vector width to use.
        FuncType: The body function type.
        unroll_factor: The unroll factor of the full-width loop.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        start_row: The first row to process.
        num_rows: The number of rows to process.
    """
    comptime rank = shape.rank

    # Decompose the first row once and step the outer indices per row; with
    # dynamic dims each decomposition costs runtime divisions, which dominates
    # when the innermost dim is small.
    var dims = StaticTuple[Int, rank](0)
    comptime for d in range(rank):
        dims[d] = Int(shape[d].value())
    var flat = _start_indices_of_nth_element[
        rank,
        type_of(shape).ParamListType,
        False,
        subvolume_rank=1,
    ](start_row, shape)

    @inline(.always)
    def func_wrapper_nd[simd_width: Int](idx: Int) {mut flat, imm func, imm}:
        flat[rank - 1] = idx
        func[simd_width](_index_coord(flat))

    # Rows are visited in runs along the next-to-last dim, with the row counter
    # in a register and the carry into the outer dims once per run. Carrying
    # the whole index from row to row measured slower than the per-row divmods
    # it replaces on Graviton.
    var rows_left = num_rows
    while rows_left > 0:
        var first = flat[rank - 2]
        var run = min(rows_left, dims[rank - 2] - first)
        for j in range(run):
            flat[rank - 2] = first + j
            _vectorize_pow2_tail[simd_width, unroll_factor=unroll_factor](
                dims[rank - 1], func_wrapper_nd
            )
        rows_left -= run
        _advance_outer_indices(flat, dims)


@inline(.always)
def _advance_outer_indices[
    rank: Int
](mut idx: StaticTuple[Int, rank], shape: StaticTuple[Int, rank]):
    """Steps `idx` to the next row, odometer-style over all but the innermost
    dim. Wraps to all zeros after the last row."""
    comptime for k in range(rank - 1):
        comptime d = rank - 2 - k
        idx[d] += 1
        if idx[d] < shape[d]:
            return
        idx[d] = 0


@inline(.always)
def _vectorize_pow2_tail[
    func: def[width: Int](idx: Int) -> None,
    //,
    simd_width: Int,
    *,
    unroll_factor: Int = 1,
](size: Int, closure: func):
    """Like `vectorize`, but runs the remainder below `simd_width` as
    descending power-of-two widths instead of width-1 calls, so a short
    innermost dim does not degrade to scalar code."""
    var simd_end = Int(align_down(UInt(size), UInt(simd_width)))
    vectorize[simd_width, unroll_factor=unroll_factor](simd_end, closure)
    var i = simd_end
    comptime for k in range(16):
        comptime width = 1 << (15 - k)
        comptime if width < simd_width:
            if size - i >= width:
                closure[width](i)
                i += width
