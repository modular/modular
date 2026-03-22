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

from std.math import ceildiv

from std.utils.index import IndexList

from .map import map
from .parallelize import _get_num_workers, sync_parallelize
from ..vectorize import vectorize
from std.algorithm.functional import _get_start_indices_of_nth_subvolume


# ===-----------------------------------------------------------------------===#
# Elementwise CPU implementations
# ===-----------------------------------------------------------------------===#


@always_inline
def _elementwise_impl_cpu[
    rank: Int,
    //,
    *,
    func: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: Int,
    use_blocking_impl: Bool = False,
](*, shape: IndexList[rank, ...]):
    """Dispatches elementwise execution on CPU to the 1D or ND implementation
    based on the rank of the input shape.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: If true the function executes without sub-tasks.

    Args:
        shape: The shape of the buffer.
    """

    def func_unified[
        width: Int, rank: Int, alignment: Int = 1
    ](indices: IndexList[rank]) unified {}:
        func[width, rank, alignment](indices)

    comptime impl = _elementwise_impl_cpu_1d if rank == 1 else _elementwise_impl_cpu_nd
    impl[
        type_of(func_unified), simd_width, use_blocking_impl=use_blocking_impl
    ](shape, func_unified)


@always_inline
def _elementwise_impl_cpu_1d[
    rank: Int,
    //,
    func: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) unified -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool,
](shape: IndexList[rank, ...], func_closure: func):
    """Executes `func[width, rank](indices)`, possibly using sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: If true the functions execute without sub-tasks.

    Args:
        shape: The shape of the buffer.
        func_closure: The closure carrying the captured state of func.
    """
    comptime assert rank == 1, "Specialization for 1D"

    comptime unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    var problem_size = shape.flattened_length()

    comptime if use_blocking_impl:

        @always_inline
        def blocking_task_fun[
            simd_width: Int
        ](idx: Int) unified {read func_closure,}:
            func_closure[simd_width, rank](IndexList[rank](idx))

        vectorize[simd_width, unroll_factor=unroll_factor](
            problem_size, blocking_task_fun
        )
        return

    var num_workers = _get_num_workers(problem_size)
    var chunk_size = ceildiv(problem_size, num_workers)

    @always_inline
    @parameter
    def task_func(i: Int):
        var start_offset = i * chunk_size
        var end_offset = min((i + 1) * chunk_size, problem_size)
        var len = end_offset - start_offset

        @always_inline
        def func_wrapper[
            simd_width: Int
        ](idx: Int) unified {read start_offset, read func_closure,}:
            var offset = start_offset + idx
            func_closure[simd_width, rank](IndexList[rank](offset))

        vectorize[simd_width, unroll_factor=unroll_factor](len, func_wrapper)

    sync_parallelize[task_func](num_workers)


@always_inline
def _elementwise_impl_cpu_nd[
    rank: Int,
    //,
    func: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) unified -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool,
](shape: IndexList[rank, ...], func_closure: func):
    """Executes `func[width, rank](indices)`, possibly using sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns
    when all sub-tasks have completed.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: If true this is a blocking op.

    Args:
        shape: The shape of the buffer.
        func_closure: The closure carrying the captured state of func.
    """
    comptime assert rank > 1, "Specialization for ND where N > 1"

    # If we know we won't do any work, return early
    if shape[rank - 1] == 0:
        return

    comptime unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    # Strategy: we parallelize over all dimensions except the innermost and
    # vectorize over the innermost dimension. We unroll the innermost dimension
    # by a factor of unroll_factor.

    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.
    var total_size: Int = shape.flattened_length()

    comptime if use_blocking_impl:

        @always_inline
        @parameter
        def blocking_task_fn(i: Int):
            var indices = _get_start_indices_of_nth_subvolume(i, shape)

            @always_inline
            def func_wrapper[
                simd_width: Int
            ](idx: Int) unified {mut indices, read func_closure,}:
                indices[rank - 1] = idx
                func_closure[simd_width, rank](indices.canonicalize())

            # We vectorize over the innermost dimension.
            vectorize[simd_width, unroll_factor=unroll_factor](
                shape[rank - 1], func_wrapper
            )

        map[blocking_task_fn](total_size // shape[rank - 1])

        return

    var num_workers = _get_num_workers(total_size)
    var parallelism_size = total_size // shape[rank - 1]
    var chunk_size = ceildiv(parallelism_size, num_workers)

    @always_inline
    @parameter
    def task_func(i: Int):
        var start_parallel_offset = i * chunk_size
        var end_parallel_offset = min((i + 1) * chunk_size, parallelism_size)

        var len = end_parallel_offset - start_parallel_offset
        if len <= 0:
            return

        for parallel_offset in range(
            start_parallel_offset, end_parallel_offset
        ):
            var indices = _get_start_indices_of_nth_subvolume(
                parallel_offset, shape
            )

            @always_inline
            def func_wrapper[
                simd_width: Int
            ](idx: Int) unified {mut indices, read func_closure,}:
                indices[rank - 1] = idx
                func_closure[simd_width, rank](indices.canonicalize())

            # We vectorize over the innermost dimension.
            vectorize[simd_width, unroll_factor=unroll_factor](
                shape[rank - 1], func_wrapper
            )

    sync_parallelize[task_func](num_workers)
