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
"""Implements higher-order functions.

You can import these APIs from the `algorithm` package. For example:

```mojo
from std.algorithm import map
```
"""

from std.collections.string.string_slice import get_static_string
from std.math import ceildiv
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.gpu.primitives.grid_controls import PDLLevel
from std.runtime.asyncrt import DeviceContextPtr
from std.runtime.tracing import Trace, TraceLevel, get_safe_task_id, trace_arg

from std.utils.index import Index, IndexList

# Re-export backend-independent implementations.
from .backend import (
    BinaryTile1DTileUnitFunc,
    Dynamic1DTileUnitFunc,
    Dynamic1DTileUnswitchUnitFunc,
    Static1DTileUnitFunc,
    Static1DTileUnitFuncWithFlag,
    Static1DTileUnitFuncWithFlags,
    Static1DTileUnswitchUnitFunc,
    Static2DTileUnitFunc,
    SwitchedFunction,
    SwitchedFunction2,
    tile,
    tile_and_unswitch,
    tile_middle_unswitch_boundaries,
    unswitch,
    vectorize,
)

# Re-export CPU implementations.
from .backend.cpu import (
    _elementwise_impl_cpu,
    _get_num_workers,
    _stencil_impl_cpu,
    map,
    parallelize,
    parallelize_over_rows,
    sync_parallelize,
)

# Re-export GPU implementations.
from .backend.gpu import _elementwise_impl_gpu, _stencil_impl_gpu


# ===-----------------------------------------------------------------------===#
# Shared Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
def _get_start_indices_of_nth_subvolume[
    rank: Int, //, subvolume_rank: Int = 1
](n: Int, shape: IndexList[rank, ...], out res: type_of(shape)):
    """Converts a flat index into the starting ND indices of the nth subvolume
    with rank `subvolume_rank`.

    For example:
        - `_get_start_indices_of_nth_subvolume[0](n, shape)` will return
        the starting indices of the nth element in shape.
        - `_get_start_indices_of_nth_subvolume[1](n, shape)` will return
        the starting indices of the nth row in shape.
        - `_get_start_indices_of_nth_subvolume[2](n, shape)` will return
        the starting indices of the nth horizontal slice in shape.

    The ND indices will iterate from right to left. I.E

        shape = (20, 5, 2, N)
        _get_start_indices_of_nth_subvolume[1](1, shape) = (0, 0, 1, 0)
        _get_start_indices_of_nth_subvolume[1](5, shape) = (0, 2, 1, 0)
        _get_start_indices_of_nth_subvolume[1](50, shape) = (5, 0, 0, 0)
        _get_start_indices_of_nth_subvolume[1](56, shape) = (5, 1, 1, 0)

    Parameters:
        rank: The rank of the ND index.
        subvolume_rank: The rank of the subvolume under consideration.

    Args:
        n: The flat index to convert (the nth subvolume to retrieve).
        shape: The shape of the ND space we are converting into.

    Returns:
        Constructed ND-index.
    """

    comptime assert (
        subvolume_rank <= rank
    ), "subvolume rank cannot be greater than indices rank"
    comptime assert subvolume_rank >= 0, "subvolume rank must be non-negative"

    comptime IntType = type_of(shape)._int_type

    # fast impls for common cases
    comptime if rank == 2 and subvolume_rank == 1:
        return {n, 0}

    comptime if rank - 1 == subvolume_rank:
        res = {0}
        res[0] = n
        return

    comptime if rank == subvolume_rank:
        return {0}

    res = {}
    # If the index type is unsigned, be sure to use unsigned div/mod operations.
    var curr_index = IntType(n)

    comptime for i in reversed(range(1, rank - subvolume_rank)):
        curr_index, res.data[i] = divmod(curr_index, IntType(shape.get[i]()))

    res.data[0] = curr_index


# ===-----------------------------------------------------------------------===#
# Elementwise
# ===-----------------------------------------------------------------------===#


@always_inline
def elementwise[
    func: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    _trace_description: StaticString = "",
](shape: Int) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: Do not invoke the function using asynchronous calls.
        target: The target to run on.
        _trace_description: Description of the trace.

    Args:
        shape: The shape of the buffer.

    Raises:
        If the operation fails.
    """

    elementwise[
        func,
        simd_width=simd_width,
        use_blocking_impl=use_blocking_impl,
        target=target,
        _trace_description=_trace_description,
    ](Index(shape))


@always_inline
def elementwise[
    rank: Int,
    //,
    func: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    _trace_description: StaticString = "",
](shape: IndexList[rank, ...]) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: Do not invoke the function using asynchronous calls.
        target: The target to run on.
        _trace_description: Description of the trace.

    Args:
        shape: The shape of the buffer.

    Raises:
        If the operation fails.
    """

    comptime assert is_cpu[target](), (
        "the target must be CPU use the elementwise which takes the"
        " DeviceContext to be able to use the GPU version"
    )

    _elementwise_impl_cpu[
        func=func, simd_width=simd_width, use_blocking_impl=use_blocking_impl
    ](shape=shape)


@always_inline
def elementwise[
    func: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    _trace_description: StaticString = "",
    pdl_level: PDLLevel = PDLLevel(),
](shape: Int, context: DeviceContext) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: Do not invoke the function using asynchronous calls.
        target: The target to run on.
        _trace_description: Description of the trace.
        pdl_level: The PDL level controlling kernel overlap behavior.

    Args:
        shape: The shape of the buffer.
        context: The device context to use.

    Raises:
        If the operation fails.
    """

    elementwise[
        func,
        simd_width=simd_width,
        use_blocking_impl=use_blocking_impl,
        target=target,
        pdl_level=pdl_level,
    ](Index(shape), context)


@always_inline
def elementwise[
    rank: Int,
    //,
    func: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    _trace_description: StaticString = "",
    pdl_level: PDLLevel = PDLLevel(),
](shape: IndexList[rank, ...], context: DeviceContext) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: Do not invoke the function using asynchronous calls.
        target: The target to run on.
        _trace_description: Description of the trace.
        pdl_level: The PDL level controlling kernel overlap behavior.

    Args:
        shape: The shape of the buffer.
        context: The device context to use.

    Raises:
        If the operation fails.
    """

    _elementwise_impl[
        func,
        simd_width,
        use_blocking_impl=use_blocking_impl,
        target=target,
        pdl_level=pdl_level,
    ](shape, context)


@always_inline
def elementwise[
    rank: Int,
    //,
    func: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    _trace_description: StaticString = "",
    pdl_level: PDLLevel = PDLLevel(1),
](shape: IndexList[rank, ...], context: DeviceContextPtr) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        rank: The rank of the buffer.
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: Do not invoke the function using asynchronous calls.
        target: The target to run on.
        _trace_description: Description of the trace.
        pdl_level: The PDL level controlling kernel overlap behavior.

    Args:
        shape: The shape of the buffer.
        context: The device context to use.

    Raises:
        If the operation fails.
    """

    @always_inline
    @parameter
    def description_fn() -> String:
        var shape_str = trace_arg("shape", shape)
        var vector_width_str = String(t"vector_width={simd_width}")
        return ";".join(Span([shape_str^, vector_width_str^]))

    # Intern the kind string as a static string so we don't allocate.
    comptime d = _trace_description
    comptime desc = String(t"({d})") if d else ""
    comptime kind = get_static_string["elementwise", desc]()

    with Trace[TraceLevel.OP, target=target](
        kind,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=get_safe_task_id(context),
    ):
        comptime if is_gpu[target]():
            _elementwise_impl_gpu[
                func=func, simd_width=UInt(simd_width), pdl_level=pdl_level
            ](shape=shape, ctx=context[])
        else:
            _elementwise_impl_cpu[
                func=func,
                simd_width=simd_width,
                use_blocking_impl=use_blocking_impl,
            ](shape=shape)


@always_inline
def _elementwise_impl[
    rank: Int,
    //,
    func: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: Int,
    /,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    pdl_level: PDLLevel = PDLLevel(),
](shape: IndexList[rank, ...], context: DeviceContext) raises:
    comptime if is_cpu[target]():
        _elementwise_impl_cpu[
            func=func,
            simd_width=simd_width,
            use_blocking_impl=use_blocking_impl,
        ](shape=shape)
    else:
        _elementwise_impl_gpu[
            func=func,
            simd_width=UInt(simd_width),
            pdl_level=pdl_level,
        ](shape=shape, ctx=context)


# ===-----------------------------------------------------------------------===#
# stencil
# ===-----------------------------------------------------------------------===#

comptime stencil = _stencil_impl_cpu
"""CPU implementation of stencil computation."""

comptime stencil_gpu = _stencil_impl_gpu
"""GPU implementation of stencil computation."""
