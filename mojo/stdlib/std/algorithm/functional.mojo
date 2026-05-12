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
from std.gpu import PDLLevel
from std.math import ceildiv
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
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
from .backend.gpu import (
    _dual_elementwise_impl_gpu,
    _elementwise_impl_gpu,
    _stencil_impl_gpu,
)


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
    _trace_description: StaticString = "elementwise",
](shape: Int, ctx: Optional[DeviceContext] = None) raises:
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
        ctx: The context to execute the work on.

    Raises:
        If the operation fails.
    """

    elementwise[
        func,
        simd_width=simd_width,
        use_blocking_impl=use_blocking_impl,
        target=target,
        _trace_description=_trace_description,
    ](Index(shape), ctx)


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
    _trace_description: StaticString = "elementwise",
](shape: IndexList[rank, ...], ctx: Optional[DeviceContext] = None) raises:
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
        ctx: The context to execute the work on.

    Raises:
        If the operation fails.
    """

    comptime assert is_cpu[target](), (
        "the target must be CPU use the elementwise which takes the"
        " DeviceContext to be able to use the GPU version"
    )

    def func_unified[
        width: Int, rank: Int, alignment: Int = 1
    ](indices: IndexList[rank]) register_passable {}:
        func[width, rank, alignment](indices)

    _elementwise_impl_cpu[
        simd_width=simd_width,
        use_blocking_impl=use_blocking_impl,
        trace_description=_trace_description,
    ](func_unified, shape=shape, ctx=ctx)


@always_inline
def elementwise[
    func: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: Int,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    pdl_level: PDLLevel = PDLLevel(1),
    _trace_description: StaticString = "elementwise",
](shape: Int, context: DeviceContext) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        func: The body function.
        simd_width: The SIMD vector width to use.
        use_blocking_impl: Do not invoke the function using asynchronous calls.
        target: The target to run on.
        pdl_level: The PDL level controlling GPU kernel overlap behavior.
        _trace_description: Description of the trace.

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
        _trace_description=_trace_description,
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
    pdl_level: PDLLevel = PDLLevel(1),
    _trace_description: StaticString = "elementwise",
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
        pdl_level: The PDL level controlling GPU kernel overlap behavior.
        _trace_description: Description of the trace.

    Args:
        shape: The shape of the buffer.
        context: The device context to use.

    Raises:
        If the operation fails.
    """

    def func_unified[
        width: Int, rank: Int, alignment: Int = 1
    ](indices: IndexList[rank]) register_passable {}:
        func[width, rank, alignment](indices)

    _elementwise_impl[
        simd_width,
        use_blocking_impl=use_blocking_impl,
        target=target,
        pdl_level=pdl_level,
        trace_description=_trace_description,
    ](func_unified, shape, context)


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
    pdl_level: PDLLevel = PDLLevel(1),
    _trace_description: StaticString = "elementwise",
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
        pdl_level: The PDL level controlling GPU kernel overlap behavior.
        _trace_description: Description of the trace.

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

            def gpu_func_unified[
                width: Int, rank: Int, alignment: Int = 1
            ](indices: IndexList[rank]) register_passable {}:
                func[width, rank, alignment](indices)

            _elementwise_impl_gpu[
                simd_width=simd_width,
                pdl_level=pdl_level,
                trace_description=kind,
            ](gpu_func_unified, shape=shape, ctx=context[])
        else:

            def cpu_func_unified[
                width: Int, rank: Int, alignment: Int = 1
            ](indices: IndexList[rank]) register_passable {}:
                func[width, rank, alignment](indices)

            _elementwise_impl_cpu[
                simd_width=simd_width,
                use_blocking_impl=use_blocking_impl,
                trace_description=_trace_description,
            ](
                cpu_func_unified,
                shape=shape,
                ctx=context.get_optional_device_context(),
            )


@always_inline
def _elementwise_impl[
    rank: Int,
    //,
    simd_width: Int,
    FuncType: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) register_passable -> None,
    /,
    *,
    use_blocking_impl: Bool = False,
    target: StaticString = "cpu",
    pdl_level: PDLLevel = PDLLevel(1),
    trace_description: StaticString = "elementwise",
](func: FuncType, shape: IndexList[rank, ...], context: DeviceContext) raises:
    @always_inline
    @parameter
    def description_fn() -> String:
        var shape_str = trace_arg("shape", shape)
        var vector_width_str = String(t"vector_width={simd_width}")
        return ";".join(Span([shape_str^, vector_width_str^]))

    # Intern the kind string as a static string so we don't allocate.
    comptime d = trace_description
    comptime desc = String(t"({d})") if d else ""
    comptime kind = get_static_string["elementwise", desc]()

    with Trace[TraceLevel.OP, target=target](
        kind,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=get_safe_task_id(context),
    ):
        comptime if is_cpu[target]():
            _elementwise_impl_cpu[
                simd_width=simd_width,
                use_blocking_impl=use_blocking_impl,
                trace_description=trace_description,
            ](func, shape=shape, ctx=Optional(context))
        else:
            _elementwise_impl_gpu[
                simd_width=simd_width,
                trace_description=trace_description,
            ](func, shape=shape, ctx=context)


# ===-----------------------------------------------------------------------===#
# Dual Elementwise (GPU only)
# ===-----------------------------------------------------------------------===#


@always_inline
def dual_elementwise[
    rank: Int,
    //,
    func_0: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    func_1: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing[_] -> None,
    simd_width: Int,
    *,
    target: StaticString = "gpu",
    pdl_level: PDLLevel = PDLLevel(1),
    _trace_description: StaticString = "dual_elementwise",
](
    shape_0: IndexList[rank],
    shape_1: IndexList[rank],
    context: DeviceContext,
) raises:
    """Executes two elementwise functions over their respective shapes in a
    single GPU kernel launch. Each thread processes elements from both shapes,
    fusing two independent elementwise passes into one.

    Parameters:
        rank: The rank of the buffers.
        func_0: The first body function.
        func_1: The second body function.
        simd_width: The SIMD vector width to use.
        target: The target to run on (must be GPU).
        pdl_level: The PDL level controlling GPU kernel overlap behavior.
        _trace_description: Description of the trace.

    Args:
        shape_0: The shape for the first function.
        shape_1: The shape for the second function.
        context: The device context to use.

    Raises:
        If the operation fails.
    """

    def func_0_unified[
        width: Int, rank: Int, alignment: Int = 1
    ](indices: IndexList[rank]) register_passable {}:
        func_0[width, rank, alignment](indices)

    def func_1_unified[
        width: Int, rank: Int, alignment: Int = 1
    ](indices: IndexList[rank]) register_passable {}:
        func_1[width, rank, alignment](indices)

    _dual_elementwise_impl[
        simd_width,
        target=target,
        pdl_level=pdl_level,
        trace_description=_trace_description,
    ](func_0_unified, func_1_unified, shape_0, shape_1, context)


@always_inline
def _dual_elementwise_impl[
    rank: Int,
    //,
    simd_width: Int,
    Func0Type: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) register_passable -> None,
    Func1Type: def[width: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) register_passable -> None,
    /,
    *,
    target: StaticString = "gpu",
    pdl_level: PDLLevel = PDLLevel(1),
    trace_description: StaticString = "dual_elementwise",
](
    func_0: Func0Type,
    func_1: Func1Type,
    shape_0: IndexList[rank],
    shape_1: IndexList[rank],
    context: DeviceContext,
) raises:
    @always_inline
    @parameter
    def description_fn() -> String:
        var s0 = trace_arg("shape_0", shape_0)
        var s1 = trace_arg("shape_1", shape_1)
        var vw = String(t"vector_width={simd_width}")
        return ";".join(Span([s0^, s1^, vw^]))

    comptime d = trace_description
    comptime desc = String(t"({d})") if d else ""
    comptime kind = get_static_string["dual_elementwise", desc]()

    with Trace[TraceLevel.OP, target=target](
        kind,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=get_safe_task_id(context),
    ):
        comptime assert is_gpu[
            target
        ](), "dual_elementwise only supports GPU target"
        _dual_elementwise_impl_gpu[
            simd_width=simd_width,
            pdl_level=pdl_level,
            trace_description=kind,
        ](
            func_0,
            func_1,
            shape_0=shape_0,
            shape_1=shape_1,
            ctx=context,
        )


# ===-----------------------------------------------------------------------===#
# stencil
# ===-----------------------------------------------------------------------===#

comptime stencil = _stencil_impl_cpu
"""Computes stencil operation in parallel.

Computes output as a function that processes input stencils, stencils are
computed as a continuous region for each output point that is determined
by map_fn : map_fn(y) -> lower_bound, upper_bound. The boundary conditions
for regions that fail out of the input domain are handled by load_fn.


Parameters:
    shape_element_type: The element dtype of the shape.
    input_shape_element_type: The element dtype of the input shape.
    rank: Input and output domain rank.
    stencil_rank: Rank of stencil subdomain slice.
    stencil_axis: Stencil subdomain axes.
    simd_width: The SIMD vector width to use.
    dtype: The input and output data dtype.
    map_fn: A function that a point in the output domain to the input co-domain.
    map_strides: A function that returns the stride for the dim.
    load_fn: A function that loads a vector of simd_width from input.
    compute_init_fn: A function that initializes vector compute over the stencil.
    compute_fn: A function the process the value computed for each point in the stencil.
    compute_finalize_fn: A function that finalizes the computation of a point in the output domain given a stencil.

Args:
    shape: The shape of the output buffer.
    input_shape: The shape of the input buffer.
    map_fn_closure: Closure mapping output points to input co-domain bounds.
    map_strides_closure: Closure returning the stride for a given dimension.
    load_fn_closure: Closure loading a SIMD vector from input.
    compute_init_fn_closure: Closure initializing the stencil accumulator.
    compute_fn_closure: Closure processing each stencil point.
    compute_finalize_fn_closure: Closure finalizing the output value.
"""

comptime stencil_gpu = _stencil_impl_gpu
"""(Naive implementation) Computes stencil operation in parallel on GPU.

Parameters:
    shape_element_type: The element dtype of the shape.
    input_shape_element_type: The element dtype of the input shape.
    rank: Input and output domain rank.
    stencil_rank: Rank of stencil subdomain slice.
    stencil_axis: Stencil subdomain axes.
    simd_width: The SIMD vector width to use.
    dtype: The input and output data dtype.
    MapFnType: A closure maps a point in the output domain to input co-domain bounds.
    MapStridesType: A closure returns the stride for each dimension.
    LoadFnType: A closure loads a SIMD vector from input.
    ComputeInitFnType: A closure initializes the stencil accumulator.
    ComputeFnType: A closure processes the value computed for each stencil point.
    ComputeFinalizeFnType: A closure finalizes the output value from the stencil result.

Args:
    ctx: The DeviceContext to use for GPU execution.
    shape: The shape of the output buffer.
    input_shape: The shape of the input buffer.
    map_func: Closure mapping output points to input co-domain bounds.
    map_strides_func: Closure returning the stride for a given dimension.
    load_func: Closure loading a SIMD vector from input.
    compute_init_func: Closure initializing the stencil accumulator.
    compute_func: Closure processing each stencil point.
    compute_finalize_func: Closure finalizing the output value.

Raises:
    If the GPU kernel launch fails.
"""
