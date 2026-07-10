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

"""This module provides functionality for building and replaying device graphs.
A device graph captures a sequence of GPU operations (such as kernel launches,
memory copies, and memsets) as a reusable graph that can be replayed at a lower
overhead than re-enqueueing each operation individually. The main entry point
is [`DeviceGraph.create()`](/docs/std/gpu/host/device_graph/DeviceGraph/#create),
which hands a [`DeviceGraphBuilder`](/docs/std/gpu/host/device_graph/DeviceGraphBuilder/)
to a scoped callback."""

from . import (
    ConstantMemoryMapping,
    Dim,
    FuncAttribute,
    LaunchAttribute,
)

from std.collections.optional import OptionalReg
from std.ffi import c_size_t, external_call, _CPointer
from std.sys import bit_width_of, size_of
from std.memory.unsafe import bitcast
from std.reflection import call_location
from std.builtin.device_passable import DevicePassable
from std.runtime.async_value import AnyAsyncValueRef

from .device_context import (
    DeviceBuffer,
    DeviceContext,
    DeviceFunction,
    HostBuffer,
    _check_dim,
    _checked,
    _CString,
    _DeviceBufferPtr,
    _DeviceContextPtr,
    _DeviceFunctionPtr,
    _DumpPath,
    _FunctionEnqueuer,
)


struct _DeviceGraphBuilderCpp:
    pass


struct _DeviceGraphCpp:
    pass


comptime _DeviceGraphBuilderPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceGraphBuilderCpp, origin]

comptime _DeviceGraphPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceGraphCpp, origin]


@fieldwise_init
struct DeviceGraphNode[arena_origin: ImmutOrigin](
    TrivialRegisterPassable, Writable
):
    """A handle to a node in an under-construction device graph.

    Returned by node-adding methods on `DeviceGraphBuilder` such as
    `add_function`, `add_copy`, and `add_memset`. The handle can be used to
    refer to the node from later API calls (for example, when expressing
    explicit dependency edges).

    Parameters:
        arena_origin: Origin of the `DeviceGraph.create` scope that produced
            this handle. Branding ties the handle's usability to that scope,
            so a node cannot be used outside the builder callback or mixed
            into a different graph.
    """

    var id: Int32
    """Opaque integer identifier of the node within its graph builder."""

    @always_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes a human-readable representation of this node handle.

        Args:
            writer: The writer to output to.
        """
        writer.write("DeviceGraphNode(id=", self.id, ")")


@doc_hidden
@fieldwise_init
struct _GraphDepArgs(TrivialRegisterPassable):
    """C ABI representation of the dependency list passed to the
    `AsyncRT_DeviceGraphBuilder_add*` exports.

    `count` is the (non-negative) number of `Int32` node ids that `ids`
    points to. When `count == 0`, `ids` may be a dangling pointer (the C
    side never dereferences it).
    """

    var ids: UnsafePointer[Int32, ImmutUntrackedOrigin]
    var count: Int64


@doc_hidden
@always_inline
def _pack_dep_args[
    o: ImmutOrigin
](deps: List[DeviceGraphNode[o]]) -> _GraphDepArgs:
    """Packs an explicit dependency list into the (ids, count) pair used by
    the AsyncRT_DeviceGraphBuilder_add* C ABI exports.

    `DeviceGraphNode` is a single-Int32 struct, so `List.unsafe_ptr()` can
    be bitcast directly to `UnsafePointer[Int32]`. The matching C++ side
    static_asserts this layout invariant in MojoBindings.cpp.

    The returned `ids` pointer borrows from the input `deps` and is only
    valid for as long as `deps` is alive at the call site.
    """
    return _GraphDepArgs(
        ids=deps.unsafe_ptr()
        .bitcast[Int32]()
        .unsafe_origin_cast[ImmutUntrackedOrigin](),
        count=Int64(len(deps)),
    )


struct DeviceGraph(ImplicitlyCopyable):
    """Represents an instantiated device graph that can be replayed.

    A `DeviceGraph` captures a sequence of GPU operations (such as kernel
    launches) as a reusable graph. Once instantiated from a
    `DeviceGraphBuilder`, the graph can be replayed multiple times at a
    lower overhead than re-enqueueing each operation individually.

    To obtain a `DeviceGraph`, use
    [`DeviceGraph.create()`](/docs/std/gpu/host/device_context/DeviceGraph/#create).
    """

    var _handle: _DeviceGraphPtr[mut=True]

    @doc_hidden
    def __init__(out self, handle: _DeviceGraphPtr[mut=True]):
        self._handle = handle

    def __init__(out self, *, copy: Self):
        """Creates a copy of an existing device graph by incrementing its
        reference count.

        Args:
            copy: The device graph to copy.
        """
        # void AsyncRT_DeviceGraph_retain(DeviceGraph *graph)
        external_call[
            "AsyncRT_DeviceGraph_retain", NoneType, _DeviceGraphPtr[mut=True]
        ](copy._handle)
        self._handle = copy._handle

    def __del__(deinit self):
        """Releases resources associated with this device graph."""
        # void AsyncRT_DeviceGraph_release(DeviceGraph *graph)
        external_call[
            "AsyncRT_DeviceGraph_release", NoneType, _DeviceGraphPtr[mut=True]
        ](self._handle)

    def replay(self) raises:
        """Replays the captured sequence of GPU operations.

        Submits the pre-captured sequence of operations for execution on the
        device. This is more efficient than re-enqueueing each operation
        individually because the graph has already been compiled and
        instantiated by the driver.

        Raises:
            If replay fails.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder

        def kernel():
            print("replaying")

        with DeviceContext() as ctx:
            var compiled_fn = ctx.compile_function[kernel]()

            def build(mut builder: DeviceGraphBuilder) raises {read}:
                _ = builder.add_function(
                    compiled_fn, grid_dim=1, block_dim=1, dependencies=[]
                )

            var graph = DeviceGraph.create(ctx, build)
            graph.replay()
            graph.replay()  # replay as many times as needed
            ctx.synchronize()
        ```
        """
        # const char *AsyncRT_DeviceGraph_replay(DeviceGraph *graph)
        _checked(
            external_call[
                "AsyncRT_DeviceGraph_replay",
                _CString[],
                _DeviceGraphPtr[mut=True],
            ](self._handle)
        )

    @staticmethod
    def create(
        ctx: DeviceContext,
        build: Some[def[o: ImmutOrigin](mut DeviceGraphBuilder[o]) raises],
    ) raises -> DeviceGraph:
        """Builds and instantiates a device graph within a scoped callback.

        Calls `build` with a fresh `DeviceGraphBuilder`, then instantiates the
        result into a replayable `DeviceGraph`. The builder, and any
        `DeviceGraphNode` handles obtained from it, are valid only for the
        duration of `build`: their origin is scoped to this call and cannot
        escape it, so a node handle cannot be stored beyond the callback or
        used with a different graph.

        Args:
            ctx: Device context for the target device.
            build: Callback that adds nodes to the supplied builder. It
                receives the builder by mutable reference and therefore
                cannot instantiate it directly; instantiation happens here
                once the callback returns.

        Returns:
            The instantiated device graph.

        Raises:
            If graph builder creation, `build`, or instantiation fails.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext, DeviceGraphBuilder

        def kernel(x: Int):
            print("Value:", x)

        with DeviceContext() as ctx:
            var compiled_fn = ctx.compile_function[kernel]()

            def build(mut builder: DeviceGraphBuilder) raises {read}:
                _ = builder.add_function(
                    compiled_fn, 42, grid_dim=1, block_dim=1, dependencies=[]
                )

            var graph = DeviceGraph.create(ctx, build)
            graph.replay()
            ctx.synchronize()
        ```
        """
        var result: _DeviceGraphBuilderPtr[mut=True] = {}
        # const char *AsyncRT_DeviceContext_createGraphBuilder(
        #     DeviceGraphBuilder **result, DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_createGraphBuilder",
                _CString[],
                UnsafePointer[
                    _DeviceGraphBuilderPtr[mut=True], origin_of(result)
                ],
                _DeviceContextPtr[mut=True],
            ](
                UnsafePointer(to=result),
                ctx._handle,
            )
        )
        var arena: Int = 0
        var builder = DeviceGraphBuilder[origin_of(arena)](result, ctx)
        build(builder)
        return builder^.instantiate()


struct DeviceGraphBuilder[arena_origin: ImmutOrigin](Movable):
    """Builder for explicit device graph construction.

    A `DeviceGraphBuilder` is handed to the callback passed to
    [`DeviceGraph.create()`](/docs/std/gpu/host/device_context/DeviceGraph/#create).
    Callers add kernel nodes via `add_function()` from within that callback,
    which then instantiates a reusable `DeviceGraph`.

    The builder, and any `DeviceGraphNode` handles it produces, are valid only
    for the duration of the callback: their origin is scoped to the
    `DeviceGraph.create` call and cannot escape it.

    Parameters:
        arena_origin: Origin of the enclosing `DeviceGraph.create` scope.

    Example:

    ```mojo
    from std.gpu.host import DeviceContext, DeviceGraphBuilder

    def kernel(x: Int):
        print("Value:", x)

    with DeviceContext() as ctx:
        var compiled_fn = ctx.compile_function[kernel]()

        def build(mut builder: DeviceGraphBuilder) raises {read}:
            _ = builder.add_function(
                compiled_fn, 42, grid_dim=1, block_dim=1, dependencies=[]
            )

        var graph = DeviceGraph.create(ctx, build)
        graph.replay()
        ctx.synchronize()
    ```
    """

    comptime Node = DeviceGraphNode[Self.arena_origin]
    """Node handle type produced by this builder, branded with the builder's
    `DeviceGraph.create` scope origin."""

    var _handle: _DeviceGraphBuilderPtr[mut=True]
    """Handle to the underlying ref-counted driver builder."""

    var _ctx: DeviceContext
    """The backing device context used to create the builder."""

    var _implicit_deps: List[Self.Node]
    """Ambient predecessor edges injected into every node added while a
    `region` scope is active.

    Outside such a scope this is empty and node-adding methods behave exactly
    as their `dependencies` argument specifies. While a scope is active,
    `region` pushes the scope's predecessor handles here so each
    `add_*` call unions them into its own `dependencies`, which is what makes
    the scope's nodes depend on the scope's incoming predecessors.
    """

    @doc_hidden
    def __init__(
        out self,
        handle: _DeviceGraphBuilderPtr[mut=True],
        ctx: DeviceContext,
    ):
        self._handle = handle
        self._ctx = ctx
        self._implicit_deps = []

    @always_inline
    def context(self) -> DeviceContext:
        """Returns the device context this builder records against.

        Unlike the `context()` accessors on buffer types, this is a non-raising
        read of the builder's stored device context (the `def` declares no
        `raises`).

        Returns:
            The `DeviceContext` backing this builder.
        """
        return self._ctx

    @doc_hidden
    @always_inline
    def _merge_implicit(
        self, var dependencies: List[Self.Node]
    ) -> List[Self.Node]:
        """Unions the active ambient predecessor set into `dependencies`.

        Returns `dependencies` unchanged when no `region` scope
        is active (the common case), so node-adding outside a scope is
        unaffected. The ambient edges are unioned in (order is irrelevant — the
        dependency list is an unordered predecessor set).
        """
        if len(self._implicit_deps) == 0:
            return dependencies^

        dependencies.extend(Span(self._implicit_deps))
        return dependencies^

    def __del__(deinit self):
        """Releases resources associated with this graph builder."""
        # void AsyncRT_DeviceGraphBuilder_release(DeviceGraphBuilder *builder)
        external_call[
            "AsyncRT_DeviceGraphBuilder_release",
            NoneType,
            _DeviceGraphBuilderPtr[mut=True],
        ](self._handle)

    @doc_hidden
    def _last_node_id(self) -> Optional[Int32]:
        """Returns the id of the most recently added node, or None if no
        nodes have been added yet.

        Cannot fail. Used by `_last_node` and `region`
        to query the builder's current state.
        """
        # int32_t AsyncRT_DeviceGraphBuilder_lastNodeIdOrNone(
        #     DeviceGraphBuilder *builder)
        var id = external_call[
            "AsyncRT_DeviceGraphBuilder_lastNodeIdOrNone",
            Int32,
            _DeviceGraphBuilderPtr[mut=True],
        ](self._handle)

        if id < 0:
            return None
        return id

    @doc_hidden
    def _last_node(self) -> Optional[Self.Node]:
        """Returns a handle to the most recently added node, or `None`
        if no nodes have been added yet.

        Used internally by the public `add_*` methods to retrieve the
        handle of a node they just added; those call sites always expect
        a `Some` result and unwrap via `.value()`. The handle is branded
        with the builder's `arena_origin` (a stable struct parameter), so it
        ties to the enclosing `DeviceGraph.create` scope.
        """
        var id = self._last_node_id()
        if id:
            return Self.Node(id.value())
        return None

    @parameter
    @always_inline
    def add_function[
        *Ts: DevicePassable
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        var dependencies: List[Self.Node] = [],
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
    ) raises -> Self.Node:
        """Adds a type-checked compiled kernel function as a node in this graph.

        Parameters:
            Ts: Argument types (must be `DevicePassable`).

        Args:
            f: The type-checked compiled function to add. Must have been
                compiled via `DeviceContext.compile_function()`.
            args: Arguments to pass to the kernel.
            grid_dim: Dimensions of the compute grid.
            block_dim: Dimensions of each thread block.
            dependencies: Explicit list of predecessor node handles. An
                empty list makes the new node a graph root with no
                predecessors; a non-empty list uses those exact handles
                as predecessors.
            cluster_dim: Cluster dimensions (optional).
            shared_mem_bytes: Amount of dynamic shared memory per block.
            attributes: Launch attributes.
            constant_memory: Constant memory mappings.

        Returns:
            A handle to the newly added kernel-dispatch node.

        Raises:
            If adding the node fails.
        """
        _check_dim["DeviceGraphBuilder.add_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceGraphBuilder.add_function", "block_dim"](
            block_dim, location=call_location()
        )
        dependencies = self._merge_implicit(dependencies^)
        # Build a transient enqueuer that pairs the builder handle with the
        # caller-supplied deps. It implements `_FunctionEnqueuer` so the
        # trait machinery in `_call_with_pack_checked` routes the call into
        # our C ABI, deps and all. (`_DeviceGraphBuilderEnqueuer` is defined
        # below `DeviceGraphBuilder` because it borrows `Self`.)
        var enqueuer = _DeviceGraphBuilderEnqueuer(self, dependencies^)
        f._call_with_pack_checked(
            enqueuer,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=call_location(),
        )
        return self._last_node().value()

    @always_inline
    def add_function[
        FuncType: def() -> None,
        //,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        func: FuncType,
        grid_dim: Dim,
        block_dim: Dim,
        *,
        var dependencies: List[Self.Node] = [],
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
    ) raises -> Self.Node:
        """Compiles and adds a capturing kernel closure as a node in this graph.

        This overload is for kernels that capture variables from their
        enclosing scope using the `{var}` capture syntax. Compilation is
        performed automatically using the `DeviceContext` that created this
        builder, so no separate compile step is needed.

        Parameters:
            FuncType: The type of the closure function (usually inferred).
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            func: The capturing kernel closure to compile and add as a graph
                node.
            grid_dim: Dimensions of the compute grid.
            block_dim: Dimensions of each thread block.
            dependencies: Explicit list of predecessor node handles. An
                empty list makes the new node a graph root with no
                predecessors; a non-empty list uses those exact handles
                as predecessors.
            cluster_dim: Cluster dimensions (optional).
            shared_mem_bytes: Amount of dynamic shared memory per block.
            attributes: Launch attributes.
            constant_memory: Constant memory mappings.

        Returns:
            A handle to the newly added kernel-dispatch node.

        Raises:
            If adding the node fails.

        Example:

        ```mojo
        from std.gpu import global_idx
        from std.gpu.host import DeviceContext, DeviceGraphBuilder

        with DeviceContext() as ctx:
            var scale: Float32 = 2.0
            var buf = ctx.enqueue_create_buffer[DType.float32](256)
            var ptr = buf.unsafe_ptr()

            def scale_kernel() {var}:
                var i = global_idx.x
                ptr[i] = Float32(i) * scale

            def build(mut builder: DeviceGraphBuilder) raises {read}:
                _ = builder.add_function(
                    scale_kernel, grid_dim=1, block_dim=256, dependencies=[]
                )

            var graph = DeviceGraph.create(ctx, build)
            graph.replay()
            ctx.synchronize()
        ```
        """
        _check_dim["DeviceGraphBuilder.add_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceGraphBuilder.add_function", "block_dim"](
            block_dim, location=call_location()
        )
        var compiled = DeviceFunction[
            FuncType.__call__,
            TypeList.of[Trait=AnyType](),
            target=DeviceContext.default_device_info.target(),
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](self._ctx)
        compiled.dump_rep[
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
        ]()
        dependencies = self._merge_implicit(dependencies^)
        # Build a transient enqueuer that pairs the builder handle with the
        # caller-supplied deps. It implements `_FunctionEnqueuer` so the
        # trait machinery in `_call_with_pack` routes the call into our
        # C ABI, deps and all. (`_DeviceGraphBuilderEnqueuer` is defined
        # below `DeviceGraphBuilder` because it borrows `Self`.)
        var enqueuer = _DeviceGraphBuilderEnqueuer(self, dependencies^)
        compiled._call_with_pack(
            enqueuer,
            func,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )
        return self._last_node().value()

    @parameter
    @always_inline
    def add_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
        link_options: StaticString = "",
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *args: *actual_arg_types,
        grid_dim: Dim,
        block_dim: Dim,
        var dependencies: List[Self.Node] = [],
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises -> Self.Node:
        """Compiles and adds a kernel function as a node in this graph.

        This overload takes the kernel as a compile-time parameter and
        compiles it automatically using the `DeviceContext` that created this
        builder, so no separate `DeviceContext.compile_function()` step is
        needed. It mirrors the parameter-based
        [`DeviceContext.enqueue_function()`](/docs/std/gpu/host/device_context/DeviceContext/#enqueue_function)
        overload for the non-graph path.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device
                function.
            func: The function to compile and add as a graph node.
            actual_arg_types: The types of the arguments being passed to the
                function.
            link_options: Additional linker flags and options as a string.
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: Dimensions of the compute grid.
            block_dim: Dimensions of each thread block.
            dependencies: Explicit list of predecessor node handles. An
                empty list makes the new node a graph root with no
                predecessors; a non-empty list uses those exact handles
                as predecessors.
            cluster_dim: Cluster dimensions (optional).
            shared_mem_bytes: Amount of dynamic shared memory per block.
            attributes: Launch attributes.
            constant_memory: Constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.

        Returns:
            A handle to the newly added kernel-dispatch node.

        Raises:
            If adding the node fails.

        You can pass the function directly to `add_function` without compiling
        it first:

        ```mojo
        from std.gpu.host import DeviceContext, DeviceGraphBuilder

        def kernel(x: Int):
            print("Value:", x)

        with DeviceContext() as ctx:
            def build(mut builder: DeviceGraphBuilder) raises {read}:
                _ = builder.add_function[kernel](
                    42, grid_dim=1, block_dim=1, dependencies=[]
                )

            var graph = DeviceGraph.create(ctx, build)
            graph.replay()
            ctx.synchronize()
        ```
        """
        _check_dim["DeviceGraphBuilder.add_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceGraphBuilder.add_function", "block_dim"](
            block_dim, location=call_location()
        )

        # If shared_mem_bytes is specified but func_attribute is not,
        # automatically set MAX_DYNAMIC_SHARED_SIZE_BYTES if needed (>48KB)
        var inferred_func_attribute = func_attribute
        if not func_attribute and shared_mem_bytes:
            var max_shared = self._ctx._get_max_dynamic_shared_memory_bytes(
                shared_mem_bytes.value()
            )
            if max_shared > 0:
                inferred_func_attribute = (
                    FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(max_shared)
                )

        var gpu_kernel = self._ctx.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            link_options=link_options,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=inferred_func_attribute)

        return self.add_function(
            gpu_kernel,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            dependencies=dependencies^,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

    def add_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype, ...],
        src_buf: HostBuffer[dtype, ...],
        *,
        var dependencies: List[Self.Node] = [],
    ) raises -> Self.Node:
        """Adds a host-to-device memcpy node to the graph.

        The number of bytes copied is determined by the size of the device
        buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Host buffer to copy from.
            dependencies: Explicit list of predecessor node handles. An
                empty list makes the new node a graph root with no
                predecessors; a non-empty list uses those exact handles
                as predecessors.

        Returns:
            A handle to the newly added memcpy node.

        Raises:
            If adding the node fails.
        """
        dependencies = self._merge_implicit(dependencies^)
        var dep_args = _pack_dep_args(dependencies)
        # const char *AsyncRT_DeviceGraphBuilder_addCopyHostToDevice(
        #     DeviceGraphBuilder *builder, DeviceBuffer *dst, const void *src,
        #     const int32_t *depIds, int64_t numDeps)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addCopyHostToDevice",
                _CString[],
            ](
                self._handle,
                dst_buf._handle,
                src_buf._host_ptr,
                dep_args.ids,
                dep_args.count,
            )
        )
        return self._last_node().value()

    def add_copy[
        dtype: DType
    ](
        self,
        dst_buf: HostBuffer[dtype, ...],
        src_buf: DeviceBuffer[dtype, ...],
        *,
        var dependencies: List[Self.Node] = [],
    ) raises -> Self.Node:
        """Adds a device-to-host memcpy node to the graph.

        The number of bytes copied is determined by the size of the device
        buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Host buffer to copy to.
            src_buf: Device buffer to copy from.
            dependencies: Explicit list of predecessor node handles. An
                empty list makes the new node a graph root with no
                predecessors; a non-empty list uses those exact handles
                as predecessors.

        Returns:
            A handle to the newly added memcpy node.

        Raises:
            If adding the node fails.
        """
        dependencies = self._merge_implicit(dependencies^)
        var dep_args = _pack_dep_args(dependencies)
        # const char *AsyncRT_DeviceGraphBuilder_addCopyDeviceToHost(
        #     DeviceGraphBuilder *builder, void *dst, DeviceBuffer *src,
        #     const int32_t *depIds, int64_t numDeps)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addCopyDeviceToHost",
                _CString[],
            ](
                self._handle,
                dst_buf._host_ptr,
                src_buf._handle,
                dep_args.ids,
                dep_args.count,
            )
        )
        return self._last_node().value()

    def add_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype, ...],
        src_buf: DeviceBuffer[dtype, ...],
        *,
        var dependencies: List[Self.Node] = [],
    ) raises -> Self.Node:
        """Adds a device-to-device memcpy node to the graph.

        Both buffers must belong to the same context as this builder;
        cross-context copies are not supported in graphs. The number of bytes
        copied is determined by the size of the source buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be the same size as
                `dst_buf`.
            dependencies: Explicit list of predecessor node handles. An
                empty list makes the new node a graph root with no
                predecessors; a non-empty list uses those exact handles
                as predecessors.

        Returns:
            A handle to the newly added memcpy node.

        Raises:
            If adding the node fails.
        """
        dependencies = self._merge_implicit(dependencies^)
        var dep_args = _pack_dep_args(dependencies)
        # const char *AsyncRT_DeviceGraphBuilder_addCopyDeviceToDevice(
        #     DeviceGraphBuilder *builder, DeviceBuffer *dst, DeviceBuffer *src,
        #     const int32_t *depIds, int64_t numDeps)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addCopyDeviceToDevice",
                _CString[],
            ](
                self._handle,
                dst_buf._handle,
                src_buf._handle,
                dep_args.ids,
                dep_args.count,
            )
        )
        return self._last_node().value()

    def add_memset[
        dtype: DType
    ](
        self,
        dst: DeviceBuffer[dtype, ...],
        val: Scalar[dtype],
        *,
        var dependencies: List[Self.Node] = [],
    ) raises -> Self.Node:
        """Adds a memset node to the graph that sets all elements of `dst` to
        `val`.

        Parameters:
            dtype: Type of the data stored in the buffer.

        Args:
            dst: Destination buffer.
            val: Value to set all elements of `dst` to.
            dependencies: Explicit list of predecessor node handles. An
                empty list makes the new node a graph root with no
                predecessors; a non-empty list uses those exact handles
                as predecessors.

        Returns:
            A handle to the newly added memset node.

        Raises:
            If adding the node fails. The underlying graph APIs cannot express
            an 8-byte memset whose high and low 32-bit halves differ as a
            single node, so such patterns will return an error.
        """
        comptime bitwidth = bit_width_of[dtype]()
        comptime assert (
            bitwidth == 8 or bitwidth == 16 or bitwidth == 32 or bitwidth == 64
        ), "bitwidth of memset dtype must be one of [8,16,32,64]"
        var value: UInt64

        comptime if bitwidth == 8:
            value = UInt64(Int(bitcast[DType.uint8, 1](val)))
        elif bitwidth == 16:
            value = UInt64(Int(bitcast[DType.uint16, 1](val)))
        elif bitwidth == 32:
            value = UInt64(bitcast[DType.uint32, 1](val))
        else:
            value = bitcast[DType.uint64, 1](val)

        dependencies = self._merge_implicit(dependencies^)
        var dep_args = _pack_dep_args(dependencies)
        # const char *AsyncRT_DeviceGraphBuilder_addSetMemory(
        #     DeviceGraphBuilder *builder, DeviceBuffer *dst, uint64_t val,
        #     size_t valSize, const int32_t *depIds, int64_t numDeps)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addSetMemory",
                _CString[],
                _DeviceGraphBuilderPtr[mut=True],
                _DeviceBufferPtr[mut=True],
                UInt64,
                c_size_t,
                UnsafePointer[Int32, ImmutAnyOrigin],
                Int64,
            ](
                self._handle,
                dst._handle,
                value,
                c_size_t(size_of[dtype]()),
                dep_args.ids.as_unsafe_any_origin(),
                dep_args.count,
            )
        )
        return self._last_node().value()

    def add_empty(
        self,
        *,
        var dependencies: List[Self.Node] = [],
    ) raises -> Self.Node:
        """Adds an empty (no-op) node to the graph.

        Empty nodes perform no work at execution time. They are used purely
        for transitive ordering: a single empty node fanned in from `m`
        predecessors and out to `n` successors expresses an `m`-to-`n`
        barrier using `m + n` edges instead of `m * n`, and serves as a
        stable handle for "the completion of this phase" when the producer
        set is not visible to the consumer.

        Args:
            dependencies: Explicit list of predecessor node handles. An
                empty list makes the new node a graph root with no
                predecessors; a non-empty list uses those exact handles
                as predecessors.

        Returns:
            A handle to the newly added empty node.

        Raises:
            If adding the node fails.
        """
        dependencies = self._merge_implicit(dependencies^)
        var dep_args = _pack_dep_args(dependencies)
        # const char *AsyncRT_DeviceGraphBuilder_addEmpty(
        #     DeviceGraphBuilder *builder, const int32_t *depIds,
        #     int64_t numDeps)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_addEmpty",
                _CString[],
                _DeviceGraphBuilderPtr[mut=True],
                UnsafePointer[Int32, ImmutAnyOrigin],
                Int64,
            ](self._handle, dep_args.ids.as_unsafe_any_origin(), dep_args.count)
        )
        return self._last_node().value()

    def region(
        mut self,
        work: Some[def[o: ImmutOrigin](mut DeviceGraphBuilder[o]) raises],
        *,
        var dependencies: List[Self.Node] = [],
    ) raises -> Self.Node:
        """Runs `work` and returns a single empty node that joins every
        node added to this builder during its execution.

        The returned handle is suitable for use as a one-element
        `dependencies=` entry on a downstream `add_*` call. The empty
        node performs no work at execution time; it exists purely as a
        fan-in barrier so the caller does not need to thread the
        producer set's individual handles to every consumer.

        Every node `work` adds also depends on the predecessors named in
        `dependencies`: while `work` runs, those handles are injected as
        ambient predecessors that each `add_*` call unions into its own
        `dependencies`. This makes the region's nodes run after the named
        predecessors without the closure having to thread the handles
        through to every `add_*` call. With the default (empty)
        `dependencies`, the region's nodes are unconstrained relative to
        earlier work.

        Args:
            work: Closure whose effects on this builder are captured. The
                builder is passed as `work`'s sole argument; the closure
                must not capture the same builder, since doing so would
                alias with this method's receiver. The closure may add
                any number of nodes (zero or more) via any of the
                `add_*` methods.
            dependencies: Predecessor node handles that every node added by
                `work` should depend on. Defaults to empty (no added
                predecessors).

        Returns:
            A handle that successors can depend on to run after everything
            `work` added. When `work` adds two or more nodes, this is a fresh
            empty node that joins them; when it adds exactly one node, that
            node is returned directly (no extra empty node); when it adds none,
            the returned empty node falls back to depending on `dependencies`
            so it still chains correctly.

        Raises:
            Anything `work` itself raises, or anything raised while
            adding the join node.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext, DeviceGraphBuilder

        with DeviceContext() as ctx:
            var buf_a = ctx.enqueue_create_buffer[DType.uint8](100)
            var buf_b = ctx.enqueue_create_buffer[DType.uint8](100)
            var buf_c = ctx.enqueue_create_buffer[DType.uint8](100)
            var host_src = ctx.enqueue_create_host_buffer[DType.uint8](100)

            def build(mut builder: DeviceGraphBuilder) raises {read}:
                def add_producers(mut b: DeviceGraphBuilder) raises {read} -> None:
                    _ = b.add_memset(buf_a, UInt8(1), dependencies=[])
                    _ = b.add_memset(buf_b, UInt8(2), dependencies=[])

                var producers_join = builder.region(add_producers)
                _ = builder.add_copy(
                    buf_c, host_src, dependencies=[producers_join]
                )

            var graph = DeviceGraph.create(ctx, build)
            graph.replay()
        ```
        """

        # Save the current set of dependencies and replace
        # self._implicit_deps with an extended version containing the original
        # plus the new dependencies.
        var saved_deps = self._implicit_deps.copy()
        self._implicit_deps.extend(Span(dependencies))

        var start_id = self._last_node_id()

        try:
            work(self)
        finally:
            # Restore the dependencies to the original value
            self._implicit_deps = saved_deps^

        var end_id = self._last_node_id()

        var deps = List[Self.Node]()

        if end_id:
            var end_val = end_id.value()
            var start_val = start_id.or_else(-1)
            deps.reserve(Int(end_val) - Int(start_val))
            for id in range(start_val + 1, end_val + 1):
                deps.append(Self.Node(Int32(id)))

        # If `work` produced no nodes, gate the join on the incoming
        # predecessors directly so a downstream consumer of the join still
        # waits for them.
        if len(deps) == 0:
            return self.add_empty(dependencies=dependencies^)

        if len(deps) == 1:
            return deps[0]

        return self.add_empty(dependencies=deps^)

    @doc_hidden
    def instantiate(var self) raises -> DeviceGraph:
        """Instantiates the constructed graph into an executable device graph.

        Finalizes the graph construction and produces a `DeviceGraph` that
        can be replayed multiple times. Called by
        `DeviceGraph.create` once the builder callback returns;
        not part of the user-facing API (the callback receives the builder by
        reference and so cannot consume it to call this directly).

        Returns:
            The instantiated device graph.

        Raises:
            If instantiation fails.
        """
        var result: _DeviceGraphPtr[mut=True] = {}
        # const char *AsyncRT_DeviceGraphBuilder_instantiate(
        #     DeviceGraph **result, DeviceGraphBuilder *builder)
        _checked(
            external_call[
                "AsyncRT_DeviceGraphBuilder_instantiate",
                _CString[],
                UnsafePointer[_DeviceGraphPtr[mut=True], origin_of(result)],
                _DeviceGraphBuilderPtr[mut=True],
            ](
                UnsafePointer(to=result),
                self._handle,
            )
        )
        return DeviceGraph(result)


@doc_hidden
struct _DeviceGraphBuilderEnqueuer[
    arena_origin: ImmutOrigin,
    builder_origin: Origin[mut=False],
](_FunctionEnqueuer):
    """Transient `_FunctionEnqueuer` pairing a `DeviceGraphBuilder` borrow
    with the dependency list for a single node addition.

    Constructed locally inside `DeviceGraphBuilder.add_function` and passed
    to `DeviceFunction._call_with_pack[_checked]` so the explicit
    dependency list can flow through the trait machinery into the C ABI
    without becoming part of the trait surface or requiring mutable state
    on `DeviceGraphBuilder` itself.

    Parameters:
        arena_origin: Origin of the enclosing `DeviceGraph.create` scope,
            shared by the borrowed builder and the dependency handles.
        builder_origin: The origin of the borrow on the parent
            `DeviceGraphBuilder`. The borrow checker enforces that this
            enqueuer cannot outlive the originating builder.
    """

    comptime Node = DeviceGraphNode[Self.arena_origin]
    """Node handle type for this enqueuer's scope origin."""

    var _builder: Pointer[
        DeviceGraphBuilder[Self.arena_origin], Self.builder_origin
    ]
    """Borrowed reference to the parent graph builder. The Mojo borrow
    checker uses `builder_origin` to ensure this enqueuer cannot outlive
    the borrow."""

    var _dependencies: List[Self.Node]
    """Explicit dependency list for the node being added. An empty list
    creates a graph root; a non-empty list specifies exact predecessor
    edges."""

    @always_inline
    def __init__(
        out self,
        ref[Self.builder_origin] builder: DeviceGraphBuilder[Self.arena_origin],
        var dependencies: List[Self.Node],
    ):
        """Initializes the transient enqueuer with a borrowed builder and
        the dependency list to apply to the next node addition.

        Args:
            builder: The parent `DeviceGraphBuilder` whose handle is used
                for the C ABI call. Borrowed for the lifetime of this
                enqueuer.
            dependencies: Explicit dependency list for the node about to
                be added. See the field docstring on `_dependencies` for
                the meaning of each value.
        """
        self._builder = Pointer(to=builder)
        self._dependencies = dependencies^

    @always_inline
    def enqueue[
        args_origin: MutOrigin, //
    ](
        self,
        func_handle: _DeviceFunctionPtr[mut=True],
        grid_dim: Dim,
        block_dim: Dim,
        shared_mem_bytes: Int,
        attributes: UnsafePointer[mut=True, LaunchAttribute, _],
        num_attributes: Int,
        args: UnsafePointer[mut=True, OpaquePointer[args_origin], _],
        arg_count: UInt32,
        arg_sizes: OptionalUnsafePointer[mut=True, UInt64, _],
    ) -> _CString[]:
        """Adds a kernel-dispatch node to the borrowed graph builder.

        Forwards to `AsyncRT_DeviceGraphBuilder_addFunctionDirect`,
        attaching the dependency list captured at construction time so it
        is applied to the node being added. See `_FunctionEnqueuer.enqueue`
        for the full contract.

        Args:
            func_handle: Handle to the compiled `DeviceFunction` to launch.
            grid_dim: Grid dimensions (number of thread blocks).
            block_dim: Block dimensions (number of threads per block).
            shared_mem_bytes: Bytes of dynamic shared memory per block.
            attributes: Pointer to the launch attributes array.
            num_attributes: Number of entries in `attributes`.
            args: Pointer to the array of argument value pointers.
            arg_count: Number of entries in `args`.
            arg_sizes: Optional pointer to the per-argument sizes in bytes.

        Returns:
            A C-string carrying an error message on failure, or an empty
            string on success.
        """
        var dep_args = _pack_dep_args(self._dependencies)
        return external_call[
            "AsyncRT_DeviceGraphBuilder_addFunctionDirect", _CString[]
        ](
            self._builder[]._handle,
            func_handle,
            grid_dim.x(),
            grid_dim.y(),
            grid_dim.z(),
            block_dim.x(),
            block_dim.y(),
            block_dim.z(),
            shared_mem_bytes,
            attributes,
            num_attributes,
            args,
            arg_count,
            arg_sizes,
            dep_args.ids,
            dep_args.count,
        )
