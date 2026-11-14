# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Backend-dispatching wrappers for the compiled-function types.

`DeviceFunction` repeats the backend's parameter list, including the kernel
function and the `Wrapper` trait that selects what actually gets compiled. The
`_Impl` arm names the backend struct for the linked backend, so all a caller
touches is the wrapper.

A compiled function is launched through an enqueuer, which supplies the backend
`enqueue` implementation, so the launch lives in `device_context_dispatch` and
not on these structs: `ContextLaunchTarget` here is the pair of methods one
enqueuer implements.
"""

from std.builtin.device_passable import DevicePassable
from std.collections.optional import OptionalReg
from std.compile.compile import CompiledFunctionInfo
from std.memory import ArcPointer
from std.reflection import SourceLocation
from std.sys import size_of
from std.sys.info import CompilationTarget

from max._driver.max_license.device_context_hal.device_context_hal import (
    HALDeviceFunction,
    _check_exec_config_launch_support,
    _launch_attributes_ptr,
)
from max._driver.max_license.hal import FunctionHandle, Stream, get_device_spec
from max._driver.max_license.device_context_dispatch.device_context import (
    DeviceContext,
)
from .device_context import use_legacy_backend
from max.gpu.host._launch_args import (
    DenseArgs,
    DeviceFunctionWrapper,
    _IdentityWrapper,
)
from max.gpu.host._legacy.device_context import (
    LegacyDeviceExternalFunction,
    LegacyDeviceFunction,
    _FunctionEnqueuer,
)
from max.gpu.host._legacy.constant_memory_mapping import ConstantMemoryMapping
from max.gpu.host._legacy.dim import Dim
from max.gpu.host._legacy.func_attribute import FuncAttribute
from max.gpu.host._legacy.launch_attribute import LaunchAttribute
from max.gpu.host.func_attribute import Attribute
from max.gpu.host.info import default_accelerator
from max.gpu.host.launch_config import GridBlockExecutionConfig


trait _DispatchEnqueuer:
    """A context or stream that can launch a compiled function.

    Names the backend's own enqueuer for each backend, so a wrapper can hand the
    launch to the backend that is linked.
    """

    comptime _LegacyEnqueuer: _FunctionEnqueuer & Deinitable

    def _legacy_enqueuer(self) -> Self._LegacyEnqueuer:
        ...

    def _hal_stream(self) -> ArcPointer[Stream[get_device_spec[0]()]]:
        ...


struct DeviceFunction[
    ret_type: AnyType,
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(*args: *declared_arg_types) thin -> ret_type,
    Wrapper: DeviceFunctionWrapper = _IdentityWrapper,
    *,
    target: CompilationTarget = default_accelerator(),
    compile_options: StaticString = (target.default_compile_options()),
    link_options: StaticString = "",
    _ptxas_info_verbose: Bool = False,
](ImplicitlyCopyable):
    """Represents a compiled device function for GPU execution.

    This struct encapsulates a compiled GPU function that can be launched on a
    device.

    Parameters:
        ret_type: The return type of `func` (inferred).
        declared_arg_types: The argument types of `func` (inferred).
        func: The kernel that was compiled.
        Wrapper: The wrapper the kernel was compiled through.
        target: The target architecture it was compiled for.
        compile_options: The compilation options used.
        link_options: The linker options used.
        _ptxas_info_verbose: Whether verbose PTX assembly output is enabled.
    """

    # ===-------------------------------------------------------------------===#
    # Backend selection
    # ===-------------------------------------------------------------------===#

    comptime _Legacy = LegacyDeviceFunction[
        Self.func,
        Self.Wrapper,
        target=Self.target,
        compile_options=Self.compile_options,
        link_options=Self.link_options,
        _ptxas_info_verbose=Self._ptxas_info_verbose,
    ]
    comptime _HAL = HALDeviceFunction[
        Self.func, Self.Wrapper, target=Self.target
    ]
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _impl: Self._Impl

    def __init__(
        out self,
        ctx: DeviceContext[Self.target],
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
        name: StaticString = "",
    ) raises:
        """Compiles `func` for the device of `ctx`.

        Args:
            ctx: The device context to compile for.
            func_attribute: An attribute to use when compiling the function.
            name: The symbol name to give the function. The HAL backend
                ignores it.

        Raises:
            If the compilation fails.
        """
        comptime if use_legacy_backend():
            self._impl = Self._Legacy(
                ctx._impl,
                func_attribute=func_attribute,
                name=name,
            )
        else:
            self._impl = Self._HAL(
                ctx._impl,
                func_attribute=func_attribute,
            )

    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend function.

        Args:
            _impl: The backend function to take ownership of.
        """
        self._impl = _impl^

    @doc_hidden
    def _call_with_pack_checked[
        HostArgs: TypeList[Trait=RegisterPassable, ...],
        //,
        *Ts: DevicePassable,
        ContextT: _DispatchEnqueuer,
    ](
        imm self,
        ctx: ContextT,
        *args: *Ts,
        host_args: Tuple[*HostArgs] = (),
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: Optional[SourceLocation] = None,
    ) raises:
        """Encodes `args`, passes `host_args` by host layout, and launches.

        Args:
            ctx: The context or stream to launch on.
            args: The device arguments.
            host_args: The host-only arguments.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: The dynamic shared memory in bytes.
            attributes: The launch attributes.
            constant_memory: The constant memory mappings.
            location: The source location of the call.

        Raises:
            If the launch fails.
        """
        comptime if use_legacy_backend():
            self._impl._call_with_pack_checked(
                ctx._legacy_enqueuer(),
                *args,
                host_args=host_args,
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
                location=location,
            )
        else:
            self._impl._call_with_pack_checked(
                ctx._hal_stream(),
                *args,
                host_args=host_args,
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
                location=location,
            )

    # ===-------------------------------------------------------------------===#
    # Escape hatches
    # ===-------------------------------------------------------------------===#

    def unwrap_legacy(self) -> Self._Legacy:
        """Returns the underlying legacy function.

        Only valid in a build whose backend is the legacy one.

        Returns:
            The legacy function this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return self._impl

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL function.

        Only valid in a build whose backend is the HAL one.

        Returns:
            The HAL function this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return self._impl

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def get_attribute(self, attr: Attribute) raises -> Int:
        """Returns the value of a function attribute.

        The HAL reports none, so this is legacy-only.

        Args:
            attr: The attribute to query.

        Returns:
            The attribute's value.

        Raises:
            If querying the attribute fails.
        """
        comptime assert (
            use_legacy_backend()
        ), "get_attribute is only available on the legacy DeviceContext"
        return self._impl.get_attribute(attr)

    def occupancy_max_active_blocks_per_multiprocessor(
        self, block_size: Int, dynamic_shared_mem_size: Int
    ) raises -> Int:
        """Returns the maximum number of active blocks per multiprocessor for
        the given function.

        Args:
            block_size: The number of threads per block.
            dynamic_shared_mem_size: The size of dynamically allocated shared
                memory in bytes.

        Returns:
            The maximum number of active blocks that can run concurrently per
            multiprocessor.

        Raises:
            If the occupancy calculation fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl.occupancy_max_active_blocks_per_multiprocessor(
                block_size, dynamic_shared_mem_size
            )
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return Int(
            self._impl._inner[]
            ._context[]
            .function_occupancy_max_active_blocks(
                self._impl._inner[]._func_handle,
                Int32(block_size),
                UInt64(dynamic_shared_mem_size),
            )
        )


@fieldwise_init
struct _ExternalFunctionInner(Movable):
    """Holder for an externally-loaded HAL `FunctionHandle`.

    Borrows the handle from the context's loaded-function cache, which owns it
    and its `RuntimeBundle`, so there is nothing to unload here. Distinct from
    `_DeviceFunctionInner` because external functions don't carry
    `CompiledFunctionInfo`.
    """

    var _func_handle: FunctionHandle


@fieldwise_init
struct _HALDeviceExternalFunction(ImplicitlyCopyable, Movable):
    """The state of a `DeviceExternalFunction` on the HAL backend."""

    # The context keeps the device alive for as long as the function exists.
    var _ctx: DeviceContext.Unknown
    var _inner: ArcPointer[_ExternalFunctionInner]


struct DeviceExternalFunction[
    *, target: CompilationTarget = default_accelerator()
](Movable):
    """A device function loaded from PTX/SASS assembly rather than compiled
    from Mojo.

    This loads and runs pre-compiled GPU functions from assembly code, which is
    useful for integrating with existing CUDA/HIP code or for using specialized
    assembly optimizations.

    Parameters:
        target: The accelerator the function is loaded for.
    """

    # ===-------------------------------------------------------------------===#
    # Backend selection
    # ===-------------------------------------------------------------------===#

    comptime _Legacy = LegacyDeviceExternalFunction[target=Self.target]
    comptime _HAL = _HALDeviceExternalFunction
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _impl: Self._Impl

    def __init__(
        out self,
        ctx: DeviceContext[Self.target],
        *,
        var function_name: String,
        var asm: String,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Loads a function from assembly.

        Args:
            ctx: The device context to load the function on.
            function_name: The name of the function in the assembly.
            asm: The assembly containing the function.
            func_attribute: An attribute to use when loading the function. The
                HAL backend ignores it.

        Raises:
            If the load fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl = Self._Legacy(
                ctx._impl,
                function_name=function_name^,
                asm=asm^,
                func_attribute=func_attribute,
            )
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        var func_handle = ctx._impl._context[].get_or_load_function_dynamic(
            asm, function_name
        )
        self._impl = Self._HAL(
            ctx.into_unknown_target(),
            ArcPointer(_ExternalFunctionInner(func_handle)),
        )

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext[Self.target],
        info: CompiledFunctionInfo,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Loads a function from the result of compiling it to object code.

        Args:
            ctx: The device context to load the function on.
            info: The result of compiling the function to object code.
            func_attribute: An attribute to use when loading the function. The
                HAL backend ignores it.

        Raises:
            If `info` is not object code or the load fails.
        """
        if info.emission_kind != "object":
            raise Error("the function is not compiled to object code")
        self = Self(
            ctx,
            function_name=info.function_name,
            asm=info.asm,
            func_attribute=func_attribute,
        )

    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend function.

        Args:
            _impl: The backend function to take ownership of.
        """
        self._impl = _impl^

    # ===-------------------------------------------------------------------===#
    # Escape hatches
    # ===-------------------------------------------------------------------===#

    def unwrap_legacy(deinit self) -> Self._Legacy:
        """Returns the underlying legacy function.

        Only valid in a build whose backend is the legacy one.

        Returns:
            The legacy function this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return self._impl^

    def unwrap_hal(deinit self) -> Self._HAL:
        """Returns the underlying HAL function.

        Only valid in a build whose backend is the HAL one.

        Returns:
            The HAL function this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return self._impl^

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def get_attribute(self, attr: Attribute) raises -> Int:
        """Returns the value of a function attribute.

        The HAL reports none, so this is legacy-only.

        Args:
            attr: The attribute to query.

        Returns:
            The attribute's value.

        Raises:
            If querying the attribute fails.
        """
        comptime assert (
            use_legacy_backend()
        ), "get_attribute is only available on the legacy DeviceContext"
        return self._impl.get_attribute(attr)

    # ===-------------------------------------------------------------------===#
    # HAL launch
    # ===-------------------------------------------------------------------===#

    @inline(.always)
    def _call_with_pack[
        *Ts: AnyType,
    ](
        self,
        stream: ArcPointer[Stream[get_device_spec[0]()]],
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: Optional[SourceLocation] = None,
    ) raises where not use_legacy_backend():
        # Launch `attributes` (e.g. PDL / programmatic stream serialization,
        # cluster dimensions) are forwarded to the plugin via the attribute
        # array assembled below. `constant_memory` mappings DO change
        # semantics and are not yet plumbed through, so refuse them rather
        # than silently dropping them.
        if case .Some(cluster_dim) = cluster_dim:
            attributes.append(LaunchAttribute.from_cluster_dim(cluster_dim))
        if constant_memory:
            raise Error(
                "HAL DeviceContext.enqueue_function does not support"
                " `constant_memory` mappings."
            )

        comptime num_args = Ts.length
        var dense_args = DenseArgs[ImmutAnyOrigin, num_args + 1]()

        comptime for i in range(num_args):
            dense_args.addrs.unsafe_get(i) = (
                Pointer(to=args[i])
                .unsafe_bitcast[NoneType]()
                .as_unsafe_any_origin()
            )

        def _populate_arg_sizes[i: Int]() {mut dense_args}:
            dense_args.sizes.unsafe_get(i) = UInt64(size_of[Ts[i]]())

        comptime for i in range(num_args):
            _populate_arg_sizes[i]()

        # Kernels that use `with PDL()` emit `griddepcontrol` instructions
        # and require the launch to be configured with the matching attribute;
        # dropping it faults the launch.
        var attr_ptr = _launch_attributes_ptr(attributes)

        stream[].execute(
            self._impl._inner[]._func_handle,
            grid=(
                UInt32(grid_dim.x()),
                UInt32(grid_dim.y()),
                UInt32(grid_dim.z()),
            ),
            block=(
                UInt32(block_dim.x()),
                UInt32(block_dim.y()),
                UInt32(block_dim.z()),
            ),
            args=dense_args.addrs.unsafe_ptr(),
            arg_sizes=dense_args.sizes.unsafe_ptr(),
            num_args=UInt32(num_args),
            shared_mem_bytes=UInt32(shared_mem_bytes.or_else(0)),
            attributes=attr_ptr,
            num_attributes=UInt32(len(attributes)),
        )

        # Keep attributes alive until the plugin has read them.
        _ = attributes^

    @inline(.always)
    def _call_with_pack[
        ExecutionConfigType: GridBlockExecutionConfig,
        //,
        *Ts: AnyType,
    ](
        self,
        stream: ArcPointer[Stream[get_device_spec[0]()]],
        execution_config: ExecutionConfigType,
        *args: *Ts,
        location: Optional[SourceLocation] = None,
    ) raises where not use_legacy_backend():
        _check_exec_config_launch_support[ExecutionConfigType, Self.target]()

        comptime num_args = Ts.length
        var dense_args = DenseArgs[ImmutAnyOrigin, num_args + 1]()

        comptime for i in range(num_args):
            dense_args.addrs.unsafe_get(i) = (
                Pointer(to=args[i])
                .unsafe_bitcast[NoneType]()
                .as_unsafe_any_origin()
            )

        def _populate_arg_sizes[i: Int]() {mut dense_args}:
            dense_args.sizes.unsafe_get(i) = UInt64(size_of[Ts[i]]())

        comptime for i in range(num_args):
            _populate_arg_sizes[i]()

        stream[].execute(
            self._impl._inner[]._func_handle,
            execution_config,
            args=dense_args.addrs.unsafe_ptr(),
            arg_sizes=dense_args.sizes.unsafe_ptr(),
            num_args=UInt32(num_args),
        )
