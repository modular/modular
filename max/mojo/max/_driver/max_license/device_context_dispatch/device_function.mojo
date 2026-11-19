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
from std.memory import ArcPointer, Pointer
from std.builtin.rebind import downcast
from std.math import align_up
from std.reflection import SourceLocation, reflect
from std.sys import get_defined_bool, size_of
from std.sys.info import CompilationTarget

from max._driver.max_license.hal._machine import DeviceSpec
from max._driver.max_license.hal.execution_config import (
    GPUExecutionConfiguration,
    check_config_fields_launchable,
)
from max._plugin import MaxPluginForTarget
from max.gpu.host import (
    ClusterExecutionConfig,
    ExecutionConfig,
    LaunchAttributeID,
    NearComputeGeneralPurposeScratchpadExecutionConfig,
    _launch_args,
)
from max.gpu.host._driver_common import _check_dim_in_range
from max.gpu.host.compile import _compile_code
from max.gpu.host.nvidia import LaunchAttributeHolderExecutionConfig
from max._driver.max_license.hal import (
    Context,
    FunctionHandle,
    Stream,
    get_device_spec,
)
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
    DefaultDeviceTypeEncoder,
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


def _launch_attributes_ptr(
    mut attributes: List[LaunchAttribute],
) -> OptionalReg[OpaquePointer[MutUntrackedOrigin]]:
    # The caller keeps `attributes` alive until the plugin has read them.
    if len(attributes) == 0:
        return None
    return (
        attributes.unsafe_ptr()
        .unsafe_bitcast[NoneType]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


def _check_exec_config_launch_support[
    ExecutionConfigType: ExecutionConfig, target: CompilationTarget
]():
    comptime if conforms_to(ExecutionConfigType, ClusterExecutionConfig):
        comptime assert conforms_to(
            DeviceContext[target].ExecutionConfigType, ClusterExecutionConfig
        ), "cluster launch is not supported on this device"

    comptime if conforms_to(
        ExecutionConfigType, LaunchAttributeHolderExecutionConfig
    ):
        comptime assert conforms_to(
            DeviceContext[target].ExecutionConfigType,
            LaunchAttributeHolderExecutionConfig,
        ), "this target does not support launch attributes"

    check_config_fields_launchable[
        ExecutionConfigType, "HAL DeviceContext.enqueue_function"
    ]()


def _check_config_dims[
    func_name_for_msg: StringLiteral,
    ExecutionConfigType: ExecutionConfig,
](execution_config: ExecutionConfigType, *, location: SourceLocation) raises:
    """Checks a grid/block config's dims; any other config has none."""
    comptime if conforms_to(ExecutionConfigType, GridBlockExecutionConfig):
        _check_dim_in_range[func_name_for_msg, "grid_dim"](
            execution_config.get_grid_dim(), location=location
        )
        _check_dim_in_range[func_name_for_msg, "block_dim"](
            execution_config.get_block_dim(), location=location
        )


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


@fieldwise_init
struct _DeviceFunctionInner(Movable):
    """Wrapper around a HAL-loaded `FunctionHandle`.

    The function handle (and the `RuntimeBundle` defining it) is owned by the
    `Context`'s function cache and stays loaded for the context's lifetime;
    the held `Context` `ArcPointer` keeps it valid.
    """

    var _func_handle: FunctionHandle
    var _context: ArcPointer[Context[get_device_spec[0]()]]


@fieldwise_init
struct _HALDeviceFunction[
    ret_type: AnyType,
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(*args: *declared_arg_types) thin -> ret_type,
    Wrapper: DeviceFunctionWrapper = _IdentityWrapper,
    *,
    target: CompilationTarget = default_accelerator(),
](ImplicitlyCopyable, Movable):
    """The state of a `DeviceFunction` on the HAL backend."""

    comptime _target = DeviceSpec.target[get_device_spec[0]()]()
    comptime _wrapped_func = Self.Wrapper.wrap[Self.func]

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _ctx: DeviceContext.Unknown
    var _inner: ArcPointer[_DeviceFunctionInner]
    # The compilation result for `Self._wrapped_func`: its assembly and
    # linkage name.
    var _func_info: CompiledFunctionInfo[
        type_of(Self._wrapped_func), Self._wrapped_func, Self._target
    ]


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
    device. It provides methods for managing the function's lifecycle, copying
    data to constant memory, and dumping debug information such as assembly
    code, LLVM IR, or SASS code.

    `DeviceFunction` is typically created through the `compile_function()`
    method of a `DeviceContext` rather than directly instantiated.

    Example:

    ```mojo
    from max.gpu.host import DeviceContext

    def my_kernel():
        # Kernel implementation
        pass

    with DeviceContext() as ctx:
        # Compile the kernel
        var kernel = ctx.compile_function[my_kernel]()
        # Enqueue the kernel for execution
        ctx.enqueue_function(kernel, grid_dim=1, block_dim=1)
    ```

    Parameters:
        ret_type: The return type of `func` (inferred).
        declared_arg_types: The argument types of `func` (inferred). Launch
            arguments are checked against these.
        func: The kernel that was compiled.
        Wrapper: Selects the function that is actually compiled for `func`,
            such as a generated entry point. Defaults to compiling `func` as
            written.
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
    comptime _HAL = _HALDeviceFunction[
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
            func_attribute: An attribute to use when compiling the function. The
                HAL backend accepts only `MAX_DYNAMIC_SHARED_SIZE_BYTES`, and it
                has no effect.
            name: The symbol name to give the function. The HAL backend
                ignores it.

        Raises:
            If the compilation or function loading fails, or if the HAL backend
            is given an unsupported function attribute.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl = Self._Legacy(
                ctx._impl,
                func_attribute=func_attribute,
                name=name,
            )
            return

        if case .Some(func_attribute) = func_attribute:
            # The CUDA plugin lifts the dynamic shared memory cap to the
            # device's opt-in limit when it loads a function, and Metal and
            # HIP have no cap, so a requested cap needs nothing more.
            if (
                func_attribute.attribute
                != Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            ):
                raise Error(
                    "the function attribute '",
                    func_attribute.attribute,
                    "' is not currently supported",
                )
        var info = ctx._impl._context[]._compile_inner[
            type_of(Self._HAL._wrapped_func),
            Self._HAL._wrapped_func,
            compile_options=Self.compile_options,
            link_options=Self.link_options,
        ]()
        # The context's function cache owns the loaded module: repeated
        # constructions of the same kernel reuse the cached module.
        var func_handle = ctx._impl._context[].get_or_load_function(
            info.asm, info.function_name
        )
        self._impl = Self._HAL(
            ctx.into_unknown_target(),
            ArcPointer(_DeviceFunctionInner(func_handle, ctx._impl._context)),
            info,
        )

    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend function.

        Args:
            _impl: The backend function to take ownership of.
        """
        self._impl = _impl^

    @doc_hidden
    def __init__(out self, *, _legacy: Self._Legacy) where use_legacy_backend():
        """Wraps a legacy function.

        Only available in a build whose backend is the legacy one.

        Args:
            _legacy: The legacy function to wrap.
        """
        self._impl = _legacy

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
            # Launch `attributes` (e.g. PDL / programmatic stream
            # serialization, cluster dimensions) are forwarded to the plugin
            # via the attribute array assembled below. `constant_memory`
            # mappings DO change semantics and are not yet plumbed through, so
            # refuse them rather than silently dropping them.
            if case .Some(cluster_dim) = cluster_dim:
                attributes.append(LaunchAttribute.from_cluster_dim(cluster_dim))
            if constant_memory:
                raise Error(
                    "HAL DeviceContext.enqueue_function does not support"
                    " `constant_memory` mappings."
                )
            debug_assert(
                shared_mem_bytes.or_else(0) >= 0,
                "shared_mem_bytes must be non-negative",
            )
            var config = GPUExecutionConfiguration(grid_dim, block_dim)
            config.set_near_compute_scratchpad_usage(
                shared_mem_bytes.or_else(0)
            )

            comptime for idx in range(HostArgs.length):
                Self._check_trailing_host[HostArgs[idx], Ts.length + idx]()

            self._do_call_with_pack_checked[*Ts](
                ctx._hal_stream(),
                config,
                *args,
                host_args=host_args,
                attributes=attributes^,
                location=location,
            )

    # ===-------------------------------------------------------------------===#
    # HAL launch
    # ===-------------------------------------------------------------------===#

    @inline(.always)
    @staticmethod
    def _check_trailing_host[Host: RegisterPassable, index: Int]():
        """Require `Host` to be the declared kernel argument at `index`."""
        comptime declared_host_type = Self.declared_arg_types[index]
        comptime assert Host == declared_host_type, String(
            "host argument of type '",
            reflect[Host].name(),
            "' does not match the declared function argument type '",
            reflect[declared_host_type].name(),
            "'",
        )

    @inline(.always)
    def _check_pdl_wait[
        o: ImmOrigin
    ](
        self, attributes: Span[LaunchAttribute, o]
    ) raises where not use_legacy_backend():
        """Rejects a PDL launch of a kernel that never waits on its predecessor.

        This is the HAL copy of the legacy `DeviceFunction._check_pdl_wait`,
        which documents what the check covers and what it misses. It is off
        unless the build passes `-D MAX_CHECK_PDL_WAIT=True`.

        Parameters:
            o: The origin of `attributes`.

        Args:
            attributes: The launch attributes of the enqueue being checked.

        Raises:
            If a PDL launch targets a kernel whose PTX has no
            `griddepcontrol.wait`.
        """
        comptime if (
            get_defined_bool["MAX_CHECK_PDL_WAIT", False]()
            and Self._HAL._target.is_nvidia_gpu()
        ):
            for ref attribute in attributes:
                # The value is a Boolean in the first word; a caller can pass
                # the attribute with `False` to switch PDL off for the launch.
                if (
                    attribute.id
                    == LaunchAttributeID.PROGRAMMATIC_STREAM_SERIALIZATION
                    and attribute.value._storage[0] != 0
                ):
                    # The loaded function is an object file, so compile the
                    # kernel to PTX to look for the wait.
                    var ptx = _compile_code[
                        Self._HAL._wrapped_func,
                        emission_kind="asm",
                        target=Self._HAL._target,
                    ]().asm
                    if "griddepcontrol.wait" not in ptx:
                        raise Error(
                            "kernel '",
                            self._impl._func_info.function_name,
                            (
                                "' is launched with the PDL attribute but its"
                                " PTX has no `griddepcontrol.wait`; bind a"
                                " non-OFF `pdl_level` in the kernel's"
                                " instantiation, or call"
                                " `wait_on_dependent_grids()` before its first"
                                " read of predecessor output"
                            ),
                        )
                    return

    @inline(.always)
    def _call_closure_checked[
        FuncType: ImplicitlyCopyable & RegisterPassable & def() -> None,
        ContextT: _DispatchEnqueuer,
    ](
        self,
        ctx: ContextT,
        closure: FuncType,
        *,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim],
        shared_mem_bytes: OptionalReg[Int],
        var attributes: List[LaunchAttribute],
        var constant_memory: List[ConstantMemoryMapping],
        location: Optional[SourceLocation],
    ) raises where not use_legacy_backend():
        # Encoding makes a captured `DevicePointer` a device address.
        comptime if conforms_to(FuncType, DevicePassable):
            self._call_with_pack_checked(
                ctx,
                rebind[downcast[FuncType, DevicePassable]](closure),
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
                location=location,
            )
        else:
            self._call_with_pack_checked(
                ctx,
                host_args=(closure,),
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
                location=location,
            )

    @inline(.always)
    def _do_call_closure_checked[
        ExecutionConfigType: ExecutionConfig,
        FuncType: ImplicitlyCopyable & RegisterPassable & def() -> None,
    ](
        self,
        stream: ArcPointer[Stream[get_device_spec[0]()]],
        execution_config: ExecutionConfigType,
        closure: FuncType,
        location: Optional[SourceLocation],
    ) raises where not use_legacy_backend():
        # Encoding makes a captured `DevicePointer` a device address.
        comptime if conforms_to(FuncType, DevicePassable):
            self._do_call_with_pack_checked(
                stream,
                execution_config,
                rebind[downcast[FuncType, DevicePassable]](closure),
                location=location,
            )
        else:
            self._do_call_with_pack_checked(
                stream,
                execution_config,
                host_args=(closure,),
                location=location,
            )

    @inline(.always)
    def _do_call_with_pack_checked[
        ExecutionConfigType: ExecutionConfig,
        HostArgs: TypeList[Trait=RegisterPassable, ...],
        //,
        *Ts: DevicePassable,
    ](
        self,
        stream: ArcPointer[Stream[get_device_spec[0]()]],
        execution_config: ExecutionConfigType,
        *args: *Ts,
        host_args: Tuple[*HostArgs] = (),
        var attributes: List[LaunchAttribute] = [],
        location: Optional[SourceLocation] = None,
    ) raises where not use_legacy_backend():
        # Every launch, from a DeviceContext or a DeviceStream, ends up here,
        # so this is the one place the plugin's launcher has to be honored.
        comptime _HALHooks = MaxPluginForTarget[Self.target].HalHooks
        _check_exec_config_launch_support[ExecutionConfigType, Self.target]()
        comptime is_grid_block = conforms_to(
            ExecutionConfigType, GridBlockExecutionConfig
        )
        comptime launcher_takes_config = (
            not is_grid_block and _HALHooks.Launcher.launches_config
        )
        comptime launcher_takes_grid_block = (
            is_grid_block and _HALHooks.Launcher.overrides_grid_block
        )
        comptime if launcher_takes_config or launcher_takes_grid_block:
            comptime if HostArgs.length != 0:
                raise Error(
                    "the active HAL plugin's launcher doesn't take host-only"
                    " arguments"
                )
            # It only gets its own config or the grid and block, so refuse
            # anything else rather than drop it.
            var num_attributes = len(attributes)
            comptime if conforms_to(
                ExecutionConfigType, LaunchAttributeHolderExecutionConfig
            ):
                num_attributes += len(execution_config.get_launch_attributes())
            var clustered = False
            comptime if conforms_to(
                ExecutionConfigType, ClusterExecutionConfig
            ):
                var cluster_dim = execution_config.get_cluster_dim()
                clustered = (
                    cluster_dim.x() != 1
                    or cluster_dim.y() != 1
                    or cluster_dim.z() != 1
                )
            if num_attributes != 0 or clustered:
                raise Error(
                    "the active HAL plugin's launcher doesn't take cluster"
                    " dims or launch attributes"
                )
            stream[]._launch[
                Self.Wrapper,
                Self.declared_arg_types,
                "HAL DeviceContext.enqueue_function",
            ](self._impl._inner[]._func_handle, execution_config, *args)
        elif is_grid_block:
            self._do_call_grid_block_with_pack_checked(
                stream,
                rebind[downcast[ExecutionConfigType, GridBlockExecutionConfig]](
                    execution_config
                ),
                *args,
                host_args=host_args,
                attributes=attributes^,
                location=location,
            )
        else:
            comptime assert False, (
                "the active HAL plugin's launcher doesn't take this execution"
                " config"
            )

    @inline(.always)
    def _do_call_grid_block_with_pack_checked[
        ExecutionConfigType: GridBlockExecutionConfig,
        HostArgs: TypeList[Trait=RegisterPassable, ...],
        //,
        *Ts: DevicePassable,
    ](
        self,
        stream: ArcPointer[Stream[get_device_spec[0]()]],
        execution_config: ExecutionConfigType,
        *args: *Ts,
        host_args: Tuple[*HostArgs],
        var attributes: List[LaunchAttribute],
        location: Optional[SourceLocation],
    ) raises where not use_legacy_backend():
        comptime extra_host_count = HostArgs.length

        self._check_pdl_wait(Span(attributes))
        comptime if conforms_to(
            ExecutionConfigType, LaunchAttributeHolderExecutionConfig
        ):
            self._check_pdl_wait(execution_config.get_launch_attributes())

        var shared_mem_bytes: Int = 0

        comptime if conforms_to(
            ExecutionConfigType,
            NearComputeGeneralPurposeScratchpadExecutionConfig,
        ):
            shared_mem_bytes = (
                execution_config.get_near_compute_scratchpad_usage()
            )

        debug_assert(
            shared_mem_bytes >= 0,
            "shared_mem_bytes must be non-negative",
        )

        comptime num_encoded = Ts.length
        comptime num_passed_args = num_encoded + extra_host_count
        var validated_args = _launch_args._validate_encodable_arguments[
            Self._HAL._target,
            DeclaredArgTypes=Self.declared_arg_types,
            ActualArgTypes=Ts,
            host_args_count=extra_host_count,
        ]()

        var num_translated_args = validated_args[0]
        comptime for i in range(extra_host_count):
            comptime size = size_of[HostArgs[i]]()
            # Skip zero-sized types (ZSTs).
            comptime if size != 0:
                num_translated_args += 1

        var translated_arg_offsets = validated_args[1].copy()

        def calculate_args_size() -> Int:
            var tmp_args_size = 8  # reserve 8 extra bytes for alignment
            comptime for i in range(num_encoded):
                comptime actual_arg_type = Ts[i]
                tmp_args_size += align_up(
                    size_of[actual_arg_type.device_type](), 8
                )
            return tmp_args_size

        comptime args_size = calculate_args_size()

        var translated_args = Array[Byte, args_size](uninitialized=True)
        var start_addr = Int(translated_args.unsafe_ptr())
        var extra_align = align_up(start_addr, 8) - start_addr

        # Allocation made at function scope to outlive the enqueue, otherwise
        # the stack bytes are free to be reused immediately. A kernel may take
        # no arguments; keep the arrays non-empty.
        var dense_args = DenseArgs[ImmutAnyOrigin, max(num_passed_args, 1)]()

        var translated_arg_idx = 0

        var device_type_encoder = DefaultDeviceTypeEncoder()

        comptime for i in range(num_encoded):
            var translated_arg_offset = translated_arg_offsets[i]
            if translated_arg_offset >= 0:
                comptime actual_arg_type = Ts[i]
                var first_word_addr = Pointer(
                    to=translated_args.unsafe_ptr()[
                        unsafe_offset=translated_arg_offset + extra_align
                    ]
                ).unsafe_bitcast[NoneType]()
                args[i]._to_device_type(device_type_encoder, first_word_addr)
                dense_args.addrs.unsafe_get(
                    translated_arg_idx
                ) = first_word_addr.as_unsafe_any_origin()
                dense_args.sizes.unsafe_get(translated_arg_idx) = UInt64(
                    size_of[
                        actual_arg_type.device_type,
                        target=device_type_encoder.target(),
                    ]()
                )
                translated_arg_idx += 1

        comptime for i in range(extra_host_count):
            comptime size = size_of[HostArgs[i]]()
            # Skip zero-sized types (ZSTs).
            if size == 0:
                continue

            var addr = (
                Pointer(to=host_args[i])
                .unsafe_bitcast[NoneType]()
                .as_unsafe_any_origin()
            )

            dense_args.addrs.unsafe_get(translated_arg_idx) = addr
            dense_args.sizes.unsafe_get(translated_arg_idx) = UInt64(size)
            translated_arg_idx += 1

        # Kernels that use `with PDL()` emit `griddepcontrol` instructions and
        # require the launch to be configured with the matching attribute;
        # dropping it faults the launch.
        var attr_ptr = _launch_attributes_ptr(attributes)

        stream[].execute(
            self._impl._inner[]._func_handle,
            execution_config,
            args=dense_args.addrs.unsafe_ptr(),
            arg_sizes=dense_args.sizes.unsafe_ptr(),
            num_args=UInt32(num_translated_args),
            attributes=attr_ptr,
            num_attributes=UInt32(len(attributes)),
        )

        # Keep these alive until the plugin has read them.
        _ = attributes^
        _ = translated_args^
        _ = dense_args^

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
        comptime __guard not use_legacy_backend() else:
            return self._impl.occupancy_max_active_blocks_per_multiprocessor(
                block_size, dynamic_shared_mem_size
            )

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
        comptime __guard not use_legacy_backend() else:
            self._impl = Self._Legacy(
                ctx._impl,
                function_name=function_name^,
                asm=asm^,
                func_attribute=func_attribute,
            )
            return

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

        @__allow_legacy_any_origin_fields
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

        @__allow_legacy_any_origin_fields
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
