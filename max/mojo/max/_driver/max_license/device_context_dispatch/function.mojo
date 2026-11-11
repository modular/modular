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
from std.reflection import SourceLocation
from std.sys.info import CompilationTarget

from max._driver.max_license.device_context_hal.device_context_hal import (
    HALDeviceExternalFunction,
    HALDeviceFunction,
    _HALFunctionEnqueuer,
)
from max._driver.max_license.device_context_dispatch.device_context import (
    DeviceContext,
)
from .device_context import use_legacy_backend
from max.gpu.host._launch_args import DeviceFunctionWrapper, _IdentityWrapper
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


trait _DispatchEnqueuer:
    """A context or stream that can launch a compiled function.

    Names the backend's own enqueuer for each backend, so a wrapper can hand the
    launch to the backend that is linked.
    """

    comptime _LegacyEnqueuer: _FunctionEnqueuer & Deinitable
    comptime _HALEnqueuer: _HALFunctionEnqueuer & Deinitable

    def _legacy_enqueuer(self) -> Self._LegacyEnqueuer:
        ...

    def _hal_enqueuer(self) -> Self._HALEnqueuer:
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
    comptime _HAL = HALDeviceFunction[Self.func, Self.Wrapper]
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
            self._impl = rebind_var[Self._Impl](
                Self._Legacy(
                    rebind[DeviceContext[Self.target]._Legacy](ctx._impl),
                    func_attribute=func_attribute,
                    name=name,
                )
            )
        else:
            self._impl = rebind_var[Self._Impl](
                Self._HAL(
                    rebind[DeviceContext[Self.target]._HAL](ctx._impl),
                    func_attribute=func_attribute,
                )
            )

    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend function.

        Args:
            _impl: The backend function to take ownership of.
        """
        self._impl = _impl^

    @doc_hidden
    @staticmethod
    def _wrap_legacy(var impl: Self._Legacy) -> Self:
        """Builds a wrapper around a legacy backend function.

        Args:
            impl: The backend function to take ownership of.

        Returns:
            The wrapper around `impl`.
        """
        var typed = rebind_var[Self._Impl](impl^)
        var wrapper = Self(_impl=typed^)
        return wrapper^

    @doc_hidden
    @staticmethod
    def _wrap_hal(var impl: Self._HAL) -> Self:
        """Builds a wrapper around a HAL backend function.

        Args:
            impl: The backend function to take ownership of.

        Returns:
            The wrapper around `impl`.
        """
        var typed = rebind_var[Self._Impl](impl^)
        var wrapper = Self(_impl=typed^)
        return wrapper^

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
            rebind[Self._Legacy](self._impl)._call_with_pack_checked(
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
            rebind[Self._HAL](self._impl)._call_with_pack_checked(
                ctx._hal_enqueuer(),
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
        return rebind[Self._Legacy](self._impl)

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL function.

        Only valid in a build whose backend is the HAL one.

        Returns:
            The HAL function this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return rebind[Self._HAL](self._impl)

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
        return rebind[Self._Legacy](self._impl).get_attribute(attr)

    def occupancy_max_active_blocks_per_multiprocessor(
        self, block_size: Int, dynamic_shared_mem_size: Int
    ) raises -> Int:
        """Returns the maximum active blocks per multiprocessor for a launch.

        Args:
            block_size: Threads per thread block.
            dynamic_shared_mem_size: Dynamic shared memory per block, in bytes.

        Returns:
            The maximum number of active blocks per multiprocessor.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](
                self._impl
            ).occupancy_max_active_blocks_per_multiprocessor(
                block_size, dynamic_shared_mem_size
            )
        else:
            return rebind[Self._HAL](
                self._impl
            ).occupancy_max_active_blocks_per_multiprocessor(
                block_size, dynamic_shared_mem_size
            )


struct DeviceExternalFunction[
    *, target: CompilationTarget = default_accelerator()
](Movable):
    """A device function loaded from assembly rather than compiled from Mojo."""

    # ===-------------------------------------------------------------------===#
    # Backend selection
    # ===-------------------------------------------------------------------===#

    comptime _Legacy = LegacyDeviceExternalFunction[target=Self.target]
    comptime _HAL = HALDeviceExternalFunction
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
        comptime if use_legacy_backend():
            self._impl = rebind_var[Self._Impl](
                Self._Legacy(
                    rebind[DeviceContext[Self.target]._Legacy](ctx._impl),
                    function_name=function_name^,
                    asm=asm^,
                    func_attribute=func_attribute,
                )
            )
        else:
            self._impl = rebind_var[Self._Impl](
                Self._HAL(
                    rebind[DeviceContext[Self.target]._HAL](ctx._impl),
                    function_name^,
                    asm,
                )
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

    @doc_hidden
    @staticmethod
    def _wrap_legacy(var impl: Self._Legacy) -> Self:
        """Builds a wrapper around a legacy backend function.

        Args:
            impl: The backend function to take ownership of.

        Returns:
            The wrapper around `impl`.
        """
        var typed = rebind_var[Self._Impl](impl^)
        var wrapper = Self(_impl=typed^)
        return wrapper^

    @doc_hidden
    @staticmethod
    def _wrap_hal(var impl: Self._HAL) -> Self:
        """Builds a wrapper around a HAL backend function.

        Args:
            impl: The backend function to take ownership of.

        Returns:
            The wrapper around `impl`.
        """
        var typed = rebind_var[Self._Impl](impl^)
        var wrapper = Self(_impl=typed^)
        return wrapper^

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
        return rebind_var[Self._Legacy](self._impl^)

    def unwrap_hal(deinit self) -> Self._HAL:
        """Returns the underlying HAL function.

        Only valid in a build whose backend is the HAL one.

        Returns:
            The HAL function this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return rebind_var[Self._HAL](self._impl^)

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
        return rebind[Self._Legacy](self._impl).get_attribute(attr)
