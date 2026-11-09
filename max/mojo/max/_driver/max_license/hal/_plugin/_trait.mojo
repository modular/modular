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

from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std._plugin.selector import Plugin
from std.sys.info import CompilationTarget

from max.gpu.host._launch_args import DeviceFunctionWrapper, _IdentityWrapper
from max.gpu.host._legacy.constant_memory_mapping import ConstantMemoryMapping
from max.gpu.host._legacy.dim import Dim
from max.gpu.host._legacy.func_attribute import FuncAttribute
from max.gpu.host._legacy.launch_attribute import LaunchAttribute

from ..device import DeviceSpec
from ..plugins import FunctionHandle
from ..stream import Stream

from .._machine import DeviceRef
from max.gpu.host import ExecutionConfig
from ..execution_config.cpu import CPUExecutionConfiguration
from ..extensions.traits import DefaultHALExtensions, HALExtensions


trait HalPlugin(Plugin):
    comptime HookFor[target: CompilationTarget]: HalPluginHooks


trait HalLauncher:
    """Launches kernels for a plugin whose launches aren't grid/block shaped.

    The HAL DeviceContext hands a grid/block `enqueue_function` to the
    launcher when `overrides_grid_block` is set, with the compiled function's
    wrapper and declared argument types so the launcher can pack the
    arguments its runtime expects.
    """

    comptime overrides_grid_block: Bool
    """Whether `enqueue_grid_block` replaces grid/block launches."""

    @staticmethod
    def enqueue_grid_block[
        spec: DeviceSpec,
        Wrapper: DeviceFunctionWrapper,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        *Ts: AnyType,
    ](
        mut stream: Stream[spec],
        func: FunctionHandle,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
    ) raises:
        """Launches `func` on `stream` with `args`."""
        ...


struct DefaultHalLauncher(HalLauncher):
    """Leaves grid/block launches to the HAL DeviceContext."""

    comptime overrides_grid_block = False

    @staticmethod
    def enqueue_grid_block[
        spec: DeviceSpec,
        Wrapper: DeviceFunctionWrapper,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        *Ts: AnyType,
    ](
        mut stream: Stream[spec],
        func: FunctionHandle,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
    ) raises:
        raise Error("the active HAL plugin has no grid/block launcher")


@fieldwise_init
struct HalLaunchOptions(Movable):
    """The launch arguments of a grid/block `enqueue_function` call."""

    var grid_dim: Dim
    var block_dim: Dim
    var cluster_dim: OptionalReg[Dim]
    var shared_mem_bytes: OptionalReg[Int]
    var attributes: List[LaunchAttribute]
    var constant_memory: List[ConstantMemoryMapping]
    var func_attribute: OptionalReg[FuncAttribute]


trait HalEnqueuer:
    """Takes over `enqueue_function` entirely.

    When `overrides` is set, every `enqueue_function` on a HAL DeviceContext
    or DeviceStream hands its inputs to the matching method here before doing
    anything itself, so nothing is compiled, checked or packed unless the
    plugin does it. `launch` is the caller's execution config, or a
    `HalLaunchOptions` holding its grid/block arguments.
    """

    comptime overrides: Bool
    """Whether this enqueuer replaces `enqueue_function`."""

    @staticmethod
    def enqueue_function[
        spec: DeviceSpec,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        HostArgs: TypeList[Trait=RegisterPassable, ...],
        LaunchT: AnyType,
        //,
        func: def(*args: *declared_arg_types) thin -> None,
        *Ts: AnyType,
    ](
        mut stream: Stream[spec],
        launch: LaunchT,
        *args: *Ts,
        host_args: Tuple[*HostArgs],
    ) raises:
        """Compiles and launches `func`, as `enqueue_function[func]` does."""
        ...

    @staticmethod
    def enqueue_closure[
        spec: DeviceSpec,
        FuncType: ImplicitlyCopyable & RegisterPassable & def() -> None,
        LaunchT: AnyType,
        //,
    ](mut stream: Stream[spec], launch: LaunchT, closure: FuncType) raises:
        """Compiles and launches `closure`."""
        ...

    @staticmethod
    def enqueue_compiled[
        spec: DeviceSpec,
        LaunchT: AnyType,
        //,
        Wrapper: DeviceFunctionWrapper,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        *Ts: AnyType,
    ](
        mut stream: Stream[spec],
        func: FunctionHandle,
        launch: LaunchT,
        *args: *Ts,
    ) raises:
        """Launches a function `compile_function` already compiled."""
        ...

    @staticmethod
    def enqueue_external[
        spec: DeviceSpec,
        LaunchT: AnyType,
        //,
        *Ts: AnyType,
    ](
        mut stream: Stream[spec],
        func: FunctionHandle,
        launch: LaunchT,
        *args: *Ts,
    ) raises:
        """Launches an externally loaded function."""
        ...


struct DefaultHalEnqueuer(HalEnqueuer):
    """Leaves `enqueue_function` to the HAL DeviceContext."""

    comptime overrides = False

    @staticmethod
    def enqueue_function[
        spec: DeviceSpec,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        HostArgs: TypeList[Trait=RegisterPassable, ...],
        LaunchT: AnyType,
        //,
        func: def(*args: *declared_arg_types) thin -> None,
        *Ts: AnyType,
    ](
        mut stream: Stream[spec],
        launch: LaunchT,
        *args: *Ts,
        host_args: Tuple[*HostArgs],
    ) raises:
        raise Error("the active HAL plugin has no enqueuer")

    @staticmethod
    def enqueue_closure[
        spec: DeviceSpec,
        FuncType: ImplicitlyCopyable & RegisterPassable & def() -> None,
        LaunchT: AnyType,
        //,
    ](mut stream: Stream[spec], launch: LaunchT, closure: FuncType) raises:
        raise Error("the active HAL plugin has no enqueuer")

    @staticmethod
    def enqueue_compiled[
        spec: DeviceSpec,
        LaunchT: AnyType,
        //,
        Wrapper: DeviceFunctionWrapper,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        *Ts: AnyType,
    ](
        mut stream: Stream[spec],
        func: FunctionHandle,
        launch: LaunchT,
        *args: *Ts,
    ) raises:
        raise Error("the active HAL plugin has no enqueuer")

    @staticmethod
    def enqueue_external[
        spec: DeviceSpec,
        LaunchT: AnyType,
        //,
        *Ts: AnyType,
    ](
        mut stream: Stream[spec],
        func: FunctionHandle,
        launch: LaunchT,
        *args: *Ts,
    ) raises:
        raise Error("the active HAL plugin has no enqueuer")


trait HalPluginHooks:
    """Compile-time hook interface for pluggable stdlib behavior.

    Most hooks are `comptime OptionalReg[Callable]` fields; call sites invoke
    `comptime if CurrentPlugin.xxx_fn: return comptime(CurrentPlugin.xxx_fn.value())(...)`,
    so implementors that leave a hook at `None` add zero cost.

    A few hooks (`abort_fn`, `debug_assert_emit_fn`) are required
    `@staticmethod` trait methods rather than `OptionalReg` fields, because
    their dispatch sites lie on `OptionalReg.value()`'s own instantiation
    path — an `OptionalReg` field would re-enter that template via its own
    `debug_assert` and deadlock comptime instantiation.
    """

    comptime ExecutionConfigType[device: DeviceRef]: ExecutionConfig

    comptime extensions: HALExtensions = DefaultHALExtensions

    comptime FunctionWrapper: DeviceFunctionWrapper = _IdentityWrapper
    """Selects the function `compile_function` compiles for a user kernel,
    such as a generated entry point the plugin's runtime launches."""

    comptime Launcher: HalLauncher = DefaultHalLauncher
    """Launches grid/block `enqueue_function` calls the plugin takes over."""

    comptime Enqueuer: HalEnqueuer = DefaultHalEnqueuer
    """Replaces `enqueue_function` outright, before anything is compiled."""


# ===-----------------------------------------------------------------------===#
# DefaultPlugin
# ===-----------------------------------------------------------------------===#


struct DefaultHalPlugin(HalPlugin):
    """Default `HalPluginHooks` implementation used when no plugin is active.

    Every hook is left at its `HalPluginHooks` default, so the built-in stdlib
    code paths are preserved.
    """

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["default"]()
    comptime HookFor[target: CompilationTarget] = DefaultHalPluginHooks[target]


struct DefaultHalPluginHooks[target: CompilationTarget](HalPluginHooks):
    comptime ExecutionConfigType[device: DeviceRef] = CPUExecutionConfiguration

    comptime extensions = DefaultHALExtensions
