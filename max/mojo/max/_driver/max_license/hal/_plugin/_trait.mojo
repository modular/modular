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

from max.gpu.host._launch_args import DeviceFunctionWrapper, _IdentityWrapper
from max.gpu.host._legacy.constant_memory_mapping import ConstantMemoryMapping
from max.gpu.host._legacy.dim import Dim
from max.gpu.host._legacy.func_attribute import FuncAttribute
from max.gpu.host._legacy.launch_attribute import LaunchAttribute

from ..device import DeviceSpec
from ..kernel_args import DenseArgPacker, HalArgPacker, HalKernelArgs
from ..plugins import FunctionHandle
from ..status import STATUS_INVALID_ARG, HALError
from ..queue import Queue
from ..stream import Stream

from .._machine import DeviceRef
from std.builtin.rebind import downcast
from max.gpu.host import ExecutionConfig, GridBlockExecutionConfig
from ..execution_config.cpu_execution_config import CPUExecutionConfiguration
from ..extensions.cpu import CPUHALExtensions
from ..extensions.traits import DefaultHALExtensions, HALExtensions


trait HalLauncher:
    """Launches a kernel with arguments packed by the plugin's `ArgPacker`.

    `DefaultHalLauncher` launches grid/block configs. A plugin with its own
    launch ABI supplies a launcher that launches the configs it supports, can
    forward grid/block configs to `DefaultHalLauncher.enqueue`, and rejects
    any other config. The launcher also decides which grid and block
    dimensions are valid.

    The caller keeps the packed arguments alive until `enqueue` returns.
    """

    @staticmethod
    def enqueue[
        spec: DeviceSpec,
        ExecutionConfigType: ExecutionConfig,
        Args: HalKernelArgs,
        o: ImmOrigin,
        //,
    ](
        queue: Queue[spec],
        func: FunctionHandle,
        args: Args,
        execution_config: ExecutionConfigType,
        attributes: Span[LaunchAttribute, o],
    ) raises HALError:
        """Launches `func` on `queue` with the packed `args` as
        `execution_config` describes, applying the caller's launch
        `attributes`, such as a cluster dimension or programmatic dependent
        launch. A launcher that can't apply them raises."""
        ...


struct DefaultHalLauncher(HalLauncher):
    """Launches grid/block configs with the packed argument slots.

    A launch whose grid or block has a component outside `[1, Int32.MAX]`
    raises. Any other kind of config fails to compile.
    """

    @staticmethod
    def enqueue[
        spec: DeviceSpec,
        ExecutionConfigType: ExecutionConfig,
        Args: HalKernelArgs,
        o: ImmOrigin,
        //,
    ](
        queue: Queue[spec],
        func: FunctionHandle,
        args: Args,
        execution_config: ExecutionConfigType,
        attributes: Span[LaunchAttribute, o],
    ) raises HALError:
        comptime assert conforms_to(
            ExecutionConfigType, GridBlockExecutionConfig
        ), "the active HAL plugin's launcher only takes grid/block configs"
        ref config = rebind[
            downcast[ExecutionConfigType, GridBlockExecutionConfig]
        ](execution_config)
        _check_dims_in_range(config.get_grid_dim(), "grid_dim")
        _check_dims_in_range(config.get_block_dim(), "block_dim")
        var slots = args.slots()
        var attribute_ptr = OptionalReg[OpaquePointer[MutUntrackedOrigin]](None)
        if len(attributes) != 0:
            attribute_ptr = (
                attributes.unsafe_ptr()
                .unsafe_bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .unsafe_origin_cast[MutUntrackedOrigin]()
            )
        queue.execute(
            func,
            config,
            args=slots.addrs.unsafe_ptr(),
            arg_sizes=slots.sizes.unsafe_ptr(),
            num_args=UInt32(args.num_slots()),
            attributes=attribute_ptr,
            num_attributes=UInt32(len(attributes)),
        )
        _ = slots^


def _check_dims_in_range(dim: Dim, name: StaticString) raises HALError:
    # Drivers take launch dims as 32-bit values, so a larger one would be
    # truncated.
    _check_dim_in_range(dim.x(), name, "x")
    _check_dim_in_range(dim.y(), name, "y")
    _check_dim_in_range(dim.z(), name, "z")


def _check_dim_in_range(
    value: Int, name: StaticString, axis: StaticString
) raises HALError:
    if value <= 0 or value > Int(Int32.MAX):
        raise HALError(
            STATUS_INVALID_ARG,
            message=String(
                name, ".", axis, " must be in [1, Int32.MAX], but it is ", value
            ),
        )


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

    Its hooks take the caller's stream rather than a queue, as
    `HalLauncher`'s do: only a DeviceContext or DeviceStream reaches them,
    always with a stream, and an enqueuer that issues more than one
    operation for a call can order all of them after the stream's earlier
    work through it.
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

    comptime Launcher: HalLauncher = DefaultHalLauncher
    """Launches every typed launch once its arguments are packed."""

    comptime ArgPacker: HalArgPacker = DenseArgPacker
    """Packs a launch's arguments into the slots the plugin's runtime reads,
    for a typed `Queue.execute` and for every launch the `Launcher` takes."""

    comptime Enqueuer: HalEnqueuer = DefaultHalEnqueuer
    """Replaces `enqueue_function` outright, before anything is compiled."""


# ===-----------------------------------------------------------------------===#
# DefaultPlugin
# ===-----------------------------------------------------------------------===#


struct DefaultHalPluginHooks(HalPluginHooks):
    comptime ExecutionConfigType[device: DeviceRef] = CPUExecutionConfiguration

    comptime extensions = DefaultHALExtensions


struct CPUHalPluginHooks(HalPluginHooks):
    """The host CPU's HAL hooks: the default ones plus its own extensions."""

    comptime ExecutionConfigType[device: DeviceRef] = CPUExecutionConfiguration

    comptime extensions: HALExtensions = CPUHALExtensions
