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
"""Kernel arguments for a typed `Queue.execute`.

Defines the device memory handles a launch takes, the packed arguments a
plugin's launcher reads, and the plugin hook that packs them.
"""

from std.builtin.device_passable import (
    DevicePassable,
    DevicePointerLike,
    DeviceTypeEncoder,
)
from std.builtin.rebind import downcast
from std.math import align_up
from std.reflection import reflect
from std.sys import size_of
from std.sys.info import _TargetType, _current_target

from max.gpu.host._launch_args import (
    DenseArgs,
    DeviceFunctionWrapper,
    check_encodable_argument,
)

from .context import Context
from .device import DeviceSpec
from .plugin import M_driver_memory_view
from .status import HALError


trait DeviceMemoryHandle:
    """A handle to device memory that a kernel's pointer argument can take.

    Kernels declare pointers, but a HAL allocation is a handle that only its
    plugin can turn into a kernel argument, so a launch takes the memory view
    the handle names and the plugin's packer encodes it. Device pointers and
    `TileTensor`s already carry their address and are passed as
    `DevicePassable` values.
    """

    def memory_view[
        spec: DeviceSpec
    ](self, context: Context[spec]) raises HALError -> M_driver_memory_view:
        """Returns the memory handle and byte range this handle refers to.

        Parameters:
            spec: The device the launch runs on.

        Args:
            context: The context of the queue the launch runs on.

        Returns:
            The view a kernel's pointer argument reads through.

        Raises:
            If the memory isn't `context`'s.
        """
        ...


def view_device_address[
    spec: DeviceSpec
](context: Context[spec], view: M_driver_memory_view) raises HALError -> UInt64:
    """Returns the device address a kernel reads `view` at.

    Parameters:
        spec: The device the launch runs on.

    Args:
        context: The context the viewed memory belongs to.
        view: The memory and byte range to resolve.

    Returns:
        The allocation's device address plus the view's offset.

    Raises:
        If the plugin can't report the allocation's address.
    """
    return (
        context._raw[].get_memory_property["address", UInt64](view.memory)
        + view.offset
    )


def _is_kernel_arg[T: AnyType]() -> Bool:
    """Returns whether a typed `Queue.execute` can pass a `T` argument."""
    return conforms_to(T, DeviceMemoryHandle) or conforms_to(T, DevicePassable)


def encode_target[spec: DeviceSpec]() -> _TargetType:
    """Returns the target whose data layout a launch on `spec`'s device
    encodes its arguments with.

    Parameters:
        spec: The device the launch runs on.

    Returns:
        The device's target, or the host's for a device without one.
    """
    comptime if spec.info:
        return DeviceSpec._mlir_target[spec]()
    else:
        return _current_target()


@fieldwise_init
struct _SpecDeviceTypeEncoder[spec: DeviceSpec](DeviceTypeEncoder):
    """Encodes `DevicePassable` values for `spec`'s device, whichever device
    the build defaults to."""

    comptime _raw_mlir_target = encode_target[Self.spec]()

    def encode_device_ptr[
        DevicePointerType: DevicePointerLike
    ](mut self, value: DevicePointerType, dst: MutOpaquePointer[_]):
        value.unsafe_ptr()._to_device_type(self, dst)


@fieldwise_init
struct KernelArgSlotKind(Equatable, TrivialRegisterPassable):
    """How a launch reads the bytes an argument slot points at."""

    var _value: UInt8

    comptime VALUE = Self(_value=0)
    """The slot points at the argument's encoded bytes, which the launch
    copies."""

    comptime DEVICE_ADDRESS = Self(_value=1)
    """The slot points at the 64-bit device address of memory the kernel
    reads through."""


trait HalKernelArgs(Deinitable, Movable):
    """The packed arguments of one launch, as the plugin's runtime reads them.

    Holds the bytes its slots point at, so it must outlive the enqueue that
    reads them. The core packs the arguments with the plugin's `ArgPacker`
    and hands them to the plugin's launcher, or to the queue's raw `execute`.
    """

    comptime device_spec: DeviceSpec
    """The device the arguments are encoded for."""

    comptime max_slots: Int
    """The capacity of the slot arrays `slots` and `slot_kinds` return."""

    def num_slots(self) -> Int:
        """Returns the number of slots the launch reads."""
        ...

    def slots[
        origin: ImmOrigin
    ](ref[origin] self, out result: DenseArgs[origin, Self.max_slots]):
        """Returns the address and size of each slot.

        Parameters:
            origin: The origin of `self`, which the slot addresses point into.

        Returns:
            The first `num_slots()` addresses and sizes, in the order the
            kernel reads them.
        """
        ...

    def slot_kinds(self) -> Array[KernelArgSlotKind, Self.max_slots]:
        """Returns how the launch reads each slot.

        Returns:
            The kind of each of the first `num_slots()` slots; by default
            every slot is a `VALUE`.
        """
        return Array[KernelArgSlotKind, Self.max_slots](
            fill=KernelArgSlotKind.VALUE
        )


trait HalArgPacker:
    """Packs a launch's arguments into the slots the plugin's runtime reads.

    The default suits devices whose runtime takes an array of argument
    addresses; a plugin with its own argument ABI overrides it.
    """

    comptime Packed[
        spec: DeviceSpec,
        Wrapper: DeviceFunctionWrapper,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        Ts: TypeList[Trait=AnyType, ...],
    ]: HalKernelArgs
    """The packed arguments of a launch on `spec`'s device of a kernel
    compiled behind `Wrapper`, declaring `declared_arg_types`, with arguments
    of types `Ts`."""

    @staticmethod
    def pack[
        spec: DeviceSpec,
        //,
        Wrapper: DeviceFunctionWrapper,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        *Ts: AnyType,
    ](context: Context[spec], *args: *Ts) raises HALError -> Self.Packed[
        spec, Wrapper, declared_arg_types, Ts
    ]:
        """Packs `args` for a launch on `context`'s device.

        Parameters:
            spec: The device the launch runs on.
            Wrapper: The wrapper the kernel was compiled behind.
            declared_arg_types: The kernel's declared argument types.
            Ts: The argument types.

        Args:
            context: The context of the queue the launch runs on.
            args: The arguments.

        Returns:
            The packed arguments.

        Raises:
            If a handle's device address can't be resolved.
        """
        ...


struct DenseKernelArgs[
    spec: DeviceSpec,
    declared_arg_types: TypeList[Trait=AnyType, ...],
    Ts: TypeList[Trait=AnyType, ...],
](HalKernelArgs):
    """Arguments packed back to back, each encoded as the kernel declares it.

    Every argument takes one slot holding its device encoding, the layout
    `DeviceContext.enqueue_function` launches with, and a device memory
    handle takes one slot holding its device address. Zero-sized arguments
    take no slot.

    Parameters:
        spec: The device the arguments are encoded for.
        declared_arg_types: The kernel's declared argument types.
        Ts: The argument types, each a `DeviceMemoryHandle` or
            `DevicePassable`.
    """

    comptime device_spec = Self.spec
    comptime _Encoder = _SpecDeviceTypeEncoder[Self.spec]
    comptime _encoder_target = Self._Encoder.target()

    @staticmethod
    def _encoded_size[T: AnyType]() -> Int:
        comptime if conforms_to(T, DeviceMemoryHandle):
            return size_of[UInt64]()
        elif conforms_to(T, DevicePassable):
            return size_of[
                downcast[T, DevicePassable].device_type,
                target=Self._encoder_target,
            ]()
        else:
            return 0

    @staticmethod
    def _offset[index: Int]() -> Int:
        var offset = 0
        comptime for i in range(index):
            offset += align_up(Self._encoded_size[Self.Ts[i]](), 8)
        return offset

    @staticmethod
    def _slot_count[index: Int]() -> Int:
        var count = 0
        comptime for i in range(index):
            comptime if Self._encoded_size[Self.Ts[i]]() != 0:
                count += 1
        return count

    comptime _num_slots = Self._slot_count[Self.Ts.length]()
    comptime max_slots = max(Self._num_slots, 1)
    # Words rather than bytes, so every slot is 8-byte aligned.
    comptime _num_words = max(Self._offset[Self.Ts.length]() // 8, 1)

    var _words: Array[UInt64, Self._num_words]

    def __init__[
        *ArgTs: AnyType
    ](out self, context: Context[Self.spec], *args: *ArgTs) raises HALError:
        """Encodes `args` as `declared_arg_types` declares them.

        Parameters:
            ArgTs: The argument types, which must be `Ts`.

        Args:
            context: The context of the queue the launch runs on.
            args: The arguments.

        Raises:
            If a handle's device address can't be resolved.
        """
        comptime assert ArgTs.length == Self.Ts.length
        self._words = Array[UInt64, Self._num_words](fill=0)
        var encoder = Self._Encoder()

        comptime for i in range(ArgTs.length):
            comptime T = ArgTs[i]
            comptime assert T == Self.Ts[i]
            comptime declared = Self.declared_arg_types[i]
            var dst = (
                self._words.unsafe_ptr()
                .unsafe_bitcast[UInt8]()
                .unsafe_offset(Self._offset[i]())
                .unsafe_bitcast[NoneType]()
            )
            comptime if conforms_to(T, DeviceMemoryHandle):
                comptime assert (
                    size_of[declared, target=Self._encoder_target]()
                    == size_of[UInt64]()
                ), String(
                    "argument #",
                    i,
                    (
                        " is a device memory handle, which fills a pointer, but"
                        " the kernel declares it as '"
                    ),
                    reflect[declared].name(),
                    "'",
                )
                ref handle = rebind[downcast[T, DeviceMemoryHandle]](args[i])
                dst.unsafe_bitcast[UInt64]()[] = view_device_address(
                    context, handle.memory_view(context)
                )
            else:
                comptime DevT = downcast[T, DevicePassable]
                check_encodable_argument[
                    Self._encoder_target,
                    index=i,
                    declared_arg_type=declared,
                    actual_arg_type=DevT,
                ]()
                rebind[DevT](args[i])._to_device_type(encoder, dst)

    def num_slots(self) -> Int:
        return Self._num_slots

    def slots[
        origin: ImmOrigin
    ](ref[origin] self, out result: DenseArgs[origin, Self.max_slots]):
        result = DenseArgs[origin, Self.max_slots]()
        var base = (
            self._words.unsafe_ptr()
            .unsafe_bitcast[UInt8]()
            .unsafe_origin_cast[origin]()
        )
        comptime for i in range(Self.Ts.length):
            comptime size = Self._encoded_size[Self.Ts[i]]()
            comptime if size != 0:
                comptime slot = Self._slot_count[i]()
                result.addrs.unsafe_get(slot) = base.unsafe_offset(
                    Self._offset[i]()
                ).unsafe_bitcast[NoneType]()
                result.sizes.unsafe_get(slot) = UInt64(size)


struct DenseArgPacker(HalArgPacker):
    """Packs arguments as `DenseKernelArgs`, for devices whose runtime takes
    an array of argument addresses."""

    comptime Packed[
        spec: DeviceSpec,
        Wrapper: DeviceFunctionWrapper,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        Ts: TypeList[Trait=AnyType, ...],
    ] = DenseKernelArgs[spec, declared_arg_types, Ts]

    @staticmethod
    def pack[
        spec: DeviceSpec,
        //,
        Wrapper: DeviceFunctionWrapper,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        *Ts: AnyType,
    ](context: Context[spec], *args: *Ts) raises HALError -> Self.Packed[
        spec, Wrapper, declared_arg_types, Ts
    ]:
        return DenseKernelArgs[spec, declared_arg_types, Ts](context, *args)
