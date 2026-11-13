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
"""Shared helpers for packing GPU kernel launch arguments.

These are implementation details of `DeviceFunction`'s launch paths (see
`device_context.mojo` and `_device_context_metal.mojo`) and are shared by the
legacy and HAL device contexts; they are not part of the public API.
"""

from std.sys import CompilationTarget, align_of, size_of

from std.math import align_up

from std.builtin.device_passable import DevicePassable
from std.reflection import reflect


struct DenseArgs[addrs_origin: Origin, length: Int]:
    """Storage for the per-argument address and size slots of a kernel launch.

    Slot `i` of `addrs` points at the bytes of the `i`-th launch
    argument, and slot `i` of `sizes` is that argument's size in
    bytes. Callers fill both arrays and pass pointers to them to the enqueuer.

    Parameters:
        addrs_origin: The origin the argument address pointers borrow from.
        length: The number of argument slots.
    """

    var addrs: Array[OpaquePointer[Self.addrs_origin], Self.length]
    var sizes: Array[UInt64, Self.length]

    def __init__(out self):
        """Initializes the address slots uninitialized and the sizes to zero."""
        self.addrs = Array[OpaquePointer[Self.addrs_origin], Self.length](
            uninitialized=True
        )
        self.sizes = Array[UInt64, Self.length](fill=0)


trait DeviceFunctionWrapper:
    """Selects the function that `DeviceFunction` compiles for a user kernel.

    Most targets compile the kernel as written. A target that launches through
    a generated entry point supplies a wrapper that produces it.
    """

    comptime Entry[
        ret_type: AnyType,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        f: def(*args: *declared_arg_types) thin -> ret_type,
    ]: TrivialRegisterPassable
    """The type of the function compiled for `f`."""

    comptime wrap[
        ret_type: AnyType,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        f: def(*args: *declared_arg_types) thin -> ret_type,
    ]: Self.Entry[f]
    """The function compiled for `f`."""


struct _IdentityWrapper(DeviceFunctionWrapper):
    """Compiles the kernel as written, with no entry point."""

    comptime Entry[
        ret_type: AnyType,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        f: def(*args: *declared_arg_types) thin -> ret_type,
    ] = type_of(f)

    comptime wrap[
        ret_type: AnyType,
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        f: def(*args: *declared_arg_types) thin -> ret_type,
    ]: Self.Entry[f] = f


@fieldwise_init
struct SlotLayout(TrivialRegisterPassable):
    """The size and alignment, in bytes, of a kernel argument slot."""

    var size: Int
    var align: Int

    @staticmethod
    def of[T: AnyType, target: CompilationTarget]() -> Self:
        """Returns `T`'s layout per `size_of` and `align_of`."""
        return Self(size_of[T, target=target](), align_of[T, target=target]())


trait DeviceArgAcceptor:
    """A kernel argument type that can be filled by a host value of another
    type, because that value already encodes to the bytes this type expects.

    Lets a device-only argument type, which the host can't build, take a host
    value directly.
    """

    @staticmethod
    def _accepts_device_arg[ActualT: DevicePassable]() -> Bool:
        """Returns whether an `ActualT` may fill an argument of this type.

        Parameters:
            ActualT: The type passed for the argument.

        Returns:
            True if an `ActualT` may fill an argument declared as this type.
        """
        ...

    @staticmethod
    def _device_arg_slot[
        ActualT: DevicePassable, target: CompilationTarget
    ]() -> SlotLayout:
        """Returns the argument slot an accepted `ActualT` fills.

        The launcher still checks it against `ActualT.device_type`.

        Parameters:
            ActualT: The type passed for the argument.
            target: The compilation target of the kernel.

        Returns:
            The slot's size and alignment.
        """
        ...


def _declared_slot[
    Actual: AnyType, Declared: AnyType, target: CompilationTarget
]() -> Optional[SlotLayout]:
    """Returns the slot a `Declared` argument filled from a host `Actual`
    occupies, or `None` if `Actual` may not fill it.

    `Actual` must encode to `Declared`, be accepted by it, or match it field by
    field under this rule. Acceptors supply layouts the host cannot compute."""
    comptime if Actual == Declared:
        return SlotLayout.of[Declared, target]()
    elif conforms_to(Actual, DevicePassable):
        comptime accepted = _acceptor_slot[Declared, Actual, target]()
        comptime if Actual._is_implicitly_encodable_to[Declared]():
            return SlotLayout.of[Declared, target]()
        elif accepted:
            return accepted
        else:
            return _fields_slot[Actual.device_type, Declared, target]()
    elif conforms_to(Actual, RegisterPassable):
        return _fields_slot[Actual, Declared, target]()
    else:
        return None


def _acceptor_slot[
    Declared: AnyType, Actual: DevicePassable, target: CompilationTarget
]() -> Optional[SlotLayout]:
    comptime if conforms_to(Declared, DeviceArgAcceptor):
        comptime if Declared._accepts_device_arg[Actual]():
            return Declared._device_arg_slot[Actual, target]()
    return None


def _fields_slot[
    Actual: AnyType, Declared: AnyType, target: CompilationTarget
]() -> Optional[SlotLayout]:
    # Ignores `@align(N)` on `Declared`; the launcher's assert against the
    # encoded type catches the mismatch.
    comptime if _walkable_pair[Declared, Actual]():
        comptime a_fields = reflect[Actual].field_types()
        comptime d_fields = reflect[Declared].field_types()
        var size = 0
        var align = 1
        comptime for i in range(a_fields.length):
            comptime field = _declared_slot[a_fields[i], d_fields[i], target]()
            comptime __guard case .Some(slot) = field else:
                return None
            size = align_up(size, slot.align) + slot.size
            align = max(align, slot.align)
        return SlotLayout(align_up(size, align), align)
    else:
        return None


def _walkable_pair[Declared: AnyType, Actual: AnyType]() -> Bool:
    # Only the structs `encode_fields` walks; a leaf's MLIR-typed storage does
    # not reflect.
    comptime if not (
        conforms_to(Declared, DevicePassable)
        or conforms_to(Declared, RegisterPassable)
    ) or not (
        conforms_to(Actual, DevicePassable)
        or conforms_to(Actual, RegisterPassable)
    ):
        return False
    else:
        comptime d = reflect[Declared]
        comptime a = reflect[Actual]
        comptime if not d.is_struct() or not a.is_struct():
            return False
        # Reflected as structs but not walkable; see
        # `_contains_device_passable_field`.
        elif d.base_name() == "_RegTuple" or d.base_name() == "Bool":
            return False
        elif a.base_name() == "_RegTuple" or a.base_name() == "Bool":
            return False
        else:
            return d.field_count() == a.field_count() and d.field_count() != 0


# TODO(MSTDL-3307): Move this back to being a method on DeviceFunction
@inline(.always)
def _validate_encodable_arguments[
    target: CompilationTarget,
    *,
    DeclaredArgTypes: TypeList[Trait=AnyType, ...],
    ActualArgTypes: TypeList[Trait=DevicePassable, ...],
    host_args_count: Int = 0,
]() -> Tuple[Int, Array[Int, ActualArgTypes.length]]:
    """Validate that the passed (actual) argument types are encodable to the
    declared types in the user kernel function, and handle skipping zero-sized
    argument types.

    Any zero-sized types (ZSTs) present in `ActualArgTypes` are skipped: they
    get an offset of `-1` and are not counted in the returned argument count.
    """
    comptime assert (
        DeclaredArgTypes.length == ActualArgTypes.length + host_args_count
    ), "Wrong number of arguments to enqueue"

    # For each argument determine the size of the device dtype and
    # calculate the offset into a contiguous memory area which will
    # be used to remap the passed arguments into the device dtypes.
    var tmp_arg_offset = 0
    var translated_arg_offsets = Array[Int, ActualArgTypes.length](
        uninitialized=True
    )
    var num_translated_args = 0

    comptime for i in range(ActualArgTypes.length):
        comptime actual_arg_type = ActualArgTypes[i]
        check_encodable_argument[
            target,
            index=i,
            declared_arg_type=DeclaredArgTypes[i],
            actual_arg_type=actual_arg_type,
        ]()

        var aligned_type_size = align_up(
            size_of[actual_arg_type.device_type, target=target](), 8
        )
        if aligned_type_size != 0:
            num_translated_args += 1
            translated_arg_offsets[i] = tmp_arg_offset
            tmp_arg_offset += aligned_type_size
        else:
            # Zero-sized types (ZSTs) should not be stored directly.
            translated_arg_offsets[i] = -1

    return (num_translated_args, translated_arg_offsets^)


@inline(.always)
def check_encodable_argument[
    target: CompilationTarget,
    *,
    index: Int,
    declared_arg_type: AnyType,
    actual_arg_type: DevicePassable,
]():
    """Checks at compile time that an argument of type `actual_arg_type` can
    occupy the kernel argument at `index`, declared as `declared_arg_type`,
    once it's encoded for `target`.

    Parameters:
        target: The device the argument is encoded for.
        index: The argument's position, for the error message.
        declared_arg_type: The type the kernel declares the argument as.
        actual_arg_type: The type of the value passed.
    """

    def declared_arg_type_name() -> String:
        comptime if conforms_to(declared_arg_type, DevicePassable):
            return declared_arg_type.get_type_name()
        else:
            return reflect[declared_arg_type].name()

    # Check that the given argument can occupy the kernel's declared
    # argument slot after device encoding.
    comptime declared_slot = _declared_slot[
        actual_arg_type, declared_arg_type, target
    ]()

    comptime assert declared_slot, String(
        "argument #",
        index,
        " of type '",
        actual_arg_type.get_type_name(),
        "' is not encodable as the declared function argument type '",
        declared_arg_type_name(),
        "'",
    )

    # The kernel reads the slot as `declared_arg_type`, but the bytes
    # written are `actual_arg_type.device_type`. Field-wise encodability
    # compares field types, not layouts, so check the layouts agree.
    comptime if case .Some(declared) = declared_slot:
        comptime host_args_hint = (
            ". If the argument's host bytes are already valid on the"
            " device, pass it via `host_args=` to skip encoding"
        )
        comptime device = SlotLayout.of[actual_arg_type.device_type, target]()
        comptime assert declared.size == device.size, String(
            "argument #",
            index,
            " of type '",
            actual_arg_type.get_type_name(),
            "' encodes to ",
            device.size,
            " bytes but the declared function argument type '",
            declared_arg_type_name(),
            "' is ",
            declared.size,
            " bytes",
            host_args_hint,
        )
        comptime assert declared.align == device.align, String(
            "argument #",
            index,
            " of type '",
            actual_arg_type.get_type_name(),
            "' encodes with alignment ",
            device.align,
            " but the declared function argument type '",
            declared_arg_type_name(),
            "' has alignment ",
            declared.align,
            host_args_hint,
        )
