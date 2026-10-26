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
        comptime declared_arg_type = DeclaredArgTypes[i]
        comptime actual_arg_type = ActualArgTypes[i]

        def declared_arg_type_name() -> String:
            comptime if conforms_to(declared_arg_type, DevicePassable):
                return declared_arg_type.get_type_name()
            else:
                return reflect[declared_arg_type].name()

        # Check that the given argument can occupy the kernel's declared
        # argument slot after device encoding.
        comptime is_convertible: Bool = actual_arg_type._is_implicitly_encodable_to[
            declared_arg_type
        ]()

        comptime assert is_convertible, String(
            "argument #",
            i,
            " of type '",
            actual_arg_type.get_type_name(),
            "' is not encodable as the declared function argument type '",
            declared_arg_type_name(),
            "'",
        )

        # The kernel reads the slot as `declared_arg_type`, but the bytes
        # written are `actual_arg_type.device_type`. Field-wise encodability
        # compares field types, not layouts, so check the layouts agree.
        comptime host_args_hint = (
            ". If the argument's host bytes are already valid on the device,"
            " pass it via `host_args=` to skip encoding"
        )
        comptime declared_size = size_of[declared_arg_type, target=target]()
        comptime device_size = size_of[
            actual_arg_type.device_type, target=target
        ]()
        comptime assert declared_size == device_size, String(
            "argument #",
            i,
            " of type '",
            actual_arg_type.get_type_name(),
            "' encodes to ",
            device_size,
            " bytes but the declared function argument type '",
            declared_arg_type_name(),
            "' is ",
            declared_size,
            " bytes",
            host_args_hint,
        )
        comptime declared_align = align_of[declared_arg_type, target=target]()
        comptime device_align = align_of[
            actual_arg_type.device_type, target=target
        ]()
        comptime assert declared_align == device_align, String(
            "argument #",
            i,
            " of type '",
            actual_arg_type.get_type_name(),
            "' encodes with alignment ",
            device_align,
            " but the declared function argument type '",
            declared_arg_type_name(),
            "' has alignment ",
            declared_align,
            host_args_hint,
        )

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
