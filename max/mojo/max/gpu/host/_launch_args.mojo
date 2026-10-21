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
`device_context.mojo` and `_device_context_metal.mojo`); they are not part of
the public API.
"""

from std.collections.optional import Optional
from std.sys import CompilationTarget, size_of

from std.math import align_up

from std.builtin.device_passable import DevicePassable


@inline(.always)
def _compact_zero_sized_capture_slots(
    dense_args_addrs: Pointer[OpaquePointer[MutAnyOrigin], MutUntrackedOrigin],
    capture_sizes: Pointer[UInt64, ImmUntrackedOrigin],
    num_leading_args: Int,
    num_captures: Int,
    dense_args_sizes: Optional[Pointer[UInt64, MutUntrackedOrigin]] = None,
) -> Int:
    """Compacts the capture slots of a packed launch-argument array in place,
    dropping zero-sized captures.

    NOTE(MOCO-4233): This is a hack resulting from legacy closures. A
    fully-static-layout capture (e.g. a `TileTensor`'s layout struct) is
    emitted as a zero-sized cross-device capture, but the device kernel
    elides it from its declared parameter list. The driver reads the packed
    argument array positionally against that parameter list, so a zero-sized
    slot preceding a real argument shifts every following argument by one:
    the kernel dereferences the wrong value (a hard illegal-address fault on
    CUDA), while Metal validates argument sizes and rejects any `== 0`.
    Dropping the zero-sized slots keeps the packed array aligned with the
    device kernel's parameter order.

    The capture slots start at `dense_args_addrs[num_leading_args]` (as
    written by the compiler-generated `populate` function) and are compacted
    in place, preserving relative order. Slots at and beyond the returned
    count are left stale and must not be read.

    Args:
        dense_args_addrs: The packed per-argument value pointers:
            `num_leading_args` argument slots followed by `num_captures`
            capture slots.
        capture_sizes: The store size in bytes of each capture, as recorded
            by the compiler.
        num_leading_args: The number of non-capture argument slots preceding
            the captures.
        num_captures: The number of capture slots.
        dense_args_sizes: Optional parallel per-argument sizes array; when
            provided, the surviving captures' sizes are written to it in
            lockstep with the compacted address slots.

    Returns:
        The effective argument count — `num_leading_args` plus the number of
        non-zero-sized captures. This is the slot count the device kernel
        actually reads, and must be the count passed to the launch call.
    """
    var effective_argc = num_leading_args
    for i in range(num_captures):
        if capture_sizes[unsafe_offset=i] != 0:
            dense_args_addrs[unsafe_offset=effective_argc] = dense_args_addrs[
                unsafe_offset=num_leading_args + i
            ]
            if case .Some(dense_args_sizes) = dense_args_sizes:
                dense_args_sizes[unsafe_offset=effective_argc] = capture_sizes[
                    unsafe_offset=i
                ]
            effective_argc += 1
    return effective_argc


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
