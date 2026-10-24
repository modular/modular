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

from std.builtin.device_passable import (
    DevicePassable,
    DevicePointerLike,
    DeviceTypeEncoder,
)
from std.collections.optional import Optional
from std.memory import unsafe_stack_allocation
from std.reflection import SourceLocation
from std.sys import size_of
from ..info import _device_type_encoder_target

from .device_context import (
    _checked_call,
    _DeviceBufferCpp,
    _DeviceFunctionPtr,
    _FunctionEnqueuer,
    DeviceContext,
)

from . import Dim, LaunchAttribute


@doc_hidden
@fieldwise_init
struct MetalEnqueueFunctionArgs:
    """Passes through Metal specific kernel launch data through to the
    driver."""

    var args: Pointer[OpaquePointer[MutUntrackedOrigin], MutUntrackedOrigin]
    var arg_sizes: Pointer[UInt64, ImmUntrackedOrigin]
    var arg_is_device_ptr: Pointer[Bool, MutUntrackedOrigin]
    var buffers: Optional[
        Pointer[
            Pointer[_DeviceBufferCpp, MutUntrackedOrigin], MutUntrackedOrigin
        ]
    ]
    var num_buffers: Int32


struct MetalDeviceTypeEncoder(DeviceTypeEncoder):
    """Provides a Metal specific implementation of the `DeviceTypeEncoder`
    trait."""

    var _buffers: List[Pointer[_DeviceBufferCpp, MutUntrackedOrigin]]

    def __init__(out self):
        """Initializes the encoder with an empty buffer list."""
        self._buffers = []

    comptime _raw_mlir_target = _device_type_encoder_target()

    def encode_device_ptr[
        DevicePointerType: DevicePointerLike
    ](mut self, value: DevicePointerType, dst: MutOpaquePointer[_],):
        """Encodes a device pointer into `dst`.

        Parameters:
            DevicePointerType: The type of the device pointer.

        Args:
            value: The device pointer to encode into `dst`.
            dst: The opaque destination pointer to encode into.
        """
        value.unsafe_ptr()._to_device_type(self, dst)

        var handle = value._buffer_handle()
        debug_assert(
            Bool(handle), "Metal device pointers must have a buffer handle"
        )
        self._buffers.append(handle.value().unsafe_bitcast[_DeviceBufferCpp]())


@inline(.always)
def call_with_pack_metal[
    func: Some[TrivialRegisterPassable],
    ContextT: _FunctionEnqueuer,
    num_args: Int,
](
    ctx: ContextT,
    func_handle: _DeviceFunctionPtr[mut=True],
    device_context: DeviceContext,
    dense_args_addrs: Pointer[OpaquePointer[MutAnyOrigin], MutUntrackedOrigin],
    dense_args_sizes: Pointer[UInt64, MutUntrackedOrigin],
    grid_dim: Dim,
    block_dim: Dim,
    shared_mem_bytes: Int,
    attributes_ptr: Pointer[LaunchAttribute, MutAnyOrigin],
    num_attributes: Int,
    location: SourceLocation,
) raises:
    """Wraps the arg pack in `MetalEnqueueFunctionArgs` and enqueues via `ctx`.

    Used by `DeviceFunction._call_with_pack` (the unchecked path) so callers
    that don't have `DevicePassable` arguments can still launch on Metal. The
    sizes pointer is sourced directly from `dense_args_sizes` without going
    through `MetalDeviceTypeEncoder`.

    Parameters:
        func: The function symbol used by `_checked_call` for error messages.
        ContextT: The `_FunctionEnqueuer` implementation receiving the launch.
        num_args: The compile-time count of caller-supplied arguments in
            `dense_args_addrs`.

    Args:
        ctx: The enqueuer that dispatches the kernel.
        func_handle: Handle to the compiled `DeviceFunction` to launch.
        device_context: The device context backing the function, used for
            error reporting in `_checked_call`.
        dense_args_addrs: Pre-populated per-argument value pointers, owned by
            the caller.
        dense_args_sizes: Pre-populated per-argument sizes in bytes, owned by
            the caller.
        grid_dim: Grid dimensions for the kernel launch.
        block_dim: Block dimensions for the kernel launch.
        shared_mem_bytes: Bytes of dynamic shared memory per block.
        attributes_ptr: Pointer to the launch attributes array.
        num_attributes: Number of entries in `attributes_ptr`.
        location: Source location threaded through `_checked_call` for error
            messages.
    """
    # Unchecked path: no `DevicePassable` encoding, so no arg is treated as
    # a device pointer for buffer binding purposes.
    #
    # `is_dev_inline` is an `Array` (not `unsafe_stack_allocation`) so the
    # compiler tracks its lifetime as a named local. Without this, the
    # stack slot can be reused for `metal_args` below — `metal_args`'s
    # `arg_is_device_ptr` field still points at the old slot, which now
    # holds part of the `metal_args` struct rather than the bool array.
    var is_dev_inline = Array[Bool, max(num_args, 1)](fill=False)
    var dense_args_is_device_ptr = (
        is_dev_inline.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )

    var metal_args = MetalEnqueueFunctionArgs(
        dense_args_addrs.unsafe_bitcast[OpaquePointer[MutUntrackedOrigin]](),
        dense_args_sizes,
        dense_args_is_device_ptr,
        None,
        Int32(0),
    )

    var metal_args_addrs = unsafe_stack_allocation[
        1, OpaquePointer[origin_of(metal_args)]
    ]()
    metal_args_addrs[] = Pointer(to=metal_args).unsafe_bitcast[NoneType]()

    _checked_call[func](
        ctx.enqueue(
            func_handle,
            grid_dim,
            block_dim,
            shared_mem_bytes,
            attributes_ptr,
            num_attributes,
            metal_args_addrs,
            UInt32(num_args),
            Optional[Pointer[UInt64, MutUntrackedOrigin]](),
        ),
        device_context=device_context,
        location=location,
    )


@inline(.always)
def call_with_pack_checked_metal[
    HostArgs: TypeList[Trait=RegisterPassable, ...],
    //,
    func: Some[TrivialRegisterPassable],
    *Ts: DevicePassable,
    ContextT: _FunctionEnqueuer,
    num_passed_args: Int,
](
    ctx: ContextT,
    *args: *Ts,
    host_args: Tuple[*HostArgs],
    func_handle: _DeviceFunctionPtr[mut=True],
    device_context: DeviceContext,
    translated_arg_offsets: Array[Int, num_passed_args],
    extra_align: Int,
    translated_args_ptr: Pointer[Byte, MutAnyOrigin],
    dense_args_addrs: Pointer[OpaquePointer[MutAnyOrigin], MutUntrackedOrigin],
    grid_dim: Dim,
    block_dim: Dim,
    shared_mem_bytes: Int,
    attributes_ptr: Pointer[LaunchAttribute, MutAnyOrigin],
    num_attributes: Int,
    location: SourceLocation,
) raises:
    """Encodes Metal kernel arguments and enqueues the function via `ctx`.

    Parameters:
        func: The function symbol used by `_checked_call` for error messages.
        Ts: The host argument types being passed to the kernel.
        ContextT: The `_FunctionEnqueuer` implementation receiving the launch.
        num_passed_args: The compile-time count of arguments in `args`.

    Args:
        ctx: The enqueuer that dispatches the kernel.
        args: The host-side kernel arguments to encode.
        host_args: Pack of host-layout arguments to encode.
        func_handle: Handle to the compiled `DeviceFunction` to launch.
        device_context: The device context backing the function, used for
            error reporting in `_checked_call`.
        translated_arg_offsets: Per-argument offsets into `translated_args_ptr`,
            or `-1` for zero-sized arguments that should be skipped.
        extra_align: Alignment padding to add to each `translated_arg_offset`.
        translated_args_ptr: Storage for the encoded device-side argument
            bytes, owned by the caller.
        dense_args_addrs: Storage for the per-argument value pointers, owned
            by the caller.
        grid_dim: Grid dimensions for the kernel launch.
        block_dim: Block dimensions for the kernel launch.
        shared_mem_bytes: Bytes of dynamic shared memory per block.
        attributes_ptr: Pointer to the launch attributes array.
        num_attributes: Number of entries in `attributes_ptr`.
        location: Source location threaded through `_checked_call` for error
            messages.
    """
    comptime extra_host_count = host_args.Ts.length
    comptime num_slots = max(num_passed_args + extra_host_count, 1)

    # Using `Array` (rather than `unsafe_stack_allocation`) gives the compiler
    # a named local whose origin flows into the `MetalEnqueueFunctionArgs`
    # pointer fields below — that origin dependency keeps the storage
    # alive across the `ctx.enqueue` call, preventing stack-slot reuse
    # for `metal_args` from clobbering the sizes/is-device-ptr arrays.
    var sizes_inline = Array[UInt64, num_slots](fill=0)
    var is_dev_inline = Array[Bool, num_slots](fill=False)

    var dense_args_sizes = sizes_inline.unsafe_ptr().unsafe_origin_cast[
        MutUntrackedOrigin
    ]()
    var dense_args_is_device_ptr = (
        is_dev_inline.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )

    # Since we skip over zero sized declared dtypes when passing arguments
    # we need to know the current count of arguments pushed.
    var translated_arg_idx = 0

    # The device type encoder is passed into
    # `DevicePassable._to_device_type()` to enable target specific encoding
    # of device types.
    var device_type_encoder = MetalDeviceTypeEncoder()

    comptime for i in range(num_passed_args):
        # If the arg offset is negative then the corresponding declared
        # dtype is zero sized and we do not push the argument to the kernel.
        var translated_arg_offset = translated_arg_offsets[i]
        if translated_arg_offset < 0:
            continue

        comptime actual_arg_type = Ts[i]
        var first_word_addr = Pointer(
            to=translated_args_ptr[
                unsafe_offset=translated_arg_offset + extra_align
            ]
        ).unsafe_bitcast[NoneType]()
        # Snapshot the encoder's buffer count so we can detect
        # whether this arg's encoding pushed exactly one device
        # buffer. Combined with the pointer-sized arg check
        # below, this distinguishes a bare `DevicePointer` /
        # `DeviceBuffer` arg from a nested struct that happens
        # to contain one or more device pointers.
        var buffers_before = len(device_type_encoder._buffers)
        args[i]._to_device_type(device_type_encoder, first_word_addr)

        var arg_size = size_of[
            actual_arg_type.device_type,
            target=device_type_encoder.target(),
        ]()
        dense_args_addrs[unsafe_offset=translated_arg_idx] = first_word_addr
        dense_args_sizes[unsafe_offset=translated_arg_idx] = UInt64(arg_size)
        dense_args_is_device_ptr[unsafe_offset=translated_arg_idx] = (
            len(device_type_encoder._buffers) - buffers_before == 1
            and arg_size == size_of[OpaquePointer[MutAnyOrigin]]()
        )
        translated_arg_idx += 1

    comptime for i in range(extra_host_count):
        comptime arg_size = size_of[host_args.Ts[i], device_context.target]()
        # Skip zero-sized types (ZSTs).
        if arg_size == 0:
            continue

        var arg_addr = (
            Pointer(to=host_args[i])
            .unsafe_bitcast[NoneType]()
            .unsafe_mut_cast[True]()
            .as_unsafe_any_origin()
        )
        dense_args_addrs[unsafe_offset=translated_arg_idx] = arg_addr
        dense_args_sizes[unsafe_offset=translated_arg_idx] = UInt64(arg_size)
        translated_arg_idx += 1

    var metal_args = MetalEnqueueFunctionArgs(
        dense_args_addrs.unsafe_bitcast[OpaquePointer[MutUntrackedOrigin]](),
        dense_args_sizes,
        dense_args_is_device_ptr,
        Optional[
            Pointer[
                Pointer[_DeviceBufferCpp, MutUntrackedOrigin],
                MutUntrackedOrigin,
            ]
        ](
            device_type_encoder._buffers.unsafe_ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ]()
        ),
        Int32(len(device_type_encoder._buffers)),
    )

    var metal_args_addrs = unsafe_stack_allocation[
        1, OpaquePointer[MutAnyOrigin]
    ]()
    metal_args_addrs[] = (
        Pointer(to=metal_args).unsafe_bitcast[NoneType]().as_unsafe_any_origin()
    )

    _checked_call[func](
        ctx.enqueue(
            func_handle,
            grid_dim,
            block_dim,
            shared_mem_bytes,
            attributes_ptr,
            num_attributes,
            metal_args_addrs,
            UInt32(translated_arg_idx),
            Optional[Pointer[UInt64, MutUntrackedOrigin]](),
        ),
        device_context=device_context,
        location=location,
    )
