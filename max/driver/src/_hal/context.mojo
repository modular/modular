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
"""HAL Context — per-device context for memory and queue operations."""

from . import Device, Queue
from .plugin import (
    RawDriver,
    OutParam,
    ContextHandle,
    FunctionHandle,
    RuntimeBundleHandle,
    M_driver_static_bundle,
    M_driver_slice,
    M_driver_bundle_compilation_options,
    M_driver_dlpack_device,
)

from .buffer import Buffer, BufferView
from .device import DeviceSpec
from .stream import Stream

from .status import STATUS_SUCCESS, HALError

from std.memory import (
    ImmutPointer,
    ArcPointer,
    UnsafePointer,
    UnsafeMaybeUninit,
)
from std.memory.arc_pointer import WeakPointer

from std.compile import CompiledFunctionInfo

from std.compile.compile import _Info, _get_emission_kind_id

from std.reflection import get_linkage_name

from std.collections.string.string_slice import _get_kgen_string

from std.sys.info import CompilationTarget, is_triple, _TargetType


#
def _bundle_file_type[target: _TargetType]() -> StaticString:
    """Returns the file_type string each HAL plugin expects for bundle_load."""
    if is_triple["amdgcn-amd-amdhsa", target]():
        return "hsaco"
    if is_triple["nvptx64-nvidia-cuda", target]():
        return "cubin"
    return "object"


@fieldwise_init
struct Context[device_spec: DeviceSpec](ImplicitlyDeletable, Movable):
    """A context loaded on a specific device.

    Represents a runtime handle to an initialized
    and usable device.

    This type is potentially expensive to construct,
    so information gathering for device selection
    should ideally be done on Device before creating
    a Context.

    Parameters:
        device_spec: The compilation target this context is set up for.
    """

    var _handle: ContextHandle
    var _device: ArcPointer[Device[Self.device_spec]]
    var _raw: ArcPointer[RawDriver]
    var _self_ref: WeakPointer[Self]

    @staticmethod
    def _create(
        out _self: ArcPointer[Self], device: Device[Self.device_spec]
    ) raises HALError:
        _self = ArcPointer(Self(device))
        _self[]._self_ref = WeakPointer(downgrade=_self)

    @doc_hidden
    def __init__(
        out self: Self,
        ref device: Device[Self.device_spec],
    ) raises HALError:
        self._device = device._self_ref.try_upgrade().value()
        self._raw = device._raw
        self._self_ref = WeakPointer[Self]()

        ref raw = self._raw[]

        var context_handle_uninit = UnsafeMaybeUninit[ContextHandle]()
        var status = raw._raw.context_create.f(
            device._handle, OutParam[ContextHandle](to=context_handle_uninit)
        )

        if status != STATUS_SUCCESS:
            var err = raw.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to create context from device: {err.message}"
                ),
            )

        self._handle = context_handle_uninit.unsafe_assume_init_ref()

    def __del__(deinit self):
        try:
            self._raw[].destroy_context(self._handle)
        except e:
            print("warning: destroy_context failed:", e)

    def handle[
        origin: Origin, //
    ](ref[origin] self) -> UnsafePointer[ContextHandle.type, origin]:
        """Returns the raw context handle, tying its origin back to `self`.

        Accessing `self._handle` directly hands out a `ContextHandle` whose
        origin is `MutExternalOrigin`, so it keeps nothing alive.

        Parameters:
            origin: The origin of the borrow of `self`, inferred at the call
                site.

        Returns:
            The context handle carrying `self`'s origin.
        """
        return self._handle.unsafe_mut_cast[origin.mut]().unsafe_origin_cast[
            origin
        ]()

    # ===-------------------------------------------------------------------===#
    # Queries
    # ===-------------------------------------------------------------------===#

    def get_driver_name(self) raises HALError -> String:
        """Returns the API name reported by the plugin backing this context.

        This is the plugin's own identity (e.g. "CUDA", "Metal", "HIP"),
        not the loader label from the plugin spec, so it is stable however
        the plugin was loaded.
        """
        return self._raw[].get_api_name()

    def get_device_id(self) -> Int64:
        """Returns the id of the device this context is bound to."""
        return self._device[].id

    def get_dlpack_device(
        self, pinned: Bool
    ) raises HALError -> M_driver_dlpack_device:
        """Returns the DLPack `(device_type, device_id)` for this context."""
        return self._device[].get_dlpack_device(pinned)

    # ===-------------------------------------------------------------------===#
    # Synchronous copies
    # ===-------------------------------------------------------------------===#

    def copy_to_device_sync(
        self,
        dst: BufferView,
        src: UnsafePointer[mut=False, UInt8, _],
    ) raises HALError:
        """Copies `dst.byte_size` bytes from host memory into `dst`, blocking
        until complete."""
        self._raw[].copy_to_device_sync(self._handle, dst._view, src)

    def copy_from_device_sync(
        self,
        dst: UnsafePointer[mut=True, UInt8, _],
        src: BufferView,
    ) raises HALError:
        """Copies `src.byte_size` bytes from `src` into host memory, blocking
        until complete."""
        self._raw[].copy_from_device_sync(self._handle, dst, src._view)

    def copy_intra_device_sync(
        self,
        dst: BufferView,
        src: BufferView,
    ) raises HALError:
        """Copies `dst.byte_size` bytes from `src` into `dst`, blocking until
        complete."""
        debug_assert(
            src.byte_size() >= dst.byte_size(),
            "copy_intra_device_sync source view smaller than destination",
        )
        self._raw[].copy_intra_device_sync(self._handle, dst._view, src._view)

    def _compile_inner[
        fn_type: TrivialRegisterPassable,
        func: fn_type,
    ](self) raises -> CompiledFunctionInfo[
        fn_type, func, Self.device_spec._mlir_target()
    ]:
        comptime target = Self.device_spec._mlir_target()
        comptime emission_kind_id = _get_emission_kind_id[
            "object"
        ]().__mlir_index__()

        var offload = __mlir_op.`kgen.compile_offload`[
            target_type=Self.device_spec._mlir_target(),
            emission_kind=emission_kind_id,
            emission_option=_get_kgen_string[
                CompilationTarget[
                    Self.device_spec._mlir_target()
                ].default_compile_options()
            ](),
            emission_link_option=_get_kgen_string[""](),
            func=func,
            _type=_Info,
        ]()

        return CompiledFunctionInfo[fn_type, func, target](
            asm=StaticString(offload.asm),
            function_name=get_linkage_name[func, target=target](),
            module_name=StaticString(offload.module_name),
            num_captures=Int(mlir_value=offload.num_captures),
            capture_sizes=offload.capture_sizes,
            emission_kind="object",
        )

    def compile[
        fn_type: TrivialRegisterPassable,
        func: fn_type,
    ](self) raises -> Tuple[
        RuntimeBundle,
        CompiledFunctionInfo[fn_type, func, Self.device_spec._mlir_target()],
    ]:
        var compiled_info = self._compile_inner[fn_type, func]()
        var bundle = self.load_bundle(compiled_info.asm)
        return (bundle^, compiled_info)

    def load_bundle[
        asm_origin: ImmutOrigin
    ](
        self, asm: StringSlice[origin=asm_origin]
    ) raises HALError -> RuntimeBundle:
        """Loads a runtime bundle from pre-compiled binary bytes."""
        # Each plugin expects a specific file_type string. PTX text is
        # accepted by `cuModuleLoadDataEx` even when file_type="cubin".
        comptime target = Self.device_spec._mlir_target()
        comptime file_type = _bundle_file_type[target]()

        var static_bundle = M_driver_static_bundle(
            mapped_data=M_driver_slice(
                data=Pointer(to=asm.unsafe_ptr()[]),
                size=UInt64(asm.byte_length()),
            ),
            file_type=Pointer(to=file_type.unsafe_ptr()[]),
            file_type_len=UInt64(file_type.byte_length()),
        )

        var opts = M_driver_bundle_compilation_options(
            debug_level=rebind[ImmutPointer[Int8, ImmutUntrackedOrigin]](
                "".unsafe_ptr()
            ),
            debug_level_len=UInt64(0),
            optimization_level=Int32(-1),
        )

        ref raw = self._raw[]
        var runtime_bundle = UnsafeMaybeUninit[RuntimeBundleHandle]()
        var status = raw._raw.bundle_load.f(
            self._handle,
            UnsafePointer(to=static_bundle),
            UnsafePointer(to=opts),
            OutParam[RuntimeBundleHandle](to=runtime_bundle),
        )
        if status != STATUS_SUCCESS:
            var err = raw.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to load bundle: {err.message}"),
            )

        return RuntimeBundle(
            _handle=runtime_bundle.unsafe_assume_init_ref(),
            _context_handle=self._handle,
            _raw=self._raw,
        )

    # ===-------------------------------------------------------------------===#
    # Queue operations
    # ===-------------------------------------------------------------------===#

    def create_queue(
        self,
    ) raises HALError -> ArcPointer[Queue[Self.device_spec]]:
        return Queue[Self.device_spec]._create(self)

    # ===-------------------------------------------------------------------===#
    # Stream operations
    # ===-------------------------------------------------------------------===#

    def create_stream(
        self,
    ) raises HALError -> ArcPointer[Stream[Self.device_spec]]:
        return Stream[Self.device_spec]._create(self)

    # ===-------------------------------------------------------------------===#
    # Memory operations
    # ===-------------------------------------------------------------------===#

    def alloc_sync(
        self, byte_size: UInt64
    ) raises HALError -> Buffer[Self.device_spec]:
        return Buffer[Self.device_spec](
            _handle=self._raw[].alloc_sync(self._handle, byte_size),
            byte_size=byte_size,
            is_host_pinned=False,
            _context=self._self_ref.try_upgrade().value(),
        )

    def free_sync(self, var mem: Buffer[Self.device_spec]) raises HALError:
        self._raw[].free_sync(self._handle, mem._handle)

    def alloc_host_pinned(
        self, byte_size: UInt64
    ) raises HALError -> Buffer[Self.device_spec]:
        return Buffer[Self.device_spec](
            _handle=self._raw[].alloc_pinned(self._handle, byte_size),
            byte_size=byte_size,
            is_host_pinned=True,
            _context=self._self_ref.try_upgrade().value(),
        )

    def free_host_pinned(
        self, var mem: Buffer[Self.device_spec]
    ) raises HALError:
        self._raw[].free_pinned(self._handle, mem._handle)

    def wrap_memory(
        self, address: UInt64, byte_size: UInt64, owning: Bool = False
    ) raises HALError -> Buffer[Self.device_spec]:
        """Wraps an existing device memory region in a Buffer.

        With `owning=False` (the default) the region is externally owned:
        the plugin never frees it, freeing the buffer releases only the
        plugin's bookkeeping, and the caller must keep the underlying
        allocation alive for the buffer's lifetime. With `owning=True`
        the buffer frees the region through the plugin's normal path,
        which is only valid for an address that came from this plugin's
        own allocator (e.g. one released with `unwrap_memory`).
        """
        return Buffer[Self.device_spec](
            _handle=self._raw[].wrap_memory(
                self._handle, address, byte_size, owning
            ),
            byte_size=byte_size,
            is_host_pinned=False,
            _context=self._self_ref.try_upgrade().value(),
        )

    def unwrap_memory(
        self, mem: Buffer[Self.device_spec]
    ) raises HALError -> UInt64:
        """Releases ownership of the region under `mem`, returning its address.

        After this call the buffer is non-owning — freeing it releases
        only the plugin's bookkeeping — and the caller is responsible for
        the region at the returned address.
        """
        return self._raw[].unwrap_memory(self._handle, mem._handle)

    def memory_get_address(
        self, mem: Buffer[Self.device_spec]
    ) raises HALError -> UInt64:
        """Get the GPU address of a device memory allocation."""
        return self._raw[].get_memory_property["address", UInt64](mem._handle)

    def memory_get_host_address[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, mem: Buffer[Self.device_spec]) raises HALError -> UnsafePointer[
        UInt8, origin
    ]:
        """Get a host-dereferenceable pointer to a host-accessible allocation.

        Host-accessible memory is any allocation on a host (CPU) device and
        pinned GPU allocations (`alloc_host_pinned`); on unified-memory GPUs
        every allocation qualifies. The plugin is the authority on residency:
        """
        return UnsafePointer[UInt8, origin](
            unsafe_from_address=Int(
                self._raw[].get_memory_property["host_address", UInt64](
                    mem._handle
                )
            )
        )

    # ===-------------------------------------------------------------------===#
    # Function execution
    # ===-------------------------------------------------------------------===#

    def load_function(
        self, bundle: RuntimeBundle, var name: String
    ) raises HALError -> FunctionHandle:
        return self._raw[].load_function(self._handle, bundle._handle, name)

    def unload_function(self, func: FunctionHandle) raises HALError:
        self._raw[].unload_function(self._handle, func)


@fieldwise_init
struct RuntimeBundle(Movable):
    var _handle: RuntimeBundleHandle
    var _context_handle: ContextHandle
    var _raw: ArcPointer[RawDriver]

    def __del__(deinit self):
        try:
            self._raw[].unload_bundle(self._context_handle, self._handle)
        except e:
            print("warning: unload_bundle failed:", e)
