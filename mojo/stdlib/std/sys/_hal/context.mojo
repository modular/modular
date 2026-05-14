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

from .plugin import (
    RawDriver,
    OutParam,
    ContextHandle,
    FunctionHandle,
    MemoryHandle,
    RuntimeBundleHandle,
    StaticBundleHandle,
    CompilationOptionsHandle,
    M_driver_static_bundle,
    M_driver_slice,
    M_driver_bundle_compilation_options,
)

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

from std.sys.info import is_triple, _TargetType


#
def _bundle_file_type[target: _TargetType]() -> StaticString:
    """Returns the file_type string each HAL plugin expects for bundle_load."""
    if is_triple["amdgcn-amd-amdhsa", target]():
        return "hsaco"
    if is_triple["nvptx64-nvidia-cuda", target]():
        return "cubin"
    return "object"


@fieldwise_init
struct Context[device_spec: DeviceSpec](ImplicitlyDestructible, Movable):
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

    def _compile_inner[
        fn_type: TrivialRegisterPassable,
        func: fn_type,
    ](self) raises -> CompiledFunctionInfo[
        fn_type, func, Self.device_spec.target.value
    ]:
        comptime target = Self.device_spec.target.value
        comptime emission_kind_id = _get_emission_kind_id[
            "object"
        ]()._mlir_value

        var offload = __mlir_op.`kgen.compile_offload`[
            target_type=Self.device_spec.target.value,
            emission_kind=emission_kind_id,
            emission_option=_get_kgen_string[
                Self.device_spec.target.default_compile_options()
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
    ](self) raises -> Tuple[RuntimeBundle, String]:
        var compiled_info = self._compile_inner[fn_type, func]()

        # Build the M_driver_static_bundle from the compiled object code.
        # Each plugin expects a specific file_type string.
        comptime target = Self.device_spec.target.value
        comptime file_type = _bundle_file_type[target]()

        var asm_data = compiled_info.asm
        var static_bundle = M_driver_static_bundle(
            mapped_data=M_driver_slice(
                data=rebind[ImmutPointer[UInt8, ImmutAnyOrigin]](
                    asm_data.unsafe_ptr()
                ),
                size=UInt64(asm_data.byte_length()),
            ),
            file_type=rebind[ImmutPointer[Int8, ImmutAnyOrigin]](
                file_type.unsafe_ptr()
            ),
            file_type_len=UInt64(file_type.byte_length()),
        )

        var opts = M_driver_bundle_compilation_options(
            debug_level=rebind[ImmutPointer[Int8, ImmutAnyOrigin]](
                "".unsafe_ptr()
            ),
            debug_level_len=UInt64(0),
            optimization_level=Int32(-1),
        )

        ref raw = self._raw[]
        var runtime_bundle = UnsafeMaybeUninit[RuntimeBundleHandle]()
        var status = raw._raw.bundle_load.f(
            self._handle,
            rebind[StaticBundleHandle](UnsafePointer(to=static_bundle)),
            rebind[CompilationOptionsHandle](UnsafePointer(to=opts)),
            OutParam[RuntimeBundleHandle](to=runtime_bundle),
        )
        if status != STATUS_SUCCESS:
            var err = raw.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to load bundle: {err.message}"),
            )

        return (
            RuntimeBundle(
                _handle=runtime_bundle.unsafe_assume_init_ref(),
                _context_handle=self._handle,
                _raw=self._raw,
            ),
            compiled_info.function_name,
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

    def alloc_sync(self, byte_size: UInt64) raises HALError -> Buffer:
        return Buffer(
            self._raw[].alloc_sync(self._handle, byte_size), byte_size
        )

    def free_sync(self, var mem: Buffer) raises HALError:
        self._raw[].free_sync(self._handle, mem._handle)

    def memory_get_address(self, mem: Buffer) raises HALError -> UInt64:
        """Get the GPU address of a device memory allocation."""
        return self._raw[].get_memory_property["address", UInt64](mem._handle)

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
struct Buffer(Movable):
    """A device memory allocation.

    Tracks the allocation mode and byte size.
    """

    # TODO(Sawyer): decide Buffer ownership. Currently leaks unless the user
    # calls `Context.free_sync`. Either give Buffer a destructor (requires
    # carrying `ContextHandle` + `ArcPointer[RawDriver]`) or document it as a
    # non-owning view and remove `Movable`.

    var _handle: MemoryHandle
    var byte_size: UInt64


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
