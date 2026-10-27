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
from ._plugin._overlay import HAL_PLUGINS

from .buffer import Buffer, BufferView
from .device import DeviceSpec
from .extensions.traits import SupportsPeerAccessV1
from .stream import Stream

from .status import STATUS_SUCCESS, STATUS_UNKNOWN_SYMBOL, HALError

from std.atomic import Atomic
from std.collections import Dict
from std.collections.optional import OptionalReg
from std.memory import (
    ImmPointer,
    ArcPointer,
    Layout,
    MutOpaquePointer,
    ThinAllocation,
    Pointer,
    MaybeUninit,
    alloc,
    dealloc,
)
from std.memory.arc_pointer import WeakPointer
from std.utils.lock import SpinWaiter

from std.compile import CompiledFunctionInfo

from std.compile.compile import _Info, _get_emission_kind_id

from std.reflection import get_linkage_name

from std.collections.string.string_span import _get_kgen_string

from std.sys.info import CompilationTarget


#
def _bundle_file_type[target: CompilationTarget]() -> StaticString:
    """Returns the file_type string each HAL plugin expects for bundle_load."""
    comptime _Target = CompilationTarget[_mlir_value=target._mlir_value]
    if _Target._is_triple_arch["amdgpu"]():
        return "hsaco"
    if target.is_triple["nvptx64-nvidia-cuda"]():
        return "cubin"
    return "object"


@fieldwise_init
struct _CachedFunction(Movable):
    """One cached loaded function: the handle and the bundle that defines it.
    The bundle must outlive the handle, so the entry keeps both and both are
    released together at context teardown."""

    var handle: FunctionHandle
    var bundle: RuntimeBundle


def _code_digest[origin: ImmOrigin](code: StringSlice[origin=origin]) -> UInt64:
    """Hashes `code`'s bytes, identifying a code image by content."""
    return hash_bytes(code.as_bytes())


@fieldwise_init
struct Context[device_spec: DeviceSpec](Deinitable, Movable):
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

    comptime HALExtensions = HAL_PLUGINS.for_target[
        DeviceSpec._mlir_target[Self.device_spec]()
    ].HookFor[DeviceSpec.target[Self.device_spec]()].extensions

    var _extensions: Pointer[Self.HALExtensions, ImmUntrackedOrigin]

    # Loaded functions keyed by code content, so byte-identical code dedupes
    # across the separate buffers it arrives in (one per MEF kernel archive).
    # Entries own their handle and bundle, released at context teardown.
    var _function_cache: Dict[String, _CachedFunction]
    # Address-keyed aliases into `_function_cache`, keeping the per-launch path
    # off the content hash. Sound only because a cached buffer is never freed
    # while this context lives, so an address cannot come to hold other bytes.
    var _function_alias: Dict[String, String]
    var _function_cache_lock: Pointer[Int64, MutUntrackedOrigin]

    comptime _device_spec_target = DeviceSpec.target[Self.device_spec]()

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
        # Points into the Arc'd `_device`, which keeps the extensions
        # instance alive, so the erased origin is sound here.
        self._extensions = (
            Pointer(to=self._device[].extensions())
            .as_imm()
            .unsafe_origin_cast[ImmUntrackedOrigin]()
        )
        self._raw = device._raw
        self._self_ref = WeakPointer[Self]()
        self._function_cache = {}
        self._function_alias = {}
        self._function_cache_lock = alloc(Layout[Int64](length=1)).unsafe_leak()
        self._function_cache_lock[] = 0

        ref raw = self._raw[]

        var context_handle_uninit = MaybeUninit[ContextHandle]()
        var status = raw._raw.base.context_create.get()(
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

        self._handle = context_handle_uninit.unsafe_assume_init()

    def __deinit__(deinit self):
        while len(self._function_cache) > 0:
            try:
                var entry = self._function_cache.popitem()
                try:
                    self.unload_function(entry.value.handle)
                except e:
                    print("warning: unload_function failed:", e)
                _ = entry^
            except e:
                break
        dealloc(
            ThinAllocation(
                unsafe_owned_ptr=self._function_cache_lock
            ).unsafe_with_layout({length = 1})
        )
        try:
            self._raw[].destroy_context(self._handle)
        except e:
            print("warning: destroy_context failed:", e)

    def extensions(
        imm self,
    ) -> ref[origin_of(self)] Self.HALExtensions:
        """Returns a ref to the HAL extension carrier object that
        lives on the `Device` this is peripherally derived from.

        Returns:
            An immutable ref to the extension carrier.
        """

        # SAFETY: this ref is valid up through destruction of `self`,
        # as it must live on the `Device` that `self` was peripherally
        # derived from and that `self` keeps alive
        # through `self._context`.
        return self._extensions.unsafe_origin_cast[origin_of(self)]()[]

    def handle[
        origin: Origin, //
    ](ref[origin] self) -> Pointer[ContextHandle.T, origin]:
        """Returns the raw context handle, tying its origin back to `self`.

        Accessing `self._handle` directly hands out a `ContextHandle` whose
        origin is `MutUntrackedOrigin`, so it keeps nothing alive.

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

    def enable_peer_access(
        self, peer: Context[Self.device_spec]
    ) raises HALError where conforms_to(
        Device[Self.device_spec].Extensions, SupportsPeerAccessV1
    ):
        """Makes `peer`'s memory addressable from this context. Idempotent."""
        var peer_access = self._device[].extensions().peer_access()
        if not peer_access:
            raise HALError(
                STATUS_UNKNOWN_SYMBOL,
                message="plugin does not export peer access",
            )
        peer_access.value().enable_peer_access(
            self._raw[], self._handle, peer._handle
        )

    def get_dlpack_device(
        self, pinned: Bool
    ) raises HALError -> M_driver_dlpack_device:
        """Returns the DLPack `(device_type, device_id)` for this context."""
        return self._device[].get_dlpack_device(pinned)

    # ===-------------------------------------------------------------------===#
    # Synchronous copies
    # ===-------------------------------------------------------------------===#

    def copy_to_device(
        self,
        dst: BufferView,
        src: Pointer[mut=False, UInt8, _],
    ) raises HALError:
        """Copies `dst.byte_size` bytes from host memory into `dst`, blocking
        until complete."""
        self._raw[].copy_to_device_sync(self._handle, dst._view, src)

    def copy_from_device(
        self,
        dst: Pointer[mut=True, UInt8, _],
        src: BufferView,
    ) raises HALError:
        """Copies `src.byte_size` bytes from `src` into host memory, blocking
        until complete."""
        self._raw[].copy_from_device_sync(self._handle, dst, src._view)

    def copy_intra_device(
        self,
        dst: BufferView,
        src: BufferView,
    ) raises HALError:
        """Copies `dst.byte_size` bytes from `src` into `dst`, blocking until
        complete."""
        debug_assert(
            src.byte_size() >= dst.byte_size(),
            "copy_intra_device source view smaller than destination",
        )
        self._raw[].copy_intra_device_sync(self._handle, dst._view, src._view)

    def set_memory(self, dst: BufferView, value: UInt8) raises HALError:
        """Sets every byte of `dst` to `value`, blocking until complete."""
        self._raw[].set_memory_sync(self._handle, dst._view, value)

    def fill(
        self, dst: BufferView, value: UInt64, value_size: UInt64
    ) raises HALError:
        """Fills `dst` with a repeated `value_size`-byte `value`, blocking until
        complete. `value_size` must be one of 1, 2, 4, or 8."""
        self._raw[].fill_sync(self._handle, dst._view, value, value_size)

    @inline(.always)
    def _compile_inner[
        fn_type: TrivialRegisterPassable,
        func: fn_type,
    ](self) -> CompiledFunctionInfo[fn_type, func, Self._device_spec_target,]:
        comptime target = Self._device_spec_target
        comptime emission_kind_id = _get_emission_kind_id[
            "object"
        ]().__mlir_index__()

        var offload = __mlir_op.`kgen.compile_offload`[
            target_type=target._mlir_value,
            emission_kind=emission_kind_id,
            emission_option=_get_kgen_string[
                target.default_compile_options()
            ](),
            emission_link_option=_get_kgen_string[""](),
            func=func,
            _type=_Info,
        ]()

        return CompiledFunctionInfo[fn_type, func, target](
            asm=StaticString(offload.asm),
            function_name=get_linkage_name[func, target=target](),
            module_name=StaticString(offload.module_name),
            emission_kind="object",
        )

    @inline(.always)
    def compile[
        fn_type: TrivialRegisterPassable,
        func: fn_type,
    ](self) raises HALError -> Tuple[
        RuntimeBundle,
        CompiledFunctionInfo[
            fn_type,
            func,
            Self._device_spec_target,
        ],
    ]:
        var compiled_info = self._compile_inner[fn_type, func]()
        var bundle = self.load_bundle(compiled_info.asm)
        return (bundle^, compiled_info)

    def load_bundle[
        asm_origin: ImmOrigin
    ](
        self, asm: StringSlice[origin=asm_origin]
    ) raises HALError -> RuntimeBundle:
        """Loads a runtime bundle from pre-compiled binary bytes."""
        # Each plugin expects a specific file_type string. PTX text is
        # accepted by `cuModuleLoadDataEx` even when file_type="cubin".
        comptime file_type = _bundle_file_type[Self._device_spec_target]()

        var static_bundle = M_driver_static_bundle(
            mapped_data=M_driver_slice(
                data=Pointer(to=asm.unsafe_ptr()[]),
                size=UInt64(asm.byte_length()),
            ),
            file_type=Pointer(
                to=file_type.unsafe_ptr().unsafe_bitcast[Int8]()[]
            ),
            file_type_len=UInt64(file_type.byte_length()),
        )

        var opts = M_driver_bundle_compilation_options(
            debug_level=rebind[ImmPointer[Int8, ImmUntrackedOrigin]]("".ptr()),
            debug_level_len=UInt64(0),
            optimization_level=Int32(-1),
        )

        ref raw = self._raw[]
        var runtime_bundle = MaybeUninit[RuntimeBundleHandle]()
        var status = raw._raw.base.bundle_load.get()(
            self._handle,
            Pointer(to=static_bundle),
            Pointer(to=opts),
            OutParam[RuntimeBundleHandle](to=runtime_bundle),
        )
        if status != STATUS_SUCCESS:
            var err = raw.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to load bundle: {err.message}"),
            )

        return RuntimeBundle(
            _handle=runtime_bundle.unsafe_assume_init(),
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
            _extensions=self._extensions,
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
            _extensions=self._extensions,
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
            _extensions=self._extensions,
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
    ](self, mem: Buffer[Self.device_spec]) raises HALError -> Pointer[
        UInt8, origin
    ]:
        """Get a host-dereferenceable pointer to a host-accessible allocation.

        Host-accessible memory is any allocation on a host (CPU) device and
        pinned GPU allocations (`alloc_host_pinned`); on unified-memory GPUs
        every allocation qualifies. The plugin is the authority on residency:
        """
        return Pointer[UInt8, origin](
            unsafe_from_address=Int(
                self._raw[].get_memory_property["host_address", UInt64](
                    mem._handle
                )
            )
        )

    # ===-------------------------------------------------------------------===#
    # Current-context state
    # ===-------------------------------------------------------------------===#

    def set_current(self) raises HALError:
        """Sets this context as the calling thread's current context."""
        self._raw[].set_context_current(self._handle)

    def get_current_driver_context(self) raises HALError -> Int:
        """Returns the calling thread's active driver-level context as an
        opaque handle."""
        return self._raw[].get_current_driver_context(self._handle)

    def set_current_driver_context(self, driver_context: Int) raises HALError:
        """Installs a handle previously obtained from
        `get_current_driver_context` as the calling thread's current context.
        """
        self._raw[].set_current_driver_context(self._handle, driver_context)

    def get_memory_info(self) raises HALError -> Tuple[UInt64, UInt64]:
        """Returns the free and total device memory, in bytes."""
        return self._raw[].get_context_memory_info(self._handle)

    # ===-------------------------------------------------------------------===#
    # Function execution
    # ===-------------------------------------------------------------------===#

    def load_function(
        self, bundle: RuntimeBundle, var name: String
    ) raises HALError -> FunctionHandle:
        return self._raw[].load_function(self._handle, bundle._handle, name)

    def set_function_attribute(
        self, func: FunctionHandle, attribute: Int32, value: Int32
    ) raises HALError:
        """Sets a backend function attribute (e.g. the dynamic shared-memory
        cap) on a loaded function."""
        self._raw[].set_function_attribute(self._handle, func, attribute, value)

    def _function_cache_lock_acquire(self):
        var expected = Int64(0)
        var waiter = SpinWaiter()
        while not Atomic.compare_exchange(
            self._function_cache_lock, expected, 1
        ):
            waiter.wait()
            expected = 0

    def _function_cache_lock_release(self):
        Atomic.store(self._function_cache_lock, 0)

    @staticmethod
    def _function_cache_key(
        discriminator: UInt64,
        byte_length: Int,
        name: String,
        attr_code: OptionalReg[Int32],
        attr_value: Int32,
    ) -> String:
        """Builds a cache key from a code discriminator -- a content hash for
        `_function_cache`, the buffer address for `_function_alias` -- plus the
        code length, the linkage name and the optional function attribute. The
        length pairs with the hash so a 64-bit collision alone cannot alias two
        different code images."""
        var key = String(t"{discriminator}:{byte_length}:{name}")
        if case .Some(attr_code) = attr_code:
            key += String(t"#{attr_code}={attr_value}")
        return key^

    def _function_cache_find(
        mut self, key: String
    ) -> OptionalReg[FunctionHandle]:
        """Content-key lookup; acquires and releases the cache lock.

        The key fully identifies the loaded function (see
        `get_or_load_function`), so this is a plain 1:1 lookup."""
        self._function_cache_lock_acquire()
        var result = OptionalReg[FunctionHandle]()
        try:
            result = self._function_cache._find_ref(key).handle
        except e:
            pass
        self._function_cache_lock_release()
        return result

    def _function_alias_find(
        mut self, addr_key: String
    ) -> OptionalReg[FunctionHandle]:
        """Hot-path lookup: address key -> content key -> handle, resolved under
        a single lock acquisition."""
        self._function_cache_lock_acquire()
        var result = OptionalReg[FunctionHandle]()
        try:
            ref content_key = self._function_alias._find_ref(addr_key)
            result = self._function_cache._find_ref(content_key).handle
        except e:
            pass
        self._function_cache_lock_release()
        return result

    def _function_alias_insert(
        mut self, var addr_key: String, var content_key: String
    ):
        """Points `addr_key` at an existing `_function_cache` entry so later
        launches from the same buffer skip hashing the code."""
        self._function_cache_lock_acquire()
        self._function_alias[addr_key^] = content_key^
        self._function_cache_lock_release()

    def _function_cache_insert(
        mut self,
        var key: String,
        var entry: _CachedFunction,
    ) -> OptionalReg[FunctionHandle]:
        """Inserts `entry` under `key`, re-checking for a load race; acquires
        and releases the cache lock.

        If another thread cached `key` while the caller was loading, discards
        the duplicate, unloading its function BEFORE its `entry` (and so
        the bundle that defines the function) is destroyed, and returns the
        cached entry's handle. Otherwise inserts `entry` and returns None."""
        self._function_cache_lock_acquire()
        var raced = OptionalReg[FunctionHandle]()
        try:
            raced = self._function_cache._find_ref(key).handle
        except e:
            pass
        if raced:
            self._function_cache_lock_release()
            # The duplicate's function must be unloaded before `entry` --
            # and with it the `RuntimeBundle` defining that function -- is
            # destroyed; destroying the bundle first invalidates the
            # handle.
            try:
                self.unload_function(entry.handle)
            except e:
                print("warning: unload_function failed:", e)
            _ = entry^
            return raced
        self._function_cache[key^] = entry^
        self._function_cache_lock_release()
        return None

    def _get_or_load_function_keyed[
        asm_origin: ImmOrigin
    ](
        mut self,
        asm: StringSlice[origin=asm_origin],
        name: String,
        content_key: String,
        attr_code: OptionalReg[Int32],
        attr_value: Int32,
    ) raises HALError -> FunctionHandle:
        """Returns the cached function for `content_key`, loading `asm` on first
        use."""
        var found = self._function_cache_find(content_key)
        if case .Some(found) = found:
            return found

        # Load outside the lock so concurrent first-launches don't serialize.
        var bundle = self.load_bundle(asm)
        var func = self.load_function(bundle, name.copy())
        if case .Some(attr_code) = attr_code:
            self.set_function_attribute(func, attr_code, attr_value)

        var raced = self._function_cache_insert(
            content_key.copy(),
            _CachedFunction(func, bundle^),
        )
        if case .Some(raced) = raced:
            # Another thread cached this function while we were loading;
            # the insert discarded our duplicate load. Use the cached one.
            return raced
        return func

    def get_or_load_function(
        mut self,
        asm: StaticString,
        name: String,
        attr_code: OptionalReg[Int32] = None,
        attr_value: Int32 = 0,
    ) raises HALError -> FunctionHandle:
        """Returns the loaded function for `name`, loading `asm` on first use.

        The per-launch entry point: looks `asm` up by buffer address and only
        hashes the code image on a miss. `asm` must be `StaticString` for that
        alias to be sound. Callers holding code with a narrower lifetime want
        `get_or_load_function_dynamic`.
        """
        var addr_key = Self._function_cache_key(
            UInt64(Int(asm.unsafe_ptr())),
            asm.byte_length(),
            name,
            attr_code,
            attr_value,
        )
        var found = self._function_alias_find(addr_key)
        if case .Some(found) = found:
            return found

        var content_key = Self._function_cache_key(
            _code_digest(asm), asm.byte_length(), name, attr_code, attr_value
        )
        var func = self._get_or_load_function_keyed(
            asm, name, content_key, attr_code, attr_value
        )
        self._function_alias_insert(addr_key^, content_key^)
        return func

    def get_or_load_function_dynamic[
        asm_origin: ImmOrigin
    ](
        mut self,
        asm: StringSlice[origin=asm_origin],
        name: String,
        attr_code: OptionalReg[Int32] = None,
        attr_value: Int32 = 0,
    ) raises HALError -> FunctionHandle:
        """Returns the loaded function for `name`, loading `asm` on first use.

        Keyed purely on code content, so `asm` may have any lifetime; it records
        no address alias and therefore hashes the code image on every call.
        """
        var content_key = Self._function_cache_key(
            _code_digest(asm), asm.byte_length(), name, attr_code, attr_value
        )
        return self._get_or_load_function_keyed(
            asm, name, content_key, attr_code, attr_value
        )

    def compile_function_inline[
        func: Some[TrivialRegisterPassable]
    ](mut self) raises HALError -> FunctionHandle:
        """Returns the loaded function that is passed as parameter, loading the
        corresponding asm on first use.
        """
        var info = self._compile_inner[type_of(func), func]()
        return self.get_or_load_function(info.asm, info.function_name)

    def unload_function(self, func: FunctionHandle) raises HALError:
        self._raw[].unload_function(self._handle, func)

    def function_occupancy_max_active_blocks(
        self,
        func: FunctionHandle,
        block_size: Int32,
        dynamic_shared_memory_bytes: UInt64,
    ) raises HALError -> Int32:
        """Returns the max number of resident blocks per multiprocessor for
        the function at the given block size and dynamic shared-memory usage.
        """
        return self._raw[].function_occupancy_max_active_blocks(
            self._handle, func, block_size, dynamic_shared_memory_bytes
        )

    def tensor_map_encode_tiled(
        self,
        tensor_map: MutOpaquePointer[_],
        data_type: Int32,
        rank: Int32,
        global_device_address: UInt64,
        global_dim: Pointer[mut=False, UInt64, _],
        global_strides: Pointer[mut=False, UInt64, _],
        box_dim: Pointer[mut=False, UInt32, _],
        element_strides: Pointer[mut=False, UInt32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises HALError:
        """Encodes a tiled TMA descriptor into `tensor_map` for this
        context's device."""
        self._raw[].tensor_map_encode_tiled(
            self._handle,
            tensor_map,
            data_type,
            rank,
            global_device_address,
            global_dim,
            global_strides,
            box_dim,
            element_strides,
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
        )

    def tensor_map_encode_im2col(
        self,
        tensor_map: MutOpaquePointer[_],
        data_type: Int32,
        rank: Int32,
        global_device_address: UInt64,
        global_dim: Pointer[mut=False, UInt64, _],
        global_strides: Pointer[mut=False, UInt64, _],
        pixel_box_lower_corner: Pointer[mut=False, Int32, _],
        pixel_box_upper_corner: Pointer[mut=False, Int32, _],
        channels_per_pixel: Int32,
        pixels_per_column: Int32,
        element_strides: Pointer[mut=False, UInt32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises HALError:
        """Encodes an im2col TMA descriptor into `tensor_map` for this
        context's device."""
        self._raw[].tensor_map_encode_im2col(
            self._handle,
            tensor_map,
            data_type,
            rank,
            global_device_address,
            global_dim,
            global_strides,
            pixel_box_lower_corner,
            pixel_box_upper_corner,
            channels_per_pixel,
            pixels_per_column,
            element_strides,
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
        )


@fieldwise_init
struct RuntimeBundle(Movable):
    var _handle: RuntimeBundleHandle
    var _context_handle: ContextHandle
    var _raw: ArcPointer[RawDriver]

    def __deinit__(deinit self):
        try:
            self._raw[].unload_bundle(self._context_handle, self._handle)
        except e:
            print("warning: unload_bundle failed:", e)
