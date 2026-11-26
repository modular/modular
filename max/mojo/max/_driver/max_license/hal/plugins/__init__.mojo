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
from std.collections.string.string_span import get_static_string

from std.ffi import (
    __fn_type_is_cabi,
    _DLHandle,
    OwnedDLHandle,
    RTLD,
    CStringSpan,
    UnsafeUnion,
    c_uchar,
    c_char,
    c_uint,
    c_int,
)

from std.memory import (
    alloc,
    ImmPointer,
    MutOpaquePointer,
    MutPointer,
    OpaquePointer,
    Pointer,
    MaybeUninit,
)

from ..status import (
    STATUS_SUCCESS,
    STATUS_BAD_DRIVER,
    STATUS_INCOMPATIBLE_VERSION,
    STATUS_UNKNOWN_ERROR,
    STATUS_UNKNOWN_PROPERTY_NAME,
    STATUS_UNKNOWN_SYMBOL,
    HALError,
)

from .base import RawPluginBase
from .dcc_graph_capture import RawPluginGraphCapture

# ===-----------------------------------------------------------------------===#
# Shared plugin structs across FFI
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct M_driver_slice[origin: ImmOrigin](TrivialRegisterPassable):
    var data: ImmPointer[UInt8, Self.origin]
    var size: UInt64


@fieldwise_init
struct M_driver_static_bundle[origin: ImmOrigin](TrivialRegisterPassable):
    var mapped_data: M_driver_slice[Self.origin]
    var file_type: ImmPointer[Int8, ImmStaticOrigin]
    var file_type_len: UInt64


@fieldwise_init
struct M_driver_dim(TrivialRegisterPassable):
    var x: UInt32
    var y: UInt32
    var z: UInt32


@fieldwise_init
struct M_driver_dlpack_device(TrivialRegisterPassable):
    var device_type: Int32
    var device_id: Int32


@fieldwise_init
struct M_driver_queue_execute_config_gpu[attr_origin: MutOrigin](
    TrivialRegisterPassable
):
    var grid: M_driver_dim
    var block: M_driver_dim
    var shared_mem_bytes: UInt32
    var attributes: OptionalReg[OpaquePointer[Self.attr_origin]]
    var num_attributes: UInt32


@fieldwise_init
struct M_driver_queue_execute_mode(TrivialRegisterPassable):
    var value: Int32

    comptime GPU = Self(value=0)


@fieldwise_init
struct M_driver_queue_execute_config[attr_origin: MutOrigin]:
    var mode: M_driver_queue_execute_mode
    var config: UnsafeUnion[M_driver_queue_execute_config_gpu[Self.attr_origin]]


@fieldwise_init
struct M_driver_bundle_compilation_options(TrivialRegisterPassable):
    var debug_level: ImmPointer[Int8, ImmUntrackedOrigin]
    var debug_level_len: UInt64
    var optimization_level: Int32


# ===-----------------------------------------------------------------------===#
# Opaque plugin structs for handles.
# ===-----------------------------------------------------------------------===#


struct M_driver_driver:
    pass


struct M_driver_device:
    pass


struct M_driver_context:
    pass


struct M_driver_queue:
    pass


struct M_driver_event:
    pass


struct M_driver_function:
    pass


struct M_driver_memory:
    pass


struct M_driver_runtime_bundle:
    pass


struct M_driver_flag:
    pass


# ===-----------------------------------------------------------------------===#
# Version negotiation struct — must match C layout of M_driver_version.
# ===-----------------------------------------------------------------------===#

comptime M_DRIVER_INTERFACE_VERSION_MAJOR: UInt32 = 0
comptime M_DRIVER_INTERFACE_VERSION_MINOR: UInt32 = 1
comptime M_DRIVER_INTERFACE_VERSION_PATCH: UInt32 = 0


@fieldwise_init
struct DriverVersion(TrivialRegisterPassable, Writable):
    """Mirrors C `M_driver_version { uint32_t major, minor, patch; }`."""

    var major: UInt32
    var minor: UInt32
    var patch: UInt32

    def accepts(self, plugin: Self) -> Bool:
        """Reports whether a host at this version may use a plugin at `plugin`.

        The rule is the one `M_driver_version` documents in
        `SDK/include/HAL/M_driver_common.h`: equal major versions, and a plugin
        minor version at least the host's.

        Args:
            plugin: The version the plugin implements.

        Returns:
            True if the plugin's version is compatible with this one.
        """
        return plugin.major == self.major and plugin.minor >= self.minor

    def write_to(self, mut writer: Some[Writer]):
        """Writes the version as `major.minor.patch`.

        Args:
            writer: The writer to write to.
        """
        writer.write(self.major, ".", self.minor, ".", self.patch)


comptime HOST_DRIVER_VERSION = DriverVersion(
    major=M_DRIVER_INTERFACE_VERSION_MAJOR,
    minor=M_DRIVER_INTERFACE_VERSION_MINOR,
    patch=M_DRIVER_INTERFACE_VERSION_PATCH,
)
"""The driver interface version this build implements."""


comptime DriverHandle = ExternalHandle[M_driver_driver]
comptime DeviceHandle = ExternalHandle[M_driver_device]
comptime ContextHandle = ExternalHandle[M_driver_context]
comptime QueueHandle = ExternalHandle[M_driver_queue]
comptime EventHandle = ExternalHandle[M_driver_event]
comptime FunctionHandle = ExternalHandle[M_driver_function]
comptime MemoryHandle = ExternalHandle[M_driver_memory]
comptime RuntimeBundleHandle = ExternalHandle[M_driver_runtime_bundle]
comptime FlagHandle = ExternalHandle[M_driver_flag]


@fieldwise_init
struct M_driver_memory_view(Copyable, Movable):
    var memory: MemoryHandle
    var offset: UInt64
    var size: UInt64


comptime MemoryViewHandle = ImmPointer[M_driver_memory_view, _]

# ===-----------------------------------------------------------------------===#
# C API function pointer types
# ===-----------------------------------------------------------------------===#

# A handle to memory the raw HAL plugin owns. This means ownership
# lives outside of Mojo, so leans on UntrackedOrigin.
comptime ExternalHandle[mut: Bool = False, //, T: AnyType] = Pointer[
    T, UntrackedOrigin[mut=mut]
]

# A handle to a Mojo-owned value passed by pointer into a plugin FFI call.
comptime InternalHandle[
    mut: Bool = False, //, T: AnyType, origin: Origin[mut=mut]
] = Pointer[T, origin]

comptime OutParam[T: TrivialRegisterPassable, origin: MutOrigin] = Pointer[
    MaybeUninit[T], origin
]

comptime PluginResultCode = Int64


trait HALPluginConstructable(Deinitable, TrivialRegisterPassable):
    def __init__(
        out self, var handle: _DLHandle, so_path: StringSpan
    ) raises HALError:
        self = Self(handle, so_path, use_reflection=())

    def __init__(
        out self,
        var handle: _DLHandle,
        so_path: StringSpan,
        *,
        use_reflection: Tuple[],
    ) raises HALError:
        # This looks very weird, but is indeed sound here.
        # All HALFunction are TrivialRegisterPassable,
        # so can not have any deinit or move action.
        # This is all here to make sure we don't have `undef` or `poison`
        # bits surviving the constructor.

        # We first mark this entire struct as initialized, inheriting
        # those undef values. We don't _use_ any of those values
        # before a write, though.
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

        # We then pun a ref to each field into a
        # MaybeUninit[FieldTy] so that we can write into it
        # with something that should reliably actually initialize the type.
        # A Pointer and unsafe_write may be fine here now, but
        # MaybeUninit should continue to guarantee a `write` will
        # properly initialize the backing storage.
        comptime r = reflect[Self]
        comptime for i in range(r.field_count()):
            comptime field = r.field_at[i]
            comptime if conforms_to(field.T, HALPluginConstructable):
                var uninit = Pointer(to=r.field_ref[i](self)).unsafe_bitcast[
                    MaybeUninit[field.T]
                ]()
                uninit[].write(field.T(handle, so_path))

                # TODO: ideally something like this would work, and be less gross:
                # r.field_ref[i](self) = field.T(handle, so_path)


# Every plugin entry point is a `HALFunction` field, and each module that loads
# a plugin inlines all of their constructors. Keeping the lookup and its error
# path in one shared function, rather than in each constructor, keeps that
# per-module code small.
@no_inline
def _resolve_hal_symbol(
    lib: _DLHandle, name: StaticString, so_path: StringSpan, required: Bool
) raises HALError -> Optional[Pointer[NoneType, MutUntrackedOrigin]]:
    """Looks up a plugin symbol, raising if a required one is missing.

    Args:
        lib: The loaded plugin library.
        name: The symbol name. Must be nul-terminated.
        so_path: The plugin path, for the error message.
        required: Whether a missing symbol is an error.

    Returns:
        The symbol's address, or `None` if it is optional and missing.

    Raises:
        `STATUS_UNKNOWN_SYMBOL` if `required` is set and the symbol is missing.
    """
    var symbol = lib.get_symbol[NoneType](cstr_name=name.as_c_string_span())
    if not symbol and required:
        raise HALError(
            STATUS_UNKNOWN_SYMBOL,
            message=String(
                t"required symbol {name} was not present in plugin .so"
                t" {so_path}"
            ),
        )
    return symbol


struct HALFunction[
    name: StaticString, fn_type: TrivialRegisterPassable, required: Bool = True
](
    Boolable,
    Copyable,
    HALPluginConstructable,
    Movable,
    TrivialRegisterPassable,
):
    """
    HALFunction handles loading of a HAL function symbol from a HAL plugin.

    Parameters:
        name: The symbol name of the function to be loaded from the HAL plugin.
        fn_type: The function type of the function to be loaded from the HAL plugin.
        required: If True, loading the provided HAL plugin will fail if the symbol is not present.
                  If False, loading will continue but accessing the symbol becomes fallible.
                  This should be used for symbols that are only optionally implemented rather
                  than being part of the base capability set.
    """

    var _f: OptionalReg[Self.fn_type]

    def __init__(out self, lib: _DLHandle, so_path: StringSpan) raises HALError:
        comptime assert __fn_type_is_cabi[Self.fn_type](), (
            "fn_type must conform to C ABI; declare the RawPlugin field"
            ' as `def(...) thin abi("C") -> PluginResultCode`'
        )

        var symbol = _resolve_hal_symbol(
            lib, get_static_string[Self.name](), so_path, Self.required
        )
        if case .Some(address) = symbol:
            self._f = Pointer(to=address).unsafe_bitcast[Self.fn_type]()[]
        else:
            self._f = None

    def __bool__(self) -> Bool:
        comptime if Self.required:
            return True
        else:
            return Bool(self._f)

    def get(self) -> Self.fn_type where Self.required:
        return self._f.unsafe_value()

    def try_get(self) -> Optional[Self.fn_type]:
        return self._f

    # TODO: figure out if we can do fn type reflection to be able to do
    # typewise variadic call through to the fn ptr
    # @always_inline
    # def __call__(self, *args: *_) -> _:
    #    return self.f(*args)


struct RawPlugin(Movable):
    """Loaded HAL plugin with resolved function pointers.

    Owns the dlopen'd shared library handle and all resolved symbols.
    Dropping this struct dlcloses the library.
    """

    var _lib: OwnedDLHandle
    var name: String
    var so_path: String

    var base: RawPluginBase[]
    var graph_capture: Optional[RawPluginGraphCapture]

    @staticmethod
    def _try_init_optional_component[
        Component: RawPluginComponent
    ](
        mut handle: _DLHandle,
        ref so_path: String,
        out field: Optional[Component],
    ) raises HALError:
        try:
            var component = Component(handle, so_path)
            field = Optional(component)
        except e:
            if e.status == STATUS_UNKNOWN_SYMBOL:
                field = Optional[Component]()
            else:
                raise e^

    def __init__(
        out self, var lib: OwnedDLHandle, *, name: String, so_path: String
    ) raises HALError:
        self._lib = lib^
        self.name = name
        self.so_path = so_path
        var handle = self._lib.borrow()

        self.base = type_of(self.base)(handle, so_path)
        self.graph_capture = Self._try_init_optional_component[
            RawPluginGraphCapture
        ](handle, so_path)

    def _create_init_driver(mut self) raises HALError -> DriverHandle:
        """Creates the plugin's driver, following the `M_driver_create`
        contract in `SDK/include/HAL/M_driver_driver.h`.
        """
        var version = HOST_DRIVER_VERSION
        var handle = MaybeUninit(OptionalReg[DriverHandle]())
        var status = self.base.create.get()(
            MutPointer(to=version),
            OutParam[OptionalReg[DriverHandle]](to=handle),
        )
        var driver = handle.unsafe_assume_init()

        if status == STATUS_INCOMPATIBLE_VERSION or (
            status == STATUS_SUCCESS
            and not HOST_DRIVER_VERSION.accepts(version)
        ):
            if case .Some(d) = driver:
                _ = self.base.destroy.get()(d)
            # A plugin built before the header required the write-back refuses
            # without one, which would otherwise read as the host's own
            # version.
            if (
                status == STATUS_INCOMPATIBLE_VERSION
                and version.major == HOST_DRIVER_VERSION.major
                and version.minor == HOST_DRIVER_VERSION.minor
                and version.patch == HOST_DRIVER_VERSION.patch
            ):
                raise HALError(
                    STATUS_INCOMPATIBLE_VERSION,
                    message=String(
                        t"driver plugin '{self.so_path}' rejected this build's"
                        t" driver interface {HOST_DRIVER_VERSION} without"
                        t" reporting the version it implements"
                    ),
                )
            raise HALError(
                STATUS_INCOMPATIBLE_VERSION,
                message=String(
                    t"driver plugin '{self.so_path}' implements driver"
                    t" interface {version}, which is incompatible with this"
                    t" build's {HOST_DRIVER_VERSION}: the major versions must"
                    t" match and the plugin's minor version must be at least"
                    t" this build's"
                ),
            )

        if status != STATUS_SUCCESS:
            var detail = String("the plugin returned no handle to describe it")
            if case .Some(d) = driver:
                detail = self.status_message(d, status).message
                _ = self.base.destroy.get()(d)
            raise HALError(
                status,
                message=String(
                    t"driver plugin '{self.so_path}' failed to create its"
                    t" driver: {detail}"
                ),
            )

        if case .Some(d) = driver:
            return d
        raise HALError(
            STATUS_BAD_DRIVER,
            message=String(
                t"driver plugin '{self.so_path}' reported success from"
                t" M_driver_create without a driver"
            ),
        )

    def status_message(self, driver: DriverHandle, status: Int64) -> HALError:
        """Retrieves the plugin's message for `status` through `driver`.

        `driver` may be a handle that a failed `M_driver_create` returned only
        to describe its failure.
        """
        var buf = List[Int8](length=1024, fill=0)
        var ret = self.base.status_message.get()(
            driver,
            status,
            MutPointer(to=buf.unsafe_ptr()[]),
            Int64(len(buf)),
        )
        if ret != STATUS_SUCCESS:
            return HALError(
                status, message="(problem retrieving status message)"
            )
        try:
            return HALError(
                status,
                message=CStringSpan(
                    unsafe_from_ptr=Pointer(to=buf[0]).unsafe_bitcast[c_char]()
                ),
            )
        except:
            return HALError(status, message="(failed to decode status message)")


trait RawPluginComponent(Deinitable, HALPluginConstructable, Movable):
    pass
