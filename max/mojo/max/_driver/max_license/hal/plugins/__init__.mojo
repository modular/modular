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


# ===-----------------------------------------------------------------------===#
# Version negotiation struct — must match C layout of M_driver_version.
# ===-----------------------------------------------------------------------===#

comptime M_DRIVER_INTERFACE_VERSION_MAJOR: UInt32 = 0
comptime M_DRIVER_INTERFACE_VERSION_MINOR: UInt32 = 1
comptime M_DRIVER_INTERFACE_VERSION_PATCH: UInt32 = 0


@fieldwise_init
struct DriverVersion(TrivialRegisterPassable):
    """Mirrors C `M_driver_version { uint32_t major, minor, patch; }`."""

    var major: UInt32
    var minor: UInt32
    var patch: UInt32


comptime DriverHandle = ExternalHandle[M_driver_driver]
comptime DeviceHandle = ExternalHandle[M_driver_device]
comptime ContextHandle = ExternalHandle[M_driver_context]
comptime QueueHandle = ExternalHandle[M_driver_queue]
comptime EventHandle = ExternalHandle[M_driver_event]
comptime FunctionHandle = ExternalHandle[M_driver_function]
comptime MemoryHandle = ExternalHandle[M_driver_memory]
comptime RuntimeBundleHandle = ExternalHandle[M_driver_runtime_bundle]


@fieldwise_init
struct M_driver_memory_view(Copyable, Movable):
    var memory: MemoryHandle
    var offset: UInt64
    var size: UInt64


comptime MemoryViewHandle[origin: ImmOrigin] = ImmPointer[
    M_driver_memory_view, origin
]

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
        out self, var handle: _DLHandle, so_path: StringSlice
    ) raises HALError:
        self = Self(handle, so_path, use_reflection=())

    def __init__(
        out self,
        var handle: _DLHandle,
        so_path: StringSlice,
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

    def __init__(
        out self, lib: _DLHandle, so_path: StringSlice
    ) raises HALError:
        comptime assert __fn_type_is_cabi[Self.fn_type](), (
            "fn_type must conform to C ABI; declare the RawPlugin field"
            ' as `def(...) thin abi("C") -> PluginResultCode`'
        )

        var symbol_exists = lib.check_symbol(Self.name)

        comptime if Self.required:
            if not symbol_exists:
                raise HALError(
                    STATUS_UNKNOWN_SYMBOL,
                    message=String(
                        t"required symbol {Self.name} was not present in plugin"
                        t" .so {so_path}"
                    ),
                )

        if symbol_exists:
            self._f = lib.get_function[Self.fn_type](Self.name)
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
    var graph_capture: Optional[RawPluginGraphCapture[]]

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
        out self, var lib: OwnedDLHandle, so_path: String
    ) raises HALError:
        self._lib = lib^
        self.name = so_path
        self.so_path = so_path
        var handle = self._lib.borrow()

        self.base = type_of(self.base)(handle, so_path)
        self.graph_capture = Self._try_init_optional_component[
            RawPluginGraphCapture[]
        ](handle, so_path)

    def _create_init_driver(mut self) raises HALError -> DriverHandle:
        var handle = MaybeUninit[DriverHandle]()
        var version = DriverVersion(
            major=M_DRIVER_INTERFACE_VERSION_MAJOR,
            minor=M_DRIVER_INTERFACE_VERSION_MINOR,
            patch=M_DRIVER_INTERFACE_VERSION_PATCH,
        )

        var status = self.base.create.get()(
            ImmPointer(to=version),
            OutParam[DriverHandle](to=handle),
        )
        if status != STATUS_SUCCESS:
            raise HALError(
                status,
                message=String(
                    t"failed to initialise driver plugin from {self.so_path}"
                ),
            )

        var driver_handle = handle.unsafe_assume_init()

        return driver_handle


trait RawPluginComponent(Deinitable, HALPluginConstructable, Movable):
    pass
