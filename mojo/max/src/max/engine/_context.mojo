# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections.optional import Optional
from sys.ffi import DLHandle
from sys.param_env import is_defined

from max._utils import call_dylib_func, exchange
from max.driver import Device
from memory import UnsafePointer

from ._status import Status

alias MODULAR_PRODUCTION = is_defined["MODULAR_PRODUCTION"]()


@value
@register_passable("trivial")
struct AllocatorType:
    var value: Int32
    # This needs to map M_AllocatorType enum on the C API side.
    alias SYSTEM = Int32(0)
    alias CACHING = Int32(1)

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int32):
        self.value = value

    @always_inline("nodebug")
    fn __ne__(self, rhs: Int32) -> Bool:
        return self.value != rhs.value


@value
@register_passable("trivial")
struct CRuntimeConfig:
    var ptr: UnsafePointer[NoneType]

    alias FreeRuntimeConfigFnName = "M_freeRuntimeConfig"
    alias SetAllocatorTypeFnName = "M_setAllocatorType"
    alias SetDeviceFnName = "M_setDevice"
    alias SetMaxContextFnName = "M_setMaxContext"
    alias SetAPILanguageFnName = "M_setAPILanguage"

    @implicit
    fn __init__(out self, ptr: UnsafePointer[NoneType]):
        self.ptr = ptr

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeRuntimeConfigFnName, self)

    fn set_device(self, lib: DLHandle, device: Device):
        call_dylib_func(lib, Self.SetDeviceFnName, self, device._cdev)

    fn set_api_language(self, lib: DLHandle, source: String):
        call_dylib_func(
            lib, Self.SetAPILanguageFnName, self, source.unsafe_ptr()
        )

    fn set_allocator_type(self, lib: DLHandle, allocator_type: AllocatorType):
        call_dylib_func(lib, Self.SetAllocatorTypeFnName, self, allocator_type)

    fn set_max_context(
        self, lib: DLHandle, max_context: UnsafePointer[NoneType]
    ) -> None:
        call_dylib_func(lib, Self.SetMaxContextFnName, self, max_context)


struct RuntimeConfig:
    var ptr: CRuntimeConfig
    var lib: DLHandle

    alias NewRuntimeConfigFnName = "M_newRuntimeConfig"

    fn __init__(
        out self,
        lib: DLHandle,
        device: Device,
        allocator_type: AllocatorType = AllocatorType.CACHING,
        max_context: UnsafePointer[NoneType] = UnsafePointer[NoneType](),
    ):
        self.ptr = call_dylib_func[CRuntimeConfig](
            lib, Self.NewRuntimeConfigFnName
        )

        if max_context:
            # `mojo-run` already has an existing `M::Context`.
            # Set the runtime config to reuse this existing context, rather
            # than trying to recreate a new one.
            self.ptr.set_max_context(lib, max_context)

        self.lib = lib

        if allocator_type != AllocatorType.CACHING:
            self.ptr.set_allocator_type(self.lib, allocator_type)

        self.ptr.set_device(self.lib, device)

        self.ptr.set_api_language(self.lib, "mojo")

    fn __moveinit__(out self, owned existing: Self):
        self.ptr = exchange[CRuntimeConfig](
            existing.ptr, UnsafePointer[NoneType]()
        )
        self.lib = existing.lib

    fn borrow_ptr(self) -> CRuntimeConfig:
        """
        Borrow the underlying C ptr.
        """
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)


@value
@register_passable("trivial")
struct CRuntimeContext:
    var ptr: UnsafePointer[NoneType]

    @implicit
    fn __init__(out self, ptr: UnsafePointer[NoneType]):
        self.ptr = ptr

    alias FreeRuntimeContextFnName = "M_freeRuntimeContext"

    fn free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeRuntimeContextFnName, self)


struct RuntimeContext:
    var ptr: CRuntimeContext
    var lib: DLHandle

    alias NewRuntimeContextFnName = "M_newRuntimeContext"
    alias SetDebugPrintOptionsFnName = "M_setDebugPrintOptions"

    fn __init__(out self, owned config: RuntimeConfig, lib: DLHandle):
        var status = Status(lib)
        self.ptr = call_dylib_func[CRuntimeContext](
            lib,
            Self.NewRuntimeContextFnName,
            config.borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            print(status.__str__())
            self.ptr = UnsafePointer[NoneType]()
        _ = config^
        self.lib = lib

    fn __moveinit__(out self, owned existing: Self):
        self.ptr = exchange[CRuntimeContext](
            existing.ptr, UnsafePointer[NoneType]()
        )
        self.lib = existing.lib

    fn borrow_ptr(self) -> CRuntimeContext:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.lib

    fn set_debug_print_options(
        mut self,
        style: PrintStyle,
        precision: UInt,
        owned output_directory: String,
    ):
        _ = call_dylib_func[CRuntimeContext](
            self.lib,
            Self.SetDebugPrintOptionsFnName,
            self.ptr,
            style.style,
            precision,
            output_directory.unsafe_cstr_ptr(),
        )


@value
@register_passable("trivial")
struct PrintStyle:
    var style: Int32

    alias COMPACT = PrintStyle(0)
    alias FULL = PrintStyle(1)
    alias BINARY = PrintStyle(2)
    alias NONE = PrintStyle(3)
