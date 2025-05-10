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

from sys.ffi import DLHandle

from max._utils import call_dylib_func, get_lib_path_from_cfg
from max.tensor import TensorSpec
from memory import ArcPointer, UnsafePointer

from ._status import _CStatus


struct ManagedDLHandle:
    var lib: DLHandle

    @implicit
    fn __init__(out self, lib: DLHandle):
        self.lib = lib

    fn __moveinit__(out self, owned existing: Self):
        self.lib = existing.lib

    fn get_handle(self) -> DLHandle:
        return self.lib

    fn __del__(owned self):
        if self.lib:
            self.lib.close()


@value
struct DriverLibrary:
    var lib: ArcPointer[ManagedDLHandle]

    alias device_type = UnsafePointer[NoneType]
    alias device_memory_type = UnsafePointer[NoneType]

    alias destroy_device_fn_sig = fn (Self.device_type) -> None
    var destroy_device_fn: Self.destroy_device_fn_sig

    alias create_cpu_device_fn_sig = fn (Int, _CStatus) -> Self.device_type
    var create_cpu_device_fn: Self.create_cpu_device_fn_sig

    alias create_accelerator_device_fn_sig = fn (
        Int, _CStatus
    ) -> Self.device_type
    var create_accelerator_device_fn: Self.create_accelerator_device_fn_sig

    alias copy_device_fn_sig = fn (Self.device_type) -> Self.device_type
    var copy_device_fn: Self.copy_device_fn_sig

    alias free_device_data_fn_sig = fn (
        Self.device_type, UnsafePointer[UInt8], _CStatus
    ) -> None
    var free_device_data_fn: Self.free_device_data_fn_sig

    alias get_device_desc_fn_sig = fn (Self.device_type) -> UnsafePointer[UInt8]
    var get_device_desc_fn: Self.get_device_desc_fn_sig

    alias create_device_memory_fn_sig = fn (
        UnsafePointer[TensorSpec], Self.device_type, _CStatus
    ) -> Self.device_memory_type
    var create_device_memory_fn: Self.create_device_memory_fn_sig

    alias destroy_device_memory_fn_sig = fn (Self.device_memory_type) -> None
    var destroy_device_memory_fn: Self.destroy_device_memory_fn_sig

    alias copy_device_memory_fn_sig = fn (
        Self.device_memory_type, Self.device_memory_type, _CStatus
    ) -> None
    var copy_device_memory_fn: Self.copy_device_memory_fn_sig

    alias get_data_fn_sig = fn (Self.device_memory_type) -> UnsafePointer[UInt8]
    var get_data_fn: Self.get_data_fn_sig

    alias accelerator_count_fn_sig = fn () -> Int
    var accelerator_count_fn: Self.accelerator_count_fn_sig

    fn __init__(out self) raises:
        var lib = DLHandle(_get_driver_path())
        self.destroy_device_fn = lib.get_function[Self.destroy_device_fn_sig](
            "M_destroyDevice"
        )
        self.create_cpu_device_fn = lib.get_function[
            Self.create_cpu_device_fn_sig
        ]("M_createCPUDevice")
        self.create_accelerator_device_fn = lib.get_function[
            Self.create_accelerator_device_fn_sig
        ]("M_createAcceleratorDevice")
        self.copy_device_fn = lib.get_function[Self.copy_device_fn_sig](
            "M_copyDevice"
        )
        self.free_device_data_fn = lib.get_function[
            Self.free_device_data_fn_sig
        ]("M_freeDeviceData")
        self.get_device_desc_fn = lib.get_function[Self.get_device_desc_fn_sig](
            "M_getDeviceDesc"
        )
        self.create_device_memory_fn = lib.get_function[
            Self.create_device_memory_fn_sig
        ]("M_createDeviceMemory")
        self.destroy_device_memory_fn = lib.get_function[
            Self.destroy_device_memory_fn_sig
        ]("M_destroyDeviceMemory")
        self.copy_device_memory_fn = lib.get_function[
            Self.copy_device_memory_fn_sig
        ]("M_copyDeviceMemory")
        self.get_data_fn = lib.get_function[Self.get_data_fn_sig]("M_getData")
        self.accelerator_count_fn = lib.get_function[
            Self.accelerator_count_fn_sig
        ]("M_getAcceleratorCount")
        self.lib = ArcPointer[ManagedDLHandle](lib)

    fn get_handle(self) -> DLHandle:
        return self.lib[].get_handle()


fn _get_driver_path() raises -> String:
    return get_lib_path_from_cfg(".driver_lib", "MAX Driver")
