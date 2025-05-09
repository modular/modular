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

from collections import Optional, OptionalReg
from os import abort
from pathlib import Path

from gpu.host import DeviceContext
from gpu.host import DeviceFunction as AcceleratorFunction
from gpu.host import Dim, FuncAttribute
from gpu.host._compile import _get_gpu_target
from max._utils import call_dylib_func
from memory import UnsafePointer
from runtime.asyncrt import DeviceContextPtr

from utils import Variant

from ._driver_library import DriverLibrary
from ._status import Status
from .device import Device, _CDevice, _get_driver_path


@deprecated("use gpu.host.DeviceContext.number_of_devices() instead")
fn accelerator_count() raises -> Int:
    var lib = DriverLibrary()
    return lib.accelerator_count_fn()


@deprecated("use gpu.host.DeviceContext() instead")
fn accelerator(gpu_id: Int = 0) raises -> Device:
    var lib = DriverLibrary()
    var status = Status(lib)
    var device = lib.create_accelerator_device_fn(gpu_id, status.impl)
    if status:
        raise String(status)
    var accelerator_dev = Device(
        lib,
        owned_ptr=device,
    )
    return accelerator_dev


# TODO: Make this polymorphic on Device type.
@value
struct CompiledDeviceKernel[func_type: AnyTrivialRegType, //, func: func_type]:
    var _compiled_func: AcceleratorFunction[
        func,
        Optional[__mlir_type[`!kgen.variadic<`, AnyType, `>`]](None),
        target = _get_gpu_target(),
    ]
    alias LaunchArg = Variant[Dim, Int]

    @parameter
    fn __call__[
        *Ts: AnyType
    ](self, device: Device, *args: *Ts, **kwargs: Self.LaunchArg,) raises:
        """Launch a compiled kernel on `device`.

        Note: launch is async which means that you must keep `args` and `device`
        alive manually until execution of the DeviceFunction finishes.

        Args:
            device: The Device on which to launch the kernel.
            args: Arguments which will be passed to the kernel on the device.
                **These arguments must all be `register_passable` types**.
            kwargs:
                grid_dim (Dim): Dimensions of grid the kernel is launched on.
                block_dim (Dim): Dimensions of block the kernel is launched on.
                shared_mem_bytes (Int): Dynamic shared memory size available to kernel.
        """

        if "gpu" not in String(device):
            raise "launch() expects GPU device."

        if "grid_dim" not in kwargs or "block_dim" not in kwargs:
            raise "launch() requires grid_dim and block_dim to be specified."

        var grid_dim = kwargs["grid_dim"]
        var block_dim = kwargs["block_dim"]
        var shared_mem_bytes = kwargs.find("shared_mem_bytes").or_else(0)

        var device_context = call_dylib_func[DeviceContextPtr](
            device._lib.value().get_handle(), "M_getDeviceContext", device._cdev
        )
        # need to call _enqueue function, not enqueue_function, otherwise the whole
        # pack is passed as a single argument
        device_context[]._enqueue_function_unchecked(
            self._compiled_func,
            args,
            grid_dim=grid_dim[Dim],
            block_dim=block_dim[Dim],
            shared_mem_bytes=shared_mem_bytes[Int],
        )


@deprecated("use gpu.host.DeviceContext.enqueue_function() instead")
struct Accelerator:
    @staticmethod
    fn check_compute_capability(device: Device) -> Bool:
        """Checks if a device is compatible with MAX.

        Returns:
            True if the device is compatible with MAX, False otherwise.
        """
        var device_context = call_dylib_func[DeviceContextPtr](
            device._lib.value().get_handle(), "M_getDeviceContext", device._cdev
        )
        return device_context[].is_compatible()

    @staticmethod
    fn compile[
        func_type: AnyTrivialRegType, //, func: func_type
    ](device: Device) raises -> CompiledDeviceKernel[func]:
        """Compiles a function which can be executed on device.

        Args:
            device: Device for which to compile the function. The returned CompiledDeviceKernel
                can execute on a different Device, as long as the device architecture matches.
        Returns:
            Kernel which can be launched on a Device.

        """
        if "gpu" not in String(device):
            raise "compile() expects GPU device."

        var device_context = call_dylib_func[DeviceContextPtr](
            device._lib.value().get_handle(), "M_getDeviceContext", device._cdev
        )

        var accelerator_func = device_context[].compile_function[
            func, _target = _get_gpu_target()
        ]()
        return CompiledDeviceKernel(accelerator_func)
