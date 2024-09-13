# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import abort
from .device import Device, _get_driver_path, _CDevice
from max._utils import call_dylib_func
from sys.ffi import DLHandle
from ._driver_library import DriverLibrary
from gpu.host import (
    DeviceContext,
    KernelProfilingInfo,
    DeviceFunction as CUDAFunction,
    Dim,
    FuncAttribute,
    CacheConfig,
)
from pathlib import Path
from collections import Optional, OptionalReg
from collections.dict import OwnedKwargsDict
from utils import Variant
from gpu.host._compile import _get_nvptx_target
from ._status import Status


fn cuda_device(gpu_id: Int = 0) raises -> Device:
    var lib = DriverLibrary()
    var status = Status(lib)
    var device = lib.create_cuda_device_fn(gpu_id, status.impl)
    if status:
        raise str(status)
    var cuda_dev = Device(
        lib,
        owned_ptr=device,
    )
    return cuda_dev


fn check_compute_capability(device: Device) raises:
    """Checks if a device is compatible with MAX. Will raise an exception if
    CUDA version is below 8.0."""
    var device_context = call_dylib_func[UnsafePointer[DeviceContext]](
        device._lib.value().get_handle(), "M_getDeviceContext", device._cdev
    )
    device_context[].is_compatible()


# TODO: Make this polymorphic on Device type.
@value
struct CompiledDeviceKernel[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    target: __mlir_type.`!kgen.target` = _get_nvptx_target(),
]:
    var _compiled_func: CUDAFunction[func, target=target]
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

        if "cuda" not in str(device):
            raise "launch() expects CUDA device."

        if "grid_dim" not in kwargs or "block_dim" not in kwargs:
            raise "launch() requires grid_dim and block_dim to be specified."

        var grid_dim = kwargs["grid_dim"]
        var block_dim = kwargs["block_dim"]
        var shared_mem_bytes = kwargs.find("shared_mem_bytes").or_else(0)

        var device_context = call_dylib_func[UnsafePointer[DeviceContext]](
            device._lib.value().get_handle(), "M_getDeviceContext", device._cdev
        )
        # need to call _enqueue function, not enqueue_function, otherwise the whole
        # pack is passed as a single argument
        device_context[]._enqueue_function(
            self._compiled_func,
            args,
            grid_dim=grid_dim[Dim],
            block_dim=block_dim[Dim],
            shared_mem_bytes=shared_mem_bytes[Int],
        )


alias CompileArg = Variant[Int, Path, Bool]


@value
struct CUDACompiledKernelArgs:
    var max_registers: Optional[Int]
    var threads_per_block: Optional[Int]

    @staticmethod
    fn _get_opt[
        T: CollectionElement
    ](kwargs: OwnedKwargsDict[CompileArg], key: String) raises -> Optional[T]:
        return kwargs.find(key).value()[T] if key in kwargs else Optional[T]()

    fn __init__(inout self, kwargs: OwnedKwargsDict[CompileArg]) raises:
        self.max_registers = Self._get_opt[Int](kwargs, "max_registers")
        self.threads_per_block = Self._get_opt[Int](kwargs, "threads_per_block")


fn compile[
    func_type: AnyTrivialRegType, //,
    func: func_type,
    # TODO: would like this to be an Optional but need to workaround MOCO-1039
    target_arch: StringLiteral = "sm_80",
](device: Device, **kwargs: CompileArg) raises -> CompiledDeviceKernel[
    func, target = _get_nvptx_target[target_arch]()
] as out:
    """Compiles a function which can be executed on device.

    Args:
        device: Device for which to compile the function. The returned CompiledDeviceKernel
            can execute on a different Device, as long as the device architecture matches.
        kwargs:
            verbose (Bool): Prints verbose log messages from cuModuleLoadEx during compilation/linking.
            max_registers (Int): Limits the max of registers that can be used by your kernel.
            threads_per_block (Int): Block size that will be used to launch the kernel. Can help
                the compiler decide how to tradeoff resources (e.g. registers).
    Returns:
        Kernel which can be launched on a Device.

    """
    if "cuda" not in str(device):
        raise "compile() expects CUDA device."

    var device_context = call_dylib_func[UnsafePointer[DeviceContext]](
        device._lib.value().get_handle(), "M_getDeviceContext", device._cdev
    )

    var compile_args = CUDACompiledKernelArgs(kwargs)
    var cuda_func = device_context[].compile_function[
        func,
        target = out.target,
        _is_failable=False,
    ](
        max_registers=OptionalReg(
            compile_args.max_registers.value()
        ) if compile_args.max_registers else OptionalReg[Int](None),
        threads_per_block=OptionalReg(
            compile_args.threads_per_block.value()
        ) if compile_args.threads_per_block else OptionalReg[Int](None),
    )
    return CompiledDeviceKernel(cuda_func)
