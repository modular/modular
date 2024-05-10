# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional
from sys.ffi import DLHandle
from max.engine._utils import call_dylib_func
from max.engine._utils import get_lib_path_from_cfg
from pathlib import Path


struct CPUDescriptor:
    var numa_id: Int

    fn __init__(inout self, *, numa_id: Optional[Int] = None):
        self.numa_id = numa_id.value()[] if numa_id else -1


fn _get_driver_path() raises -> String:
    return get_lib_path_from_cfg(".driver_lib", "MAX Driver")


struct Device:
    var lib: DLHandle
    var _ptr: DTypePointer[DType.invalid]

    fn __init__(inout self, descriptor: CPUDescriptor = CPUDescriptor()) raises:
        alias func_name_create = "M_createCPUDevice"
        self.lib = DLHandle(_get_driver_path())
        self._ptr = call_dylib_func[DTypePointer[DType.invalid]](
            self.lib, func_name_create, descriptor.numa_id
        )

    fn info(self) -> Int:
        alias func_name_info = "M_getInfo"
        return call_dylib_func[Int](self.lib, func_name_info, self._ptr)

    fn __del__(owned self):
        alias func_name_destroy = "M_destroyCPUDevice"
        self.lib.close()
        call_dylib_func[NoneType](self.lib, func_name_destroy, self._ptr)
