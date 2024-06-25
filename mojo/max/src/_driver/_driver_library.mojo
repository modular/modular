# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys.ffi import DLHandle
from memory.arc import Arc


struct ManagedDLHandle:
    var lib: DLHandle

    fn __init__(inout self, lib: DLHandle):
        self.lib = lib

    fn __moveinit__(inout self, owned existing: Self):
        self.lib = existing.lib

    fn get_handle(self) -> DLHandle:
        return self.lib

    fn __del__(owned self):
        self.lib.close()


@value
struct DriverLibrary:
    var lib: Arc[ManagedDLHandle]

    fn __init__(inout self, owned handle: ManagedDLHandle):
        self.lib = Arc(handle^)

    fn get_handle(self) -> DLHandle:
        return self.lib[].get_handle()
