# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import getenv
from sys import external_call
from sys.ffi import DLHandle
from memory.unsafe import DTypePointer
from ._status import Status
from ._utils import call_dylib_func, exchange, CString
from tensor import TensorSpec
from pathlib import Path


fn _get_engine_path() raises -> String:
    var engine_lib_path_str_ptr = external_call[
        "KGEN_CompilerRT_getMAXConfigValue", DTypePointer[DType.int8]
    ](".engine_lib")

    if not engine_lib_path_str_ptr:
        raise "cannot get the location of AI engine library from modular.cfg"

    # this transfers ownership of the underlying data buffer allocated in
    # `KGEN_CompilerRT_getMAXConfigValue` so that it can be destroyed by Mojo.
    var engine_lib_path = String._from_bytes(engine_lib_path_str_ptr)

    if not Path(engine_lib_path).exists():
        raise "AI engine library not found at " + engine_lib_path
    return engine_lib_path


struct _EngineImpl:
    """Represents an instance of Modular AI Engine."""

    """Handle to Modular AI Engine library."""
    var lib: DLHandle
    var can_close_lib: Bool

    alias VersionFnName = "M_version"

    fn __init__(inout self, path: String):
        self.lib = DLHandle(path)
        self.can_close_lib = True

    fn __moveinit__(inout self, owned existing: Self):
        self.lib = existing.lib
        self.can_close_lib = exchange[Bool](existing.can_close_lib, False)

    fn get_version(self) -> String:
        """Returns version of modular AI engine.

        Returns:
            Version as string.
        """
        var version = call_dylib_func[CString](self.lib, Self.VersionFnName)
        return version.__str__()

    fn __enter__(owned self) -> Self:
        return self^

    fn __del__(owned self):
        if self.can_close_lib:
            self.lib.close()
