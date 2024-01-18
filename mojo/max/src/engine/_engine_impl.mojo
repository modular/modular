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
from ._utils import *
from tensor import TensorSpec
from pathlib import Path


fn _get_engine_path() raises -> String:
    let engine_lib_path_str_ptr = external_call[
        "KGEN_CompilerRT_getConfigValue", DTypePointer[DType.int8]
    ]("max.engine_lib")

    # this transfers ownership of the underlying data buffer allocated in
    # `KGEN_CompilerRT_getConfigValue` so that it can be destroyed by Mojo.
    let pathlen = len(StringRef(engine_lib_path_str_ptr))
    let engine_lib_path = String(
        engine_lib_path_str_ptr, pathlen + 1
    )  # account for the terminator

    if not engine_lib_path:
        raise "cannot get the location of AI engine library from modular.cfg"

    if not Path(engine_lib_path).exists():
        raise "AI engine library not found at " + engine_lib_path
    return engine_lib_path


struct _EngineImpl:
    """Represents an instance of Modular AI Engine."""

    """Handle to Modular AI Engine library."""
    var lib: DLHandle
    var can_close_lib: Bool

    alias VersionFnName = "M_version"

    fn __init__(inout self, path: StringRef):
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
        let version = call_dylib_func[CString](self.lib, Self.VersionFnName)
        return version.__str__()

    fn __enter__(owned self) -> Self:
        return self ^

    fn __del__(owned self):
        if self.can_close_lib:
            self.lib._del_old()
