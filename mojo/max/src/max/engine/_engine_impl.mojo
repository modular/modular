# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import external_call
from sys.ffi import DLHandle
from memory import UnsafePointer
from max._utils import call_dylib_func, exchange, CString, get_lib_path_from_cfg
from pathlib import Path


fn _get_engine_path() raises -> String:
    return get_lib_path_from_cfg(".engine_lib", "AI engine lib")


struct _EngineImpl:
    """Represents an instance of Modular AI Engine."""

    """Handle to Modular AI Engine library."""
    var lib: DLHandle
    var can_close_lib: Bool

    alias VersionFnName = "M_version"

    fn __init__(out self, path: String):
        self.lib = DLHandle(path)
        self.can_close_lib = True

    fn __moveinit__(out self, owned existing: Self):
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
