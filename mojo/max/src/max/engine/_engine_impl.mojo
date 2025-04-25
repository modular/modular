# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path
from sys import external_call
from sys.ffi import _OwnedDLHandle, DLHandle, _find_dylib

from max._utils import CString, call_dylib_func, exchange, get_lib_path_from_cfg
from memory import UnsafePointer


fn _get_engine_path() raises -> String:
    return get_lib_path_from_cfg(".engine_lib", "AI engine lib")


struct _EngineImpl:
    """Represents an instance of Modular AI Engine."""

    var owned_lib: _OwnedDLHandle
    # FIXME, lib should not be accessed directly.
    var lib: DLHandle
    """Handle to Modular AI Engine library."""

    alias VersionFnName = "M_version"

    @implicit
    fn __init__(out self, path: String):
        self.owned_lib = _find_dylib["Modular AI Engine"](path)
        self.lib = self.owned_lib.handle()

    fn __moveinit__(out self, owned existing: Self):
        self.owned_lib = existing.owned_lib^
        self.lib = existing.lib^

    fn get_version(self) -> String:
        """Returns version of modular AI engine.

        Returns:
            Version as string.
        """
        var version = call_dylib_func[CString](self.lib, Self.VersionFnName)
        return version.__str__()

    fn __enter__(owned self) -> Self:
        return self^
