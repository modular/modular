# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Provides information about MAX Engine, such as the version.
"""
from ._engine_impl import _get_engine_path, _EngineImpl


fn get_version() raises -> String:
    """Returns the current MAX Engine version.

    Returns:
        Version as string.
    """
    var path = _get_engine_path()
    var version = _EngineImpl(path._strref_dangerous()).get_version()
    path._strref_keepalive()
    return version
