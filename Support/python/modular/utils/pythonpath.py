# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Utilities to fetch paths of python libraries.
#
# ===----------------------------------------------------------------------=== #
from importlib import machinery
import sys
import os
from pathlib import Path
from sysconfig import get_config_var as var
from itertools import product
from find_libpython import find_libpython
from typing import Optional


def get_lib_path(base_path: Path, name: str) -> Optional[Path]:
    """
    Get the path of `name`.
    Returns:
        Path to the `name` library.
    """
    loader_details = (
        machinery.ExtensionFileLoader,
        machinery.EXTENSION_SUFFIXES + [".dylib"],
    )

    extfinder = machinery.FileFinder(str(base_path), loader_details)
    ext_specs = extfinder.find_spec(name)
    return None if ext_specs is None else Path(str(ext_specs.origin))


def get_libtorch_python_path() -> Optional[Path]:
    """
    Get the path of libtorch_python.
    Returns:
        Path to the libtorch_python library.
    """
    try:
        import torch
    except ImportError:
        return None

    torch_path = Path(os.path.dirname(torch.__file__)) / "lib"
    return get_lib_path(torch_path, "libtorch_python")


def get_libtorchvision_path() -> Optional[Path]:
    """
    Get the path of libtorchvision.
    Returns:
        Path to the torchvision library.
    """
    try:
        import torchvision
    except ImportError:
        return None

    torchvision_path = Path(os.path.dirname(torchvision.__file__))
    return get_lib_path(torchvision_path, "_C")


def get_libpython() -> Optional[Path]:
    """
    Get the path of libpython shared library.

    The function tries to locate libpython in several possible locations and
    returns `None` if it couldn't find libpython in any of them.

    Returns:
        Path to the libpython shared library or None if it wasn't found.
    """
    is_windows = os.name == "nt"
    if is_windows:
        ext = "dll"
    elif os.name == "posix":
        ext = "dylib" if sys.platform == "darwin" else "so"

    pyver = var("py_version_short")
    folders = [Path(var("LIBPL")), Path(var("LIBDIR"))]
    binaries = [var("LDLIBRARY"), f"libpython{pyver}.{ext}"]

    # Make a list of potential locations of libtorch shared lib - we'll start
    # with what `find_libpython` returns and then add some more possible
    # locations. We then go over these locations until we find an existing file
    # in one of them.
    paths = [Path(str(find_libpython()))]
    for folder, binary in product(folders, binaries):
        paths.append(folder / binary)
    for path in paths:
        if path.exists() and path.is_file() and path.suffix == f".{ext}":
            return path
    return None
