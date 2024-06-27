# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

from modular.utils.typing import Optional
from modular.utils.pythonpath import (
    get_lib_path,
    get_libtorch_python_path,
    get_libtorchvision_path,
    get_libpython,
)


def check_filepath(path: Optional[Path]):
    if path:
        assert path.exists() and path.is_file()


def test_get_lib_path():
    try:
        import torch
    except ImportError:
        return

    torch_path = Path(os.path.dirname(torch.__file__)) / "lib"
    c10_path = get_lib_path(torch_path, "libc10")
    check_filepath(c10_path)


def test_get_libtorch_python_path():
    torch_python_path = get_libtorch_python_path()
    check_filepath(torch_python_path)


def test_get_libtorchvision_path():
    torchvision_path = get_libtorchvision_path()
    check_filepath(torchvision_path)


def test_get_libpython():
    libpython_path = get_libpython()
    check_filepath(libpython_path)
