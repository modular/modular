# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import platform
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from modular.utils import logging
from modular.utils.subprocess import get_command_output


def has_gpu():
    """Check if the system has an NVidia GPU we can target.

    Returns:
        Boolean indicating if the system has an NVidia GPU that we can target.
    """
    if platform.system() != "Linux":
        return False

    try:
        return get_command_output(["cuda-query"])
    except Exception:
        return False


@contextmanager
def set_env_var(env_var: str, tmp_val: Optional[str]) -> Iterator[None]:
    """Temporarily set an environment variable to a different value.

    This is modeled as a context manager that yields nothing.

    Args:
        env_var: name of the environment variable as a string.
        tmp_val: temporary value for the variable in the context. If None, the
            variable will be temporarily removed.
    """

    # Store the old value, and remove the environment variable if it exists.
    old_val = os.environ.pop(env_var, None)

    # If a value is given, we set the variable to that value
    if tmp_val is not None:
        os.environ[env_var] = tmp_val

    try:
        yield
    finally:
        if old_val is None:
            # Delete if it didn't exist before and exists now.
            os.environ.pop(env_var, None)
        else:
            # We reset to the original value.
            os.environ[env_var] = old_val


def get_ordinal(n: int) -> str:
    """Get the string ordinal for an integer.

    Args:
        n: the integer.

    Returns:
        The ordinal as a string, for example '1st'.
    """
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return f"{n}{suffix}"


def create_symlink(
    destination: Path, src: Path, target_is_directory: bool = False
):
    """Links the destination to the src.

    Links the destination to the src. If the destination already exists, then we
    remove or unlink it. If one cannot link due to permission issues, then the
    src is copied.

    Args:
        destination (Path): the symlink to be created
        src (Path): the source ("true") to link to
    """
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            destination.unlink()
        else:
            shutil.rmtree(destination, ignore_errors=True)
    try:
        destination.symlink_to(src, target_is_directory)
    except OSError as e:
        if e.args[0] in (22, 1314):
            # 22 is the error code on Windows for required privileges exception
            # is not held by a client.
            # 1314 is the error code on Windows for failing to symlink due to
            # missing admin privileges.
            # If the link fails because of either of these errors, try copying
            # the build directory instead. For more, see:
            # https://docs.python.org/3/library/os.html#os.symlink
            logging.warning(
                f'Failed to link "{destination}" to "{src}"'
                f" due to permission issues (code {e.args[0]})."
                "Copying instead."
            )
            if target_is_directory:
                shutil.copytree(src, destination)
            else:
                shutil.copy(src, destination)
        else:
            raise


def create_dir_symlink(destination_dir: Path, src_dir: Path):
    """Links the destination to the src directory.

    Links the destination directory to the src (true) directory. If the
    destination directory already exists, then we remove or unlink it. If one
    cannot link the directory due to permission issues, then the directory is
    copied.

    Args:
        destination_dir (Path): the symlink directory to be created
        src_dir (Path): the source ("true") directory to link to
    """
    create_symlink(destination_dir, src_dir, target_is_directory=True)


def modular_dtype_to_np_dtype(dtype: str) -> np.dtype:
    """Converts a string representing a Modular dtype to its NumPy dtype.

    Args:
        dtype: the string representing the Modular dtype.

    Returns:
        The corresponding NumPy dtype.
    """
    np_dtype = {
        "bool": np.bool_,
        "si8": np.int8,
        "int8": np.int8,
        "si16": np.int16,
        "int16": np.int16,
        "si32": np.int32,
        "int32": np.int32,
        "si64": np.int64,
        "int64": np.int64,
        "ui8": np.uint8,
        "uint8": np.uint8,
        "ui16": np.uint16,
        "uint16": np.uint16,
        "ui32": np.uint32,
        "uint32": np.uint32,
        "ui64": np.uint64,
        "uint64": np.uint64,
        "f16": np.float16,
        "float16": np.float16,
        "f32": np.float32,
        "float32": np.float32,
        "f64": np.float64,
        "float64": np.float64,
    }.setdefault(dtype, None)

    if np_dtype is None:
        raise RuntimeError(f"unrecognized dtype {dtype}")

    return np_dtype
