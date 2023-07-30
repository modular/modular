# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from modular.utils import logging
from modular.utils.typing import Iterator, Optional


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
    if destination_dir.exists():
        if destination_dir.is_symlink():
            destination_dir.unlink()
        else:
            shutil.rmtree(destination_dir, ignore_errors=True)
    try:
        destination_dir.symlink_to(src_dir, target_is_directory=True)
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
                f'Failed to link directory "{destination_dir}" to "{src_dir}"'
                f" due to permission issues (code {e.args[0]})."
                "Copying the directories instead."
            )
            shutil.copytree(src_dir, destination_dir)
        else:
            raise
