# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import subprocess
from pathlib import Path
from typing import Any, Iterable, Union

from modular.utils import logging

ShellCommand = Iterable[Union[str, Path]]


def run_shell_command(
    cmd: ShellCommand, *, check: bool = True, **kwargs: Any
) -> subprocess.CompletedProcess:
    """Runs a shell command using the arguments provided.

    This is essentially a wrapper around subprocess.run, with more reasonable
    default arguments, and some debug logging.

    Args:
        cmd: shell command to run.
        check: see subprocess.run for semantics.
        **kwargs: see subprocess.run for semantics
            (https://docs.python.org/3/library/subprocess.html#subprocess.run).

    Returns:
        A subprocess.CompletedProcess object.
    """
    if "shell" in kwargs:
        raise ValueError("shell support has been removed")
    cmdline = subprocess.list2cmdline(cmd)
    logging.debug(f"Running command: {cmdline}")
    kwargs.update({"check": check})
    return subprocess.run(list(map(str, cmd)), **kwargs)


def get_command_output(cmd: ShellCommand, **kwargs: Any) -> str:
    """A wrapper over run_shell_command that captures stdout into a string.

    Args:
        cmd: shell command to run.
        **kwargs: see run_shell_command for semantics. Passing capture_output is
            not allowed.

    Returns:
        Captured stdout of the command as a string.

    Raises:
        ValueError: if the capture_output keyword argument is specified.
    """
    if "capture_output" in kwargs:
        raise ValueError(
            "Cannot pass capture_output when using get_command_output"
        )
    proc = run_shell_command(cmd, capture_output=True, **kwargs)
    return proc.stdout.decode("utf-8").rstrip()
