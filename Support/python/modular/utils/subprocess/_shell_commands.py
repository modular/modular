# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import itertools
import subprocess
from pathlib import Path

from modular.utils import logging
from modular.utils.typing import Any, Iterable, Iterator, TypeVar, Union

ShellCommand = Iterable[Union[str, Path]]


def run_shell_command(
    cmd: ShellCommand, *, shell: bool = True, check: bool = True, **kwargs: Any
) -> subprocess.CompletedProcess:
    """Runs a shell command using the arguments provided.

    This is essentially a wrapper around subprocess.run, with more reasonable
    default arguments, and some debug logging.

    Args:
        cmd: shell command to run.
        shell: see subprocess.run for semantics.
        check: see subprocess.run for semantics.
        **kwargs: see subprocess.run for semantics
            (https://docs.python.org/3/library/subprocess.html#subprocess.run).

    Returns:
        A subprocess.CompletedProcess object.
    """
    cmdline = subprocess.list2cmdline(cmd)
    logging.debug(f"Running command: {cmdline}")
    kwargs.update({"shell": shell, "check": check})
    return subprocess.run(cmdline if shell else list(cmd), **kwargs)


def run_chained_commands(
    commands: Iterable[ShellCommand], **kwargs: Any
) -> subprocess.CompletedProcess:
    """Runs a sequence of shell commands chained into a single command in order.

    The function fails immediately if any of the commands fail.

    Args:
        commands: shell commands to run.
        **kwargs: see run_shell_command for semantics.
    """

    # TODO: this should be exposed as a separate utility
    _T, _S = TypeVar("_T"), TypeVar("_S")

    def interleave(it: Iterable[_T], separator: _S) -> Iterator[Union[_T, _S]]:
        """
        Inserts the seperator between each element of the given iterable.
        """
        for idx, cmd in enumerate(it):
            if idx:
                yield separator
            yield cmd

    return run_shell_command(
        itertools.chain.from_iterable(interleave(commands, ("&&",))), **kwargs
    )


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
