# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

__doc__ = """
Utility module for dealing with subprocesses

This module implements tools to uniformly dispatch and log subprocesses from
Python, and to reduce import clutter by mirroring some commonly used
functionality of the built-in subprocess module. Users are encouraged to use
this library in order to take advantage of unified logging and alternative
default arguments.
"""

from subprocess import CalledProcessError, CompletedProcess, list2cmdline

from ._shell_commands import ShellCommand, get_command_output, run_shell_command

# Remove from the namespace so that it's not visible to users.
del _shell_commands  # type: ignore
