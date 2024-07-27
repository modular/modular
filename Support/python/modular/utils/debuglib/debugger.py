# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Module used to acquire and interact with an LLDB python instance."""

import configparser
import importlib
import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from modular.utils.debuglib.lldbtypes import (
    SBDebugger,
    SBFrame,
    SBProcess,
    SBTarget,
    SBThread,
)

lldb: Any = None
_debugger: Optional[SBDebugger] = None


def _load_lldb() -> Any:
    """Loads the lldb python module and caches it."""
    if os.getenv("TEST_TMPDIR"):
        # Skip import side effect in tests
        return

    global lldb
    if lldb is None:
        # Fortunately the path to the module can be gotten from LLDB itself.
        lldb_lib = subprocess.check_output(
            ["mojo", "debug", "-Xlldb=-P"], text=True
        ).strip()
        if lldb_lib != "<COULD NOT FIND PATH>":
            sys.path.insert(0, lldb_lib)
            try:
                lldb = importlib.import_module("lldb")
            except ImportError:
                pass
    return lldb


# We need this to ensure the lldb module is loaded automatically for all
# consumers that want to easily access its `static` constructors.
_load_lldb()


def _setup_singleton_debugger() -> Optional[SBDebugger]:
    """Creates a working global debugger instance and caches it."""
    global _debugger
    if _debugger is None and lldb is not None:
        _debugger = lldb.SBDebugger.Create(False)  # source_init_files=False

        if _debugger is not None:
            # This sets up the debugger for running in the test environment.
            if init_file := os.environ.get("LLDB_TEST_INIT_FILE", None):
                _debugger.HandleCommand(f"command source {init_file}")

            # This is needed to ensure that we wait for the debugger to stop any
            # time we issue stepping commands.
            _debugger.SetAsync(False)

            cfg = configparser.ConfigParser()
            cfg.read(
                os.path.join(os.environ["MODULAR_DERIVED_PATH"], "modular.cfg")
            )
            plugin_path = cfg.get(
                section="mojo", option="lldb_plugin_path"
            ).strip(";")
            _debugger.HandleCommand(f"plugin load {plugin_path}")
    return _debugger


def get_debugger() -> SBDebugger:
    _setup_singleton_debugger()
    """Acquire a singleton instance of a debugger"""
    return _debugger  # type: ignore


def run_command_for_context(
    command: str, context: Any
) -> Tuple[bool, str, str]:
    """Execute the provided command using the provided context (thread, process
    or frame).
    print its output and error.

    Returns a tuple [success, output, error].

    Note: it's better to use this instead of debugger.HandleCommand() because
    it doesn't work nicely if multiple targets exist at once, which happens
    when multiple test files are executed simultaneously."""
    result = lldb.SBCommandReturnObject()
    exe_ctx = lldb.SBExecutionContext(context)
    get_debugger().GetCommandInterpreter().HandleCommand(
        command, exe_ctx, result
    )

    output = str(result.GetOutput())
    error = str(result.GetError())

    return (result.Succeeded(), output, error)


def dump_command_for_context(command: str, context: Any) -> bool:
    """Similar to run_command_for_context, but the output and error are printed
    right away.

    Returns True if and only if the command succeeded."""
    [success, output, error] = run_command_for_context(command, context)
    if len(output) > 0:
        print(output)
    if len(error) > 0:
        print(error)

    return success


@dataclass
class StopContext:
    """This dataclass is used to hold the context of the debugger at a given
    stop. It resembles the LLDB ExecutionContext class."""

    target: SBTarget
    process: SBProcess
    thread: SBThread
    frame: SBFrame

    def resume(self) -> Optional["StopContext"]:
        """Resume the process and return the StopContext once it stops, unless
        the process finished, in which case None is returned."""
        self.process.Continue()
        if self.process.GetState() == lldb.eStateStopped:
            thread = self.process.GetSelectedThread()
            return StopContext(
                self.target, self.process, thread, thread.GetFrameAtIndex(0)
            )
        return None

    def step_into(self) -> Optional["StopContext"]:
        """Step into the current thread and return the StopContext once it
        stops, unless the process finished, in which case None is returned."""
        self.thread.StepInto()
        if self.process.GetState() == lldb.eStateStopped:
            thread = self.process.GetSelectedThread()
            return StopContext(
                self.target, self.process, thread, thread.GetFrameAtIndex(0)
            )
        return None

    def step_over(self) -> Optional["StopContext"]:
        """Step over the current thread and return the StopContext once it
        stops, unless the process finished, in which case None is returned."""
        self.thread.StepOver()
        if self.process.GetState() == lldb.eStateStopped:
            thread = self.process.GetSelectedThread()
            return StopContext(
                self.target, self.process, thread, thread.GetFrameAtIndex(0)
            )
        return None

    def dump_command(self, command: str) -> bool:
        """Run the given command using the current frame as context and print
        its output and error."""
        return dump_command_for_context(command, self.frame)

    def run_command(self, command: str) -> Tuple[bool, str, str]:
        """Run the given command using the current frame as context and return
        a tuple [success, output, error]."""
        return run_command_for_context(command, self.frame)


def run_target(target: Any) -> StopContext:
    # We use this command because it nicely uses all the defaults from
    # the lldb init file, unlike debugger.Launch.
    with io.StringIO() as buf, redirect_stdout(buf):
        assert dump_command_for_context("run", target)

    process = target.GetProcess()
    assert process.IsValid()
    # This ensures the process didn't exit
    assert process.GetState() == lldb.eStateStopped

    thread = process.GetSelectedThread()
    return StopContext(target, process, thread, thread.GetFrameAtIndex(0))
