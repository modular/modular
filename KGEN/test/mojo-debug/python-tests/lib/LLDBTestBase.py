# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import configparser
import contextlib
import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional, Union

_lldb: Any = None


def load_lldb() -> Any:
    """Loads the lldb python module and cache it."""
    global _lldb
    if _lldb is None:
        # Fortunately the path to the module can be gotten from LLDB itself.
        lldb_lib = subprocess.check_output(
            ["mojo", "debug", "-P"], text=True
        ).strip()
        sys.path.insert(0, lldb_lib)
        _lldb = importlib.import_module("lldb")
    return _lldb


lldb = load_lldb()


def handle_command_for_context(
    command: str, debugger: Any, context: Any
) -> bool:
    """Execute the provided command using the provided context (thread, process
    or frame) and print its output and error.

    Return `True` if the command succeeded.

    Note: it's better to use this instead of debugger.HandleCommand() because
    it doesn't work nicely if multiple targets exist at once, which happens
    when multiple test files are executed simultaneously."""

    result = lldb.SBCommandReturnObject()
    exe_ctx = lldb.SBExecutionContext(context)
    debugger.GetCommandInterpreter().HandleCommand(command, exe_ctx, result)

    output = str(result.GetOutput())
    error = str(result.GetError())

    if len(output) > 0:
        print(output)
    if len(error) > 0:
        print(error)

    return result.Succeeded()


@dataclass
class StopContext:
    """This dataclass is used to hold the context of the debugger at a given
    stop. It resembles the LLDB ExecutionContext class."""

    target: Any
    process: Any
    thread: Any
    frame: Any

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

    def handle_command(self, command: str) -> bool:
        """Handle the given command using the current frame as context"""
        return handle_command_for_context(
            command, self.target.GetDebugger(), self.frame
        )


class SourceFile:
    """Class that represents a source file in the /Inputs directory."""

    def __init__(self, file_name: str):
        self.path = Path(__file__).parent.parent / "Inputs" / file_name

    def find_lines_with_text(self, text: str) -> Generator[int, None, None]:
        """Generate the 1-indexed line numbers at which the given text is found
        in the source file."""
        with open(self.path, "r") as source:
            i = 1
            for line in source.readlines():
                if text in line:
                    yield i
                i += 1


class LLDBTestBase:
    """Base class for all tests that interact with LLDB using the SB API."""

    def setup_class(self):
        """Ensures that the lldb module is loaded and that a debugger is created
        with Mojo support."""
        load_lldb()

        self.debugger = lldb.SBDebugger.Create(False)  # no load lldb init files

        # This sets up the debugger for running in the test environment.
        self.debugger.HandleCommand(
            "command source " + os.environ["LLDB_TEST_INIT_FILE"]
        )

        # This is needed to ensure that we wait for the debugger to stop any
        # time we issue stepping commands.
        self.debugger.SetAsync(False)

        cfg = configparser.ConfigParser()
        cfg.read(
            os.path.join(os.environ["MODULAR_DERIVED_PATH"], "modular.cfg")
        )
        plugin_path = cfg.get(section="mojo", option="lldb_plugin_path").strip(
            ";"
        )
        self.debugger.HandleCommand(f"plugin load {plugin_path}")

    def build(self, source: SourceFile, output_path: Path) -> None:
        subprocess.run(
            [
                "mojo",
                "build",
                "--debug-level",
                "full",
                "-O0",
                "--debug-info-language",
                "Mojo",
                "--no-alnum-symbols",
                source.path,
                "-o",
                output_path,
            ],
            check=True,
        )

    def set_breakpoints_from_comments(self, source: SourceFile, target: Any):
        """Traverses the input file looking for the `# breakpoint` comment, and
        places a breakpoint at the lines where it appears"""
        for line in source.find_lines_with_text("# breakpoint"):
            bp = target.BreakpointCreateByLocation(str(source.path), line)
            assert (
                bp.GetNumLocations() == 1
            ), f"Couldn't set a breakpoint at {str(source.path)}:{line}"

    @contextlib.contextmanager
    def build_and_launch(
        self, source_or_file_name: Union[SourceFile, str]
    ) -> Generator[Any, None, None]:
        """Builds the given source file, then creates a target with the
        resultant binary, places breakpoints on all the locations with the
        `# breakpoint` comment, and yields at the first stop."""

        source = SourceFile(source_or_file_name) if isinstance(
            source_or_file_name, str
        ) else source_or_file_name

        # We build the input as a precompiled binary in a temporary folder
        # instead of jitting because of an issue with the debug info generation
        # on mac (24462).
        with tempfile.TemporaryDirectory() as out_dir:
            bin_file = Path(out_dir) / (source.path.name + ".exe")
            self.build(source, bin_file)

            target = self.debugger.CreateTarget(str(bin_file))
            assert target.IsValid()

            self.set_breakpoints_from_comments(source, target)

            # We use this command because it nicely uses all the defaults from
            # the lldb init file, unlike debugger.Launch.
            assert handle_command_for_context("run", self.debugger, target)

            process = target.GetProcess()
            assert process.IsValid()
            # This ensures the process didn't exit
            assert process.GetState() == lldb.eStateStopped

            thread = process.GetSelectedThread()
            yield StopContext(
                target, process, thread, thread.GetFrameAtIndex(0)
            )
