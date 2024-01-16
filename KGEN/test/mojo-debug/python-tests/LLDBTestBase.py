# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import contextlib
import unittest
from pathlib import Path
from typing import Any, Generator, Union

import pytest

from modular.utils.debuglib.debugger import (
    StopContext,
    get_debugger,
    lldb,
    run_target,
)
from modular.utils.debuglib.lldbtypes import SBTarget
from modular.utils.debuglib.sources import MojoBinary, MojoSource


class LLDBTestBase(unittest.TestCase):
    """Base class for all tests that interact with LLDB using the SB API."""

    def setup_class(self):
        if lldb is None:
            pytest.skip("The scripting bridge for LLDB is not available.")
        pass

    def create_test_input_source(self, file_name: str) -> MojoSource:
        """Create a MojoSource that lives in the /Inputs directory."""
        return MojoSource(Path(__file__).parent / "Inputs" / file_name)

    def set_breakpoints_from_comments(self, source: MojoSource, target: Any):
        """Traverses the input file looking for the `# breakpoint` comment, and
        places a breakpoint at the lines where it appears"""
        for line in source.find_lines_with_text("# breakpoint"):
            bp = target.BreakpointCreateByLocation(str(source.path), line)
            assert (
                bp.GetNumLocations() == 1
            ), f"Couldn't set a breakpoint at {str(source.path)}:{line}"

    @contextlib.contextmanager
    def build_and_launch(
        self, source_or_file_name: Union[MojoSource, str]
    ) -> Generator[StopContext, None, None]:
        """Builds the given source file, then creates a target with the
        resultant binary, places breakpoints on all the locations with the
        `# breakpoint` comment, and yields at the first stop."""

        source = self.create_test_input_source(
            source_or_file_name
        ) if isinstance(source_or_file_name, str) else source_or_file_name

        # TODO(28608): support a test mode for JIT debugging besides AOT.
        with MojoBinary(source) as bin_file:
            target: SBTarget = get_debugger().CreateTarget(str(bin_file))
            assert target.IsValid()

            self.set_breakpoints_from_comments(source, target)

            yield run_target(target)
