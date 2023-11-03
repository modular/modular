# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.LLDBTestBase import LLDBTestBase, SourceFile


class TestIssueStepIntoInlinedNoDebugInfo(LLDBTestBase):
    def test(self):
        """FIXME(24731): There's an issue caused by inlined functions emitting
        line table entries pointing to the beginning of the caller function or
        to the file but without line information.

        Look how between lines 12 and 13, which are sequential instructions
        corresponding to the call to `abs`, which is an inlined function with no
        debug info. There are some instructions at line 8 and others without
        line:

        0x00000001000037e4: KGEN/test/mojo-lldb-dwarf/python-tests/Inputs/step_into_inlined.mojo:12:16
        0x0000000100003818: KGEN/test/mojo-lldb-dwarf/python-tests/Inputs/step_into_inlined.mojo
        0x000000010000381c: KGEN/test/mojo-lldb-dwarf/python-tests/Inputs/step_into_inlined.mojo:8:1
        0x0000000100003820: KGEN/test/mojo-lldb-dwarf/python-tests/Inputs/step_into_inlined.mojo:13:10
        """

        source = SourceFile("step_into_inlined_no_debug_info.mojo")
        with self.build_and_launch(source) as ctx:
            ctx = ctx.step_into()
            main_decl_line = next(source.find_lines_with_text("fn main():"))
            # This is wrong, stepping into should have taken us to the line
            # after the breakpoint.
            assert ctx.frame.GetLineEntry().GetLine() == main_decl_line
