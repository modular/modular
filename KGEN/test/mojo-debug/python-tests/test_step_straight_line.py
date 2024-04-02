# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase


class TestStepStraightLine(LLDBTestBase):
    def test_step_straight_line(self):
        """Checks stepping straight line code."""

        source = self.create_test_input_source("step_straight_line.mojo")
        function_header_line = next(source.find_lines_with_text("fn main()"))
        with self.build_and_launch(source) as ctx:
            line = ctx.frame.GetLineEntry().GetLine()
            prev_line = line
            while line != function_header_line:
                # TODO(#35853) step straight line odd behavior with always inline functions.
                # assert line >= prev_line

                ctx = ctx.step_over()
                assert ctx is not None
                prev_line = line
                line = ctx.frame.GetLineEntry().GetLine()
