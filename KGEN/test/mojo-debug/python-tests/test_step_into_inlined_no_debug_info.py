# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase


class TestStepIntoInlinedNoDebugInfo(LLDBTestBase):
    def test_stepping_into_lined_no_debug_info(self):
        """Checks line info for inlined function with no debuginfo."""

        source = self.create_test_input_source(
            "step_into_inlined_no_debug_info.mojo"
        )
        with self.build_and_launch(source) as ctx:
            ctx = ctx.step_into()
            assert ctx is not None
            expected_line = next(
                source.find_lines_with_text("# expected after step-into")
            )
            assert ctx.frame.GetLineEntry().GetLine() == expected_line
