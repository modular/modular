# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase


class TestFunctionBeforeStructParsing(LLDBTestBase):
    def test_function_before_struct_parsing(self):
        """Tests that DWARF parsing is done correctly when LLDB parses a function
        source name before its owning struct.
        This happens when the debug session starts with a single breakpoint
        within a struct method."""

        with self.build_and_launch("point.mojo") as ctx:
            var = ctx.frame.FindVariable("x")
            assert var.GetValueAsSigned(0) == 1

            ctx = ctx.step_over()
            assert ctx is not None

            p1 = ctx.frame.FindVariable("p1")
            assert p1.GetTypeName() == "!lit.declref<@point::@Point>"
