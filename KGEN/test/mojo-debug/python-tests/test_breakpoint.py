# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase


class TestBreakpoint(LLDBTestBase):
    def test_breakpoint(self):
        """
        Test the breakpoint() intrinsic.
        """

        with self.build_and_launch("breakpoint.mojo") as ctx:
            sum = ctx.frame.FindVariable("sum")
            assert sum.GetValueAsUnsigned() == 36
