# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.debugger import lldb
from lib.LLDBTestBase import LLDBTestBase


class TestSample(LLDBTestBase):
    def test_assignment(self):
        """Make sure struct field offset calculation is correct by accessing
        struct members."""

        with self.build_and_launch("struct_field_offset.mojo") as ctx:
            struct = ctx.frame.FindVariable("p")
            assert struct.GetNumChildren() == 2
            first = struct.GetChildAtIndex(0)
            assert first.GetChildAtIndex(0).GetValueAsSigned(-1) == 42
            second = struct.GetChildAtIndex(1)
            assert second.GetChildAtIndex(0).GetValueAsSigned(-1) == 3735928559
