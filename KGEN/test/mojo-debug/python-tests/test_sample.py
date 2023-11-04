# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.LLDBTestBase import LLDBTestBase, SourceFile
from lib.debugger import lldb


class TestSample(LLDBTestBase):
    def test_sample(self):
        """Sample test that ensures that the basic attributes of a primitive
        variables can be read after parsing the corresponding DWARF. This test
        serves as an example on how to interact with the SB API for querying
        variable and type information."""

        with self.build_and_launch("sample.mojo") as ctx:
            var = ctx.frame.FindVariable("an_int")
            assert var.GetValue() == "-420"
            assert var.GetTypeName() == "si64"
            assert var.GetDisplayTypeName() == "si64"
            assert var.GetType().GetTypeFlags() | lldb.eTypeIsInteger
            assert var.GetValueAsSigned(-420)

            ctx = ctx.resume()
            assert ctx
            var = ctx.frame.FindVariable("another_int")
            assert var.GetValue() == "420"
