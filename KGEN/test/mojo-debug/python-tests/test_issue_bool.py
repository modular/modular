# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase


class TestBool(LLDBTestBase):
    def test_bool(self):
        """
        There's no SB API for getting the bit size of a type, but the byte size
        should be 1.
        """

        with self.build_and_launch("bool.mojo") as ctx:
            true = ctx.frame.FindVariable("true")
            assert true.GetTypeName() == "!pop.scalar<bool>"
            assert true.GetByteSize() == 1
            assert true.GetChildAtIndex(0).GetValueAsUnsigned(2) == 1
            assert true.GetSummary() == "True"

            false = ctx.frame.FindVariable("false")
            assert false.GetTypeName() == "!pop.scalar<bool>"
            assert false.GetByteSize() == 1
            assert false.GetChildAtIndex(0).GetValueAsUnsigned(2) == 0
            assert false.GetSummary() == "False"

            other = ctx.frame.FindVariable("other")
            assert other.GetTypeName() == "!pop.scalar<bool>"
            assert other.GetByteSize() == 1
            assert other.GetChildAtIndex(0).GetValueAsUnsigned(2) == 1
            assert other.GetSummary() == "True"
