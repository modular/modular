# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.LLDBTestBase import LLDBTestBase


class TestIssueBool(LLDBTestBase):
    def test(self):
        """FIXME(23022):
        Booleans are emitted wrong:
          - They are reported to be of type `bool`, but `!pop.scalar<bool>`
            would be the correct type.

        Once these issues get fixed, the summary provider for bools should just
        work.

        There's no SB API for getting the bit size of a type, but the byte size
        should be 1.
        """

        with self.build_and_launch("bool.mojo") as ctx:
            true = ctx.frame.FindVariable("true")
            # This should be !pop.scalar<bool>
            assert true.GetTypeName() == "i1"
            assert true.GetByteSize() == 1
            assert true.GetValueAsUnsigned(2) == 1
            # We need to show True in the summary
            assert true.GetSummary() != "True"

            false = ctx.frame.FindVariable("false")
            # This should be !pop.scalar<bool>
            assert false.GetTypeName() == "i1"
            assert false.GetByteSize() == 1
            assert false.GetValueAsUnsigned(2) == 0
            # We need to show False in the summary
            assert false.GetSummary() != "False"

            other = ctx.frame.FindVariable("other")
            # This should be !pop.scalar<bool>
            assert other.GetTypeName() == "i1"
            assert other.GetByteSize() == 1
            assert other.GetValueAsUnsigned(2) == 1
            # We need to show True in the summary
            assert other.GetSummary() != "True"
