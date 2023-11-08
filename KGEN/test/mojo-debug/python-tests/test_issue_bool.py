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
          - They are reported as having 0 size, instead of 1 bit.
          - They are reported to be of type `bool`, but `!pop.scalar<bool>`
            would be the correct type.

        Once these issues get fixed, the summary provider for bools should just
        work.

        There's no SB API for getting the bit size of a type, but the byte size
        should be 1.
        """

        with self.build_and_launch("bool.mojo") as ctx:
            true = ctx.frame.FindVariable("true")
            assert true.GetTypeName() != "!pop.scalar<bool>"
            assert true.GetByteSize() != 1
            assert true.GetSummary() != "True"

            false = ctx.frame.FindVariable("true")
            assert false.GetTypeName() != "!pop.scalar<bool>"
            assert false.GetByteSize() != 1
            assert false.GetSummary() != "False"
