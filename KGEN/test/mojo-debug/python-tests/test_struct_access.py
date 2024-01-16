# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase

from modular.utils.debuglib.lldbtypes import SBValue


class TestSample(LLDBTestBase):
    @staticmethod
    def check_int_pair(struct: SBValue, first: int, second: int):
        assert struct.GetDisplayTypeName() == "MyPair"
        assert struct.GetNumChildren() == 2
        first_field = struct.GetChildAtIndex(0)
        assert first_field.GetValueAsSigned(first - 1) == first
        second_field = struct.GetChildAtIndex(1)
        assert second_field.GetValueAsSigned(second - 1) == second

    def test_access(self):
        """Make sure struct variable is tracked correctly."""

        with self.build_and_launch("struct_access.mojo") as ctx:
            p = ctx.frame.FindVariable("p")
            TestSample.check_int_pair(p, 1, 2)

            ctx = ctx.resume()
            assert ctx
            p = ctx.frame.FindVariable("p")
            TestSample.check_int_pair(p, 3, 4)

            ctx = ctx.resume()
            assert ctx
            pp = ctx.frame.FindVariable("pp")
            assert pp.GetDisplayTypeName() == "MyPairPair"
            assert pp.GetNumChildren() == 2
            pp_first = pp.GetChildAtIndex(0)
            TestSample.check_int_pair(pp_first, 5, 6)
            pp_second = pp.GetChildAtIndex(1)
            TestSample.check_int_pair(pp_second, 7, 8)
