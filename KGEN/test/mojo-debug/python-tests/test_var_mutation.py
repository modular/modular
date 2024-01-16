# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase

from modular.utils.debuglib.debugger import StopContext, lldb


class TestSample(LLDBTestBase):
    @staticmethod
    def assert_si64_int(ctx: StopContext, name: str, expected: int):
        var = ctx.frame.FindVariable(name)
        assert var.GetValue() == str(expected)
        assert var.GetTypeName() == "si64"
        assert var.GetDisplayTypeName() == "si64"
        assert var.GetType().GetTypeFlags() | lldb.eTypeIsInteger
        assert var.GetValueAsSigned(expected - 1) == expected

    def test_assignment(self):
        """Make sure basic var mutation assignment is tracked."""

        with self.build_and_launch("var_mutation_assignment.mojo") as ctx:
            TestSample.assert_si64_int(ctx, "i", 5)
            TestSample.assert_si64_int(ctx, "j", 7)

            ctx = ctx.resume()
            assert ctx
            TestSample.assert_si64_int(ctx, "i", 15)
            TestSample.assert_si64_int(ctx, "j", 7)

            ctx = ctx.resume()
            assert ctx
            TestSample.assert_si64_int(ctx, "i", 15)
            TestSample.assert_si64_int(ctx, "j", 13)

            ctx = ctx.resume()
            assert ctx
            TestSample.assert_si64_int(ctx, "i", 2)
            TestSample.assert_si64_int(ctx, "j", 13)

    def test_iteration(self):
        """Make sure changes to basic loop index variable is tracked."""

        with self.build_and_launch("var_mutation_iteration.mojo") as ctx:
            for i in range(3):
                assert ctx
                TestSample.assert_si64_int(ctx, "i", i)
                ctx = ctx.resume()
