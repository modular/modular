# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase


class TestDynamicVector(LLDBTestBase):
    def test(self):
        """Tests that DynamicVector can be parsed correctly and its data
        formatter works correctly as well."""

        with self.build_and_launch("dynamic_vector.mojo") as ctx:
            var = ctx.frame.FindVariable("point_vec")
            assert var.GetSummary() == "(size 2)"
            assert var.GetValueForExpressionPath("[0].x").GetValue() == "1"
            assert var.GetValueForExpressionPath("[1].y").GetValue() == "-2"

            ctx = ctx.resume()
            var = ctx.frame.FindVariable("int_vec")
            assert var.GetSummary() == "(size 2)[1, 2]"

            ctx = ctx.resume()
            var = ctx.frame.FindVariable("int_vec")
            assert (
                var.GetSummary()
                == "(size 103)[1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, ...]"
            )
