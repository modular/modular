# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase


class TestList(LLDBTestBase):
    def test_dynamic_vector(self):
        """Tests that List can be parsed correctly and its data
        formatter works correctly as well."""

        # TODO(#29440): Re-enable asserts when debugger uses SourceName for structs.
        with self.build_and_launch("dynamic_vector.mojo") as ctx:
            var = ctx.frame.FindVariable("point_vec")
            assert var.GetSummary() == "(size 2)"
            assert var.GetValueForExpressionPath("[0].x").GetValue() == "1"
            assert var.GetValueForExpressionPath("[1].y").GetValue() == "-2"

            ctx = ctx.resume()
            assert ctx is not None
            var = ctx.frame.FindVariable("int_vec")
            assert var.GetSummary() == "(size 2)[1, 2]"

            ctx = ctx.resume()
            assert ctx is not None
            var = ctx.frame.FindVariable("int_vec")
            assert (
                var.GetSummary()
                == "(size 103)[1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, ...]"
            )
