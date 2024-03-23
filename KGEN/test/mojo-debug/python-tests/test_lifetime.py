# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase


class TestLifetime(LLDBTestBase):
    def assert_var_not_available(self, ctx, name):
        var = ctx.frame.FindVariable(name)
        assert "variable not available" in var.GetError().GetCString()

    def simple_int(self):
        """
        Ensures that the lifetime for a very simple program that uses an Int
        is correct."""
        with self.build_and_launch("int.mojo") as ctx:
            foo = ctx.frame.FindVariable("foo")
            assert foo.GetValueAsUnsigned(-1) == 42

    def test_full_eager_destruction(self):
        """Ensures that if a variable is completely destroyed eagerly, the
        lifetime of the value is reflected in DWARF."""
        with self.build_and_launch("full_eager_destruction.mojo") as ctx:
            text = ctx.frame.FindVariable("text")
            assert text.GetSummary()
            self.assert_var_not_available(ctx, "number")
            self.assert_var_not_available(ctx, "simd")

            for i in range(2):
                ctx = ctx.resume()
                assert ctx
                number = ctx.frame.FindVariable("number")
                assert number.GetValueAsSigned(0) == 8
                self.assert_var_not_available(ctx, "text")
                self.assert_var_not_available(ctx, "simd")

            # Nothing is alive coming out of the loop.
            ctx = ctx.resume()
            assert ctx
            self.assert_var_not_available(ctx, "text")
            self.assert_var_not_available(ctx, "number")
            self.assert_var_not_available(ctx, "simd")

            # Nothing is alive in the else-block as it's past the last use of
            # all variables.
            ctx = ctx.resume()
            assert ctx
            self.assert_var_not_available(ctx, "text")
            self.assert_var_not_available(ctx, "number")
            self.assert_var_not_available(ctx, "simd")
