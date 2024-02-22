# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase


class TestTupleAndSIMD(LLDBTestBase):
    def test_bool(self):
        """
        Test that we can parse a StaticTuple and a SIMD with similar entries.
        In DWARF, they are both array types but SIMD is marked as vector, which
        means that the layout is different.
        """

        with self.build_and_launch("static_tuple.mojo") as ctx:
            [success, output, error] = ctx.run_command("v")
            assert (
                """(array<4, scalar<si16>>) tuple = {
  [0] = ([0] = 1)
  [1] = ([0] = 2)
  [2] = ([0] = 3)
  [3] = ([0] = 4)
}
(simd<4, si16>) simd = ([0] = 1, [1] = 2, [2] = 3, [3] = 4)"""
                in output
            )
