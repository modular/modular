# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase


class TestInvalidScalar(LLDBTestBase):
    def test_invalid_scalar(self):
        """Ensures that `scalar<invalid>` is correctly considered as having 0
        bytes. It also expects to have an inner data type of `None`.
        A very common case where this is found is `Dict`, so we start with a
        `Dict` variable and navigate its members until reaching the invalid value.
        """

        with self.build_and_launch("invalid.mojo") as ctx:
            dict = ctx.frame.FindVariable("dict")
            _index = dict.GetChildMemberWithName("_index")
            data = _index.GetChildMemberWithName("data")
            assert data.GetTypeName() == "!kgen.pointer<scalar<invalid>>"
            invalid_scalar = data.GetChildAtIndex(0)
            assert invalid_scalar.GetTypeName() == "!pop.scalar<invalid>"
            assert invalid_scalar.GetByteSize() == 0
            none = invalid_scalar.GetChildAtIndex(0)
            assert none.GetTypeName() == "!kgen.none"
            assert none.GetByteSize() == 0
