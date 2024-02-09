# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from LLDBTestBase import LLDBTestBase


class TestStrings(LLDBTestBase):
    def test_strings(self):
        """Ensures that String and StringLiteral can be parsed correctly from
        memory"""

        with self.build_and_launch("strings.mojo") as ctx:
            literal = ctx.frame.FindVariable("literal")
            s1 = ctx.frame.FindVariable("s1")
            s2 = ctx.frame.FindVariable("s2")
            s3 = ctx.frame.FindVariable("s3")
            # StringLiterals, being built-in, provide the underlying strings
            # as value. On the other hand, Structs, being parsed by a data
            # formatter, provides the underlying string as a Summary, following
            # C++'s convention in LLDB.
            assert literal.GetValue() == '"string_literal"'
            assert s1.GetSummary() == '"let_string"'
            assert '"012345678910111213141' in s2.GetSummary()

            # TODO(#31429): This test currently doesn't work when the standard
            # library is built with debug information.
            # assert s3.GetSummary() == '""'
