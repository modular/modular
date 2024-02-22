# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from LLDBTestBase import LLDBTestBase


class TestStrings(LLDBTestBase):
    def test_strings(self):
        """Ensures that String and StringLiteral can be parsed correctly from
        memory"""

        with self.build_and_launch("strings.mojo") as ctx:
            st = ctx.frame.FindVariable("st")
            assert '"012345678910111213141' in st.GetSummary()

            ctx = ctx.resume()
            assert ctx is not None

            literal = ctx.frame.FindVariable("literal")
            s1 = ctx.frame.FindVariable("s1")
            # FIXME(29497): reenable tests over s2 and s3.
            # s2 = ctx.frame.FindVariable("s2")
            # s3 = ctx.frame.FindVariable("s3")
            s4 = ctx.frame.FindVariable("s4")

            # StringLiterals, being built-in, provide the underlying strings
            # as value. On the other hand, String, being parsed by a data
            # formatter, provides the underlying string as a Summary, following
            # C++'s convention in LLDB.
            assert literal.GetValue() == '"string_literal"'
            assert s1.GetSummary() == '"let_string"'
            # assert '"012345678910111213141' in s2.GetSummary()
            # TODO(#31429): This test currently doesn't work when the standard
            # library is built with debug information.
            # assert s3.GetSummary() == '""'
            assert '"012345678910111213141' in s4.GetSummary()
