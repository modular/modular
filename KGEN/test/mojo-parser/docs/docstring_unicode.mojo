# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Verify that \u and \U unicode escape sequences in docstrings are resolved
# correctly. Docstrings use the same Lexer::getStringLiteralValue path as
# regular strings, so no special handling is needed — this test confirms that.


# CHECK: #lit.doc.string<"ASCII: h, Latin: \C3\A9, CJK: \E4\B8\AD, Emoji: \F0\9F\98\80"
def unicode_escapes():
    """ASCII: \u0068, Latin: \u00E9, CJK: \u4E2D, Emoji: \U0001F600"""
    pass
