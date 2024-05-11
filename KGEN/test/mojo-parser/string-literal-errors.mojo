# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics -split-input-file %s


# The octal escape sequence in string literals \ooo can have variable length.
fn testOctal():
    var x = "A\0"
    x = "A\01"
    x = "A\012"


# // -----


fn testTripleQuote():
    # expected-error @below {{invalid escape sequence}}
    var x = """$\s$"""
