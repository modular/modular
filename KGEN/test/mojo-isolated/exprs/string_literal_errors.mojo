# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics -split-input-file %s


# The octal escape sequence in string literals \ooo can have variable length.
fn testOctal():
    alias x = "A\0"
    alias y = "A\01"
    alias z = "A\012"


# // -----


fn testTripleQuote():
    # expected-error @below {{invalid escape sequence}}
    var x = """$\s$"""
