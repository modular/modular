# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from test_package.module import dont_inline_me


# Prevent this function from being inlined into modules importing it. We wish to
# test that even when its definition exists in a separate package module (and
# that it, in turn, calls a function defined in another separate package), the
# module importing this function can invoke it and its dependent symbols.
@no_inline
def dont_inline_me_either():
    dont_inline_me()
