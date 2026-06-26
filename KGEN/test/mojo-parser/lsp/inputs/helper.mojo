# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Helper module imported by ../signature_only_imports.mojo. Not a test itself
# (excluded via lit.cfg.py). The distinctive integer literals in the body let
# the importing test assert whether this body was resolved.


def compute_secret_value() -> Int:
    var a = 7
    var b = 6
    return a * b
