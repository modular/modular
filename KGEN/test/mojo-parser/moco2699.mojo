# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --verify-diagnostics %s

trait a:
    # expected-error @below {{nested struct in a trait not supported here}}
    struct b
trait c(a):
