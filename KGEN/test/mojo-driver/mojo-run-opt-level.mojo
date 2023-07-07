# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# We can run this file with optimizations disabled.
# RUN: mojo-driver run --no-optimization %s
# RUN: mojo-driver run -O0 %s

from IO import print


fn main() -> None:
    print("ok")
