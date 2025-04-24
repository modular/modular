# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-build -j 1 %s
# RUN: %mojo-build --num-threads 5 %s

# This is a very simple test that just checks that we can pass --num-threads command line argument.
fn main():
    pass
