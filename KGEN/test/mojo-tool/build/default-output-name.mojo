# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Test that, when no output file path argument is provided, `mojo build` creates
# files in the current workfing directory, using nice names that are based on
# the input file name.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm default-output-name || true
# RUN: mojo build %s
# RUN: test -x default-output-name

# RUN: rm default-output-name-2 || true
# RUN: mojo build %S/inputs/default-output-name-2.mojo
# RUN: test -x default-output-name-2


fn main():
    pass
