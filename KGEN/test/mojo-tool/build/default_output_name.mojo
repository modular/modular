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

# RUN: rm default_output_name || true
# RUN: %mojo-build %s
# RUN: test -x default_output_name

# RUN: rm default_output_name_2 || true
# RUN: %mojo-build %S/inputs/default_output_name_2.mojo
# RUN: test -x default_output_name_2


def main():
    pass
