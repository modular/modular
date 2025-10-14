# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Test that when using `--emit`, `mojo build` creates the output files in the
# current working directory, using nice names that are based on the input file
# name.
#
# ===----------------------------------------------------------------------=== #

# RUN: rm %S/emit_output_name.ll || true
# RUN: %mojo-build --emit llvm %s
# RUN: test -e %S/emit_output_name.ll

# RUN: rm %S/emit_output_name.ll || true
# RUN: %mojo-build %s --emit llvm
# RUN: test -e %S/emit_output_name.ll

# RUN: rm %S/emit_output_name.s || true
# RUN: %mojo-build --emit asm %s
# RUN: test -e %S/emit_output_name.s

# RUN: rm %S/emit_output_name.s || true
# RUN: %mojo-build %s --emit asm
# RUN: test -e %S/emit_output_name.s


fn main():
    pass
