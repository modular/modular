# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir %t
# RUN: %mojo-build %s -o %t/output.o --emit object

# COM: Check that `file` recognizes it as an object file
# RUN: file %t/output.o | FileCheck %s

# COM: Full output:
# COM:   - macOS: "output.o: Mach-O 64-bit object arm64"
# COM:   - Linux: "output.o: ELF 64-bit LSB relocatable, ARM aarch64, version 1 (SYSV), not stripped"
# CHECK: output.o: {{(Mach-O 64-bit object|ELF 64-bit LSB relocatable)}}


fn main():
    pass
