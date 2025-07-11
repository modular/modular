# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mkdir %t
# COM: For sake of keeping the test commands simple, this hard-codes the use of
# COM:   `.dylib`, even on platforms like Linux where .so would otherwise be used.
# RUN: %mojo-build %s -o %t/example.dylib --emit shared-lib

# COM: Check that this file compiled to a dynamic library:
# RUN: test -f %t/example.dylib

# COM: Check that `file` recognizes it as a dynamic library
# RUN: file %t/example.dylib | FileCheck %s

# COM: Full output:
# COM:   - macOS: "example.dylib: Mach-O 64-bit dynamically linked shared library arm64"
# COM:   - Linux: "ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, with debug_info, not stripped"
# CHECK: dynamically linked


@export
fn foo():
    pass
