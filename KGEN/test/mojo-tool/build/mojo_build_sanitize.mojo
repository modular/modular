# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# Note: Don't run with pre-existing sanitizers to ensure sanitizers work in a
#       clean environment.
# UNSUPPORTED: asan,msan,tsan
# TODO: Support windows when we build with sanitizers.
# TODO: Mac requires using a non-apple clang, as our sanitizers are different.
# UNSUPPORTED: system-darwin

# RUN: not %mojo-build %s --sanitize unknown -o %t 2>&1 | FileCheck %s --check-prefix=ERROR

# ERROR: error: invalid sanitizer 'unknown', expected one of: `address` or `thread`

# Check that we have the expected sanitizer symbols in our built executables.

# RUN: %mojo-build %s --sanitize=address -o %t
# RUN: llvm-objdump %t -t | FileCheck %s --check-prefix=ASAN

# RUN: %mojo-build %s --sanitize thread -o %t
# RUN: llvm-objdump %t -t | FileCheck %s --check-prefix=TSAN

# ASAN: __asan_init
# TSAN: __tsan_init


def main():
    print("sanitizer")
    return
