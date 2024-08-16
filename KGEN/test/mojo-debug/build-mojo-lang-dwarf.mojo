# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# LLDB fails with asan because it's built by default with python support in the
# CI, and python fails asan.
# UNSUPPORTED: asan


# COM: Using the default language Mojo
# RUN: mojo build --debug-level=full -O0 %s -o %t
# RUN: mojo debug -X -o -X 'image lookup -vs build-mojo-lang-dwarf::foo()' -X -b %t | FileCheck %s --check-prefix CHECK-MOJO
# CHECK-MOJO: language = "mojo"

# COM: Setting explicitly the language C
# RUN: mojo build --debug-level full -O0 --debug-info-language C %s -o %t
# RUN: mojo debug -X -o -X 'image lookup -vs build-mojo-lang-dwarf::foo()' -X -b %t | FileCheck %s --check-prefix CHECK-C


# CHECK-C: language = "c"
fn foo():
    pass


fn main():
    foo()
