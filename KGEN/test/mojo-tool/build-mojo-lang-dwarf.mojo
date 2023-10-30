# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# COM: Setting the language to Mojo and using non-C-like symbols
# RUN: mojo build --debug-level full -O0 --debug-info-language Mojo --no-alnum-symbols %s -o %t
# RUN: mojo debug %t -o 'image lookup -vs $build-mojo-lang-dwarf::main()' -b | FileCheck %s --check-prefix CHECK-MOJO
# CHECK-MOJO: language = "mojo"


# COM: Using the default language C
# RUN: mojo build --debug-level full -O0 %s -o %t
# RUN: mojo debug %t -o 'image lookup -vs _build_mojo_lang_dwarf_fookAtAtAtA6A6AoApA' -b | FileCheck %s --check-prefix CHECK-C

# COM: Setting explicitly the language C
# RUN: mojo build --debug-level full -O0 --debug-info-language C %s -o %t
# RUN: mojo debug %t -o 'image lookup -vs _build_mojo_lang_dwarf_fookAtAtAtA6A6AoApA' -b | FileCheck %s --check-prefix CHECK-C


# CHECK-C: language = "c"
fn foo():
    pass


fn main():
    foo()
