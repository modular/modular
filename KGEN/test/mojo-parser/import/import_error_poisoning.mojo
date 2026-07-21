# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A failed import must not poison its name for the rest of the compilation:
# inputs/imported_module.mojo genuinely exists, so it must still be importable
# after the lookup of a same-named submodule of test_package fails. The bad
# call at the end is the tripwire proving the real module was bound: it is
# only diagnosed when `imported_module` resolves to the real module rather
# than binding a silent error state.

# RUN: %parse-mojo-isolated -verify-diagnostics -I=%S/inputs %s

# expected-error @+1 {{unable to locate module 'imported_module'}}
import test_package.imported_module

import imported_module

def main():
    # expected-error @+1 {{invalid call to 'imported_fn': unexpected argument}}
    imported_module.imported_fn(42)
