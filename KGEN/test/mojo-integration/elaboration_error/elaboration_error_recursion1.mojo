# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -elaborate -elaboration-max-depth=128 --elaboration-error-verbose=no-params %s --verify-diagnostics
# RUN: not mojo build -elaboration-max-depth=128 --elaboration-error-verbose=no-params %s 2>&1 | FileCheck %s

from collections.string.string_slice import _get_kgen_string
from sys import env_get_bool


# expected-note @below {{function instantiation failed}}
# expected-note @below {{remaining errors after}}
# expected-note-re @below {{error recurses {{[0-9]+}} times}}
# expected-note-re @below {{elaborator expansion is {{[0-9]+}} levels deep - infinite recursion?}}
fn self_recursion[i: Int]() -> Int:
    # expected-note @below {{call expansion failed}}
    # expected-warning @below {{self recursive call will cause an infinite loop}}
    var x = self_recursion[i + 1]()
    return x


# expected-error @below {{function instantiation failed}}
fn main():
    # expected-note @below {{call expansion failed}}
    _ = self_recursion[1]()


# CHECK: self recursive call will cause an infinite loop
# CHECK: error recurses {{[0-9]+}} times
# CHECK: elaborator expansion is {{[0-9]+}} levels deep - infinite recursion?
