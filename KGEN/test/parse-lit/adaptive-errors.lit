# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s -verify-diagnostics -I %S/../mojo-examples/


@adaptive
fn foo():
    let b = 3
    return


# expected-note @below {{non-adaptive candidate here}}
fn foo():
    let b = 5
    return


fn main():
    # expected-error @below {{ambiguous call to 'foo', multiple implementations detected but not all are marked adaptive}}
    foo()
    return
