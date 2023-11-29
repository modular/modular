# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s -verify-diagnostics -mojo-disable-builtins -mojo-disable-parser-caching

alias Int = __mlir_type.index

@adaptive
fn foo():
    return


# expected-note @below {{non-adaptive candidate here}}
fn foo():
    return


struct Stuff[*size: Int]:
    pass


# expected-note @+2 {{previous definition here}}
@adaptive
fn fred[c: Int]() -> Stuff[c]:
    pass


# expected-error @+2 {{redefinition of function 'fred' cannot overload on return type only}}
@adaptive
fn fred[c: Int]() -> Stuff[c, c]:
    pass


@register_passable("trivial")
struct TrivialStuff[*size: Int]:
    pass


# expected-note @+2 {{previous definition here}}
@adaptive
fn waldo[c: Int]() -> TrivialStuff[c]:
    pass


# expected-error @+2 {{redefinition of function 'waldo' cannot overload on return type only}}
@adaptive
fn waldo[c: Int]() -> TrivialStuff[c, c]:
    pass


fn bar():
    # expected-error @below {{ambiguous call to 'foo', multiple implementations detected but not all are marked adaptive}}
    foo()
