# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -I %S/inputs -verify-diagnostics %s

# Extensions must apply only to the struct decl they were declared on, not to
# any struct that shares its name. `ext_ident_x` and `ext_ident_y` each define
# their own `Foo` plus an extension of it; the explicit import below makes
# `ext_ident_y`'s `Foo` the visible one, but both extensions sit in scope under
# the name key "extension:Foo". Member lookup currently matches extensions to
# the type by name only, so `ext_ident_x`'s members leak onto the wrong `Foo`:
# the static resolves silently to the other module's function, and the
# instance method fails self-unification with a baffling "cannot be converted
# from 'Foo' to 'Foo'" instead of a missing-attribute error.

# See MOCO-4406.
#
# TODO: Remove the XFAIL once member lookup filters collected extensions by
# their resolved target decl (compare ExtensionDeclOp::getTargetStruct against
# the type's own symbol, as findExtensionsInScopeForStruct already does).
# XFAIL: *

from ext_ident_x import *
from ext_ident_y import Foo


def main():
    var f = Foo()
    _ = f.y_inst()
    _ = Foo.y_static()
    # expected-error @+1 {{'Foo' value has no attribute 'x_inst'}}
    _ = f.x_inst()
    # expected-error @+1 {{'Foo' value has no attribute 'x_static'}}
    _ = Foo.x_static()
