# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Tests the ways in which a module can import itself

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=%S/inputs %s

import import_self

def main():
    import_self.foo() # ok

def foo(): pass

# // -----

# TODO: This test is encoding current behaviour - do we want to support this?

# expected-note @below {{cannot overload with this non-function definition}}
# expected-error @below {{attempt to resolve a recursive reference to declaration}}
# expected-note @below {{referenced from here}}
# expected-note @below {{by declaration}}
# expected-note @below {{referenced through this use}}
from import_self import foo

def main():
    # expected-error @below {{use of unknown declaration 'import_self'}}
    import_self.foo()

# expected-error @+1 {{invalid redefinition of 'foo'}}
def foo(): pass
