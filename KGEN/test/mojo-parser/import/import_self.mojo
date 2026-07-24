# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A standalone module cannot import its own bare name: it would only ever
# resolve to itself, silently shadowing any same-named system package.
# Self-imports inside packages are unaffected (see
# import_relative_self_reexport.mojo and import_self_named_package.mojo).

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics %s

# expected-error @+1 {{module 'import_self' cannot import itself}}
import import_self

def main():
    import_self.foo()

def foo(): pass

# // -----

# expected-error @+1 {{module 'import_self' cannot import itself}}
from import_self import foo

def main():
    foo()

# // -----

# Aliasing doesn't make the self-import any less ambiguous.

# expected-error @+1 {{module 'import_self' cannot import itself}}
import import_self as myself

def main():
    myself.foo()

def foo(): pass

# // -----

# A dotted path is rejected at its first component.

# expected-error @+1 {{module 'import_self' cannot import itself}}
import import_self.submodule

def main():
    pass
