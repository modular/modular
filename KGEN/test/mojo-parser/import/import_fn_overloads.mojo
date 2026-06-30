# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Tests that a package can define overloads of a function in multiple modules,
# bring them together in its __init__.mojo, and have users import them in a
# variety of ways.

# RUN: %parse-mojo-isolated -split-input-file -I=%S/inputs -verify-diagnostics %s

import fn_overload_package

def main():
    _ = fn_overload_package.foo(42)      # ok
    _ = fn_overload_package.foo("hello") # ok

# // -----

import fn_overload_package.module1
import fn_overload_package.module2

def main():
    _ = fn_overload_package.module1.foo(42)      # ok
    _ = fn_overload_package.module2.foo("hello") # ok

# // -----

from fn_overload_package import foo

def main():
    _ = foo(42)      # ok
    _ = foo("hello") # ok

# // -----

from fn_overload_package.module1 import foo

def main():
    _ = foo(42)
    # expected-error @+1 {{invalid call to 'foo': value passed to 'x' cannot be converted from 'StringLiteral["hello"]' to 'Int'}}
    _ = foo("hello")

# // -----

from fn_overload_package.module2 import foo

def main():
    # expected-error @+1 {{invalid call to 'foo': value passed to 'x' cannot be converted from 'IntLiteral[42]' to 'String'}}
    _ = foo(42)
    _ = foo("hello")
