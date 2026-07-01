# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Tests that a user sees a useful error message if they use the wrong import
# syntax, trying to import through a *module*. Only packages can have modules or
# packages imported from them.

# RUN: %parse-mojo-isolated -split-input-file -I=%S/inputs -verify-diagnostics %s

# expected-error @+1 {{'module1' is a module, not a package; it has no nested module or package 'bar'}}
import import_through_module.nested_package.module1.bar

# // -----

# expected-error @+1 {{'module1' is a module, not a package; it has no nested module or package 'module2'}}
import import_through_module.nested_package.module1.module2.bar

# // -----

# expected-error @+1 {{'module1' is a module, not a package; it has no nested module or package 'module2'}}
from import_through_module.nested_package.module1.module2 import bar

# // -----

# expected-error @+1 {{'module1' is a module, not a package; it has no nested module or package 'bar'}}
from import_through_module.nested_package.module1.bar import woof

# // -----

# expected-error @+1 {{'module1' is a module, not a package; it has no nested module or package 'module2'}}
from import_through_module.nested_package.module1.module2.bar import woof
