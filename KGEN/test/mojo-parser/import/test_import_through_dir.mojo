# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Tests that you can import modules and packages through (nested) directories
# using the same syntax as through packages.

# RUN: %parse-mojo-isolated -split-input-file -I=%S/inputs -verify-diagnostics %s

import import_through_dir.module

# // -----

from import_through_dir import module

# // -----

from import_through_dir.module import foo

# // -----

# Nested directories

import import_through_dir.nested_dir.module

# // -----

from import_through_dir.nested_dir import module

# // -----

from import_through_dir.nested_dir.module import bar

# // -----

# Packages inside nested directories

import import_through_dir.nested_dir.nested_package.module

# // -----

from import_through_dir.nested_dir.nested_package import module

# // -----

from import_through_dir.nested_dir.nested_package import baz

# // -----

from import_through_dir.nested_dir.nested_package.module import baz2
