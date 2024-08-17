# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# FIXME(#26472): Mojo on windows doesn't support emoji file extensions.
# UNSUPPORTED: windows

# Test the various error cases of imports. The run line also checks that we
# properly handle the case of an invalid import directory.

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics -I=unknownincludedir -I=%S -I=%S/test_package %s

# expected-error @+1 {{expected module name}}
import --

# // -----

# expected-error @+1 {{expected name to bind import}}
import foo as --

# // -----

# expected-error @+1 {{unable to locate module 'there_cant_be_a_module_named_this'}}
import there_cant_be_a_module_named_this

# // -----

# expected-error @+1 {{expected module name}}
from --

# // -----

# expected-error @+1 {{expected 'import' after module name}}
from foo --

# // -----

# expected-error @+1 {{expected construct name to import}}
from imported_module import ()

# // -----

# expected-error @+1 {{expected name to import 'foo' as}}
from imported_module import (foo as --)

# // -----

# Tuple import allows an optional trailing comma.
from imported_module import (imported_fn,)

# expected-error @+1 {{expected construct name to import}}
from imported_module import imported_fn,

# // -----

# expected-error @+1 {{expected ')' after import list}}
from imported_module import (imported_fn --

# // -----

# expected-error @below {{cannot import relative to a top-level package}}
from .module import foo

# // -----

# expected-error @below {{unable to locate module 'unknown_nested_module'}}
from test_package.unknown_nested_module import bar

# // -----

import test_package

fn assignPackageModule():
  test_package = test_package

# // -----

# Check that we can't directly import `test_package.module_in_package` just
# because `test_package` is in the path.

# expected-error @below {{unable to locate module 'module_in_package'}}
import module_in_package

# Check that we don't crash on an invalid use of the missing imported decl.
# expected-error @below {{expressions are not yet supported at the file scope level}}
module_in_package

# // -----

# expected-error @below {{unable to locate module 'does_not_exist'}}
import imported_module.does_not_exist

from imported_module import *

fn test_import():
  imported_fn()

  # expected-error @below {{use of unknown declaration '_ignored_wildcard_fn'}}
  _ignored_wildcard_fn()

from imported_module import (
  # expected-error @below {{module 'imported_module' does not contain 'there_cant_be_a_decl_named_this'}}
  there_cant_be_a_decl_named_this
)

# expected-error @below {{unable to locate module 'there_cant_be_a_module_named_this'}}
from there_cant_be_a_module_named_this import there_cant_be_another_decl_named_this

# expected-note @below {{previous definition here}}
fn already_defined_fn():
  return

# expected-error @below {{invalid redefinition of 'already_defined_fn'}}
from imported_module import imported_fn as already_defined_fn

# expected-error @below {{ambiguous import}}
from test_bad_package.extension_dup import getExtension

# // -----

import imported_module
import test_package

fn baz():
    # expected-error @below {{module 'imported_module' is not callable; did you mean to call imported_module.imported_module?}}
    imported_module()
    # expected-error @below {{module 'imported_module' is not subscriptable; did you mean to subscript imported_module.imported_module?}}
    imported_module[`0`]
    # expected-error @below {{module 'test_package' is not callable}}
    test_package()
    # expected-error @below {{module 'test_package' is not subscriptable}}
    test_package[`0`]
