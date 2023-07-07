# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -split-input-file -verify-diagnostics -I=%S %s

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

# Check that we properly allow import of an imported decl.

from imported_module import *

fn import_of_import(arg: Float64):
  pass

# // -----

# expected-error @below {{cannot import relative to a top-level package}}
from .module import foo

# // -----

# expected-error @below {{unable to locate module 'unknown_nested_module'}}
from test_package.unknown_nested_module import bar

# // -----

from imported_module import *

fn test_import():
  imported_fn()

  # expected-error @below {{use of unknown declaration '_ignored_wildcard_fn'}}
  _ignored_wildcard_fn()

# expected-error @below {{module 'imported_module' does not contain 'there_cant_be_a_decl_named_this'}}
from imported_module import there_cant_be_a_decl_named_this

# expected-error @below {{unable to locate module 'there_cant_be_a_module_named_this'}}
from there_cant_be_a_module_named_this import there_cant_be_another_decl_named_this

# expected-note @below {{previous definition here}}
fn already_defined_fn():
  return

# expected-error @below {{invalid redefinition of 'already_defined_fn'}}
from imported_module import imported_fn as already_defined_fn
