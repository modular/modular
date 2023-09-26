# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This is a module doc."""

# RUN: kgen-translate -import-mojo %s | FileCheck %s

from test_package import method_defined_in_init

# CHECK: #[[MODULE_DOC:.*]] = #lit.doc.string<"This is a module doc."
# CHECK: #[[ALIAS_DOC:.*]] = #lit.doc.string<"This is an alias doc."
# CHECK: #[[GLOBAL_VAR_DOC:.*]] = #lit.doc.string<"This is a global variable doc."
# CHECK: #[[STRUCT_DOC:.*]] = #lit.doc.string<"This is a struct doc."
# CHECK: #[[STRUCT_FIELD_DOC:.*]] = #lit.doc.string<"This is a struct field doc."
# CHECK: #[[FUNCTION_DOC:.*]] = #lit.doc.string<"This is a function doc."
# CHECK: #[[TRAIT_DOC:.*]] = #lit.doc.string<"This is a trait doc."
# CHECK: #[[TRAIT_FUNCTION_DOC:.*]] = #lit.doc.string<"This is a trait function doc."
# CHECK: #[[PACKAGE_DOC:.*]] = #lit.doc.string<"This is a test package."

# CHECK: lit.file_module @"$parser-doc"{{.*}}docString = #[[MODULE_DOC]]
# CHECK: lit.alias.decl {{.*}}AliasType{{.*}}docString = #[[ALIAS_DOC]]
# CHECK: lit.globalvar.decl @value{{.*}}docString = #[[GLOBAL_VAR_DOC]]
# CHECK: lit.struct.decl @Struct{{.*}}docString = #[[STRUCT_DOC]]
# CHECK: lit.struct.field value{{.*}}docString = #[[STRUCT_FIELD_DOC]]
# CHECK: lit.func @"foo()"{{.*}}docString = #[[FUNCTION_DOC]]

# CHECK: lit.package @"$test_package"{{.*}}docString = #[[PACKAGE_DOC]]

alias AliasType = __mlir_type.`!kgen.mlirtype`
"""This is an alias doc."""

let value = 10
"""This is a global variable doc."""

struct Struct:
  """This is a struct doc."""

    var value: Int
    """This is a struct field doc."""

fn foo():
  """This is a function doc."""
  method_defined_in_init()
  return

##===----------------------------------------------------------------------===##
# Traits
##===----------------------------------------------------------------------===##

trait Trait:
  """This is a trait doc."""

  fn f(self: Self):
    """This is a trait function doc."""
    ...

