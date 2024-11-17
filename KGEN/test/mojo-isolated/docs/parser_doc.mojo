# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This is a module doc."""

# RUN: %parse-mojo-isolated %s | FileCheck %s

from docs_package import documented_method_defined_in_init

# CHECK: #[[MODULE_DOC:.*]] = #lit.doc.string<"This is a module doc."
# CHECK: #[[ALIAS_DOC:.*]] = #lit.doc.string<"This is an alias doc."
# CHECK: #[[GLOBAL_VAR_DOC:.*]] = #lit.doc.string<"This is a global variable doc."
# CHECK: #[[STRUCT_DOC:.*]] = #lit.doc.string<"This is a struct doc."
# CHECK: #[[STRUCT_FIELD_DOC:.*]] = #lit.doc.string<"This is a struct field doc."
# CHECK: #[[FUNCTION_DOC:.*]] = #lit.doc.string<"This is a function doc."
# CHECK: #[[TRAIT_DOC:.*]] = #lit.doc.string<"This is a trait doc."
# CHECK: #[[TRAIT_FUNCTION_DOC:.*]] = #lit.doc.string<"This is a trait function doc."
# CHECK: #[[PACKAGE_DOC:.*]] = #lit.doc.string<"This is a test package."
# CHECK: #[[IMPORTED_FUNC_DOC:.*]] = #lit.doc.string<"This is an imported method."

# CHECK: lit.file_module @parser_doc{{.*}}docString = #[[MODULE_DOC]]
# CHECK: lit.alias.decl {{.*}}AliasType{{.*}}docString = #[[ALIAS_DOC]]
# CHECK: lit.globalvar.decl @value{{.*}}docString = #[[GLOBAL_VAR_DOC]]
# CHECK: lit.struct.decl @Struct{{.*}}docString = #[[STRUCT_DOC]]
# CHECK: lit.struct.field value{{.*}}docString = #[[STRUCT_FIELD_DOC]]
# CHECK: lit.func @"foo()"{{.*}}docString = #[[FUNCTION_DOC]]

# CHECK: lit.package @docs_package{{.*}}docString = #[[PACKAGE_DOC]]
# CHECK: lit.func @"documented_method_defined_in_init()"{{.*}}docString = #[[IMPORTED_FUNC_DOC]]

alias AliasType = __mlir_type.`!kgen.type`
"""This is an alias doc."""

var value = __mlir_attr.`10 : index`
"""This is a global variable doc."""

struct Struct:
  """This is a struct doc."""

    var value: __mlir_type.index
    """This is a struct field doc."""

fn foo():
  """This is a function doc."""
  documented_method_defined_in_init()
  return

trait AnyType:
  """A stub for the AnyType trait to allow decoupling from the builtins."""
  pass

trait Trait:
  """This is a trait doc."""

  fn f(self):
    """This is a trait function doc."""
    ...
