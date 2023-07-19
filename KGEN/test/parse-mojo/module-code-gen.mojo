# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -verify-diagnostics -import-mojo | FileCheck %s

from String import String

##===----------------------------------------------------------------------===##
# Runtime Closures
##===----------------------------------------------------------------------===##

# CHECK: lit.file_module @"$module-code-gen" {
# CHECK-NEXT: lit.struct.decl @"_CW_
# CHECK:  lit.struct.field field0 : !pop.pointer<array<0, i1>>
# CHECK: !kgen.signature<(!pop.pointer<@{{.*}}::@String> byref_result, !pop.pointer<@{{.*}}::@String> borrow_in_mem) capturing -> !lit.none>
fn makes_escaping_closure(m: String, z:String, y:Bool) -> fn(String) escaping -> String:
   fn myclosure(n:String) -> String:
      return n + m
   return myclosure
