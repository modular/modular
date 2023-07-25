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
# CHECK-NEXT:    lit.struct.decl @"_CI_$module-code-gen_\22($String::String,$String::String)\22throws"
# CHECK-NEXT:      lit.struct.field field0 : !kgen.declref<@"$String"::@String>
# CHECK-NEXT:    }
# CHECK-NEXT:    lit.struct.decl @"_CI_$module-code-gen_\22($String::String,$String::String)\22"
# CHECK-NEXT:      lit.struct.field field0 : !kgen.declref<@"$String"::@String>
# CHECK-NEXT:    }
# CHECK-NEXT: lit.struct.decl @"_CW_$module-code-gen_\22(,$String::String)\22"
# CHECK-NEXT:     lit.struct.field field0 : !pop.pointer<array<0, i1>>
# CHECK-NEXT:     lit.struct.field dtor : !kgen.signature<(!pop.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:     lit.struct.field copy : !kgen.signature<(!pop.pointer<array<0, i1>> init_self, !pop.pointer<array<0, i1>> borrow_in_mem) -> !lit.none>
# CHECK-NEXT:     lit.struct.field move : !kgen.signature<(!pop.pointer<array<0, i1>> init_self, !pop.pointer<array<0, i1>> owned_in_mem) -> !lit.none>  
# CHECK-NEXT: lit.func @"__del__
# CHECK-NEXT:   [[DTOR_PTR:%.*]] = lit.struct.gep %self[dtor]
# CHECK-NEXT:   [[DTOR:%.*]] = pop.load [[DTOR_PTR]] : !pop.pointer<!kgen.signature<(!pop.pointer<array<0, i1>>) -> !lit.none>>
# CHECK-NEXT:   [[IMPL_PTR:%.*]] = lit.struct.gep %self[field0]
# CHECK-NEXT:   [[IMPL:%.*]] = pop.load [[IMPL_PTR]] : !pop.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   kgen.call_signature [[DTOR]]([[IMPL]]) : (!pop.pointer<array<0, i1>>) -> !lit.none
# CHECK-NEXT:   kgen.param.constant
# CHECK-NEXT:   lit.ownership.mark.destroyed %self
# CHECK: lit.func @"__moveinit__
# CHECK-NEXT:   [[SELF_IMPL_PTR:%.*]] = lit.struct.gep %self[field0]
# CHECK-NEXT:   [[EXISTING_IMPL_PTR:%.*]] = lit.struct.gep %existing[field0]
# CHECK-NEXT:   [[SELF_IMPL:%.*]] = pop.load [[SELF_IMPL_PTR]] : !pop.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   [[EXISTING_IMPL:%.*]] = pop.load [[EXISTING_IMPL_PTR]] : !pop.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   [[MOVE_PTR:%.*]] = lit.struct.gep %self[move]
# CHECK-NEXT:   [[MOVE:%.*]] = pop.load [[MOVE_PTR]]
# CHECK-NEXT:   kgen.call_signature [[MOVE]]([[SELF_IMPL]], [[EXISTING_IMPL]])
# CHECK-NEXT:   kgen.param.constant
# CHECK-NEXT:   lit.ownership.mark.destroyed %existing
# CHECK: lit.func @"__copyinit__
# CHECK-NEXT:   [[SELF_IMPL_PTR:%.*]] = lit.struct.gep %self[field0]
# CHECK-NEXT:   [[EXISTING_IMPL_PTR:%.*]] = lit.struct.gep %existing[field0]
# CHECK-NEXT:   [[SELF_IMPL:%.*]] = pop.load [[SELF_IMPL_PTR]] : !pop.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   [[EXISTING_IMPL:%.*]] = pop.load [[EXISTING_IMPL_PTR]] : !pop.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   [[COPY_PTR:%.*]] = lit.struct.gep %self[copy]
# CHECK-NEXT:   [[COPY:%.*]] = pop.load [[COPY_PTR]]
# CHECK-NEXT:   kgen.call_signature [[COPY]]([[SELF_IMPL]], [[EXISTING_IMPL]])

# CHECK: lit.func @"makes_escaping_closure({{.*}}::String,{{.*}}::String,{{.*}}::Bool)"
# CHECK-SAME: (%m: !pop.pointer<{{.*}}@String> borrow_in_mem, %z: !pop.pointer<{{.*}}@String> borrow_in_mem, %y: !kgen.declref<{{.*}}@Bool> borrow)
# CHECK-SAME:  -> !kgen.signature<(!pop.pointer<{{.*}}@String> byref_result, !pop.pointer<{{.*}}@String> borrow_in_mem) capturing -> !lit.none> 
# CHECK-SAME: attributes {isParametric, specialFnKind = 0 : i8} {
fn makes_escaping_closure(m: String, z:String, y:Bool) -> fn(String) escaping -> String:
   fn dummy(n:String) escaping -> String:
      return n + m
   fn duplicate(n:String) escaping -> String:
      return n + m
   fn two_effects(n:String) escaping raises -> String:
      return n + m
   fn myclosure(n:String) -> String:
      return n + m
   return myclosure
