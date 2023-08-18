# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -verify-diagnostics -import-mojo -split-input-file | FileCheck %s

##===----------------------------------------------------------------------===##
# Duplicates
##===----------------------------------------------------------------------===##

fn foo1(x:String, y:String, z:Int, u: __mlir_type.index) -> String:
   return x

fn foo2(x:String, y:String, z:Int, u: __mlir_type.index) -> String:
   return y

# CHECK-COUNT-1: lit.struct.decl @"_CI_
fn makes_escaping_closure(m: String, z:String, y:Bool):
   let register_passable_var: Int = 3
   let mlir_type_var: __mlir_type.index = register_passable_var.value
   fn dummy(n:String) escaping -> String:
      return foo1(n,m,register_passable_var, mlir_type_var)
   fn duplicate(n:String) escaping -> String:
      return foo2(n,m,register_passable_var, mlir_type_var)

# // -----

##===----------------------------------------------------------------------===##
# Closure Impl Methods
##===----------------------------------------------------------------------===##

struct String:
   var size: Int
   fn __init__(inout self, sz:Int):
      self.size = sz

   fn __copyinit__(inout self, existing: Self):
      pass

   fn __moveinit__(inout self, owned existing: Self):
      pass

   fn __del__(owned self):
      pass

fn foo1(x:String, y:String, z:Int, u: __mlir_type.index) -> String:
   return x

# CHECK:    lit.struct.decl @"_CI_
# CHECK-NEXT:      lit.struct.field field0 : index
# CHECK-NEXT:      lit.struct.field field1 : !kgen.declref<{{.*}}::@Int>
# CHECK-NEXT:      lit.struct.field field2 : !kgen.declref<{{.*}}::@String>
# CHECK-NEXT:      lit.func @"__del__
# CHECK-NEXT:      [[VAR0:%.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT:      lit.ownership.mark.destroyed %self
# CHECK-NEXT:      lit.return [[VAR0]] : !lit.none
# CHECK-NEXT:      lit.end_func
# CHECK-NEXT:      }

# CHECK-NEXT:    lit.func @"__copyinit__
# CHECK-SAME:    (%self: !pop.pointer<@{{.*}}::@"_CI_{{.*}}"> init_self,
# CHECK-SAME:    %existing: !pop.pointer<@{{.*}}::@"_CI_{{.*}}"> borrow_in_mem) -> !lit.none attributes {specialFnKind = 3 : i8} {
# CHECK-NEXT:    [[V0:%.*]] = lit.struct.gep %self[field0] : <index>
# CHECK-NEXT:    [[V1:%.*]] = lit.struct.gep %existing[field0] : <index>
# CHECK-NEXT:    [[V2:%.*]] = pop.load [[V1]] : !pop.pointer<index>
# CHECK-NEXT:    pop.store [[V2]], [[V0]] : !pop.pointer<index>
# CHECK-NEXT:    [[V3:%.*]] = lit.struct.gep %self[field1] : <{{.*}}::@Int>
# CHECK-NEXT:    [[V4:%.*]] = lit.struct.gep %existing[field1] : <{{.*}}::@Int>
# CHECK-NEXT:    [[V5:%.*]] = pop.load [[V4]] : !pop.pointer<{{.*}}::@Int>
# CHECK-NEXT:    pop.store [[V5]], [[V3]] : !pop.pointer<{{.*}}::@Int>
# CHECK-NEXT:    [[V6:%.*]] = lit.struct.gep %self[field2] : <@{{.*}}::@String>
# CHECK-NEXT:    [[V7:%.*]] = lit.struct.gep %existing[field2] : <@{{.*}}::@String>
# CHECK-NEXT:    [[V8:%.*]] = kgen.call @{{.*}}::@String::@"__copyinit__
# CHECK-NEXT:    [[V9:%.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT:    lit.return [[V9]] : !lit.none
# CHECK-NEXT:    lit.end_func
# CHECK-NEXT:    }

# CHECK-NEXT:      lit.func @"__moveinit__
# CHECK-SAME:      (%self: !pop.pointer<@"{{.*}}"::@"_CI_{{.*}}"> init_self, %existing:
# CHECK-SAME:      !pop.pointer<@"{{.*}}"::@"_CI_{{.*}}"> owned_in_mem) -> !lit.none attributes {specialFnKind = 4 : i8} {
# CHECK-NEXT:        [[W0:%.*]] = lit.struct.gep %self[field0] : <index>
# CHECK-NEXT:        [[W1:%.*]] = lit.struct.gep %existing[field0] : <index>
# CHECK-NEXT:        [[W2:%.*]] = lit.load.consume %1 : !pop.pointer<index>
# CHECK-NEXT:        pop.store [[W2]], [[W0]] : !pop.pointer<index>
# CHECK-NEXT:        [[W3:%.*]] = lit.struct.gep %self[field1]
# CHECK-NEXT:        [[W4:%.*]] = lit.struct.gep %existing[field1]
# CHECK-NEXT:        [[W5:%.*]] = lit.load.consume [[W4]]
# CHECK-NEXT:        pop.store [[W5]], [[W3]]
# CHECK-NEXT:        [[W6:%.*]] = lit.struct.gep %self[field2]
# CHECK-NEXT:        [[W7:%.*]] = lit.struct.gep %existing[field2]
# CHECK-NEXT:        [[W8:%.*]] = kgen.call @{{.*}}__moveinit__{{.*}}"([[W6]], [[W7]])
# CHECK-NEXT:        [[W9:%.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT:        lit.ownership.mark.destroyed %existing
# CHECK-NEXT:        lit.return %9 : !lit.none
# CHECK-NEXT:        lit.end_func
# CHECK-NEXT: }

# CHECK-NEXT: lit.func @"__init__
# CHECK-NEXT: [[Q0:%.*]] = lit.struct.gep %self[field0] : <index>
# CHECK-NEXT: pop.store %field0, [[Q0]] : !pop.pointer<index>
# CHECK-NEXT: [[Q1:%.*]] = lit.struct.gep %self[field1]
# CHECK-NEXT: pop.store %field1, [[Q1]]
# CHECK-NEXT: [[Q2:%.*]] = lit.struct.gep %self[field2]
# CHECK-NEXT: [[Q3:%.*]] = kgen.call @{{.*}}@String::@"__moveinit__{{.*}}"([[Q2]], %field2)
# CHECK-NEXT: [[Q4:%.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT: lit.return [[Q4]] : !lit.none
# CHECK-NEXT: lit.end_func
fn makes_escaping_closure(m: String, z:String, y:Bool):
   let register_passable_var: Int = 3
   let mlir_type_var: __mlir_type.index = register_passable_var.value
   fn dummy(n:String) escaping -> String:
      return foo1(n,m,register_passable_var, mlir_type_var)

# // -----

##===----------------------------------------------------------------------===##
# Nested Function Signature Multiple Effects
##===----------------------------------------------------------------------===##

# CHECK:    lit.struct.decl @"_CI_{{.*}}throws"
# CHECK-NEXT:      lit.struct.field field0 : !kgen.declref<@{{.*}}::@String>
fn makes_escaping_closure(m: String):
   fn two_effects(n:String) escaping raises -> String:
      return n + m

# // -----

##===----------------------------------------------------------------------===##
# Escaping Return Type
##===----------------------------------------------------------------------===##

# CHECK: lit.struct.decl @"_CW_{{.*}}(,{{.*}}::String)\22"
# CHECK-NEXT:     lit.struct.field field0 : !pop.pointer<array<0, i1>>
# CHECK-NEXT:     lit.struct.field dtor : !kgen.signature<(!pop.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:     lit.struct.field copy : !kgen.signature<(!pop.pointer<array<0, i1>> init_self, !pop.pointer<array<0, i1>> borrow_in_mem) -> !lit.none>
# CHECK-NEXT:     lit.struct.field move : !kgen.signature<(!pop.pointer<array<0, i1>> init_self, !pop.pointer<array<0, i1>> owned_in_mem) -> !lit.none>
# CHECK-NEXT: lit.func @"__del__
# CHECK-NEXT:   [[DTOR_PTR:%.*]] = lit.struct.gep %self[dtor]
# CHECK-NEXT:   [[DTOR:%.*]] = pop.load [[DTOR_PTR]]
# CHECK-NEXT:   [[IMPL_PTR:%.*]] = lit.struct.gep %self[field0]
# CHECK-NEXT:   [[IMPL:%.*]] = pop.load [[IMPL_PTR]]
# CHECK-NEXT:   kgen.call_signature [[DTOR]]([[IMPL]])
# CHECK-NEXT:   kgen.param.constant
# CHECK-NEXT:   lit.ownership.mark.destroyed %self
# CHECK: lit.func @"__copyinit__
# CHECK-NEXT:   [[SELF_IMPL_PTR:%.*]] = lit.struct.gep %self[field0]
# CHECK-NEXT:   [[EXISTING_IMPL_PTR:%.*]] = lit.struct.gep %existing[field0]
# CHECK-NEXT:   [[SELF_IMPL:%.*]] = pop.load [[SELF_IMPL_PTR]] : !pop.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   [[EXISTING_IMPL:%.*]] = pop.load [[EXISTING_IMPL_PTR]] : !pop.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   [[COPY_PTR:%.*]] = lit.struct.gep %self[copy]
# CHECK-NEXT:   [[COPY:%.*]] = pop.load [[COPY_PTR]]
# CHECK-NEXT:   kgen.call_signature [[COPY]]([[SELF_IMPL]], [[EXISTING_IMPL]])
# CHECK: lit.func @"__moveinit__
# CHECK-NEXT:   [[SELF_IMPL_PTR:%.*]] = lit.struct.gep %self[field0]
# CHECK-NEXT:   [[EXISTING_IMPL_PTR:%.*]] = lit.struct.gep %existing[field0]
# CHECK-NEXT:   [[SELF_IMPL:%.*]] = pop.load [[SELF_IMPL_PTR]]
# CHECK-NEXT:   [[EXISTING_IMPL:%.*]] = pop.load [[EXISTING_IMPL_PTR]]
# CHECK-NEXT:   [[MOVE_PTR:%.*]] = lit.struct.gep %self[move]
# CHECK-NEXT:   [[MOVE:%.*]] = pop.load [[MOVE_PTR]]
# CHECK-NEXT:   kgen.call_signature [[MOVE]]([[SELF_IMPL]], [[EXISTING_IMPL]])
# CHECK-NEXT:   kgen.param.constant
# CHECK-NEXT:   lit.ownership.mark.destroyed %existing

# CHECK: lit.func @"returns_escaping_closure({{.*}}::String)"
# CHECK-SAME: (%m: !pop.pointer<{{.*}}@String> borrow_in_mem) -> !kgen.signature<(!pop.pointer<@{{.*}}::@String> byref_result, !pop.pointer<@{{.*}}::@String> borrow_in_mem) capturing -> !lit.none>
fn returns_escaping_closure(m: String) -> fn(String) escaping -> String:
   fn myclosure(n:String) -> String:
      return n + m
   return myclosure

# // -----

##===----------------------------------------------------------------------===##
# Captures With No Move
##===----------------------------------------------------------------------===##

struct StringNoMove:
   var size: Int
   fn __init__(inout self, sz:Int):
      self.size = sz

   fn __copyinit__(inout self, existing: Self):
      pass

   fn __del__(owned self):
      pass

   fn __add__(self, existing: Self) -> Int:
      return 42

# CHECK: lit.struct.decl @"_CI_{{.*}}_\22({{.*}}::StringNoMove)\22"
# CHECK: lit.struct.field field0 : !kgen.declref<{{.*}}::@StringNoMove>
# CHECK: lit.func @"__del__({{.*}}::_CI_{{.*}}_\22({{.*}}::StringNoMove)\22)"
# CHECK: lit.func @"__copyinit__(${{.*}}::_CI_${{.*}}_\22(${{.*}}::StringNoMove)\22=&,${{.*}}::_CI_${{.*}}_\22(${{.*}}::StringNoMove)\22)"
# CHECK: lit.func @"__moveinit__(${{.*}}::_CI_${{.*}}_\22(${{.*}}::StringNoMove)
fn makes_escaping_closure_from_nomove(m: StringNoMove) -> Int:
   fn foo() escaping -> Int:
      return m + m
   return 43

# // -----

##===----------------------------------------------------------------------===##
# Closure Wrapper Initializer
##===----------------------------------------------------------------------===##

# CHECK: lit.func @"__init__{{.*}}"(%self: !pop.pointer<@{{.*}}::@"_CW_{{.*}}"> init_self, %impl: !pop.pointer<@{{.*}}::@"_CI_{{.*}}"> borrow_in_mem) -> !lit.none attributes {specialFnKind = 2 : i8} {

# CHECK-NEXT: %[[V7:.*]] = lit.struct.gep %self[move]
# CHECK-NEXT: %[[V8:.*]] = kgen.create_closure [{{.*}}]()
# CHECK-NEXT: pop.store %[[V8]], %[[V7]]

# CHECK-NEXT: %[[V9:.*]] = lit.struct.gep %self[copy]
# CHECK-NEXT: %[[V10:.*]] = kgen.create_closure [{{.*}}]()
# CHECK-NEXT: pop.store %[[V10]], %[[V9]]

# CHECK-NEXT: %[[V5:.*]] = lit.struct.gep %self[dtor]
# CHECK-NEXT: %[[V6:.*]] = kgen.create_closure [{{.*}}]()
# CHECK-NEXT: pop.store %[[V6]], %[[V5]] : !pop.pointer<!kgen.signature<(!pop.pointer<array<0, i1>>) -> !lit.none>>

# Allocate memory on heap
# CHECK-NEXT:  %index = kgen.param.constant = <get_sizeof(!kgen.declref<@[[CI_TYPE:.*]]>, current_target())>
# CHECK-NEXT:  %index_0 = kgen.param.constant = <get_alignof(!kgen.declref<@[[CI_TYPE]]>, current_target())>
# CHECK-NEXT:  %[[V0:.*]] = pop.aligned_alloc %index_0, %index : <@[[CI_TYPE]]>

# Copy source (stack) into target (heap)
# CHECK-NEXT:  %[[V1:.*]] = kgen.call @[[CI_TYPE]]::@"__moveinit__{{.*}}(%[[V0]], %impl)

# Store heap pointer in ClosureWrapper field
# CHECK-NEXT:  %[[V2:.*]] = lit.struct.gep %self[field0] : <pointer<array<0, i1>>> from <@{{.*}}::@"_CW_{{.*}}">
# CHECK-NEXT:  %[[V3:.*]] = pop.pointer.bitcast %[[V0]] : !pop.pointer<@[[CI_TYPE]]> to !pop.pointer<array<0, i1>>
# CHECK-NEXT:  pop.store %[[V3]], %[[V2]] : !pop.pointer<pointer<array<0, i1>>>

# CHECK-NEXT:  %[[V4:.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT:  lit.return %[[V4]] : !lit.none
# CHECK-NEXT:  lit.end_func
# CHECK-NEXT:  }

# CHECK: lit.func @"__init__{{.*}}"(%self: !pop.pointer<@{{.*}}::@"_CW_{{.*}}"> init_self, %impl: !pop.pointer<@{{.*}}::@"_CI_${{.*}}

# CHECK: lit.func @"_CW_{{.*}}_dtor__CI_{{.*}}"(%self: !pop.pointer<array<0, i1>>) -> !lit.none
# CHECK-NEXT: %0 = pop.pointer.bitcast %self
# CHECK-NEXT: pop.aligned_free %0

# CHECK: lit.func @"_CW_{{.*}}_copyinit__CI_{{.*}}"(%self: !pop.pointer<array<0, i1>> init_self, %existing: !pop.pointer<array<0, i1>> borrow_in_mem) -> !lit.none
# CHECK-NEXT: %[[W0:.*]]  = pop.pointer.bitcast %self
# CHECK-NEXT: %[[W1:.*]] = pop.pointer.bitcast %existing
# CHECK-NEXT: %[[W2:.*]]  = kgen.call @{{.*}}__copyinit__{{.*}}(%[[W0]], %[[W1]])

# CHECK: lit.func @"_CW_{{.*}}_moveinit__CI_{{.*}}"(%self: !pop.pointer<array<0, i1>> init_self, %existing: !pop.pointer<array<0, i1>> owned_in_mem) -> !lit.none
# CHECK-NEXT: %[[W0:.*]]  = pop.pointer.bitcast %self
# CHECK-NEXT: %[[W1:.*]] = pop.pointer.bitcast %existing
# CHECK-NEXT: %[[W2:.*]]  = kgen.call @{{.*}}__moveinit__{{.*}}(%[[W0]], %[[W1]])
fn materialize_escaping_closure(m: String, z:String):
   fn dummy(n:String) escaping -> String:
      return n + m + z
   fn dupe(n: String) escaping -> String:
      return z + n + m
   fn unique(n: String) escaping -> String:
      return m + n
   let x = dummy
