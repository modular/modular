# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -verify-diagnostics -import-mojo -split-input-file | FileCheck %s

##===----------------------------------------------------------------------===##
# Duplicates
##===----------------------------------------------------------------------===##

# CHECK-LABEL: module {

fn foo1(x:String, y:String, z:Int, u: __mlir_type.index) -> String:
   return x

fn foo2(x:String, y:String, z:Int, u: __mlir_type.index) -> String:
   return y

# CHECK-COUNT-1: lit.struct.decl @"_CI_

# CHECK-LABEL: lit.func @"makes_escaping_closure
fn makes_escaping_closure(m: String, z:String, y:Bool):
   let register_passable_var: Int = 3
   let mlir_type_var: __mlir_type.index = register_passable_var.value
   fn dummy(n:String) escaping -> String:
      return foo1(n,m,register_passable_var, mlir_type_var)
   fn duplicate(n:String) escaping -> String:
      return foo2(n,m,register_passable_var, mlir_type_var)

# // -----

# CHECK-LABEL: module {

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
# CHECK-NEXT:      lit.struct.field field0 : !String
# CHECK-NEXT:      lit.struct.field field1 : !Int
# CHECK-NEXT:      lit.struct.field field2 : index
# CHECK-NEXT:      lit.func @"__del__
# CHECK-NEXT:      [[VAR0:%.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT:      lit.ownership.mark.destroyed %self
# CHECK-NEXT:      lit.return [[VAR0]] : !lit.none
# CHECK-NEXT:      lit.end_func
# CHECK-NEXT:      }

# CHECK-NEXT:    lit.func @"__copyinit__
# CHECK-SAME:    (%self: !kgen.pointer<{{.*}}> init_self,
# CHECK-SAME:    %existing: !kgen.pointer<{{.*}}> borrow_in_mem) -> !lit.none attributes {specialFnKind = 3 : i8} {
# CHECK-NEXT:    [[V0:%.*]] = lit.struct.gep %self[field0] : <!String>
# CHECK-NEXT:    [[V1:%.*]] = lit.struct.gep %existing[field0] : <!String>
# CHECK-NEXT:    [[V2:%.*]] = kgen.call @{{.*}}__copyinit__{{.*}}"([[V0]], [[V1]])
# CHECK-NEXT:    [[V3:%.*]] = lit.struct.gep %self[field1] : <!Int>
# CHECK-NEXT:    [[V4:%.*]] = lit.struct.gep %existing[field1] : <!Int>
# CHECK-NEXT:    [[V5:%.*]] = pop.load [[V4]] : !kgen.pointer<!Int>
# CHECK-NEXT:    pop.store [[V5]], [[V3]] : !kgen.pointer<!Int>
# CHECK-NEXT:    [[V6:%.*]] = lit.struct.gep %self[field2] : <index>
# CHECK-NEXT:    [[V7:%.*]] = lit.struct.gep %existing[field2] : <index>
# CHECK-NEXT:    [[V8:%.*]] = pop.load [[V7]] : !kgen.pointer<index>
# CHECK-NEXT:    pop.store [[V8]], [[V6]] : !kgen.pointer<index>
# CHECK-NEXT:    [[V9:%.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT:    lit.return [[V9]] : !lit.none
# CHECK-NEXT:    lit.end_func
# CHECK-NEXT:    }

# CHECK-NEXT:      lit.func @"__moveinit__
# CHECK-SAME:      (%self: !kgen.pointer<{{.*}}> init_self, %existing:
# CHECK-SAME:      !kgen.pointer<{{.*}}> owned_in_mem) -> !lit.none attributes {specialFnKind = 4 : i8} {
# CHECK-NEXT:      [[W0:%.*]] = lit.struct.gep %self[field0] : <!String>
# CHECK-NEXT:      [[W1:%.*]] = lit.struct.gep %existing[field0] : <!String>
# CHECK-NEXT:      [[W2:%.*]] = kgen.call @{{.*}}__moveinit__{{.*}}"([[W0]], [[W1]])
# CHECK-NEXT:      [[W3:%.*]] = lit.struct.gep %self[field1] : <!Int>
# CHECK-NEXT:      [[W4:%.*]] = lit.struct.gep %existing[field1] : <!Int>
# CHECK-NEXT:      [[W5:%.*]] = lit.load.consume [[W4]] : !kgen.pointer<!Int>
# CHECK-NEXT:      pop.store [[W5]], [[W3]] : !kgen.pointer<!Int>
# CHECK-NEXT:      [[W6:%.*]] = lit.struct.gep %self[field2] : <index>
# CHECK-NEXT:      [[W7:%.*]] = lit.struct.gep %existing[field2] : <index>
# CHECK-NEXT:      [[W8:%.*]] = lit.load.consume [[W7]] : !kgen.pointer<index>
# CHECK-NEXT:      pop.store [[W8]], [[W6]] : !kgen.pointer<index>
# CHECK-NEXT:      [[W9:%.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT:      lit.ownership.mark.destroyed %existing
# CHECK-NEXT:      lit.return %9 : !lit.none
# CHECK-NEXT:      lit.end_func
# CHECK-NEXT: }

# CHECK-NEXT: lit.func @"__init__
# CHECK-NEXT: [[Q0:%.*]] = lit.struct.gep %self[field0] : <!String>
# CHECK-NEXT: [[Q1:%.*]] = kgen.call @{{.*}}::@"__moveinit__{{.*}}"([[Q0]], %field0)
# CHECK-NEXT: [[Q2:%.*]] = lit.struct.gep %self[field1] : <!Int>
# CHECK-NEXT: pop.store %field1, [[Q2]] : !kgen.pointer<!Int>
# CHECK-NEXT: [[Q3:%.*]] = lit.struct.gep %self[field2] : <index>
# CHECK-NEXT: pop.store %field2, [[Q3]] : !kgen.pointer<index>
# CHECK-NEXT: [[Q4:%.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT: lit.return [[Q4]] : !lit.none
# CHECK-NEXT: lit.end_func

# CHECK-LABEL: lit.func @"makes_escaping_closure
fn makes_escaping_closure(m: String, z:String, y:Bool):
   let register_passable_var: Int = 3
   let mlir_type_var: __mlir_type.index = register_passable_var.value
   fn dummy(n:String) escaping -> String:
      return foo1(n,m,register_passable_var, mlir_type_var)

# // -----

# CHECK-LABEL: module {

##===----------------------------------------------------------------------===##
# Nested Function Signature Multiple Effects
##===----------------------------------------------------------------------===##

# CHECK:    lit.struct.decl @"_CI_{{.*}}throws|escaping
# CHECK-NEXT:      lit.struct.field field0 : !String

# CHECK: lit.struct.decl @"_CW_{{.*}}::String)

# CHECK-LABEL: lit.func @"makes_escaping_closure
fn makes_escaping_closure(m: String):
   fn two_effects(n:String) escaping raises -> String:
      return n + m

# // -----

# CHECK-LABEL: module {

##===----------------------------------------------------------------------===##
# Escaping Return Type
##===----------------------------------------------------------------------===##

# CHECK: lit.struct.decl @"_CW_{{.*}}::String)
# CHECK-NEXT:     lit.struct.field field0 : !kgen.pointer<array<0, i1>>
# CHECK-NEXT:     lit.struct.field dtor : !kgen.signature<("self": !kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:     lit.struct.field copy : !kgen.signature<("ptrToImpl": !kgen.pointer<pointer<array<0, i1>>> borrow, "other": !kgen.pointer<array<0, i1>> borrow_in_mem) -> !lit.none>
# CHECK-NEXT:     lit.struct.field call : !kgen.signature<("__result__": !kgen.pointer<!String> byref_result, "self": !kgen.pointer<array<0, i1>> borrow_in_mem, "n": !kgen.pointer<!String> borrow_in_mem) -> !lit.none>
# CHECK-NEXT: lit.func @"__del__
# CHECK-NEXT: [[PTR_TO_IMPL:%.*]] = lit.struct.gep %self[field0] : <pointer<array<0, i1>>>
# CHECK-NEXT: [[OPAQUE_IMPL:%.*]] = pop.load [[PTR_TO_IMPL]] : !kgen.pointer<pointer<array<0, i1>>>
# CHECK-NEXT: %index0 = kgen.param.constant = <0>
# CHECK-NEXT: [[SCALAR_IMPL:%.*]] = pop.pointer_to_index [[OPAQUE_IMPL]] : !kgen.pointer<array<0, i1>> to !pop.scalar<index>
# CHECK-NEXT: [[INDEX_IMPL:%.*]] = pop.cast_to_builtin [[SCALAR_IMPL]] : !pop.scalar<index> to index
# CHECK-NEXT: [[IS_NULL:%.*]] = index.cmp eq([[INDEX_IMPL]], %index0)
# CHECK-NEXT: hlcf.if [[IS_NULL]] {
# CHECK-NEXT: kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT: lit.ownership.mark.destroyed %self
# CHECK-NEXT: lit.return
# CHECK-NEXT: hlcf.yield
# CHECK-NEXT: } else {
# CHECK-NEXT: hlcf.yield
# CHECK-NEXT: }
# CHECK-NEXT: [[DTOR_PTR:%.*]] = lit.struct.gep %self[dtor] : <("self": !kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT: [[DTOR:%.*]] = pop.load [[DTOR_PTR]] : !kgen.pointer<("self": !kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT: kgen.call_signature [[DTOR]]([[OPAQUE_IMPL]]) : ("self": !kgen.pointer<array<0, i1>>) -> !lit.none
# CHECK-NEXT: kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT: lit.ownership.mark.destroyed %self
# CHECK-NEXT: lit.return %8 : !lit.none
# CHECK-NEXT: lit.end_func

# CHECK: lit.func @"__copyinit__
# CHECK-NEXT:   [[P0:%.*]] = lit.struct.gep %self[field0] : <pointer<array<0, i1>>>
# CHECK-NEXT:   [[existing_impl:%.*]] = lit.struct.gep %existing[field0] : <pointer<array<0, i1>>>
# CHECK-NEXT:   [[loaded_existing_impl:%.*]] = pop.load [[existing_impl]] : !kgen.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   pop.store [[loaded_existing_impl]], [[P0]] : !kgen.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   [[P1:%.*]] = lit.struct.gep %self[dtor] : <(!kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:   [[P2:%.*]] = lit.struct.gep %existing[dtor] : <(!kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:   [[P3:%.*]] = pop.load [[P2]] : !kgen.pointer<(!kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:   pop.store [[P3]], [[P1]] : !kgen.pointer<(!kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:   [[P4:%.*]] = lit.struct.gep %self[copy]
# CHECK-NEXT:   [[P5:%.*]] = lit.struct.gep %existing[copy]
# CHECK-NEXT:   [[P6:%.*]] = pop.load [[P5]]
# CHECK-NEXT:   pop.store [[P6]], [[P4]]
# CHECK-NEXT:   [[P7:%.*]] = lit.struct.gep %self[call]
# CHECK-NEXT:   [[P8:%.*]] = lit.struct.gep %existing[call]
# CHECK-NEXT:   [[P9:%.*]] = pop.load [[P8]]
# CHECK-NEXT:   pop.store [[P9]], [[P7]]
# CHECK-NEXT:   kgen.param.constant: !lit.none
# CHECK-NEXT:   [[EXISTING_IMPL_PTR:%.*]] = lit.struct.gep %existing[field0]
# CHECK-NEXT:   [[EXISTING_IMPL:%.*]] = pop.load [[EXISTING_IMPL_PTR]] : !kgen.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   [[COPY_PTR:%.*]] = lit.struct.gep %self[copy]
# CHECK-NEXT:   [[SELF_IMPL_PTR:%.*]] = lit.struct.gep %self[field0] : <pointer<array<0, i1>>>
# CHECK-NEXT:   [[COPY:%.*]] = pop.load [[COPY_PTR]]
# CHECK-NEXT:   kgen.call_signature [[COPY]]([[SELF_IMPL_PTR]], [[EXISTING_IMPL]])
# CHECK: lit.func @"__moveinit__
# CHECK-NEXT:   [[M0:%.*]] = lit.struct.gep %self[field0] : <pointer<array<0, i1>>>
# CHECK-NEXT:   [[mov_existing_impl:%.*]] = lit.struct.gep %existing[field0] : <pointer<array<0, i1>>>
# CHECK-NEXT:   [[mov_loaded_existing_impl:%.*]] = lit.load.consume [[mov_existing_impl]] : !kgen.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   pop.store [[mov_loaded_existing_impl]], [[M0]] : !kgen.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   [[M1:%.*]] = lit.struct.gep %self[dtor] : <(!kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:   [[M2:%.*]] = lit.struct.gep %existing[dtor] : <(!kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:   [[M3:%.*]] = lit.load.consume [[M2]] : !kgen.pointer<(!kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:   pop.store [[M3]], [[M1]] : !kgen.pointer<(!kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:   [[M4:%.*]] = lit.struct.gep %self[copy]
# CHECK-NEXT:   [[M5:%.*]] = lit.struct.gep %existing[copy]
# CHECK-NEXT:   [[M6:%.*]] = lit.load.consume [[M5]]
# CHECK-NEXT:   pop.store [[M6]], [[M4]]
# CHECK-NEXT:   [[M7:%.*]] = lit.struct.gep %self[call]
# CHECK-NEXT:   [[M8:%.*]] = lit.struct.gep %existing[call]
# CHECK-NEXT:   [[M9:%.*]] = lit.load.consume [[M8]]
# CHECK-NEXT:   pop.store [[M9]], [[M7]]
# CHECK-NEXT:   %pointer = kgen.param.constant: pointer<array<0, i1>> = <0>
# CHECK-NEXT:   [[V0:%.*]] = lit.struct.gep %existing[field0] : <pointer<array<0, i1>>>
# CHECK-NEXT:   pop.store %pointer, [[V0]] : !kgen.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   [[V3:%.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT:   lit.ownership.mark.destroyed %existing

# CHECK: lit.func @"returns_escaping_closure({{.*}}::String)"
# CHECK-SAME: (%__result__: !kgen.pointer<{{.*}}> byref_result, %m: !kgen.pointer<!String> borrow_in_mem) -> !lit.none
fn returns_escaping_closure(m: String) -> fn(n:String) escaping -> String:
   fn myclosure(n:String) escaping -> String:
      return n + m
   return myclosure

# // -----

# CHECK-LABEL: module {

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

# CHECK: lit.struct.decl @"_CI_{{.*}}::StringNoMove)
# CHECK: lit.struct.field field0 : !StringNoMove
# CHECK: lit.func @"__del__
# CHECK: lit.func @"__copyinit__
# CHECK: lit.func @"__moveinit__
fn makes_escaping_closure_from_nomove(m: StringNoMove) -> Int:
   fn foo() escaping -> Int:
      return m + m
   return 43

# CHECK-LABEL: lit.struct.decl @StringNoMove

# // -----

# CHECK-LABEL: module {

##===----------------------------------------------------------------------===##
# Closure Wrapper Initializer
##===----------------------------------------------------------------------===##

@value
struct Mem:
  var str: String
  fn __add__(self, other:Mem) -> Mem:
     return Mem(self.str + other.str)

# CHECK: lit.struct.decl @"_CW_

# CHECK: lit.func @"__init__{{.*}}"(%self: !kgen.pointer<!escaping1> init_self, %impl: !kgen.pointer<!escaping> borrow_in_mem)
# CHECK-NEXT: %[[callPtr:.*]] = lit.struct.gep %self[call]
# CHECK-NEXT: %[[ptrToCall:.*]] = kgen.create_closure [("__result__": !kgen.pointer<!Mem> byref_result, "self": !kgen.pointer<array<0, i1>> borrow_in_mem, "n": !kgen.pointer<!Mem> borrow_in_mem) -> !lit.none
# CHECK-NEXT: pop.store %[[ptrToCall]], %[[callPtr]]

# CHECK-NEXT: %[[V5:.*]] = lit.struct.gep %self[dtor]
# CHECK-NEXT: %[[V6:.*]] = kgen.create_closure [{{.*}}]()
# CHECK-NEXT: pop.store %[[V6]], %[[V5]] : !kgen.pointer<("self": !kgen.pointer<array<0, i1>>) -> !lit.none>

# CHECK-NEXT: %[[V9:.*]] = lit.struct.gep %self[copy]
# CHECK-NEXT: %[[V10:.*]] = kgen.create_closure [{{.*}}]()
# CHECK-NEXT: pop.store %[[V10]], %[[V9]]

# Allocate memory on heap
# CHECK-NEXT:  %index = kgen.param.constant = <get_sizeof(!escaping, current_target())>
# CHECK-NEXT:  %index_0 = kgen.param.constant = <get_alignof(!escaping, current_target())>
# CHECK-NEXT:  %[[V0:.*]] = pop.aligned_alloc %index_0, %index : <!escaping>

# Copy source (stack) into target (heap)
# CHECK-NEXT:  %[[V1:.*]] = kgen.call {{.*}}__moveinit__{{.*}}(%[[V0]], %impl)

# Store heap pointer in ClosureWrapper field
# CHECK-NEXT:  %[[V2:.*]] = lit.struct.gep %self[field0]
# CHECK-NEXT:  %[[V3:.*]] = pop.pointer.bitcast %[[V0]] : !kgen.pointer<!escaping> to !kgen.pointer<array<0, i1>>
# CHECK-NEXT:  pop.store %[[V3]], %[[V2]] : !kgen.pointer<pointer<array<0, i1>>>

# CHECK-NEXT:  %[[V4:.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT:  lit.return %[[V4]] : !lit.none
# CHECK-NEXT:  lit.end_func
# CHECK-NEXT:  }

# CHECK: lit.struct.decl @Mem

# CHECK: lit.func @"_CW_{{.*}}_dtor__CI_{{.*}}"(%self: !kgen.pointer<array<0, i1>>) -> !lit.none
# CHECK-NEXT: %0 = pop.pointer.bitcast %self
# CHECK-NEXT: pop.aligned_free %0

# CHECK: lit.func @"_CW_{{.*}}_call__CI_{{.*}}"(%__result__: !kgen.pointer<!Mem> byref_result, %self: !kgen.pointer<array<0, i1>> borrow_in_mem, %n: !kgen.pointer<!Mem> borrow_in_mem) -> !lit.none
# CHECK-NEXT: %[[A0:.*]] = pop.pointer.bitcast %self
# CHECK-NEXT: %[[A1:.*]] = kgen.call @{{.*}}@"__call__{{.*}}"(%__result__, %[[A0]], %n)
# CHECK-NEXT: lit.return %[[A1]] : !lit.none
# CHECK-NEXT: lit.end_func

fn materialize_escaping_closure(m: Mem):
   fn unique(n: Mem) escaping -> Mem:
      return m + n

# // -----

# CHECK-LABEL: module {

##===----------------------------------------------------------------------===##
# Pointer Captures
##===----------------------------------------------------------------------===##

# CHECK: lit.struct.field field0 : !kgen.pointer<!String>
# CHECK: lit.struct.field field1 : !kgen.pointer<!String>
# CHECK: lit.struct.field field2 : !kgen.pointer<!Int>
# CHECK: lit.struct.field field3 : !kgen.pointer<index>
# CHECK: lit.struct.field field4 : !String
# CHECK: lit.struct.field field5 : !String
# CHECK: lit.struct.field field6 : !Int

# CHECK: lit.func @"__init__({{.*}}_CI_{{.*}} init_self,
# CHECK-SAME: %field0: !kgen.pointer<!String>, %field1: !kgen.pointer<!String>, %field2: !kgen.pointer<!Int>,
# CHECK-SAME: %field3: !kgen.pointer<index>, %field4: !kgen.pointer<!String> owned_in_mem, %field5: !kgen.pointer<!String> owned_in_mem
fn doNothing(x:__mlir_type[`!kgen.pointer<`, String, `>`], y:__mlir_type[`!kgen.pointer<`, String, `>`]):
   pass

fn doNothingAgain(x:__mlir_type[`!kgen.pointer<`, Int, `>`], y:__mlir_type.`!kgen.pointer<index>`, w:String, local:String, size:Int):
   pass

# CHECK-LABEL: lit.func @"makes_escaping_closure
fn makes_escaping_closure(x: __mlir_type[`!kgen.pointer<`, Int, `>`],
                          y: __mlir_type.`!kgen.pointer<index>`,
                          z: __mlir_type[`!kgen.pointer<`, String, `>`],
                          w: String):
   let local:String = "1234"
   let alignment = 0
   let size = 1
   let local_ptr: __mlir_type[`!kgen.pointer<`, String, `>`] = __mlir_op.`pop.aligned_alloc`[_type : __mlir_type[`!kgen.pointer<`, String, `>`]](alignment.value, size.value)
   fn do_stuff_with_pointers() escaping -> NoneType:
      doNothing(local_ptr, z)
      doNothingAgain(x, y, local, w, size)

# // -----

# CHECK-LABEL: module {

##===----------------------------------------------------------------------===##
# Owned/ByRef Captures
##===----------------------------------------------------------------------===##

fn foo(x:Int, y:String, z: String):
   pass

# CHECK: lit.struct.field field0 : !Int
# CHECK: lit.struct.field field1 : !String
# CHECK: lit.struct.field field2 : !String
# CHECK: lit.func @"__init__{{.*}}"(%self: !kgen.pointer<{{.*}}> init_self, %field0: !Int, %field1: !kgen.pointer<!String> owned_in_mem, %field2: !kgen.pointer<!String> owned_in_mem)

# CHECK-LABEL: lit.func @"makes_escaping_closure_3
fn makes_escaping_closure_3(owned x: Int,
                          owned y: String,
                          inout z: String):
   fn take_owned_and_escape() escaping -> NoneType:
      foo(x, y, z)

# // -----

# CHECK-LABEL: module {

##===----------------------------------------------------------------------===##
# Multiple References
##===----------------------------------------------------------------------===##

# CHECK: lit.struct.decl @"_CI_
# CHECK-NEXT: lit.struct.field field0 : !Int
# CHECK-NEXT: lit.func @"__copyinit__
fn foo():
   let w = 5
   fn bar() escaping -> Int:
      let x = w + w
      return x

# // -----

# CHECK-LABEL: module {

##===----------------------------------------------------------------------===##
# SLValues
##===----------------------------------------------------------------------===##

fn make_pointer() -> __mlir_type.`!kgen.pointer<index>`:
   let alignment = 0
   let size = 8
   return __mlir_op.`pop.aligned_alloc`[
           _type : __mlir_type.`!kgen.pointer<index>`
       ](alignment.value, size.value)

# CHECK: lit.struct.decl @"_CI_
# CHECK-NEXT: lit.struct.field field0 : !kgen.pointer<index>
# CHECK-NEXT: lit.struct.field field1 : !kgen.pointer<index>
# CHECK-NEXT: lit.struct.field field2 : !Int
# CHECK-NEXT: lit.struct.field field3 : !Int

# CHECK: (%self: !kgen.pointer<{{.*}}> init_self, %field0: !kgen.pointer<index>, %field1: !kgen.pointer<index>, %field2: !Int, %field3: !Int)
fn foo(owned y:Int):
  var w = 5
  var q = make_pointer()
  let u = make_pointer()
  fn bar() escaping -> Int:
     __mlir_op.`pop.aligned_free`(q)
     __mlir_op.`pop.aligned_free`(u)
     y = y + 1
     w = w + 1
     return w

# // -----

# CHECK-LABEL: module {

##===----------------------------------------------------------------------===##
# Closure Impl Call
##===----------------------------------------------------------------------===##

# CHECK: lit.func @"__call__({{.*}}_CI_{{.*}})"(%self: !kgen.pointer<{{.*}}> borrow_in_mem, %q: !Int, %ww: !Int borrow) -> !lit.none
# CHECK-NEXT: %[[V0:.*]] = lit.struct.gep %self[field0] : <!Int>
# CHECK-NEXT: %[[V1:.*]] = lit.struct.gep %self[field1] : <!Int>
# CHECK-NEXT: %q_0 = lit.varlet.decl "q" var synth : <!Int>
# CHECK-NEXT: pop.store %q, %q_0 : !kgen.pointer<!Int>
# CHECK-NEXT: %[[V2:.*]] = pop.load %[[V0]] : !kgen.pointer<!Int>
# CHECK-NEXT: %[[V3:.*]] = pop.load %[[V0]] : !kgen.pointer<!Int>
# CHECK-NEXT: %[[V4:.*]] = kgen.call @{{.*}}::@Int::@"__add__{{.*}}"(%[[V2]], %[[V3]]) : ("self": !Int borrow, "rhs": !Int borrow) -> !Int
# CHECK-NEXT: pop.store %[[V4]], %[[V0]] : !kgen.pointer<!Int>
# CHECK-NEXT: %[[V5:.*]] = pop.load %[[V1]] : !kgen.pointer<!Int>
# CHECK-NEXT: %[[V6:.*]] = kgen.call @{{.*}}@"print{{.*}}"(%[[V5]]) : ("x": !Int borrow) -> !lit.none
# CHECK-NEXT: %[[V7:.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT: lit.return %[[V7]] : !lit.none
# CHECK-NEXT: lit.end_func

# CHECK: lit.func @"__call__({{.*}}_CI_{{.*}})"(%self: !kgen.pointer<{{.*}}> borrow_in_mem, %p: !Int borrow) -> !lit.none
# CHECK-NEXT: %[[W0:.*]] = lit.struct.gep %self[field0] : <!String>
# CHECK-NEXT: %[[W1:.*]] = kgen.call @{{.*}}::@"print{{.*}}"(%[[W0]])
# CHECK-NEXT: %[[W2:.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT: lit.return %[[W2]] : !lit.none
# CHECK-NEXT: lit.end_func

# CHECK: lit.func @"__call__({{.*}}_CI_{{.*}})"(%self: !kgen.pointer<{{.*}}> borrow_in_mem) -> index
# CHECK-NEXT: %[[W0:.*]] = lit.struct.gep %self[field0] : <index>
# CHECK-NEXT: %[[W1:.*]] = pop.load %[[W0]] : !kgen.pointer<index>
# CHECK-NEXT: %[[W2:.*]] = lit.struct.gep %self[field1] : <!Int>
# CHECK-NEXT: %[[W3:.*]] = lit.struct.gep %[[W2]][value] : <index> from <!Int>
# CHECK-NEXT: %[[W4:.*]] = pop.load %[[W3]] : !kgen.pointer<index>
# CHECK-NEXT: %[[W5:.*]] = index.mul %[[W1]], %[[W4]]
# CHECK-NEXT: lit.return %[[W5]] : index
# CHECK-NEXT: lit.end_func

# CHECK: lit.func @"__call__({{.*}}_CI_{{.*}}"(%__result__: !kgen.pointer<!String> byref_result, %self: !kgen.pointer<{{.*}}> borrow_in_mem, %y: !kgen.pointer<!String> borrow_in_mem) -> !lit.none
# CHECK-NEXT: %[[W0:.*]] = lit.struct.gep %self[field0] : <!String>
# CHECK-NEXT: %__call_result_tmp__ = lit.varlet.decl "__call_result_tmp__" var synth : <!String>
# CHECK-NEXT: %[[W2:.*]] = kgen.call @{{.*}}__add__{{.*}}(%__call_result_tmp__, %[[W0]], %y)
# CHECK-NEXT: %[[W3:.*]] = kgen.call @{{.*}}__copyinit__{{.*}}(%__result__, %__call_result_tmp__)
# CHECK-NEXT: %[[W4:.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT: lit.return %[[W4]] : !lit.none
# CHECK-NEXT: lit.end_func
fn make_diff_closures(m:String, z: __mlir_type.index, owned w: Int):
   var x = w
   fn ret_mem(y: String) escaping -> String:
      return m + y
   fn ret_mlir_type() escaping -> __mlir_type.index:
      return __mlir_op.`index.mul`(z, w.value)
   fn ret_none(p: Int) escaping -> NoneType:
      print(m)
   fn capture_slvalue(owned q: Int, ww: Int) escaping -> NoneType:
      x = x + x
      print(w)

# // -----

# CHECK-LABEL: module {

##===----------------------------------------------------------------------===##
# Closure Wrapper Call
##===----------------------------------------------------------------------===##

# CHECK: lit.struct.field call : !kgen.signature<("self": !kgen.pointer<array<0, i1>> borrow_in_mem, "n": !kgen.pointer<!String> borrow_in_mem, "j": !Int borrow) -> !Int>
# CHECK: lit.func @"__call__{{.*}}"(%self: !kgen.pointer<{{.*}}> borrow_in_mem, %n: !kgen.pointer<!String> borrow_in_mem, %j: !Int borrow) -> !Int
# CHECK-NEXT: [[closure_impl_ref0:%.*]] = lit.struct.gep %self[field0]
# CHECK-NEXT: [[closure_impl0:%.*]] = pop.load [[closure_impl_ref0]] : !kgen.pointer<pointer<array<0, i1>>>
# CHECK-NEXT: [[casting_call_ref0:%.*]] = lit.struct.gep %self[call]
# CHECK-NEXT: [[casting_call0:%.*]] = pop.load [[casting_call_ref0]]
# CHECK-NEXT: [[result_of_typed_call0:%.*]] = kgen.call_signature [[casting_call0]]([[closure_impl0]], %n, %j)
# CHECK-NEXT: lit.return [[result_of_typed_call0]] : !Int
# CHECK-NEXT: lit.end_func
# CHECK-NEXT: }
# CHECK: lit.struct.field call : !kgen.signature<("__result__": !kgen.pointer<!String> byref_result, "self": !kgen.pointer<array<0, i1>> borrow_in_mem, "n": !kgen.pointer<!String> borrow_in_mem) -> !lit.none>
# CHECK: lit.func @"__call__{{.*}}"(%__result__: !kgen.pointer<!String> byref_result, %self: !kgen.pointer<{{.*}}> borrow_in_mem, %n: !kgen.pointer<!String> borrow_in_mem) -> !lit.none
# CHECK-NEXT: [[closure_impl_ref:%.*]] = lit.struct.gep %self[field0]
# CHECK-NEXT: [[closure_impl:%.*]] = pop.load [[closure_impl_ref]] : !kgen.pointer<pointer<array<0, i1>>>
# CHECK-NEXT: [[casting_call_ref:%.*]] = lit.struct.gep %self[call]
# CHECK-NEXT: [[casting_call:%.*]] = pop.load [[casting_call_ref]]
# CHECK-NEXT: [[result_of_typed_call:%.*]] = kgen.call_signature [[casting_call]](%__result__, [[closure_impl]], %n)
# CHECK-NEXT: lit.return [[result_of_typed_call]] : !lit.none

# CHECK-LABEL: lit.func @"makes_escaping_closure
fn makes_escaping_closure(m: String):
   fn myclosure(n:String) escaping -> String:
      return n+m
   fn myclosure2(n:String, j:Int) escaping -> Int:
      return m.__len__()

# // -----

# CHECK-LABEL: module {

##===----------------------------------------------------------------------===##
# Closure Impl Instantiation
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"makes_escaping_closure
fn makes_escaping_closure(m: String):
   # CHECK: %anonymous2A = lit.varlet.decl "anonymous*" var synth
   # CHECK-NEXT: %anonymous2A_0 = lit.varlet.decl "anonymous*" var synth : <!String>
   # CHECK-NEXT: kgen.call @"{{.*}}@"__copyinit__{{.*}}"(%anonymous2A_0, %m)
   # CHECK-NEXT: kgen.call @{{.*}}::@"__init__{{.*}}"(%anonymous2A, %anonymous2A_0)
   # CHECK-NEXT: %anonymous2A_1 = lit.varlet.decl "anonymous*" var synth
   # CHECK-NEXT: kgen.call @{{.*}}::@"__init__{{.*}}"(%anonymous2A_1, %anonymous2A)
   fn myclosure_with_mem_types(n:String) escaping -> String:
      return n+m

# // -----

# CHECK-LABEL: module {

# CHECK-LABEL: lit.func @"makes_escaping_closure
fn makes_escaping_closure(z: Int):
   let w = z * z
   var a = w
   # CHECK: %anonymous2A = lit.varlet.decl "anonymous*" var synth
   # CHECK-NEXT: %[[A:.*]] = pop.load %a : !kgen.pointer<!Int>
   # CHECK-NEXT: kgen.call @{{.*}}::@"__init__{{.*}}"(%anonymous2A, %[[A]], %w)
   # CHECK-NEXT: %anonymous2A_0 = lit.varlet.decl "anonymous*" var synth
   # CHECK-NEXT: kgen.call @{{.*}}::@"__init__{{.*}}"(%anonymous2A_0, %anonymous2A)
   fn myclosure_with_reg_types(x:Int) escaping -> Int:
      a = a + 1
      return x + w

# // -----

# CHECK-LABEL: module {

##===----------------------------------------------------------------------===##
# Nested Closures
##===----------------------------------------------------------------------===##

# CHECK-NEXT: lit.file_module @"$[[F:.*]]"

# CHECK-COUNT-2: lit.struct.decl @"_CI_

# CHECK-LABEL: lit.func @"makes_escaping_closure

# CHECK: %anonymous2A = lit.varlet.decl "anonymous*" var synth
# CHECK-NEXT: %anonymous2A_0 = lit.varlet.decl "anonymous*" var synth : <!String>
# CHECK-NEXT: [[V0:%.*]] = kgen.call @"{{.*}}::@String::@"__copyinit__({{.*}})"(%anonymous2A_0, %m)
# CHECK-NEXT: [[V1:%.*]] = kgen.call {{.*}}CI_$[[F]]_{{.*}}"::@"__init__{{.*}}(%anonymous2A, %anonymous2A_0)
# CHECK-NEXT: %anonymous2A_1 = lit.varlet.decl "anonymous*" var synth
# CHECK-NEXT: %2 = kgen.call {{.*}}CW_{{.*}}__init__{{.*}}(%anonymous2A_1, %anonymous2A)
# CHECK-NEXT: [[V3:%.*]] = kgen.param.constant: !lit.none = <#lit.none>
# CHECK-NEXT: lit.return [[V3]] : !lit.none
# CHECK-NEXT: lit.end_func
fn makes_escaping_closure(m: String):
   fn myclosure(n:String) escaping -> String:
      fn nested_nested(k:String, l:String) escaping -> String:
         return n+k
      return n+m

# // -----

# CHECK-LABEL: module {

##===----------------------------------------------------------------------===##
# Copy Constructor
##===----------------------------------------------------------------------===##

# CHECK: lit.func @"__copyinit__{{.*}}(%self: !kgen.pointer<!escaping1> init_self, %existing: !kgen.pointer<!escaping1> borrow_in_mem) -> !lit.none attributes {specialFnKind = 3 : i8} {
# CHECK-NEXT:   [[M0:%.*]] = lit.struct.gep %self[field0] : <pointer<array<0, i1>>>
# CHECK-NEXT:   [[existing_impl:%.*]] = lit.struct.gep %existing[field0] : <pointer<array<0, i1>>>
# CHECK-NEXT:   [[loaded_existing_impl:%.*]] = pop.load [[existing_impl]] : !kgen.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   pop.store [[loaded_existing_impl]], [[M0]] : !kgen.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   [[M1:%.*]] = lit.struct.gep %self[dtor] : <(!kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:   [[M2:%.*]] = lit.struct.gep %existing[dtor] : <(!kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:   [[M3:%.*]] = pop.load [[M2]] : !kgen.pointer<(!kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:   pop.store [[M3]], [[M1]] : !kgen.pointer<(!kgen.pointer<array<0, i1>>) -> !lit.none>
# CHECK-NEXT:   [[M4:%.*]] = lit.struct.gep %self[copy]
# CHECK-NEXT:   [[M5:%.*]] = lit.struct.gep %existing[copy]
# CHECK-NEXT:   [[M6:%.*]] = pop.load [[M5]]
# CHECK-NEXT:   pop.store [[M6]], [[M4]]
# CHECK-NEXT:   [[M7:%.*]] = lit.struct.gep %self[call]
# CHECK-NEXT:   [[M8:%.*]] = lit.struct.gep %existing[call]
# CHECK-NEXT:   [[M9:%.*]] = pop.load [[M8]]
# CHECK-NEXT:   pop.store [[M9]], [[M7]]
# CHECK-NEXT:   kgen.param.constant: !lit.none
# CHECK-NEXT:   [[W0:%.*]] = lit.struct.gep %existing[field0]
# CHECK-NEXT:   [[W1:%.*]] = pop.load [[W0]] : !kgen.pointer<pointer<array<0, i1>>>
# CHECK-NEXT:   [[W2:%.*]] = lit.struct.gep %self[copy]
# CHECK-NEXT:   [[W3:%.*]] = lit.struct.gep %self[field0] : <pointer<array<0, i1>>>
# CHECK-NEXT:   [[W4:%.*]] = pop.load [[W2]]

# Call the copy constructor member with the uninitialized self and the untyped existing impl.
# CHECK-NEXT:  [[W5:%.*]] = kgen.call_signature [[W4]]([[W3]], [[W1]]) : ("ptrToImpl": !kgen.pointer<pointer<array<0, i1>>> borrow, "other": !kgen.pointer<array<0, i1>> borrow_in_mem) -> !lit.none
# CHECK-NEXT:  lit.return
# CHECK-NEXT:  lit.end_func

# CHECK-LABEL: lit.func @"materialize_escaping_closure

# CHECK:      lit.func @"_CW_{{.*}}_copyinit__CI_{{.*}}"(%ptrToImpl: !kgen.pointer<pointer<array<0, i1>>> borrow, %other: !kgen.pointer<array<0, i1>> borrow_in_mem) -> !lit.none attributes {specialFnKind = 0 : i8} {

# Allocate memory on the heap for impl and copy existing contents into it.
# CHECK-NEXT:  %index = kgen.param.constant = <get_sizeof(
# CHECK-NEXT:  %index_0 = kgen.param.constant = <get_alignof(
# CHECK-NEXT:  [[V0:%.*]] = pop.aligned_alloc %index_0, %index
# CHECK-NEXT:  [[V1:%.*]] = pop.pointer.bitcast %other
# CHECK-NEXT:  [[V2:%.*]] = kgen.call {{.*}}__copyinit__(${{.*}}::_CI_${{.*}}"([[V0]], [[V1]])

# Store the address of the heap allocated memory into the self.
# CHECK-NEXT:  [[V4:%.*]] = pop.pointer.bitcast [[V0]]
# CHECK-NEXT:  pop.store [[V4]], %ptrToImpl : !kgen.pointer<pointer<array<0, i1>>>

fn materialize_escaping_closure(m: String):
   fn unique(n: String) escaping -> String:
      return m + n
