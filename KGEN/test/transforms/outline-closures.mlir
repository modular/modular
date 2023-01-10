// RUN: kgen-opt %s -allow-unregistered-dialect -outline-closures | kgen-opt -allow-unregistered-dialect | FileCheck %s

// COM: This shouldn't change at all, save for whatever canonicalizations happen at parse time.
// CHECK-LABEL: @call_region<fn: <A -> index>() force_inline -> index -> index>() force_inline -> index
kgen.generator @call_region<fn: <A -> index>() force_inline ->index -> index>() force_inline -> index {
  // CHECK-NEXT: kgen.param.declare BoundFn: <() -> index>() force_inline -> index = <bind_signature(:<A -> index>() force_inline -> index fn, 2)>
  kgen.param.declare BoundFn: <() -> index>() force_inline -> index = <bind_signature(:<A -> index>() force_inline -> index fn, 2)>
  // CHECK-NEXT: %0 = kgen.call_param[<() -> index>() force_inline -> index: BoundFn]<() -> Result>()
  %0 = kgen.call_param[<() -> index>() force_inline -> index: BoundFn]<() -> Result>()
  // CHECK-NEXT: kgen.return<Result> %0
  kgen.return<Result> %0 : index
}

// CHECK-LABEL: kgen.struct.decl @raiseClosure_context
// CHECK-NEXT: field_0 : index
// CHECK-NEXT: field_1 : !pop.scalar<index>

// CHECK-LABEL: pop.compiler.global_variable @raiseClosure_context_var : !kgen.declref<@raiseClosure_context>

// COM: This is the region hoisted out into a generator.
// CHECK-LABEL: kgen.generator @raiseClosure_0<Jefffffffffff, C, A, B -> index>(
// CHECK-SAME:                   [[ARG:%arg[0-9]+]]: index, [[ARGARG:%arg[0-9]+]]: !pop.scalar<index>) force_inline -> index
// CHECK-NEXT: [[CST:%[0-9]+]] = kgen.param.constant = <add(mul(B, Jefffffffffff, -1), mul(A, Jefffffffffff), mul(C, Jefffffffffff))>
// CHECK-NEXT: [[CASTCST:%[0-9]+]] = pop.cast_from_builtin [[CST]] : index to !pop.scalar<index>
// CHECK-NEXT: [[CASTARG:%[0-9]+]] = pop.cast_from_builtin [[ARG]] : index to !pop.scalar<index>
// CHECK-NEXT: [[ADD:%[0-9]+]] = pop.add [[CASTCST]], [[CASTARG]] : !pop.scalar<index>
// CHECK-NEXT: [[RES:%[0-9]+]] = pop.add [[ADD]], [[ARGARG]] : !pop.scalar<index>
// CHECK-NEXT: [[CASTRES:%[0-9]+]] = pop.cast_to_builtin [[RES]] : !pop.scalar<index> to index
// CHECK-NEXT: kgen.return<add(mul(A, -1), C)> [[CASTRES]]

// COM: This is the wrapper that loads values from the global variable.
// CHECK-LABEL: kgen.generator @raiseClosure_wrapper<Jefffffffffff, C, A, B -> index>() force_inline -> index
// CHECK-NEXT:   [[PTR:%[0-9]+]] = pop.compiler.global_load @raiseClosure_context_var : !kgen.declref<@raiseClosure_context>
// CHECK-NEXT:   [[VAL:%[0-9]+]] = kgen.struct.extract [[PTR]][field_0] : index from !kgen.declref<@raiseClosure_context>
// CHECK-NEXT:   [[ARG:%[0-9]+]] = kgen.struct.extract [[PTR]][field_1] : !pop.scalar<index> from !kgen.declref<@raiseClosure_context>
// CHECK-NEXT:   [[RES:%[0-9]+]] = kgen.call @raiseClosure_0<Jefffffffffff = Jefffffffffff, C = C, A = A, B = B -> __resultParam_0>([[VAL]], [[ARG]]) : (index, !pop.scalar<index>) force_inline -> index
// CHECK-NEXT:   kgen.return<__resultParam_0> [[RES]] : index

// CHECK-LABEL: kgen.generator @raiseClosure
kgen.generator @raiseClosure<Jefffffffffff -> index>(%arg0: !pop.scalar<index>) -> (index, index) {
  %cst = index.constant 0
  kgen.param.declare C = <15>
  kgen.param.declare.region Fn = <A, B -> index>() force_inline -> index {
    %0 = kgen.param.constant = <mul(add(sub(A, B), C), Jefffffffffff)>
    %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
    %2 = pop.cast_from_builtin %cst : index to !pop.scalar<index>
    %3 = pop.add %1, %2 : !pop.scalar<index>
    %4 = pop.add %3, %arg0 : !pop.scalar<index>
    %5 = pop.cast_to_builtin %4 : !pop.scalar<index> to index
    kgen.return<sub(C, A)> %5 : index
  }
  // CHECK: [[STRUCT:%[0-9]+]] = kgen.struct.create(%idx0, %arg0) : (index, !pop.scalar<index>) -> !kgen.declref<@raiseClosure_context>
  // CHECK-NEXT: pop.compiler.global_store @raiseClosure_context_var, [[STRUCT]] : !kgen.declref<@raiseClosure_context>
  // CHECK: kgen.param.declare Fn: <A, B -> index>() force_inline -> index = <@raiseClosure_wrapper<Jefffffffffff = Jefffffffffff, C = C, A = #kgen.unbound, B = #kgen.unbound>>
  // CHECK: kgen.param.declare BoundFn: <A -> index>() force_inline -> index = <bind_signature(:<A, B -> index>() force_inline -> index Fn, #kgen.unbound, 1)>
  kgen.param.declare BoundFn: <A -> index>() force_inline -> index = <bind_signature(:<A, B -> index>() force_inline -> index Fn, #kgen.unbound, 1)>
  // CHECK: kgen.call @call_region<fn: <A -> index>() force_inline -> index = BoundFn -> Result>() : () force_inline -> index
  %0 = kgen.call @call_region<fn: <A -> index>() force_inline ->index = BoundFn -> Result>() : () force_inline -> index
  %1 = kgen.param.constant = <Result>
  // CHECK: kgen.return<Result>
  kgen.return<Result> %0, %1 : index, index
}

// CHECK-LABEL: kgen.struct.decl @raise2Closures_context

// CHECK-LABEL: pop.compiler.global_variable @raise2Closures_context_var : !kgen.declref<@raise2Closures_context>
// CHECK-LABEL: kgen.generator @raise2Closures_1() force_inline
// CHECK-NEXT:    kgen.return

// CHECK-LABEL: kgen.generator @raise2Closures_wrapper() force_inline
// CHECK-NEXT:    %0 = pop.compiler.global_load @raise2Closures_context_var : !kgen.declref<@raise2Closures_context>
// CHECK-NEXT:    kgen.call @raise2Closures_1() : () force_inline -> ()
// CHECK-NEXT:    kgen.return

// CHECK-LABEL: kgen.struct.decl @raise2Closures_context_2
// CHECK-NEXT:    kgen.struct.field field_0 : index

// CHECK-LABEL: pop.compiler.global_variable @raise2Closures_context_var_3 : !kgen.declref<@raise2Closures_context_2>
// CHECK-LABEL: kgen.generator @raise2Closures_4<C, A -> index>(%arg0: index) force_inline -> index
// CHECK-NEXT:    %0 = kgen.param.constant = <add(A, C)>
// CHECK-NEXT:    %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
// CHECK-NEXT:    %2 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
// CHECK-NEXT:    %3 = pop.add %1, %2 : !pop.scalar<index>
// CHECK-NEXT:    %4 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
// CHECK-NEXT:    kgen.return<add(mul(A, -1), C)> %4 : index

// CHECK-LABEL: kgen.generator @raise2Closures_wrapper_5<C, A -> index>() force_inline -> index
// CHECK-NEXT:    %0 = pop.compiler.global_load @raise2Closures_context_var_3 : !kgen.declref<@raise2Closures_context_2>
// CHECK-NEXT:    %1 = kgen.struct.extract %0[field_0] : index from !kgen.declref<@raise2Closures_context_2>
// CHECK-NEXT:    %2 = kgen.call @raise2Closures_4<C = C, A = A -> __resultParam_0>(%1) : (index) force_inline -> index
// CHECK-NEXT:    kgen.return<__resultParam_0> %2 : index


// CHECK-LABEL: kgen.generator @raise2Closures
kgen.generator @raise2Closures() {
  %cst = index.constant 0
  kgen.param.declare C = <15>

  // CHECK: [[STRUCT:%[0-9]+]] = kgen.struct.create() : () -> !kgen.declref<@raise2Closures_context>
  // CHECK-NEXT: pop.compiler.global_store @raise2Closures_context_var, [[STRUCT]] : !kgen.declref<@raise2Closures_context>
  // CHECK-NEXT: kgen.param.declare Empty: <>() force_inline -> () = <@raise2Closures_wrapper>
  kgen.param.declare.region Empty = () force_inline -> () {
    kgen.return
  }

  // CHECK: [[STRUCT2:%[0-9]+]] = kgen.struct.create(%idx0) : (index) -> !kgen.declref<@raise2Closures_context_2>
  // CHECK-NEXT: pop.compiler.global_store @raise2Closures_context_var_3, [[STRUCT2]] : !kgen.declref<@raise2Closures_context_2>
  // CHECK-NEXT: kgen.param.declare Fn: <A -> index>() force_inline -> index = <@raise2Closures_wrapper_5<C = C, A = #kgen.unbound>>
  kgen.param.declare.region Fn = <A -> index>() force_inline -> index {
    %0 = kgen.param.constant = <add(A, C)>
    %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
    %2 = pop.cast_from_builtin %cst : index to !pop.scalar<index>
    %3 = pop.add %1, %2 : !pop.scalar<index>
    %5 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
    kgen.return<sub(C, A)> %5 : index
  }

  // CHECK: kgen.call @call_region<fn: <A -> index>() force_inline -> index = Fn -> Result>() : () force_inline -> index
  %0 = kgen.call @call_region<fn: <A -> index>() force_inline ->index = Fn -> Result>() : () force_inline -> index
  %1 = kgen.param.constant = <Result>
  // CHECK: kgen.return
  kgen.return
}
