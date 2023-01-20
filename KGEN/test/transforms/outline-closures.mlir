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
// CHECK-NEXT:   [[PTR:%[0-9]+]] = pop.compiler.global_load "raiseClosure_context_var_0" : !pop.struct<index, scalar<index>>
// CHECK-NEXT:   [[VAL:%[0-9]+]] = pop.struct.get [[PTR]][0] : !pop.struct<index, scalar<index>>
// CHECK-NEXT:   [[ARG:%[0-9]+]] = pop.struct.get [[PTR]][1] : !pop.struct<index, scalar<index>>
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
  // CHECK: [[STRUCT:%[0-9]+]] = pop.struct.construct(%idx0, %arg0) : !pop.struct<index, scalar<index>>
  // CHECK-NEXT: pop.compiler.global_store "raiseClosure_context_var_0", [[STRUCT]] : !pop.struct<index, scalar<index>>
  // CHECK: kgen.param.declare Fn: <A, B -> index>() force_inline -> index = <@raiseClosure_wrapper<Jefffffffffff = Jefffffffffff, C = C, A = #kgen.unbound, B = #kgen.unbound>>
  // CHECK: kgen.param.declare BoundFn: <A -> index>() force_inline -> index = <bind_signature(:<A, B -> index>() force_inline -> index Fn, #kgen.unbound, 1)>
  kgen.param.declare BoundFn: <A -> index>() force_inline -> index = <bind_signature(:<A, B -> index>() force_inline -> index Fn, #kgen.unbound, 1)>
  // CHECK: kgen.call @call_region<fn: <A -> index>() force_inline -> index = BoundFn -> Result>() : () force_inline -> index
  %0 = kgen.call @call_region<fn: <A -> index>() force_inline ->index = BoundFn -> Result>() : () force_inline -> index
  %1 = kgen.param.constant = <Result>
  // CHECK: kgen.return<Result>
  kgen.return<Result> %0, %1 : index, index
}

// CHECK-LABEL: kgen.generator @raise2Closures_1() force_inline
// CHECK-NEXT:    kgen.return

// CHECK-LABEL: kgen.generator @raise2Closures_wrapper() force_inline
// CHECK-NEXT:    kgen.call @raise2Closures_1() : () force_inline -> ()
// CHECK-NEXT:    kgen.return

// CHECK-LABEL: kgen.generator @raise2Closures_2<C, A -> index>(%arg0: index) force_inline -> index
// CHECK-NEXT:    %0 = kgen.param.constant = <add(A, C)>
// CHECK-NEXT:    %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
// CHECK-NEXT:    %2 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
// CHECK-NEXT:    %3 = pop.add %1, %2 : !pop.scalar<index>
// CHECK-NEXT:    %4 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
// CHECK-NEXT:    kgen.return<add(mul(A, -1), C)> %4 : index

// CHECK-LABEL: kgen.generator @raise2Closures_wrapper_3<C, A -> index>() force_inline -> index
// CHECK-NEXT:    %0 = pop.compiler.global_load "raise2Closures_context_var_1" : !pop.struct<index>
// CHECK-NEXT:    %1 = pop.struct.get %0[0] :
// CHECK-NEXT:    %2 = kgen.call @raise2Closures_2<C = C, A = A -> __resultParam_0>(%1) : (index) force_inline -> index
// CHECK-NEXT:    kgen.return<__resultParam_0> %2 : index


// CHECK-LABEL: kgen.generator @raise2Closures
kgen.generator @raise2Closures() {
  %cst = index.constant 0
  // CHECK: index.constant 0
  // CHECK-NEXT: [[STRUCT2:%[0-9]+]] = pop.struct.construct(%idx0) : !pop.struct<index>
  // CHECK-NEXT: pop.compiler.global_store "raise2Closures_context_var_1", [[STRUCT2]] : !pop.struct<index>
  kgen.param.declare C = <15>

  // CHECK: kgen.param.declare Empty: <>() force_inline -> () = <@raise2Closures_wrapper>
  kgen.param.declare.region Empty = () force_inline -> () {
    kgen.return
  }

  // CHECK-NEXT: kgen.param.declare Fn: <A -> index>() force_inline -> index = <@raise2Closures_wrapper_3<C = C, A = #kgen.unbound>>
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

// CHECK-LABEL: kgen.generator @parametrizedClosure_4<T: type>(%arg0: !kgen.paramref<T>) force_inline -> !kgen.paramref<T>
// CHECK-NEXT:    kgen.return %arg0 : !kgen.paramref<T>

// CHECK-LABEL: kgen.generator @parametrizedClosure_wrapper<T: type>() force_inline -> !kgen.paramref<T>
// CHECK-NEXT:    %0 = pop.compiler.global_load "parametrizedClosure_context_var_2" : !pop.struct<T>
// CHECK-NEXT:    %1 = pop.struct.get %0[0] : !pop.struct<T>
// CHECK-NEXT:    %2 = kgen.call @parametrizedClosure_4<T: type = T>(%1) : (!kgen.paramref<T>) force_inline -> !kgen.paramref<T>
// CHECK-NEXT:    kgen.return %2 : !kgen.paramref<T>

// CHECK-LABEL: kgen.generator @parametrizedClosure<T: type>(%arg0: !kgen.paramref<T>) -> !kgen.paramref<T>
// CHECK-NEXT:    %0 = pop.struct.construct(%arg0) : !pop.struct<T>
// CHECK-NEXT:    pop.compiler.global_store "parametrizedClosure_context_var_2", %0 : !pop.struct<T>
// CHECK-NEXT:    kgen.param.declare Fn: <>() force_inline -> !kgen.paramref<T> = <@parametrizedClosure_wrapper<T: type = T>>
// CHECK-NEXT:    %1 = kgen.call_param[<>() force_inline -> !kgen.paramref<T>: Fn]()
// CHECK-NEXT:    kgen.return %1 : !kgen.paramref<T>

// CHECK-LABEL: kgen.generator @raiseParamClosure() -> f32
// CHECK-NEXT:    %0 = kgen.param.constant: !pop.scalar<f32> = <<"0">>
// CHECK-NEXT:    %1 = pop.cast_to_builtin %0 : !pop.scalar<f32> to f32
// CHECK-NEXT:    %2 = kgen.call @parametrizedClosure<T: type = f32>(%1) : (f32) -> f32
// CHECK-NEXT:    kgen.return %2 : f32


kgen.generator @parametrizedClosure<T: type>(%arg0: !kgen.paramref<T>) -> !kgen.paramref<T> {
  kgen.param.declare.region Fn = <>() force_inline -> !kgen.paramref<T> {
    kgen.return %arg0 : !kgen.paramref<T>
  }
  %1 = kgen.call_param[<>() force_inline -> !kgen.paramref<T>: Fn]()
  kgen.return %1 : !kgen.paramref<T>
}

kgen.generator @raiseParamClosure() -> f32 {
  %0 = kgen.param.constant : !pop.scalar<f32> = <<"0.000000e+00">>
  %1 = pop.cast_to_builtin %0 : !pop.scalar<f32> to f32
  %2 = kgen.call @parametrizedClosure<T: type = f32>(%1) : (f32) -> (f32)
  kgen.return %2 : f32
}

// CHECK-LABEL: @useAfterDef
kgen.generator @useAfterDef() -> index {
  %cst = index.constant 0
  // CHECK: index.constant 0
  // CHECK-NEXT: [[STRUCT2:%[0-9]+]] = pop.struct.construct(%idx0) : !pop.struct<index>
  // CHECK-NEXT: pop.compiler.global_store "useAfterDef_context_var_3", [[STRUCT2]] : !pop.struct<index>
  kgen.param.declare C = <15>

  // CHECK: kgen.call @call_region<fn: <A -> index>() force_inline -> index = Fn -> Result>() : () force_inline -> index
  %call = kgen.call @call_region<fn: <A -> index>() force_inline ->index = Fn -> Result>() : () force_inline -> index
  // CHECK-NEXT: kgen.param.constant
  %constant = kgen.param.constant = <Result>

  // CHECK-NEXT: kgen.param.declare Fn: <A -> index>() force_inline -> index = <@useAfterDef_wrapper<C = C, A = #kgen.unbound>>
  kgen.param.declare.region Fn = <A -> index>() force_inline -> index {
    %0 = kgen.param.constant = <add(A, C)>
    %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
    %2 = pop.cast_from_builtin %cst : index to !pop.scalar<index>
    %3 = pop.add %1, %2 : !pop.scalar<index>
    %5 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
    kgen.return<sub(C, A)> %5 : index
  }

  // CHECK: kgen.return
  kgen.return %call : index
}

// CHECK-LABEL: @nested
kgen.generator @nested(%pred: i1) -> index {
  kgen.param.declare C = <15>

  // CHECK: scf.if
  %if = scf.if %pred -> index {
    %cst = index.constant 0
    // CHECK-NEXT: index.constant 0
    // CHECK-NEXT: [[STRUCT2:%[0-9]+]] = pop.struct.construct(%idx0) : !pop.struct<index>
    // CHECK-NEXT: pop.compiler.global_store "nested_context_var_4", [[STRUCT2]] : !pop.struct<index>

    // CHECK-NEXT: kgen.call @call_region<fn: <A -> index>() force_inline -> index = Fn -> Result>() : () force_inline -> index
    %call = kgen.call @call_region<fn: <A -> index>() force_inline ->index = Fn -> Result>() : () force_inline -> index

    // CHECK-NEXT: kgen.param.declare
    kgen.param.declare.region Fn = <A -> index>() force_inline -> index {
      %0 = kgen.param.constant = <add(A, C)>
      %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
      %2 = pop.cast_from_builtin %cst : index to !pop.scalar<index>
      %3 = pop.add %1, %2 : !pop.scalar<index>
      %5 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
      kgen.return<sub(C, A)> %5 : index
    }
    scf.yield %call : index
  } else {
    %cst = index.constant 0
    scf.yield %cst : index
  }

  %1 = kgen.param.constant = <Result>

  // CHECK: kgen.param.declare Empty: <>() force_inline -> () = <@nested_wrapper_8>
  kgen.param.declare.region Empty = () force_inline -> () {
    kgen.return
  }

  // CHECK: kgen.return
  kgen.return %if : index
}

// CHECK-LABEL: @nested2
kgen.generator @nested2() -> index {
  %cst = index.constant 0
  %ub = index.constant 32
  %step = index.constant 1
  // CHECK-3: index.constant
  kgen.param.declare C = <15>

  // CHECK: scf.for
  %res = scf.for %iv = %cst to %ub step %step iter_args(%input = %cst) -> index {
    // CHECK-NEXT: [[STRUCT2:%[0-9]+]] = pop.struct.construct(%arg1, %idx0) : !pop.struct<index, index>
    // CHECK-NEXT: pop.compiler.global_store "nested2_context_var_5", [[STRUCT2]] : !pop.struct<index, index>
    // CHECK-NEXT: kgen.param.declare
    kgen.param.declare.region Fn = <A -> index>() force_inline -> index {
      %1 = pop.cast_from_builtin %input : index to !pop.scalar<index>
      %2 = pop.cast_from_builtin %cst : index to !pop.scalar<index>
      %3 = pop.add %1, %2 : !pop.scalar<index>
      %5 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
      kgen.return<sub(C, A)> %5 : index
    }
    // CHECK-NEXT: kgen.call @call_region<fn: <A -> index>() force_inline -> index = Fn -> Result>() : () force_inline -> index
    %call = kgen.call @call_region<fn: <A -> index>() force_inline ->index = Fn -> Result>() : () force_inline -> index
    scf.yield %call : index
  }

  // CHECK: kgen.return
  kgen.return %res : index
}
