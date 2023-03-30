// RUN: kgen-opt %s -allow-unregistered-dialect -outline-closures | kgen-opt -allow-unregistered-dialect | FileCheck %s

// COM: This shouldn't change at all, save for whatever canonicalizations happen at parse time.
// CHECK-LABEL: @call_region<fn: <A -> E>() -> index -> E>() -> index always_inline
kgen.generator @call_region<fn: <A -> E>() ->index -> E>() -> index always_inline {
  // CHECK-NEXT: kgen.param.declare BoundFn: <() -> E>() -> index = <bind_signature(:<A -> E>() -> index fn, 2)>
  kgen.param.declare BoundFn: <() -> E>() -> index = <bind_signature(:<A -> E>() -> index fn, 2)>
  // CHECK-NEXT: %0 = kgen.call_param[<() -> E>() -> index: BoundFn]<() -> Result>()
  %0 = kgen.call_param[<() -> E>() -> index: BoundFn]<() -> Result>()
  // CHECK-NEXT: kgen.param.result_bind<Result>
  kgen.param.result_bind<Result>
  // CHECK-NEXT: kgen.return %0 : index
  kgen.return %0 : index
}

// COM: This is the region hoisted out into a generator.
// CHECK-LABEL: kgen.generator @raiseClosure_Fn<Jefffffffffff, C, A, B -> E>(
// CHECK-SAME:                   [[ARG:%arg[0-9]+]]: index, [[ARGARG:%arg[0-9]+]]: !pop.scalar<index>) -> index always_inline
// CHECK-NEXT: [[CST:%[0-9]+]] = kgen.param.constant = <add(mul(B, Jefffffffffff, -1), mul(A, Jefffffffffff), mul(C, Jefffffffffff))>
// CHECK-NEXT: [[CASTCST:%[0-9]+]] = pop.cast_from_builtin [[CST]] : index to !pop.scalar<index>
// CHECK-NEXT: [[CASTARG:%[0-9]+]] = pop.cast_from_builtin [[ARG]] : index to !pop.scalar<index>
// CHECK-NEXT: [[ADD:%[0-9]+]] = pop.add [[CASTCST]], [[CASTARG]] : !pop.scalar<index>
// CHECK-NEXT: [[RES:%[0-9]+]] = pop.add [[ADD]], [[ARGARG]] : !pop.scalar<index>
// CHECK-NEXT: [[CASTRES:%[0-9]+]] = pop.cast_to_builtin [[RES]] : !pop.scalar<index> to index
// CHECK-NEXT: kgen.param.result_bind<add(mul(A, -1), C)>
// CHECK-NEXT: kgen.return [[CASTRES]]

// COM: This is the wrapper that loads values from the global variable.
// CHECK-LABEL: kgen.generator @raiseClosure_Fn_wrapper<Jefffffffffff, C, A, B -> E>() -> index always_inline
// CHECK-NEXT:   [[PTR:%[0-9]+]] = pop.compiler.global_load "raiseClosure_context_var_0" : !pop.struct<index, scalar<index>>
// CHECK-NEXT:   [[VAL:%[0-9]+]] = pop.struct.extract [[PTR]][0] : !pop.struct<index, scalar<index>>
// CHECK-NEXT:   [[ARG:%[0-9]+]] = pop.struct.extract [[PTR]][1] : !pop.struct<index, scalar<index>>
// CHECK-NEXT:   [[RES:%[0-9]+]] = kgen.call @raiseClosure_Fn<Jefffffffffff = Jefffffffffff, C = C, A = A, B = B -> *"(outlined)resultParam0" = E>([[VAL]], [[ARG]]) : (index, !pop.scalar<index>) -> index
// CHECK-NEXT:   kgen.param.result_bind<*"(outlined)resultParam0">
// CHECK-NEXT:   kgen.return [[RES]] : index

// CHECK-LABEL: kgen.generator @raiseClosure
kgen.generator @raiseClosure<Jefffffffffff -> index>(%arg0: !pop.scalar<index>) -> (index, index) {
  %cst = index.constant 0
  kgen.param.declare C = <15>
  kgen.param.declare.region Fn = <A, B -> E>() -> index always_inline {
    %0 = kgen.param.constant = <mul(add(sub(A, B), C), Jefffffffffff)>
    %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
    %2 = pop.cast_from_builtin %cst : index to !pop.scalar<index>
    %3 = pop.add %1, %2 : !pop.scalar<index>
    %4 = pop.add %3, %arg0 : !pop.scalar<index>
    %5 = pop.cast_to_builtin %4 : !pop.scalar<index> to index
    kgen.param.result_bind<sub(C, A)>
    kgen.return %5 : index
  }
  // CHECK: [[STRUCT:%[0-9]+]] = pop.struct.create(%idx0, %arg0) : !pop.struct<index, scalar<index>>
  // CHECK-NEXT: pop.compiler.global_store "raiseClosure_context_var_0", [[STRUCT]] : !pop.struct<index, scalar<index>>
  // CHECK: kgen.param.declare Fn: <A, B -> E>() -> index = <@raiseClosure_Fn_wrapper<Jefffffffffff = Jefffffffffff, C = C, A = #kgen.unbound, B = #kgen.unbound>>
  // CHECK: kgen.param.declare BoundFn: <A -> E>() -> index = <bind_signature(:<A, B -> E>() -> index Fn, #kgen.unbound, 1)>
  kgen.param.declare BoundFn: <A -> E>() -> index = <bind_signature(:<A, B -> E>() -> index Fn, #kgen.unbound, 1)>
  // CHECK: kgen.call @call_region<fn: <A -> E>() -> index = BoundFn -> Result = E>() : () -> index
  %0 = kgen.call @call_region<fn: <A -> E>() ->index = BoundFn -> Result = E>() : () -> index
  %1 = kgen.param.constant = <Result>
  // CHECK: kgen.param.result_bind<Result>
  kgen.param.result_bind<Result>
  kgen.return %0, %1 : index, index
}

// CHECK-LABEL: kgen.generator @raise2Closures_Empty() always_inline
// CHECK-NEXT:    kgen.return

// CHECK-LABEL: kgen.generator @raise2Closures_Empty_wrapper() always_inline
// CHECK-NEXT:    kgen.call @raise2Closures_Empty() : () -> ()
// CHECK-NEXT:    kgen.return

// CHECK-LABEL: kgen.generator @raise2Closures_Fn<C, A -> E>(%arg0: index) -> index always_inline
// CHECK-NEXT:    %0 = kgen.param.constant = <add(A, C)>
// CHECK-NEXT:    %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
// CHECK-NEXT:    %2 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
// CHECK-NEXT:    %3 = pop.add %1, %2 : !pop.scalar<index>
// CHECK-NEXT:    %4 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
// CHECK-NEXT:    kgen.param.result_bind<add(mul(A, -1), C)>
// CHECK-NEXT:    kgen.return %4 : index

// CHECK-LABEL: kgen.generator @raise2Closures_Fn_wrapper<C, A -> E>() -> index always_inline
// CHECK-NEXT:    %0 = pop.compiler.global_load "raise2Closures_context_var_1" : !pop.struct<index>
// CHECK-NEXT:    %1 = pop.struct.extract %0[0] :
// CHECK-NEXT:    %2 = kgen.call @raise2Closures_Fn<C = C, A = A -> *"(outlined)resultParam0" = E>(%1) : (index) -> index
// CHECK-NEXT:    kgen.param.result_bind<*"(outlined)resultParam0">
// CHECK-NEXT:    kgen.return %2 : index


// CHECK-LABEL: kgen.generator @raise2Closures
kgen.generator @raise2Closures() {
  %cst = index.constant 0
  // CHECK: index.constant 0
  // CHECK-NEXT: [[STRUCT2:%[0-9]+]] = pop.struct.create(%idx0) : !pop.struct<index>
  // CHECK-NEXT: pop.compiler.global_store "raise2Closures_context_var_1", [[STRUCT2]] : !pop.struct<index>
  kgen.param.declare C = <15>

  // CHECK: kgen.param.declare Empty: () -> () = <@raise2Closures_Empty_wrapper>
  kgen.param.declare.region Empty = () -> () always_inline {
    kgen.return
  }

  // CHECK-NEXT: kgen.param.declare Fn: <A -> E>() -> index = <@raise2Closures_Fn_wrapper<C = C, A = #kgen.unbound>>
  kgen.param.declare.region Fn = <A -> E>() -> index always_inline {
    %0 = kgen.param.constant = <add(A, C)>
    %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
    %2 = pop.cast_from_builtin %cst : index to !pop.scalar<index>
    %3 = pop.add %1, %2 : !pop.scalar<index>
    %5 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
    kgen.param.result_bind<sub(C, A)>
    kgen.return %5 : index
  }

  // CHECKparam.result_bind kgen.call @call_region<fn: <A -> E>() -> index = Fn -> Result = E>() : ()  -> index
  // CHECK: kgen.call @call_region<fn: <A -> E>() -> index = Fn -> Result = E>() : ()  -> index
  %0 = kgen.call @call_region<fn: <A -> E>() ->index = Fn -> Result = E>() : () -> index
  %1 = kgen.param.constant = <Result>
  // CHECK: kgen.return
  kgen.return
}

// CHECK-LABEL: kgen.generator @parametrizedClosure_Fn<T: type>(%arg0: !kgen.paramref<T>) -> !kgen.paramref<T> always_inline
// CHECK-NEXT:    kgen.return %arg0 : !kgen.paramref<T>

// CHECK-LABEL: kgen.generator @parametrizedClosure_Fn_wrapper<T: type>() -> !kgen.paramref<T> always_inline
// CHECK-NEXT:    %0 = pop.compiler.global_load "parametrizedClosure_context_var_2" : !pop.struct<T>
// CHECK-NEXT:    %1 = pop.struct.extract %0[0] : !pop.struct<T>
// CHECK-NEXT:    %2 = kgen.call @parametrizedClosure_Fn<T: type = T>(%1) : (!kgen.paramref<T>) -> !kgen.paramref<T>
// CHECK-NEXT:    kgen.return %2 : !kgen.paramref<T>

// CHECK-LABEL: kgen.generator @parametrizedClosure<T: type>(%arg0: !kgen.paramref<T>) -> !kgen.paramref<T>
// CHECK-NEXT:    %0 = pop.struct.create(%arg0) : !pop.struct<T>
// CHECK-NEXT:    pop.compiler.global_store "parametrizedClosure_context_var_2", %0 : !pop.struct<T>
// CHECK-NEXT:    kgen.param.declare Fn: () -> !kgen.paramref<T> = <@parametrizedClosure_Fn_wrapper<T: type = T>>
// CHECK-NEXT:    %1 = kgen.call_param[() -> !kgen.paramref<T>: Fn]()
// CHECK-NEXT:    kgen.return %1 : !kgen.paramref<T>

// CHECK-LABEL: kgen.generator @raiseParamClosure() -> f32
// CHECK-NEXT:    %0 = kgen.param.constant: scalar<f32> = <"0">
// CHECK-NEXT:    %1 = pop.cast_to_builtin %0 : !pop.scalar<f32> to f32
// CHECK-NEXT:    %2 = kgen.call @parametrizedClosure<T: type = f32>(%1) : (f32) -> f32
// CHECK-NEXT:    kgen.return %2 : f32


kgen.generator @parametrizedClosure<T: type>(%arg0: !kgen.paramref<T>) -> !kgen.paramref<T> {
  kgen.param.declare.region Fn = <>() -> !kgen.paramref<T> always_inline {
    kgen.return %arg0 : !kgen.paramref<T>
  }
  %1 = kgen.call_param[<>() -> !kgen.paramref<T>: Fn]()
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
  // CHECK-NEXT: [[STRUCT2:%[0-9]+]] = pop.struct.create(%idx0) : !pop.struct<index>
  // CHECK-NEXT: pop.compiler.global_store "useAfterDef_context_var_3", [[STRUCT2]] : !pop.struct<index>
  kgen.param.declare C = <15>

  // CHECK: kgen.call @call_region<fn: <A -> E>() -> index = Fn -> Result = E>() : () -> index
  %call = kgen.call @call_region<fn: <A -> E>() ->index = Fn -> Result = E>() : () -> index
  // CHECK-NEXT: kgen.param.constant
  %constant = kgen.param.constant = <Result>

  // CHECK-NEXT: kgen.param.declare Fn: <A -> E>() -> index = <@useAfterDef_Fn_wrapper<C = C, A = #kgen.unbound>>
  kgen.param.declare.region Fn = <A -> E>() -> index always_inline {
    %0 = kgen.param.constant = <add(A, C)>
    %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
    %2 = pop.cast_from_builtin %cst : index to !pop.scalar<index>
    %3 = pop.add %1, %2 : !pop.scalar<index>
    %5 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
    kgen.param.result_bind<sub(C, A)>
    kgen.return %5 : index
  }

  // CHECK: kgen.return
  kgen.return %call : index
}

// CHECK-LABEL: @nested
kgen.generator @nested(%pred: i1) -> index {
  kgen.param.declare C = <15>

  // CHECK: hlcf.if
  %if = hlcf.if %pred -> index {
    %cst = index.constant 0
    // CHECK-NEXT: index.constant 0
    // CHECK-NEXT: [[STRUCT2:%[0-9]+]] = pop.struct.create(%idx0) : !pop.struct<index>
    // CHECK-NEXT: pop.compiler.global_store "nested_context_var_4", [[STRUCT2]] : !pop.struct<index>

    // CHECK-NEXT: kgen.call @call_region<fn: <A -> E>() -> index = Fn -> Result = E>() : () -> index
    %call = kgen.call @call_region<fn: <A -> E>() ->index = Fn -> Result = E>() : () -> index

    // CHECK-NEXT: kgen.param.declare
    kgen.param.declare.region Fn = <A -> E>() -> index always_inline {
      %0 = kgen.param.constant = <add(A, C)>
      %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
      %2 = pop.cast_from_builtin %cst : index to !pop.scalar<index>
      %3 = pop.add %1, %2 : !pop.scalar<index>
      %5 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
      kgen.param.result_bind<sub(C, A)>
      kgen.return %5 : index
    }
    hlcf.yield %call : index
  } else {
    %cst = index.constant 0
    hlcf.yield %cst : index
  }

  %1 = kgen.param.constant = <Result>

  // CHECK: kgen.param.declare Empty: () -> () = <@nested_Empty_wrapper>
  kgen.param.declare.region Empty = () -> () always_inline {
    kgen.return
  }

  // CHECK: kgen.return
  kgen.return %if : index
}

// CHECK-LABEL: @nested2
kgen.generator @nested2() -> index {
  %cst = index.constant 0
  // CHECK: index.constant
  kgen.param.declare C = <15>

  // CHECK: hlcf.loop
  %res = hlcf.loop (%input = %cst: index) -> index {
    // CHECK-NEXT: [[STRUCT2:%[0-9]+]] = pop.struct.create(%arg0, %idx0) : !pop.struct<index, index>
    // CHECK-NEXT: pop.compiler.global_store "nested2_context_var_5", [[STRUCT2]] : !pop.struct<index, index>
    // CHECK-NEXT: kgen.param.declare
    kgen.param.declare.region Fn = <A -> E>() -> index always_inline {
      %1 = pop.cast_from_builtin %input : index to !pop.scalar<index>
      %2 = pop.cast_from_builtin %cst : index to !pop.scalar<index>
      %3 = pop.add %1, %2 : !pop.scalar<index>
      %5 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
      kgen.param.result_bind<sub(C, A)>
      kgen.return %5 : index
    }
    // CHECK-NEXT: kgen.call @call_region<fn: <A -> E>() -> index = Fn -> Result = E>() : () -> index
    %call = kgen.call @call_region<fn: <A -> E>() -> index = Fn -> Result = E>() : () -> index
    hlcf.break %call : index
  }

  // CHECK: kgen.return
  kgen.return %res : index
}

// CHECK-LABEL: kgen.generator @capture_crosses_parameter_domain
kgen.generator @capture_crosses_parameter_domain<T: type>(%arg0: !kgen.paramref<T>) {
  // CHECK: declare Fn: <A, T: type>
  kgen.param.declare.region Fn = <A, T: type>() -> !kgen.paramref<T> always_inline {
    kgen.return %arg0: !kgen.paramref<T>
  }
  kgen.return
}

// COM: We have to parametrize the wrapper on captured SSA values as well, check that this actually happens.
// CHECK-LABEL: @parametrizedSSACapture_fn_wrapper<T: type>
// CHECK-LABEL: @parametrizedSSACapture
kgen.generator @parametrizedSSACapture<T: type>(%arg0 : !kgen.paramref<T>) -> index {
  %0 = kgen.call_param[<>() -> index: fn]()
  // CHECK: kgen.param.declare fn: () -> index = <@parametrizedSSACapture_fn_wrapper<T: type = T>>
  kgen.param.declare.region fn = () -> index always_inline {
    "op.use"(%arg0) : (!kgen.paramref<T>) -> ()
    %1 = kgen.param.constant = <0>
    kgen.return %1 : index
  }
  kgen.return %0 : index
}

// COM: We should not try and capture input parameters.
// CHECK-LABEL: @dontBindInputParameters_fn_wrapper<T: type, I>
// CHECK-LABEL: @dontBindInputParameters
kgen.generator @dontBindInputParameters<T: type, I>(%arg0 : !kgen.paramref<T>) -> index {
  %0 = kgen.call_param[<>() -> index: bind_signature(: <I>() -> index fn, I)]()
  // CHECK: kgen.param.declare fn: <I>() -> index = <@dontBindInputParameters_fn_wrapper<T: type = T, I = #kgen.unbound>>
  kgen.param.declare.region fn = <I>() -> index always_inline {
    %1 = kgen.param.constant = <I>
    "use.op"(%arg0) : (!kgen.paramref<T>) -> ()
    kgen.return %1 : index
  }
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.generator @innermostCapturesThroughMid_Bot<A>
// CHECK: kgen.generator @innermostCapturesThroughMid_Bot_wrapper<A>
// CHECK: kgen.generator @innermostCapturesThroughMid_Mid<A>
// CHECK: kgen.generator @innermostCapturesThroughMid_Mid_wrapper<A>
// CHECK: kgen.generator @innermostCapturesThroughMid<A>
// CHECK-NEXT: @innermostCapturesThroughMid_Mid_wrapper<A = A>

kgen.generator @innermostCapturesThroughMid<A>() {
  kgen.param.declare.region Mid = () {
    kgen.param.declare.region Bot = () {
      kgen.param.constant = <A>
      kgen.return
    }
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @paramCaptureNestedInParamRefType_Fn<N, Vs: list<i32[N]>>
// CHECK-NEXT: constant: list<i32[N]> = <Vs>
// CHECK: kgen.generator @paramCaptureNestedInParamRefType_Fn_wrapper<N, Vs: list<i32[N]>>
// CHECK-NEXT: call @paramCaptureNestedInParamRefType_Fn<N = N, Vs: list<i32[N]> = Vs>
// CHECK: declare Fn: () -> () = <@paramCaptureNestedInParamRefType_Fn_wrapper<N = N, Vs: list<i32[N]> = Vs>>

kgen.generator @paramCaptureNestedInParamRefType<N, Vs: list<i32[N]>>() {
  kgen.param.declare.region Fn = () {
    kgen.param.constant: list<i32[N]> = <Vs>
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: @left_to_right_dependency_CaptureThemAll
// CHECK-SAME: <F: type, G: type, H: type, I: type, J: type, A,
// CHECK-SAME:  L: list<!pop.struct<F, G, H, I, J>[A]>, B: type,
// CHECK-SAME:  E: list<!kgen.list<!kgen.list<B[A]>[A]>[A]>,
// CHECK-SAME:  D: list<!kgen.list<B[A]>[A]>, C: list<B[A]>
kgen.generator @left_to_right_dependency<
    A, B: type, C: list<B[A]>, D: list<!kgen.list<B[A]>[A]>,
    E: list<!kgen.list<!kgen.list<B[A]>[A]>[A]>,
    F: type, G: type, H: type, I: type, J: type,
    K: struct<F, G, H, I, J>, L: list<!pop.struct<F, G, H, I, J>[A]>>() {
  kgen.param.declare.region CaptureThemAll = () {
    "use"() {
      a = #kgen.param.decl.ref<"L"> : !kgen.list<!pop.struct<F, G, H, I, J>[A]>,
      b = #kgen.param.decl.ref<"E"> : !kgen.list<!kgen.list<!kgen.list<B[A]>[A]>[A]>,
      c = #kgen.param.decl.ref<"D"> : !kgen.list<!kgen.list<B[A]>[A]>,
      d = #kgen.param.decl.ref<"C"> : !kgen.list<B[A]>
    } : () -> ()
    kgen.return
  }
  kgen.return
}

// CHECK-LABEL: kgen.generator @dependent_outline<a>
kgen.generator @dependent_outline<a>() {
  // CHECK-NEXT: kgen.param.declare fn: <b: type>(!pop.array<a, *0|0>) -> () =
  // CHECK-SAME: <@dependent_outline_fn_wrapper<a = a, b: type = #kgen.unbound>>
  kgen.param.declare.region fn = <b: type>(%arg0: !pop.array<a, b>) {
    kgen.return
  }
  kgen.return
}
