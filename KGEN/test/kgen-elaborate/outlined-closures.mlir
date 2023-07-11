// RUN: kgen-opt %s -verify-parameters | kgen-elaborate-opt -elaborate-generators | kgen-opt -force-inline -eliminate-dead-symbols -cleanup-compiler-globals | FileCheck %s

kgen.generator @call_region<fn: <index -> index>() -> index -> E>() -> index always_inline {
  kgen.param.declare BoundFn: <[] -> index>() -> index = <bind_signature(:<index -> index>() -> index fn, 2)>
  %0 = kgen.call_param[<[] -> index>() -> index: BoundFn]<() -> Result>()
  kgen.param.result_bind<Result>
  kgen.return %0 : index
}

kgen.generator @raiseClosure_0<C, A, B -> E>(%arg0: index) -> index always_inline {
  %0 = kgen.param.constant = <add(mul(B, -1), A, C)>
  %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
  %2 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
  %3 = pop.add %1, %2 : !pop.scalar<index>
  %4 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
  kgen.param.result_bind<add(mul(A, -1), C)>
  kgen.return %4 : index
}

kgen.generator @raiseClosure_wrapper<C, A, B -> E>() -> index always_inline {
  %0 = pop.compiler.global_load "raiseClosure_context_var" : !pop.struct<index>
  %1 = pop.struct.extract %0[0] : !pop.struct<index>
  %2 = kgen.call @raiseClosure_0<C, A, B -> __resultParam_0>(%1) : (index) -> index
  kgen.param.result_bind<__resultParam_0>
  kgen.return %2 : index
}

// COM: All this should be inlined and all that we care about is the raiseClosure func.
// CHECK-LABEL: @raiseClosure() -> (index, index)
// CHECK-NEXT: %idx0 = index.constant 0
// CHECK-NEXT: pop.struct.create(%idx0)
// CHECK: pop.struct.extract {{%[0-9]}}[0]
// CHECK: kgen.param.constant{{.*}}<16>
// CHECK-NEXT: pop.cast_from_builtin
// CHECK-NEXT: pop.cast_from_builtin
// CHECK-NEXT: pop.add
// CHECK-NEXT: pop.cast_to_builtin
// CHECK: kgen.param.constant = <13>
// CHECK-NEXT: kgen.return

kgen.generator export @raiseClosure<() -> E>() -> (index, index) {
  %idx0 = index.constant 0
  kgen.param.declare C = <15>
  %0 = pop.struct.create(%idx0) : !pop.struct<index>
  pop.compiler.global_store "raiseClosure_context_var", %0 : !pop.struct<index>
  kgen.param.declare Fn: <index, index -> index>() -> index = <@raiseClosure_wrapper<C, #kgen.unbound, #kgen.unbound>>
  kgen.param.declare BoundFn: <index -> index>() -> index = <bind_signature(:<index, index -> index>() -> index Fn, #kgen.unbound, 1)>
  %1 = kgen.call @call_region<:<index -> index>() -> index BoundFn -> Result>() : () -> index
  %2 = kgen.param.constant = <Result>
  kgen.param.result_bind<Result>
  kgen.return %1, %2 : index, index
}
