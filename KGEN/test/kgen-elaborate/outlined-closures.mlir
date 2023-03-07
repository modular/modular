// RUN: kgen-opt %s -elaborate-generators -force-inline -eliminate-dead-symbols -cleanup-compiler-globals | FileCheck %s

kgen.generator @call_region<fn: <A -> E>() -> index -> E>() -> index always_inline {
  kgen.param.declare BoundFn: <() -> E>() -> index = <bind_signature(:<A -> E>() -> index fn, 2)>
  %0 = kgen.call_param[<() -> E>() -> index: BoundFn]<() -> Result>()
  kgen.return<Result> %0 : index
}

lit.struct.decl @raiseClosure_context {
  lit.struct.field field_0 : index
}

kgen.generator @raiseClosure_0<C, A, B -> E>(%arg0: index) -> index always_inline {
  %0 = kgen.param.constant = <add(mul(B, -1), A, C)>
  %1 = pop.cast_from_builtin %0 : index to !pop.scalar<index>
  %2 = pop.cast_from_builtin %arg0 : index to !pop.scalar<index>
  %3 = pop.add %1, %2 : !pop.scalar<index>
  %4 = pop.cast_to_builtin %3 : !pop.scalar<index> to index
  kgen.return<add(mul(A, -1), C)> %4 : index
}

kgen.generator @raiseClosure_wrapper<C, A, B -> E>() -> index always_inline {
  %0 = pop.compiler.global_load "raiseClosure_context_var" : !kgen.declref<@raiseClosure_context>
  %1 = lit.struct.extract %0[field_0] : index from !kgen.declref<@raiseClosure_context>
  %2 = kgen.call @raiseClosure_0<C = C, A = A, B = B -> __resultParam_0 = E>(%1) : (index) -> index
  kgen.return<__resultParam_0> %2 : index
}

// COM: All this should be inlined and all that we care about is the raiseClosure func.
// CHECK-LABEL: @raiseClosure() -> (index, index)
// CHECK-NEXT: %idx0 = index.constant 0
// CHECK-NEXT: lit.struct.create(field_0=%idx0)
// CHECK: lit.struct.extract {{%[0-9]}}[field_0]
// CHECK: kgen.param.constant{{.*}}<16>
// CHECK-NEXT: pop.cast_from_builtin
// CHECK-NEXT: pop.cast_from_builtin
// CHECK-NEXT: pop.add
// CHECK-NEXT: pop.cast_to_builtin
// CHECK: kgen.param.constant = <13>
// CHECK-NEXT: kgen.return

kgen.generator @raiseClosure<() -> E>() -> (index, index) {
  %idx0 = index.constant 0
  kgen.param.declare C = <15>
  %0 = lit.struct.create(field_0=%idx0) : (index) -> !kgen.declref<@raiseClosure_context>
  pop.compiler.global_store "raiseClosure_context_var", %0 : !kgen.declref<@raiseClosure_context>
  kgen.param.declare Fn: <A, B -> E>() -> index = <@raiseClosure_wrapper<C = C, A = #kgen.unbound, B = #kgen.unbound>>
  kgen.param.declare BoundFn: <A -> E>() -> index = <bind_signature(:<A, B -> E>() -> index Fn, #kgen.unbound, 1)>
  %1 = kgen.call @call_region<fn: <A -> E>() -> index = BoundFn -> Result = E>() : () -> index
  %2 = kgen.param.constant = <Result>
  kgen.return<Result> %1, %2 : index, index
}

kgen.export @raiseClosure
