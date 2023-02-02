// RUN: kgen-execute %s -execute -func="simple_partial_apply:f32()" -func="add_things:f32()" -func="nested_closure:f32()" | FileCheck %s

// TODO: kgen-execute canonicalizes partial_apply -> call_indirect, which means
// that we have to split the ops across a call. In the future, when kgen-execute
// can execute llvm.funcs, we should use kgen-opt to apply the pass pipeline without
// canonicalization, then run kgen-execute.

kgen.func @call_indirect_wrapper(%closure : !pop.closure<(index) -> !pop.scalar<f32>>, %idx: index) -> f32 {
  %0 = pop.call_indirect %closure(%idx) : !pop.closure<(index) -> !pop.scalar<f32>>
  %1 = pop.cast_to_builtin %0: !pop.scalar<f32> to f32
  kgen.return %1 : f32
}

kgen.func @my_fn(%arg0: index, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  kgen.return %arg1 : !pop.scalar<f32>
}

kgen.func @simple_partial_apply() -> f32 {
  %idx = index.constant 0
  %0 = kgen.param.constant: scalar<f32> = <<"1.2">>
  %1 = kgen.addressof @my_fn : (index, !pop.scalar<f32>) -> !pop.scalar<f32>
  %2 = pop.partial_apply %1(?, %0) : (index, !pop.scalar<f32>) -> !pop.scalar<f32>
  %3 = kgen.call @call_indirect_wrapper(%2, %idx) : (!pop.closure<(index) -> !pop.scalar<f32>>, index) -> f32
  kgen.return %3 : f32
}

kgen.func @add(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  %0 = pop.add %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

kgen.func @add_things_wrapper(%addTwo: !pop.closure<(!pop.scalar<f32>) -> !pop.scalar<f32>>, %num: !pop.scalar<f32>) -> f32 {
  %0 = pop.call_indirect %addTwo(%num) : !pop.closure<(!pop.scalar<f32>) -> !pop.scalar<f32>>
  %1 = pop.cast_to_builtin %0 : !pop.scalar<f32> to f32
  kgen.return %1 : f32
}

kgen.func @add_things() -> f32 {
  %one = kgen.param.constant: scalar<f32> = <<"1.0">>
  %two = kgen.param.constant: scalar<f32> = <<"2.0">>
  %addAddr = kgen.addressof @add : (!pop.scalar<f32>, !pop.scalar<f32>) -> !pop.scalar<f32>
  %addTwo = pop.partial_apply %addAddr(?, %two) : (!pop.scalar<f32>, !pop.scalar<f32>) -> !pop.scalar<f32>
  %res = kgen.call @add_things_wrapper(%addTwo, %one) : (!pop.closure<(!pop.scalar<f32>) -> !pop.scalar<f32>>, !pop.scalar<f32>) -> f32
  kgen.return %res : f32
}

kgen.func @actually_use_it(%closure1: !pop.closure<() -> !pop.scalar<f32>>) -> !pop.scalar<f32> {
  %result = pop.call_indirect %closure1() : !pop.closure<() -> !pop.scalar<f32>>
  kgen.return %result : !pop.scalar<f32>
}

kgen.func @use_closure(%closure0 : !pop.closure<(!pop.scalar<f32>) -> !pop.scalar<f32>>) -> f32 {
  %one = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1.0">>
  %closure1 = pop.partial_apply %closure0(%one) : !pop.closure<(!pop.scalar<f32>) -> !pop.scalar<f32>>
  %result = kgen.call @actually_use_it(%closure1) : (!pop.closure<() -> !pop.scalar<f32>>) -> !pop.scalar<f32>
  %0 = pop.cast_to_builtin %result : !pop.scalar<f32> to f32
  kgen.return %0 : f32
}

kgen.func @nested_closure() -> f32 {
 %two = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2.0">>
 %addAddr = kgen.addressof @add : (!pop.scalar<f32>, !pop.scalar<f32>) -> !pop.scalar<f32>
 %addTwo = pop.partial_apply %addAddr(?, %two) : (!pop.scalar<f32>, !pop.scalar<f32>) -> !pop.scalar<f32>
 %result = kgen.call @use_closure(%addTwo) : (!pop.closure<(!pop.scalar<f32>) -> !pop.scalar<f32>>) -> f32
 kgen.return %result : f32
}

kgen.export @simple_partial_apply
kgen.export @add_things
kgen.export @nested_closure

// CHECK: --- 'simple_partial_apply' returned 1.2{{[0-9]+}}
// CHECK: --- 'add_things' returned 3.0{{[0-9]+}}
// CHECK: --- 'nested_closure' returned 3.0{{[0-9]+}}
