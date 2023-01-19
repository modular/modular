// RUN: kgen-execute %s -execute -func="simple_partial_apply:f32()" -func="add_things:f32()" | FileCheck %s

kgen.func @my_fn(%arg0: index, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  kgen.return %arg1 : !pop.scalar<f32>
}

kgen.func @simple_partial_apply() -> f32 {
  %0 = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1.2">>
  %idx = index.constant 0
  %1 = kgen.addressof @my_fn : (index, !pop.scalar<f32>) -> !pop.scalar<f32>
  %2 = pop.partial_apply %1(?, %0) : (index, !pop.scalar<f32>) -> !pop.scalar<f32>
  %3 = pop.call_indirect %2(%idx) : !pop.closure<(index) -> !pop.scalar<f32>>
  %4 = pop.cast_to_builtin %3: !pop.scalar<f32> to f32
  kgen.return %4 : f32
}

kgen.func @add(%arg0: !pop.scalar<f32>, %arg1: !pop.scalar<f32>) -> !pop.scalar<f32> {
  %0 = pop.add %arg0, %arg1 : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

kgen.func @add_things() -> f32 {
  %one = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"1.0">>
  %two = kgen.param.constant: !pop.scalar<f32> = <#pop.simd<"2.0">>
  %addAddr = kgen.addressof @add : (!pop.scalar<f32>, !pop.scalar<f32>) -> !pop.scalar<f32>
  %addTwo = pop.partial_apply %addAddr(?, %two) : (!pop.scalar<f32>, !pop.scalar<f32>) -> !pop.scalar<f32>
  %0 = pop.call_indirect %addTwo(%one) : !pop.closure<(!pop.scalar<f32>) -> !pop.scalar<f32>>
  %1 = pop.cast_to_builtin %0 : !pop.scalar<f32> to f32
  kgen.return %1 : f32
}

kgen.export[@simple_partial_apply, @add_things]

// CHECK: --- 'simple_partial_apply' returned 1.2{{[0-9]+}}
// CHECK: --- 'add_things' returned 3.0{{[0-9]+}}

