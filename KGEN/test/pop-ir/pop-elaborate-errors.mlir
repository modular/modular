// RUN: kgen-elaborate-opt -elaborate-generators %s -verify-diagnostics -split-input-file

// expected-error @below {{no viable expansions found}}
kgen.generator @impl(%a: !pop.simd<4, f32>) {
  // expected-note @below {{call expansion failed}}
  %0 = kgen.call @invalid_bitcast<2, :dtype ui32>(%a) : (!pop.simd<4, f32>) -> (!pop.simd<2, ui32>)
  kgen.return
}

// expected-note @below {{no viable expansions found}}
kgen.generator @invalid_bitcast<size, type: dtype>(%a: !pop.simd<4, f32>) -> !pop.simd<size, type> {
  // expected-note @below {{'!pop.simd<4, f32>' and result type '!pop.simd<2, ui32>' are cast incompatible}}
  %0 = pop.bitcast %a : !pop.simd<4, f32> to !pop.simd<size, type>
  kgen.return %0 : !pop.simd<size, type>
}

// -----

// expected-note @below {{failed to interpret function @out_of_range_read}}
kgen.generator @out_of_range_read() -> i32 {
  %0 = pop.stack_allocation 0 x i32
  // expected-note @below {{failed to interpret operation pop.load}}
  // expected-note @below {{address is out-of-bounds}}
  %1 = pop.load %0 : !pop.pointer<i32>
  kgen.return %1 : i32
}

// expected-error @below {{no viable expansions found}}
kgen.generator @call_it() {
  // expected-note @below {{failed to evaluate 'apply'}}
  %0 = kgen.param.constant: i32 = <apply(:() -> i32 @out_of_range_read)>
  kgen.return
}
