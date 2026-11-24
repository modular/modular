// RUN: not kgen-opt -lower-global-pop-to-llvm -verify-diagnostics %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @external_call(%a: !kgen.simd<1, ui32>) {
  pop.external_call @foo(%a) : (!kgen.simd<1, ui32>) -> ()
  // expected-error @below {{existing function with conflicting signature}}
  // expected-error @below {{failed to legalize}}
  %0 = pop.external_call @foo(%a) : (!kgen.simd<1, ui32>) -> !kgen.simd<4, f64>
  hlcf.return
}

// The negative-count case is rejected by the op verifier; see
// pop-ir/pop-errors.mlir.
kgen.func @variadic_too_many_fixed_args(%a: !kgen.simd<1, ui32>) {
  // expected-error @below {{'numFixedArgs' must not exceed the number of call operands: expected at most 1, found 2}}
  // expected-error @below {{failed to legalize}}
  pop.external_call @bar(%a)
    : (!kgen.simd<1, ui32>) -> () <numFixedArgs = 2 : index>
  hlcf.return
}

// LowerArgConventions expands the pack operand. It must not reach this pass.
kgen.func @unexpanded_pack_operand(%a: !kgen.struct<(scalar<si32>)>) {
  // expected-error @below {{'pop.external_call' op pack operand must be expanded by lower-arg-conventions}}
  // expected-error @below {{failed to legalize}}
  pop.external_call @baz(%a)
    : (!kgen.struct<(scalar<si32>)>) -> () <expandPackOperandIdx = 0>
  hlcf.return
}

}
