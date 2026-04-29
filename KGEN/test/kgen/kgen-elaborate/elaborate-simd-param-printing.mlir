// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck %s

// Test that IREvaluatorContext::printParamValue (via get_type_name) correctly
// formats SIMD-typed struct parameters for all supported dtypes.  This
// exercises all branches of POP::printDTypeValue in a single elaboration pass.
//
// Related: MOCO-3651.

kgen.struct.generator @SIMDParamStruct<
    i: !pop.scalar<si32>,
    f: !pop.scalar<f32>,
    b: !pop.scalar<bool>,
    x: !pop.scalar<index>,
    u: !pop.scalar<uindex>,
    v: !pop.simd<4, si32>
> = struct_inst<"SIMDParamStruct"> {}

// CHECK-LABEL: kgen.func export @test_simd_param_printing
kgen.generator export @test_simd_param_printing() {
  // CHECK-NEXT: constant: string = <"SIMDParamStruct[42 : SIMD[DType.int32, 1], 1.5 : SIMD[DType.float32, 1], True : SIMD[DType.bool, 1], 7 : SIMD[DType.int, 1], 8 : SIMD[DType.uint, 1], [1, 2, 3, 4] : SIMD[DType.int32, 4]]">
  kgen.param.constant: string = <#kgen.get_type_name<
    #kgen.genref<@SIMDParamStruct<
      :!pop.scalar<si32> #pop<simd 42>,
      :!pop.scalar<f32> #pop<simd "1.5">,
      :!pop.scalar<bool> #pop<simd true>,
      :!pop.scalar<index> #pop<simd 7>,
      :!pop.scalar<uindex> #pop<simd 8>,
      :!pop.simd<4, si32> #pop<simd<1, 2, 3, 4>>>>,
    false>>
  kgen.return
}
