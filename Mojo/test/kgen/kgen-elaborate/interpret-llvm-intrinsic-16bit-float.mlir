// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" -allow-unregistered-dialect | FileCheck %s
// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=true" -allow-unregistered-dialect | FileCheck %s

// The interpreter folds `llvm.*` intrinsics through LLVM's constant folder, so
// it has to lower f16 and bf16 values to LLVM's `half` and `bfloat` types.

kgen.generator @sqrt_f16(%arg0: !kgen.simd<4, f16>) -> !kgen.simd<4, f16> {
  %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.sqrt", (%arg0) : (!kgen.simd<4, f16>) -> !kgen.simd<4, f16>
  hlcf.return %0 : !kgen.simd<4, f16>
}

kgen.generator @sqrt_bf16(%arg0: !kgen.simd<4, bf16>) -> !kgen.simd<4, bf16> {
  %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.sqrt", (%arg0) : (!kgen.simd<4, bf16>) -> !kgen.simd<4, bf16>
  hlcf.return %0 : !kgen.simd<4, bf16>
}

kgen.generator @sin_f16(%arg0: !kgen.scalar<f16>) -> !kgen.scalar<f16> {
  %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.sin", (%arg0) : (!kgen.scalar<f16>) -> !kgen.scalar<f16>
  hlcf.return %0 : !kgen.scalar<f16>
}

kgen.generator @cos_bf16(%arg0: !kgen.scalar<bf16>) -> !kgen.scalar<bf16> {
  %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.cos", (%arg0) : (!kgen.scalar<bf16>) -> !kgen.scalar<bf16>
  hlcf.return %0 : !kgen.scalar<bf16>
}

// CHECK-LABEL: kgen.func export @testSqrt
kgen.generator export @testSqrt() -> !kgen.simd<4, f16> {
  kgen.param.declare H0: simd<4, f16> = <<"2.0", "4.0", "9.0", "100.0">>
  kgen.param.declare H1: simd<4, f16> = <apply(:(!kgen.simd<4, f16>) -> !kgen.simd<4, f16> @sqrt_f16, H0)>
  // CHECK: kgen.param.constant: simd<4, f16> = <<"1.4141", "2", "3", "10">>
  %0 = kgen.param.constant: !kgen.simd<4, f16> = <H1>

  kgen.param.declare B0: simd<4, bf16> = <<"2.0", "4.0", "9.0", "100.0">>
  kgen.param.declare B1: simd<4, bf16> = <apply(:(!kgen.simd<4, bf16>) -> !kgen.simd<4, bf16> @sqrt_bf16, B0)>
  // CHECK: kgen.param.constant: simd<4, bf16> = <<"1.414", "2", "3", "10">>
  %1 = kgen.param.constant: !kgen.simd<4, bf16> = <B1>

  hlcf.return %0 : !kgen.simd<4, f16>
}

// CHECK-LABEL: kgen.func export @testSinCos
kgen.generator export @testSinCos() -> !kgen.scalar<f16> {
  kgen.param.declare H0: scalar<f16> = <"0.5">
  kgen.param.declare H1: scalar<f16> = <apply(:(!kgen.scalar<f16>) -> !kgen.scalar<f16> @sin_f16, H0)>
  // CHECK: kgen.param.constant: scalar<f16> = <"0.47949">
  %0 = kgen.param.constant: !kgen.scalar<f16> = <H1>

  kgen.param.declare B0: scalar<bf16> = <"0.5">
  kgen.param.declare B1: scalar<bf16> = <apply(:(!kgen.scalar<bf16>) -> !kgen.scalar<bf16> @cos_bf16, B0)>
  // CHECK: kgen.param.constant: scalar<bf16> = <"0.8789">
  %1 = kgen.param.constant: !kgen.scalar<bf16> = <B1>

  hlcf.return %0 : !kgen.scalar<f16>
}
