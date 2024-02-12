// RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s


module attributes {M.target_info = #M.target<triple="aarch64-linux-gnu", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @fadd_scalar
// CHECK: %[[LHS:.+]] = llvm.fpext
// CHECK: %[[RHS:.+]] = llvm.fpext
// CHECK: %[[F32_RES:.+]] = llvm.fadd %[[LHS]], %[[RHS]]  {fastmathFlags = #llvm.fastmath<contract>} : f32
// CHECK: %[[F16_RES:.+]] = llvm.fptrunc %[[F32_RES]] : f32 to bf16
kgen.func @fadd_scalar(%arg0 : !pop.scalar<bf16>, %arg1: !pop.scalar<bf16>) -> !pop.scalar<bf16> {
    %0 = pop.add %arg0, %arg1 : !pop.scalar<bf16>
    kgen.return %0 : !pop.scalar<bf16>
}

// CHECK-LABEL: @fadd_vector
// CHECK: %[[LHS:.+]] = llvm.fpext
// CHECK: %[[RHS:.+]] = llvm.fpext
// CHECK: %[[F32_RES:.+]] = llvm.fadd %[[LHS]], %[[RHS]]  {fastmathFlags = #llvm.fastmath<contract>} : vector<8xf32>
// CHECK: %[[F16_RES:.+]] = llvm.fptrunc %[[F32_RES]] : vector<8xf32> to vector<8xbf16>
kgen.func @fadd_vector(%arg0 : !pop.simd<8, bf16>, %arg1: !pop.simd<8, bf16>) -> !pop.simd<8, bf16> {
    %0 = pop.add %arg0, %arg1 : !pop.simd<8, bf16>
    kgen.return %0 : !pop.simd<8, bf16>
}
}
