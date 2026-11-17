// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

// Signed floor division by a constant normally lowers to Ben Joffe's
// multiply-high sequence, which multiplies on the 2W-bit product: an i128
// multiply for a 64-bit dividend. Apple's GPU compiler rejects i128 and fails
// at pipeline creation, so on Metal a 64-bit dividend keeps the generic
// sequence. Dividends of 32 bits and narrower widen to at most i64 and keep the
// multiply-high sequence. The non-Metal magic sequence is covered by
// @floordiv_const in simd.mlir.

module attributes {M.target_info = #M.target<
    triple = "air64-apple-macosx",
    arch = "apple-m5",
    features = "+metal4_0,+air2_8_0",
    data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32",
    simd_bit_width = 128>} {

// CHECK-LABEL: @floordiv_const
kgen.func @floordiv_const(
  %arg0 : !kgen.scalar<si32>, %arg1 : !kgen.scalar<si64>
) -> (!kgen.scalar<si32>, !kgen.scalar<si64>, !kgen.scalar<si64>) {
  // CHECK-DAG: [[X:%.*]] = builtin.unrealized_conversion_cast %arg0 : !kgen.scalar<si32> to i32
  // CHECK-DAG: [[V:%.*]] = builtin.unrealized_conversion_cast %arg1 : !kgen.scalar<si64> to i64
  %c3 = kgen.param.constant: scalar<si32> = <<3>>
  %c7 = kgen.param.constant: scalar<si64> = <<7>>
  %cm7 = kgen.param.constant: scalar<si64> = <<-7>>

  // 32-bit dividend: the 2W-bit product is i64, which Metal compiles.
  // CHECK: [[M:%.*]] = llvm.mlir.constant(1431655766 : i32) : i32
  // CHECK-NEXT: [[C31:%.*]] = llvm.mlir.constant(31 : i32) : i32
  // CHECK-NEXT: [[SIGN:%.*]] = llvm.ashr [[X]], [[C31]] : i32
  // CHECK-NEXT: [[MSEL:%.*]] = llvm.add [[M]], [[SIGN]] : i32
  // CHECK-NEXT: [[XW:%.*]] = llvm.sext [[X]] : i32 to i64
  // CHECK-NEXT: [[MW:%.*]] = llvm.sext [[MSEL]] : i32 to i64
  // CHECK-NEXT: [[PROD:%.*]] = llvm.mul [[XW]], [[MW]] : i64
  // CHECK-NEXT: [[C32:%.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK-NEXT: [[HI:%.*]] = llvm.ashr [[PROD]], [[C32]] : i64
  // CHECK-NEXT: llvm.trunc [[HI]] : i64 to i32
  %0 = pop.floordiv %arg0, %c3 : !kgen.scalar<si32>

  // 64-bit dividends, positive and negative, keep the generic sequence: their
  // 2W-bit product would be an i128 multiply.
  // CHECK: [[SDIV:%.*]] = llvm.sdiv [[V]], %{{.*}} : i64
  // CHECK-NEXT: [[MUL:%.*]] = llvm.mul [[SDIV]], %{{.*}} : i64
  // CHECK-NEXT: [[CMP:%.*]] = llvm.icmp "eq" [[MUL]], [[V]] : i64
  // CHECK-NEXT: llvm.xor [[V]], %{{.*}} : i64
  // CHECK: llvm.mlir.constant(63 : i64) : i64
  // CHECK: llvm.select
  // CHECK: llvm.add
  %1 = pop.floordiv %arg1, %c7 : !kgen.scalar<si64>

  // CHECK: llvm.sdiv [[V]], %{{.*}} : i64
  // CHECK: llvm.select
  // CHECK: llvm.add
  %2 = pop.floordiv %arg1, %cm7 : !kgen.scalar<si64>

  // CHECK-NOT: i128
  hlcf.return %0, %1, %2 : !kgen.scalar<si32>, !kgen.scalar<si64>, !kgen.scalar<si64>
}

}

// -----

// The widening is disabled by target capability, not for every target: a
// non-Metal target still uses the 128-bit multiply-high sequence for i64.
module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch="sm_90", data_layout="">} {

// CHECK-LABEL: @floordiv_const_nvptx
kgen.func @floordiv_const_nvptx(%arg0: !kgen.scalar<si64>) -> !kgen.scalar<si64> {
  %c7 = kgen.param.constant: scalar<si64> = <<7>>
  // CHECK: llvm.sext %{{.*}} : i64 to i128
  // CHECK: llvm.mul {{.*}} : i128
  // CHECK: llvm.trunc %{{.*}} : i128 to i64
  // CHECK-NOT: llvm.sdiv
  %0 = pop.floordiv %arg0, %c7 : !kgen.scalar<si64>
  hlcf.return %0 : !kgen.scalar<si64>
}

}
