// RUN: kgen-opt -lower-to-llvm %s > %t
// RUN: cat %t | FileCheck %s --check-prefix=CHECK-MLIR
// RUN: kgen-translate %t --mlir-to-llvmir | FileCheck %s --check-prefix=CHECK-LLVMIR

module attributes {M.target_info = #M.target<triple = "air64-apple-macosx", arch = "apple-m3", data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", simd_bit_width = 128>} {

  // CHECK-MLIR: llvm.func @air.simdgroup.barrier(i32, i32) attributes {convergent}
  // CHECK-MLIR: llvm.func @air.wg.barrier(i32, i32) attributes {convergent}
  // CHECK-MLIR: llvm.func @air.thread_position_in_threadgroup.x() -> i32

  // CHECK-MLIR-LABEL: @test0{{.*}}"metal.kernel", "true"
  kgen.func export @test0() {
    %struct = kgen.param.constant: struct<()> = <{  }>
    // CHECK-MLIR: llvm.call @air.thread_position_in_threadgroup.x() : () -> i32
    %3 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.thread_position_in_threadgroup.x", (%struct) : (!kgen.struct<()>) -> !kgen.scalar<ui32>
    hlcf.return
  }

  // CHECK-MLIR-LABEL: @test1{{.*}}"metal.kernel", "true"
  kgen.func export @test1() {
    %struct = kgen.param.constant: struct<()> = <{  }>
    // CHECK-MLIR: llvm.call @air.thread_position_in_threadgroup.x() : () -> i32
    %3 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.thread_position_in_threadgroup.x", (%struct) : (!kgen.struct<()>) -> !kgen.scalar<ui32>
    hlcf.return
  }

  // CHECK-MLIR-LABEL: @test_barrier{{.*}}"metal.kernel", "true"
  kgen.func export @test_barrier() {
    %struct = kgen.param.constant: struct<(scalar<si32>, scalar<si32>)> = <{ 2, 1 }>
    // CHECK-MLIR: %[[PLACEHOLDER:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-MLIR: %[[BARRIER_TYPE:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-MLIR: llvm.call @air.wg.barrier(%[[BARRIER_TYPE]], %[[PLACEHOLDER]]) : (i32, i32) -> ()
    pop.call_llvm_intrinsic side_effecting<1> "llvm.air.wg.barrier", (%struct) : (!kgen.struct<(scalar<si32>, scalar<si32>)>) -> ()
    hlcf.return
  }

  kgen.func export @test_simdgroup_barrier() {
    %struct = kgen.param.constant: struct<(scalar<si32>, scalar<si32>)> = <{ 0, 4 }>
    // CHECK-MLIR: %[[PLACEHOLDER:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK-MLIR: %[[BARRIER_TYPE:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-MLIR: llvm.call @air.simdgroup.barrier(%[[BARRIER_TYPE]], %[[PLACEHOLDER]]) : (i32, i32) -> ()
    pop.call_llvm_intrinsic side_effecting<1> "llvm.air.simdgroup.barrier", (%struct) : (!kgen.struct<(scalar<si32>, scalar<si32>)>) -> ()
    hlcf.return
  }

  // CHECK-LLVMIR: declare void @air.wg.barrier(i32, i32) #[[BARRIER:[0-9]+]]
  // CHECK-LLVMIR: define{{.*}}@test0() #[[KERNEL:[0-9]+]]
  // CHECK-LLVMIR: attributes #[[BARRIER]] = { convergent }
  // CHECK-LLVMIR: attributes #[[KERNEL]] = {{.*}}"metal.kernel"="true"{{.*}}
}
