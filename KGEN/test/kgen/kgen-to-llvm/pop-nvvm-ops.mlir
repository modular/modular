// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(lower-kgen-to-llvm,llvm.func(lower-pop-to-llvm,canonicalize))' %s | FileCheck %s

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_90", data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", simd_bit_width = 128>} {
  // CHECK-LABEL: llvm.func @nvvm_wgmma_async
  kgen.func export @nvvm_wgmma_async(%arg0: !pop.scalar<si64>, %arg1: !pop.scalar<si64>, %arg2: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
    %0 = pop.cast_to_builtin %arg0 : !pop.scalar<si64> to i64
    %1 = pop.cast_to_builtin %arg1 : !pop.scalar<si64> to i64
    // CHECK-DAG: %[[VEC:.*]] = llvm.mlir.undef : vector<4xf32>
    // CHECK-DAG: %[[C3:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG: %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %5 = llvm.extractelement %arg2[%4 : i32] : vector<4xf32>
    // CHECK-DAG: %6 = llvm.extractelement %arg2[%3 : i32] : vector<4xf32>
    // CHECK-DAG: %7 = llvm.extractelement %arg2[%2 : i32] : vector<4xf32>
    // CHECK-DAG: %8 = llvm.extractelement %arg2[%1 : i32] : vector<4xf32>
    // CHECK: %[[SVAL:.*]] = llvm.inline_asm has_side_effects
    // CHECK-SAME: wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32
    // CHECK: %[[SE_0:.+]] = llvm.extractvalue %[[SVAL]][0] : !llvm.struct<(f32, f32, f32, f32)>
    // CHECK: %[[VEC_0:.+]] = llvm.insertelement %[[SE_0]], %[[VEC]][%4 : i32] : vector<4xf32>
    // CHECK: %[[SE_1:.+]] = llvm.extractvalue %[[SVAL]][1] : !llvm.struct<(f32, f32, f32, f32)>
    // CHECK: %[[VEC_1:.+]] = llvm.insertelement %[[SE_1]], %[[VEC_0]][%3 : i32] : vector<4xf32>
    // CHECK: %[[SE_2:.+]] = llvm.extractvalue %[[SVAL]][2] : !llvm.struct<(f32, f32, f32, f32)>
    // CHECK: %[[VEC_2:.+]] = llvm.insertelement %[[SE_2]], %[[VEC_1]][%2 : i32] : vector<4xf32>
    // CHECK: %[[SE_3:.+]] = llvm.extractvalue %[[SVAL]][3] : !llvm.struct<(f32, f32, f32, f32)>
    // CHECK: %[[RES:.+]] = llvm.insertelement %[[SE_3]], %[[VEC_2]][%1 : i32] : vector<4xf32>
    // return %[[RES]]
    %2 = pop.nvvm.wgmma.mma_async %0 %1 %arg2 tf32 tf32 f32 {layout_a = "row" : !kgen.string, layout_b = "col" : !kgen.string, shape_k = 8 : index, shape_m = 64 : index, shape_n = 8 : index, scale_d = 1 : index, scale_a = 1 : index, scale_b = 1 : index} : <4, f32> -> <4, f32>
    kgen.return %2 : !pop.simd<4, f32>
  }

  // CHECK-LABEL: llvm.func @nvvm_wgmma_async_scale_out
  kgen.func export @nvvm_wgmma_async_scale_out(%arg0: !pop.scalar<si64>, %arg1: !pop.scalar<si64>, %arg2: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
    %0 = pop.cast_to_builtin %arg0 : !pop.scalar<si64> to i64
    %1 = pop.cast_to_builtin %arg1 : !pop.scalar<si64> to i64
    // CHECK-DAG: %[[VEC:.*]] = llvm.mlir.undef : vector<4xf32>
    // CHECK-DAG: %[[C3:.*]] = llvm.mlir.constant(3 : i32) : i32
    // CHECK-DAG: %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-DAG: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-DAG: %5 = llvm.extractelement %arg2[%4 : i32] : vector<4xf32>
    // CHECK-DAG: %6 = llvm.extractelement %arg2[%3 : i32] : vector<4xf32>
    // CHECK-DAG: %7 = llvm.extractelement %arg2[%2 : i32] : vector<4xf32>
    // CHECK-DAG: %8 = llvm.extractelement %arg2[%1 : i32] : vector<4xf32>
    // CHECK: %[[SVAL:.*]] = llvm.inline_asm has_side_effects
    // CHECK-SAME: wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32
    // CHECK: %[[SE_0:.+]] = llvm.extractvalue %[[SVAL]][0] : !llvm.struct<(f32, f32, f32, f32)>
    // CHECK: %[[VEC_0:.+]] = llvm.insertelement %[[SE_0]], %[[VEC]][%4 : i32] : vector<4xf32>
    // CHECK: %[[SE_1:.+]] = llvm.extractvalue %[[SVAL]][1] : !llvm.struct<(f32, f32, f32, f32)>
    // CHECK: %[[VEC_1:.+]] = llvm.insertelement %[[SE_1]], %[[VEC_0]][%3 : i32] : vector<4xf32>
    // CHECK: %[[SE_2:.+]] = llvm.extractvalue %[[SVAL]][2] : !llvm.struct<(f32, f32, f32, f32)>
    // CHECK: %[[VEC_2:.+]] = llvm.insertelement %[[SE_2]], %[[VEC_1]][%2 : i32] : vector<4xf32>
    // CHECK: %[[SE_3:.+]] = llvm.extractvalue %[[SVAL]][3] : !llvm.struct<(f32, f32, f32, f32)>
    // CHECK: %[[RES:.+]] = llvm.insertelement %[[SE_3]], %[[VEC_2]][%1 : i32] : vector<4xf32>
    // return %[[RES]]
    %2 = pop.nvvm.wgmma.mma_async %0 %1 %arg2 tf32 tf32 f32 {layout_a = "row" : !kgen.string, layout_b = "col" : !kgen.string, shape_k = 8 : index, shape_m = 64 : index, shape_n = 8 : index, scale_d = 0 : index, scale_a = 1 : index, scale_b = 1 : index} : <4, f32> -> <4, f32>
    kgen.return %2 : !pop.simd<4, f32>
  }

  // CHECK-LABEL: llvm.func @nvvm_wgmma_async_inline_array
  kgen.func export @nvvm_wgmma_async_inline_array(%arg0: !pop.scalar<si64>, %arg1: !pop.scalar<si64>, %arg2: !pop.array<4, f32>) -> !pop.array<4, f32> {
    %0 = pop.cast_to_builtin %arg0 : !pop.scalar<si64> to i64
    %1 = pop.cast_to_builtin %arg1 : !pop.scalar<si64> to i64
    // CHECK-DAG: %[[ARR:.*]] = llvm.mlir.undef : !llvm.array<4 x f32>
    // CHECK-DAG: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-DAG: %2 = llvm.extractvalue %arg2[0] : !llvm.array<4 x f32> 
    // CHECK-DAG: %3 = llvm.extractvalue %arg2[1] : !llvm.array<4 x f32> 
    // CHECK-DAG: %4 = llvm.extractvalue %arg2[2] : !llvm.array<4 x f32> 
    // CHECK-DAG: %5 = llvm.extractvalue %arg2[3] : !llvm.array<4 x f32> 
    // CHECK: %[[SVAL:.*]] = llvm.inline_asm has_side_effects
    // CHECK-SAME: wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32
    // CHECK: %[[SE_0:.+]] = llvm.extractvalue %[[SVAL]][0] : !llvm.struct<(f32, f32, f32, f32)>
    // CHECK: %[[ARR_0:.+]] = llvm.insertvalue %[[SE_0]], %[[ARR]][0] : !llvm.array<4 x f32>
    // CHECK: %[[SE_1:.+]] = llvm.extractvalue %[[SVAL]][1] : !llvm.struct<(f32, f32, f32, f32)>
    // CHECK: %[[ARR_1:.+]] = llvm.insertvalue %[[SE_1]], %[[ARR_0]][1] : !llvm.array<4 x f32>
    // CHECK: %[[SE_2:.+]] = llvm.extractvalue %[[SVAL]][2] : !llvm.struct<(f32, f32, f32, f32)>
    // CHECK: %[[ARR_2:.+]] = llvm.insertvalue %[[SE_2]], %[[ARR_1]][2] : !llvm.array<4 x f32>
    // CHECK: %[[SE_3:.+]] = llvm.extractvalue %[[SVAL]][3] : !llvm.struct<(f32, f32, f32, f32)>
    // CHECK: %[[ARR_3:.+]] = llvm.insertvalue %[[SE_3]], %[[ARR_2]][3] : !llvm.array<4 x f32>
    // CHECK: llvm.return %[[ARR_3]] : !llvm.array<4 x f32>
    %2 = pop.nvvm.wgmma.mma_async.inline_array %0 %1 %arg2 tf32 tf32 f32 {layout_a = "row" : !kgen.string, layout_b = "col" : !kgen.string, shape_k = 8 : index, shape_m = 64 : index, shape_n = 8 : index, scale_d = 1 : index, scale_a = 1 : index, scale_b = 1 : index} : !pop.array<4, f32> -> !pop.array<4, f32>
    kgen.return %2 : !pop.array<4, f32>
  }

  // CHECK-LABEL: llvm.func @kgen_fp8_param_constant
  kgen.func export @kgen_fp8_param_constant() -> (f8E4M3, f8E5M2) {
    // CHECK: llvm.mlir.constant(56 : i8) : i8
    %0 = kgen.param.constant: f8E4M3 = <1.>
    // CHECK: llvm.mlir.constant(60 : i8) : i8
    %1 = kgen.param.constant: f8E5M2 = <1.>
    kgen.return %0, %1 : f8E4M3, f8E5M2
  }
}
