module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_90", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", relocation_model = "pic", simd_bit_width = 128>, llvm.data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", llvm.target_triple = "nvptx64-nvidia-cuda"} {

// RUN: kgen-opt %s -lower-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

//      CHECK: dso_local ptx_kernel void @test_async_copy_kernel
// CHECK-NEXT: call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d
// CHECK-NEXT: ret

kgen.func export @test_async_copy_kernel(%arg0: !llvm.ptr<7>, %arg1: !llvm.ptr, %arg2: !llvm.ptr<3>, %arg3: i32, %arg4: i32) {
  nvvm.cp.async.bulk.tensor.shared.cluster.global %arg0, %arg1, %arg2, box[%arg3, %arg4] : !llvm.ptr<7>, !llvm.ptr
  hlcf.return
}

// CHECK-LABEL: @abs_simd
kgen.func @abs_simd(
  %arg0: !kgen.scalar<f16>, %arg1: !kgen.simd<8, f16>,
  %arg2 : !kgen.scalar<bf16>, %arg3 : !kgen.simd<8, bf16>
) -> (
  !kgen.scalar<f16>, !kgen.simd<8, f16>, !kgen.scalar<bf16>, !kgen.simd<8, bf16>
) {
  // CHECK: [[F16:%.*]] = call half @llvm.nvvm.fabs.f16(half %0)
  %0 = pop.abs %arg0 : !kgen.scalar<f16>

  // CHECK: [[V8F_E0:%.*]] = call [[H_EXTRACT:<2 x half> @llvm.vector.extract.v2f16.v8f16]](<8 x half> %1, i64 0)
  // CHECK: [[V8F_R0:%.*]] = call <2 x half> @llvm.nvvm.fabs.v2f16(<2 x half> [[V8F_E0]])
  // CHECK: [[V8F_I0:%.*]] = call [[H_INSERT:<8 x half> @llvm.vector.insert.v8f16.v2f16]](<8 x half> poison, <2 x half> [[V8F_R0]], i64 0)
  // CHECK: [[V8F_E1:%.*]] = call [[H_EXTRACT]](<8 x half> %1, i64 2)
  // CHECK: [[V8F_R1:%.*]] = call <2 x half> @llvm.nvvm.fabs.v2f16(<2 x half> [[V8F_E1]])
  // CHECK: [[V8F_I1:%.*]] = call [[H_INSERT]](<8 x half> [[V8F_I0]], <2 x half> [[V8F_R1]], i64 2)
  // CHECK: [[V8F_E2:%.*]] = call [[H_EXTRACT]](<8 x half> %1, i64 4)
  // CHECK: [[V8F_R2:%.*]] = call <2 x half> @llvm.nvvm.fabs.v2f16(<2 x half> [[V8F_E2]])
  // CHECK: [[V8F_I2:%.*]] = call [[H_INSERT]](<8 x half> [[V8F_I1]], <2 x half> [[V8F_R2]], i64 4)
  // CHECK: [[V8F_E3:%.*]] = call [[H_EXTRACT]](<8 x half> %1, i64 6)
  // CHECK: [[V8F_R3:%.*]] = call <2 x half> @llvm.nvvm.fabs.v2f16(<2 x half> [[V8F_E3]])
  // CHECK: [[V8F_I3:%.*]] = call [[H_INSERT]](<8 x half> [[V8F_I2]], <2 x half> [[V8F_R3]], i64 6)
  %1 = pop.abs %arg1 : !kgen.simd<8, f16>

  // CHECK: [[BF16:%.*]] = call bfloat @llvm.nvvm.fabs.bf16(bfloat %2)
  %2 = pop.abs %arg2 : !kgen.scalar<bf16>

  // CHECK: [[V8B_E0:%.*]] = call [[B_EXTRACT:<2 x bfloat> @llvm.vector.extract.v2bf16.v8bf16]](<8 x bfloat> %3, i64 0)
  // CHECK: [[V8B_R0:%.*]] = call <2 x bfloat> @llvm.nvvm.fabs.v2bf16(<2 x bfloat> [[V8B_E0]])
  // CHECK: [[V8B_I0:%.*]] = call [[B_INSERT:<8 x bfloat> @llvm.vector.insert.v8bf16.v2bf16]](<8 x bfloat> poison, <2 x bfloat> [[V8B_R0]], i64 0)
  // CHECK: [[V8B_E1:%.*]] = call [[B_EXTRACT]](<8 x bfloat> %3, i64 2)
  // CHECK: [[V8B_R1:%.*]] = call <2 x bfloat> @llvm.nvvm.fabs.v2bf16(<2 x bfloat> [[V8B_E1]])
  // CHECK: [[V8B_I1:%.*]] = call [[B_INSERT]](<8 x bfloat> [[V8B_I0]], <2 x bfloat> [[V8B_R1]], i64 2)
  // CHECK: [[V8B_E2:%.*]] = call [[B_EXTRACT]](<8 x bfloat> %3, i64 4)
  // CHECK: [[V8B_R2:%.*]] = call <2 x bfloat> @llvm.nvvm.fabs.v2bf16(<2 x bfloat> [[V8B_E2]])
  // CHECK: [[V8B_I2:%.*]] = call [[B_INSERT]](<8 x bfloat> [[V8B_I1]], <2 x bfloat> [[V8B_R2]], i64 4)
  // CHECK: [[V8B_E3:%.*]] = call [[B_EXTRACT]](<8 x bfloat> %3, i64 6)
  // CHECK: [[V8B_R3:%.*]] = call <2 x bfloat> @llvm.nvvm.fabs.v2bf16(<2 x bfloat> [[V8B_E3]])
  // CHECK: [[V8B_I3:%.*]] = call [[B_INSERT]](<8 x bfloat> [[V8B_I2]], <2 x bfloat> [[V8B_R3]], i64 6)
  %3 = pop.abs %arg3 : !kgen.simd<8, bf16>

  // CHECK: ret { half, <8 x half>, bfloat, <8 x bfloat> }
  hlcf.return %0, %1, %2, %3 : !kgen.scalar<f16>, !kgen.simd<8, f16>, !kgen.scalar<bf16>, !kgen.simd<8, bf16>
}

}
