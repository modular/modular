// RUN: kgen-opt -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

!struct1 = !pop.struct<struct<f32>, array<4, f32>>

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @struct_construct
kgen.func @struct_construct(%a: !pop.struct<f32>, %b: !pop.array<4, f32>) -> !struct1 {
  // CHECK: %[[S0:.*]] = llvm.mlir.undef : !llvm.struct<(struct<(f32)>, array<4 x f32>)>
  // CHECK: %[[S1:.*]] = llvm.insertvalue %{{.*}}, %[[S0]][0]
  // CHECK: %[[S2:.*]] = llvm.insertvalue %{{.*}}, %[[S1]][1]
  %0 = pop.struct.create(%a, %b) : !struct1
  kgen.return %0 : !struct1
}

// CHECK-LABEL: @struct_construct_one
kgen.func @struct_construct_one(%a: f32) -> !pop.struct<f32> {
  // CHECK: %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(f32)>
  // CHECK: unrealized_conversion_cast %1 : !llvm.struct<(f32)> to !pop.struct<f32>
  %0 = pop.struct.create(%a) : !pop.struct<f32>
  kgen.return %0 : !pop.struct<f32>
}

// CHECK-LABEL: @struct_insert
kgen.func @struct_insert(%a: !pop.struct<f32, f32>, %b: f32) -> !pop.struct<f32, f32> {
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[0] : !llvm.struct<(f32, f32)>
  %0 = pop.struct.replace %b, %a[0] : !pop.struct<f32, f32>
  kgen.return %0 : !pop.struct<f32, f32>
}

// CHECK-LABEL: @struct_insert_one
kgen.func @struct_insert_one(%a: !pop.struct<f32>, %b: f32) -> !pop.struct<f32> {
  // CHECK: llvm.insertvalue %arg1, %0[0] : !llvm.struct<(f32)>
  %0 = pop.struct.replace %b, %a[0] : !pop.struct<f32>
  kgen.return %0 : !pop.struct<f32>
}

// CHECK-LABEL: @struct_extract
kgen.func @struct_extract(%a: !pop.struct<f32, f32>) -> f32 {
  // CHECK: llvm.extractvalue %{{.*}}[0]
  %0 = pop.struct.extract %a[0] : !pop.struct<f32, f32>
  kgen.return %0 : f32
}

// CHECK-LABEL: @struct_extract_one
kgen.func @struct_extract_one(%a: !pop.struct<f32>) -> f32 {
  // CHECK: llvm.extractvalue %0[0] : !llvm.struct<(f32)>
  %0 = pop.struct.extract %a[0] : !pop.struct<f32>
  kgen.return %0 : f32
}

// CHECK-LABEL: @struct_gep
kgen.func @struct_gep(%a: !kgen.pointer<struct<i32, i64>>) -> !kgen.pointer<i64> {
  // CHECK: llvm.getelementptr %{{.*}}[0, 1] : (!llvm.ptr<struct<(i32, i64)>>) -> !llvm.ptr<i64>
  %0 = pop.struct.gep %a[1] : <struct<i32, i64>>
  kgen.return %0 : !kgen.pointer<i64>
}

// CHECK-LABEL: @struct_gep_one
kgen.func @struct_gep_one(%a: !kgen.pointer<struct<i32>>) -> !kgen.pointer<i32> {
  // CHECK: llvm.getelementptr %0[0, 0] : (!llvm.ptr<struct<(i32)>>) -> !llvm.ptr<i32>
  %0 = pop.struct.gep %a[0] : <struct<i32>>
  kgen.return %0 : !kgen.pointer<i32>
}

}
