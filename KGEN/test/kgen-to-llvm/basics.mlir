// RUN: kgen-opt -convert-kgen-to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.func @trivial_kernel(%arg0: i32)
// CHECK-NEXT: llvm.return %arg0 : i32
kgen.kernel @trivial_kernel(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// CHECK-LABEL: llvm.func @"float_constant_f32,value=1.1283791670955126,type=f32"() -> f32
// CHECK: [[CST:%[0-9]+]] = llvm.mlir.constant(1.1283791670955126 : f64) : f64
// CHECK: [[TRUNC:%[0-9]+]] = llvm.fptrunc [[CST]] : f64 to f32
// CHECK: llvm.return [[TRUNC]] : f32
kgen.kernel @"float_constant_f32,value=1.1283791670955126,type=f32"() -> !meta.scalar<f32> {
  %0 = kgen.param.value : f64 = <1.1283791670955126>
  %1 = llvm.fptrunc %0 : f64 to f32
  %2 = meta.cast_from_builtin %1 : f32 to !meta.scalar<f32>
  kgen.return  %2 : !meta.scalar<f32>
}

// CHECK-LABEL: llvm.func @"mul_f32,type=f32"(%arg0: f32, %arg1: f32) -> f32
// CHECK: [[OUT:%[0-9]+]] = llvm.fmul %arg0, %arg1
// CHECK: llvm.return [[OUT]] : f32
kgen.kernel @"mul_f32,type=f32"(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
  %0 = meta.cast_to_builtin %arg0 : !meta.scalar<f32> to f32
  %1 = meta.cast_to_builtin %arg1 : !meta.scalar<f32> to f32
  %2 = llvm.fmul %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<f32>
  kgen.return %3 : !meta.scalar<f32>
}

// CHECK-LABEL: llvm.func @"void,type=f32"(%arg0: f32, %arg1: f32)
// CHECK: [[OUT:%[0-9]+]] = llvm.fmul %arg0, %arg1
// CHECK: llvm.return
kgen.kernel @"void,type=f32"(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) {
  %0 = meta.cast_to_builtin %arg0 : !meta.scalar<f32> to f32
  %1 = meta.cast_to_builtin %arg1 : !meta.scalar<f32> to f32
  %2 = llvm.fmul %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<f32>
  kgen.return
}

// CHECK-LABEL: llvm.func @"struct,type=f32"(%arg0: f32, %arg1: f32) -> !llvm.struct<(f32, f32)>
// CHECK: [[OUT:%[0-9]+]] = llvm.fmul %arg0, %arg1
// CHECK: [[UNDEF:%[0-9]+]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
// CHECK: [[ONE:%[0-9]+]] = llvm.insertvalue [[OUT]], [[UNDEF]][0] : !llvm.struct<(f32, f32)>
// CHECK: [[TWO:%[0-9]+]] = llvm.insertvalue [[OUT]], [[ONE]][1] : !llvm.struct<(f32, f32)>
// CHECK: llvm.return [[TWO]]
kgen.kernel @"struct,type=f32"(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> (!meta.scalar<f32>, !meta.scalar<f32>) {
  %0 = meta.cast_to_builtin %arg0 : !meta.scalar<f32> to f32
  %1 = meta.cast_to_builtin %arg1 : !meta.scalar<f32> to f32
  %2 = llvm.fmul %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<f32>
  kgen.return %3, %3 : !meta.scalar<f32>, !meta.scalar<f32>
}
