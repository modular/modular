// RUN: kgen-opt -split-input-file -convert-kgen-to-llvm="index-bitwidth=64" %s | FileCheck %s
// RUN: kgen-opt -split-input-file -convert-kgen-to-llvm="index-bitwidth=32" %s | FileCheck %s --check-prefixes=INDEX32

// CHECK-LABEL: llvm.func @trivial(%arg0: i32)
// CHECK-NEXT: llvm.return %arg0 : i32
kgen.func @trivial(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// -----

// CHECK-LABEL: llvm.func @produces_result
kgen.func @produces_result<() -> result>() {
  // CHECK: llvm.return
  kgen.return<result = 42>
}

// -----

// CHECK-LABEL: llvm.func @convert_meta_types
// CHECK-SAME: %{{.*}}: f32
// CHECK-SAME: %{{.*}}: !llvm.ptr<f32>
// CHECK-SAME: %{{.*}}: vector<4xf32>
// CHECK-SAME: %{{.*}}: !llvm.struct<(i64, ptr<i64>)>
// CHECK-SAME: %{{.*}}: !llvm.struct<(i64, ptr<f32>)>

// INDEX32-LABEL: llvm.func @convert_meta_types
// INDEX32-SAME: %{{.*}}: !llvm.struct<(i32, ptr<i64>)>
// INDEX32-SAME: %{{.*}}: !llvm.struct<(i32, ptr<f32>)>

kgen.func @convert_meta_types(
    %arg0: !meta.scalar<f32>,
    %arg1: !meta.pointer<!meta.scalar<f32>>,
    %arg2: !meta.simd<4, f32>,
    %arg3: !meta.buffer<?, si64>,
    %arg4: !meta.buffer<?, f32>) {
  kgen.return
}

// -----

// CHECK-LABEL: llvm.func @"float_constant_f32,value=1.1283791670955126,type=f32"() -> f32
// CHECK: [[CST:%[0-9]+]] = llvm.mlir.constant(1.1283791670955126 : f64) : f64
// CHECK: [[TRUNC:%[0-9]+]] = llvm.fptrunc [[CST]] : f64 to f32
// CHECK: llvm.return [[TRUNC]] : f32
kgen.func @"float_constant_f32,value=1.1283791670955126,type=f32"() -> !meta.scalar<f32> {
  %0 = kgen.param.constant : f64 = <1.1283791670955126>
  %1 = llvm.fptrunc %0 : f64 to f32
  %2 = meta.cast_from_builtin %1 : f32 to !meta.scalar<f32>
  kgen.return  %2 : !meta.scalar<f32>
}

// -----

// CHECK-LABEL: llvm.func @"mul_f32,type=f32"(%arg0: f32, %arg1: f32) -> f32
// CHECK: [[OUT:%[0-9]+]] = llvm.fmul %arg0, %arg1
// CHECK: llvm.return [[OUT]] : f32
kgen.func @"mul_f32,type=f32"(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> !meta.scalar<f32> {
  %0 = meta.cast_to_builtin %arg0 : !meta.scalar<f32> to f32
  %1 = meta.cast_to_builtin %arg1 : !meta.scalar<f32> to f32
  %2 = llvm.fmul %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<f32>
  kgen.return %3 : !meta.scalar<f32>
}

// CHECK-LABEL: llvm.func @"void,type=f32"(%arg0: f32, %arg1: f32)
// CHECK: [[OUT:%[0-9]+]] = llvm.fmul %arg0, %arg1
// CHECK: llvm.return
kgen.func @"void,type=f32"(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) {
  %0 = meta.cast_to_builtin %arg0 : !meta.scalar<f32> to f32
  %1 = meta.cast_to_builtin %arg1 : !meta.scalar<f32> to f32
  %2 = llvm.fmul %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<f32>
  kgen.return
}

// -----

// CHECK-LABEL: llvm.func @"struct,type=f32"(%arg0: f32, %arg1: f32) -> !llvm.struct<(f32, f32)>
// CHECK: [[OUT:%[0-9]+]] = llvm.fmul %arg0, %arg1
// CHECK: [[UNDEF:%[0-9]+]] = llvm.mlir.undef : !llvm.struct<(f32, f32)>
// CHECK: [[ONE:%[0-9]+]] = llvm.insertvalue [[OUT]], [[UNDEF]][0] : !llvm.struct<(f32, f32)>
// CHECK: [[TWO:%[0-9]+]] = llvm.insertvalue [[OUT]], [[ONE]][1] : !llvm.struct<(f32, f32)>
// CHECK: llvm.return [[TWO]]
kgen.func @"struct,type=f32"(%arg0: !meta.scalar<f32>, %arg1: !meta.scalar<f32>) -> (!meta.scalar<f32>, !meta.scalar<f32>) {
  %0 = meta.cast_to_builtin %arg0 : !meta.scalar<f32> to f32
  %1 = meta.cast_to_builtin %arg1 : !meta.scalar<f32> to f32
  %2 = llvm.fmul %0, %1 : f32
  %3 = meta.cast_from_builtin %2 : f32 to !meta.scalar<f32>
  kgen.return %3, %3 : !meta.scalar<f32>, !meta.scalar<f32>
}

// -----

kgen.func @trivial(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  kgen.return %arg0 : !meta.scalar<f32>
}

kgen.func @no_result(%arg0: !meta.scalar<f32>) {
  kgen.return
}

kgen.func @two_results(%arg0: !meta.scalar<f32>) -> (!meta.scalar<f32>, !meta.scalar<f32>) {
  kgen.return %arg0, %arg0 : !meta.scalar<f32>, !meta.scalar<f32>
}

// CHECK-LABEL: llvm.func @convert_call
// CHECK-SAME: %[[ARG0:.*]]: f32
kgen.func @convert_call(%arg0: !meta.scalar<f32>) {
  // CHECK: llvm.call @trivial(%[[ARG0]]) : (f32) -> f32
  %0 = kgen.call @trivial(%arg0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
  // CHECK: llvm.call @no_result(%[[ARG0]]) : (f32) -> ()
  kgen.call @no_result(%arg0) : (!meta.scalar<f32>) -> ()
  // CHECK: %[[PACK:.*]] = llvm.call @two_results(%[[ARG0]]) : (f32) -> !llvm.struct<(f32, f32)>
  %1:2 = kgen.call @two_results(%arg0) : (!meta.scalar<f32>) -> (!meta.scalar<f32>, !meta.scalar<f32>)
  // CHECK: llvm.extractvalue %[[PACK]][0]
  // CHECK: llvm.extractvalue %[[PACK]][1]
  kgen.return
}
