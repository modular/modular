// RUN: kgen-opt -split-input-file -convert-kgen-to-llvm %s | FileCheck %s
// RUN: kgen-opt -split-input-file -convert-kgen-to-llvm %s | kgen-opt -canonicalize -split-input-file | FileCheck %s -check-prefixes=CANON

// CHECK-LABEL: llvm.func @trivial_kernel(%arg0: i32)
// CHECK-NEXT: llvm.return %arg0 : i32
kgen.kernel @trivial_kernel(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// -----

// CHECK-LABEL: llvm.func @convert_meta_types
// CHECK-SAME: %{{.*}}: f32
// CHECK-SAME: %{{.*}}: !llvm.ptr<f32>
// CHECK-SAME: %{{.*}}: vector<4xf32>
// CHECK-SAME: %{{.*}}: !llvm.struct<(i64, ptr<i64>)>
// CHECK-SAME: %{{.*}}: !llvm.struct<(i64, ptr<f32>)>
kgen.kernel @convert_meta_types(
    %arg0: !meta.scalar<f32>,
    %arg1: !meta.pointer<f32>,
    %arg2: !meta.simd<4, f32>,
    %arg3: !meta.buffer<4, si64>,
    %arg4: !meta.buffer<?, f32>) {
  kgen.return
}

// -----

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

// -----

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

// -----

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

// -----

kgen.kernel @trivial_kernel(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  kgen.return %arg0 : !meta.scalar<f32>
}

kgen.kernel @no_result(%arg0: !meta.scalar<f32>) {
  kgen.return
}

kgen.kernel @two_results(%arg0: !meta.scalar<f32>) -> (!meta.scalar<f32>, !meta.scalar<f32>) {
  kgen.return %arg0, %arg0 : !meta.scalar<f32>, !meta.scalar<f32>
}

// CHECK-LABEL: llvm.func @convert_call
// CHECK-SAME: %[[ARG0:.*]]: f32
kgen.kernel @convert_call(%arg0: !meta.scalar<f32>) {
  // CHECK: llvm.call @trivial_kernel(%[[ARG0]]) : (f32) -> f32
  %0 = kgen.call @trivial_kernel(%arg0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
  // CHECK: llvm.call @no_result(%[[ARG0]]) : (f32) -> ()
  kgen.call @no_result(%arg0) : (!meta.scalar<f32>) -> ()
  // CHECK: %[[PACK:.*]] = llvm.call @two_results(%[[ARG0]]) : (f32) -> !llvm.struct<(f32, f32)>
  %1:2 = kgen.call @two_results(%arg0) : (!meta.scalar<f32>) -> (!meta.scalar<f32>, !meta.scalar<f32>)
  // CHECK: llvm.extractvalue %[[PACK]][0]
  // CHECK: llvm.extractvalue %[[PACK]][1]
  kgen.return
}

// -----

// CHECK-LABEL: llvm.func @buffer_size
// CHECK-SAME: %{{.*}}: !llvm.struct<(i64, ptr<f32>)>, %[[ARG1:.*]]: !llvm.struct<(i64, ptr<f32>)>
kgen.kernel @buffer_size(%arg0: !meta.buffer<4, f32>, %arg1: !meta.buffer<?, f32>) {
  // CHECK: llvm.mlir.constant(4 : index) : i64
  %0 = meta.buffer.size %arg0 : !meta.buffer<4, f32>
  // CHECK: llvm.extractvalue %[[ARG1]][0]
  %1 = meta.buffer.size %arg1 : !meta.buffer<?, f32>
  kgen.return
}

// -----

// CHECK-LABEL: llvm.func @buffer_address
// CHECK-SAME: %[[ARG0:.*]]:
kgen.kernel @buffer_address(%arg0: !meta.buffer<?, f32>) {
  // CHECK: llvm.extractvalue %[[ARG0]][1]
  %0 = meta.buffer.address %arg0 : !meta.buffer<?, f32>
  kgen.return
}

// -----

// CHECK-LABEL: llvm.func @buffer_cast
// CHECK-SAME: %[[ARG0:.*]]:
kgen.kernel @buffer_cast(%arg0: !meta.buffer<?, f32>) -> !meta.buffer<4, f32> {
  // CHECK-NOT: meta.buffer.cast
  %0 = meta.buffer.cast %arg0 : !meta.buffer<?, f32> to !meta.buffer<4, f32>
  // CHECK: llvm.return %[[ARG0]]
  kgen.return %0 : !meta.buffer<4, f32>
}

// -----

!buffer = !llvm.struct<(i64, ptr<f32>)>

llvm.func @impl(%arg0: !buffer) -> f32 {
  %0 = llvm.mlir.constant(1.0 : f32) : f32
  llvm.return %0 : f32
}

// FIXME: This needs to run through canonicalization to remove
// builtin.unrealized_conversion_cast ops.

// CANON-LABEL: llvm.func @buffer_kernel
// CANON-SAME: %[[BUF:.*]]: !llvm.struct<(i64, ptr<f32>)>
kgen.kernel @buffer_kernel(%arg0: !meta.buffer<?, f32>) -> f32 {
  // CANON: %[[PTR:.*]] = llvm.extractvalue %[[BUF]][1]
  %0 = meta.buffer.cast %arg0 : !meta.buffer<?, f32> to !meta.buffer<4, f32>
  %1 = meta.buffer.size %0 : !meta.buffer<4, f32>
  %2 = builtin.unrealized_conversion_cast %1 : index to i64
  %3 = meta.buffer.address %0 : !meta.buffer<4, f32>
  %4 = builtin.unrealized_conversion_cast %3 : !meta.pointer<f32> to !llvm.ptr<f32>
  %5 = builtin.unrealized_conversion_cast %0 : !meta.buffer<4, f32> to !buffer
  // CANON: llvm.load %[[PTR]]
  %6 = llvm.load %4 : !llvm.ptr<f32>
  // CANON: llvm.call @impl(%[[BUF]])
  %7 = llvm.call @impl(%5) : (!buffer) -> f32
  %8 = llvm.fadd %6, %7 : f32
  kgen.return %8 : f32
}
