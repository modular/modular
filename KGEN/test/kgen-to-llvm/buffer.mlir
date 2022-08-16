// RUN: kgen-opt -split-input-file -convert-kgen-to-llvm="index-bitwidth=64" %s | FileCheck %s
// RUN: kgen-opt -split-input-file -convert-kgen-to-llvm="index-bitwidth=32" %s | FileCheck %s --check-prefixes=INDEX32

// CHECK-LABEL: @cast_to_builtin
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @cast_to_builtin(%a: !meta.buffer<4, f32>) -> !llvm.ptr<f32> {
  // CHECK: return %[[A]]
  %0 = meta.cast_to_builtin %a : !meta.buffer<4, f32> to !llvm.ptr<f32>
  kgen.return %0 : !llvm.ptr<f32>
}

// -----

// CHECK-LABEL: @cast_from_builtin
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @cast_from_builtin(%a: !llvm.ptr<f64>) -> !meta.buffer<4, f64> {
  // CHECK: return %[[A]]
  %0 = meta.cast_from_builtin %a : !llvm.ptr<f64> to !meta.buffer<4, f64>
  kgen.return %0 : !meta.buffer<4, f64>
}

// -----

// CHECK-LABEL: @buffer_size
// INDEX32-LABEL: @buffer_size
kgen.kernel @buffer_size(%a: !meta.buffer<4, f32>) -> index {
  // CHECK: %[[S:.*]] = llvm.mlir.constant(4 : index) : i64
  // INDEX32: %[[S:.*]] = llvm.mlir.constant(4 : index) : i32
  %0 = meta.buffer.size %a : !meta.buffer<4, f32>
  // CHECK: return %[[S]]
  // INDEX32: return %[[S]]
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: @buffer_size
// INDEX32-LABEL: @buffer_size
kgen.kernel @buffer_size(%a: !meta.buffer<4, ?>) -> index {
  // CHECK: %[[S:.*]] = llvm.mlir.constant(4 : index)
  %0 = meta.buffer.size %a : !meta.buffer<4, ?>
  // CHECK: return %[[S]]
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: @buffer_size
// CHECK-SAME: %[[A:.*]]: !llvm.struct<(i64, ptr<f32>)>
// INDEX32-LABEL: @buffer_size
// INDEX32-SAME: %[[A:.*]]: !llvm.struct<(i32, ptr<f32>)>
kgen.kernel @buffer_size(%a: !meta.buffer<?, f32>) -> index {
  // CHECK: %[[S:.*]] = llvm.extractvalue %[[A]][0]
  // INDEX32: %[[S:.*]] = llvm.extractvalue %[[A]][0]
  %0 = meta.buffer.size %a : !meta.buffer<?, f32>
  // CHECK: return %[[S]] : i64
  // INDEX32: return %[[S]] : i32
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: @buffer_size
// CHECK-SAME: %[[A:.*]]: !llvm.struct<(i64, ptr<i8>, i8)>
// INDEX32-LABEL: @buffer_size
// INDEX32-SAME: %[[A:.*]]: !llvm.struct<(i32, ptr<i8>, i8)>
kgen.kernel @buffer_size(%a: !meta.buffer<?, ?>) -> index {
  // CHECK: %[[S:.*]] = llvm.extractvalue %[[A]][0]
  // INDEX32: %[[S:.*]] = llvm.extractvalue %[[A]][0]
  %0 = meta.buffer.size %a : !meta.buffer<?, ?>
  // CHECK: return %[[S]] : i64
  // INDEX32: return %[[S]] : i32
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: @buffer_dtype
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_dtype(%a: !meta.buffer<4, f32>) -> !kgen.dtype {
  // CHECK: llvm.mlir.constant({{[0-9]+}} : i8) : i8
  %0 = meta.buffer.dtype %a : !meta.buffer<4, f32>
  kgen.return %0 : !kgen.dtype
}

// -----

// CHECK-LABEL: @buffer_dtype
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_dtype(%a: !meta.buffer<?, f32>) -> !kgen.dtype {
  // CHECK: llvm.mlir.constant({{[0-9]+}} : i8) : i8
  %0 = meta.buffer.dtype %a : !meta.buffer<?, f32>
  kgen.return %0 : !kgen.dtype
}

// -----

// CHECK-LABEL: @buffer_dtype
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_dtype(%a: !meta.buffer<4, ?>) -> !kgen.dtype {
  // CHECK: %[[DTYPE:.*]] = llvm.extractvalue %[[A]][1]
  %0 = meta.buffer.dtype %a : !meta.buffer<4, ?>
  // CHECK: return %[[DTYPE]]
  kgen.return %0 : !kgen.dtype
}

// -----

// CHECK-LABEL: @buffer_dtype
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_dtype(%a: !meta.buffer<?, ?>) -> !kgen.dtype {
  // CHECK: %[[DTYPE:.*]] = llvm.extractvalue %[[A]][2]
  %0 = meta.buffer.dtype %a : !meta.buffer<?, ?>
  // CHECK: return %[[DTYPE]]
  kgen.return %0 : !kgen.dtype
}

// -----

// CHECK-LABEL: @buffer_address
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_address(%a: !meta.buffer<4, f32>) -> !meta.pointer<f32> {
  // CHECK: return %[[A]]
  %0 = meta.buffer.address %a : !meta.buffer<4, f32>
  kgen.return %0 : !meta.pointer<f32>
}

// -----

// CHECK-LABEL: @buffer_address
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_address(%a: !meta.buffer<?, f32>) -> !meta.pointer<f32> {
  // CHECK: %[[ADDR:.*]] = llvm.extractvalue %[[A]][1]
  %0 = meta.buffer.address %a : !meta.buffer<?, f32>
  // CHECK: return %[[ADDR]]
  kgen.return %0 : !meta.pointer<f32>
}

// -----

// CHECK-LABEL: @buffer_address
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_address(%a: !meta.buffer<4, ?>) -> !meta.pointer<?> {
  // CHECK: %[[ADDR:.*]] = llvm.extractvalue %[[A]][0]
  %0 = meta.buffer.address %a : !meta.buffer<4, ?>
  // CHECK: return %[[ADDR]] : !llvm.ptr<i8>
  kgen.return %0 : !meta.pointer<?>
}

// -----

// CHECK-LABEL: @buffer_address
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_address(%a: !meta.buffer<?, ?>) -> !meta.pointer<?> {
  // CHECK: %[[ADDR:.*]] = llvm.extractvalue %[[A]][1]
  %0 = meta.buffer.address %a : !meta.buffer<?, ?>
  // CHECK: return %[[ADDR]] : !llvm.ptr<i8>
  kgen.return %0 : !meta.pointer<?>
}

// -----

// CHECK-LABEL: @buffer_cast
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_cast(%a: !meta.buffer<?, ?>) -> !meta.buffer<4, f32> {
  // CHECK: %[[RAW:.*]] = llvm.extractvalue %[[A]][1]
  // CHECK: %[[PTR:.*]] = llvm.bitcast %[[RAW]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  %0 = meta.buffer.cast %a : !meta.buffer<?, ?> to !meta.buffer<4, f32>
  // CHECK: return %[[PTR]]
  kgen.return %0 : !meta.buffer<4, f32>
}

// -----

// CHECK-LABEL: @buffer_cast
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_cast(%a: !meta.buffer<4, ?>) -> !meta.buffer<4, f32> {
  // CHECK: %[[RAW:.*]] = llvm.extractvalue %[[A]][0]
  // CHECK: %[[PTR:.*]] = llvm.bitcast %[[RAW]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  %0 = meta.buffer.cast %a : !meta.buffer<4, ?> to !meta.buffer<4, f32>
  // CHECK: return %[[PTR]]
  kgen.return %0 : !meta.buffer<4, f32>
}

// -----

// CHECK-LABEL: @buffer_cast
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_cast(%a: !meta.buffer<?, f32>) -> !meta.buffer<4, f32> {
  // CHECK: %[[PTR:.*]] = llvm.extractvalue %[[A]][1]
  %0 = meta.buffer.cast %a : !meta.buffer<?, f32> to !meta.buffer<4, f32>
  // CHECK: return %[[PTR]]
  kgen.return %0 : !meta.buffer<4, f32>
}

// -----

// CHECK-LABEL: @buffer_cast
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_cast(%a: !meta.buffer<?, ?>) -> !meta.buffer<?, f32> {
  // CHECK-DAG: %[[RAW:.*]] = llvm.extractvalue %[[A]][1]
  // CHECK-DAG: %[[PTR:.*]] = llvm.bitcast %[[RAW]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.undef
  // CHECK-DAG: %[[SIZE:.*]] = llvm.extractvalue %[[A]][0]
  // CHECK-DAG: %[[S1:.*]] = llvm.insertvalue %[[SIZE]], %[[S0]][0]
  // CHECK-DAG: %[[S2:.*]] = llvm.insertvalue %[[PTR]], %[[S1]][1]
  // CHECK: return %[[S2]]
  %0 = meta.buffer.cast %a : !meta.buffer<?, ?> to !meta.buffer<?, f32>
  kgen.return %0 : !meta.buffer<?, f32>
}

// -----

// CHECK-LABEL: @buffer_cast
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_cast(%a: !meta.buffer<4, ?>) -> !meta.buffer<?, f32> {
  // CHECK-DAG: %[[RAW:.*]] = llvm.extractvalue %[[A]][0]
  // CHECK-DAG: %[[PTR:.*]] = llvm.bitcast %[[RAW]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.undef
  // CHECK-DAG: %[[SIZE:.*]] = llvm.mlir.constant(4 : index)
  // CHECK-DAG: %[[S1:.*]] = llvm.insertvalue %[[SIZE]], %[[S0]][0]
  // CHECK-DAG: %[[S2:.*]] = llvm.insertvalue %[[PTR]], %[[S1]][1]
  // CHECK: return %[[S2]]
  %0 = meta.buffer.cast %a : !meta.buffer<4, ?> to !meta.buffer<?, f32>
  kgen.return %0 : !meta.buffer<?, f32>
}

// -----

// CHECK-LABEL: @buffer_cast
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_cast(%a: !meta.buffer<4, f32>) -> !meta.buffer<?, f32> {
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.undef
  // CHECK-DAG: %[[SIZE:.*]] = llvm.mlir.constant(4 : index)
  // CHECK-DAG: %[[S1:.*]] = llvm.insertvalue %[[SIZE]], %[[S0]][0]
  // CHECK-DAG: %[[S2:.*]] = llvm.insertvalue %[[A]], %[[S1]][1]
  // CHECK: return %[[S2]]
  %0 = meta.buffer.cast %a : !meta.buffer<4, f32> to !meta.buffer<?, f32>
  kgen.return %0 : !meta.buffer<?, f32>
}

// -----

// CHECK-LABEL: @buffer_cast
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_cast(%a: !meta.buffer<?, ?>) -> !meta.buffer<4, ?> {
  // CHECK-DAG: %[[PTR:.*]] = llvm.extractvalue %[[A]][1]
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.undef
  // CHECK-DAG: %[[DTYPE:.*]] = llvm.extractvalue %[[A]][2]
  // CHECK-DAG: %[[S1:.*]] = llvm.insertvalue %[[DTYPE]], %[[S0]][1]
  // CHECK-DAG: %[[S2:.*]] = llvm.insertvalue %[[PTR]], %[[S1]][0]
  // CHECK: return %[[S2]]
  %0 = meta.buffer.cast %a : !meta.buffer<?, ?> to !meta.buffer<4, ?>
  kgen.return %0 : !meta.buffer<4, ?>
}

// -----

// CHECK-LABEL: @buffer_cast
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_cast(%a: !meta.buffer<?, f32>) -> !meta.buffer<4, ?> {
  // CHECK-DAG: %[[RAW:.*]] = llvm.extractvalue %[[A]][1]
  // CHECK-DAG: %[[PTR:.*]] = llvm.bitcast %[[RAW]] : !llvm.ptr<f32> to !llvm.ptr<i8>
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.undef
  // CHECK-DAG: %[[DTYPE:.*]] = llvm.mlir.constant
  // CHECK-DAG: %[[S1:.*]] = llvm.insertvalue %[[DTYPE]], %[[S0]][1]
  // CHECK-DAG: %[[S2:.*]] = llvm.insertvalue %[[PTR]], %[[S1]][0]
  // CHECK: return %[[S2]]
  %0 = meta.buffer.cast %a : !meta.buffer<?, f32> to !meta.buffer<4, ?>
  kgen.return %0 : !meta.buffer<4, ?>
}

// -----

// CHECK-LABEL: @buffer_cast
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_cast(%a: !meta.buffer<4, f32>) -> !meta.buffer<4, ?> {
  // CHECK-DAG: %[[PTR:.*]] = llvm.bitcast %[[A]] : !llvm.ptr<f32> to !llvm.ptr<i8>
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.undef
  // CHECK-DAG: %[[DTYPE:.*]] = llvm.mlir.constant
  // CHECK-DAG: %[[S1:.*]] = llvm.insertvalue %[[DTYPE]], %[[S0]][1]
  // CHECK-DAG: %[[S2:.*]] = llvm.insertvalue %[[PTR]], %[[S1]][0]
  // CHECK: return %[[S2]]
  %0 = meta.buffer.cast %a : !meta.buffer<4, f32> to !meta.buffer<4, ?>
  kgen.return %0 : !meta.buffer<4, ?>
}

// -----

// CHECK-LABEL: @buffer_cast
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_cast(%a: !meta.buffer<4, ?>) -> !meta.buffer<?, ?> {
  // CHECK-DAG: %[[PTR:.*]] = llvm.extractvalue %[[A]][0]
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.undef
  // CHECK-DAG: %[[SIZE:.*]] = llvm.mlir.constant
  // CHECK-DAG: %[[DTYPE:.*]] = llvm.extractvalue %[[A]][1]
  // CHECK-DAG: %[[S1:.*]] = llvm.insertvalue %[[SIZE]], %[[S0]][0]
  // CHECK-DAG: %[[S2:.*]] = llvm.insertvalue %[[DTYPE]], %[[S1]][2]
  // CHECK-DAG: %[[S3:.*]] = llvm.insertvalue %[[PTR]], %[[S2]][1]
  // CHECK: return %[[S3]]
  %0 = meta.buffer.cast %a : !meta.buffer<4, ?> to !meta.buffer<?, ?>
  kgen.return %0 : !meta.buffer<?, ?>
}

// -----

// CHECK-LABEL: @buffer_cast
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_cast(%a: !meta.buffer<?, f32>) -> !meta.buffer<?, ?> {
  // CHECK-DAG: %[[RAW:.*]] = llvm.extractvalue %[[A]][1]
  // CHECK-DAG: %[[PTR:.*]] = llvm.bitcast %[[RAW]] : !llvm.ptr<f32> to !llvm.ptr<i8>
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.undef
  // CHECK-DAG: %[[SIZE:.*]] = llvm.extractvalue %[[A]][0]
  // CHECK-DAG: %[[DTYPE:.*]] = llvm.mlir.constant
  // CHECK-DAG: %[[S1:.*]] = llvm.insertvalue %[[SIZE]], %[[S0]][0]
  // CHECK-DAG: %[[S2:.*]] = llvm.insertvalue %[[DTYPE]], %[[S1]][2]
  // CHECK-DAG: %[[S3:.*]] = llvm.insertvalue %[[PTR]], %[[S2]][1]
  // CHECK: return %[[S3]]
  %0 = meta.buffer.cast %a : !meta.buffer<?, f32> to !meta.buffer<?, ?>
  kgen.return %0 : !meta.buffer<?, ?>
}

// -----

// CHECK-LABEL: @buffer_cast
// CHECK-SAME: %[[A:.*]]:
kgen.kernel @buffer_cast(%a: !meta.buffer<4, f32>) -> !meta.buffer<?, ?> {
  // CHECK-DAG: %[[PTR:.*]] = llvm.bitcast %[[A]] : !llvm.ptr<f32> to !llvm.ptr<i8>
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.undef
  // CHECK-DAG: %[[SIZE:.*]] = llvm.mlir.constant(4 : index)
  // CHECK-DAG: %[[DTYPE:.*]] = llvm.mlir.constant({{.*}}) : i8
  // CHECK-DAG: %[[S1:.*]] = llvm.insertvalue %[[SIZE]], %[[S0]][0]
  // CHECK-DAG: %[[S2:.*]] = llvm.insertvalue %[[DTYPE]], %[[S1]][2]
  // CHECK-DAG: %[[S3:.*]] = llvm.insertvalue %[[PTR]], %[[S2]][1]
  // CHECK: return %[[S3]]
  %0 = meta.buffer.cast %a : !meta.buffer<4, f32> to !meta.buffer<?, ?>
  kgen.return %0 : !meta.buffer<?, ?>
}
