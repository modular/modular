// RUN: kgen-opt %s -pass-pipeline='builtin.module(lower-kgen-to-llvm,llvm.func(lower-pop-to-llvm,canonicalize))' | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @empty_union
// CHECK-SAME: () -> !llvm.struct<()>
kgen.func @empty_union() -> !pop.union<> {
  kgen.unreachable
}

// CHECK-LABEL: @union_create_0
kgen.func @union_create_0(%arg0: i32) -> !pop.union<i32> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<4 x i8> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 4 : i64} : i32, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 4 : i64} : !llvm.ptr -> !llvm.array<4 x i8>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.wrap %arg0 : i32 as <i32>
  kgen.return %0 : !pop.union<i32>
}

// CHECK-LABEL: @union_create_1
// Union alignment is now max of variant alignments (i8 = 1 byte).
kgen.func @union_create_1(%arg0: i8) -> !pop.union<i8> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<1 x i8> {alignment = 1 : i64} : (i64) -> !llvm.ptr
// CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
// CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 1 : i64} : i8, !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 1 : i64} : !llvm.ptr -> !llvm.array<1 x i8>
// CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.wrap %arg0 : i8 as <i8>
  kgen.return %0 : !pop.union<i8>
}

// CHECK-LABEL: @union_create_2
// Union alignment is now max of variant alignments (f64 = 8 bytes).
kgen.func @union_create_2(%arg0: f64) -> !pop.union<f64> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<8 x i8> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 8 : i64} : f64, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 8 : i64} : !llvm.ptr -> !llvm.array<8 x i8>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.array<8 x i8>
  %0 = pop.union.wrap %arg0 : f64 as <f64>
  kgen.return %0 : !pop.union<f64>
}

// CHECK-LABEL: @union_create_3
kgen.func @union_create_3(%arg0: !kgen.struct<(i32, i32)>) -> !pop.union<struct<(i32, i32)>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<8 x i8> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 4 : i64} : !llvm.struct<(i32, i32)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 4 : i64} : !llvm.ptr -> !llvm.array<8 x i8>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.array<8 x i8>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i32, i32)> as <struct<(i32, i32)>>
  kgen.return %0 : !pop.union<struct<(i32, i32)>>
}

// CHECK-LABEL: @union_create_4
kgen.func @union_create_4(%arg0: !kgen.struct<(i32, i64, i32)>) -> !pop.union<struct<(i32, i64, i32)>, array<4, i64>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<32 x i8> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 4 : i64} : !llvm.struct<(i32, i64, i32)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 4 : i64} : !llvm.ptr -> !llvm.array<32 x i8>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.array<32 x i8>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i32, i64, i32)> as <struct<(i32, i64, i32)>, array<4, i64>>
  kgen.return %0 : !pop.union<struct<(i32, i64, i32)>, array<4, i64>>
}

// CHECK-LABEL: @union_create_5
// Union alignment is now max of variant alignments (simd<2, f32> = 8 bytes).
// Note: struct type changed due to alignment-based padding.
kgen.func @union_create_5(%arg0: !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>) -> !pop.union<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<24 x i8> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 8 : i64} : !llvm.struct<(array<2 x i16>, array<4 x i8>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 8 : i64} : !llvm.ptr -> !llvm.array<24 x i8>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.array<24 x i8>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)> as <struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>
  kgen.return %0 : !pop.union<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>
}

// CHECK-LABEL: @union_create_6
// Union alignment is now max of variant alignments (pointer = 8 bytes).
kgen.func @union_create_6(%arg0: !kgen.pointer<index>) -> !pop.union<pointer<index>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<8 x i8> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 8 : i64} : !llvm.ptr -> !llvm.array<8 x i8>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.array<8 x i8>
  %0 = pop.union.wrap %arg0 : !kgen.pointer<index> as <pointer<index>>
  kgen.return %0 : !pop.union<pointer<index>>
}

// CHECK-LABEL: @union_get_0
kgen.func @union_get_0(%arg0: !pop.union<i32>) ->  i32{
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<4 x i8> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 4 : i64} : !llvm.array<4 x i8>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <i32> as i32
  kgen.return %0 : i32
}

// CHECK-LABEL: @union_get_1
// Union alignment is now max of variant alignments (f64 = 8 bytes).
kgen.func @union_get_1(%arg0: !pop.union<f64>) -> f64 {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<8 x i8> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 8 : i64} : !llvm.array<8 x i8>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 8 : i64} : !llvm.ptr -> f64
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : f64
  %0 = pop.union.unwrap %arg0 : <f64> as f64
  kgen.return %0 : f64
}

// CHECK-LABEL: @union_get_2
kgen.func @union_get_2(%arg0: !pop.union<struct<(i32, i32)>>) -> !kgen.struct<(i32, i32)>{
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<8 x i8> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 4 : i64} : !llvm.array<8 x i8>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 4 : i64} : !llvm.ptr -> !llvm.struct<(i32, i32)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(i32, i32)>
  %0 = pop.union.unwrap %arg0 : <struct<(i32, i32)>> as !kgen.struct<(i32, i32)>
  kgen.return %0 : !kgen.struct<(i32, i32)>
}

// CHECK-LABEL: @union_get_3
kgen.func @union_get_3(%arg0: !pop.union<struct<(i32, i64, i32)>, array<4, i64>>) -> !kgen.struct<(i32, i64, i32)> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<32 x i8> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 4 : i64} : !llvm.array<32 x i8>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 4 : i64} : !llvm.ptr -> !llvm.struct<(i32, i64, i32)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <struct<(i32, i64, i32)>, array<4, i64>> as !kgen.struct<(i32, i64, i32)>
  kgen.return %0 : !kgen.struct<(i32, i64, i32)>
}

// CHECK-LABEL: @union_get_4
// Union alignment is now max of variant alignments (simd<2, f32> = 8 bytes).
kgen.func @union_get_4(%arg0: !pop.union<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>) -> !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<24 x i8> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 8 : i64} : !llvm.array<24 x i8>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 8 : i64} : !llvm.ptr -> !llvm.struct<(array<2 x i16>, array<4 x i8>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>> as !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>
  kgen.return %0 : !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>
}

// CHECK-LABEL: @union_get_5
// Union alignment is now max of variant alignments (pointer = 8 bytes).
kgen.func @union_get_5(%arg0: !pop.union<pointer<index>>) -> !kgen.pointer<index> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<8 x i8> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 8 : i64} : !llvm.array<8 x i8>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <pointer<index>> as !kgen.pointer<index>
  kgen.return %0 : !kgen.pointer<index>
}

// CHECK-LABEL: @unpack_pointer
// Union alignment is now max of variant alignments (pointer = 8 bytes).
kgen.func @unpack_pointer(%arg0: !pop.union<pointer<i8>>) -> !kgen.pointer<i8> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<8 x i8> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 8 : i64} : !llvm.array<8 x i8>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <pointer<i8>> as !kgen.pointer<i8>
  kgen.return %0 : !kgen.pointer<i8>
}

// CHECK-LABEL: @union_constant_0
kgen.func @union_constant_0() -> !pop.union<i32> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<4 x i8>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG:  %[[VAL_4:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG:  %[[VAL_5:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-DAG:  %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:      %[[VAL_7:.*]] = llvm.lshr %[[VAL_4]], %[[VAL_6]] : i32
  // CHECK:      %[[VAL_8:.*]] = llvm.trunc %[[VAL_7]] : i32 to i8
  // CHECK:      %[[VAL_9:.*]] = llvm.shl %[[VAL_8]], %[[VAL_5]] : i8
  // CHECK:      %[[VAL_10:.*]] = llvm.or %[[VAL_5]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_11:.*]] = llvm.lshr %[[VAL_4]], %[[VAL_3]] : i32
  // CHECK:      %[[VAL_12:.*]] = llvm.trunc %[[VAL_11]] : i32 to i8
  // CHECK:      %[[VAL_13:.*]] = llvm.shl %[[VAL_12]], %[[VAL_5]] : i8
  // CHECK:      %[[VAL_14:.*]] = llvm.or %[[VAL_5]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_15:.*]] = llvm.lshr %[[VAL_4]], %[[VAL_2]] : i32
  // CHECK:      %[[VAL_16:.*]] = llvm.trunc %[[VAL_15]] : i32 to i8
  // CHECK:      %[[VAL_17:.*]] = llvm.shl %[[VAL_16]], %[[VAL_5]] : i8
  // CHECK:      %[[VAL_18:.*]] = llvm.or %[[VAL_5]], %[[VAL_17]] : i8
  // CHECK:      %[[VAL_19:.*]] = llvm.lshr %[[VAL_4]], %[[VAL_1]] : i32
  // CHECK:      %[[VAL_20:.*]] = llvm.trunc %[[VAL_19]] : i32 to i8
  // CHECK:      %[[VAL_21:.*]] = llvm.shl %[[VAL_20]], %[[VAL_5]] : i8
  // CHECK:      %[[VAL_22:.*]] = llvm.or %[[VAL_5]], %[[VAL_21]] : i8
  // CHECK:      %[[VAL_23:.*]] = llvm.insertvalue %[[VAL_10]], %[[VAL_0]][0] : !llvm.array<4 x i8>
  // CHECK:      %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_23]][1] : !llvm.array<4 x i8>
  // CHECK:      %[[VAL_25:.*]] = llvm.insertvalue %[[VAL_18]], %[[VAL_24]][2] : !llvm.array<4 x i8>
  // CHECK:      %[[VAL_26:.*]] = llvm.insertvalue %[[VAL_22]], %[[VAL_25]][3] : !llvm.array<4 x i8>
  %0 = kgen.param.constant: union<i32> = <{:i32 1}>
  kgen.return %0 : !pop.union<i32>
}

// CHECK-LABEL: @union_constant_1
kgen.func @union_constant_1() -> !pop.union<struct<(i32, i64, i32)>, struct<(f64, f32)>> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<16 x i8>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(56 : i64) : i64
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(48 : i64) : i64
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(40 : i64) : i64
  // CHECK-DAG:  %[[VAL_4:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK-DAG:  %[[VAL_5:.*]] = llvm.mlir.constant(24 : i64) : i64
  // CHECK-DAG:  %[[VAL_6:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-DAG:  %[[VAL_7:.*]] = llvm.mlir.constant(8 : i64) : i64
  // CHECK-DAG:  %[[VAL_8:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG:  %[[VAL_9:.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK-DAG:  %[[VAL_10:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  %[[VAL_11:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG:  %[[VAL_12:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:  %[[VAL_13:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-DAG:  %[[VAL_14:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-DAG:  %[[VAL_15:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK-DAG:  %[[VAL_16:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:      %[[VAL_17:.*]] = llvm.lshr %[[VAL_16]], %[[VAL_12]] : i32
  // CHECK:      %[[VAL_18:.*]] = llvm.trunc %[[VAL_17]] : i32 to i8
  // CHECK:      %[[VAL_19:.*]] = llvm.shl %[[VAL_18]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_20:.*]] = llvm.or %[[VAL_13]], %[[VAL_19]] : i8
  // CHECK:      %[[VAL_21:.*]] = llvm.lshr %[[VAL_16]], %[[VAL_11]] : i32
  // CHECK:      %[[VAL_22:.*]] = llvm.trunc %[[VAL_21]] : i32 to i8
  // CHECK:      %[[VAL_23:.*]] = llvm.shl %[[VAL_22]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_24:.*]] = llvm.or %[[VAL_13]], %[[VAL_23]] : i8
  // CHECK:      %[[VAL_25:.*]] = llvm.lshr %[[VAL_16]], %[[VAL_10]] : i32
  // CHECK:      %[[VAL_26:.*]] = llvm.trunc %[[VAL_25]] : i32 to i8
  // CHECK:      %[[VAL_27:.*]] = llvm.shl %[[VAL_26]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_28:.*]] = llvm.or %[[VAL_13]], %[[VAL_27]] : i8
  // CHECK:      %[[VAL_29:.*]] = llvm.lshr %[[VAL_16]], %[[VAL_9]] : i32
  // CHECK:      %[[VAL_30:.*]] = llvm.trunc %[[VAL_29]] : i32 to i8
  // CHECK:      %[[VAL_31:.*]] = llvm.shl %[[VAL_30]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_32:.*]] = llvm.or %[[VAL_13]], %[[VAL_31]] : i8
  // CHECK:      %[[VAL_33:.*]] = llvm.lshr %[[VAL_15]], %[[VAL_8]] : i64
  // CHECK:      %[[VAL_34:.*]] = llvm.trunc %[[VAL_33]] : i64 to i8
  // CHECK:      %[[VAL_35:.*]] = llvm.shl %[[VAL_34]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_36:.*]] = llvm.or %[[VAL_13]], %[[VAL_35]] : i8
  // CHECK:      %[[VAL_37:.*]] = llvm.lshr %[[VAL_15]], %[[VAL_7]] : i64
  // CHECK:      %[[VAL_38:.*]] = llvm.trunc %[[VAL_37]] : i64 to i8
  // CHECK:      %[[VAL_39:.*]] = llvm.shl %[[VAL_38]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_40:.*]] = llvm.or %[[VAL_13]], %[[VAL_39]] : i8
  // CHECK:      %[[VAL_41:.*]] = llvm.lshr %[[VAL_15]], %[[VAL_6]] : i64
  // CHECK:      %[[VAL_42:.*]] = llvm.trunc %[[VAL_41]] : i64 to i8
  // CHECK:      %[[VAL_43:.*]] = llvm.shl %[[VAL_42]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_44:.*]] = llvm.or %[[VAL_13]], %[[VAL_43]] : i8
  // CHECK:      %[[VAL_45:.*]] = llvm.lshr %[[VAL_15]], %[[VAL_5]] : i64
  // CHECK:      %[[VAL_46:.*]] = llvm.trunc %[[VAL_45]] : i64 to i8
  // CHECK:      %[[VAL_47:.*]] = llvm.shl %[[VAL_46]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_48:.*]] = llvm.or %[[VAL_13]], %[[VAL_47]] : i8
  // CHECK:      %[[VAL_49:.*]] = llvm.lshr %[[VAL_15]], %[[VAL_4]] : i64
  // CHECK:      %[[VAL_50:.*]] = llvm.trunc %[[VAL_49]] : i64 to i8
  // CHECK:      %[[VAL_51:.*]] = llvm.shl %[[VAL_50]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_52:.*]] = llvm.or %[[VAL_13]], %[[VAL_51]] : i8
  // CHECK:      %[[VAL_53:.*]] = llvm.lshr %[[VAL_15]], %[[VAL_3]] : i64
  // CHECK:      %[[VAL_54:.*]] = llvm.trunc %[[VAL_53]] : i64 to i8
  // CHECK:      %[[VAL_55:.*]] = llvm.shl %[[VAL_54]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_56:.*]] = llvm.or %[[VAL_13]], %[[VAL_55]] : i8
  // CHECK:      %[[VAL_57:.*]] = llvm.lshr %[[VAL_15]], %[[VAL_2]] : i64
  // CHECK:      %[[VAL_58:.*]] = llvm.trunc %[[VAL_57]] : i64 to i8
  // CHECK:      %[[VAL_59:.*]] = llvm.shl %[[VAL_58]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_60:.*]] = llvm.or %[[VAL_13]], %[[VAL_59]] : i8
  // CHECK:      %[[VAL_61:.*]] = llvm.lshr %[[VAL_15]], %[[VAL_1]] : i64
  // CHECK:      %[[VAL_62:.*]] = llvm.trunc %[[VAL_61]] : i64 to i8
  // CHECK:      %[[VAL_63:.*]] = llvm.shl %[[VAL_62]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_64:.*]] = llvm.or %[[VAL_13]], %[[VAL_63]] : i8
  // CHECK:      %[[VAL_65:.*]] = llvm.lshr %[[VAL_14]], %[[VAL_12]] : i32
  // CHECK:      %[[VAL_66:.*]] = llvm.trunc %[[VAL_65]] : i32 to i8
  // CHECK:      %[[VAL_67:.*]] = llvm.shl %[[VAL_66]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_68:.*]] = llvm.or %[[VAL_13]], %[[VAL_67]] : i8
  // CHECK:      %[[VAL_69:.*]] = llvm.lshr %[[VAL_14]], %[[VAL_11]] : i32
  // CHECK:      %[[VAL_70:.*]] = llvm.trunc %[[VAL_69]] : i32 to i8
  // CHECK:      %[[VAL_71:.*]] = llvm.shl %[[VAL_70]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_72:.*]] = llvm.or %[[VAL_13]], %[[VAL_71]] : i8
  // CHECK:      %[[VAL_73:.*]] = llvm.lshr %[[VAL_14]], %[[VAL_10]] : i32
  // CHECK:      %[[VAL_74:.*]] = llvm.trunc %[[VAL_73]] : i32 to i8
  // CHECK:      %[[VAL_75:.*]] = llvm.shl %[[VAL_74]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_76:.*]] = llvm.or %[[VAL_13]], %[[VAL_75]] : i8
  // CHECK:      %[[VAL_77:.*]] = llvm.lshr %[[VAL_14]], %[[VAL_9]] : i32
  // CHECK:      %[[VAL_78:.*]] = llvm.trunc %[[VAL_77]] : i32 to i8
  // CHECK:      %[[VAL_79:.*]] = llvm.shl %[[VAL_78]], %[[VAL_13]] : i8
  // CHECK:      %[[VAL_80:.*]] = llvm.or %[[VAL_13]], %[[VAL_79]] : i8
  // CHECK:      %[[VAL_81:.*]] = llvm.insertvalue %[[VAL_20]], %[[VAL_0]][0] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_82:.*]] = llvm.insertvalue %[[VAL_24]], %[[VAL_81]][1] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_83:.*]] = llvm.insertvalue %[[VAL_28]], %[[VAL_82]][2] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_84:.*]] = llvm.insertvalue %[[VAL_32]], %[[VAL_83]][3] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_85:.*]] = llvm.insertvalue %[[VAL_36]], %[[VAL_84]][4] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_86:.*]] = llvm.insertvalue %[[VAL_40]], %[[VAL_85]][5] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_87:.*]] = llvm.insertvalue %[[VAL_44]], %[[VAL_86]][6] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_88:.*]] = llvm.insertvalue %[[VAL_48]], %[[VAL_87]][7] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_89:.*]] = llvm.insertvalue %[[VAL_52]], %[[VAL_88]][8] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_90:.*]] = llvm.insertvalue %[[VAL_56]], %[[VAL_89]][9] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_91:.*]] = llvm.insertvalue %[[VAL_60]], %[[VAL_90]][10] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_92:.*]] = llvm.insertvalue %[[VAL_64]], %[[VAL_91]][11] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_93:.*]] = llvm.insertvalue %[[VAL_68]], %[[VAL_92]][12] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_94:.*]] = llvm.insertvalue %[[VAL_72]], %[[VAL_93]][13] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_95:.*]] = llvm.insertvalue %[[VAL_76]], %[[VAL_94]][14] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_96:.*]] = llvm.insertvalue %[[VAL_80]], %[[VAL_95]][15] : !llvm.array<16 x i8>
  %0 = kgen.param.constant: union<struct<(i32, i64, i32)>, struct<(f64, f32)>> = <{:struct<(i32, i64, i32)> { 1, 2, 3 }}>
  kgen.return %0 : !pop.union<struct<(i32, i64, i32)>, struct<(f64, f32)>>
}

// CHECK-LABEL: @union_constant_2
kgen.func @union_constant_2() -> !pop.union<i1, i2, i3, i4, i5, i6> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<1 x i8>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(1 : i4) : i4
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(0 : i4) : i4
  // CHECK:      %[[VAL_4:.*]] = llvm.lshr %[[VAL_1]], %[[VAL_3]] : i4
  // CHECK:      %[[VAL_5:.*]] = llvm.zext %[[VAL_4]] : i4 to i8
  // CHECK:      %[[VAL_6:.*]] = llvm.shl %[[VAL_5]], %[[VAL_2]] : i8
  // CHECK:      %[[VAL_7:.*]] = llvm.or %[[VAL_2]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_0]][0] : !llvm.array<1 x i8>
  %0 = kgen.param.constant: union<i1, i2, i3, i4, i5, i6> = <{:i4 1}>
  kgen.return %0 : !pop.union<i1, i2, i3, i4, i5, i6>
}

// CHECK-LABEL: @union_wrap_nonempty_with_empty_sibling
// Union contains a non-empty struct variant followed by an empty struct variant.
// The non-empty variant comes first so its {align=1, type=i8} was previously
// overwritten by the empty struct's {align=1, type=null} via the '>=' path,
// causing a null dereference in getTypeSizeInBits (MOCO-3275).
// The union lowers to !llvm.array<1 x i8> (maxSize=1).
kgen.func @union_wrap_nonempty_with_empty_sibling(%arg0: !kgen.struct<(i8)>) -> !pop.union<struct<(i8)>, struct<()>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<1 x i8> {alignment = 1 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 1 : i64} : !llvm.struct<(i8)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 1 : i64} : !llvm.ptr -> !llvm.array<1 x i8>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.array<1 x i8>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i8)> as <struct<(i8)>, struct<()>>
  kgen.return %0 : !pop.union<struct<(i8)>, struct<()>>
}

// CHECK-LABEL: @union_wrap_empty_with_nonempty_sibling
// CHECK-SAME:  () -> !llvm.array<1 x i8>
// Wrap the empty struct variant into the same union type. lower-kgen-to-llvm
// eliminates the zero-size struct argument and replaces its use with undef.
kgen.func @union_wrap_empty_with_nonempty_sibling(%arg0: !kgen.struct<()>) -> !pop.union<struct<(i8)>, struct<()>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.mlir.undef : !llvm.struct<()>
  // CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<1 x i8> {alignment = 1 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_2]] : !llvm.ptr
  // CHECK:           llvm.store %[[VAL_1]], %[[VAL_2]] {alignment = 1 : i64} : !llvm.struct<()>, !llvm.ptr
  // CHECK:           %[[VAL_3:.*]] = llvm.load %[[VAL_2]] {alignment = 1 : i64} : !llvm.ptr -> !llvm.array<1 x i8>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_2]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_3]] : !llvm.array<1 x i8>
  %0 = pop.union.wrap %arg0 : !kgen.struct<()> as <struct<(i8)>, struct<()>>
  kgen.return %0 : !pop.union<struct<(i8)>, struct<()>>
}

// CHECK-LABEL: @union_wrap_nonempty_empty_first
// Same union but with empty struct declared first; this order did not crash
// before the fix, but is included as a regression guard.
kgen.func @union_wrap_nonempty_empty_first(%arg0: !kgen.struct<(i8)>) -> !pop.union<struct<()>, struct<(i8)>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<1 x i8> {alignment = 1 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 1 : i64} : !llvm.struct<(i8)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 1 : i64} : !llvm.ptr -> !llvm.array<1 x i8>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.array<1 x i8>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i8)> as <struct<()>, struct<(i8)>>
  kgen.return %0 : !pop.union<struct<()>, struct<(i8)>>
}

// CHECK-LABEL: @union_unwrap_nonempty_with_empty_sibling
// Unwrap the non-empty variant from a union that also has an empty struct.
kgen.func @union_unwrap_nonempty_with_empty_sibling(%arg0: !pop.union<struct<(i8)>, struct<()>>) -> !kgen.struct<(i8)> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<1 x i8> {alignment = 1 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 1 : i64} : !llvm.array<1 x i8>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 1 : i64} : !llvm.ptr -> !llvm.struct<(i8)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(i8)>
  %0 = pop.union.unwrap %arg0 : <struct<(i8)>, struct<()>> as !kgen.struct<(i8)>
  kgen.return %0 : !kgen.struct<(i8)>
}

// Regression test for MOCO-3900: a union used to be lowered to
// !llvm.struct<(max_align_t, padding)>, reusing one member's own field as
// that struct's first field. For !kgen.struct<(i8, i1)>, both fields have
// the same alignment, and a tie-break bug in `getTypeABIAlignAndType` always
// picked the *last* tied field — here, the i1 from Bool. But i1 is only 1
// bit of real data even though it takes a whole byte in memory. Loading the
// *other* field's byte through that i1 threw away 7 of its 8 bits, silently
// destroying the leading i8's value. Now the union is just a byte array, so
// this can't happen.
// CHECK-LABEL: @union_wrap_struct_with_trailing_bool
kgen.func @union_wrap_struct_with_trailing_bool(%arg0: !kgen.struct<(i8, i1)>) -> !pop.union<struct<(i8, i1)>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<2 x i8> {alignment = 1 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 1 : i64} : !llvm.struct<(i8, i1)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 1 : i64} : !llvm.ptr -> !llvm.array<2 x i8>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i8, i1)> as <struct<(i8, i1)>>
  kgen.return %0 : !pop.union<struct<(i8, i1)>>
}

// Unwrap side of the same regression: the union must round-trip back to
// !kgen.struct<(i8, i1)> without losing the leading i8's bits.
// CHECK-LABEL: @union_unwrap_struct_with_trailing_bool
kgen.func @union_unwrap_struct_with_trailing_bool(%arg0: !pop.union<struct<(i8, i1)>>) -> !kgen.struct<(i8, i1)> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.array<2 x i8> {alignment = 1 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] {alignment = 1 : i64} : !llvm.array<2 x i8>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] {alignment = 1 : i64} : !llvm.ptr -> !llvm.struct<(i8, i1)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <struct<(i8, i1)>> as !kgen.struct<(i8, i1)>
  kgen.return %0 : !kgen.struct<(i8, i1)>
}

// Same bug, compile-time-constant version: `VariantHelper::materializeLLVMUnion`
// packed a constant's bytes into storage slots, and it also sized the first
// slot after the (buggy) representative field. Sizing that first slot as
// i1 (1 bit) instead of a full byte pushed every later byte 7 bits out of
// place, corrupting the whole constant.
// CHECK-LABEL: @union_constant_struct_with_trailing_bool
kgen.func @union_constant_struct_with_trailing_bool() -> !pop.union<struct<(i8, i1)>> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<2 x i8>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(false) : i1
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(true) : i1
  // CHECK-DAG:  %[[VAL_4:.*]] = llvm.mlir.constant(-86 : i8) : i8
  // CHECK:      %[[VAL_5:.*]] = llvm.lshr %[[VAL_4]], %[[VAL_2]] : i8
  // CHECK:      %[[VAL_6:.*]] = llvm.trunc %[[VAL_5]] : i8 to i8
  // CHECK:      %[[VAL_7:.*]] = llvm.shl %[[VAL_6]], %[[VAL_2]] : i8
  // CHECK:      %[[VAL_8:.*]] = llvm.or %[[VAL_2]], %[[VAL_7]] : i8
  // CHECK:      %[[VAL_9:.*]] = llvm.lshr %[[VAL_3]], %[[VAL_1]] : i1
  // CHECK:      %[[VAL_10:.*]] = llvm.zext %[[VAL_9]] : i1 to i8
  // CHECK:      %[[VAL_11:.*]] = llvm.shl %[[VAL_10]], %[[VAL_2]] : i8
  // CHECK:      %[[VAL_12:.*]] = llvm.or %[[VAL_2]], %[[VAL_11]] : i8
  // CHECK:      %[[VAL_13:.*]] = llvm.insertvalue %[[VAL_8]], %[[VAL_0]][0] : !llvm.array<2 x i8>
  // CHECK:      %[[VAL_14:.*]] = llvm.insertvalue %[[VAL_12]], %[[VAL_13]][1] : !llvm.array<2 x i8>
  %0 = kgen.param.constant: union<struct<(i8, i1)>> = <{:struct<(i8, i1)> { 170, 1 }}>
  kgen.return %0 : !pop.union<struct<(i8, i1)>>
}

}
