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
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i32)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : i32, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i32)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.wrap %arg0 : i32 as <i32>
  kgen.return %0 : !pop.union<i32>
}

// CHECK-LABEL: @union_create_1
kgen.func @union_create_1(%arg0: i8) -> !pop.union<i8> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i8)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
// CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
// CHECK:           llvm.store %arg0, %[[VAL_1]] : i8, !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i8)>
// CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.wrap %arg0 : i8 as <i8>
  kgen.return %0 : !pop.union<i8>
}

// CHECK-LABEL: @union_create_2
kgen.func @union_create_2(%arg0: f64) -> !pop.union<f64> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(f64)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : f64, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(f64)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(f64)>
  %0 = pop.union.wrap %arg0 : f64 as <f64>
  kgen.return %0 : !pop.union<f64>
}

// CHECK-LABEL: @union_create_3
kgen.func @union_create_3(%arg0: !kgen.struct<(i32, i32)>) -> !pop.union<struct<(i32, i32)>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i32, array<4 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(i32, i32)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i32, array<4 x i8>)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(i32, array<4 x i8>)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i32, i32)> as <struct<(i32, i32)>>
  kgen.return %0 : !pop.union<struct<(i32, i32)>>
}

// CHECK-LABEL: @union_create_4
kgen.func @union_create_4(%arg0: !kgen.struct<(i32, i64, i32)>) -> !pop.union<struct<(i32, i64, i32)>, array<4, i64>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i64, array<24 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(i32, i64, i32)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i64, array<24 x i8>)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(i64, array<24 x i8>)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i32, i64, i32)> as <struct<(i32, i64, i32)>, array<4, i64>>
  kgen.return %0 : !pop.union<struct<(i32, i64, i32)>, array<4, i64>>
}

// CHECK-LABEL: @union_create_5
kgen.func @union_create_5(%arg0: !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>) -> !pop.union<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(f32, array<16 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(f32, array<16 x i8>)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(f32, array<16 x i8>)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)> as <struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>
  kgen.return %0 : !pop.union<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>
}

// CHECK-LABEL: @union_create_6
kgen.func @union_create_6(%arg0: !kgen.pointer<index>) -> !pop.union<pointer<index>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.ptr, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(ptr)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(ptr)>
  %0 = pop.union.wrap %arg0 : !kgen.pointer<index> as <pointer<index>>
  kgen.return %0 : !pop.union<pointer<index>>
}

// CHECK-LABEL: @union_get_0
kgen.func @union_get_0(%arg0: !pop.union<i32>) ->  i32{
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i32)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(i32)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i32
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <i32> as i32
  kgen.return %0 : i32
}

// CHECK-LABEL: @union_get_1
kgen.func @union_get_1(%arg0: !pop.union<f64>) -> f64 {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(f64)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(f64)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> f64
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : f64
  %0 = pop.union.unwrap %arg0 : <f64> as f64
  kgen.return %0 : f64
}

// CHECK-LABEL: @union_get_2
kgen.func @union_get_2(%arg0: !pop.union<struct<(i32, i32)>>) -> !kgen.struct<(i32, i32)>{
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i32, array<4 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(i32, array<4 x i8>)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i32, i32)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(i32, i32)>
  %0 = pop.union.unwrap %arg0 : <struct<(i32, i32)>> as !kgen.struct<(i32, i32)>
  kgen.return %0 : !kgen.struct<(i32, i32)>
}

// CHECK-LABEL: @union_get_3
kgen.func @union_get_3(%arg0: !pop.union<struct<(i32, i64, i32)>, array<4, i64>>) -> !kgen.struct<(i32, i64, i32)> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i64, array<24 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(i64, array<24 x i8>)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i32, i64, i32)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <struct<(i32, i64, i32)>, array<4, i64>> as !kgen.struct<(i32, i64, i32)>
  kgen.return %0 : !kgen.struct<(i32, i64, i32)>
}

// CHECK-LABEL: @union_get_4
kgen.func @union_get_4(%arg0: !pop.union<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>) -> !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(f32, array<16 x i8>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(f32, array<16 x i8>)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>> as !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>
  kgen.return %0 : !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>
}

// CHECK-LABEL: @union_get_5
kgen.func @union_get_5(%arg0: !pop.union<pointer<index>>) -> !kgen.pointer<index> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(ptr)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <pointer<index>> as !kgen.pointer<index>
  kgen.return %0 : !kgen.pointer<index>
}

// CHECK-LABEL: @unpack_pointer
kgen.func @unpack_pointer(%arg0: !pop.union<pointer<i8>>) -> !kgen.pointer<i8> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(ptr)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <pointer<i8>> as !kgen.pointer<i8>
  kgen.return %0 : !kgen.pointer<i8>
}

// CHECK-LABEL: @union_constant_0
kgen.func @union_constant_0() -> !pop.union<i32> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:      %[[VAL_3:.*]] = llvm.lshr %[[VAL_1]], %[[VAL_2]] : i32
  // CHECK:      %[[VAL_4:.*]] = llvm.trunc %[[VAL_3]] : i32 to i32
  // CHECK:      %[[VAL_5:.*]] = llvm.shl %[[VAL_4]], %[[VAL_2]] : i32
  // CHECK:      %[[VAL_6:.*]] = llvm.or %[[VAL_2]], %[[VAL_5]] : i32
  // CHECK:      %[[VAL_7:.*]] = llvm.insertvalue %[[VAL_6]], %[[VAL_0]][0] : !llvm.struct<(i32)>
  %0 = kgen.param.constant: union<i32> = <{:i32 1}>
  kgen.return %0 : !pop.union<i32>
}

// CHECK-LABEL: @union_constant_1
kgen.func @union_constant_1() -> !pop.union<struct<(i32, i64, i32)>, struct<(f64, f32)>> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<8 x i8>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.undef : !llvm.struct<(f64, array<8 x i8>)>
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  %[[VAL_4:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG:  %[[VAL_5:.*]] = llvm.mlir.constant(56 : i64) : i64
  // CHECK-DAG:  %[[VAL_6:.*]] = llvm.mlir.constant(48 : i64) : i64
  // CHECK-DAG:  %[[VAL_7:.*]] = llvm.mlir.constant(40 : i64) : i64
  // CHECK-DAG:  %[[VAL_8:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK-DAG:  %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:  %[[VAL_10:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-DAG:  %[[VAL_11:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG:  %[[VAL_12:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-DAG:  %[[VAL_13:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK-DAG:  %[[VAL_14:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:      %[[VAL_15:.*]] = llvm.lshr %[[VAL_14]], %[[VAL_9]] : i32
  // CHECK:      %[[VAL_16:.*]] = llvm.zext %[[VAL_15]] : i32 to i64
  // CHECK:      %[[VAL_17:.*]] = llvm.shl %[[VAL_16]], %[[VAL_11]] : i64
  // CHECK:      %[[VAL_18:.*]] = llvm.or %[[VAL_11]], %[[VAL_17]] : i64
  // CHECK:      %[[VAL_19:.*]] = llvm.lshr %[[VAL_13]], %[[VAL_11]] : i64
  // CHECK:      %[[VAL_20:.*]] = llvm.trunc %[[VAL_19]] : i64 to i64
  // CHECK:      %[[VAL_21:.*]] = llvm.shl %[[VAL_20]], %[[VAL_8]] : i64
  // CHECK:      %[[VAL_22:.*]] = llvm.or %[[VAL_18]], %[[VAL_21]] : i64
  // CHECK:      %[[VAL_23:.*]] = llvm.lshr %[[VAL_13]], %[[VAL_8]] : i64
  // CHECK:      %[[VAL_24:.*]] = llvm.trunc %[[VAL_23]] : i64 to i8
  // CHECK:      %[[VAL_25:.*]] = llvm.shl %[[VAL_24]], %[[VAL_10]] : i8
  // CHECK:      %[[VAL_26:.*]] = llvm.or %[[VAL_10]], %[[VAL_25]] : i8
  // CHECK:      %[[VAL_27:.*]] = llvm.lshr %[[VAL_13]], %[[VAL_7]] : i64
  // CHECK:      %[[VAL_28:.*]] = llvm.trunc %[[VAL_27]] : i64 to i8
  // CHECK:      %[[VAL_29:.*]] = llvm.shl %[[VAL_28]], %[[VAL_10]] : i8
  // CHECK:      %[[VAL_30:.*]] = llvm.or %[[VAL_10]], %[[VAL_29]] : i8
  // CHECK:      %[[VAL_31:.*]] = llvm.lshr %[[VAL_13]], %[[VAL_6]] : i64
  // CHECK:      %[[VAL_32:.*]] = llvm.trunc %[[VAL_31]] : i64 to i8
  // CHECK:      %[[VAL_33:.*]] = llvm.shl %[[VAL_32]], %[[VAL_10]] : i8
  // CHECK:      %[[VAL_34:.*]] = llvm.or %[[VAL_10]], %[[VAL_33]] : i8
  // CHECK:      %[[VAL_35:.*]] = llvm.lshr %[[VAL_13]], %[[VAL_5]] : i64
  // CHECK:      %[[VAL_36:.*]] = llvm.trunc %[[VAL_35]] : i64 to i8
  // CHECK:      %[[VAL_37:.*]] = llvm.shl %[[VAL_36]], %[[VAL_10]] : i8
  // CHECK:      %[[VAL_38:.*]] = llvm.or %[[VAL_10]], %[[VAL_37]] : i8
  // CHECK:      %[[VAL_39:.*]] = llvm.lshr %[[VAL_12]], %[[VAL_9]] : i32
  // CHECK:      %[[VAL_40:.*]] = llvm.trunc %[[VAL_39]] : i32 to i8
  // CHECK:      %[[VAL_41:.*]] = llvm.shl %[[VAL_40]], %[[VAL_10]] : i8
  // CHECK:      %[[VAL_42:.*]] = llvm.or %[[VAL_10]], %[[VAL_41]] : i8
  // CHECK:      %[[VAL_43:.*]] = llvm.lshr %[[VAL_12]], %[[VAL_4]] : i32
  // CHECK:      %[[VAL_44:.*]] = llvm.trunc %[[VAL_43]] : i32 to i8
  // CHECK:      %[[VAL_45:.*]] = llvm.shl %[[VAL_44]], %[[VAL_10]] : i8
  // CHECK:      %[[VAL_46:.*]] = llvm.or %[[VAL_10]], %[[VAL_45]] : i8
  // CHECK:      %[[VAL_47:.*]] = llvm.lshr %[[VAL_12]], %[[VAL_3]] : i32
  // CHECK:      %[[VAL_48:.*]] = llvm.trunc %[[VAL_47]] : i32 to i8
  // CHECK:      %[[VAL_49:.*]] = llvm.shl %[[VAL_48]], %[[VAL_10]] : i8
  // CHECK:      %[[VAL_50:.*]] = llvm.or %[[VAL_10]], %[[VAL_49]] : i8
  // CHECK:      %[[VAL_51:.*]] = llvm.lshr %[[VAL_12]], %[[VAL_2]] : i32
  // CHECK:      %[[VAL_52:.*]] = llvm.trunc %[[VAL_51]] : i32 to i8
  // CHECK:      %[[VAL_53:.*]] = llvm.shl %[[VAL_52]], %[[VAL_10]] : i8
  // CHECK:      %[[VAL_54:.*]] = llvm.or %[[VAL_10]], %[[VAL_53]] : i8
  // CHECK:      %[[VAL_55:.*]] = llvm.bitcast %[[VAL_22]] : i64 to f64
  // CHECK:      %[[VAL_56:.*]] = llvm.insertvalue %[[VAL_55]], %[[VAL_1]][0] : !llvm.struct<(f64, array<8 x i8>)>
  // CHECK:      %[[VAL_57:.*]] = llvm.insertvalue %[[VAL_26]], %[[VAL_0]][0] : !llvm.array<8 x i8>
  // CHECK:      %[[VAL_58:.*]] = llvm.insertvalue %[[VAL_30]], %[[VAL_57]][1] : !llvm.array<8 x i8>
  // CHECK:      %[[VAL_59:.*]] = llvm.insertvalue %[[VAL_34]], %[[VAL_58]][2] : !llvm.array<8 x i8>
  // CHECK:      %[[VAL_60:.*]] = llvm.insertvalue %[[VAL_38]], %[[VAL_59]][3] : !llvm.array<8 x i8>
  // CHECK:      %[[VAL_61:.*]] = llvm.insertvalue %[[VAL_42]], %[[VAL_60]][4] : !llvm.array<8 x i8>
  // CHECK:      %[[VAL_62:.*]] = llvm.insertvalue %[[VAL_46]], %[[VAL_61]][5] : !llvm.array<8 x i8>
  // CHECK:      %[[VAL_63:.*]] = llvm.insertvalue %[[VAL_50]], %[[VAL_62]][6] : !llvm.array<8 x i8>
  // CHECK:      %[[VAL_64:.*]] = llvm.insertvalue %[[VAL_54]], %[[VAL_63]][7] : !llvm.array<8 x i8>
  // CHECK:      %[[VAL_65:.*]] = llvm.insertvalue %[[VAL_64]], %[[VAL_56]][1] : !llvm.struct<(f64, array<8 x i8>)>
  %0 = kgen.param.constant: union<struct<(i32, i64, i32)>, struct<(f64, f32)>> = <{:struct<(i32, i64, i32)> { 1, 2, 3 }}>
  kgen.return %0 : !pop.union<struct<(i32, i64, i32)>, struct<(f64, f32)>>
}

// CHECK-LABEL: @union_constant_2
kgen.func @union_constant_2() -> !pop.union<i1, i2, i3, i4, i5, i6> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(i6)>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(1 : i4) : i4
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(0 : i6) : i6
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(0 : i4) : i4
  // CHECK:      %[[VAL_4:.*]] = llvm.lshr %[[VAL_1]], %[[VAL_3]] : i4
  // CHECK:      %[[VAL_5:.*]] = llvm.zext %[[VAL_4]] : i4 to i6
  // CHECK:      %[[VAL_6:.*]] = llvm.shl %[[VAL_5]], %[[VAL_2]] : i6
  // CHECK:      %[[VAL_7:.*]] = llvm.or %[[VAL_2]], %[[VAL_6]] : i6
  // CHECK:      %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_0]][0] : !llvm.struct<(i6)>
  %0 = kgen.param.constant: union<i1, i2, i3, i4, i5, i6> = <{:i4 1}>
  kgen.return %0 : !pop.union<i1, i2, i3, i4, i5, i6>
}

}
