// RUN: kgen-opt %s -pass-pipeline='builtin.module(lower-kgen-to-llvm,llvm.func(lower-pop-to-llvm,canonicalize))' | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @empty_union
// CHECK-SAME: () -> !llvm.struct<()>
kgen.func @empty_union() -> !pop.union<> {
  hlcf.unreachable
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
  hlcf.return %0 : !pop.union<i32>
}

// CHECK-LABEL: @union_create_1
// Union alignment is now max of variant alignments (i8 = 1 byte).
kgen.func @union_create_1(%arg0: i8) -> !pop.union<i8> {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i8)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
// CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
// CHECK:           llvm.store %arg0, %[[VAL_1]] : i8, !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i8)>
// CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.wrap %arg0 : i8 as <i8>
  hlcf.return %0 : !pop.union<i8>
}

// CHECK-LABEL: @union_create_2
// Union alignment is now max of variant alignments (f64 = 8 bytes).
kgen.func @union_create_2(%arg0: f64) -> !pop.union<f64> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(f64)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : f64, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(f64)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(f64)>
  %0 = pop.union.wrap %arg0 : f64 as <f64>
  hlcf.return %0 : !pop.union<f64>
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
  hlcf.return %0 : !pop.union<struct<(i32, i32)>>
}

// CHECK-LABEL: @union_create_4
kgen.func @union_create_4(%arg0: !kgen.struct<(i32, i64, i32)>) -> !pop.union<struct<(i32, i64, i32)>, array<4, i64>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i64, array<3 x i64>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(i32, i64, i32)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i64, array<3 x i64>)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(i64, array<3 x i64>)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i32, i64, i32)> as <struct<(i32, i64, i32)>, array<4, i64>>
  hlcf.return %0 : !pop.union<struct<(i32, i64, i32)>, array<4, i64>>
}

// CHECK-LABEL: @union_create_5
// Union alignment is now max of variant alignments (simd<2, f32> = 8 bytes).
// Note: struct type changed due to alignment-based padding.
kgen.func @union_create_5(%arg0: !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>) -> !pop.union<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(f32, array<20 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(array<2 x i16>, array<4 x i8>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(f32, array<20 x i8>)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(f32, array<20 x i8>)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)> as <struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>
  hlcf.return %0 : !pop.union<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>
}

// CHECK-LABEL: @union_create_6
// Union alignment is now max of variant alignments (pointer = 8 bytes).
kgen.func @union_create_6(%arg0: !kgen.pointer<index>) -> !pop.union<pointer<index>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.ptr, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(ptr)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(ptr)>
  %0 = pop.union.wrap %arg0 : !kgen.pointer<index> as <pointer<index>>
  hlcf.return %0 : !pop.union<pointer<index>>
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
  hlcf.return %0 : i32
}

// CHECK-LABEL: @union_get_1
// Union alignment is now max of variant alignments (f64 = 8 bytes).
kgen.func @union_get_1(%arg0: !pop.union<f64>) -> f64 {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(f64)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(f64)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> f64
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : f64
  %0 = pop.union.unwrap %arg0 : <f64> as f64
  hlcf.return %0 : f64
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
  hlcf.return %0 : !kgen.struct<(i32, i32)>
}

// CHECK-LABEL: @union_get_3
kgen.func @union_get_3(%arg0: !pop.union<struct<(i32, i64, i32)>, array<4, i64>>) -> !kgen.struct<(i32, i64, i32)> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i64, array<3 x i64>)> {alignment = 4 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(i64, array<3 x i64>)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i32, i64, i32)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <struct<(i32, i64, i32)>, array<4, i64>> as !kgen.struct<(i32, i64, i32)>
  hlcf.return %0 : !kgen.struct<(i32, i64, i32)>
}

// CHECK-LABEL: @union_get_4
// Union alignment is now max of variant alignments (simd<2, f32> = 8 bytes).
kgen.func @union_get_4(%arg0: !pop.union<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>) -> !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(f32, array<20 x i8>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(f32, array<20 x i8>)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(array<2 x i16>, array<4 x i8>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>> as !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>
  hlcf.return %0 : !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>
}

// CHECK-LABEL: @union_get_5
// Union alignment is now max of variant alignments (pointer = 8 bytes).
kgen.func @union_get_5(%arg0: !pop.union<pointer<index>>) -> !kgen.pointer<index> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(ptr)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <pointer<index>> as !kgen.pointer<index>
  hlcf.return %0 : !kgen.pointer<index>
}

// CHECK-LABEL: @unpack_pointer
// Union alignment is now max of variant alignments (pointer = 8 bytes).
kgen.func @unpack_pointer(%arg0: !pop.union<pointer<i8>>) -> !kgen.pointer<i8> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(ptr)> {alignment = 8 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(ptr)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  %0 = pop.union.unwrap %arg0 : <pointer<i8>> as !kgen.pointer<i8>
  hlcf.return %0 : !kgen.pointer<i8>
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
  hlcf.return %0 : !pop.union<i32>
}

// CHECK-LABEL: @union_constant_1
kgen.func @union_constant_1() -> !pop.union<struct<(i32, i64, i32)>, struct<(f64, f32)>> {
  // CHECK: %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<1 x i64>
  // CHECK: %[[VAL_1:.*]] = llvm.mlir.undef : !llvm.struct<(f64, array<1 x i64>)>
  // CHECK: %[[VAL_2:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK: %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[VAL_4:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %[[VAL_5:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: %[[VAL_6:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[VAL_8:.*]] = llvm.lshr %[[VAL_7]], %[[VAL_3]] : i32
  // CHECK: %[[VAL_9:.*]] = llvm.zext %[[VAL_8]] : i32 to i64
  // CHECK: %[[VAL_10:.*]] = llvm.shl %[[VAL_9]], %[[VAL_4]] : i64
  // CHECK: %[[VAL_11:.*]] = llvm.or %[[VAL_4]], %[[VAL_10]] : i64
  // CHECK: %[[VAL_12:.*]] = llvm.lshr %[[VAL_6]], %[[VAL_4]] : i64
  // CHECK: %[[VAL_13:.*]] = llvm.trunc %[[VAL_12]] : i64 to i64
  // CHECK: %[[VAL_14:.*]] = llvm.shl %[[VAL_13]], %[[VAL_2]] : i64
  // CHECK: %[[VAL_15:.*]] = llvm.or %[[VAL_11]], %[[VAL_14]] : i64
  // CHECK: %[[VAL_16:.*]] = llvm.lshr %[[VAL_6]], %[[VAL_2]] : i64
  // CHECK: %[[VAL_17:.*]] = llvm.trunc %[[VAL_16]] : i64 to i64
  // CHECK: %[[VAL_18:.*]] = llvm.shl %[[VAL_17]], %[[VAL_4]] : i64
  // CHECK: %[[VAL_19:.*]] = llvm.or %[[VAL_4]], %[[VAL_18]] : i64
  // CHECK: %[[VAL_20:.*]] = llvm.lshr %[[VAL_5]], %[[VAL_3]] : i32
  // CHECK: %[[VAL_21:.*]] = llvm.zext %[[VAL_20]] : i32 to i64
  // CHECK: %[[VAL_22:.*]] = llvm.shl %[[VAL_21]], %[[VAL_2]] : i64
  // CHECK: %[[VAL_23:.*]] = llvm.or %[[VAL_19]], %[[VAL_22]] : i64
  // CHECK: %[[VAL_24:.*]] = llvm.bitcast %[[VAL_15]] : i64 to f64
  // CHECK: %[[VAL_25:.*]] = llvm.insertvalue %[[VAL_24]], %[[VAL_1]][0] : !llvm.struct<(f64, array<1 x i64>)>
  // CHECK: %[[VAL_26:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_0]][0] : !llvm.array<1 x i64>
  // CHECK: %[[VAL_27:.*]] = llvm.insertvalue %[[VAL_26]], %[[VAL_25]][1] : !llvm.struct<(f64, array<1 x i64>)>
  // CHECK: llvm.return %[[VAL_27]] : !llvm.struct<(f64, array<1 x i64>)>
  %0 = kgen.param.constant: union<struct<(i32, i64, i32)>, struct<(f64, f32)>> = <{:struct<(i32, i64, i32)> { 1, 2, 3 }}>
  hlcf.return %0 : !pop.union<struct<(i32, i64, i32)>, struct<(f64, f32)>>
}

// CHECK-LABEL: @union_constant_2
// All members are sub-byte-width (i1..i6): falls back to i8, not the old i6.
kgen.func @union_constant_2() -> !pop.union<i1, i2, i3, i4, i5, i6> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(i8)>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(1 : i4) : i4
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(0 : i4) : i4
  // CHECK:      %[[VAL_4:.*]] = llvm.lshr %[[VAL_1]], %[[VAL_3]] : i4
  // CHECK:      %[[VAL_5:.*]] = llvm.zext %[[VAL_4]] : i4 to i8
  // CHECK:      %[[VAL_6:.*]] = llvm.shl %[[VAL_5]], %[[VAL_2]] : i8
  // CHECK:      %[[VAL_7:.*]] = llvm.or %[[VAL_2]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_0]][0] : !llvm.struct<(i8)>
  %0 = kgen.param.constant: union<i1, i2, i3, i4, i5, i6> = <{:i4 1}>
  hlcf.return %0 : !pop.union<i1, i2, i3, i4, i5, i6>
}

// CHECK-LABEL: @union_wrap_nonempty_with_empty_sibling
// Union contains a non-empty struct variant followed by an empty struct variant.
// The non-empty variant comes first so its {align=1, type=i8} was previously
// overwritten by the empty struct's {align=1, type=null} via the '>=' path,
// causing a null dereference in getTypeSizeInBits (MOCO-3275).
// The union lowers to !llvm.struct<(i8)>: maxSize=1, maxAlignTp=i8, remLen=0.
kgen.func @union_wrap_nonempty_with_empty_sibling(%arg0: !kgen.struct<(i8)>) -> !pop.union<struct<(i8)>, struct<()>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i8)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(i8)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i8)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(i8)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i8)> as <struct<(i8)>, struct<()>>
  hlcf.return %0 : !pop.union<struct<(i8)>, struct<()>>
}

// CHECK-LABEL: @union_wrap_empty_with_nonempty_sibling
// CHECK-SAME:  () -> !llvm.struct<(i8)>
// Wrap the empty struct variant into the same union type. lower-kgen-to-llvm
// eliminates the zero-size struct argument and replaces its use with undef.
kgen.func @union_wrap_empty_with_nonempty_sibling(%arg0: !kgen.struct<()>) -> !pop.union<struct<(i8)>, struct<()>> {
  // CHECK-DAG:       %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK-DAG:       %[[VAL_1:.*]] = llvm.mlir.undef : !llvm.struct<()>
  // CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i8)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_2]] : !llvm.ptr
  // CHECK:           llvm.store %[[VAL_1]], %[[VAL_2]] : !llvm.struct<()>, !llvm.ptr
  // CHECK:           %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.struct<(i8)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_2]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_3]] : !llvm.struct<(i8)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<()> as <struct<(i8)>, struct<()>>
  hlcf.return %0 : !pop.union<struct<(i8)>, struct<()>>
}

// CHECK-LABEL: @union_wrap_nonempty_empty_first
// Same union but with empty struct declared first; this order did not crash
// before the fix, but is included as a regression guard.
kgen.func @union_wrap_nonempty_empty_first(%arg0: !kgen.struct<(i8)>) -> !pop.union<struct<()>, struct<(i8)>> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i8)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(i8)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i8)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(i8)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i8)> as <struct<()>, struct<(i8)>>
  hlcf.return %0 : !pop.union<struct<()>, struct<(i8)>>
}

// CHECK-LABEL: @union_unwrap_nonempty_with_empty_sibling
// Unwrap the non-empty variant from a union that also has an empty struct.
kgen.func @union_unwrap_nonempty_with_empty_sibling(%arg0: !pop.union<struct<(i8)>, struct<()>>) -> !kgen.struct<(i8)> {
  // CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[VAL_1:.*]] = llvm.alloca %[[VAL_0]] x !llvm.struct<(i8)> {alignment = 1 : i64} : (i64) -> !llvm.ptr
  // CHECK:           llvm.intr.lifetime.start %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.store %arg0, %[[VAL_1]] : !llvm.struct<(i8)>, !llvm.ptr
  // CHECK:           %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.struct<(i8)>
  // CHECK:           llvm.intr.lifetime.end %[[VAL_1]] : !llvm.ptr
  // CHECK:           llvm.return %[[VAL_2]] : !llvm.struct<(i8)>
  %0 = pop.union.unwrap %arg0 : <struct<(i8)>, struct<()>> as !kgen.struct<(i8)>
  hlcf.return %0 : !kgen.struct<(i8)>
}

// Regression test for MOCO-3900: representative must be i8, not i1.
// CHECK-LABEL: @union_wrap_struct_with_trailing_bool
// CHECK-SAME:  -> !llvm.struct<(i8, array<1 x i8>)>
kgen.func @union_wrap_struct_with_trailing_bool(%arg0: !kgen.struct<(i8, i1)>) -> !pop.union<struct<(i8, i1)>> {
  // CHECK:      llvm.store %arg0, %{{.*}} : !llvm.struct<(i8, i1)>, !llvm.ptr
  // CHECK:      %[[V:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i8, array<1 x i8>)>
  // CHECK:      llvm.return %[[V]] : !llvm.struct<(i8, array<1 x i8>)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i8, i1)> as <struct<(i8, i1)>>
  hlcf.return %0 : !pop.union<struct<(i8, i1)>>
}

// CHECK-LABEL: @union_unwrap_struct_with_trailing_bool
// CHECK-SAME:  -> !llvm.struct<(i8, i1)>
kgen.func @union_unwrap_struct_with_trailing_bool(%arg0: !pop.union<struct<(i8, i1)>>) -> !kgen.struct<(i8, i1)> {
  // CHECK:      %[[V:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i8, i1)>
  // CHECK:      llvm.return %[[V]] : !llvm.struct<(i8, i1)>
  %0 = pop.union.unwrap %arg0 : <struct<(i8, i1)>> as !kgen.struct<(i8, i1)>
  hlcf.return %0 : !kgen.struct<(i8, i1)>
}

// Same regression, compile-time-constant path. 170 = 0xAA = -86 as i8.
// CHECK-LABEL: @union_constant_struct_with_trailing_bool
kgen.func @union_constant_struct_with_trailing_bool() -> !pop.union<struct<(i8, i1)>> {
  // CHECK-DAG:  %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i8, array<1 x i8>)>
  // CHECK-DAG:  %[[BYTE:.*]] = llvm.mlir.constant(-86 : i8) : i8
  // CHECK:      %[[F0:.*]] = llvm.insertvalue %{{.*}}, %[[UNDEF]][0] : !llvm.struct<(i8, array<1 x i8>)>
  // CHECK:      %{{.*}} = llvm.insertvalue %{{.*}}, %[[F0]][1] : !llvm.struct<(i8, array<1 x i8>)>
  %0 = kgen.param.constant: union<struct<(i8, i1)>> = <{:struct<(i8, i1)> { 170, 1 }}>
  hlcf.return %0 : !pop.union<struct<(i8, i1)>>
}

// Reversed field order: proves the fix is order-independent.
// CHECK-LABEL: @union_wrap_struct_with_leading_bool
// CHECK-SAME:  -> !llvm.struct<(i8, array<1 x i8>)>
kgen.func @union_wrap_struct_with_leading_bool(%arg0: !kgen.struct<(i1, i8)>) -> !pop.union<struct<(i1, i8)>> {
  // CHECK:      llvm.store %arg0, %{{.*}} : !llvm.struct<(i1, i8)>, !llvm.ptr
  // CHECK:      %[[V:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i8, array<1 x i8>)>
  // CHECK:      llvm.return %[[V]] : !llvm.struct<(i8, array<1 x i8>)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i1, i8)> as <struct<(i1, i8)>>
  hlcf.return %0 : !pop.union<struct<(i1, i8)>>
}

// CHECK-LABEL: @union_unwrap_struct_with_leading_bool
// CHECK-SAME:  -> !llvm.struct<(i1, i8)>
kgen.func @union_unwrap_struct_with_leading_bool(%arg0: !pop.union<struct<(i1, i8)>>) -> !kgen.struct<(i1, i8)> {
  // CHECK:      %[[V:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i1, i8)>
  // CHECK:      llvm.return %[[V]] : !llvm.struct<(i1, i8)>
  %0 = pop.union.unwrap %arg0 : <struct<(i1, i8)>> as !kgen.struct<(i1, i8)>
  hlcf.return %0 : !kgen.struct<(i1, i8)>
}

// CHECK-LABEL: @union_constant_struct_with_leading_bool
kgen.func @union_constant_struct_with_leading_bool() -> !pop.union<struct<(i1, i8)>> {
  // CHECK-DAG:  %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i8, array<1 x i8>)>
  // CHECK-DAG:  %[[BYTE:.*]] = llvm.mlir.constant(-86 : i8) : i8
  // CHECK:      llvm.insertvalue %{{.*}}[0] : !llvm.struct<(i8, array<1 x i8>)>
  // CHECK:      llvm.insertvalue %{{.*}}[1] : !llvm.struct<(i8, array<1 x i8>)>
  %0 = kgen.param.constant: union<struct<(i1, i8)>> = <{:struct<(i1, i8)> { 1, 170 }}>
  hlcf.return %0 : !pop.union<struct<(i1, i8)>>
}

// Same check with i8/i1 as direct union members, not nested in a struct.
// CHECK-LABEL: @union_wrap_direct_i8_i1
// CHECK-SAME:  -> !llvm.struct<(i8)>
kgen.func @union_wrap_direct_i8_i1(%arg0: i8) -> !pop.union<i8, i1> {
  // CHECK:      llvm.store %arg0, %{{.*}} : i8, !llvm.ptr
  // CHECK:      %[[V:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i8)>
  // CHECK:      llvm.return %[[V]] : !llvm.struct<(i8)>
  %0 = pop.union.wrap %arg0 : i8 as <i8, i1>
  hlcf.return %0 : !pop.union<i8, i1>
}

// CHECK-LABEL: @union_wrap_direct_i1_i8
// CHECK-SAME:  -> !llvm.struct<(i8)>
kgen.func @union_wrap_direct_i1_i8(%arg0: i8) -> !pop.union<i1, i8> {
  // CHECK:      llvm.store %arg0, %{{.*}} : i8, !llvm.ptr
  // CHECK:      %[[V:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i8)>
  // CHECK:      llvm.return %[[V]] : !llvm.struct<(i8)>
  %0 = pop.union.wrap %arg0 : i8 as <i1, i8>
  hlcf.return %0 : !pop.union<i1, i8>
}

// A real Optional/Variant is a struct containing a union field; guards that
// no trailing padding field is emitted for it (MOCO-4405).
// CHECK-LABEL: @variant_in_struct
// CHECK-SAME:  (%{{.*}}: !llvm.struct<(struct<(f64)>, i8)> {{.*}}) -> !llvm.struct<(struct<(f64)>, i8)>
// CHECK-NOT:   array<
kgen.func @variant_in_struct(%arg0: !kgen.struct<(union<i64, f64>, i8)>) -> !kgen.struct<(union<i64, f64>, i8)> {
  // CHECK:      llvm.return %arg0 : !llvm.struct<(struct<(f64)>, i8)>
  hlcf.return %arg0 : !kgen.struct<(union<i64, f64>, i8)>
}

// Regression test: `f80`'s alloc size (padded for alignment) is wider than
// its store size, which previously undersized the tail array and crashed.
// CHECK-LABEL: @union_f80
// CHECK-SAME:  -> !llvm.struct<(f80)>
kgen.func @union_f80(%arg0: !pop.union<f80>) -> !pop.union<f80> {
  // CHECK: llvm.return %arg0 : !llvm.struct<(f80)>
  hlcf.return %arg0 : !pop.union<f80>
}

// Regression test: `i15` is excluded as a representative (unsafe), but its
// true alignment (2 bytes) exceeds the safe fallback `i8`'s (1 byte). The
// representative is widened to `i16` -- matching the union's true alignment
// natively, with no tail array needed -- rather than silently under-reporting
// it. Order-independent, both declaration orders synthesize the same `i16`.
// CHECK-LABEL: @union_i8_i15
// CHECK-SAME:  -> !llvm.struct<(i16)>
kgen.func @union_i8_i15(%arg0: i8) -> !pop.union<i8, i15> {
  // CHECK: llvm.return %{{.*}} : !llvm.struct<(i16)>
  %0 = pop.union.wrap %arg0 : i8 as <i8, i15>
  hlcf.return %0 : !pop.union<i8, i15>
}

// CHECK-LABEL: @union_i15_i8
// CHECK-SAME:  -> !llvm.struct<(i16)>
kgen.func @union_i15_i8(%arg0: i8) -> !pop.union<i15, i8> {
  // CHECK: llvm.return %{{.*}} : !llvm.struct<(i16)>
  %0 = pop.union.wrap %arg0 : i8 as <i15, i8>
  hlcf.return %0 : !pop.union<i15, i8>
}


// Regression test for MOCO-4689: simd<N, bool> is N bytes to KGEN but converts
// to vector<N x i1>, which LLVM allocates in ceil(N/8). Sizing the payload from
// the converted type undersized it below the representative the alignment step
// then picks.
// CHECK-LABEL: @union_simd_bool_2
// CHECK-SAME:  -> !llvm.struct<(i16)>
kgen.func @union_simd_bool_2(%arg0: !kgen.simd<2, bool>) -> !pop.union<!kgen.simd<2, bool>> {
  // CHECK: llvm.return %{{.*}} : !llvm.struct<(i16)>
  %0 = pop.union.wrap %arg0 : !kgen.simd<2, bool> as <!kgen.simd<2, bool>>
  hlcf.return %0 : !pop.union<!kgen.simd<2, bool>>
}

// CHECK-LABEL: @union_simd_bool_4
// CHECK-SAME:  -> !llvm.struct<(i32)>
kgen.func @union_simd_bool_4(%arg0: !kgen.simd<4, bool>) -> !pop.union<!kgen.simd<4, bool>> {
  // CHECK: llvm.return %{{.*}} : !llvm.struct<(i32)>
  %0 = pop.union.wrap %arg0 : !kgen.simd<4, bool> as <!kgen.simd<4, bool>>
  hlcf.return %0 : !pop.union<!kgen.simd<4, bool>>
}

// A sibling member wide enough to dominate the payload masked the bug before,
// and must keep lowering to the same 8-byte union.
// CHECK-LABEL: @union_simd_bool_2_i64
// CHECK-SAME:  -> !llvm.struct<(i64)>
kgen.func @union_simd_bool_2_i64(%arg0: i64) -> !pop.union<!kgen.simd<2, bool>, i64> {
  // CHECK: llvm.return %{{.*}} : !llvm.struct<(i64)>
  %0 = pop.union.wrap %arg0 : i64 as <!kgen.simd<2, bool>, i64>
  hlcf.return %0 : !pop.union<!kgen.simd<2, bool>, i64>
}

// i16 is the wider member to LLVM, simd<4, bool> to KGEN, so the payload must
// size on the latter. The representative widens to i32 to carry the union's
// alignment, leaving no tail.
// CHECK-LABEL: @union_simd_bool_4_i16
// CHECK-SAME:  -> !llvm.struct<(i32)>
kgen.func @union_simd_bool_4_i16(%arg0: i16) -> !pop.union<!kgen.simd<4, bool>, i16> {
  // CHECK: llvm.return %{{.*}} : !llvm.struct<(i32)>
  %0 = pop.union.wrap %arg0 : i16 as <!kgen.simd<4, bool>, i16>
  hlcf.return %0 : !pop.union<!kgen.simd<4, bool>, i16>
}

}
