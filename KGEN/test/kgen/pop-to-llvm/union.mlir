// RUN: kgen-opt %s -pass-pipeline='builtin.module(lower-kgen-to-llvm,llvm.func(lower-pop-to-llvm,canonicalize))' | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @empty_union
// CHECK-SAME: () -> !llvm.struct<()>
kgen.func @empty_union() -> !pop.union<> {
  kgen.unreachable
}

// CHECK-LABEL: @union_create_0
kgen.func @union_create_0(%arg0: i32) -> !pop.union<i32> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(0 : i32) 
  // CHECK:      %[[VAL_2:.*]] = llvm.lshr %{{.*}}, %[[VAL_1]] : i32
  // CHECK:      %[[VAL_3:.*]] = llvm.trunc %[[VAL_2]] : i32 to i32
  // CHECK:      %[[VAL_4:.*]] = llvm.shl %[[VAL_3]], %[[VAL_1]] : i32
  // CHECK:      %[[VAL_5:.*]] = llvm.or %[[VAL_1]], %[[VAL_4]] : i32
  // CHECK:      %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_0]][0] : !llvm.struct<(i32)>
  %0 = pop.union.wrap %arg0 : i32 as <i32>
  kgen.return %0 : !pop.union<i32>
}

// CHECK-LABEL: @union_create_1
kgen.func @union_create_1(%arg0: i8) -> !pop.union<i8> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(i8)>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK:      %[[VAL_2:.*]] = llvm.lshr %{{.*}}, %[[VAL_1]] : i8
  // CHECK:      %[[VAL_3:.*]] = llvm.trunc %[[VAL_2]] : i8 to i8
  // CHECK:      %[[VAL_4:.*]] = llvm.shl %[[VAL_3]], %[[VAL_1]] : i8
  // CHECK:      %[[VAL_5:.*]] = llvm.or %[[VAL_1]], %[[VAL_4]] : i8
  // CHECK:      %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_5]], %[[VAL_0]][0] : !llvm.struct<(i8)>
  %0 = pop.union.wrap %arg0 : i8 as <i8>
  kgen.return %0 : !pop.union<i8>
}

// CHECK-LABEL: @union_create_2
kgen.func @union_create_2(%arg0: f64) -> !pop.union<f64> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.struct<(f64)>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK:      %[[VAL_2:.*]] = llvm.bitcast %{{.*}} : f64 to i64
  // CHECK:      %[[VAL_3:.*]] = llvm.lshr %[[VAL_2]], %[[VAL_1]] : i64
  // CHECK:      %[[VAL_4:.*]] = llvm.trunc %[[VAL_3]] : i64 to i64
  // CHECK:      %[[VAL_5:.*]] = llvm.shl %[[VAL_4]], %[[VAL_1]] : i64
  // CHECK:      %[[VAL_6:.*]] = llvm.or %[[VAL_1]], %[[VAL_5]] : i64
  // CHECK:      %[[VAL_7:.*]] = llvm.bitcast %[[VAL_6]] : i64 to f64
  // CHECK:      %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_0]][0] : !llvm.struct<(f64)>
  %0 = pop.union.wrap %arg0 : f64 as <f64>
  kgen.return %0 : !pop.union<f64>
}

// CHECK-LABEL: @union_create_3
kgen.func @union_create_3(%arg0: !kgen.struct<(i32, i32)>) -> !pop.union<struct<(i32, i32)>> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<4 x i8>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.undef : !llvm.struct<(i32, array<4 x i8>)>
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  %[[VAL_4:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG:  %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:  %[[VAL_6:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK:      %[[VAL_7:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i32, i32)>
  // CHECK:      %[[VAL_8:.*]] = llvm.lshr %[[VAL_7]], %[[VAL_5]] : i32
  // CHECK:      %[[VAL_9:.*]] = llvm.trunc %[[VAL_8]] : i32 to i32
  // CHECK:      %[[VAL_10:.*]] = llvm.shl %[[VAL_9]], %[[VAL_5]] : i32
  // CHECK:      %[[VAL_11:.*]] = llvm.or %[[VAL_5]], %[[VAL_10]] : i32
  // CHECK:      %[[VAL_12:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(i32, i32)>
  // CHECK:      %[[VAL_13:.*]] = llvm.lshr %[[VAL_12]], %[[VAL_5]] : i32
  // CHECK:      %[[VAL_14:.*]] = llvm.trunc %[[VAL_13]] : i32 to i8
  // CHECK:      %[[VAL_15:.*]] = llvm.shl %[[VAL_14]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_16:.*]] = llvm.or %[[VAL_6]], %[[VAL_15]] : i8
  // CHECK:      %[[VAL_17:.*]] = llvm.lshr %[[VAL_12]], %[[VAL_4]] : i32
  // CHECK:      %[[VAL_18:.*]] = llvm.trunc %[[VAL_17]] : i32 to i8
  // CHECK:      %[[VAL_19:.*]] = llvm.shl %[[VAL_18]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_20:.*]] = llvm.or %[[VAL_6]], %[[VAL_19]] : i8
  // CHECK:      %[[VAL_21:.*]] = llvm.lshr %[[VAL_12]], %[[VAL_3]] : i32
  // CHECK:      %[[VAL_22:.*]] = llvm.trunc %[[VAL_21]] : i32 to i8
  // CHECK:      %[[VAL_23:.*]] = llvm.shl %[[VAL_22]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_24:.*]] = llvm.or %[[VAL_6]], %[[VAL_23]] : i8
  // CHECK:      %[[VAL_25:.*]] = llvm.lshr %[[VAL_12]], %[[VAL_2]] : i32
  // CHECK:      %[[VAL_26:.*]] = llvm.trunc %[[VAL_25]] : i32 to i8
  // CHECK:      %[[VAL_27:.*]] = llvm.shl %[[VAL_26]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_28:.*]] = llvm.or %[[VAL_6]], %[[VAL_27]] : i8
  // CHECK:      %[[VAL_29:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_1]][0] : !llvm.struct<(i32, array<4 x i8>)>
  // CHECK:      %[[VAL_30:.*]] = llvm.insertvalue %[[VAL_16]], %[[VAL_0]][0] : !llvm.array<4 x i8>
  // CHECK:      %[[VAL_31:.*]] = llvm.insertvalue %[[VAL_20]], %[[VAL_30]][1] : !llvm.array<4 x i8>
  // CHECK:      %[[VAL_32:.*]] = llvm.insertvalue %[[VAL_24]], %[[VAL_31]][2] : !llvm.array<4 x i8>
  // CHECK:      %[[VAL_33:.*]] = llvm.insertvalue %[[VAL_28]], %[[VAL_32]][3] : !llvm.array<4 x i8>
  // CHECK:      %[[VAL_34:.*]] = llvm.insertvalue %[[VAL_33]], %[[VAL_29]][1] : !llvm.struct<(i32, array<4 x i8>)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i32, i32)> as <struct<(i32, i32)>>
  kgen.return %0 : !pop.union<struct<(i32, i32)>>
}

// CHECK-LABEL: @union_create_4
kgen.func @union_create_4(%arg0: !kgen.struct<(i32, i64, i32)>) -> !pop.union<struct<(i32, i64, i32)>, array<4, i64>> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<24 x i8>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.undef : !llvm.struct<(i64, array<24 x i8>)>
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  %[[VAL_4:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG:  %[[VAL_5:.*]] = llvm.mlir.constant(56 : i64) : i64
  // CHECK-DAG:  %[[VAL_6:.*]] = llvm.mlir.constant(48 : i64) : i64
  // CHECK-DAG:  %[[VAL_7:.*]] = llvm.mlir.constant(40 : i64) : i64
  // CHECK-DAG:  %[[VAL_8:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK-DAG:  %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:  %[[VAL_10:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG:  %[[VAL_11:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK:      %[[VAL_12:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i32, i64, i32)>
  // CHECK:      %[[VAL_13:.*]] = llvm.lshr %[[VAL_12]], %[[VAL_9]] : i32
  // CHECK:      %[[VAL_14:.*]] = llvm.zext %[[VAL_13]] : i32 to i64
  // CHECK:      %[[VAL_15:.*]] = llvm.shl %[[VAL_14]], %[[VAL_10]] : i64
  // CHECK:      %[[VAL_16:.*]] = llvm.or %[[VAL_10]], %[[VAL_15]] : i64
  // CHECK:      %[[VAL_17:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(i32, i64, i32)>
  // CHECK:      %[[VAL_18:.*]] = llvm.lshr %[[VAL_17]], %[[VAL_10]] : i64
  // CHECK:      %[[VAL_19:.*]] = llvm.trunc %[[VAL_18]] : i64 to i64
  // CHECK:      %[[VAL_20:.*]] = llvm.shl %[[VAL_19]], %[[VAL_8]] : i64
  // CHECK:      %[[VAL_21:.*]] = llvm.or %[[VAL_16]], %[[VAL_20]] : i64
  // CHECK:      %[[VAL_22:.*]] = llvm.lshr %[[VAL_17]], %[[VAL_8]] : i64
  // CHECK:      %[[VAL_23:.*]] = llvm.trunc %[[VAL_22]] : i64 to i8
  // CHECK:      %[[VAL_24:.*]] = llvm.shl %[[VAL_23]], %[[VAL_11]] : i8
  // CHECK:      %[[VAL_25:.*]] = llvm.or %[[VAL_11]], %[[VAL_24]] : i8
  // CHECK:      %[[VAL_26:.*]] = llvm.lshr %[[VAL_17]], %[[VAL_7]] : i64
  // CHECK:      %[[VAL_27:.*]] = llvm.trunc %[[VAL_26]] : i64 to i8
  // CHECK:      %[[VAL_28:.*]] = llvm.shl %[[VAL_27]], %[[VAL_11]] : i8
  // CHECK:      %[[VAL_29:.*]] = llvm.or %[[VAL_11]], %[[VAL_28]] : i8
  // CHECK:      %[[VAL_30:.*]] = llvm.lshr %[[VAL_17]], %[[VAL_6]] : i64
  // CHECK:      %[[VAL_31:.*]] = llvm.trunc %[[VAL_30]] : i64 to i8
  // CHECK:      %[[VAL_32:.*]] = llvm.shl %[[VAL_31]], %[[VAL_11]] : i8
  // CHECK:      %[[VAL_33:.*]] = llvm.or %[[VAL_11]], %[[VAL_32]] : i8
  // CHECK:      %[[VAL_34:.*]] = llvm.lshr %[[VAL_17]], %[[VAL_5]] : i64
  // CHECK:      %[[VAL_35:.*]] = llvm.trunc %[[VAL_34]] : i64 to i8
  // CHECK:      %[[VAL_36:.*]] = llvm.shl %[[VAL_35]], %[[VAL_11]] : i8
  // CHECK:      %[[VAL_37:.*]] = llvm.or %[[VAL_11]], %[[VAL_36]] : i8
  // CHECK:      %[[VAL_38:.*]] = llvm.extractvalue %{{.*}}[2] : !llvm.struct<(i32, i64, i32)>
  // CHECK:      %[[VAL_39:.*]] = llvm.lshr %[[VAL_38]], %[[VAL_9]] : i32
  // CHECK:      %[[VAL_40:.*]] = llvm.trunc %[[VAL_39]] : i32 to i8
  // CHECK:      %[[VAL_41:.*]] = llvm.shl %[[VAL_40]], %[[VAL_11]] : i8
  // CHECK:      %[[VAL_42:.*]] = llvm.or %[[VAL_11]], %[[VAL_41]] : i8
  // CHECK:      %[[VAL_43:.*]] = llvm.lshr %[[VAL_38]], %[[VAL_4]] : i32
  // CHECK:      %[[VAL_44:.*]] = llvm.trunc %[[VAL_43]] : i32 to i8
  // CHECK:      %[[VAL_45:.*]] = llvm.shl %[[VAL_44]], %[[VAL_11]] : i8
  // CHECK:      %[[VAL_46:.*]] = llvm.or %[[VAL_11]], %[[VAL_45]] : i8
  // CHECK:      %[[VAL_47:.*]] = llvm.lshr %[[VAL_38]], %[[VAL_3]] : i32
  // CHECK:      %[[VAL_48:.*]] = llvm.trunc %[[VAL_47]] : i32 to i8
  // CHECK:      %[[VAL_49:.*]] = llvm.shl %[[VAL_48]], %[[VAL_11]] : i8
  // CHECK:      %[[VAL_50:.*]] = llvm.or %[[VAL_11]], %[[VAL_49]] : i8
  // CHECK:      %[[VAL_51:.*]] = llvm.lshr %[[VAL_38]], %[[VAL_2]] : i32
  // CHECK:      %[[VAL_52:.*]] = llvm.trunc %[[VAL_51]] : i32 to i8
  // CHECK:      %[[VAL_53:.*]] = llvm.shl %[[VAL_52]], %[[VAL_11]] : i8
  // CHECK:      %[[VAL_54:.*]] = llvm.or %[[VAL_11]], %[[VAL_53]] : i8
  // CHECK:      %[[VAL_55:.*]] = llvm.insertvalue %[[VAL_21]], %[[VAL_1]][0] : !llvm.struct<(i64, array<24 x i8>)>
  // CHECK:      %[[VAL_56:.*]] = llvm.insertvalue %[[VAL_25]], %[[VAL_0]][0] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_57:.*]] = llvm.insertvalue %[[VAL_29]], %[[VAL_56]][1] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_58:.*]] = llvm.insertvalue %[[VAL_33]], %[[VAL_57]][2] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_59:.*]] = llvm.insertvalue %[[VAL_37]], %[[VAL_58]][3] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_60:.*]] = llvm.insertvalue %[[VAL_42]], %[[VAL_59]][4] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_61:.*]] = llvm.insertvalue %[[VAL_46]], %[[VAL_60]][5] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_62:.*]] = llvm.insertvalue %[[VAL_50]], %[[VAL_61]][6] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_63:.*]] = llvm.insertvalue %[[VAL_54]], %[[VAL_62]][7] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_64:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_63]][8] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_65:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_64]][9] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_66:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_65]][10] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_67:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_66]][11] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_68:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_67]][12] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_69:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_68]][13] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_70:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_69]][14] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_71:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_70]][15] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_72:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_71]][16] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_73:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_72]][17] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_74:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_73]][18] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_75:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_74]][19] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_76:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_75]][20] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_77:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_76]][21] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_78:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_77]][22] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_79:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_78]][23] : !llvm.array<24 x i8>
  // CHECK:      %[[VAL_80:.*]] = llvm.insertvalue %[[VAL_79]], %[[VAL_55]][1] : !llvm.struct<(i64, array<24 x i8>)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(i32, i64, i32)> as <struct<(i32, i64, i32)>, array<4, i64>>
  kgen.return %0 : !pop.union<struct<(i32, i64, i32)>, array<4, i64>>
}

// CHECK-LABEL: @union_create_5
kgen.func @union_create_5(%arg0: !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>) -> !pop.union<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : !llvm.array<16 x i8>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.undef : vector<2xf32>
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.undef : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG:  %[[VAL_4:.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK-DAG:  %[[VAL_5:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG:  %[[VAL_6:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  %[[VAL_7:.*]] = llvm.mlir.constant(0 : i16) : i16
  // CHECK-DAG:  %[[VAL_8:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:  %[[VAL_9:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK:      %[[VAL_10:.*]] = llvm.extractvalue %{{.*}}[0, 0] : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK:      %[[VAL_11:.*]] = llvm.lshr %[[VAL_10]], %[[VAL_7]] : i16
  // CHECK:      %[[VAL_12:.*]] = llvm.zext %[[VAL_11]] : i16 to i32
  // CHECK:      %[[VAL_13:.*]] = llvm.shl %[[VAL_12]], %[[VAL_8]] : i32
  // CHECK:      %[[VAL_14:.*]] = llvm.or %[[VAL_8]], %[[VAL_13]] : i32
  // CHECK:      %[[VAL_15:.*]] = llvm.extractvalue %{{.*}}[0, 1] : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK:      %[[VAL_16:.*]] = llvm.lshr %[[VAL_15]], %[[VAL_7]] : i16
  // CHECK:      %[[VAL_17:.*]] = llvm.zext %[[VAL_16]] : i16 to i32
  // CHECK:      %[[VAL_18:.*]] = llvm.shl %[[VAL_17]], %[[VAL_6]] : i32
  // CHECK:      %[[VAL_19:.*]] = llvm.or %[[VAL_14]], %[[VAL_18]] : i32
  // CHECK:      %[[VAL_20:.*]] = llvm.extractvalue %{{.*}}[1, 0, 0] : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK:      %[[VAL_21:.*]] = llvm.lshr %[[VAL_20]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_22:.*]] = llvm.trunc %[[VAL_21]] : i8 to i8
  // CHECK:      %[[VAL_23:.*]] = llvm.shl %[[VAL_22]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_24:.*]] = llvm.or %[[VAL_9]], %[[VAL_23]] : i8
  // CHECK:      %[[VAL_25:.*]] = llvm.extractvalue %{{.*}}[1, 0, 1] : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK:      %[[VAL_26:.*]] = llvm.lshr %[[VAL_25]], %[[VAL_8]] : i32
  // CHECK:      %[[VAL_27:.*]] = llvm.trunc %[[VAL_26]] : i32 to i8
  // CHECK:      %[[VAL_28:.*]] = llvm.shl %[[VAL_27]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_29:.*]] = llvm.or %[[VAL_9]], %[[VAL_28]] : i8
  // CHECK:      %[[VAL_30:.*]] = llvm.lshr %[[VAL_25]], %[[VAL_5]] : i32
  // CHECK:      %[[VAL_31:.*]] = llvm.trunc %[[VAL_30]] : i32 to i8
  // CHECK:      %[[VAL_32:.*]] = llvm.shl %[[VAL_31]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_33:.*]] = llvm.or %[[VAL_9]], %[[VAL_32]] : i8
  // CHECK:      %[[VAL_34:.*]] = llvm.lshr %[[VAL_25]], %[[VAL_6]] : i32
  // CHECK:      %[[VAL_35:.*]] = llvm.trunc %[[VAL_34]] : i32 to i8
  // CHECK:      %[[VAL_36:.*]] = llvm.shl %[[VAL_35]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_37:.*]] = llvm.or %[[VAL_9]], %[[VAL_36]] : i8
  // CHECK:      %[[VAL_38:.*]] = llvm.lshr %[[VAL_25]], %[[VAL_4]] : i32
  // CHECK:      %[[VAL_39:.*]] = llvm.trunc %[[VAL_38]] : i32 to i8
  // CHECK:      %[[VAL_40:.*]] = llvm.shl %[[VAL_39]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_41:.*]] = llvm.or %[[VAL_9]], %[[VAL_40]] : i8
  // CHECK:      %[[VAL_42:.*]] = llvm.extractvalue %{{.*}}[1, 1] : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK:      %[[VAL_43:.*]] = llvm.extractelement %[[VAL_42]]{{\[}}%[[VAL_8]] : i32] : vector<2xf32>
  // CHECK:      %[[VAL_44:.*]] = llvm.bitcast %[[VAL_43]] : f32 to i32
  // CHECK:      %[[VAL_45:.*]] = llvm.lshr %[[VAL_44]], %[[VAL_8]] : i32
  // CHECK:      %[[VAL_46:.*]] = llvm.trunc %[[VAL_45]] : i32 to i8
  // CHECK:      %[[VAL_47:.*]] = llvm.shl %[[VAL_46]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_48:.*]] = llvm.or %[[VAL_9]], %[[VAL_47]] : i8
  // CHECK:      %[[VAL_49:.*]] = llvm.lshr %[[VAL_44]], %[[VAL_5]] : i32
  // CHECK:      %[[VAL_50:.*]] = llvm.trunc %[[VAL_49]] : i32 to i8
  // CHECK:      %[[VAL_51:.*]] = llvm.shl %[[VAL_50]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_52:.*]] = llvm.or %[[VAL_9]], %[[VAL_51]] : i8
  // CHECK:      %[[VAL_53:.*]] = llvm.lshr %[[VAL_44]], %[[VAL_6]] : i32
  // CHECK:      %[[VAL_54:.*]] = llvm.trunc %[[VAL_53]] : i32 to i8
  // CHECK:      %[[VAL_55:.*]] = llvm.shl %[[VAL_54]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_56:.*]] = llvm.or %[[VAL_9]], %[[VAL_55]] : i8
  // CHECK:      %[[VAL_57:.*]] = llvm.lshr %[[VAL_44]], %[[VAL_4]] : i32
  // CHECK:      %[[VAL_58:.*]] = llvm.trunc %[[VAL_57]] : i32 to i8
  // CHECK:      %[[VAL_59:.*]] = llvm.shl %[[VAL_58]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_60:.*]] = llvm.or %[[VAL_9]], %[[VAL_59]] : i8
  // CHECK:      %[[VAL_61:.*]] = llvm.extractelement %[[VAL_42]]{{\[}}%[[VAL_3]] : i32] : vector<2xf32>
  // CHECK:      %[[VAL_62:.*]] = llvm.bitcast %[[VAL_61]] : f32 to i32
  // CHECK:      %[[VAL_63:.*]] = llvm.lshr %[[VAL_62]], %[[VAL_8]] : i32
  // CHECK:      %[[VAL_64:.*]] = llvm.trunc %[[VAL_63]] : i32 to i8
  // CHECK:      %[[VAL_65:.*]] = llvm.shl %[[VAL_64]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_66:.*]] = llvm.or %[[VAL_9]], %[[VAL_65]] : i8
  // CHECK:      %[[VAL_67:.*]] = llvm.lshr %[[VAL_62]], %[[VAL_5]] : i32
  // CHECK:      %[[VAL_68:.*]] = llvm.trunc %[[VAL_67]] : i32 to i8
  // CHECK:      %[[VAL_69:.*]] = llvm.shl %[[VAL_68]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_70:.*]] = llvm.or %[[VAL_9]], %[[VAL_69]] : i8
  // CHECK:      %[[VAL_71:.*]] = llvm.lshr %[[VAL_62]], %[[VAL_6]] : i32
  // CHECK:      %[[VAL_72:.*]] = llvm.trunc %[[VAL_71]] : i32 to i8
  // CHECK:      %[[VAL_73:.*]] = llvm.shl %[[VAL_72]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_74:.*]] = llvm.or %[[VAL_9]], %[[VAL_73]] : i8
  // CHECK:      %[[VAL_75:.*]] = llvm.lshr %[[VAL_62]], %[[VAL_4]] : i32
  // CHECK:      %[[VAL_76:.*]] = llvm.trunc %[[VAL_75]] : i32 to i8
  // CHECK:      %[[VAL_77:.*]] = llvm.shl %[[VAL_76]], %[[VAL_9]] : i8
  // CHECK:      %[[VAL_78:.*]] = llvm.or %[[VAL_9]], %[[VAL_77]] : i8
  // CHECK:      %[[VAL_79:.*]] = llvm.bitcast %[[VAL_19]] : i32 to f32
  // CHECK:      %[[VAL_80:.*]] = llvm.insertelement %[[VAL_79]], %[[VAL_1]]{{\[}}%[[VAL_8]] : i32] : vector<2xf32>
  // CHECK:      %[[VAL_81:.*]] = llvm.bitcast %[[VAL_8]] : i32 to f32
  // CHECK:      %[[VAL_82:.*]] = llvm.insertelement %[[VAL_81]], %[[VAL_80]]{{\[}}%[[VAL_3]] : i32] : vector<2xf32>
  // CHECK:      %[[VAL_83:.*]] = llvm.insertvalue %[[VAL_82]], %[[VAL_2]][0] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_84:.*]] = llvm.insertvalue %[[VAL_24]], %[[VAL_0]][0] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_85:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_84]][1] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_86:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_85]][2] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_87:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_86]][3] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_88:.*]] = llvm.insertvalue %[[VAL_29]], %[[VAL_87]][4] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_89:.*]] = llvm.insertvalue %[[VAL_33]], %[[VAL_88]][5] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_90:.*]] = llvm.insertvalue %[[VAL_37]], %[[VAL_89]][6] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_91:.*]] = llvm.insertvalue %[[VAL_41]], %[[VAL_90]][7] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_92:.*]] = llvm.insertvalue %[[VAL_48]], %[[VAL_91]][8] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_93:.*]] = llvm.insertvalue %[[VAL_52]], %[[VAL_92]][9] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_94:.*]] = llvm.insertvalue %[[VAL_56]], %[[VAL_93]][10] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_95:.*]] = llvm.insertvalue %[[VAL_60]], %[[VAL_94]][11] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_96:.*]] = llvm.insertvalue %[[VAL_66]], %[[VAL_95]][12] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_97:.*]] = llvm.insertvalue %[[VAL_70]], %[[VAL_96]][13] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_98:.*]] = llvm.insertvalue %[[VAL_74]], %[[VAL_97]][14] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_99:.*]] = llvm.insertvalue %[[VAL_78]], %[[VAL_98]][15] : !llvm.array<16 x i8>
  // CHECK:      %[[VAL_100:.*]] = llvm.insertvalue %[[VAL_99]], %[[VAL_83]][1] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  %0 = pop.union.wrap %arg0 : !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)> as <struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>
  kgen.return %0 : !pop.union<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>
}

// CHECK-LABEL: @union_create_6
kgen.func @union_create_6(%arg0: !kgen.pointer<index>) -> !pop.union<pointer<index>> {
    // CHECK: llvm.ptrtoint
  %0 = pop.union.wrap %arg0 : !kgen.pointer<index> as <pointer<index>>
  kgen.return %0 : !pop.union<pointer<index>>
}

// CHECK-LABEL: @union_get_0
kgen.func @union_get_0(%arg0: !pop.union<i32>) ->  i32{
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:      %[[VAL_1:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i32)>
  // CHECK:      %[[VAL_2:.*]] = llvm.lshr %[[VAL_1]], %[[VAL_0]] : i32
  // CHECK:      %[[VAL_3:.*]] = llvm.trunc %[[VAL_2]] : i32 to i32
  // CHECK:      %[[VAL_4:.*]] = llvm.shl %[[VAL_3]], %[[VAL_0]] : i32
  // CHECK:      %[[VAL_5:.*]] = llvm.or %[[VAL_0]], %[[VAL_4]] : i32
  %0 = pop.union.unwrap %arg0 : <i32> as i32
  kgen.return %0 : i32
}

// CHECK-LABEL: @union_get_1
kgen.func @union_get_1(%arg0: !pop.union<f64>) -> f64 {
    // CHECK: llvm.bitcast %{{.*}} : i64 to f64
  %0 = pop.union.unwrap %arg0 : <f64> as f64
  kgen.return %0 : f64
}

// CHECK-LABEL: @union_get_2
kgen.func @union_get_2(%arg0: !pop.union<struct<(i32, i32)>>) -> !kgen.struct<(i32, i32)>{
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-DAG:  %[[VAL_4:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:  %[[VAL_5:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32)>
  // CHECK:      %[[VAL_6:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i32, array<4 x i8>)>
  // CHECK:      %[[VAL_7:.*]] = llvm.extractvalue %{{.*}}[1, 0] : !llvm.struct<(i32, array<4 x i8>)>
  // CHECK:      %[[VAL_8:.*]] = llvm.extractvalue %{{.*}}[1, 1] : !llvm.struct<(i32, array<4 x i8>)>
  // CHECK:      %[[VAL_9:.*]] = llvm.extractvalue %{{.*}}[1, 2] : !llvm.struct<(i32, array<4 x i8>)>
  // CHECK:      %[[VAL_10:.*]] = llvm.extractvalue %{{.*}}[1, 3] : !llvm.struct<(i32, array<4 x i8>)>
  // CHECK:      %[[VAL_11:.*]] = llvm.lshr %[[VAL_6]], %[[VAL_4]] : i32
  // CHECK:      %[[VAL_12:.*]] = llvm.trunc %[[VAL_11]] : i32 to i32
  // CHECK:      %[[VAL_13:.*]] = llvm.shl %[[VAL_12]], %[[VAL_4]] : i32
  // CHECK:      %[[VAL_14:.*]] = llvm.or %[[VAL_4]], %[[VAL_13]] : i32
  // CHECK:      %[[VAL_15:.*]] = llvm.insertvalue %[[VAL_14]], %[[VAL_5]][0] : !llvm.struct<(i32, i32)>
  // CHECK:      %[[VAL_16:.*]] = llvm.lshr %[[VAL_7]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_17:.*]] = llvm.zext %[[VAL_16]] : i8 to i32
  // CHECK:      %[[VAL_18:.*]] = llvm.shl %[[VAL_17]], %[[VAL_4]] : i32
  // CHECK:      %[[VAL_19:.*]] = llvm.or %[[VAL_4]], %[[VAL_18]] : i32
  // CHECK:      %[[VAL_20:.*]] = llvm.lshr %[[VAL_8]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_21:.*]] = llvm.zext %[[VAL_20]] : i8 to i32
  // CHECK:      %[[VAL_22:.*]] = llvm.shl %[[VAL_21]], %[[VAL_2]] : i32
  // CHECK:      %[[VAL_23:.*]] = llvm.or %[[VAL_19]], %[[VAL_22]] : i32
  // CHECK:      %[[VAL_24:.*]] = llvm.lshr %[[VAL_9]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_25:.*]] = llvm.zext %[[VAL_24]] : i8 to i32
  // CHECK:      %[[VAL_26:.*]] = llvm.shl %[[VAL_25]], %[[VAL_1]] : i32
  // CHECK:      %[[VAL_27:.*]] = llvm.or %[[VAL_23]], %[[VAL_26]] : i32
  // CHECK:      %[[VAL_28:.*]] = llvm.lshr %[[VAL_10]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_29:.*]] = llvm.zext %[[VAL_28]] : i8 to i32
  // CHECK:      %[[VAL_30:.*]] = llvm.shl %[[VAL_29]], %[[VAL_0]] : i32
  // CHECK:      %[[VAL_31:.*]] = llvm.or %[[VAL_27]], %[[VAL_30]] : i32
  // CHECK:      %[[VAL_32:.*]] = llvm.insertvalue %[[VAL_31]], %[[VAL_15]][1] : !llvm.struct<(i32, i32)>
  %0 = pop.union.unwrap %arg0 : <struct<(i32, i32)>> as !kgen.struct<(i32, i32)>
  kgen.return %0 : !kgen.struct<(i32, i32)>
}

// CHECK-LABEL: @union_get_3
kgen.func @union_get_3(%arg0: !pop.union<struct<(i32, i64, i32)>, array<4, i64>>) -> !kgen.struct<(i32, i64, i32)> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(56 : i64) : i64
  // CHECK-DAG:  %[[VAL_4:.*]] = llvm.mlir.constant(48 : i64) : i64
  // CHECK-DAG:  %[[VAL_5:.*]] = llvm.mlir.constant(40 : i64) : i64
  // CHECK-DAG:  %[[VAL_6:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-DAG:  %[[VAL_7:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK-DAG:  %[[VAL_8:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG:  %[[VAL_9:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG:  %[[VAL_10:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i64, i32)>
  // CHECK:      %[[VAL_11:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i64, array<24 x i8>)>
  // CHECK:      %[[VAL_12:.*]] = llvm.extractvalue %{{.*}}[1, 0] : !llvm.struct<(i64, array<24 x i8>)>
  // CHECK:      %[[VAL_13:.*]] = llvm.extractvalue %{{.*}}[1, 1] : !llvm.struct<(i64, array<24 x i8>)>
  // CHECK:      %[[VAL_14:.*]] = llvm.extractvalue %{{.*}}[1, 2] : !llvm.struct<(i64, array<24 x i8>)>
  // CHECK:      %[[VAL_15:.*]] = llvm.extractvalue %{{.*}}[1, 3] : !llvm.struct<(i64, array<24 x i8>)>
  // CHECK:      %[[VAL_16:.*]] = llvm.extractvalue %{{.*}}[1, 4] : !llvm.struct<(i64, array<24 x i8>)>
  // CHECK:      %[[VAL_17:.*]] = llvm.extractvalue %{{.*}}[1, 5] : !llvm.struct<(i64, array<24 x i8>)>
  // CHECK:      %[[VAL_18:.*]] = llvm.extractvalue %{{.*}}[1, 6] : !llvm.struct<(i64, array<24 x i8>)>
  // CHECK:      %[[VAL_19:.*]] = llvm.extractvalue %{{.*}}[1, 7] : !llvm.struct<(i64, array<24 x i8>)>
  // CHECK:      %[[VAL_20:.*]] = llvm.lshr %[[VAL_11]], %[[VAL_8]] : i64
  // CHECK:      %[[VAL_21:.*]] = llvm.trunc %[[VAL_20]] : i64 to i32
  // CHECK:      %[[VAL_22:.*]] = llvm.shl %[[VAL_21]], %[[VAL_9]] : i32
  // CHECK:      %[[VAL_23:.*]] = llvm.or %[[VAL_9]], %[[VAL_22]] : i32
  // CHECK:      %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_10]][0] : !llvm.struct<(i32, i64, i32)>
  // CHECK:      %[[VAL_25:.*]] = llvm.lshr %[[VAL_11]], %[[VAL_7]] : i64
  // CHECK:      %[[VAL_26:.*]] = llvm.trunc %[[VAL_25]] : i64 to i64
  // CHECK:      %[[VAL_27:.*]] = llvm.shl %[[VAL_26]], %[[VAL_8]] : i64
  // CHECK:      %[[VAL_28:.*]] = llvm.or %[[VAL_8]], %[[VAL_27]] : i64
  // CHECK:      %[[VAL_29:.*]] = llvm.lshr %[[VAL_12]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_30:.*]] = llvm.zext %[[VAL_29]] : i8 to i64
  // CHECK:      %[[VAL_31:.*]] = llvm.shl %[[VAL_30]], %[[VAL_7]] : i64
  // CHECK:      %[[VAL_32:.*]] = llvm.or %[[VAL_28]], %[[VAL_31]] : i64
  // CHECK:      %[[VAL_33:.*]] = llvm.lshr %[[VAL_13]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_34:.*]] = llvm.zext %[[VAL_33]] : i8 to i64
  // CHECK:      %[[VAL_35:.*]] = llvm.shl %[[VAL_34]], %[[VAL_5]] : i64
  // CHECK:      %[[VAL_36:.*]] = llvm.or %[[VAL_32]], %[[VAL_35]] : i64
  // CHECK:      %[[VAL_37:.*]] = llvm.lshr %[[VAL_14]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_38:.*]] = llvm.zext %[[VAL_37]] : i8 to i64
  // CHECK:      %[[VAL_39:.*]] = llvm.shl %[[VAL_38]], %[[VAL_4]] : i64
  // CHECK:      %[[VAL_40:.*]] = llvm.or %[[VAL_36]], %[[VAL_39]] : i64
  // CHECK:      %[[VAL_41:.*]] = llvm.lshr %[[VAL_15]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_42:.*]] = llvm.zext %[[VAL_41]] : i8 to i64
  // CHECK:      %[[VAL_43:.*]] = llvm.shl %[[VAL_42]], %[[VAL_3]] : i64
  // CHECK:      %[[VAL_44:.*]] = llvm.or %[[VAL_40]], %[[VAL_43]] : i64
  // CHECK:      %[[VAL_45:.*]] = llvm.insertvalue %[[VAL_44]], %[[VAL_24]][1] : !llvm.struct<(i32, i64, i32)>
  // CHECK:      %[[VAL_46:.*]] = llvm.lshr %[[VAL_16]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_47:.*]] = llvm.zext %[[VAL_46]] : i8 to i32
  // CHECK:      %[[VAL_48:.*]] = llvm.shl %[[VAL_47]], %[[VAL_9]] : i32
  // CHECK:      %[[VAL_49:.*]] = llvm.or %[[VAL_9]], %[[VAL_48]] : i32
  // CHECK:      %[[VAL_50:.*]] = llvm.lshr %[[VAL_17]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_51:.*]] = llvm.zext %[[VAL_50]] : i8 to i32
  // CHECK:      %[[VAL_52:.*]] = llvm.shl %[[VAL_51]], %[[VAL_2]] : i32
  // CHECK:      %[[VAL_53:.*]] = llvm.or %[[VAL_49]], %[[VAL_52]] : i32
  // CHECK:      %[[VAL_54:.*]] = llvm.lshr %[[VAL_18]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_55:.*]] = llvm.zext %[[VAL_54]] : i8 to i32
  // CHECK:      %[[VAL_56:.*]] = llvm.shl %[[VAL_55]], %[[VAL_1]] : i32
  // CHECK:      %[[VAL_57:.*]] = llvm.or %[[VAL_53]], %[[VAL_56]] : i32
  // CHECK:      %[[VAL_58:.*]] = llvm.lshr %[[VAL_19]], %[[VAL_6]] : i8
  // CHECK:      %[[VAL_59:.*]] = llvm.zext %[[VAL_58]] : i8 to i32
  // CHECK:      %[[VAL_60:.*]] = llvm.shl %[[VAL_59]], %[[VAL_0]] : i32
  // CHECK:      %[[VAL_61:.*]] = llvm.or %[[VAL_57]], %[[VAL_60]] : i32
  // CHECK:      %[[VAL_62:.*]] = llvm.insertvalue %[[VAL_61]], %[[VAL_45]][2] : !llvm.struct<(i32, i64, i32)>
  %0 = pop.union.unwrap %arg0 : <struct<(i32, i64, i32)>, array<4, i64>> as !kgen.struct<(i32, i64, i32)>
  kgen.return %0 : !kgen.struct<(i32, i64, i32)>
}

// CHECK-LABEL: @union_get_4
kgen.func @union_get_4(%arg0: !pop.union<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>) -> !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)> {
  // CHECK-DAG:  %[[VAL_0:.*]] = llvm.mlir.undef : vector<2xf32>
  // CHECK-DAG:  %[[VAL_1:.*]] = llvm.mlir.constant(24 : i32) : i32
  // CHECK-DAG:  %[[VAL_2:.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK-DAG:  %[[VAL_3:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-DAG:  %[[VAL_4:.*]] = llvm.mlir.undef : !llvm.struct<(i8, i32)>
  // CHECK-DAG:  %[[VAL_5:.*]] = llvm.mlir.undef : !llvm.struct<(struct<(i8, i32)>, vector<2xf32>)>
  // CHECK-DAG:  %[[VAL_6:.*]] = llvm.mlir.constant(16 : i32) : i32
  // CHECK-DAG:  %[[VAL_7:.*]] = llvm.mlir.constant(0 : i16) : i16
  // CHECK-DAG:  %[[VAL_8:.*]] = llvm.mlir.undef : !llvm.array<2 x i16>
  // CHECK-DAG:  %[[VAL_9:.*]] = llvm.mlir.undef : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK-DAG:  %[[VAL_10:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG:  %[[VAL_11:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK:      %[[VAL_12:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_13:.*]] = llvm.extractelement %[[VAL_12]]{{\[}}%[[VAL_11]] : i32] : vector<2xf32>
  // CHECK:      %[[VAL_14:.*]] = llvm.bitcast %[[VAL_13]] : f32 to i32
  // CHECK:      %[[VAL_15:.*]] = llvm.extractvalue %{{.*}}[1, 0] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_16:.*]] = llvm.extractvalue %{{.*}}[1, 4] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_17:.*]] = llvm.extractvalue %{{.*}}[1, 5] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_18:.*]] = llvm.extractvalue %{{.*}}[1, 6] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_19:.*]] = llvm.extractvalue %{{.*}}[1, 7] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_20:.*]] = llvm.extractvalue %{{.*}}[1, 8] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_21:.*]] = llvm.extractvalue %{{.*}}[1, 9] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_22:.*]] = llvm.extractvalue %{{.*}}[1, 10] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_23:.*]] = llvm.extractvalue %{{.*}}[1, 11] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_24:.*]] = llvm.extractvalue %{{.*}}[1, 12] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_25:.*]] = llvm.extractvalue %{{.*}}[1, 13] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_26:.*]] = llvm.extractvalue %{{.*}}[1, 14] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_27:.*]] = llvm.extractvalue %{{.*}}[1, 15] : !llvm.struct<(vector<2xf32>, array<16 x i8>)>
  // CHECK:      %[[VAL_28:.*]] = llvm.lshr %[[VAL_14]], %[[VAL_11]] : i32
  // CHECK:      %[[VAL_29:.*]] = llvm.trunc %[[VAL_28]] : i32 to i16
  // CHECK:      %[[VAL_30:.*]] = llvm.shl %[[VAL_29]], %[[VAL_7]] : i16
  // CHECK:      %[[VAL_31:.*]] = llvm.or %[[VAL_7]], %[[VAL_30]] : i16
  // CHECK:      %[[VAL_32:.*]] = llvm.insertvalue %[[VAL_31]], %[[VAL_8]][0] : !llvm.array<2 x i16>
  // CHECK:      %[[VAL_33:.*]] = llvm.lshr %[[VAL_14]], %[[VAL_6]] : i32
  // CHECK:      %[[VAL_34:.*]] = llvm.trunc %[[VAL_33]] : i32 to i16
  // CHECK:      %[[VAL_35:.*]] = llvm.shl %[[VAL_34]], %[[VAL_7]] : i16
  // CHECK:      %[[VAL_36:.*]] = llvm.or %[[VAL_7]], %[[VAL_35]] : i16
  // CHECK:      %[[VAL_37:.*]] = llvm.insertvalue %[[VAL_36]], %[[VAL_32]][1] : !llvm.array<2 x i16>
  // CHECK:      %[[VAL_38:.*]] = llvm.insertvalue %[[VAL_37]], %[[VAL_9]][0] : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK:      %[[VAL_39:.*]] = llvm.lshr %[[VAL_15]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_40:.*]] = llvm.trunc %[[VAL_39]] : i8 to i8
  // CHECK:      %[[VAL_41:.*]] = llvm.shl %[[VAL_40]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_42:.*]] = llvm.or %[[VAL_3]], %[[VAL_41]] : i8
  // CHECK:      %[[VAL_43:.*]] = llvm.insertvalue %[[VAL_42]], %[[VAL_4]][0] : !llvm.struct<(i8, i32)>
  // CHECK:      %[[VAL_44:.*]] = llvm.lshr %[[VAL_16]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_45:.*]] = llvm.zext %[[VAL_44]] : i8 to i32
  // CHECK:      %[[VAL_46:.*]] = llvm.shl %[[VAL_45]], %[[VAL_11]] : i32
  // CHECK:      %[[VAL_47:.*]] = llvm.or %[[VAL_11]], %[[VAL_46]] : i32
  // CHECK:      %[[VAL_48:.*]] = llvm.lshr %[[VAL_17]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_49:.*]] = llvm.zext %[[VAL_48]] : i8 to i32
  // CHECK:      %[[VAL_50:.*]] = llvm.shl %[[VAL_49]], %[[VAL_2]] : i32
  // CHECK:      %[[VAL_51:.*]] = llvm.or %[[VAL_47]], %[[VAL_50]] : i32
  // CHECK:      %[[VAL_52:.*]] = llvm.lshr %[[VAL_18]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_53:.*]] = llvm.zext %[[VAL_52]] : i8 to i32
  // CHECK:      %[[VAL_54:.*]] = llvm.shl %[[VAL_53]], %[[VAL_6]] : i32
  // CHECK:      %[[VAL_55:.*]] = llvm.or %[[VAL_51]], %[[VAL_54]] : i32
  // CHECK:      %[[VAL_56:.*]] = llvm.lshr %[[VAL_19]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_57:.*]] = llvm.zext %[[VAL_56]] : i8 to i32
  // CHECK:      %[[VAL_58:.*]] = llvm.shl %[[VAL_57]], %[[VAL_1]] : i32
  // CHECK:      %[[VAL_59:.*]] = llvm.or %[[VAL_55]], %[[VAL_58]] : i32
  // CHECK:      %[[VAL_60:.*]] = llvm.insertvalue %[[VAL_59]], %[[VAL_43]][1] : !llvm.struct<(i8, i32)>
  // CHECK:      %[[VAL_61:.*]] = llvm.insertvalue %[[VAL_60]], %[[VAL_5]][0] : !llvm.struct<(struct<(i8, i32)>, vector<2xf32>)>
  // CHECK:      %[[VAL_62:.*]] = llvm.lshr %[[VAL_20]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_63:.*]] = llvm.zext %[[VAL_62]] : i8 to i32
  // CHECK:      %[[VAL_64:.*]] = llvm.shl %[[VAL_63]], %[[VAL_11]] : i32
  // CHECK:      %[[VAL_65:.*]] = llvm.or %[[VAL_11]], %[[VAL_64]] : i32
  // CHECK:      %[[VAL_66:.*]] = llvm.lshr %[[VAL_21]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_67:.*]] = llvm.zext %[[VAL_66]] : i8 to i32
  // CHECK:      %[[VAL_68:.*]] = llvm.shl %[[VAL_67]], %[[VAL_2]] : i32
  // CHECK:      %[[VAL_69:.*]] = llvm.or %[[VAL_65]], %[[VAL_68]] : i32
  // CHECK:      %[[VAL_70:.*]] = llvm.lshr %[[VAL_22]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_71:.*]] = llvm.zext %[[VAL_70]] : i8 to i32
  // CHECK:      %[[VAL_72:.*]] = llvm.shl %[[VAL_71]], %[[VAL_6]] : i32
  // CHECK:      %[[VAL_73:.*]] = llvm.or %[[VAL_69]], %[[VAL_72]] : i32
  // CHECK:      %[[VAL_74:.*]] = llvm.lshr %[[VAL_23]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_75:.*]] = llvm.zext %[[VAL_74]] : i8 to i32
  // CHECK:      %[[VAL_76:.*]] = llvm.shl %[[VAL_75]], %[[VAL_1]] : i32
  // CHECK:      %[[VAL_77:.*]] = llvm.or %[[VAL_73]], %[[VAL_76]] : i32
  // CHECK:      %[[VAL_78:.*]] = llvm.bitcast %[[VAL_77]] : i32 to f32
  // CHECK:      %[[VAL_79:.*]] = llvm.insertelement %[[VAL_78]], %[[VAL_0]]{{\[}}%[[VAL_11]] : i32] : vector<2xf32>
  // CHECK:      %[[VAL_80:.*]] = llvm.lshr %[[VAL_24]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_81:.*]] = llvm.zext %[[VAL_80]] : i8 to i32
  // CHECK:      %[[VAL_82:.*]] = llvm.shl %[[VAL_81]], %[[VAL_11]] : i32
  // CHECK:      %[[VAL_83:.*]] = llvm.or %[[VAL_11]], %[[VAL_82]] : i32
  // CHECK:      %[[VAL_84:.*]] = llvm.lshr %[[VAL_25]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_85:.*]] = llvm.zext %[[VAL_84]] : i8 to i32
  // CHECK:      %[[VAL_86:.*]] = llvm.shl %[[VAL_85]], %[[VAL_2]] : i32
  // CHECK:      %[[VAL_87:.*]] = llvm.or %[[VAL_83]], %[[VAL_86]] : i32
  // CHECK:      %[[VAL_88:.*]] = llvm.lshr %[[VAL_26]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_89:.*]] = llvm.zext %[[VAL_88]] : i8 to i32
  // CHECK:      %[[VAL_90:.*]] = llvm.shl %[[VAL_89]], %[[VAL_6]] : i32
  // CHECK:      %[[VAL_91:.*]] = llvm.or %[[VAL_87]], %[[VAL_90]] : i32
  // CHECK:      %[[VAL_92:.*]] = llvm.lshr %[[VAL_27]], %[[VAL_3]] : i8
  // CHECK:      %[[VAL_93:.*]] = llvm.zext %[[VAL_92]] : i8 to i32
  // CHECK:      %[[VAL_94:.*]] = llvm.shl %[[VAL_93]], %[[VAL_1]] : i32
  // CHECK:      %[[VAL_95:.*]] = llvm.or %[[VAL_91]], %[[VAL_94]] : i32
  // CHECK:      %[[VAL_96:.*]] = llvm.bitcast %[[VAL_95]] : i32 to f32
  // CHECK:      %[[VAL_97:.*]] = llvm.insertelement %[[VAL_96]], %[[VAL_79]]{{\[}}%[[VAL_10]] : i32] : vector<2xf32>
  // CHECK:      %[[VAL_98:.*]] = llvm.insertvalue %[[VAL_97]], %[[VAL_61]][1] :
  // CHECK:      %[[VAL_99:.*]] = llvm.insertvalue %[[VAL_98]], %[[VAL_38]][1] :
  %0 = pop.union.unwrap %arg0 : <struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>> as !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>
  kgen.return %0 : !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>
}

// CHECK-LABEL: @union_get_5
kgen.func @union_get_5(%arg0: !pop.union<pointer<index>>) -> !kgen.pointer<index> {
    // CHECK: llvm.inttoptr
  %0 = pop.union.unwrap %arg0 : <pointer<index>> as !kgen.pointer<index>
  kgen.return %0 : !kgen.pointer<index>
}

// CHECK-LABEL: @unpack_pointer
kgen.func @unpack_pointer(%arg0: !pop.union<pointer<i8>>) -> !kgen.pointer<i8> {
    // CHECK: trunc %{{.*}} : i64 to i64
    // CHECK: inttoptr %{{.*}} : i64 to !llvm.ptr
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
