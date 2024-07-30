// RUN: kgen-opt %s -lower-kgen-to-llvm -canonicalize | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @variant_create_0
kgen.func @variant_create_0(%arg0: i32) -> !kgen.variant<i32> {
  // CHECK-DAG: %[[DISCR:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-DAG: %[[S0:.*]] = llvm.mlir.undef : !llvm.array<1 x i64>
  // CHECK-DAG: %[[S2:.*]] = llvm.mlir.undef : !llvm.struct<(array<1 x i64>, i8)>
  // CHECK-DAG: %[[I32_0:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: %[[I64_0:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK: %[[P0:.*]] = llvm.lshr %arg0, %[[I32_0]]
  // CHECK: %[[P1:.*]] = llvm.zext %[[P0]] : i32 to i64
  // CHECK: %[[P2:.*]] = llvm.shl %[[P1]], %[[I64_0]]
  // CHECK: %[[P3:.*]] = llvm.or %[[I64_0]], %[[P2]]
  // CHECK: %[[S1:.*]] = llvm.insertvalue %[[P3]], %[[S0]][0]
  // CHECK: %[[S3:.*]] = llvm.insertvalue %[[S1]], %[[S2]][0]
  // CHECK: %[[S4:.*]] = llvm.insertvalue %[[DISCR]], %[[S3]][1]
  %0 = kgen.variant.create %arg0, 0 : <i32>
  kgen.return %0 : !kgen.variant<i32>
}

// CHECK-LABEL: @variant_create_1
kgen.func @variant_create_1(%arg0: i8) -> !kgen.variant<i8> {
  // CHECK-DAG: %[[I64_0:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK-DAG: %[[I8_0:.*]] = llvm.mlir.constant(0 : i8)
  // CHECK: %[[P3:.*]] = llvm.lshr %arg0, %[[I8_0]] : i8
  // CHECK: %[[P4:.*]] = llvm.zext %[[P3]] : i8 to i64
  // CHECK: %[[P5:.*]] = llvm.shl %[[P4]], %[[I64_0]] : i64
  // CHECK: %[[P6:.*]] = llvm.or %[[I64_0]], %[[P5]] : i64
  // CHECK: llvm.insertvalue %[[P6]], %{{.*}}[0] : !llvm.array<1 x i64>
  %0 = kgen.variant.create %arg0, 0 : <i8>
  kgen.return %0 : !kgen.variant<i8>
}

// CHECK-LABEL: @variant_create_2
kgen.func @variant_create_2(%arg0: f64) -> !kgen.variant<f64> {
  // CHECK: %[[P2:.*]] = llvm.bitcast %arg0 : f64 to i64
  // CHECK: %[[P3:.*]] = llvm.lshr %[[P2]], %{{.*}} : i64
  // CHECK: %[[P4:.*]] = llvm.trunc %[[P3]] : i64 to i64
  // CHECK: %[[P5:.*]] = llvm.shl %[[P4]], %{{.*}} : i64
  %0 = kgen.variant.create %arg0, 0 : <f64>
  kgen.return %0 : !kgen.variant<f64>
}

// CHECK-LABEL: @variant_create_3
kgen.func @variant_create_3(%arg0: !kgen.struct<(i32, i32)>) -> !kgen.variant<struct<(i32, i32)>> {
  // CHECK: %[[P5:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.struct<(i32, i32)>
  // CHECK: %[[P6:.*]] = llvm.lshr %[[P5]], %{{.*}}  : i32
  // CHECK: %[[P7:.*]] = llvm.zext %[[P6]] : i32 to i64
  // CHECK: %[[P8:.*]] = llvm.shl %[[P7]], %{{.*}} : i64
  // CHECK: %[[P9:.*]] = llvm.or %{{.*}}, %[[P8]] : i64
  // CHECK: %[[P10:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(i32, i32)>
  // CHECK: %[[P11:.*]] = llvm.lshr %[[P10]], %{{.*}} : i32
  // CHECK: %[[P12:.*]] = llvm.zext %[[P11]] : i32 to i64
  // CHECK: %[[P13:.*]] = llvm.shl %[[P12]], %{{.*}} : i64
  // CHECK: %[[P14:.*]] = llvm.or %[[P9]], %[[P13]] : i64
  // CHECK: llvm.insertvalue %[[P14]], %{{.*}}[0] : !llvm.array<1 x i64>
  %0 = kgen.variant.create %arg0, 0 : <struct<(i32, i32)>>
  kgen.return %0 : !kgen.variant<struct<(i32, i32)>>
}

// CHECK-LABEL: @variant_create_4
kgen.func @variant_create_4(%arg0: !kgen.struct<(i32, i64, i32)>) -> !kgen.variant<struct<(i32, i64, i32)>, array<4, i64>> {
  // CHECK-DAG: %[[I64_32:.*]] = llvm.mlir.constant(32 : i64)
  // CHECK-DAG: %[[I32_0:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: %[[I64_0:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK-DAG: %[[P24:.*]] = llvm.mlir.undef : !llvm.array<4 x i64>
  // CHECK: %[[P5:.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i64, i32)>
  // CHECK: %[[P6:.*]] = llvm.lshr %[[P5]], %[[I32_0]] : i32
  // CHECK: %[[P7:.*]] = llvm.zext %[[P6]] : i32 to i64
  // CHECK: %[[P8:.*]] = llvm.shl %[[P7]], %[[I64_0]] : i64
  // CHECK: %[[P9:.*]] = llvm.or %[[I64_0]], %[[P8]] : i64
  // CHECK: %[[P10:.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i64, i32)>
  // CHECK: %[[P11:.*]] = llvm.lshr %[[P10]], %[[I64_0]] : i64
  // CHECK: %[[P12:.*]] = llvm.trunc %[[P11]] : i64 to i64
  // CHECK: %[[P13:.*]] = llvm.shl %[[P12]], %[[I64_32]] : i64
  // CHECK: %[[P14:.*]] = llvm.or %[[P9]], %[[P13]] : i64
  // CHECK: %[[P15:.*]] = llvm.lshr %[[P10]], %[[I64_32]] : i64
  // CHECK: %[[P16:.*]] = llvm.trunc %[[P15]] : i64 to i64
  // CHECK: %[[P17:.*]] = llvm.shl %[[P16]], %[[I64_0]] : i64
  // CHECK: %[[P18:.*]] = llvm.or %[[I64_0]], %[[P17]] : i64
  // CHECK: %[[P19:.*]] = llvm.extractvalue %arg0[2] : !llvm.struct<(i32, i64, i32)>
  // CHECK: %[[P20:.*]] = llvm.lshr %[[P19]], %[[I32_0]] : i32
  // CHECK: %[[P21:.*]] = llvm.zext %[[P20]] : i32 to i64
  // CHECK: %[[P22:.*]] = llvm.shl %[[P21]], %[[I64_32]] : i64
  // CHECK: %[[P23:.*]] = llvm.or %[[P18]], %[[P22]] : i64
  // CHECK: %[[P25:.*]] = llvm.insertvalue %[[P14]], %[[P24]][0] : !llvm.array<4 x i64>
  // CHECK: %[[P26:.*]] = llvm.insertvalue %[[P23]], %[[P25]][1] : !llvm.array<4 x i64>
  // CHECK: %[[P27:.*]] = llvm.insertvalue %[[I64_0]], %[[P26]][2] : !llvm.array<4 x i64>
  // CHECK: %[[P28:.*]] = llvm.insertvalue %[[I64_0]], %[[P27]][3] : !llvm.array<4 x i64>
  %0 = kgen.variant.create %arg0, 0 : <struct<(i32, i64, i32)>, array<4, i64>>
  kgen.return %0 : !kgen.variant<struct<(i32, i64, i32)>, array<4, i64>>
}

// CHECK-LABEL: @variant_create_5
kgen.func @variant_create_5(%arg0: !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>) -> !kgen.variant<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>> {
  // CHECK-DAG: %[[I32_1:.*]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: %[[I32_0:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: %[[I64_32:.*]] = llvm.mlir.constant(32 : i64)
  // CHECK-DAG: %[[I8_0:.*]] = llvm.mlir.constant(0 : i8)
  // CHECK-DAG: %[[I64_16:.*]] = llvm.mlir.constant(16 : i64)
  // CHECK-DAG: %[[I16_0:.*]] = llvm.mlir.constant(0 : i16)
  // CHECK-DAG: %[[I64_0:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK: %[[P12:.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK: %[[P13:.*]] = llvm.extractvalue %[[P12]][0] : !llvm.array<2 x i16>
  // CHECK: %[[P14:.*]] = llvm.lshr %[[P13]], %[[I16_0]] : i16
  // CHECK: %[[P15:.*]] = llvm.zext %[[P14]] : i16 to i64
  // CHECK: %[[P16:.*]] = llvm.shl %[[P15]], %[[I64_0]] : i64
  // CHECK: %[[P17:.*]] = llvm.or %[[I64_0]], %[[P16]] : i64
  // CHECK: %[[P18:.*]] = llvm.extractvalue %[[P12]][1] : !llvm.array<2 x i16>
  // CHECK: %[[P19:.*]] = llvm.lshr %[[P18]], %[[I16_0]] : i16
  // CHECK: %[[P20:.*]] = llvm.zext %[[P19]] : i16 to i64
  // CHECK: %[[P21:.*]] = llvm.shl %[[P20]], %[[I64_16]] : i64
  // CHECK: %[[P22:.*]] = llvm.or %[[P17]], %[[P21]] : i64
  // CHECK: %[[P23:.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK: %[[P24:.*]] = llvm.extractvalue %[[P23]][0] : !llvm.struct<(struct<(i8, i32)>, vector<2xf32>)>
  // CHECK: %[[P25:.*]] = llvm.extractvalue %[[P24]][0] : !llvm.struct<(i8, i32)>
  // CHECK: %[[P26:.*]] = llvm.lshr %[[P25]], %[[I8_0]] : i8
  // CHECK: %[[P27:.*]] = llvm.zext %[[P26]] : i8 to i64

  // COM: The second struct element is aligned to 8 bytes.
  // CHECK: %[[P28:.*]] = llvm.shl %[[P27]], %[[I64_0]] : i64
  // CHECK: %[[P29:.*]] = llvm.or %[[I64_0]], %[[P28]] : i64
  // CHECK: %[[P30:.*]] = llvm.extractvalue %[[P24]][1] : !llvm.struct<(i8, i32)>
  // CHECK: %[[P31:.*]] = llvm.lshr %[[P30]], %[[I32_0]] : i32
  // CHECK: %[[P32:.*]] = llvm.zext %[[P31]] : i32 to i64
  // CHECK: %[[P33:.*]] = llvm.shl %[[P32]], %[[I64_32]] : i64
  // CHECK: %[[P34:.*]] = llvm.or %[[P29]], %[[P33]] : i64
  // CHECK: %[[P39:.*]] = llvm.extractvalue %[[P23]][1] : !llvm.struct<(struct<(i8, i32)>, vector<2xf32>)>
  // CHECK: %[[P40:.*]] = llvm.extractelement %[[P39]][%[[I32_0]] : i32] : vector<2xf32>
  // CHECK: %[[P41:.*]] = llvm.bitcast %[[P40]] : f32 to i32
  // CHECK: %[[P42:.*]] = llvm.lshr %[[P41]], %[[I32_0]] : i32
  // CHECK: %[[P43:.*]] = llvm.zext %[[P42]] : i32 to i64
  // CHECK: %[[P44:.*]] = llvm.shl %[[P43]], %[[I64_0]] : i64
  // CHECK: %[[P45:.*]] = llvm.or %[[I64_0]], %[[P44]] : i64
  // CHECK: %[[P46:.*]] = llvm.extractelement %[[P39]][%[[I32_1]] : i32] : vector<2xf32>
  // CHECK: %[[P47:.*]] = llvm.bitcast %[[P46]] : f32 to i32
  // CHECK: %[[P48:.*]] = llvm.lshr %[[P47]], %[[I32_0]] : i32
  // CHECK: %[[P49:.*]] = llvm.zext %[[P48]] : i32 to i64
  // CHECK: %[[P50:.*]] = llvm.shl %[[P49]], %[[I64_32]] : i64
  // CHECK: %[[P51:.*]] = llvm.or %[[P45]], %[[P50]] : i64
  // CHECK: %[[S0:.*]] = llvm.insertvalue %[[P22]], %{{.*}}[0] : !llvm.array<3 x i64>
  // CHECK: %[[S1:.*]] = llvm.insertvalue %[[P34]], %[[S0]][1] : !llvm.array<3 x i64>
  // CHECK: %[[S2:.*]] = llvm.insertvalue %[[P51]], %[[S1]][2] : !llvm.array<3 x i64>
  // CHECK: insertvalue %[[S2]], %{{.*}}[0]
  %0 = kgen.variant.create %arg0, 0 : <struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>
  kgen.return %0 : !kgen.variant<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>
}

// CHECK-LABEL: @variant_create_6
kgen.func @variant_create_6(%arg0: !kgen.pointer<index>) -> !kgen.variant<pointer<index>> {
  // CHECK: llvm.ptrtoint
  %0 = kgen.variant.create %arg0, 0 : <pointer<index>>
  kgen.return %0 : !kgen.variant<pointer<index>>
}

// CHECK-LABEL: @variant_get_0
kgen.func @variant_get_0(%arg0: !kgen.variant<i32>) ->  i32{
  // CHECK-DAG: %[[C0:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK-DAG: %[[C0_i32:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK: %[[P4:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x i64>
  // CHECK: %[[P5:.*]] = llvm.lshr %[[P4]], %[[C0]] : i64
  // CHECK: %[[P6:.*]] = llvm.shl %[[P5]], %[[C0]] : i64
  // CHECK: %[[P7:.*]] = llvm.trunc %[[P6]] : i64 to i32
  // CHECK: %[[P8:.*]] = llvm.or %[[C0_i32]], %[[P7]] : i32
  %0 = kgen.variant.get %arg0, 0 : <i32>
  kgen.return %0 : i32
}

// CHECK-LABEL: @variant_get_1
kgen.func @variant_get_1(%arg0: !kgen.variant<f64>) -> f64 {
  // CHECK: llvm.bitcast %{{.*}} : i64 to f64
  %0 = kgen.variant.get %arg0, 0 : <f64>
  kgen.return %0 : f64
}

// CHECK-LABEL: @variant_get_2
kgen.func @variant_get_2(%arg0: !kgen.variant<struct<(i32, i32)>>) -> !kgen.struct<(i32, i32)>{
  // CHECK-DAG: %[[C32_i64:.*]] = llvm.mlir.constant(32 : i64)
  // CHECK-DAG: %[[C0_i64:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK-DAG: %[[C0_i32:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: %[[P6:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32)>
  // CHECK: %[[P5:.*]] = llvm.extractvalue %{{.*}}[0] : !llvm.array<1 x i64>
  // CHECK: %[[P7:.*]] = llvm.lshr %[[P5]], %[[C0_i64]] : i64
  // CHECK: %[[P8:.*]] = llvm.shl %[[P7]], %[[C0_i64]] : i64
  // CHECK: %[[P9:.*]] = llvm.trunc %[[P8]] : i64 to i32
  // CHECK: %[[P10:.*]] = llvm.or %[[C0_i32]], %[[P9]] : i32
  // CHECK: %[[P11:.*]] = llvm.insertvalue %[[P10]], %[[P6]][0] : !llvm.struct<(i32, i32)>
  // CHECK: %[[P12:.*]] = llvm.lshr %[[P5]], %[[C32_i64]] : i64
  // CHECK: %[[P13:.*]] = llvm.shl %[[P12]], %[[C0_i64]] : i64
  // CHECK: %[[P14:.*]] = llvm.trunc %[[P13]] : i64 to i32
  // CHECK: %[[P15:.*]] = llvm.or %[[C0_i32]], %[[P14]] : i32
  // CHECK: %[[P16:.*]] = llvm.insertvalue %[[P15]], %[[P11]][1] : !llvm.struct<(i32, i32)>
  %0 = kgen.variant.get %arg0, 0 : <struct<(i32, i32)>>
  kgen.return %0 : !kgen.struct<(i32, i32)>
}

// CHECK-LABEL: @variant_get_3
kgen.func @variant_get_3(%arg0: !kgen.variant<struct<(i32, i64, i32)>, array<4, i64>>) -> !kgen.struct<(i32, i64, i32)> {
  // CHECK: %[[C32_i64:.*]] = llvm.mlir.constant(32 : i64)
  // CHECK: %[[C0_i64:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK: %[[C0_i32:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK: %[[P7:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i64, i32)>
  // CHECK: %[[P4:.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(array<4 x i64>, i8)>
  // CHECK: %[[P5:.*]] = llvm.extractvalue %[[P4]][0] : !llvm.array<4 x i64>
  // CHECK: %[[P6:.*]] = llvm.extractvalue %[[P4]][1] : !llvm.array<4 x i64>
  // CHECK: %[[P8:.*]] = llvm.lshr %[[P5]], %[[C0_i64]] : i64
  // CHECK: %[[P9:.*]] = llvm.shl %[[P8]], %[[C0_i64]] : i64
  // CHECK: %[[P10:.*]] = llvm.trunc %[[P9]] : i64 to i32
  // CHECK: %[[P11:.*]] = llvm.or %[[C0_i32]], %[[P10]] : i32
  // CHECK: %[[P12:.*]] = llvm.insertvalue %[[P11]], %[[P7]][0] : !llvm.struct<(i32, i64, i32)>
  // CHECK: %[[P13:.*]] = llvm.lshr %[[P5]], %[[C32_i64]] : i64
  // CHECK: %[[P14:.*]] = llvm.shl %[[P13]], %[[C0_i64]] : i64
  // CHECK: %[[P15:.*]] = llvm.trunc %[[P14]] : i64 to i64
  // CHECK: %[[P16:.*]] = llvm.or %[[C0_i64]], %[[P15]] : i64
  // CHECK: %[[P17:.*]] = llvm.lshr %[[P6]], %[[C0_i64]] : i64
  // CHECK: %[[P18:.*]] = llvm.shl %[[P17]], %[[C32_i64]] : i64
  // CHECK: %[[P19:.*]] = llvm.trunc %[[P18]] : i64 to i64
  // CHECK: %[[P20:.*]] = llvm.or %[[P16]], %[[P19]] : i64
  // CHECK: %[[P21:.*]] = llvm.insertvalue %[[P20]], %[[P12]][1] : !llvm.struct<(i32, i64, i32)>
  // CHECK: %[[P22:.*]] = llvm.lshr %[[P6]], %[[C32_i64]] : i64
  // CHECK: %[[P23:.*]] = llvm.shl %[[P22]], %[[C0_i64]] : i64
  // CHECK: %[[P24:.*]] = llvm.trunc %[[P23]] : i64 to i32
  // CHECK: %[[P25:.*]] = llvm.or %[[C0_i32]], %[[P24]] : i32
  // CHECK: %[[P26:.*]] = llvm.insertvalue %[[P25]], %[[P21]][2] : !llvm.struct<(i32, i64, i32)>
  %0 = kgen.variant.get %arg0, 0 : <struct<(i32, i64, i32)>, array<4, i64>>
  kgen.return %0 : !kgen.struct<(i32, i64, i32)>
}

// CHECK-LABEL: @variant_get_4
kgen.func @variant_get_4(%arg0: !kgen.variant<struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>) -> !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)> {
  // CHECK-DAG: %[[C1_i32:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[C0_i32:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[C32_i64:.*]] = llvm.mlir.constant(32 : i64) : i64
  // CHECK-DAG: %[[C0_i8:.*]] = llvm.mlir.constant(0 : i8) : i8
  // CHECK-DAG: %[[C16_i64:.*]] = llvm.mlir.constant(16 : i64) : i64
  // CHECK-DAG: %[[C0_i64:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-DAG: %[[C0_i16:.*]] = llvm.mlir.constant(0 : i16) : i16
  // CHECK-DAG: %[[P15:.*]] = llvm.mlir.undef : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK-DAG: %[[P16:.*]] = llvm.mlir.undef : !llvm.array<2 x i16>
  // CHECK-DAG: %[[P28:.*]] = llvm.mlir.undef : !llvm.struct<(struct<(i8, i32)>, vector<2xf32>)>
  // CHECK-DAG: %[[P29:.*]] = llvm.mlir.undef : !llvm.struct<(i8, i32)>
  // CHECK-DAG: %[[P45:.*]] = llvm.mlir.undef : vector<2xf32>
  // CHECK: %[[P11:.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(array<3 x i64>, i8)>
  // CHECK: %[[P12:.*]] = llvm.extractvalue %[[P11]][0] : !llvm.array<3 x i64>
  // CHECK: %[[P13:.*]] = llvm.extractvalue %[[P11]][1] : !llvm.array<3 x i64>
  // CHECK: %[[P14:.*]] = llvm.extractvalue %[[P11]][2] : !llvm.array<3 x i64>
  // CHECK: %[[P17:.*]] = llvm.lshr %[[P12]], %[[C0_i64]] : i64
  // CHECK: %[[P18:.*]] = llvm.shl %[[P17]], %[[C0_i64]] : i64
  // CHECK: %[[P19:.*]] = llvm.trunc %[[P18]] : i64 to i16
  // CHECK: %[[P20:.*]] = llvm.or %[[C0_i16]], %[[P19]] : i16
  // CHECK: %[[P21:.*]] = llvm.insertvalue %[[P20]], %[[P16]][0] : !llvm.array<2 x i16>
  // CHECK: %[[P22:.*]] = llvm.lshr %[[P12]], %[[C16_i64]] : i64
  // CHECK: %[[P23:.*]] = llvm.shl %[[P22]], %[[C0_i64]] : i64
  // CHECK: %[[P24:.*]] = llvm.trunc %[[P23]] : i64 to i16
  // CHECK: %[[P25:.*]] = llvm.or %[[C0_i16]], %[[P24]] : i16
  // CHECK: %[[P26:.*]] = llvm.insertvalue %[[P25]], %[[P21]][1] : !llvm.array<2 x i16>
  // CHECK: %[[P27:.*]] = llvm.insertvalue %[[P26]], %[[P15]][0] : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  // CHECK: %[[P30:.*]] = llvm.lshr %[[P13]], %[[C0_i64]] : i64
  // CHECK: %[[P31:.*]] = llvm.shl %[[P30]], %[[C0_i64]] : i64
  // CHECK: %[[P32:.*]] = llvm.trunc %[[P31]] : i64 to i8
  // CHECK: %[[P33:.*]] = llvm.or %[[C0_i8]], %[[P32]] : i8
  // CHECK: %[[P34:.*]] = llvm.insertvalue %[[P33]], %[[P29]][0] : !llvm.struct<(i8, i32)>
  // CHECK: %[[P35:.*]] = llvm.lshr %[[P13]], %[[C32_i64]] : i64
  // CHECK: %[[P36:.*]] = llvm.shl %[[P35]], %[[C0_i64]] : i64
  // CHECK: %[[P37:.*]] = llvm.trunc %[[P36]] : i64 to i32
  // CHECK: %[[P38:.*]] = llvm.or %[[C0_i32]], %[[P37]] : i32
  // CHECK: %[[P43:.*]] = llvm.insertvalue %[[P38]], %[[P34]][1] : !llvm.struct<(i8, i32)>
  // CHECK: %[[P44:.*]] = llvm.insertvalue %[[P43]], %[[P28]][0] : !llvm.struct<(struct<(i8, i32)>, vector<2xf32>)>
  // CHECK: %[[P46:.*]] = llvm.lshr %[[P14]], %[[C0_i64]] : i64
  // CHECK: %[[P47:.*]] = llvm.shl %[[P46]], %[[C0_i64]] : i64
  // CHECK: %[[P48:.*]] = llvm.trunc %[[P47]] : i64 to i32
  // CHECK: %[[P49:.*]] = llvm.or %[[C0_i32]], %[[P48]] : i32
  // CHECK: %[[P50:.*]] = llvm.bitcast %[[P49]] : i32 to f32
  // CHECK: %[[P51:.*]] = llvm.insertelement %[[P50]], %[[P45]][%[[C0_i32]] : i32] : vector<2xf32>
  // CHECK: %[[P52:.*]] = llvm.lshr %[[P14]], %[[C32_i64]] : i64
  // CHECK: %[[P53:.*]] = llvm.shl %[[P52]], %[[C0_i64]] : i64
  // CHECK: %[[P54:.*]] = llvm.trunc %[[P53]] : i64 to i32
  // CHECK: %[[P55:.*]] = llvm.or %[[C0_i32]], %[[P54]] : i32
  // CHECK: %[[P60:.*]] = llvm.bitcast %[[P55]] : i32 to f32
  // CHECK: %[[P61:.*]] = llvm.insertelement %[[P60]], %[[P51]][%[[C1_i32]] : i32] : vector<2xf32>
  // CHECK: %[[P62:.*]] = llvm.insertvalue %[[P61]], %[[P44]][1] : !llvm.struct<(struct<(i8, i32)>, vector<2xf32>)>
  // CHECK: %[[P63:.*]] = llvm.insertvalue %[[P62]], %[[P27]][1] : !llvm.struct<(array<2 x i16>, struct<(struct<(i8, i32)>, vector<2xf32>)>)>
  %0 = kgen.variant.get %arg0, 0 : <struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>>
  kgen.return %0 : !kgen.struct<(array<2, i16>, struct<(struct<(i8, i32)>, simd<2, f32>)>)>
}

// CHECK-LABEL: @variant_get_5
kgen.func @variant_get_5(%arg0: !kgen.variant<pointer<index>>) -> !kgen.pointer<index> {
  // CHECK: llvm.inttoptr
  %0 = kgen.variant.get %arg0, 0 : <pointer<index>>
  kgen.return %0 : !kgen.pointer<index>
}

// CHECK-LABEL: @unpack_pointer
kgen.func @unpack_pointer(%arg0: !kgen.variant<pointer<i8>>) -> !kgen.pointer<i8> {
  // CHECK: trunc %{{.*}} : i64 to i64
  // CHECK: inttoptr %{{.*}} : i64 to !llvm.ptr
  %0 = kgen.variant.get %arg0, 0 : <pointer<i8>>
  kgen.return %0 : !kgen.pointer<i8>
}

}
