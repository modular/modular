// RUN: kgen-opt -split-input-file -allow-unregistered-dialect -lower-global-pop-to-llvm %s | FileCheck %s --check-prefix=CHECK
// RUN: kgen-opt -split-input-file -allow-unregistered-dialect -lower-global-pop-to-llvm --mlir-print-debuginfo %s | FileCheck %s --check-prefix=CHECK-DEBUG-INFO

module attributes {M.target_info = #M.target<triple = "air64-apple-macosx", arch = "", data_layout = "", simd_bit_width = 128>} {
  // CHECK-LABEL: @test_air_cos_f16_mangling
  llvm.func @test_air_cos_f16_mangling(%arg0: !llvm.struct<(f16)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(f16)> to !kgen.struct<(scalar<f16>)>
    // CHECK: llvm.call @air.cos.f16
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.cos", (%input) : (!kgen.struct<(scalar<f16>)>) -> !kgen.scalar<f16>
    llvm.return
  }

  // CHECK-LABEL: @test_air_cos_f32_mangling
  llvm.func @test_air_cos_f32_mangling(%arg0: !llvm.struct<(f32)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(f32)> to !kgen.struct<(scalar<f32>)>
    // CHECK: llvm.call @air.cos.f32
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.cos", (%input) : (!kgen.struct<(scalar<f32>)>) -> !kgen.scalar<f32>
    llvm.return
  }

  // CHECK-LABEL: @test_air_cos_bf16_mangling
  llvm.func @test_air_cos_bf16_mangling(%arg0: !llvm.struct<(bf16)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(bf16)> to !kgen.struct<(scalar<bf16>)>
    // CHECK: llvm.call @air.cos.bf16
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.cos", (%input) : (!kgen.struct<(scalar<bf16>)>) -> !kgen.scalar<bf16>
    llvm.return
  }

  // COM: There's no cos for integer type, but it's needed to verify mangling is correct
  // CHECK-LABEL: @test_air_cos_i32_mangling
  llvm.func @test_air_cos_i32_mangling(%arg0: !llvm.struct<(i32)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(i32)> to !kgen.struct<(scalar<si32>)>
    // CHECK: llvm.call @air.cos.i32
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.cos", (%input) : (!kgen.struct<(scalar<si32>)>) -> !kgen.scalar<si32>
    llvm.return
  }

  // COM: There's no cos for integer type, but it's needed to verify mangling is correct
  // CHECK-LABEL: @test_air_cos_i128_mangling
  llvm.func @test_air_cos_i128_mangling(%arg0: !llvm.struct<(i128)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(i128)> to !kgen.struct<(scalar<si128>)>
    // CHECK: llvm.call @air.cos.i128
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.cos", (%input) : (!kgen.struct<(scalar<si128>)>) -> !kgen.scalar<si128>
    llvm.return
  }

  // CHECK-LABEL: @test_air_max_bf16_mangling
  kgen.func @test_air_max_bf16_mangling(%arg0: !kgen.scalar<bf16>, %arg1: !kgen.scalar<bf16>) {
    // CHECK: llvm.call @air.fmax.bf16
    %0 = pop.max %arg0, %arg1 : !kgen.scalar<bf16>
    llvm.return
  }

  // CHECK-LABEL: @test_air_cos_v2f16_mangling
  llvm.func @test_air_cos_v2f16_mangling(%arg0: !llvm.struct<(vector<2 x f16>)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(vector<2 x f16>)> to !kgen.struct<(!kgen.simd<2, f16>)>
    // CHECK: llvm.call @air.cos.v2f16
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.cos", (%input) : (!kgen.struct<(!kgen.simd<2, f16>)>) -> !kgen.simd<2, f16>
    llvm.return
  }

  // CHECK-LABEL: @test_air_min_f16_mangling
  kgen.func @test_air_min_f16_mangling(%arg0: !kgen.scalar<f16>, %arg1: !kgen.scalar<f16>) {
    // CHECK: llvm.call @air.fmin.f16
    %0 = pop.min %arg0, %arg1 : !kgen.scalar<f16>
    llvm.return
  }

  // CHECK-LABEL: @test_air_min_si32_mangling
  kgen.func @test_air_min_si32_mangling(%arg0: !kgen.scalar<si32>, %arg1: !kgen.scalar<si32>) {
    // CHECK: llvm.call @air.min.s.i32
    %0 = pop.min %arg0, %arg1 : !kgen.scalar<si32>
    llvm.return
  }

  // CHECK-LABEL: @test_air_min_ui32_mangling
  kgen.func @test_air_min_ui32_mangling(%arg0: !kgen.scalar<ui32>, %arg1: !kgen.scalar<ui32>) {
    // CHECK: llvm.call @air.min.u.i32
    %0 = pop.min %arg0, %arg1 : !kgen.scalar<ui32>
    llvm.return
  }

  // CHECK-LABEL: @test_air_min_v2i32_mangling
  kgen.func @test_air_min_v2i32_mangling(%arg0: !kgen.simd<2, ui32>, %arg1: !kgen.simd<2, ui32>) {
    // CHECK: llvm.call @air.min.u.v2i32
    %0 = pop.min %arg0, %arg1 : !kgen.simd<2, ui32>
    llvm.return
  }

  // CHECK-LABEL: @test_air_max_f16_mangling
  kgen.func @test_air_max_f16_mangling(%arg0: !kgen.scalar<f16>, %arg1: !kgen.scalar<f16>) {
    // CHECK: llvm.call @air.fmax.f16
    %0 = pop.max %arg0, %arg1 : !kgen.scalar<f16>
    llvm.return
  }

  // CHECK-LABEL: @test_air_max_si32_mangling
  kgen.func @test_air_max_si32_mangling(%arg0: !kgen.scalar<si32>, %arg1: !kgen.scalar<si32>) {
    // CHECK: llvm.call @air.max.s.i32
    %0 = pop.max %arg0, %arg1 : !kgen.scalar<si32>
    llvm.return
  }

  // CHECK-LABEL: @test_air_max_ui32_mangling
  kgen.func @test_air_max_ui32_mangling(%arg0: !kgen.scalar<ui32>, %arg1: !kgen.scalar<ui32>) {
    // CHECK: llvm.call @air.max.u.i32
    %0 = pop.max %arg0, %arg1 : !kgen.scalar<ui32>
    llvm.return
  }

  // CHECK-LABEL: @test_air_max_v2i32_mangling
  kgen.func @test_air_max_v2i32_mangling(%arg0: !kgen.simd<2, ui32>, %arg1: !kgen.simd<2, ui32>) {
    // CHECK: llvm.call @air.max.u.v2i32
    %0 = pop.max %arg0, %arg1 : !kgen.simd<2, ui32>
    llvm.return
  }

  kgen.func @test_pop_cast_all_types(
    %si8_val: !kgen.scalar<si8>,
    %ui8_val: !kgen.scalar<ui8>,
    %si16_val: !kgen.scalar<si16>,
    %ui16_val: !kgen.scalar<ui16>,
    %si32_val: !kgen.scalar<si32>,
    %ui32_val: !kgen.scalar<ui32>,
    %f16_val: !kgen.scalar<f16>,
    %bf16_val: !kgen.scalar<bf16>,
    %f32_val: !kgen.scalar<f32>
  ) {
    // CHECK-LABEL:   kgen.func @test_pop_cast_all_types(
    // CHECK-SAME: %[[ARG0:.*]]: !kgen.scalar<si8>,
    // CHECK-SAME: %[[ARG1:.*]]: !kgen.scalar<ui8>,
    // CHECK-SAME: %[[ARG2:.*]]: !kgen.scalar<si16>,
    // CHECK-SAME: %[[ARG3:.*]]: !kgen.scalar<ui16>,
    // CHECK-SAME: %[[ARG4:.*]]: !kgen.scalar<si32>,
    // CHECK-SAME: %[[ARG5:.*]]: !kgen.scalar<ui32>,
    // CHECK-SAME: %[[ARG6:.*]]: !kgen.scalar<f16>,
    // CHECK-SAME: %[[ARG7:.*]]: !kgen.scalar<bf16>,
    // CHECK-SAME: %[[ARG8:.*]]: !kgen.scalar<f32>) {
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG8]] : !kgen.scalar<f32> to f32
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG7]] : !kgen.scalar<bf16> to bf16
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[ARG6]] : !kgen.scalar<f16> to f16
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_3:.*]] = builtin.unrealized_conversion_cast %[[ARG5]] : !kgen.scalar<ui32> to i32
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_4:.*]] = builtin.unrealized_conversion_cast %[[ARG4]] : !kgen.scalar<si32> to i32
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_5:.*]] = builtin.unrealized_conversion_cast %[[ARG3]] : !kgen.scalar<ui16> to i16
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_6:.*]] = builtin.unrealized_conversion_cast %[[ARG2]] : !kgen.scalar<si16> to i16
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_7:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !kgen.scalar<ui8> to i8
    // CHECK: %[[UNREALIZED_CONVERSION_CAST_8:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !kgen.scalar<si8> to i8
    // ========================================================================
    // from si8
    // ========================================================================
    // CHECK: %[[CALL_0:.*]] = llvm.call @air.convert.f.f16.s.i8(%[[UNREALIZED_CONVERSION_CAST_8]]) : (i8) -> f16
    %si8_to_f16 = pop.cast %si8_val : !kgen.scalar<si8> to !kgen.scalar<f16>

    // CHECK: %[[CALL_1:.*]] = llvm.call @air.convert.f.bf16.s.i8(%[[UNREALIZED_CONVERSION_CAST_8]]) : (i8) -> bf16
    %si8_to_bf16 = pop.cast %si8_val : !kgen.scalar<si8> to !kgen.scalar<bf16>

    // CHECK: %[[CALL_2:.*]] = llvm.call @air.convert.f.f32.s.i8(%[[UNREALIZED_CONVERSION_CAST_8]]) : (i8) -> f32
    %si8_to_f32 = pop.cast %si8_val : !kgen.scalar<si8> to !kgen.scalar<f32>

    // ========================================================================
    // from ui8
    // ========================================================================
    // CHECK: %[[CALL_3:.*]] = llvm.call @air.convert.f.f16.u.i8(%[[UNREALIZED_CONVERSION_CAST_7]]) : (i8) -> f16
    %ui8_to_f16 = pop.cast %ui8_val : !kgen.scalar<ui8> to !kgen.scalar<f16>

    // CHECK: %[[CALL_4:.*]] = llvm.call @air.convert.f.bf16.u.i8(%[[UNREALIZED_CONVERSION_CAST_7]]) : (i8) -> bf16
    %ui8_to_bf16 = pop.cast %ui8_val : !kgen.scalar<ui8> to !kgen.scalar<bf16>

    // CHECK: %[[CALL_5:.*]] = llvm.call @air.convert.f.f32.u.i8(%[[UNREALIZED_CONVERSION_CAST_7]]) : (i8) -> f32
    %ui8_to_f32 = pop.cast %ui8_val : !kgen.scalar<ui8> to !kgen.scalar<f32>

    // ========================================================================
    // from si16
    // ========================================================================
    // CHECK: %[[CALL_6:.*]] = llvm.call @air.convert.f.f16.s.i16(%[[UNREALIZED_CONVERSION_CAST_6]]) : (i16) -> f16
    %si16_to_f16 = pop.cast %si16_val : !kgen.scalar<si16> to !kgen.scalar<f16>

    // CHECK: %[[CALL_7:.*]] = llvm.call @air.convert.f.bf16.s.i16(%[[UNREALIZED_CONVERSION_CAST_6]]) : (i16) -> bf16
    %si16_to_bf16 = pop.cast %si16_val : !kgen.scalar<si16> to !kgen.scalar<bf16>

    // CHECK: %[[CALL_8:.*]] = llvm.call @air.convert.f.f32.s.i16(%[[UNREALIZED_CONVERSION_CAST_6]]) : (i16) -> f32
    %si16_to_f32 = pop.cast %si16_val : !kgen.scalar<si16> to !kgen.scalar<f32>

    // ========================================================================
    // from ui16
    // ========================================================================
    // CHECK: %[[CALL_9:.*]] = llvm.call @air.convert.f.f16.u.i16(%[[UNREALIZED_CONVERSION_CAST_5]]) : (i16) -> f16
    %ui16_to_f16 = pop.cast %ui16_val : !kgen.scalar<ui16> to !kgen.scalar<f16>

    // CHECK: %[[CALL_10:.*]] = llvm.call @air.convert.f.bf16.u.i16(%[[UNREALIZED_CONVERSION_CAST_5]]) : (i16) -> bf16
    %ui16_to_bf16 = pop.cast %ui16_val : !kgen.scalar<ui16> to !kgen.scalar<bf16>

    // CHECK: %[[CALL_11:.*]] = llvm.call @air.convert.f.f32.u.i16(%[[UNREALIZED_CONVERSION_CAST_5]]) : (i16) -> f32
    %ui16_to_f32 = pop.cast %ui16_val : !kgen.scalar<ui16> to !kgen.scalar<f32>

    // ========================================================================
    // from si32
    // ========================================================================
    // CHECK: %[[CALL_12:.*]] = llvm.call @air.convert.f.f16.s.i32(%[[UNREALIZED_CONVERSION_CAST_4]]) : (i32) -> f16
    %si32_to_f16 = pop.cast %si32_val : !kgen.scalar<si32> to !kgen.scalar<f16>

    // CHECK: %[[CALL_13:.*]] = llvm.call @air.convert.f.bf16.s.i32(%[[UNREALIZED_CONVERSION_CAST_4]]) : (i32) -> bf16
    %si32_to_bf16 = pop.cast %si32_val : !kgen.scalar<si32> to !kgen.scalar<bf16>

    // CHECK: %[[CALL_14:.*]] = llvm.call @air.convert.f.f32.s.i32(%[[UNREALIZED_CONVERSION_CAST_4]]) : (i32) -> f32
    %si32_to_f32 = pop.cast %si32_val : !kgen.scalar<si32> to !kgen.scalar<f32>

    // ========================================================================
    // from ui32
    // ========================================================================
    // CHECK: %[[CALL_15:.*]] = llvm.call @air.convert.f.f16.u.i32(%[[UNREALIZED_CONVERSION_CAST_3]]) : (i32) -> f16
    %ui32_to_f16 = pop.cast %ui32_val : !kgen.scalar<ui32> to !kgen.scalar<f16>

    // CHECK: %[[CALL_16:.*]] = llvm.call @air.convert.f.bf16.u.i32(%[[UNREALIZED_CONVERSION_CAST_3]]) : (i32) -> bf16
    %ui32_to_bf16 = pop.cast %ui32_val : !kgen.scalar<ui32> to !kgen.scalar<bf16>

    // CHECK: %[[CALL_17:.*]] = llvm.call @air.convert.f.f32.u.i32(%[[UNREALIZED_CONVERSION_CAST_3]]) : (i32) -> f32
    %ui32_to_f32 = pop.cast %ui32_val : !kgen.scalar<ui32> to !kgen.scalar<f32>

    // ========================================================================
    // from f16
    // ========================================================================
    // CHECK: %[[CALL_18:.*]] = llvm.call @air.convert.s.i8.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> i8
    %f16_to_si8 = pop.cast %f16_val : !kgen.scalar<f16> to !kgen.scalar<si8>

    // CHECK: %[[CALL_19:.*]] = llvm.call @air.convert.u.i8.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> i8
    %f16_to_ui8 = pop.cast %f16_val : !kgen.scalar<f16> to !kgen.scalar<ui8>

    // CHECK: %[[CALL_20:.*]] = llvm.call @air.convert.s.i16.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> i16
    %f16_to_si16 = pop.cast %f16_val : !kgen.scalar<f16> to !kgen.scalar<si16>

    // CHECK: %[[CALL_21:.*]] = llvm.call @air.convert.u.i16.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> i16
    %f16_to_ui16 = pop.cast %f16_val : !kgen.scalar<f16> to !kgen.scalar<ui16>

    // CHECK: %[[CALL_22:.*]] = llvm.call @air.convert.s.i32.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> i32
    %f16_to_si32 = pop.cast %f16_val : !kgen.scalar<f16> to !kgen.scalar<si32>

    // CHECK: %[[CALL_23:.*]] = llvm.call @air.convert.u.i32.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> i32
    %f16_to_ui32 = pop.cast %f16_val : !kgen.scalar<f16> to !kgen.scalar<ui32>

    // CHECK: %[[CALL_24:.*]] = llvm.call @air.convert.f.bf16.f.f16(%[[UNREALIZED_CONVERSION_CAST_2]]) : (f16) -> bf16
    %f16_to_bf16 = pop.cast %f16_val : !kgen.scalar<f16> to !kgen.scalar<bf16>

    // ========================================================================
    // from bf16
    // ========================================================================
    // CHECK: %[[CALL_24:.*]] = llvm.call @air.convert.s.i8.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> i8
    %bf16_to_si8 = pop.cast %bf16_val : !kgen.scalar<bf16> to !kgen.scalar<si8>

    // CHECK: %[[CALL_25:.*]] = llvm.call @air.convert.u.i8.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> i8
    %bf16_to_ui8 = pop.cast %bf16_val : !kgen.scalar<bf16> to !kgen.scalar<ui8>

    // CHECK: %[[CALL_26:.*]] = llvm.call @air.convert.s.i16.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> i16
    %bf16_to_si16 = pop.cast %bf16_val : !kgen.scalar<bf16> to !kgen.scalar<si16>

    // CHECK: %[[CALL_27:.*]] = llvm.call @air.convert.u.i16.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> i16
    %bf16_to_ui16 = pop.cast %bf16_val : !kgen.scalar<bf16> to !kgen.scalar<ui16>

    // CHECK: %[[CALL_28:.*]] = llvm.call @air.convert.s.i32.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> i32
    %bf16_to_si32 = pop.cast %bf16_val : !kgen.scalar<bf16> to !kgen.scalar<si32>

    // CHECK: %[[CALL_29:.*]] = llvm.call @air.convert.u.i32.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> i32
    %bf16_to_ui32 = pop.cast %bf16_val : !kgen.scalar<bf16> to !kgen.scalar<ui32>

    // CHECK: %[[CALL_291:.*]] = llvm.call @air.convert.f.f16.f.bf16(%[[UNREALIZED_CONVERSION_CAST_1]]) : (bf16) -> f16
    %bf16_to_f16 = pop.cast %bf16_val : !kgen.scalar<bf16> to !kgen.scalar<f16>

    // ========================================================================
    // from f32
    // ========================================================================
    // CHECK: %[[CALL_30:.*]] = llvm.call @air.convert.s.i8.f.f32(%[[UNREALIZED_CONVERSION_CAST_0]]) : (f32) -> i8
    %f32_to_si8 = pop.cast %f32_val : !kgen.scalar<f32> to !kgen.scalar<si8>

    // CHECK: %[[CALL_31:.*]] = llvm.call @air.convert.u.i8.f.f32(%[[UNREALIZED_CONVERSION_CAST_0]]) : (f32) -> i8
    %f32_to_ui8 = pop.cast %f32_val : !kgen.scalar<f32> to !kgen.scalar<ui8>

    // CHECK: %[[CALL_32:.*]] = llvm.call @air.convert.s.i16.f.f32(%[[UNREALIZED_CONVERSION_CAST_0]]) : (f32) -> i16
    %f32_to_si16 = pop.cast %f32_val : !kgen.scalar<f32> to !kgen.scalar<si16>

    // CHECK: %[[CALL_33:.*]] = llvm.call @air.convert.u.i16.f.f32(%[[UNREALIZED_CONVERSION_CAST_0]]) : (f32) -> i16
    %f32_to_ui16 = pop.cast %f32_val : !kgen.scalar<f32> to !kgen.scalar<ui16>

    // CHECK: %[[CALL_34:.*]] = llvm.call @air.convert.s.i32.f.f32(%[[UNREALIZED_CONVERSION_CAST_0]]) : (f32) -> i32
    %f32_to_si32 = pop.cast %f32_val : !kgen.scalar<f32> to !kgen.scalar<si32>

    // CHECK: %[[CALL_35:.*]] = llvm.call @air.convert.u.i32.f.f32(%[[UNREALIZED_CONVERSION_CAST_0]]) : (f32) -> i32
    %f32_to_ui32 = pop.cast %f32_val : !kgen.scalar<f32> to !kgen.scalar<ui32>

    hlcf.return
  }

  // CHECK-LABEL:   @test_vector_pop_cast_all_types(
  kgen.func @test_vector_pop_cast_all_types(
    %si8_val: !kgen.simd<2, si8>,
    %ui8_val: !kgen.simd<2, ui8>,
    %si16_val: !kgen.simd<2, si16>,
    %ui16_val: !kgen.simd<2, ui16>,
    %si32_val: !kgen.simd<2, si32>,
    %ui32_val: !kgen.simd<2, ui32>,
    %f16_val: !kgen.simd<2, f16>,
    %bf16_val: !kgen.simd<2, bf16>,
    %f32_val: !kgen.simd<2, f32>
  ) {
    // CHECK:  %[[CALL_0:.*]] = llvm.call @air.convert.s.v2i16.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi16>
    // CHECK:  %[[CALL_1:.*]] = llvm.call @air.convert.u.v2i16.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi16>
    // CHECK:  %[[CALL_2:.*]] = llvm.call @air.convert.s.v2i32.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi32>
    // CHECK:  %[[CALL_3:.*]] = llvm.call @air.convert.u.v2i32.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi32>
    // CHECK:  %[[CALL_4:.*]] = llvm.call @air.convert.f.v2f16.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xf16>
    // CHECK:  %[[CALL_5:.*]] = llvm.call @air.convert.f.v2bf16.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xbf16>
    // CHECK:  %[[CALL_6:.*]] = llvm.call @air.convert.f.v2f32.s.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xf32>
    // CHECK:  %[[CALL_7:.*]] = llvm.call @air.convert.s.v2i16.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi16>
    // CHECK:  %[[CALL_8:.*]] = llvm.call @air.convert.u.v2i16.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi16>
    // CHECK:  %[[CALL_9:.*]] = llvm.call @air.convert.s.v2i32.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi32>
    // CHECK:  %[[CALL_10:.*]] = llvm.call @air.convert.u.v2i32.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xi32>
    // CHECK:  %[[CALL_11:.*]] = llvm.call @air.convert.f.v2f16.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xf16>
    // CHECK:  %[[CALL_12:.*]] = llvm.call @air.convert.f.v2bf16.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xbf16>
    // CHECK:  %[[CALL_13:.*]] = llvm.call @air.convert.f.v2f32.u.v2i8({{.*}}) : (vector<2xi8>) -> vector<2xf32>
    // CHECK:  %[[CALL_14:.*]] = llvm.call @air.convert.s.v2i8.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi8>
    // CHECK:  %[[CALL_15:.*]] = llvm.call @air.convert.u.v2i8.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi8>
    // CHECK:  %[[CALL_16:.*]] = llvm.call @air.convert.s.v2i32.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi32>
    // CHECK:  %[[CALL_17:.*]] = llvm.call @air.convert.u.v2i32.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi32>
    // CHECK:  %[[CALL_18:.*]] = llvm.call @air.convert.f.v2f16.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xf16>
    // CHECK:  %[[CALL_19:.*]] = llvm.call @air.convert.f.v2bf16.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xbf16>
    // CHECK:  %[[CALL_20:.*]] = llvm.call @air.convert.f.v2f32.s.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xf32>
    // CHECK:  %[[CALL_21:.*]] = llvm.call @air.convert.s.v2i8.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi8>
    // CHECK:  %[[CALL_22:.*]] = llvm.call @air.convert.u.v2i8.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi8>
    // CHECK:  %[[CALL_23:.*]] = llvm.call @air.convert.s.v2i32.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi32>
    // CHECK:  %[[CALL_24:.*]] = llvm.call @air.convert.u.v2i32.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xi32>
    // CHECK:  %[[CALL_25:.*]] = llvm.call @air.convert.f.v2f16.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xf16>
    // CHECK:  %[[CALL_26:.*]] = llvm.call @air.convert.f.v2bf16.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xbf16>
    // CHECK:  %[[CALL_27:.*]] = llvm.call @air.convert.f.v2f32.u.v2i16({{.*}}) : (vector<2xi16>) -> vector<2xf32>
    // CHECK:  %[[CALL_28:.*]] = llvm.call @air.convert.s.v2i8.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi8>
    // CHECK:  %[[CALL_29:.*]] = llvm.call @air.convert.u.v2i8.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi8>
    // CHECK:  %[[CALL_30:.*]] = llvm.call @air.convert.s.v2i16.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi16>
    // CHECK:  %[[CALL_31:.*]] = llvm.call @air.convert.u.v2i16.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi16>
    // CHECK:  %[[CALL_32:.*]] = llvm.call @air.convert.f.v2f16.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xf16>
    // CHECK:  %[[CALL_33:.*]] = llvm.call @air.convert.f.v2bf16.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xbf16>
    // CHECK:  %[[CALL_34:.*]] = llvm.call @air.convert.f.v2f32.s.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xf32>
    // CHECK:  %[[CALL_35:.*]] = llvm.call @air.convert.s.v2i8.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi8>
    // CHECK:  %[[CALL_36:.*]] = llvm.call @air.convert.u.v2i8.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi8>
    // CHECK:  %[[CALL_37:.*]] = llvm.call @air.convert.s.v2i16.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi16>
    // CHECK:  %[[CALL_38:.*]] = llvm.call @air.convert.u.v2i16.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xi16>
    // CHECK:  %[[CALL_39:.*]] = llvm.call @air.convert.f.v2f16.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xf16>
    // CHECK:  %[[CALL_40:.*]] = llvm.call @air.convert.f.v2bf16.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xbf16>
    // CHECK:  %[[CALL_41:.*]] = llvm.call @air.convert.f.v2f32.u.v2i32({{.*}}) : (vector<2xi32>) -> vector<2xf32>
    // CHECK:  %[[CALL_42:.*]] = llvm.call @air.convert.s.v2i8.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xi8>
    // CHECK:  %[[CALL_43:.*]] = llvm.call @air.convert.u.v2i8.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xi8>
    // CHECK:  %[[CALL_44:.*]] = llvm.call @air.convert.s.v2i16.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xi16>
    // CHECK:  %[[CALL_45:.*]] = llvm.call @air.convert.u.v2i16.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xi16>
    // CHECK:  %[[CALL_46:.*]] = llvm.call @air.convert.s.v2i32.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xi32>
    // CHECK:  %[[CALL_47:.*]] = llvm.call @air.convert.u.v2i32.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xi32>
    // CHECK:  %[[CALL_48:.*]] = llvm.call @air.convert.f.v2bf16.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xbf16>
    // CHECK:  %[[CALL_49:.*]] = llvm.call @air.convert.f.v2f32.f.v2f16({{.*}}) : (vector<2xf16>) -> vector<2xf32>
    // CHECK:  %[[CALL_50:.*]] = llvm.call @air.convert.s.v2i8.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xi8>
    // CHECK:  %[[CALL_51:.*]] = llvm.call @air.convert.u.v2i8.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xi8>
    // CHECK:  %[[CALL_52:.*]] = llvm.call @air.convert.s.v2i16.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xi16>
    // CHECK:  %[[CALL_53:.*]] = llvm.call @air.convert.u.v2i16.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xi16>
    // CHECK:  %[[CALL_54:.*]] = llvm.call @air.convert.s.v2i32.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xi32>
    // CHECK:  %[[CALL_55:.*]] = llvm.call @air.convert.u.v2i32.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xi32>
    // CHECK:  %[[CALL_56:.*]] = llvm.call @air.convert.f.v2f16.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xf16>
    // CHECK:  %[[CALL_57:.*]] = llvm.call @air.convert.f.v2f32.f.v2bf16({{.*}}) : (vector<2xbf16>) -> vector<2xf32>
    // CHECK:  %[[CALL_58:.*]] = llvm.call @air.convert.s.v2i8.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xi8>
    // CHECK:  %[[CALL_59:.*]] = llvm.call @air.convert.u.v2i8.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xi8>
    // CHECK:  %[[CALL_60:.*]] = llvm.call @air.convert.s.v2i16.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xi16>
    // CHECK:  %[[CALL_61:.*]] = llvm.call @air.convert.u.v2i16.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xi16>
    // CHECK:  %[[CALL_62:.*]] = llvm.call @air.convert.s.v2i32.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xi32>
    // CHECK:  %[[CALL_63:.*]] = llvm.call @air.convert.u.v2i32.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xi32>
    // CHECK:  %[[CALL_64:.*]] = llvm.call @air.convert.f.v2f16.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xf16>
    // CHECK:  %[[CALL_65:.*]] = llvm.call @air.convert.f.v2bf16.f.v2f32({{.*}}) : (vector<2xf32>) -> vector<2xbf16>
    %si8_to_si16 = pop.cast %si8_val : !kgen.simd<2, si8> to !kgen.simd<2, si16>

    %si8_to_ui16 = pop.cast %si8_val : !kgen.simd<2, si8> to !kgen.simd<2, ui16>

    %si8_to_si32 = pop.cast %si8_val : !kgen.simd<2, si8> to !kgen.simd<2, si32>

    %si8_to_ui32 = pop.cast %si8_val : !kgen.simd<2, si8> to !kgen.simd<2, ui32>

    %si8_to_f16 = pop.cast %si8_val : !kgen.simd<2, si8> to !kgen.simd<2, f16>

    %si8_to_bf16 = pop.cast %si8_val : !kgen.simd<2, si8> to !kgen.simd<2, bf16>

    %si8_to_f32 = pop.cast %si8_val : !kgen.simd<2, si8> to !kgen.simd<2, f32>

    %ui8_to_si16 = pop.cast %ui8_val : !kgen.simd<2, ui8> to !kgen.simd<2, si16>

    %ui8_to_ui16 = pop.cast %ui8_val : !kgen.simd<2, ui8> to !kgen.simd<2, ui16>

    %ui8_to_si32 = pop.cast %ui8_val : !kgen.simd<2, ui8> to !kgen.simd<2, si32>

    %ui8_to_ui32 = pop.cast %ui8_val : !kgen.simd<2, ui8> to !kgen.simd<2, ui32>

    %ui8_to_f16 = pop.cast %ui8_val : !kgen.simd<2, ui8> to !kgen.simd<2, f16>

    %ui8_to_bf16 = pop.cast %ui8_val : !kgen.simd<2, ui8> to !kgen.simd<2, bf16>

    %ui8_to_f32 = pop.cast %ui8_val : !kgen.simd<2, ui8> to !kgen.simd<2, f32>

    %si16_to_si8 = pop.cast %si16_val : !kgen.simd<2, si16> to !kgen.simd<2, si8>

    %si16_to_ui8 = pop.cast %si16_val : !kgen.simd<2, si16> to !kgen.simd<2, ui8>

    %si16_to_si32 = pop.cast %si16_val : !kgen.simd<2, si16> to !kgen.simd<2, si32>

    %si16_to_ui32 = pop.cast %si16_val : !kgen.simd<2, si16> to !kgen.simd<2, ui32>

    %si16_to_f16 = pop.cast %si16_val : !kgen.simd<2, si16> to !kgen.simd<2, f16>

    %si16_to_bf16 = pop.cast %si16_val : !kgen.simd<2, si16> to !kgen.simd<2, bf16>

    %si16_to_f32 = pop.cast %si16_val : !kgen.simd<2, si16> to !kgen.simd<2, f32>

    %ui16_to_si8 = pop.cast %ui16_val : !kgen.simd<2, ui16> to !kgen.simd<2, si8>

    %ui16_to_ui8 = pop.cast %ui16_val : !kgen.simd<2, ui16> to !kgen.simd<2, ui8>

    %ui16_to_si32 = pop.cast %ui16_val : !kgen.simd<2, ui16> to !kgen.simd<2, si32>

    %ui16_to_ui32 = pop.cast %ui16_val : !kgen.simd<2, ui16> to !kgen.simd<2, ui32>

    %ui16_to_f16 = pop.cast %ui16_val : !kgen.simd<2, ui16> to !kgen.simd<2, f16>

    %ui16_to_bf16 = pop.cast %ui16_val : !kgen.simd<2, ui16> to !kgen.simd<2, bf16>

    %ui16_to_f32 = pop.cast %ui16_val : !kgen.simd<2, ui16> to !kgen.simd<2, f32>

    %si32_to_si8 = pop.cast %si32_val : !kgen.simd<2, si32> to !kgen.simd<2, si8>

    %si32_to_ui8 = pop.cast %si32_val : !kgen.simd<2, si32> to !kgen.simd<2, ui8>

    %si32_to_si16 = pop.cast %si32_val : !kgen.simd<2, si32> to !kgen.simd<2, si16>

    %si32_to_ui16 = pop.cast %si32_val : !kgen.simd<2, si32> to !kgen.simd<2, ui16>

    %si32_to_f16 = pop.cast %si32_val : !kgen.simd<2, si32> to !kgen.simd<2, f16>

    %si32_to_bf16 = pop.cast %si32_val : !kgen.simd<2, si32> to !kgen.simd<2, bf16>

    %si32_to_f32 = pop.cast %si32_val : !kgen.simd<2, si32> to !kgen.simd<2, f32>

    %ui32_to_si8 = pop.cast %ui32_val : !kgen.simd<2, ui32> to !kgen.simd<2, si8>

    %ui32_to_ui8 = pop.cast %ui32_val : !kgen.simd<2, ui32> to !kgen.simd<2, ui8>

    %ui32_to_si16 = pop.cast %ui32_val : !kgen.simd<2, ui32> to !kgen.simd<2, si16>

    %ui32_to_ui16 = pop.cast %ui32_val : !kgen.simd<2, ui32> to !kgen.simd<2, ui16>

    %ui32_to_f16 = pop.cast %ui32_val : !kgen.simd<2, ui32> to !kgen.simd<2, f16>

    %ui32_to_bf16 = pop.cast %ui32_val : !kgen.simd<2, ui32> to !kgen.simd<2, bf16>

    %ui32_to_f32 = pop.cast %ui32_val : !kgen.simd<2, ui32> to !kgen.simd<2, f32>

    %f16_to_si8 = pop.cast %f16_val : !kgen.simd<2, f16> to !kgen.simd<2, si8>

    %f16_to_ui8 = pop.cast %f16_val : !kgen.simd<2, f16> to !kgen.simd<2, ui8>

    %f16_to_si16 = pop.cast %f16_val : !kgen.simd<2, f16> to !kgen.simd<2, si16>

    %f16_to_ui16 = pop.cast %f16_val : !kgen.simd<2, f16> to !kgen.simd<2, ui16>

    %f16_to_si32 = pop.cast %f16_val : !kgen.simd<2, f16> to !kgen.simd<2, si32>

    %f16_to_ui32 = pop.cast %f16_val : !kgen.simd<2, f16> to !kgen.simd<2, ui32>

    %f16_to_bf16 = pop.cast %f16_val : !kgen.simd<2, f16> to !kgen.simd<2, bf16>

    %f16_to_f32 = pop.cast %f16_val : !kgen.simd<2, f16> to !kgen.simd<2, f32>

    %bf16_to_si8 = pop.cast %bf16_val : !kgen.simd<2, bf16> to !kgen.simd<2, si8>

    %bf16_to_ui8 = pop.cast %bf16_val : !kgen.simd<2, bf16> to !kgen.simd<2, ui8>

    %bf16_to_si16 = pop.cast %bf16_val : !kgen.simd<2, bf16> to !kgen.simd<2, si16>

    %bf16_to_ui16 = pop.cast %bf16_val : !kgen.simd<2, bf16> to !kgen.simd<2, ui16>

    %bf16_to_si32 = pop.cast %bf16_val : !kgen.simd<2, bf16> to !kgen.simd<2, si32>

    %bf16_to_ui32 = pop.cast %bf16_val : !kgen.simd<2, bf16> to !kgen.simd<2, ui32>

    %bf16_to_f16 = pop.cast %bf16_val : !kgen.simd<2, bf16> to !kgen.simd<2, f16>

    %bf16_to_f32 = pop.cast %bf16_val : !kgen.simd<2, bf16> to !kgen.simd<2, f32>

    %f32_to_si8 = pop.cast %f32_val : !kgen.simd<2, f32> to !kgen.simd<2, si8>

    %f32_to_ui8 = pop.cast %f32_val : !kgen.simd<2, f32> to !kgen.simd<2, ui8>

    %f32_to_si16 = pop.cast %f32_val : !kgen.simd<2, f32> to !kgen.simd<2, si16>

    %f32_to_ui16 = pop.cast %f32_val : !kgen.simd<2, f32> to !kgen.simd<2, ui16>

    %f32_to_si32 = pop.cast %f32_val : !kgen.simd<2, f32> to !kgen.simd<2, si32>

    %f32_to_ui32 = pop.cast %f32_val : !kgen.simd<2, f32> to !kgen.simd<2, ui32>

    %f32_to_f16 = pop.cast %f32_val : !kgen.simd<2, f32> to !kgen.simd<2, f16>

    %f32_to_bf16 = pop.cast %f32_val : !kgen.simd<2, f32> to !kgen.simd<2, bf16>

    hlcf.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "air64-apple-macosx", arch = "", data_layout = "", simd_bit_width = 128>} {
  // CHECK: llvm.func @air.thread_position_in_threadgroup.x() -> i32
  // CHECK-DEBUG-INFO: llvm.func @air.thread_position_in_threadgroup.x() -> i32 loc(#[[LOCATION:.*]])
  kgen.func @test_no_debug_on_air_func_declaration() {
    %undef = llvm.mlir.undef : !llvm.struct<()> loc(unknown)
    %empty_struct = builtin.unrealized_conversion_cast %undef : !llvm.struct<()> to !kgen.struct<()> loc(unknown)
    %res = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.thread_position_in_threadgroup.x", (%empty_struct) : (!kgen.struct<()>) -> !kgen.scalar<ui32> loc("dummy.mlir":1:1)
    hlcf.return
  }
  // CHECK-DEBUG-INFO: #[[LOCATION]] = loc(unknown)
}

// -----

// Test texture intrinsic lowering with special mangling (by data type, not all arg types)
module attributes {M.target_info = #M.target<triple = "air64-apple-macosx", arch = "", data_layout = "", simd_bit_width = 128>} {
  // CHECK-LABEL: @test_air_read_texture_2d_v4f32
  llvm.func @test_air_read_texture_2d_v4f32(%arg0: !llvm.struct<(ptr, vector<2 x i32>, i32, i32)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(ptr, vector<2 x i32>, i32, i32)> to !kgen.struct<(pointer<scalar<f32>>, simd<2, ui32>, scalar<si32>, scalar<si32>)>
    // Texture intrinsics should be mangled by data type only (v4f32 from result), not all arg types
    // CHECK: llvm.call @air.read_texture_2d.v4f32
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.read_texture_2d", (%input) : (!kgen.struct<(pointer<scalar<f32>>, simd<2, ui32>, scalar<si32>, scalar<si32>)>) -> !kgen.struct<(simd<4, f32>, scalar<ui8>)>
    llvm.return
  }

  // CHECK-LABEL: @test_air_write_texture_2d_v4f32
  llvm.func @test_air_write_texture_2d_v4f32(%arg0: !llvm.struct<(ptr, vector<2 x i32>, vector<4 x f32>, i32, i32)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(ptr, vector<2 x i32>, vector<4 x f32>, i32, i32)> to !kgen.struct<(pointer<scalar<f32>>, simd<2, ui32>, simd<4, f32>, scalar<si32>, scalar<si32>)>
    // Texture intrinsics should be mangled by data type (v4f32 from 3rd arg), not all arg types
    // CHECK: llvm.call @air.write_texture_2d.v4f32
    pop.call_llvm_intrinsic side_effecting<1> "llvm.air.write_texture_2d", (%input) : (!kgen.struct<(pointer<scalar<f32>>, simd<2, ui32>, simd<4, f32>, scalar<si32>, scalar<si32>)>) -> ()
    llvm.return
  }

  // Test sample_texture intrinsic - mangled by data type
  // CHECK-LABEL: @test_air_sample_texture_2d_v4f32
  llvm.func @test_air_sample_texture_2d_v4f32(%arg0: !llvm.struct<(ptr, ptr, vector<2 x f32>, i32)>) {
    %input = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(ptr, ptr, vector<2 x f32>, i32)> to !kgen.struct<(pointer<scalar<f32>>, pointer<scalar<f32>>, simd<2, f32>, scalar<si32>)>
    // CHECK: llvm.call @air.sample_texture_2d.v4f32
    %0 = pop.call_llvm_intrinsic side_effecting<0> "llvm.air.sample_texture_2d", (%input) : (!kgen.struct<(pointer<scalar<f32>>, pointer<scalar<f32>>, simd<2, f32>, scalar<si32>)>) -> !kgen.struct<(simd<4, f32>, scalar<ui8>)>
    llvm.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "", data_layout = "">} {
  // CHECK-LABEL: kgen.func @atomic_rmw
  kgen.func @atomic_rmw(%ptr0: !kgen.pointer<scalar<si32>>,
                        %val0: !kgen.scalar<si32>,
                        %ptr1: !kgen.pointer<scalar<f32>>,
                        %val1: !kgen.scalar<f32>,
                        %ptr2: !kgen.pointer<scalar<ui32>>,
                        %val2: !kgen.scalar<ui32>) {
    // CHECK: llvm.call @air.atomic.global.add.s.i32
    %0 = pop.atomic.rmw add(%ptr0, %val0) monotonic : !kgen.pointer<scalar<si32>>

    // CHECK: llvm.call @air.atomic.global.sub.s.i32
    %1 = pop.atomic.rmw sub(%ptr0, %val0) monotonic : !kgen.pointer<scalar<si32>>

    // CHECK: llvm.call @air.atomic.global.xor.s.i32
    %2 = pop.atomic.rmw xor(%ptr0, %val0) monotonic : !kgen.pointer<scalar<si32>>

    // CHECK: llvm.call @air.atomic.global.min.s.i32
    %3 = pop.atomic.rmw min(%ptr0, %val0) monotonic : !kgen.pointer<scalar<si32>>

    // CHECK: llvm.call @air.atomic.global.max.s.i32
    %4 = pop.atomic.rmw max(%ptr0, %val0) monotonic : !kgen.pointer<scalar<si32>>

    // CHECK: llvm.call @air.atomic.global.add.f32
    %5 = pop.atomic.rmw add(%ptr1, %val1) monotonic : !kgen.pointer<scalar<f32>>

    // CHECK: llvm.call @air.atomic.global.max.u.i32
    %6 = pop.atomic.rmw max(%ptr2, %val2) monotonic : !kgen.pointer<scalar<ui32>>

    // CHECK: llvm.call @air.atomic.global.add.s.i32
    %7 = pop.atomic.rmw add(%ptr0, %val0) unordered : !kgen.pointer<scalar<si32>>

    // CHECK: llvm.call @air.atomic.global.add.s.i32
    %8 = pop.atomic.rmw add(%ptr0, %val0) monotonic : !kgen.pointer<scalar<si32>>
    hlcf.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "", data_layout = "">} {
  // CHECK-LABEL: kgen.func @atomic_cmpxchg
  kgen.func @atomic_cmpxchg(%ptr0: !kgen.pointer<scalar<si32>>,
                            %cmp0: !kgen.scalar<si32>,
                            %val0: !kgen.scalar<si32>,
                            %ptr1: !kgen.pointer<scalar<ui32>, 3>,
                            %cmp1: !kgen.scalar<ui32>,
                            %val1: !kgen.scalar<ui32>) {
    // Metal only lowers the `weak` form; AIR has no strong primitive.
    // CHECK: llvm.call @air.atomic.global.cmpxchg.weak.i32
    %0 = pop.atomic.cmpxchg weak %ptr0, %cmp0, %val0 monotonic monotonic :
                      !kgen.pointer<scalar<si32>>

    // CHECK: llvm.call @air.atomic.local.cmpxchg.weak.i32
    %1 = pop.atomic.cmpxchg weak %ptr1, %cmp1, %val1 monotonic monotonic :
                      !kgen.pointer<scalar<ui32>, 3>

    // CHECK: llvm.call @air.atomic.global.cmpxchg.weak.i32
    %2 = pop.atomic.cmpxchg weak %ptr0, %cmp0, %val0 unordered unordered :
                      !kgen.pointer<scalar<si32>>

    // Both orderings collapse to `relax` on Apple GPU, so a mismatched
    // success/failure pair lowers the same as a matched one.
    // CHECK: llvm.call @air.atomic.global.cmpxchg.weak.i32
    %3 = pop.atomic.cmpxchg weak %ptr0, %cmp0, %val0 monotonic unordered :
                      !kgen.pointer<scalar<si32>>

    // CHECK: llvm.call @air.atomic.global.cmpxchg.weak.i32
    %4 = pop.atomic.cmpxchg weak %ptr0, %cmp0, %val0 unordered monotonic :
                      !kgen.pointer<scalar<si32>>

    // CHECK: llvm.call @air.atomic.local.cmpxchg.weak.i32
    %5 = pop.atomic.cmpxchg weak %ptr1, %cmp1, %val1 monotonic unordered :
                      !kgen.pointer<scalar<ui32>, 3>

    // CHECK: llvm.call @air.atomic.local.cmpxchg.weak.i32
    %6 = pop.atomic.cmpxchg weak %ptr1, %cmp1, %val1 unordered monotonic :
                      !kgen.pointer<scalar<ui32>, 3>
    hlcf.return
  }
}
