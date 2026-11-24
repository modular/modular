// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(kgen.func(legalize-pop-operations))' %s | FileCheck %s

module attributes {M.target_info = #M.target<triple = "nvptx-nvidia-cuda", arch="sm_90", data_layout="">} {
  // CHECK-LABEL: @neg_f8e5m2
  kgen.func @neg_f8e5m2(%arg0: !kgen.scalar<f8e5m2>) -> !kgen.scalar<f8e5m2> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[NEG_0:.*]] = pop.neg %[[CAST_0]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %[[NEG_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[CAST_1]] : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
    %0 = pop.neg %arg0 : !kgen.scalar<f8e5m2>
    // CHECK-NEXT: hlcf.return %[[CAST_2]] : !kgen.scalar<f8e5m2>
    hlcf.return %0 : !kgen.scalar<f8e5m2>
  }

  // CHECK-LABEL: @neg_f8e4m3fn
  kgen.func @neg_f8e4m3fn(%arg0: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f8e4m3fn> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[NEG_0:.*]] = pop.neg %[[CAST_0]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %[[NEG_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[CAST_1]] : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    %0 = pop.neg %arg0 : !kgen.scalar<f8e4m3fn>
    // CHECK-NEXT: hlcf.return %[[CAST_2]] : !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }

  // CHECK-LABEL: @add_f8e5m2
  kgen.func @add_f8e5m2(%arg0: !kgen.scalar<f8e5m2>) -> !kgen.scalar<f8e5m2> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[ADD_0:.*]] = pop.add %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[ADD_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
    %0 = pop.add %arg0, %arg0 : !kgen.scalar<f8e5m2>
    // CHECK-NEXT: hlcf.return %[[CAST_3]] : !kgen.scalar<f8e5m2>
    hlcf.return %0 : !kgen.scalar<f8e5m2>
  }

  // CHECK-LABEL: @add_f8e4m3fn
  kgen.func @add_f8e4m3fn(%arg0: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f8e4m3fn> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[ADD_0:.*]] = pop.add %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[ADD_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    %0 = pop.add %arg0, %arg0 : !kgen.scalar<f8e4m3fn>
    // CHECK-NEXT: hlcf.return %[[CAST_3]] : !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }

  // CHECK-LABEL: @sub_f8e5m2
  kgen.func @sub_f8e5m2(%arg0: !kgen.scalar<f8e5m2>) -> !kgen.scalar<f8e5m2> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[SUB_0:.*]] = pop.sub %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[SUB_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
    %0 = pop.sub %arg0, %arg0 : !kgen.scalar<f8e5m2>
    // CHECK-NEXT: hlcf.return %[[CAST_3]] : !kgen.scalar<f8e5m2>
    hlcf.return %0 : !kgen.scalar<f8e5m2>
  }

  // CHECK-LABEL: @sub_f8e4m3fn
  kgen.func @sub_f8e4m3fn(%arg0: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f8e4m3fn> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[SUB_0:.*]] = pop.sub %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[SUB_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    %0 = pop.sub %arg0, %arg0 : !kgen.scalar<f8e4m3fn>
    // CHECK-NEXT: hlcf.return %[[CAST_3]] : !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }

  // CHECK-LABEL: @mul_f8e5m2
  kgen.func @mul_f8e5m2(%arg0: !kgen.scalar<f8e5m2>) -> !kgen.scalar<f8e5m2> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[MUL_0:.*]] = pop.mul %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[MUL_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
    %0 = pop.mul %arg0, %arg0 : !kgen.scalar<f8e5m2>
    // CHECK-NEXT: hlcf.return %[[CAST_3]] : !kgen.scalar<f8e5m2>
    hlcf.return %0 : !kgen.scalar<f8e5m2>
  }

  // CHECK-LABEL: @mul_f8e4m3fn
  kgen.func @mul_f8e4m3fn(%arg0: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f8e4m3fn> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[MUL_0:.*]] = pop.mul %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[MUL_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    %0 = pop.mul %arg0, %arg0 : !kgen.scalar<f8e4m3fn>
    // CHECK-NEXT: hlcf.return %[[CAST_3]] : !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }

  // CHECK-LABEL: @div_f8e5m2
  kgen.func @div_f8e5m2(%arg0: !kgen.scalar<f8e5m2>) -> !kgen.scalar<f8e5m2> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[DIV_0:.*]] = pop.div %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[DIV_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
    %0 = pop.div %arg0, %arg0 : !kgen.scalar<f8e5m2>
    // CHECK-NEXT: hlcf.return %[[CAST_3]] : !kgen.scalar<f8e5m2>
    hlcf.return %0 : !kgen.scalar<f8e5m2>
  }

  // CHECK-LABEL: @div_f8e4m3fn
  kgen.func @div_f8e4m3fn(%arg0: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f8e4m3fn> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[DIV_0:.*]] = pop.div %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[DIV_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    %0 = pop.div %arg0, %arg0 : !kgen.scalar<f8e4m3fn>
    // CHECK-NEXT: hlcf.return %[[CAST_3]] : !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }

  // CHECK-LABEL: @rem_f8e5m2
  kgen.func @rem_f8e5m2(%arg0: !kgen.scalar<f8e5m2>) -> !kgen.scalar<f8e5m2> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[REM_0:.*]] = pop.rem %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[REM_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    %0 = pop.rem %arg0, %arg0 : !kgen.scalar<f8e5m2>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
    hlcf.return %0 : !kgen.scalar<f8e5m2>
  }

  // CHECK-LABEL: @rem_f8e4m3fn
  kgen.func @rem_f8e4m3fn(%arg0: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f8e4m3fn> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[REM_0:.*]] = pop.rem %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[REM_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    %0 = pop.rem %arg0, %arg0 : !kgen.scalar<f8e4m3fn>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }

  // CHECK-LABEL: @max_f8e5m2
  kgen.func @max_f8e5m2(%arg0: !kgen.scalar<f8e5m2>) -> !kgen.scalar<f8e5m2> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[MAX_0:.*]] = pop.max %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[MAX_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    %0 = pop.max %arg0, %arg0 : !kgen.scalar<f8e5m2>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
    hlcf.return %0 : !kgen.scalar<f8e5m2>
  }

  // CHECK-LABEL: @max_f8e4m3fn
  kgen.func @max_f8e4m3fn(%arg0: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f8e4m3fn> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[MAX_0:.*]] = pop.max %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[MAX_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    %0 = pop.max %arg0, %arg0 : !kgen.scalar<f8e4m3fn>
    // CHECK-NEXT: hlcf.return %[[CAST_3]] : !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }

  // CHECK-LABEL: @min_f8e5m2
  kgen.func @min_f8e5m2(%arg0: !kgen.scalar<f8e5m2>) -> !kgen.scalar<f8e5m2> {
    // CHECK-SAME:      %[[ARG0:.*]]: !kgen.scalar<f8e5m2>) -> !kgen.scalar<f8e5m2> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[MIN_0:.*]] = pop.min %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[MIN_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
    %0 = pop.min %arg0, %arg0 : !kgen.scalar<f8e5m2>
    // CHECK-NEXT: hlcf.return %[[CAST_3]] : !kgen.scalar<f8e5m2>
    hlcf.return %0 : !kgen.scalar<f8e5m2>
  }

  // CHECK-LABEL: @min_f8e4m3fn
  kgen.func @min_f8e4m3fn(%arg0: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f8e4m3fn> {
    // CHECK-NEXT: %[[CAST_0:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_1:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[MIN_0:.*]] = pop.min %[[CAST_0]], %[[CAST_1]] : !kgen.scalar<f16>
    // CHECK-NEXT: %[[CAST_2:.*]] = pop.cast %[[MIN_0]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[CAST_3:.*]] = pop.cast %[[CAST_2]] : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    %0 = pop.min %arg0, %arg0 : !kgen.scalar<f8e4m3fn>
    // CHECK-NEXT: hlcf.return %[[CAST_3]] : !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }

  kgen.func @cast_f16_to_f8e5m2(%arg0: !kgen.scalar<f16>) -> !kgen.scalar<f8e5m2> {
    // CHECK: %[[F32:.*]] = pop.cast %arg0 : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[FP8:.*]] = pop.cast %[[F32]] : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
    %0 = pop.cast %arg0 : !kgen.scalar<f16> to !kgen.scalar<f8e5m2>
    // CHECK-NEXT: hlcf.return %[[FP8]] : !kgen.scalar<f8e5m2>
    hlcf.return %0 : !kgen.scalar<f8e5m2>
  }

  kgen.func @cast_f16_to_f8e4m3fn(%arg0: !kgen.scalar<f16>) -> !kgen.scalar<f8e4m3fn> {
    // CHECK: %[[F32:.*]] = pop.cast %arg0 : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[FP8:.*]] = pop.cast %[[F32]] : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>
    %0 = pop.cast %arg0 : !kgen.scalar<f16> to !kgen.scalar<f8e4m3fn>
    // CHECK-NEXT: hlcf.return %[[FP8]] : !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }

  // CHECK-LABEL: @cast_bf16_to_f8e5m2
  kgen.func @cast_bf16_to_f8e5m2(%arg0: !kgen.scalar<bf16>) -> !kgen.scalar<f8e5m2> {
    // CHECK: %[[F32:.*]] = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[FP8:.*]] = pop.cast %[[F32]] : !kgen.scalar<f32> to !kgen.scalar<f8e5m2>
    %0 = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f8e5m2>
    // CHECK-NEXT: hlcf.return %[[FP8]] : !kgen.scalar<f8e5m2>
    hlcf.return %0 : !kgen.scalar<f8e5m2>
  }

  // CHECK-LABEL: @cast_bf16_to_f8e4m3fn
  kgen.func @cast_bf16_to_f8e4m3fn(%arg0: !kgen.scalar<bf16>) -> !kgen.scalar<f8e4m3fn> {
    // CHECK: %[[F32:.*]] = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[FP8:.*]] = pop.cast %[[F32]] : !kgen.scalar<f32> to !kgen.scalar<f8e4m3fn>

    %0 = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f8e4m3fn>
    // CHECK-NEXT: hlcf.return %[[FP8]] : !kgen.scalar<f8e4m3fn>
    hlcf.return %0 : !kgen.scalar<f8e4m3fn>
  }

  // CHECK-LABEL: @cast_bf16_to_f16
  kgen.func @cast_bf16_to_f16(%arg0: !kgen.scalar<bf16>) -> !kgen.scalar<f16> {
    // CHECK: %[[F32:.*]] = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK: %[[F16:.*]] = pop.cast %[[F32]] : !kgen.scalar<f32> to !kgen.scalar<f16>
    %0 = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f16>
    // CHECK-NEXT: hlcf.return %[[F16]] : !kgen.scalar<f16>
    hlcf.return %0 : !kgen.scalar<f16>
  }

  // CHECK-LABEL: @cast_f8e5m2_to_f32
  kgen.func @cast_f8e5m2_to_f32(%arg0: !kgen.scalar<f8e5m2>) -> !kgen.scalar<f32> {
    // CHECK: %[[F16:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[F32:.*]] = pop.cast %[[F16]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    %0 = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f32>
    // CHECK-NEXT: hlcf.return %[[F32]] : !kgen.scalar<f32>
    hlcf.return %0 : !kgen.scalar<f32>
  }

  // CHECK-LABEL: @cast_f8e4m3fn_to_f32
  kgen.func @cast_f8e4m3fn_to_f32(%arg0: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f32> {
    // CHECK: %[[F16:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[F32:.*]] = pop.cast %[[F16]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    %0 = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f32>
    // CHECK-NEXT: hlcf.return %[[F32]] : !kgen.scalar<f32>
    hlcf.return %0 : !kgen.scalar<f32>
  }

  // CHECK-LABEL: @cast_f8e5m2_to_bf16
  kgen.func @cast_f8e5m2_to_bf16(%arg0: !kgen.scalar<f8e5m2>) -> !kgen.scalar<bf16> {
    // CHECK: %[[F16:.*]] = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[F32:.*]] = pop.cast %[[F16]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[BF16:.*]] = pop.cast %[[F32]] : !kgen.scalar<f32> to !kgen.scalar<bf16>
    %0 = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<bf16>
    // CHECK-NEXT: hlcf.return %[[BF16]] : !kgen.scalar<bf16>
    hlcf.return %0 : !kgen.scalar<bf16>
  }

  // CHECK-LABEL: @cast_f8e4m3fn_to_bf16
  kgen.func @cast_f8e4m3fn_to_bf16(%arg0: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<bf16> {
    // CHECK: %[[F16:.*]] = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    // CHECK-NEXT: %[[F32:.*]] = pop.cast %[[F16]] : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[BF16:.*]] = pop.cast %[[F32]] : !kgen.scalar<f32> to !kgen.scalar<bf16>
    %0 = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<bf16>
    // CHECK-NEXT: hlcf.return %[[BF16]] : !kgen.scalar<bf16>
    hlcf.return %0 : !kgen.scalar<bf16>
  }

  // CHECK-LABEL: @cast_f16_to_bf16
  kgen.func @cast_f16_to_bf16(%arg0: !kgen.scalar<f16>) -> !kgen.scalar<bf16> {
    // CHECK: %[[F32:.*]] = pop.cast %arg0 : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK: %[[BF16:.*]] = pop.cast %[[F32]] : !kgen.scalar<f32> to !kgen.scalar<bf16>
    %0 = pop.cast %arg0 : !kgen.scalar<f16> to !kgen.scalar<bf16>
    // CHECK-NEXT: hlcf.return %[[BF16]] : !kgen.scalar<bf16>
    hlcf.return %0 : !kgen.scalar<bf16>
  }

  // COM: Negative tests for which legalization shouldn't kick-in

  // CHECK-LABEL: @neg_f32
  kgen.func @neg_f32(%arg0: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
    // CHECK: pop.neg %arg0 : !kgen.scalar<f32>
    %0 = pop.neg %arg0 : !kgen.scalar<f32>
    hlcf.return %0 : !kgen.scalar<f32>
  }

  // CHECK-LABEL: @add_f32
  kgen.func @add_f32(%arg0: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
    // CHECK: pop.add %arg0, %arg0 : !kgen.scalar<f32>
    %0 = pop.add %arg0, %arg0 : !kgen.scalar<f32>
    hlcf.return %0 : !kgen.scalar<f32>
  }

  // CHECK-LABEL: @cast_f8e5m2_to_f16
  kgen.func @cast_f8e5m2_to_f16(%arg0: !kgen.scalar<f8e5m2>) -> !kgen.scalar<f16> {
    // CHECK: pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    %0 = pop.cast %arg0 : !kgen.scalar<f8e5m2> to !kgen.scalar<f16>
    hlcf.return %0 : !kgen.scalar<f16>
  }

  // CHECK-LABEL: @cast_f8e4m3fn_to_f16
  kgen.func @cast_f8e4m3fn_to_f16(%arg0: !kgen.scalar<f8e4m3fn>) -> !kgen.scalar<f16> {
    // CHECK: pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    %0 = pop.cast %arg0 : !kgen.scalar<f8e4m3fn> to !kgen.scalar<f16>
    hlcf.return %0 : !kgen.scalar<f16>
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "nvptx-nvidia-cuda", arch="sm_80", data_layout="">} {
  // CHECK-LABEL: @cast_f16_to_bf16
  kgen.func @cast_f16_to_bf16(%arg0: !kgen.scalar<f16>) -> !kgen.scalar<bf16> {
    // CHECK: %[[F32:.*]] = pop.cast %arg0 : !kgen.scalar<f16> to !kgen.scalar<f32>
    // CHECK: %[[BF16:.*]] = pop.cast %[[F32]] : !kgen.scalar<f32> to !kgen.scalar<bf16>
    %0 = pop.cast %arg0 : !kgen.scalar<f16> to !kgen.scalar<bf16>
    // CHECK-NEXT: hlcf.return %[[BF16]] : !kgen.scalar<bf16>
    hlcf.return %0 : !kgen.scalar<bf16>
  }

  // CHECK-LABEL: @cast_bf16_to_f16
  kgen.func @cast_bf16_to_f16(%arg0: !kgen.scalar<bf16>) -> !kgen.scalar<f16> {
    // CHECK: %[[F32:.*]] = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK: %[[F16:.*]] = pop.cast %[[F32]] : !kgen.scalar<f32> to !kgen.scalar<f16>
    %0 = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f16>
    // CHECK-NEXT: hlcf.return %[[F16]] : !kgen.scalar<f16>
    hlcf.return %0 : !kgen.scalar<f16>
  }
}

// -----

// Metal M1 (apple-m1): same as M2 — bf16 arithmetic must be legalized to f32.
module attributes {M.target_info = #M.target<triple = "air64-apple-macosx", arch = "apple-m1", data_layout = "">} {
  // CHECK-LABEL: @mul_bf16_m1
  kgen.func @mul_bf16_m1(%arg0: !kgen.scalar<bf16>) -> !kgen.scalar<bf16> {
    // CHECK-NEXT: %[[A:.*]] = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[B:.*]] = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[MUL:.*]] = pop.mul %[[A]], %[[B]] : !kgen.scalar<f32>
    // CHECK-NEXT: %[[RES:.*]] = pop.cast %[[MUL]] : !kgen.scalar<f32> to !kgen.scalar<bf16>
    %0 = pop.mul %arg0, %arg0 : !kgen.scalar<bf16>
    // CHECK-NEXT: hlcf.return %[[RES]] : !kgen.scalar<bf16>
    hlcf.return %0 : !kgen.scalar<bf16>
  }
}

// -----

// Metal M2 (apple-m2): bf16 arithmetic has no native hardware support and must
// be legalized to f32.
module attributes {M.target_info = #M.target<triple = "air64-apple-macosx", arch = "apple-m2", data_layout = "">} {
  // CHECK-LABEL: @add_bf16_m2
  kgen.func @add_bf16_m2(%arg0: !kgen.scalar<bf16>) -> !kgen.scalar<bf16> {
    // CHECK-NEXT: %[[A:.*]] = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[B:.*]] = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[ADD:.*]] = pop.add %[[A]], %[[B]] : !kgen.scalar<f32>
    // CHECK-NEXT: %[[RES:.*]] = pop.cast %[[ADD]] : !kgen.scalar<f32> to !kgen.scalar<bf16>
    %0 = pop.add %arg0, %arg0 : !kgen.scalar<bf16>
    // CHECK-NEXT: hlcf.return %[[RES]] : !kgen.scalar<bf16>
    hlcf.return %0 : !kgen.scalar<bf16>
  }

  // CHECK-LABEL: @mul_bf16_m2
  kgen.func @mul_bf16_m2(%arg0: !kgen.scalar<bf16>) -> !kgen.scalar<bf16> {
    // CHECK-NEXT: %[[A:.*]] = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[B:.*]] = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[MUL:.*]] = pop.mul %[[A]], %[[B]] : !kgen.scalar<f32>
    // CHECK-NEXT: %[[RES:.*]] = pop.cast %[[MUL]] : !kgen.scalar<f32> to !kgen.scalar<bf16>
    %0 = pop.mul %arg0, %arg0 : !kgen.scalar<bf16>
    // CHECK-NEXT: hlcf.return %[[RES]] : !kgen.scalar<bf16>
    hlcf.return %0 : !kgen.scalar<bf16>
  }

  // CHECK-LABEL: @fma_bf16_m2
  kgen.func @fma_bf16_m2(%arg0: !kgen.scalar<bf16>, %arg1: !kgen.scalar<bf16>, %arg2: !kgen.scalar<bf16>) -> !kgen.scalar<bf16> {
    // CHECK-NEXT: %[[A:.*]] = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[B:.*]] = pop.cast %arg1 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[C:.*]] = pop.cast %arg2 : !kgen.scalar<bf16> to !kgen.scalar<f32>
    // CHECK-NEXT: %[[FMA:.*]] = pop.fma %[[A]], %[[B]], %[[C]] : !kgen.scalar<f32>
    // CHECK-NEXT: %[[RES:.*]] = pop.cast %[[FMA]] : !kgen.scalar<f32> to !kgen.scalar<bf16>
    %0 = pop.fma %arg0, %arg1, %arg2 : !kgen.scalar<bf16>
    // CHECK-NEXT: hlcf.return %[[RES]] : !kgen.scalar<bf16>
    hlcf.return %0 : !kgen.scalar<bf16>
  }

  // CHECK-LABEL: @mul_simd_bf16_m2
  kgen.func @mul_simd_bf16_m2(%arg0: !kgen.simd<4, bf16>) -> !kgen.simd<4, bf16> {
    // CHECK-NEXT: %[[A:.*]] = pop.cast %arg0 : !kgen.simd<4, bf16> to !kgen.simd<4, f32>
    // CHECK-NEXT: %[[B:.*]] = pop.cast %arg0 : !kgen.simd<4, bf16> to !kgen.simd<4, f32>
    // CHECK-NEXT: %[[MUL:.*]] = pop.mul %[[A]], %[[B]] : !kgen.simd<4, f32>
    // CHECK-NEXT: %[[RES:.*]] = pop.cast %[[MUL]] : !kgen.simd<4, f32> to !kgen.simd<4, bf16>
    %0 = pop.mul %arg0, %arg0 : !kgen.simd<4, bf16>
    // CHECK-NEXT: hlcf.return %[[RES]] : !kgen.simd<4, bf16>
    hlcf.return %0 : !kgen.simd<4, bf16>
  }

  // Int<->bf16 casts are handled by AIR intrinsics and must not be legalized.
  // CHECK-LABEL: @cast_ui64_to_bf16_m2
  kgen.func @cast_ui64_to_bf16_m2(%arg0: !kgen.scalar<ui64>) -> !kgen.scalar<bf16> {
    // CHECK-NEXT: %0 = pop.cast %arg0 : !kgen.scalar<ui64> to !kgen.scalar<bf16>
    // CHECK-NOT: pop.cast
    %0 = pop.cast %arg0 : !kgen.scalar<ui64> to !kgen.scalar<bf16>
    // CHECK-NEXT: hlcf.return %0 : !kgen.scalar<bf16>
    hlcf.return %0 : !kgen.scalar<bf16>
  }

  // CHECK-LABEL: @cast_bf16_to_ui64_m2
  kgen.func @cast_bf16_to_ui64_m2(%arg0: !kgen.scalar<bf16>) -> !kgen.scalar<ui64> {
    // CHECK-NEXT: %0 = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<ui64>
    // CHECK-NOT: pop.cast
    %0 = pop.cast %arg0 : !kgen.scalar<bf16> to !kgen.scalar<ui64>
    // CHECK-NEXT: hlcf.return %0 : !kgen.scalar<ui64>
    hlcf.return %0 : !kgen.scalar<ui64>
  }
}

// -----

// Metal M3 (apple-m3): bf16 arithmetic is natively supported — no legalization.
module attributes {M.target_info = #M.target<triple = "air64-apple-macosx", arch = "apple-m3", data_layout = "">} {
  // CHECK-LABEL: @add_bf16_m3
  kgen.func @add_bf16_m3(%arg0: !kgen.scalar<bf16>) -> !kgen.scalar<bf16> {
    // CHECK-NEXT: %0 = pop.add %arg0, %arg0 : !kgen.scalar<bf16>
    // CHECK-NOT: pop.cast
    %0 = pop.add %arg0, %arg0 : !kgen.scalar<bf16>
    // CHECK-NEXT: hlcf.return %0 : !kgen.scalar<bf16>
    hlcf.return %0 : !kgen.scalar<bf16>
  }

  // CHECK-LABEL: @mul_bf16_m3
  kgen.func @mul_bf16_m3(%arg0: !kgen.scalar<bf16>) -> !kgen.scalar<bf16> {
    // CHECK-NEXT: %0 = pop.mul %arg0, %arg0 : !kgen.scalar<bf16>
    // CHECK-NOT: pop.cast
    %0 = pop.mul %arg0, %arg0 : !kgen.scalar<bf16>
    // CHECK-NEXT: hlcf.return %0 : !kgen.scalar<bf16>
    hlcf.return %0 : !kgen.scalar<bf16>
  }

  // CHECK-LABEL: @mul_simd_bf16_m3
  kgen.func @mul_simd_bf16_m3(%arg0: !kgen.simd<4, bf16>) -> !kgen.simd<4, bf16> {
    // CHECK-NEXT: %0 = pop.mul %arg0, %arg0 : !kgen.simd<4, bf16>
    // CHECK-NOT: pop.cast
    %0 = pop.mul %arg0, %arg0 : !kgen.simd<4, bf16>
    // CHECK-NEXT: hlcf.return %0 : !kgen.simd<4, bf16>
    hlcf.return %0 : !kgen.simd<4, bf16>
  }
}
