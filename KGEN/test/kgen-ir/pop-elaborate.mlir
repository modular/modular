// RUN: kgen-opt -split-input-file -elaborate-generators %s | FileCheck %s

// CHECK-LABEL: @"int_to_wider_int
kgen.generator @int_to_wider_int<type: dtype>() -> !meta.scalar<type> {
  // CHECK: pop.constant(4 : ui32)
  %0 = pop.constant(4 : i8) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_wider_int<type: dtype = ui32>() : () -> !meta.scalar<ui32>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_same_width_int
kgen.generator @int_to_same_width_int<type: dtype>() -> !meta.scalar<type> {
  // CHECK: pop.constant(65532 : ui16)
  %0 = pop.constant(-4 : i16) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_same_width_int<type: dtype = ui16>() : () -> !meta.scalar<ui16>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_shorter_int
kgen.generator @int_to_shorter_int<type: dtype>() -> !meta.scalar<type> {
  // CHECK: pop.constant(42 : si8)
  %0 = pop.constant(42 : si64) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_shorter_int<type: dtype = si8>() : () -> !meta.scalar<si8>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_float
kgen.generator @int_to_float<type: dtype>() -> !meta.scalar<type> {
  // CHECK: pop.constant(5.{{0+}}e+02 : f32)
  %0 = pop.constant(500) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_float<type: dtype = f32>() : () -> !meta.scalar<f32>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_float
kgen.generator @int_to_float<type: dtype>() -> !meta.scalar<type> {
  // CHECK: pop.constant(-5.{{0+}}e+02 : f32)
  %0 = pop.constant(-500) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_float<type: dtype = f32>() : () -> !meta.scalar<f32>
  kgen.return
}

// -----

// CHECK-LABEL: @"float_to_int
kgen.generator @float_to_int<type: dtype>() -> !meta.scalar<type> {
  // CHECK: pop.constant(15 : ui8)
  %0 = pop.constant(15.0) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @float_to_int<type: dtype = ui8>() : () -> !meta.scalar<ui8>
  kgen.return
}

// -----

// CHECK-LABEL: @"float_to_int
kgen.generator @float_to_int<type: dtype>() -> !meta.scalar<type> {
  // CHECK: pop.constant(-15 : si8)
  %0 = pop.constant(-15.0) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @float_to_int<type: dtype = si8>() : () -> !meta.scalar<si8>
  kgen.return
}

// -----

// CHECK-LABEL: @"float_to_float
kgen.generator @float_to_float<type: dtype>() -> !meta.scalar<type> {
  // CHECK: pop.constant(1.195{{.*}}e+00 : bf16)
  %0 = pop.constant(1.2) : !meta.scalar<type>
  kgen.return %0 : !meta.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @float_to_float<type: dtype = bf16>() : () -> !meta.scalar<bf16>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_vec
kgen.generator @int_vec<type: dtype>() -> !meta.simd<4, type> {
  // CHECK: pop.constant(dense<[1, 2, 3, 4]> : vector<4xsi8>)
  %0 = pop.constant(dense<[1., 2., 3., 4.]> : vector<4xf32>) : !meta.simd<4, type>
  kgen.return %0 : !meta.simd<4, type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_vec<type: dtype = si8>() : () -> !meta.simd<4, si8>
  kgen.return
}

// -----

// CHECK-LABEL: @"flt_vec
kgen.generator @flt_vec<type: dtype>() -> !meta.simd<4, type> {
  // CHECK: pop.constant(dense<[1.{{.*}}, 2.{{.*}}, 3.{{.*}}, 4.{{.*}}]> : vector<4xbf16>)
  %0 = pop.constant(dense<[1, 2, 3, 4]> : vector<4xi64>) : !meta.simd<4, type>
  kgen.return %0 : !meta.simd<4, type>
}

kgen.generator @impl() {
  %0 = kgen.call @flt_vec<type: dtype = bf16>() : () -> !meta.simd<4, bf16>
  kgen.return
}
