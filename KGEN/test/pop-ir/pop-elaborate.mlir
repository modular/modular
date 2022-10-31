// RUN: kgen-opt -split-input-file -elaborate-generators %s | FileCheck %s

// CHECK-LABEL: @"int_to_wider_int
kgen.generator @int_to_wider_int<type: dtype>() -> !pop.simd<1, type> {
  // CHECK: pop.constant(#M.dense_array<4> : vector<1xui32>)
  %0 = pop.constant(4 : i8) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_wider_int<type: dtype = ui32>() : () -> !pop.simd<1, ui32>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_same_width_int
kgen.generator @int_to_same_width_int<type: dtype>() -> !pop.simd<1, type> {
  // CHECK: pop.constant(#M.dense_array<65532> : vector<1xui16>)
  %0 = pop.constant(-4 : i16) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_same_width_int<type: dtype = ui16>() : () -> !pop.simd<1, ui16>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_shorter_int
kgen.generator @int_to_shorter_int<type: dtype>() -> !pop.simd<1, type> {
  // CHECK: pop.constant(#M.dense_array<42> : vector<1xsi8>)
  %0 = pop.constant(42 : si64) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_shorter_int<type: dtype = si8>() : () -> !pop.simd<1, si8>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_float
kgen.generator @int_to_float<type: dtype>() -> !pop.simd<1, type> {
  // CHECK: pop.constant(#M.dense_array<5.{{0+}}e+02> : vector<1xf32>)
  %0 = pop.constant(500) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_float<type: dtype = f32>() : () -> !pop.simd<1, f32>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_float
kgen.generator @int_to_float<type: dtype>() -> !pop.simd<1, type> {
  // CHECK: pop.constant(#M.dense_array<-5.{{0+}}e+02> : vector<1xf32>)
  %0 = pop.constant(-500) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_float<type: dtype = f32>() : () -> !pop.simd<1, f32>
  kgen.return
}

// -----

// CHECK-LABEL: @"float_to_int
kgen.generator @float_to_int<type: dtype>() -> !pop.simd<1, type> {
  // CHECK: pop.constant(#M.dense_array<15> : vector<1xui8>)
  %0 = pop.constant(15.0) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

kgen.generator @impl() {
  %0 = kgen.call @float_to_int<type: dtype = ui8>() : () -> !pop.simd<1, ui8>
  kgen.return
}

// -----

// CHECK-LABEL: @"float_to_int
kgen.generator @float_to_int<type: dtype>() -> !pop.simd<1, type> {
  // CHECK: pop.constant(#M.dense_array<-15> : vector<1xsi8>)
  %0 = pop.constant(-15.0) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

kgen.generator @impl() {
  %0 = kgen.call @float_to_int<type: dtype = si8>() : () -> !pop.simd<1, si8>
  kgen.return
}

// -----

// CHECK-LABEL: @"float_to_float
kgen.generator @float_to_float<type: dtype>() -> !pop.simd<1, type> {
  // CHECK: pop.constant(#M.dense_array<1.195{{.*}}e+00> : vector<1xbf16>)
  %0 = pop.constant(1.2) : !pop.simd<1, type>
  kgen.return %0 : !pop.simd<1, type>
}

kgen.generator @impl() {
  %0 = kgen.call @float_to_float<type: dtype = bf16>() : () -> !pop.simd<1, bf16>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_vec
kgen.generator @int_vec<type: dtype>() -> !pop.simd<4, type> {
  // CHECK: pop.constant(#M.dense_array<1, 2, 3, 4> : vector<4xsi8>)
  %0 = pop.constant(#M.dense_array<1., 2., 3., 4.> : vector<4xf32>) : !pop.simd<4, type>
  kgen.return %0 : !pop.simd<4, type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_vec<type: dtype = si8>() : () -> !pop.simd<4, si8>
  kgen.return
}

// -----

// CHECK-LABEL: @"flt_vec
kgen.generator @flt_vec<type: dtype>() -> !pop.simd<4, type> {
  // CHECK: pop.constant(#M.dense_array<1.{{.*}}, 2.{{.*}}, 3.{{.*}}, 4.{{.*}}> : vector<4xbf16>)
  %0 = pop.constant(#M.dense_array<1, 2, 3, 4> : vector<4xi64>) : !pop.simd<4, type>
  kgen.return %0 : !pop.simd<4, type>
}

kgen.generator @impl() {
  %0 = kgen.call @flt_vec<type: dtype = bf16>() : () -> !pop.simd<4, bf16>
  kgen.return
}

// -----

// CHECK-LABEL: @"splat_constant
kgen.generator @splat_constant<size>() -> !pop.simd<size, f32> {
  // CHECK: pop.constant(#M.dense_array<{{.*}}> : vector<8xf32>)
  %0 = pop.constant(0.0 : f32) : !pop.simd<size, f32>
  kgen.return %0 : !pop.simd<size, f32>
}

kgen.generator @impl() {
  %0 = kgen.call @splat_constant<size = 8>() : () -> !pop.simd<8, f32>
  kgen.return
}

// -----

// CHECK-LABEL: @"splat_constant,type=!pop.simd<1, si32>"
// CHECK: pop.constant(#M.dense_array<1> : vector<1xsi32>) : !pop.simd<1, si32>

// CHECK-LABEL: @"splat_constant,type=!pop.simd<1, f32>"
// CHECK: pop.constant(#M.dense_array<1.{{0+}}e+00> : vector<1xf32>) : !pop.simd<1, f32>

// CHECK-LABEL: @"splat_constant,type=!pop.simd<4, si32>"
// CHECK: pop.constant(#M.dense_array<1, 1, 1, 1> : vector<4xsi32>) : !pop.simd<4, si32>

kgen.generator @splat_constant<type: type>() -> !kgen.paramref<type> {
  %0 = pop.constant(1) : !kgen.paramref<type>
  kgen.return %0 : !kgen.paramref<type>
}

kgen.generator @impl() {
  %0 = kgen.call @splat_constant<type: type = !pop.simd<1, si32>>() : () -> !pop.simd<1, si32>
  %1 = kgen.call @splat_constant<type: type = !pop.simd<1, f32>>() : () -> !pop.simd<1, f32>
  %2 = kgen.call @splat_constant<type: type = !pop.simd<4, si32>>() : () -> !pop.simd<4, si32>
  kgen.return
}

// -----

// CHECK-LABEL: @"array_constant
kgen.generator @array_constant<dtype: dtype>() {
  // CHECK: pop.global_constant(#M.dense_array<1.{{0+}}e+00, 2.{{0+}}e+00> : !M.array<2xf32>) : !pop.array<2, simd<1, f32>>
  %0 = pop.global_constant(#M.dense_array<1, 2> : !M.array<2xi32>) : !pop.array<2, simd<1, dtype>>
  kgen.return
}

kgen.generator @impl() {
  kgen.call @array_constant<dtype: dtype = f32>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: @"array_constant
kgen.generator @array_constant<size>() {
  // CHECK: pop.global_constant(#M.dense_array<1, 1> : !M.array<2xui64>) : !pop.array<2, simd<1, ui64>>
  %0 = pop.global_constant(1 : ui64) : !pop.array<size, simd<1, ui64>>
  kgen.return
}

kgen.generator @impl() {
  kgen.call @array_constant<size = 2>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: @reify_i1
kgen.generator @reify_i1() {
  // CHECK: pop.constant(true) : !pop.simd<1, bool>
  %0 = pop.constant(1:i1) : !pop.simd<1, bool>
  kgen.return
}
