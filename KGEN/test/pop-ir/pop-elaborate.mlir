// RUN: kgen-opt -split-input-file -elaborate-generators %s | FileCheck %s

// CHECK-LABEL: @"int_to_wider_int
kgen.generator @int_to_wider_int<type: dtype>() -> !pop.scalar<type> {
  // CHECK: pop.constant(#M.dense_array<4> : vector<1xui32>)
  %0 = pop.constant(4 : i8) : !pop.scalar<type>
  kgen.return %0 : !pop.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_wider_int<type: dtype = ui32>() : () -> !pop.scalar<ui32>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_same_width_int
kgen.generator @int_to_same_width_int<type: dtype>() -> !pop.scalar<type> {
  // CHECK: pop.constant(#M.dense_array<65532> : vector<1xui16>)
  %0 = pop.constant(-4 : i16) : !pop.scalar<type>
  kgen.return %0 : !pop.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_same_width_int<type: dtype = ui16>() : () -> !pop.scalar<ui16>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_shorter_int
kgen.generator @int_to_shorter_int<type: dtype>() -> !pop.scalar<type> {
  // CHECK: pop.constant(#M.dense_array<42> : vector<1xsi8>)
  %0 = pop.constant(42 : si64) : !pop.scalar<type>
  kgen.return %0 : !pop.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_shorter_int<type: dtype = si8>() : () -> !pop.scalar<si8>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_float
kgen.generator @int_to_float<type: dtype>() -> !pop.scalar<type> {
  // CHECK: pop.constant(#M.dense_array<5.{{0+}}e+02> : vector<1xf32>)
  %0 = pop.constant(500) : !pop.scalar<type>
  kgen.return %0 : !pop.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_float<type: dtype = f32>() : () -> !pop.scalar<f32>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_float
kgen.generator @int_to_float<type: dtype>() -> !pop.scalar<type> {
  // CHECK: pop.constant(#M.dense_array<-5.{{0+}}e+02> : vector<1xf32>)
  %0 = pop.constant(-500) : !pop.scalar<type>
  kgen.return %0 : !pop.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_float<type: dtype = f32>() : () -> !pop.scalar<f32>
  kgen.return
}

// -----

// CHECK-LABEL: @"float_to_int
kgen.generator @float_to_int<type: dtype>() -> !pop.scalar<type> {
  // CHECK: pop.constant(#M.dense_array<15> : vector<1xui8>)
  %0 = pop.constant(15.0) : !pop.scalar<type>
  kgen.return %0 : !pop.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @float_to_int<type: dtype = ui8>() : () -> !pop.scalar<ui8>
  kgen.return
}

// -----

// CHECK-LABEL: @"float_to_int
kgen.generator @float_to_int<type: dtype>() -> !pop.scalar<type> {
  // CHECK: pop.constant(#M.dense_array<-15> : vector<1xsi8>)
  %0 = pop.constant(-15.0) : !pop.scalar<type>
  kgen.return %0 : !pop.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @float_to_int<type: dtype = si8>() : () -> !pop.scalar<si8>
  kgen.return
}

// -----

// CHECK-LABEL: @"float_to_float
kgen.generator @float_to_float<type: dtype>() -> !pop.scalar<type> {
  // CHECK: pop.constant(#M.dense_array<1.195{{.*}}e+00> : vector<1xbf16>)
  %0 = pop.constant(1.2) : !pop.scalar<type>
  kgen.return %0 : !pop.scalar<type>
}

kgen.generator @impl() {
  %0 = kgen.call @float_to_float<type: dtype = bf16>() : () -> !pop.scalar<bf16>
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

// CHECK-LABEL: @"splat_constant,type=!pop.scalar<si32>"
// CHECK: pop.constant(#M.dense_array<1> : vector<1xsi32>) : !pop.scalar<si32>

// CHECK-LABEL: @"splat_constant,type=!pop.scalar<f32>"
// CHECK: pop.constant(#M.dense_array<1.{{0+}}e+00> : vector<1xf32>) : !pop.scalar<f32>

// CHECK-LABEL: @"splat_constant,type=!pop.simd<4, si32>"
// CHECK: pop.constant(#M.dense_array<1, 1, 1, 1> : vector<4xsi32>) : !pop.simd<4, si32>

kgen.generator @splat_constant<type: type>() -> !kgen.paramref<type> {
  %0 = pop.constant(1) : !kgen.paramref<type>
  kgen.return %0 : !kgen.paramref<type>
}

kgen.generator @impl() {
  %0 = kgen.call @splat_constant<type: type = !pop.scalar<si32>>() : () -> !pop.scalar<si32>
  %1 = kgen.call @splat_constant<type: type = !pop.scalar<f32>>() : () -> !pop.scalar<f32>
  %2 = kgen.call @splat_constant<type: type = !pop.simd<4, si32>>() : () -> !pop.simd<4, si32>
  kgen.return
}

// -----

// CHECK-LABEL: @"reify_array,dtype=f32"
kgen.generator @reify_array<dtype: dtype>() -> !pop.array<1, scalar<dtype>> {
  // CHECK-NEXT: pop.constant(#M.dense_array<1.{{0+}}e+00> : !M.array<1xf32>)
  %0 = pop.constant(#M.dense_array<1> : !M.array<1xi32>) : !pop.array<1, scalar<dtype>>
  kgen.return %0 : !pop.array<1, scalar<dtype>>
}

kgen.generator @impl() {
  %0 = kgen.call @reify_array<dtype: dtype = f32>() : () -> !pop.array<1, scalar<f32>>
  kgen.return
}

// -----

// CHECK-LABEL: @"array_constant
kgen.generator @array_constant<dtype: dtype>() {
  // CHECK: pop.global_constant(#M.dense_array<1.{{0+}}e+00, 2.{{0+}}e+00> : !M.array<2xf32>) : !pop.array<2, scalar<f32>>
  %0 = pop.global_constant(#M.dense_array<1, 2> : !M.array<2xi32>) : !pop.array<2, scalar<dtype>>
  kgen.return
}

kgen.generator @impl() {
  kgen.call @array_constant<dtype: dtype = f32>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: @"array_constant
kgen.generator @array_constant<size>() {
  // CHECK: pop.global_constant(#M.dense_array<1, 1> : !M.array<2xui64>) : !pop.array<2, scalar<ui64>>
  %0 = pop.global_constant(1 : ui64) : !pop.array<size, scalar<ui64>>
  kgen.return
}

kgen.generator @impl() {
  kgen.call @array_constant<size = 2>() : () -> ()
  kgen.return
}

// -----

// CHECK-LABEL: @reify_i1
kgen.generator @reify_i1() {
  // CHECK: pop.constant(true) : !pop.scalar<bool>
  %0 = pop.constant(1:i1) : !pop.scalar<bool>
  kgen.return
}

// -----

// CHECK-LABEL: @"int_to_bool
kgen.generator @int_to_bool<DT:dtype>() -> !pop.scalar<DT> {
  //CHECK: pop.constant(#M.dense_array<true> : vector<1xui1>) : !pop.scalar<bool>
  %0 = pop.constant(1:si32) : !pop.scalar<DT>
  kgen.return %0 : !pop.scalar<DT>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_bool<DT:dtype=bool>() : () -> !pop.scalar<bool>
  kgen.return
}

// -----

// CHECK-LABEL: @"float_to_bool
kgen.generator @float_to_bool<DT:dtype>() -> !pop.scalar<DT> {
  //CHECK: pop.constant(#M.dense_array<false> : vector<1xui1>) : !pop.scalar<bool>
  %0 = pop.constant(0.0:f32) : !pop.scalar<DT>
  kgen.return %0 : !pop.scalar<DT>
}

kgen.generator @impl() {
  %0 = kgen.call @float_to_bool<DT:dtype=bool>() : () -> !pop.scalar<bool>
  kgen.return
}
// -----

// CHECK-LABEL: @"int_to_bool_vec
kgen.generator @int_to_bool_vec<type: dtype>() -> !pop.simd<4, type> {
  // CHECK: pop.constant(#M.dense_array<true, false, false, true> : vector<4xui1>) : !pop.simd<4, bool>
  %0 = pop.constant(#M.dense_array<1, 0, -0, -3> : vector<4xsi32>) : !pop.simd<4, type>
  kgen.return %0 : !pop.simd<4, type>
}

kgen.generator @impl() {
  %0 = kgen.call @int_to_bool_vec<type: dtype = bool>() : () -> !pop.simd<4, bool>
  kgen.return
}
// -----

// CHECK-LABEL: @"float_to_bool_vec
kgen.generator @float_to_bool_vec<type: dtype>() -> !pop.simd<4, type> {
  // CHECK: pop.constant(#M.dense_array<true, false, true, true> : vector<4xui1>) : !pop.simd<4, bool>
  %0 = pop.constant(#M.dense_array<1.0, 0.0, 4.13, -3.125> : vector<4xf32>) : !pop.simd<4, type>
  kgen.return %0 : !pop.simd<4, type>
}

kgen.generator @impl() {
  %0 = kgen.call @float_to_bool_vec<type: dtype = bool>() : () -> !pop.simd<4, bool>
  kgen.return
}
// -----

// CHECK-LABEL: @"materializeConstant,C=2.5
kgen.generator @materializeConstant<C: f32>() -> !pop.scalar<f32> {
  // CHECK-NEXT: pop.constant(#M.dense_array<2.5{{.*}}> : vector<1xf32>)
  %0 = pop.constant(#kgen.param.decl.ref<"C"> : f32) : !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

kgen.generator @doIt() -> !pop.scalar<f32> {
  %0 = kgen.call @materializeConstant<C: f32 = 2.5>() : () -> !pop.scalar<f32>
  kgen.return %0 : !pop.scalar<f32>
}

// -----

kgen.func @store_load_array(%arg0: !pop.array<2, i24>) -> !pop.array<2, i24> {
  %0 = pop.stack_allocation 1 x !pop.array<2, i24>
  pop.store %arg0, %0 : !pop.pointer<array<2, i24>>
  %1 = pop.load %0 : !pop.pointer<array<2, i24>>
  kgen.return %1 : !pop.array<2, i24>
}

kgen.func @store_load_pointer(%arg0: i32) -> i32 {
  %0 = pop.stack_allocation 1 x i32
  pop.store %arg0, %0 : !pop.pointer<i32>
  %1 = pop.stack_allocation 1 x !pop.pointer<i32>
  pop.store %0, %1 : !pop.pointer<pointer<i32>>
  %2 = pop.load %1 : !pop.pointer<pointer<i32>>
  %3 = pop.load %2 : !pop.pointer<i32>
  kgen.return %3 : i32
}

// FIXME: Can't call generators from a meta context.
kgen.func @store_load_simd(%arg0: !pop.simd<2, f32>) -> !pop.simd<2, f32> {
  %0 = pop.stack_allocation 1 x !pop.simd<2, f32>
  pop.store %arg0, %0 : !pop.pointer<simd<2, f32>>
  %1 = pop.load %0 : !pop.pointer<simd<2, f32>>
  kgen.return %1 : !pop.simd<2, f32>
}

kgen.func @store_load_simd_packed(%arg0: !pop.simd<2, si4>) -> !pop.simd<2, si4> {
  %0 = pop.stack_allocation 1 x !pop.simd<2, si4>
  pop.store %arg0, %0 : !pop.pointer<simd<2, si4>>
  %1 = pop.load %0 : !pop.pointer<simd<2, si4>>
  kgen.return %1 : !pop.simd<2, si4>
}

kgen.func @store_load_simd_packed_i2(%arg0: !pop.simd<6, ui2>) -> !pop.simd<6, ui2> {
  %0 = pop.stack_allocation 1 x !pop.simd<6, ui2>
  pop.store %arg0, %0 : !pop.pointer<simd<6, ui2>>
  %1 = pop.load %0 : !pop.pointer<simd<6, ui2>>
  kgen.return %1 : !pop.simd<6, ui2>
}

kgen.func @store_load_struct(%arg0: !pop.struct<i8, i16, f64>) -> !pop.struct<i8, i16, f64> {
  %0 = pop.stack_allocation 1 x !pop.struct<i8, i16, f64>
  pop.store %arg0, %0 : !pop.pointer<struct<i8, i16, f64>>
  %1 = pop.load %0 : !pop.pointer<struct<i8, i16, f64>>
  kgen.return %1 : !pop.struct<i8, i16, f64>
}

kgen.func @i24_pair_bitcast(%arg0: !pop.array<2, i24>) -> i64 {
  %0 = pop.stack_allocation 2 x i24
  %1 = pop.pointer.bitcast %0 : !pop.pointer<i24> to !pop.pointer<array<2, i24>>
  pop.store %arg0, %1 : !pop.pointer<array<2, i24>>
  %2 = pop.pointer.bitcast %0 : !pop.pointer<i24> to !pop.pointer<i64>
  %3 = pop.load %2 : !pop.pointer<i64>
  kgen.return %3 : i64
}

kgen.func @i8_i16_i32_bitcast(%arg0: !pop.struct<i8, i16, i32>) -> i64 {
  %0 = pop.stack_allocation 1 x i64
  %1 = pop.pointer.bitcast %0 : !pop.pointer<i64> to !pop.pointer<struct<i8, i16, i32>>
  pop.store %arg0, %1 : !pop.pointer<struct<i8, i16, i32>>
  %2 = pop.load %0 : !pop.pointer<i64>
  kgen.return %2 : i64
}

kgen.func @i8_vec_bitcast(%arg0: !pop.simd<2, si8>) -> i16 {
  %0 = pop.stack_allocation 1 x !pop.simd<2, si8>
  pop.store %arg0, %0 : !pop.pointer<simd<2, si8>>
  %1 = pop.pointer.bitcast %0 : !pop.pointer<simd<2, si8>> to !pop.pointer<i16>
  %2 = pop.load %1 : !pop.pointer<i16>
  kgen.return %2 : i16
}

kgen.func @i2_vec_bitcast(%arg0: !pop.simd<4, ui2>) -> ui8 {
  %0 = pop.stack_allocation 1 x !pop.simd<4, ui2>
  pop.store %arg0, %0 : !pop.pointer<simd<4, ui2>>
  %1 = pop.pointer.bitcast %0 : !pop.pointer<simd<4, ui2>> to !pop.pointer<ui8>
  %2 = pop.load %1 : !pop.pointer<ui8>
  kgen.return %2 : ui8
}

// CHECK-LABEL: kgen.func @do_it
kgen.generator @do_it() {
  // CHECK-NEXT: #pop.array<123, 456>
  kgen.param.constant: !pop.array<2, i24> = <apply(
    :(!pop.array<2, i24>) -> !pop.array<2, i24> @store_load_array, #pop.array<123, 456>)>

  // CHECK-NEXT: <555>
  kgen.param.constant: i32 = <apply(
    :(i32) -> i32 @store_load_pointer, 555)>

  // CHECK-NEXT: #pop.simd<"1.25", "2.25">
  kgen.param.constant: !pop.simd<2, f32> = <apply(
    :(!pop.simd<2, f32>) -> !pop.simd<2, f32> @store_load_simd, #pop.simd<"1.25", "2.25">)>

  // CHECK-NEXT: #pop.simd<-7, 7>
  kgen.param.constant: !pop.simd<2, si4> = <apply(
    :(!pop.simd<2, si4>) -> !pop.simd<2, si4> @store_load_simd_packed, #pop.simd<-7, 7>)>

  // CHECK-NEXT: #pop.simd<0, 1, 2, 3, 3, 2>
  kgen.param.constant: !pop.simd<6, ui2> = <apply(
    :(!pop.simd<6, ui2>) -> !pop.simd<6, ui2> @store_load_simd_packed_i2, #pop.simd<0, 1, 2, 3, 3, 2>)>

  // CHECK-NEXT: #pop.struct<120, 32112, 1.125{{0+}}e+00>
  kgen.param.constant: !pop.struct<i8, i16, f64> = <apply(
    :(!pop.struct<i8, i16, f64>) -> !pop.struct<i8, i16, f64> @store_load_struct, #pop.struct<120, 32112, 1.125>)>

  // CHECK-NEXT: <1099511627792>
  kgen.param.constant: i64 = <apply(
    :(!pop.array<2, i24>) -> i64 @i24_pair_bitcast, #pop.array<16, 256>)>

  // CHECK-NEXT: <8590983192>
  kgen.param.constant: i64 = <apply(
    :(!pop.struct<i8, i16, i32>) -> i64 @i8_i16_i32_bitcast, #pop.struct<24, 16, 2>)>

  // CHECK-NEXT: <1026>
  kgen.param.constant: i16 = <apply(
    :(!pop.simd<2, si8>) -> i16 @i8_vec_bitcast, #pop.simd<2, 4>)>

  // CHECK-NEXT: <229>
  kgen.param.constant: ui8 = <apply(
    :(!pop.simd<4, ui2>) -> ui8 @i2_vec_bitcast, #pop.simd<1, 1, 2, 3>)>
  kgen.return
}
