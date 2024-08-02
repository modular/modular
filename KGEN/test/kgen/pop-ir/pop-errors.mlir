// RUN: kgen-opt %s -verify-diagnostics -split-input-file

kgen.func @pop_select_simd(
    // expected-note @below {{prior use here}}
    %arg0: !pop.scalar<bool>,
    %arg1: !pop.simd<4, si32>,
    %arg2: !pop.simd<4, si32>
  ) -> !pop.simd<4, si32> {
  // expected-error @below {{use of value '%arg0' expects different type than prior uses: '!pop.simd<4, bool>' vs '!pop.scalar<bool>'}}
  %0 = pop.simd.select %arg0, %arg1, %arg2 : !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

kgen.func @pop_select_simd(
    // expected-note @below {{prior use here}}
    %arg0: !pop.simd<8, bool>,
    %arg1: !pop.simd<4, si32>,
    %arg2: !pop.simd<4, si32>
  ) -> !pop.simd<4, si32> {
  // expected-error @below {{use of value '%arg0' expects different type than prior uses: '!pop.simd<4, bool>' vs '!pop.simd<8, bool>'}}
  %0 = pop.simd.select %arg0, %arg1, %arg2 : !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

kgen.generator @bitcast_scalar(%a: !pop.scalar<f32>) {
  // expected-error @below {{'pop.bitcast' op operand type '!pop.scalar<f32>' and result type '!pop.scalar<si8>' are cast incompatible}}
  %0 = pop.bitcast %a : !pop.scalar<f32> to !pop.scalar<si8>
  kgen.return
}

// -----

kgen.generator @bitcast_simd(%a: !pop.simd<4, f32>) {
  // expected-error @below {{'pop.bitcast' op operand type '!pop.simd<4, f32>' and result type '!pop.simd<8, f32>' are cast incompatible}}
  %0 = pop.bitcast %a : !pop.simd<4, f32> to !pop.simd<8, f32>
  kgen.return
}

// -----

kgen.generator @bitcast_simd(%a: !pop.simd<4, f32>) {
  // expected-error @below {{'pop.bitcast' op operand type '!pop.simd<4, f32>' and result type '!pop.simd<4, f64>' are cast incompatible}}
  %0 = pop.bitcast %a : !pop.simd<4, f32> to !pop.simd<4, f64>
  kgen.return
}

// -----

kgen.generator @cast_simd_size<type: dtype>(%a: !pop.simd<2, type>) {
  // expected-error @below {{are cast incompatible}}
  %0 = pop.cast %a : !pop.simd<2, type> to !pop.simd<4, type>
  kgen.return
}

// -----

kgen.generator @cast_simd_size<size, type: dtype>(%a: !pop.simd<size, type>) {
  // expected-error @below {{are cast incompatible}}
  %0 = pop.cast %a : !pop.simd<size, type> to !pop.simd<add(size, 1), type>
  kgen.return
}

// -----

kgen.generator @simd_shuffle(%a: !pop.simd<2, f32>) {
  // expected-error @below {{expected result dtype to match operand dtypes}}
  %0 = pop.simd.shuffle <2, f32> %a, %a -> <1, f64> :array<1, index> [1]
  kgen.return
}

// -----

kgen.generator @simd_shuffle<type: dtype>(%a: !pop.simd<2, f32>) {
  // expected-error @below {{expected result dtype to match operand dtypes}}
  %0 = pop.simd.shuffle <2, f32> %a, %a -> <1, type> :array<1, index> [1]
  kgen.return
}

// -----

kgen.generator @simd_shuffle<size>(%a: !pop.simd<2, f32>) {
  // expected-error @below {{mask element 4 is out of bounds}}
  %0 = pop.simd.shuffle <2, f32> %a, %a -> <1, f32> :array<1, index> [4]
  kgen.return
}

// -----

kgen.func @cast_from_builtin_type(%arg0: si32) {
  // expected-error @below {{cannot convert from scalar dtype ui32 to 'si32'}}
  %0 = pop.cast_from_builtin %arg0 : si32 to !pop.scalar<ui32>
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector(%arg0: !pop.simd<4, f32>) {
  // expected-error @below {{expected a rank 1 non-scalable vector}}
  %0 = pop.cast_to_builtin %arg0 : !pop.simd<4, f32> to f32
  kgen.return
}

// -----

kgen.generator @cast_simd_to_vector<size>(%arg0: !pop.simd<size, f32>) {
  // expected-error @below {{cannot convert   %0 = pop.cast_to_builtin %arg0 : !pop.simd<size, f32> to vector<4xi32>
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector(%arg0: !pop.simd<4, f32>) {
  // expected-error @below {{expected vector<4xT>}}
  %0 = pop.cast_to_builtin %arg0 : !pop.simd<4, f32> to vector<8xf32>
  kgen.return
}

// -----

kgen.func @simd_splat(%arg0: !pop.scalar<f32>) {
  // expected-error @below {{'pop.simd.splat' op requires a non-negative size}}
  %0 = pop.simd.splat %arg0 : !pop.simd<-1, f32>
  kgen.return
}

// -----

kgen.func @invalid_array_create(%arg0: i32) {
  // expected-error @below {{expected 2 operands to create array but got 1}}
  %0 = pop.array.create [%arg0] : !pop.array<2, i32>
  kgen.return
}

// -----

kgen.func @array_out_of_bounds(%arg0: !pop.array<1, i32>) {
  // expected-error @below {{'pop.array.get' op array index out of bounds: -1}}
  %0 = pop.array.get %arg0[-1] : !pop.array<1, i32>
  kgen.return
}

// -----

kgen.func @array_out_of_bounds(%arg0: !pop.array<1, i32>) {
  // expected-error @below {{'pop.array.get' op array index out of bounds: 2}}
  %0 = pop.array.get %arg0[2] : !pop.array<1, i32>
  kgen.return
}

// -----

kgen.func @repeat_zero() {
  // expected-error @below {{requires at least one operand to create an array whose size is non-zero}}
  %0 = pop.array.repeat [] : !pop.array<1, i32>
  kgen.return
}

// -----

kgen.func @array_repeat_crash(%arg0: index) {
  // expected-error @below {{'pop.array.repeat' op requires a non-negative size}}
  %0 = "pop.array.repeat"(%arg0) : (index) -> !pop.array<-1, index>
  kgen.return
}

// -----

kgen.generator @pack_create<Ts: variadic<!kgen.type>>(%arg0: f32, %arg1: si8) {
  // expected-error @below {{operand types cannot be inferred for resulting pack type '!kgen.pack<Ts>'}}
  %0 = kgen.pack.create(%arg0, %arg1) : !kgen.pack<Ts>
  kgen.return
}

// -----

// CHECK-LABEL: @pack_attr
kgen.generator @pack_attr<Ts: variadic<i32>>() {
  // expected-error @below {{pack attribute expected a variadic constant type, but got #kgen.param.decl.ref<"Ts"> : !kgen.variadic<type>}}
  %0 = kgen.param.constant: !kgen.pack<Ts> = <<>>
  kgen.return
}

// -----

kgen.func @struct_gep_type(%a: !kgen.pointer<struct<(i32)>>) {
  // expected-error @below {{'kgen.struct.gep' op element index 1 out of bounds (>=1)}}
  %0 = "kgen.struct.gep"(%a) { index = 1 : index } : (!kgen.pointer<struct<(i32)>>) -> !kgen.pointer<i32>
  kgen.return
}

// -----

kgen.func @struct_gep_type(%a: !kgen.pointer<struct<(i32)>>) {
  // expected-error @below {{'kgen.struct.gep' op result type 'i64' does not match struct element type at index 0: 'i32'}}
  %0 = "kgen.struct.gep"(%a) { index = 0 : index } : (!kgen.pointer<struct<(i32)>>) -> !kgen.pointer<i64>
  kgen.return
}

// -----

kgen.func @func() {
  // expected-error @below {{'kgen.global.address' op does not reference a `pop.global` operation}}
  kgen.global.address @func : <i32>
  kgen.return
}

// -----

kgen.func @global_ctor() {
  kgen.return
}

kgen.global @global_var : i32 [@global_ctor, @global_ctor](2)

kgen.func @func() {
  // expected-error @below {{'kgen.global.address' op result type does not match global type 'i32'}}
  kgen.global.address @global_var : <i64>
  kgen.return
}

// -----

kgen.func @fence() {
  // expected-error @below {{'pop.fence' op can be given only acquire, release, acq_rel, and seq_cst orderings}}
  pop.fence not_atomic
  kgen.return
}

// -----

kgen.func @fence() {
  // expected-error @below {{'pop.fence' op can be given only acquire, release, acq_rel, and seq_cst orderings}}
  pop.fence monotonic
  kgen.return
}

// -----

kgen.func @fence() {
  // expected-error @below {{'pop.fence' op can be given only acquire, release, acq_rel, and seq_cst orderings}}
  pop.fence unordered
  kgen.return
}

// -----

kgen.func @variant_bitcast_oob(%arg0: !kgen.pointer<variant<i32>>) {
  // expected-error @below {{variant index 1 is out of bounds in range [0, 1)}}
  %0 = "pop.variant.bitcast"(%arg0) {index = 1 : index} : (!kgen.pointer<variant<i32>>) -> !kgen.pointer<i32>
  kgen.return
}

// -----

kgen.func @variant_bitcast_oob(%arg0: !kgen.pointer<variant<i32>>) {
  // expected-error @below {{variant element at index 0 expected type 'i32' but result has type 'i64'}}
  %0 = "pop.variant.bitcast"(%arg0) {index = 0 : index} : (!kgen.pointer<variant<i32>>) -> !kgen.pointer<i64>
  kgen.return
}

// -----

kgen.func @variant_discr_gep_type(%arg0: !kgen.pointer<variant<i32, i64>>) {
  // expected-error @below {{variant expected discriminant bitwidth to be 8 but result returns uint with width 16}}
  %0 = pop.variant.discr_gep %arg0 : <variant<i32, i64>> as <scalar<ui16>>
  kgen.return
}

// -----

kgen.func @invalid_union() {
  // expected-error @below {{value type 'i64' is not a union element type of '!pop.union<i32>'}}
  kgen.param.constant: union<i32> = <{:i64 42}>
  kgen.return
}

// -----

kgen.func @invalid_union_bitcast(%arg0: !kgen.pointer<union<i32>>) {
  // expected-error @below {{result pointer element type 'i64' is not an element type of '!pop.union<i32>'}}
  pop.union.bitcast %arg0 : <union<i32>> as <i64>
  kgen.return
}

// -----

kgen.func @invalid_union_wrap(%arg0: i32) {
  // expected-error @below {{operand type 'i32' is not an element type of '!pop.union<i64>'}}
  %0 = pop.union.wrap %arg0 : i32 as <i64>
  kgen.return
}

// -----

kgen.func @invalid_union_unwrap(%arg0: !pop.union<i32>) {
  // expected-error @below {{result type 'i64' is not an element type of '!pop.union<i32>'}}
  %0 = pop.union.unwrap %arg0 : <i32> as i64
  kgen.return
}
