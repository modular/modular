// RUN: kgen-opt %s -verify-diagnostics -split-input-file

// -----

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
  // expected-error @below {{cannot cast between SIMD types of different sizes}}
  %0 = pop.cast %a : !pop.simd<2, type> to !pop.simd<4, type>
  kgen.return
}

// -----

kgen.generator @cast_simd_size<size, type: dtype>(%a: !pop.simd<size, type>) {
  // expected-error @below {{cannot cast between SIMD types of different sizes}}
  %0 = pop.cast %a : !pop.simd<size, type> to !pop.simd<add(size, 1), type>
  kgen.return
}

// -----

kgen.generator @simd_shuffle(%a: !pop.simd<2, f32>) {
  // expected-error @below {{expected result dtype to match operand dtypes}}
  %0 = pop.simd.shuffle <2, f32> %a, %a -> <1, f64> [1]
  kgen.return
}

// -----

kgen.generator @simd_shuffle<type: dtype>(%a: !pop.simd<2, f32>) {
  // expected-error @below {{expected result dtype to match operand dtypes}}
  %0 = pop.simd.shuffle <2, f32> %a, %a -> <1, type> [1]
  kgen.return
}

// -----

kgen.generator @simd_shuffle<size>(%a: !pop.simd<2, f32>) {
  // expected-error @below {{mask element 4 is out of bounds}}
  %0 = pop.simd.shuffle <2, f32> %a, %a -> <1, f32> [4]
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
  // expected-error @below {{cannot convert from SIMD dtype f32 to vector element 'i32'}}
  %0 = pop.cast_to_builtin %arg0 : !pop.simd<size, f32> to vector<4xi32>
  kgen.return
}

// -----

kgen.func @cast_simd_to_vector(%arg0: !pop.simd<4, f32>) {
  // expected-error @below {{expected vector<4xT>}}
  %0 = pop.cast_to_builtin %arg0 : !pop.simd<4, f32> to vector<8xf32>
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

kgen.generator @pack_create<Ts: variadic<!kgen.mlirtype>>(%arg0: f32, %arg1: si8) {
  // expected-error @below {{operand types cannot be inferred for resulting pack type '!pop.pack<Ts>'}}
  %0 = pop.pack.create(%arg0, %arg1) : !pop.pack<Ts>
  kgen.return
}

// -----

// CHECK-LABEL: @pack_attr
kgen.generator @pack_attr<Ts: variadic<i32>>() {
  // expected-error @below {{pack attribute expected a variadic constant type, but got #kgen.param.decl.ref<"Ts"> : !kgen.variadic<!kgen.mlirtype>}}
  %0 = kgen.param.constant: !pop.pack<Ts> = <<>>
  kgen.return
}

// -----

kgen.func @pack_index_negative(%pack: !pop.pack<[si8]>) {
  // expected-error @below {{pack element index must not be negative}}
  %0 = pop.pack.get %pack[-1] : <[si8]>
  kgen.return
}

// -----

kgen.generator @pack_index_negative<Ts: variadic<f32>>(%pack: !pop.pack<Ts>) {
  // expected-error @below {{pack element index must not be negative}}
  %0 = pop.pack.get %pack[-2] : <Ts> -> f32
  kgen.return
}

// -----

kgen.func @pack_index_out_of_bounds(%pack: !pop.pack<[si8, ui8]>) {
  // expected-error @below {{pack element index out of bounds}}
  %0 = pop.pack.get %pack[2] : <[si8, ui8]>
  kgen.return
}

// -----

kgen.func @struct_gep_type(%a: !pop.pointer<struct<i32>>) {
  // expected-error @below {{'pop.struct.gep' op element index 1 out of bounds (>=1)}}
  %0 = "pop.struct.gep"(%a) { index = 1 : index } : (!pop.pointer<struct<i32>>) -> !pop.pointer<i32>
  kgen.return
}

// -----

kgen.func @struct_gep_type(%a: !pop.pointer<struct<i32>>) {
  // expected-error @below {{'pop.struct.gep' op result type 'i64' does not match struct element type at index 0: 'i32'}}
  %0 = "pop.struct.gep"(%a) { index = 0 : index } : (!pop.pointer<struct<i32>>) -> !pop.pointer<i64>
  kgen.return
}

// -----

// expected-error @below {{'pop.global' op expected initializer argument to be type '!pop.pointer<i32>'}}
pop.global @global_var(2) : i32, (%arg0: i32) {
}, (%arg0: !pop.pointer<i32>) {
}

// -----

// expected-error @below {{'pop.global' op expected destructor argument to be type '!pop.pointer<i32>'}}
pop.global @global_var(2) : i32, (%arg0: !pop.pointer<i32>) {
}, (%arg0: i32) {
}

// -----

// expected-error @below {{'pop.global' op expected initializer region to have one argument}}
"pop.global"() ({
^bb0:
}, {
^bb0(%arg0: !pop.pointer<i32>):
}) {sym_name = "global_var", priority = 2 : i32, type = i32} : () -> ()

// -----

// expected-error @below {{'pop.global' op expected destructor region to have one argument}}
"pop.global"() ({
^bb0(%arg0: !pop.pointer<i32>):
}, {
^bb0:
}) {sym_name = "global_var", priority = 2 : i32, type = i32} : () -> ()

// -----

kgen.func @func() {
  // expected-error @below {{'pop.global.address' op does not reference a `pop.global` operation}}
  pop.global.address @func : <i32>
  kgen.return
}

// -----

pop.global @global_var(2) : i32, (%arg0: !pop.pointer<i32>) {}, (%arg0: !pop.pointer<i32>) {}

kgen.func @func() {
  // expected-error @below {{'pop.global.address' op result type does not match global type 'i32'}}
  pop.global.address @global_var : <i64>
  kgen.return
}
