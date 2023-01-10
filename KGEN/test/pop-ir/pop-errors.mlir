// RUN: kgen-opt %s -verify-diagnostics -split-input-file

kgen.generator @pop_constant<type: type>() {
  // expected-error @below {{expected integer or float attribute for unspecified result type}}
  %0 = pop.constant(dense<0> : vector<1xi32>) : !kgen.paramref<type>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{cannot convert from attribute type 'f32' to dtype si64}}
  %0 = pop.constant(32.0 : f32) : !pop.scalar<si64>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{expected array elements attribute for vector constant with known size}}
  %0 = pop.constant(dense<0> : vector<1xi32>) : !pop.scalar<si32>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{expected attribute type to be vector<2xT>}}
  %0 = pop.constant(#M.dense_array<0.0, 0.0> : tensor<2xf32>) : !pop.simd<2, f32>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{expected attribute type to be vector<2xT>}}
  %0 = pop.constant(#M.dense_array<0, 0, 0, 0> : vector<2x2xsi32>) : !pop.simd<2, si32>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{expected attribute type to be vector<2xT>}}
  %0 = pop.constant(#M.dense_array<0> : vector<1xsi32>) : !pop.simd<2, si32>
  kgen.return
}

// -----

kgen.func @pop_constant() {
  // expected-error @below {{cannot convert from attribute type 'i32' to dtype si32}}
  %0 = pop.constant(#M.dense_array<0, 0> : vector<2xi32>) : !pop.simd<2, si32>
  kgen.return
}

// -----

kgen.generator @pop_constant<size>() {
  // expected-error @below {{expected integer or float attribute for vector constant of unspecified size}}
  %0 = pop.constant(#M.dense_array<0, 0> : vector<2xsi32>) : !pop.simd<size, si32>
  kgen.return
}

// -----

// COM: copysign is not defined on non-floating point types

kgen.func @pop_copysign(%arg0 : !pop.scalar<si32>, %arg1 : !pop.scalar<si32>) -> !pop.scalar<si32> {
  // expected-error @below {{whose element type is either unbound or a floating-point dtype}}
  %0 = pop.copysign %arg0, %arg1 : !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

// -----

// COM: copysign is not defined on non-floating point types

kgen.func @pop_copysign(%arg0 : !pop.simd<4, si32>, %arg1 : !pop.simd<4, si32>) -> !pop.simd<4, si32> {
  // expected-error @below {{whose element type is either unbound or a floating-point dtype}}
  %0 = pop.copysign %arg0, %arg1 : !pop.simd<4, si32>
  kgen.return %0 : !pop.simd<4, si32>
}

// -----

kgen.func @pop_select_simd(
    // expected-note @below {{prior use here}}
    %arg0: !pop.scalar<bool>,
    %arg1: !pop.simd<4, si32>,
    %arg2: !pop.simd<4, si32>
  ) -> !pop.simd<4, si32> {
  // expected-error @below {{use of value '%arg0' expects different type than prior uses: '!pop.simd<4, bool>' vs '!pop.scalar<bool>'}}
  %0 = pop.select %arg0, %arg1, %arg2 : !pop.simd<4, si32>
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
  %0 = pop.select %arg0, %arg1, %arg2 : !pop.simd<4, si32>
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

kgen.func @variant_visit_invalid_case(%a: !pop.variant<i32>) {
  // expected-error @below {{'pop.variant.visit' op type case 'f32' is not a possible variant type of '!pop.variant<i32>'}}
  pop.variant.visit %a : !pop.variant<i32>
  case (%v: f32) {
    pop.yield
  }
  kgen.return
}

// -----

kgen.func @variant_visit_duplicate_case(%a: !pop.variant<i32>) {
  // expected-error @below {{'pop.variant.visit' op duplicate type case 'i32'}}
  pop.variant.visit %a : !pop.variant<i32>
  case (%v: i32) {
    pop.yield
  }
  case (%v: i32) {
    pop.yield
  }
  kgen.return
}

// -----

kgen.func @variant_visit_bad_regions(%a: !pop.variant<i32, f32>) {
  // expected-error @below {{'pop.variant.visit' op expected 2 regions when all type cases are present}}
  "pop.variant.visit"(%a) ({
    pop.yield
  }) {cases = #kgen<type.array[i32, f32]>} : (!pop.variant<i32, f32>) -> ()
  kgen.return
}

// -----

kgen.func @variant_visit_bad_regions(%a: !pop.variant<i32, f32>) {
  // expected-error @below {{'pop.variant.visit' op expected 1 regions plus a default region when not all case types are present}}
  "pop.variant.visit"(%a) ({
  ^bb0(%v: i32):
    pop.yield
  }) {cases = #kgen<type.array[i32]>} : (!pop.variant<i32, f32>) -> ()
  kgen.return
}

// -----

kgen.func @variant_visit_bad_regions(%a: !pop.variant<i32>) {
  // expected-error @below {{'pop.variant.visit' op expected default region to have zero arguments}}
  "pop.variant.visit"(%a) ({
  ^bb0(%v: i32):
    pop.yield
  }) {cases = #kgen<type.array[]>} : (!pop.variant<i32>) -> ()
  kgen.return
}

// -----

kgen.func @variant_visit_bad_yield(%a: !pop.variant<i32>) {
  // expected-error @below {{'pop.variant.visit' op operand types of region #0 yield do not match result types}}
  pop.variant.visit %a : !pop.variant<i32>
  case (%v: i32) {
    // expected-note @below {{see terminator here}}
    pop.yield %v : i32
  }
  kgen.return
}

// -----

kgen.func @variant_visit_bad_arguments(%a: !pop.variant<i32>) {
  // expected-error @below {{'pop.variant.visit' op expected region #0 to have one argument}}
  "pop.variant.visit"(%a) ({
    pop.yield
  }) {cases = #kgen<type.array[i32]>} : (!pop.variant<i32>) -> ()
  kgen.return
}


// -----

kgen.func @variant_visit_bad_arguments(%a: !pop.variant<i32>) {
  // expected-error @below {{'pop.variant.visit' op expected region #0 argument type to be 'i32'}}
  "pop.variant.visit"(%a) ({
  ^bb0(%v: f32):
    pop.yield
  }) {cases = #kgen<type.array[i32]>} : (!pop.variant<i32>) -> ()
  kgen.return
}

// -----

kgen.func @variant_asm_missing_default(%a: !pop.variant<i32, f32>) {
  // expected-error @below {{'pop.variant.visit' op expected 1 regions plus a default region when not all case types are present}}
  pop.variant.visit %a : !pop.variant<i32, f32>
  case (%v: i32) {
    pop.yield
  }
  kgen.return
}

// -----

kgen.func @variant_visit_bad_terminator(%a: !pop.variant<i32>) {
  // expected-error @below {{'pop.variant.visit' op region #0 expected `pop.yield` terminator}}
  pop.variant.visit %a : !pop.variant<i32>
  default {
    // expected-note @below {{see invalid terminator here}}
    scf.yield
  }
  kgen.return
}

// -----

kgen.func @list_index_out_of_bounds(%list : !kgen.list<index[0]>) {
  // expected-error @below {{'pop.list.get' op list index out-of-range}}
  %0 = pop.list.get %list[0] : <index[0]>
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
  // expected-error @below {{'pop.struct.gep' op result type 'i64' does not match struct element type at index 0: #kgen.concretetype.constant<i32> : !kgen.mlirtype}}
  %0 = "pop.struct.gep"(%a) { index = 0 : index } : (!pop.pointer<struct<i32>>) -> !pop.pointer<i64>
  kgen.return
}

// -----

kgen.generator @partial_apply(%arg0: () -> ()) {
  // expected-error @below {{'pop.partial_apply' op expected indices to be sorted ascending}}
  "pop.partial_apply"(%arg0) {boundInputs = array<i64: 1, 0>} : (() -> ()) -> !pop.closure<() -> ()>
  kgen.return
}

// -----

kgen.generator @partial_apply(%arg0: () -> (), %arg1: i32) {
  // expected-error @below {{'pop.partial_apply' op mismatch between number of indices and inputs: 0 vs 1}}
  "pop.partial_apply"(%arg0, %arg1) {boundInputs = array<i64>} : (() -> (), i32) ->  !pop.closure<() -> ()>
  kgen.return
}

// -----

kgen.generator @partial_apply(%arg0: () -> (), %arg1: i32) {
  // expected-error @below {{'pop.partial_apply' op bound input index is out of range: 0}}
  "pop.partial_apply"(%arg0, %arg1) {boundInputs = array<i64: 0>} : (() -> (), i32) -> !pop.closure<() -> ()>
  kgen.return
}

// -----

kgen.generator @partial_apply(%arg0: (i32, i32) -> (), %arg1: i32, %arg2: i32) {
  // expected-error @below {{'pop.partial_apply' op duplicate bound input index: 0}}
  "pop.partial_apply"(%arg0, %arg1, %arg2) {boundInputs = array<i64: 0, 0>} : ((i32, i32) -> (), i32, i32) -> !pop.closure<(i32, i32) -> ()>
  kgen.return
}

// -----

kgen.generator @partial_apply(%arg0: (i32) -> (), %arg1: i64) {
  // expected-error @below {{'pop.partial_apply' op input bound to argument #0}}
  "pop.partial_apply"(%arg0, %arg1) {boundInputs = array<i64: 0>} : ((i32) -> (), i64) -> !pop.closure<() -> ()>
  kgen.return
}

// -----

kgen.generator @partial_apply(%arg0: (i16, i32, i64) -> (), %arg1: i32) {
  // expected-error @below {{'pop.partial_apply' op result signature does not match}}
  "pop.partial_apply"(%arg0, %arg1) {boundInputs = array<i64: 1>} : ((i16, i32, i64) -> (), i32) -> !pop.closure<(i32, i64) -> ()>
  kgen.return
}

// -----

kgen.generator @partial_apply_syntax(%arg0: (i8) -> ()) {
  // expected-error @below {{custom op 'pop.partial_apply' expected '?' or an operand in binding list}}
  pop.partial_apply %arg0([])
  kgen.return
}

// -----

kgen.generator @partial_apply_syntax(%arg0: !kgen.signature<[], [], (i8) -> ()>, %arg1: i8, %arg2: i8) {
  // expected-error @below {{custom op 'pop.partial_apply' there are more bound inputs than arguments}}
  pop.partial_apply %arg0(%arg1, %arg2) : (i8) -> ()
  kgen.return
}

// -----

kgen.generator @partial_apply_syntax(%arg0: !kgen.signature<[], [], (i8) -> ()>, %arg1: i8) {
  // expected-error @below {{custom op 'pop.partial_apply' expected callee type to be a function type or closure type.}}
  pop.partial_apply %arg0(%arg1) : !kgen.signature<[], [], (i8) -> ()>
  kgen.return
}

// -----

kgen.generator @call_indirect(%arg0: !kgen.signature<[], [], (i8) -> ()>, %arg1: i8) {
  // expected-error @below {{custom op 'pop.call_indirect' the callee type must be a function type or a closure type.}}
  pop.call_indirect %arg0(%arg1) : !kgen.signature<[], [], (i8) -> ()>
  kgen.return
}
