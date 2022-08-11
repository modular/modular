// RUN: kgen-opt -allow-unregistered-dialect %s -verify-diagnostics -split-input-file -o /dev/null

kgen.generator @test() {
  // expected-error @+1 {{invalid use of parameter with no declaration "p"}}
  "someop" () {
    attr = #kgen.param.decl.ref<p> : i1
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @test<p1>() {
  // expected-note @-1 {{previous declaration here}}
  "someop" () { // expected-error {{redeclaration of parameter "p1"}}
    paramDecls = #kgen<param.decls[p1 : i4]>
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @test<p1>() {
  // expected-error @+1 {{invalid ParamDeclAttr outside of paramDecls attribute}}
  "someop" () {
    notParamDecls = #kgen<param.decl p1: i4>
  } : () -> ()
  kgen.return
}

// -----

kgen.generator @parameter_results<p1 -> r1: i4>() {
  // expected-error @+1 {{parameter #0 is named "r7" but should be "r1"}}
  kgen.return<r7: i4 = 7>
}

// -----

"someop" () {
  use1 = #kgen.param.expr<add,
  // expected-error @+2 {{failed to parse ParamOperatorAttr parameter}}
  // expected-error @+1 {{parameter reference requires a type}}
                          #kgen.param.decl.ref<"p1">, 42 : si64>
} : () -> ()

// -----

"someop" () {
  // expected-error @+1 {{shl must have two operands}}
  use1 = #kgen.param.expr<shl, 1 : si32, 2 : si32, 3 : si32>
} : () -> ()

// -----

"someop" () {
  // expected-error @+1 {{operand type mismatch}}
  use1 = #kgen.param.expr<shl, 1 : si32, 2 : ui32>
} : () -> ()

// -----

// expected-error @+1 {{'kgen.param.value' shl must have two operands}}
%0 = kgen.param.value = <shl(p1, p2, p3)>

// -----

// expected-error @+1 {{'kgen.param.value' unknown expression invalid_op}}
%0 = kgen.param.value = <invalid_op(p1, p2, p3)>

// -----

// expected-error @+1 {{operator requires an index type}}
%0 = kgen.param.value : i32 = <shl(1, 2)>

// -----

// expected-error @+1 {{integer literal not valid for specified type}}
kgen.param.value : !kgen.dtype = <mul(1, 4)>

// -----

// expected-error @+1 {{kgen.dtype.constant requires !kgen.dtype type}}
kgen.param.value : i8 = <#kgen.dtype.constant<f32>>

// -----

// expected-error @+1 {{kgen.dtype.constant requires !kgen.dtype type}}
kgen.param.value : i8 = <f32>

// -----

kgen.generator @scalar_params_verbose<n>(%x :
// expected-error @+1 {{expected '!kgen.dtype', but got 'index'}}
           !meta.scalar<#kgen.param.decl.ref<"n"> : index>) {
  kgen.return
}

// -----

// expected-error @+1 {{invalid use of parameter with no declaration "abc"}}
kgen.generator @scalar_params_verbose(%x : !meta.scalar<abc>) {
  kgen.return
}

// -----

kgen.generator @dtype_params() {
  // expected-error @+1 {{invalid use of parameter with no declaration "type"}}
  %y = "someop" () {} : () -> !meta.scalar<type>
  kgen.return
 }

// -----

// expected-note @+2 {{parameter defined with type 'ui32'}}
// expected-error @+1 {{reference to parameter "n" with incorrect type 'index'}}
kgen.generator @scalar_params_verbose<n : ui32>(%x : !meta.buffer<n, f32>) {
  kgen.return
}

// -----

// expected-error @+1 {{'@undefined' does not reference a valid callee}}
kgen.call @undefined() : () -> ()

// -----

kgen.generator @g1(%x : i32) {
  // expected-error @+1 {{caller has 1 input but callee expects 0}}
  kgen.call @g2(%x) : (i32) -> ()
  kgen.return
}
kgen.generator @g2<>() { // expected-note {{callee declared here}}
  kgen.return
}

// -----

// expected-error @+1 {{kgen.kernel only allows output parameters, not input parameters}}
kgen.kernel @bad_kernel_param<x>() {
  kgen.return
}

// -----

kgen.generator @g1(%x : i32) {
  // expected-error @+1 {{expected '('}}
  kgen.call @g2<()> : (i32) -> ()
  kgen.return
}
kgen.generator @g2<()>() {
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @only_returns<p1 -> p2>() {
  kgen.return <p2 = p1>
}

kgen.kernel @test_only_returns() {
  // expected-error @+1 {{caller has 0 input parameters but callee expects 1}}
  kgen.call @only_returns<()->p2>() : () -> ()
  kgen.return
}

// -----

kgen.kernel @meta_buffer_size(%arg0: i32) -> index {
  // expected-error @+1 {{'meta.buffer.size' op operand #0 must be parameterized buffer type, but got 'i32'}}
  %0 = meta.buffer.size %arg0 : i32
  kgen.return %0 : index
}
// -----

kgen.kernel @bad_cast_size(%arg0 : !meta.buffer<42, f32>) -> !meta.buffer<1, f32> {
  // expected-error @+1 {{'meta.buffer.cast' op expected the size of the input buffer (42 : index) to be equal to the size you are casting it to (1 : index), or one of them to be unknown.}}
  %0 = meta.buffer.cast %arg0 : !meta.buffer<42, f32> to !meta.buffer<1, f32>
  kgen.return %0 : !meta.buffer<1, f32>
}

// -----
kgen.kernel @bad_cast_dtype(%arg0 : !meta.buffer<42, f32>) -> !meta.buffer<42, f64> {
  // expected-error @+1 {{'meta.buffer.cast' op expected the dtype of the input buffer (#kgen.dtype.constant<f32> : !kgen.dtype) to the same as to the dtype you are casting to (#kgen.dtype.constant<f64> : !kgen.dtype), or one of them to be unknown.}}
  %0 = meta.buffer.cast %arg0 : !meta.buffer<42, f32> to !meta.buffer<42, f64>
  kgen.return %0 : !meta.buffer<42, f64>
}

// -----

kgen.kernel @meta_buffer_dtype(%arg0: i32) -> !kgen.dtype {
  // expected-error @+1 {{'meta.buffer.dtype' op operand #0 must be parameterized buffer type, but got 'i32'}}
  %0 = meta.buffer.dtype %arg0 : i32
  kgen.return %0 : !kgen.dtype
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @only_returns<() -> p2: i4>() {
  kgen.return <p2: i4 = 2>
}

kgen.kernel @test_only_returns() {
  // expected-error @+1 {{caller output parameter #0 has type 'index' but callee expected type 'i4'}}
  kgen.call @only_returns<()->p2>() : () -> ()
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @fn<p2>() {
  kgen.return
}

kgen.kernel @input_param_name() {
  // expected-error @+1 {{caller input parameter #0 has name "p1" but callee expected name "p2"}}
  kgen.call @fn<p1 = 42>() : () -> ()
  kgen.return
}

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @fn(%a: i1) -> i1 {
  kgen.return %a : i1
}

kgen.kernel @result_type(%a: i1) {
  // expected-error @+1 {{caller result #0 has type 'f32' but callee expected type 'i1'}}
  kgen.call @fn(%a) : (i1) -> f32
  kgen.return
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf<size>(si32) -> si32

// expected-error @+1 {{generator has 2 input parameters but interface expects 1}}
kgen.generator @bad<size, size2>(%arg0: si32) -> si32
  implements @itf {
  kgen.return %arg0 : si32
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf<size>(si32, si32) -> si32

// expected-error @+1 {{generator argument #0 has type 'ui32' but interface expected type 'si32'}}
kgen.generator @bad<size>(%arg0: ui32, %arg1: si32) -> si32
  implements @itf {
  kgen.return %arg1 : si32
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf<size>(si32) -> si32

// expected-error @+1 {{generator has 0 input parameters but interface expects 1}}
kgen.generator @bad<() -> size>(%arg0: si32) -> si32 implements @itf {
  kgen.return<size = 42> %arg0 : si32
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf<size>()

// expected-error @+1 {{generator input parameter #0 has name "barf" but interface expected name "size"}}
kgen.generator @bad<barf>() implements @itf {
  kgen.return
}

// -----

// expected-note @+1 {{interface declared here}}
kgen.generator.interface @itf<() -> result>(si32)

// expected-error @+1 {{generator result parameter #0 has name "size" but interface expected name "result"}}
kgen.generator @bad<() -> size: i8>(%arg0: si32) implements @itf {
  kgen.return<size:i8 = 42>
}

// -----

// expected-error @+1 {{failed to satisfy constraint: TypedAttr instance valid parameter expression}}
%0 = kgen.param.value : i32 = <[:i32]>

// -----

kgen.generator.interface @take_and_return<p1 -> r1>()

// expected-error @+1 {{invalid cyclic reference between operations defining and using parameters}}
kgen.kernel @self_cyclic() {
  // Uses r1 and defines r1
  kgen.call @take_and_return<p1 = r1 -> r1>() : () -> ()
  // expected-note @-1 {{this operation uses parameter "r1", which is defined by itself}}
  kgen.return
}

// -----

kgen.generator.interface @take_and_return<p1 -> r1>()

// expected-error @+1 {{invalid cyclic reference between operations defining and using parameters}}
kgen.kernel @mutually_recursive() {
  // Uses r2 and defines r1
  kgen.call @take_and_return<p1 = r2 -> r1>() : () -> ()
  // expected-note @-1 {{this operation uses parameter "r2", which is defined by the first operation}}

  // Uses r1 and defines r2
  kgen.call @take_and_return<p1 = r1 -> r2>() : () -> ()
  // expected-note @-1 {{this operation uses parameter "r1", which is defined by:}}

  kgen.return
}

// -----

kgen.kernel @cast_from_builtin_type(%arg0: !meta.scalar<f32>) {
  // expected-error @+1 {{'meta.cast_to_builtin' op result #0 must be integer or floating-point or vector of any type values, but got '!meta.scalar<f32>'}}
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<f32> to !meta.scalar<f32>
  kgen.return
}

// -----

kgen.kernel @cast_from_builtin_type(%arg0: !meta.scalar<f32>) {
  // expected-error @+1 {{'meta.cast_to_builtin' op does not support casting '!meta.scalar<f32>' to 'i8'}}
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<f32> to i8
  kgen.return
}

// -----

kgen.kernel @cast_from_builtin_type(%arg0: !meta.scalar<f32>) {
  // expected-error @+1 {{'meta.cast_to_builtin' op does not support casting '!meta.scalar<f32>' to 'f64'}}
  %0 = meta.cast_to_builtin %arg0: !meta.scalar<f32> to f64
  kgen.return
}

// -----

kgen.kernel @cast_simd_to_vector(%arg0: !meta.simd<4, f32>) {
  // expected-error @+1 {{does not support casting '!meta.simd<4, f32>' to 'f32'}}
  %0 = meta.cast_to_builtin %arg0 : !meta.simd<4, f32> to f32
  kgen.return
}

// -----

kgen.kernel @cast_simd_to_vector(%arg0: !meta.simd<4, f32>) {
  // expected-error @+1 {{vector type should not be scalable}}
  %0 = meta.cast_to_builtin %arg0 : !meta.simd<4, f32> to vector<[4]xf32>
  kgen.return
}

// -----

kgen.kernel @cast_simd_to_vector(%arg0: !meta.simd<4, f32>) {
  // expected-error @+1 {{expected a rank 1 vector}}
  %0 = meta.cast_to_builtin %arg0 : !meta.simd<4, f32> to vector<4x4xf32>
  kgen.return
}

// -----

kgen.kernel @cast_simd_to_vector(%arg0: !meta.simd<4, f32>) {
  // expected-error @+1 {{dimensions do not match}}
  %0 = meta.cast_to_builtin %arg0 : !meta.simd<4, f32> to vector<5xf32>
  kgen.return
}

// -----

kgen.kernel @cast_simd_to_vector(%arg0: !meta.simd<4, f32>) {
  // expected-error @+1 {{element types do not match}}
  %0 = meta.cast_to_builtin %arg0 : !meta.simd<4, f32> to vector<4xf64>
  kgen.return
}

// -----

kgen.kernel @cast_from_meta_type(%arg0: !meta.scalar<f32>) {
  // expected-error @+1 {{'meta.cast_from_builtin' op operand #0 must be integer or floating-point or vector of any type values, but got '!meta.scalar<f32>'}}
  %0 = meta.cast_from_builtin %arg0: !meta.scalar<f32> to !meta.scalar<f32>
  kgen.return
}

// -----

kgen.kernel @cast_from_meta_type(%arg0: f64) {
  // expected-error @+1 {{'meta.cast_from_builtin' op does not support casting 'f64' to '!meta.scalar<f32>'}}
  %0 = meta.cast_from_builtin %arg0: f64 to !meta.scalar<f32>
  kgen.return
}

// -----

// expected-error @+1 {{invalid use of parameter with no declaration "he1ght"}}
kgen.generator @constrained<width, height>()
  constraints <eq(width, 42), "thing", eq(he1ght, 42), "other"> {
  kgen.return
}

// -----

// expected-error @+1 {{invalid use of parameter with no declaration "he1ght"}}
kgen.generator.interface @constrained<width, height>()
  constraints <eq(width, 42), "width", eq(he1ght, 42), "height">

// -----

// expected-error @+1 {{invalid use of parameter with no declaration "ty2"}}
kgen.generator.interface @badTypes<ty1 : dtype>(%a : !meta.scalar<ty2>)

// -----

// expected-note @+1 {{callee declared here}}
kgen.generator @callee<type: dtype>(%x: !meta.scalar<type>) {
  kgen.return
}

kgen.generator @caller<type : dtype>(%arg0: !meta.scalar<type>) {
  // expected-error @+1 {{caller input #0 has type '!meta.scalar<type>' but callee expected type '!meta.scalar<f64>'}}
  kgen.call @callee<type: dtype = f64>(%arg0) : (!meta.scalar<type>) -> ()
  kgen.return
}
