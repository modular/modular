// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.generator @param_expr
kgen.generator @param_expr<p1, p2, int1: i1, type: dtype, type2: dtype, mlirType: type>()  {
  // Generic attr syntax in generic ops
  // CHECK: "someop"() {
  "someop" () {
    // CHECK-SAME: use1 = #kgen.param.expr<add, #kgen.param.decl.ref<"p1", index> : index, 42 : index>
    use1 = #kgen.param.expr<add, #kgen.param.decl.ref<"p1", index>, 42 : index> : index,
    // CHECK-SAME: use2 = #kgen.param.expr<add, #kgen.param.decl.ref<"p1", index> : index, 43 : index>
    use2 = #kgen.param.expr<add, 1 : index, #kgen.param.decl.ref<"p1", index>, 42 : index> : index,
    // CHECK-SAME: use3 = 3 : index
    use3 = #kgen.param.expr<add, 1 : index, 2 : index> : index
  } : () -> ()
  // Generic syntax in known contexts

  // CHECK: = kgen.param.constant = <add(p1, 42)>
  %0 = kgen.param.constant = <#kgen.param.expr<add, #kgen.param.decl.ref<"p1", index>, 42 : index>>

  // CHECK: = kgen.param.constant = <add(mul(p2, p2), p1, 42)>
  %1 = kgen.param.constant = <add(p1, 42, mul(p2, p2))>

  // CHECK: = kgen.param.constant = <mul(p1, p2, 84)>
  %2 = kgen.param.constant = <mul(p1, 42, add(p2, p2))>

  // CHECK: = kgen.param.constant: i1 = <eq(p1, 42)>
  %3 = kgen.param.constant: i1 = <eq(42, p1)>

  // CHECK: = kgen.param.constant: i1 = <0>
  %4 = kgen.param.constant: i1 = <eq(41, 42)>

  // CHECK: = kgen.param.constant: i1 = <1>
  %5 = kgen.param.constant: i1 = <1>

  // CHECK: = kgen.param.constant: i1 = <eq(:dtype type, f32)>
  %6 = kgen.param.constant: i1 = <eq(:dtype type, f32)>

  // CHECK: = kgen.param.constant: i1 = <0>
  %7 = kgen.param.constant: i1 = <eq(:dtype bf16, f16)>

  // CHECK: = kgen.param.constant: i1 = <in(p1, [add(p2, 1), p2, 1, 3])>
  %8 = kgen.param.constant: i1 = <in(p1, [3, 1, p2, add(p2, 1), 1])>

  // CHECK: = kgen.param.constant: i1 = <0>
  %9 = kgen.param.constant: i1 = <in(0, [1, 2])>

  // CHECK: = kgen.param.constant: i1 = <0>
  %10 = kgen.param.constant: i1 = <in(p1, [])>

  // CHECK: = kgen.param.constant: i1 = <1>
  %11 = kgen.param.constant: i1 = <in(p1, [p1, 1])>

  // CHECK: = kgen.param.constant: i1 = <eq(p1, 1)>
  %12 = kgen.param.constant: i1 = <in(p1, [1])>

  // CHECK: = kgen.param.constant: i1 = <in(:dtype f32, [type, f64])>
  %13 = kgen.param.constant: i1 = <in(:dtype f32, [f64, type, f64, type])>

  // CHECK: = kgen.param.constant: i1 = <0>
  %14 = kgen.param.constant: i1 = <in(:dtype f32, [si64, f64])>

  // CHECK: = kgen.param.constant: i1 = <0>
  %15 = kgen.param.constant: i1 = <in(:dtype type, [])>

  // CHECK: = kgen.param.constant: i1 = <1>
  %16 = kgen.param.constant: i1 = <in(:dtype type, [type, f32])>

  // CHECK: = kgen.param.constant: i1 = <in(:dtype type, [type2, f32])>
  %17 = kgen.param.constant: i1 = <in(:dtype type, [type2, f32])>

  // CHECK: = kgen.param.constant: i1 = <eq(:dtype type, f32)>
  %18 = kgen.param.constant: i1 = <in(:dtype type, [f32])>

  // The only binary operation that signless i1 supports is xor.
  // CHECK: = kgen.param.constant: i1 = <not(int1)>
  %19 = kgen.param.constant: i1 = <xor(int1, 1)>

  // CHECK: = kgen.param.constant: i1 = <not(int1)>
  %20 = kgen.param.constant: i1 = <not(int1)>

  // CHECK: = kgen.param.constant: i1 = <ne(:dtype type, f32)>
  %21 = kgen.param.constant: i1 = <xor(eq(:dtype type, f32), 1)>

  // CHECK: = kgen.param.constant: dtype = <get_dtype(mlirType)>
  %22 = kgen.param.constant: dtype = <get_dtype(mlirType)>

  // CHECK: = kgen.param.constant: dtype = <get_dtype(mlirType)>
  %23 = kgen.param.constant: dtype = <get_dtype(!pop.scalar<get_dtype(mlirType)>)>

  // CHECK: = kgen.param.constant: dtype = <f32>
  %24 = kgen.param.constant: dtype = <get_dtype(!pop.scalar<f32>)>

  // CHECK: = kgen.param.constant: i1 = <1>
  %25 = kgen.param.constant: i1 = <le(5, 9)>

  // CHECK: = kgen.param.constant = <get_sizeof(mlirType)>
  %26 = kgen.param.constant = <get_sizeof(mlirType)>

  // CHECK: = kgen.param.constant = <get_alignof(mlirType)>
  %27 = kgen.param.constant = <get_alignof(mlirType)>

  // CHECK: = kgen.param.constant = <max(p1, 2)>
  %28 = kgen.param.constant = <max(p1, 2)>

  // CHECK: = kgen.param.constant = <4>
  %29 = kgen.param.constant = <max(-2, 4)>

  // CHECK: = kgen.param.constant = <max(p1, p2, 5)>
  %30 = kgen.param.constant = <max(4, p1, p2, 5, p1, p2)>

  // CHECK: = kgen.param.constant = <min(p1, 2)>
  %31 = kgen.param.constant = <min(p1, 2)>

  // CHECK: = kgen.param.constant = <-2>
  %32 = kgen.param.constant = <min(-2, 4)>

  // CHECK: = kgen.param.constant = <min(p1, p2, 4)>
  %33 = kgen.param.constant = <min(4, p1, p2, 5, p1, p2)>

  // CHECK: = kgen.param.constant = <-4>
  %34 = kgen.param.constant = <neg(4)>

  // CHECK: = kgen.param.constant = <-6>
  %35 = kgen.param.constant = <neg(add(2, 4))>

  // CHECK: = kgen.param.constant = <mul(p1, -1)>
  %36 = kgen.param.constant = <neg(p1)>

  // CHECK: = kgen.param.constant = <add(mul(p2, -1), p1)>
  %37 = kgen.param.constant = <sub(p1, p2)>

  // CHECK: = kgen.param.constant = <5>
  %38 = kgen.param.constant = <sub(9, 4)>

  kgen.return
}

// CHECK-LABEL: kgen.generator @int1_aliases
kgen.generator @int1_aliases<p1, p2, int1: i1, type: dtype>()  {

  // CHECK: = kgen.param.constant: i1 = <ne(:dtype type, f32)>
  %0 = kgen.param.constant: i1 = <ne(:dtype type, f32)>

  // CHECK: = kgen.param.constant: i1 = <ne(p1, 42)>
  %1 = kgen.param.constant: i1 = <ne(p1, 42)>

  // CHECK: = kgen.param.constant: i1 = <not(int1)>
  %2 = kgen.param.constant: i1 = <not(int1)>

  // CHECK: = kgen.param.constant: i1 = <ge(p1, p2)>
  %3 = kgen.param.constant: i1 = <ge(p1, p2)>

  // CHECK: = kgen.param.constant: i1 = <ge(p1, 43)>
  %4 = kgen.param.constant: i1 = <gt(p1, 42)>

  // CHECK: = kgen.param.constant: i1 = <ge(p1, 42)>
  %5 = kgen.param.constant: i1 = <ge(p1, 42)>

  // CHECK: = kgen.param.constant: i1 = <ge(p1, 4)>
  %6 = kgen.param.constant: i1 = <le(4, p1)>

  // CHECK: = kgen.param.constant: i1 = <ge(p1, 5)>
  %7 = kgen.param.constant: i1 = <lt(4, p1)>

  // Shouldn't fold `index` constant expressions that differ for 32-/64-bit
  // targets without target info.
  // CHECK: = kgen.param.constant = <div(6000000000, 4)>
  %8 = kgen.param.constant = <div(6000000000, 4)> // 6B/4 differs.

  // CHECK: = kgen.param.constant = <8589934592>
  %9 = kgen.param.constant = <shl(1, 33)>

  // CHECK: = kgen.param.constant: i1 = <not(in(p1, [add(p2, 1), p2, 1, 3]))>
  %10 = kgen.param.constant: i1 = <not_in(p1, [3, 1, p2, add(p2, 1), 1])>

  // CHECK: = kgen.param.constant: i1 = <1>
  %11 = kgen.param.constant: i1 = <not_in(0, [1, 2])>

  // CHECK: = kgen.param.constant: i1 = <1>
  %12 = kgen.param.constant: i1 = <not_in(p1, [])>

  // CHECK: = kgen.param.constant: i1 = <0>
  %13 = kgen.param.constant: i1 = <not_in(p1, [p1, 1])>

  // CHECK: = kgen.param.constant: i1 = <ne(p1, 1)>
  %14 = kgen.param.constant: i1 = <not_in(p1, [1])>

  // CHECK: = kgen.param.constant: i1 = <not(in(:dtype f32, [type, f64]))>
  %15 = kgen.param.constant: i1 = <not_in(:dtype f32, [f64, type, f64, type])>

  // CHECK: = kgen.param.constant: i1 = <1>
  %16 = kgen.param.constant: i1 = <not_in(:dtype f32, [si64, f64])>

  // CHECK: = kgen.param.constant: i1 = <1>
  %17 = kgen.param.constant: i1 = <not_in(:dtype type, [])>

  // CHECK: = kgen.param.constant: i1 = <0>
  %18 = kgen.param.constant: i1 = <not_in(:dtype type, [type, f32])>

  // CHECK: = kgen.param.constant: i1 = <ne(:dtype type, f32)>
  %19 = kgen.param.constant: i1 = <not_in(:dtype type, [f32])>

  // This can't be folded because it is target specific: true on 32-bit and
  // false on 64-bit.
  // CHECK: = kgen.param.constant: i1 = <in(0, [4294967296, 8589934592])>
  %20 = kgen.param.constant: i1 = <in(0, [shl(1, 32), shl(2, 32)])>

  kgen.return
}

// CHECK-LABEL: kgen.generator @param_canonicalize
kgen.generator @param_canonicalize<p1, p2>()  {
  // CHECK: = kgen.param.constant = <add(mul(p1, 4), mul(p2, 4))>
  kgen.param.constant = <mul(add(p1, p2), 4)>

  // CHECK: = kgen.param.constant = <add(mul(p2, p2), p1, 42)>
  kgen.param.constant = <add(p1, 42, mul(p2, p2))>

  // CHECK: = kgen.param.constant = <add(mul(p1, 3), 42)>
  kgen.param.constant = <add(p1, 42, mul(p1, 2))>

  kgen.param.constant = <mul(p1, 1)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <mul(p1, 0, p2)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <and(12, 6)>  // CHECK: kgen.param.constant = <4>
  kgen.param.constant = <or(12, 6)>  // CHECK: kgen.param.constant = <14>
  kgen.param.constant = <xor(4, 6)>  // CHECK: kgen.param.constant = <2>
  kgen.param.constant = <shl(p1, 2)>  // CHECK: kgen.param.constant = <mul(p1, 4)>
  kgen.param.constant = <shl(p1, 0)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <shr(p1, 0)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <div(p1, 1)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <mod(p1, 1)>  // CHECK: kgen.param.constant = <0>

  kgen.param.declare square = <mul(p1, p1)>  // CHECK: kgen.param.declare square = <mul(p1, p1)>
  kgen.param.constant = <square>  // CHECK: kgen.param.constant = <square>

  kgen.return
}

// CHECK-LABEL: kgen.generator @datalayout_operators()
kgen.generator @datalayout_operators() {
  // CHECK-NEXT: <4>
  %0 = kgen.param.constant = <get_sizeof(i32)>
  // CHECK-NEXT: <3>
  %1 = kgen.param.constant = <get_sizeof(i20)>
  // CHECK-NEXT: <8>
  %2 = kgen.param.constant = <get_sizeof(f64)>
  // CHECK-NEXT: <8>
  %3 = kgen.param.constant = <get_sizeof(index)>
  // CHECK-NEXT: <2>
  %4 = kgen.param.constant = <get_sizeof(vector<2xi8>)>
  // CHECK-NEXT: <4>
  %5 = kgen.param.constant = <get_sizeof(vector<3xi8>)>

  // CHECK-NEXT: <4>
  %6 = kgen.param.constant = <get_alignof(i32)>
  // CHECK-NEXT: <4>
  %7 = kgen.param.constant = <get_alignof(i20)>
  // CHECK-NEXT: <8>
  %8 = kgen.param.constant = <get_alignof(f64)>
  // CHECK-NEXT: <8>
  %9 = kgen.param.constant = <get_alignof(index)>
  // CHECK-NEXT: <4>
  %10 = kgen.param.constant = <get_alignof(vector<3xi8>)>

  kgen.return
}

// DTYPES
// CHECK-LABEL: kgen.generator @dtype_params<dt: dtype, *"f32", *"ui32">()
kgen.generator @dtype_params<dt: !kgen.dtype, *"f32", *"ui32">() {

  // Make sure that kgen keywords are printed properly escaped.
  // CHECK: kgen.param.constant = <add(*"f32", *"ui32")>
  kgen.param.constant  = <add(*"f32", *"ui32")>

  // CHECK: kgen.param.constant: dtype = <f32>
  kgen.param.constant: !kgen.dtype = <#kgen.dtype.constant<f32>>

  // CHECK: kgen.param.constant: dtype = <f32>
  kgen.param.constant: !kgen.dtype = <f32>
  // CHECK: kgen.param.constant: dtype = <ui128>
  kgen.param.constant: dtype = <ui128>
  kgen.return
}

// MLIR TYPES
// CHECK-LABEL: kgen.generator @type_params<dt: dtype, typeParam: type>()
kgen.generator @type_params<dt: dtype, typeParam: type>()
// CHECK: constraints <[eq(:type typeParam, !pop.scalar<f32>), "f32 scalarzzz", #{{.*}}]> {
   constraints <[eq(:type typeParam, !pop.scalar<f32>), "f32 scalarzzz"]>
 {
  // CHECK: kgen.param.declare ty1: type = <!pop.scalar<f32>>
  kgen.param.declare ty1: type = <!pop.scalar<f32>>

  // CHECK: kgen.param.declare ty2: type = <!pop.scalar<dt>>
  kgen.param.declare ty2: type = <!pop.scalar<dt>>

  // This op returns an SSA value whose type is specified by a type parameter.
  // CHECK: "someop"() : () -> !kgen.paramref<ty2>
  "someop"() : () -> !kgen.paramref<ty2>

  // kgen.paramref auto-folds non-parameterized types on construction.
  // CHECK: "someop"() : () -> !pop.scalar<f32>
  "someop"() : () -> !kgen.paramref<!pop.scalar<f32>>

  kgen.return
}

// STRING TYPES
// CHECK-LABEL: kgen.generator @string_params<a: string, b: string>()
kgen.generator @string_params<a: string, b: string>()
// CHECK: constraints <
// CHECK-NEXT: [eq(:string a, b), "samesies only", #loc{{.*}}],
// CHECK-NEXT: [in(:string a, [b, "foo"]), "samesies or foo", #loc{{.*}}]> {
   constraints <[eq(:string a, b), "samesies only"],
                 [in(:string a, [b, "foo"]), "samesies or foo"]>
 {
  // CHECK: kgen.param.declare s1: string = <"exciting">
  kgen.param.declare s1: string = <"exciting">

  //kgen.param.declare s2: string = <concat("hello ", "world", "!!11oneone">

  kgen.return
}


// REGION TYPES
// CHECK-LABEL: kgen.generator @region_params<
kgen.generator @region_params
  // CHECK-SAME: r1: (si32) -> si32,
  <r1: <()>(si32) -> si32,
   // This has an input and output parameter.
   // CHECK-SAME: r2: <() -> i1>() -> (),
   r2: <() -> i1>() -> (),
   // This uses a different parameter.
   // CHECK-SAME: r3: <dt: dtype>() -> !zap.buffer<4, dt>
   r3: <dt: dtype>() -> !zap.buffer<4, dt>
   >() {
  // use unaryFn
  kgen.return
}

kgen.generator @takeUnary<unaryFn: (!pop.scalar<si32>) -> !pop.scalar<si32>>() {
  // use unaryFn
  kgen.return
}

kgen.func @doubleExample(%arg0: !pop.scalar<si32>) -> !pop.scalar<si32> {
  %0 = pop.add %arg0, %arg0: !pop.scalar<si32>
  kgen.return %0 : !pop.scalar<si32>
}

kgen.generator @test_region() {
  // CHECK: kgen.call @takeUnary<
  // CHECK-SAME: unaryFn: (!pop.scalar<si32>) -> !pop.scalar<si32> = @doubleExample>()
  kgen.call @takeUnary<
     unaryFn : (!pop.scalar<si32>) -> !pop.scalar<si32> = @doubleExample>() : () -> ()

  kgen.return
}

// CHECK-LABEL: @testTargetInfo
kgen.generator @testTargetInfo() {
  // CHECK: kgen.param.constant = <"darwin-arm64-21.0">
  %0 = kgen.param.constant = <"darwin-arm64-21.0">
  kgen.return
}

// COM: Test that `index` parses to the builtin MLIR type and that `*"index"`
// COM: roundtrips as an escaped parameter name.

// CHECK-LABEL: @mlir_builtin_types
// CHECK-SAME: <*"index": type>
// CHECK-SAME: %[[ARG0:.*]]: !pop.pointer<index>
// CHECK-SAME: %[[ARG1:.*]]: !pop.pointer<*"index">
kgen.generator @mlir_builtin_types<*"index": type>(
  %arg0: !pop.pointer<index>, %arg1: !pop.pointer<*"index">
) -> (index, !kgen.paramref<*"index">) {
  // CHECK: %[[V0:.*]] = pop.load %[[ARG0]] : !pop.pointer<index>
  %0 = pop.load %arg0 : !pop.pointer<index>
  // CHECK: %[[V1:.*]] = pop.load %[[ARG1]] : !pop.pointer<*"index">
  %1 = pop.load %arg1 : !pop.pointer<*"index">
  // CHECK: return %[[V0]], %[[V1]] : index, !kgen.paramref<*"index">
  kgen.return %0, %1 : index, !kgen.paramref<*"index">
}

kgen.struct.decl @A {}
kgen.struct.decl @B {}

// CHECK-LABEL: @symbol_exprs
kgen.generator @symbol_exprs() {
  // CHECK: <add(get_sizeof(!kgen.ref<@A>), get_sizeof(!kgen.ref<@B>))>
  %0 = kgen.param.constant = <add(get_sizeof(!kgen.ref<@A>), get_sizeof(!kgen.ref<@B>))>
  kgen.return
}

kgen.generator @takeFnContextualType<ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> {
  %0 = kgen.call_param[()->!kgen.paramref<ty>: fn]()
  kgen.return %0: !kgen.paramref<ty>
}

kgen.func @sillyFn() -> index {
  %0 = kgen.param.constant = <42>
  kgen.return %0: index
}

// CHECK-LABEL:  kgen.generator public @elaborateFnWithContextualType() -> index {
// CHECK:  kgen.call @takeFnContextualType<ty: type = index, fn: () -> index = @sillyFn>() : () -> index
kgen.generator public @elaborateFnWithContextualType() -> index {
  %0 = kgen.call @takeFnContextualType<ty: type = index, fn: ()->index = @sillyFn>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: @elaborateFnWithContextualType2()
kgen.generator public @elaborateFnWithContextualType2() -> index {
  kgen.param.declare fn: <ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> = <@takeFnContextualType>

  // CHECK: kgen.param.declare boundFn: () -> index =
  // CHECK-SAME: <bind_signature(:<ty: type, fn: () -> !kgen.paramref<ty>>() -> !kgen.paramref<ty> fn, :type index, :() -> index @sillyFn)>
  kgen.param.declare boundFn: ()->index =
    <bind_signature(:<ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> fn,
                    :type index, :()->index @sillyFn)>
  %0 = kgen.call_param[()->index: boundFn]()

  kgen.return %0 : index
}
