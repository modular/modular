// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.generator @param_expr
kgen.generator @param_expr<p1, p2, int1: i1, int2: i1, type: dtype, type2: dtype, mlirType: type, fn: (index) -> index>()  {
  // Generic attr syntax in generic ops
  // CHECK: "someop"() {
  "someop" () {
    // CHECK-SAME: use1 = #kgen.param.expr<add, #kgen.param.decl.ref<"p1"> : index, 42 : index>
    use1 = #kgen.param.expr<add, #kgen.param.decl.ref<"p1"> : index, 42 : index> : index,
    // CHECK-SAME: use2 = #kgen.param.expr<add, #kgen.param.decl.ref<"p1"> : index, 43 : index>
    use2 = #kgen.param.expr<add, 1 : index, #kgen.param.decl.ref<"p1"> : index, 42 : index> : index,
    // CHECK-SAME: use3 = 3 : index
    use3 = #kgen.param.expr<add, 1 : index, 2 : index> : index,

    // Type folding.
    // CHECK-SAME: use4 = #kgen.param.decl.ref<"mlirType"> : !kgen.mlirtype
    use4 = #kgen.parameterizedtype.constant<!kgen.paramref<mlirType>> : !kgen.mlirtype


  } : () -> ()
  // Generic syntax in known contexts

  // CHECK: = kgen.param.constant = <add(p1, 42)>
  %0 = kgen.param.constant = <#kgen.param.expr<add, #kgen.param.decl.ref<"p1"> : index, 42 : index>>

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

  // CHECK: = kgen.param.constant: i1 = <1>
  %22 = kgen.param.constant: i1 = <le(5, 9)>

  // CHECK: = kgen.param.constant = <get_sizeof(mlirType, #kgen.target<{{.*}}>)>
  %23 = kgen.param.constant = <get_sizeof(mlirType, #kgen<target host>)>

  // CHECK: = kgen.param.constant = <get_alignof(mlirType, #kgen.target<{{.*}}>)>
  %24 = kgen.param.constant = <get_alignof(mlirType, #kgen<target host>)>


  // CHECK: = kgen.param.constant = <max(p1, 2)>
  %25 = kgen.param.constant = <max(p1, 2)>

  // CHECK: = kgen.param.constant = <4>
  %26 = kgen.param.constant = <max(-2, 4)>

  // CHECK: = kgen.param.constant = <max(p1, p2, 5)>
  %27 = kgen.param.constant = <max(4, p1, p2, 5, p1, p2)>

  // CHECK: = kgen.param.constant = <min(p1, 2)>
  %28 = kgen.param.constant = <min(p1, 2)>

  // CHECK: = kgen.param.constant = <-2>
  %29 = kgen.param.constant = <min(-2, 4)>

  // CHECK: = kgen.param.constant = <min(p1, p2, 4)>
  %30 = kgen.param.constant = <min(4, p1, p2, 5, p1, p2)>

  // CHECK: = kgen.param.constant = <-4>
  %31 = kgen.param.constant = <neg(4)>

  // CHECK: = kgen.param.constant = <-6>
  %32 = kgen.param.constant = <neg(add(2, 4))>

  // CHECK: = kgen.param.constant = <mul(p1, -1)>
  %33 = kgen.param.constant = <neg(p1)>

  // CHECK: = kgen.param.constant = <add(mul(p2, -1), p1)>
  %34 = kgen.param.constant = <sub(p1, p2)>

  // CHECK: = kgen.param.constant = <5>
  %35 = kgen.param.constant = <sub(9, 4)>

  // CHECK: = kgen.param.constant: i1 = <1>
  %36 = kgen.param.constant : i1 = <eq(:i1 int1, int1)>

  // CHECK: = kgen.param.constant: i1 = <eq(:i1 int1, int2)>
  %37 = kgen.param.constant : i1 = <eq(:i1 int1, int2)>

  // CHECK: = kgen.param.constant = <apply(:(index) -> index fn, p1)>
  %38 = kgen.param.constant = <apply(:(index) -> index fn, p1)>

  kgen.return
}

// CHECK-LABEL: @fixed_width_integers
kgen.generator @fixed_width_integers<p1: i32, p2: i32>() {
  // CHECK-NEXT: constant: i32 = <add(p1, p2)>
  %0 = kgen.param.constant: i32 = <add(p1, p2)>

  // CHECK-NEXT: constant: i32 = <11>
  %1 = kgen.param.constant: i32 = <add(5, 6)>

  // CHECK-NEXT: constant: i32 = <div(p2, p1)>
  %2 = kgen.param.constant: i32 = <div(p2, p1)>

  // CHECK-NEXT: constant: i32 = <2>
  %3 = kgen.param.constant: i32 = <div(12, 5)>

  // CHECK-NEXT: constant: i1 = <lt(:i32 p2, p1)>
  %4 = kgen.param.constant: i1 = <lt(:i32 p2, p1)>

  kgen.return
}

// CHECK-LABEL: @signed_unsigned_integers
kgen.generator @signed_unsigned_integers<ps: si8, pu: ui8>() {
  // CHECK-NEXT: constant: ui8 = <255>
  %0 = kgen.param.constant: ui8 = <max(pu, 255)>

  // CHECK-NEXT: constant: si8 = <127>
  %1 = kgen.param.constant: si8 = <max(pu, 127)>

  // CHECK-NEXT: constant: ui8 = <0>
  %2 = kgen.param.constant: ui8 = <min(pu, 0)>

  // CHECK-NEXT: constant: si8 = <-128>
  %3 = kgen.param.constant: si8 = <min(pu, -128)>

  // CHECK-NEXT: constant: ui8 = <5>
  %4 = kgen.param.constant: ui8 = <min(250, 5)>

  // CHECK-NEXT: constant: si8 = <-5>
  %5 = kgen.param.constant: si8 = <min(-5, 5)>

  // CHECK-NEXT: constant: ui8 = <250>
  %6 = kgen.param.constant: ui8 = <max(250, 5)>

  // CHECK-NEXT: constant: si8 = <5>
  %7 = kgen.param.constant: si8 = <max(-5, 5)>

  kgen.return
}

// CHECK-LABEL: @eq_compare_anything
kgen.generator @eq_compare_anything() {
  // CHECK-NEXT: <1>
  %0 = kgen.param.constant: i1 = <eq(:f32 1.5, 1.5)>

  // CHECK-NEXT: <0>
  %1 = kgen.param.constant: i1 = <ne(:f32 1.5, 1.5)>
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

  // CHECK: = kgen.param.constant = <get_list_element(:list<index[2]> [1, 2], p1)>
  %22 = kgen.param.constant = <get_list_element(:list<index[2]> [1, 2], p1)>

  kgen.return
}

// CHECK-LABEL: kgen.generator @param_canonicalize
kgen.generator @param_canonicalize<p1, p2>() {
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

  // CHECK: = <3>
  kgen.param.constant = <apply(:(index) -> index #kgen.expr.func<(A) -> add(A, 1)>, 2)>

  // CHECK: fn1: (index) -> index = <#kgen.expr.func<(B) -> add(B, 1)>>
  kgen.param.declare fn1: (index) -> index = <bind_signature(:<A>(index) -> index #kgen.expr.func<(B) -> add(A, B)>, 1)>

  // CHECK: fn2: (index) -> index = <bind_signature(:<A>(index) -> index #kgen.expr.func<(B) -> add(A, B)>, p1)>
  kgen.param.declare fn2: (index) -> index = <bind_signature(:<A>(index) -> index #kgen.expr.func<(B) -> add(A, B)>, p1)>

  // CHECK = <eq(p1, ?)>
  kgen.param.declare unknown: i1 = <eq(?, p1)>
  // CHECK: = <0>
  kgen.param.declare unknownEq: i1 = <eq(:dtype ?, f32)>
  // CHECK: = <1>
  kgen.param.declare unknownEqItself: i1 = <eq(:dtype ?, ?)>
  // CHECK: = <0>
  kgen.param.declare unknownEqIndex: i1 = <eq(?, 1)>
  // CHECK: = <1>
  kgen.param.declare unknownEqItselfIndex: i1 = <eq(?, ?)>

  // CHECK: <eq(:() -> index fn3, #kgen.expr.func<() -> 0>)>
  kgen.param.declare fn3: () -> index = <#kgen.expr.func<() -> 1>>
  kgen.param.declare compareFns: i1 = <eq(:() -> index fn3, #kgen.expr.func<() -> 0>)>

  // CHECK: <eq(:list<index[2]> list, [1, 2])>
  kgen.param.declare list: list<index[2]> = <[3, 4]>
  kgen.param.declare compareLists: i1 = <eq(:list<index[2]> [1, 2], list)>

  // CHECK: = <2>
  kgen.param.constant = <get_list_element(:list<index[2]> [1, 2], 1)>

  // CHECK: = <get_list_element(:list<index[1]> [1], 1)>
  kgen.param.constant = <get_list_element(:list<index[1]> [1], 1)>

  kgen.return
}

// CHECK-LABEL: kgen.generator @datalayout_operators()
kgen.generator @datalayout_operators() {
  // CHECK-NEXT: <4>
  %0 = kgen.param.constant = <get_sizeof(i32, #kgen<target host>)>
  // CHECK-NEXT: <3>
  %1 = kgen.param.constant = <get_sizeof(i20, #kgen<target host>)>
  // CHECK-NEXT: <8>
  %2 = kgen.param.constant = <get_sizeof(f64, #kgen<target host>)>
  // CHECK-NEXT: <8>
  %3 = kgen.param.constant = <get_sizeof(index, #kgen<target host>)>

  // CHECK-NEXT: <4>
  %4 = kgen.param.constant = <get_alignof(i32, #kgen<target host>)>
  // CHECK-NEXT: <4>
  %5 = kgen.param.constant = <get_alignof(i20, #kgen<target host>)>
  // CHECK-NEXT: <8>
  %6 = kgen.param.constant = <get_alignof(f64, #kgen<target host>)>
  // CHECK-NEXT: <8>
  %7 = kgen.param.constant = <get_alignof(index, #kgen<target host>)>

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

// TARGET TYPES
// CHECK-LABEL: kgen.generator @target_params<t0: target, t1: target>()
kgen.generator @target_params<t0: target, t1: target>()
  // CHECK: constraints <[target_eq(:target t0, t1),
  constraints <[target_eq(:target t0, t1), "must support target!!"]> {
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_params2<t0: target>()
kgen.generator @target_params2<t0: target>()
  // CHECK: constraints <[target_eq(:target t0, #kgen.target<triple = "triple", cpu = "cpu", features = "features", pointer_bit_width = 24, simd_bit_width = 4>),
  constraints <[target_eq(:target t0, #kgen.target<triple="triple", cpu="cpu", features="features", pointer_bit_width=24, simd_bit_width=4>), "must support target!!"]> {
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_host<t0: target>()
kgen.generator @target_host<t0: target>()
  // CHECK: constraints <[target_eq(:target #kgen.target<triple = "{{.*}}", cpu = "{{.*}}", features = "{{.*}}", pointer_bit_width = {{[0-9]+}}, simd_bit_width = {{[0-9]+}}>
  constraints <[target_eq(:target #kgen<target host>, t0), "must support target!!"]> {
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_has_feature<t0: target>()
kgen.generator @target_has_feature<t0: target>()
  constraints <[target_has_feature(t0, "avx"), "must support avx!"]> {
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_is_os<t0: target>()
kgen.generator @target_is_os<t0: target>()
  constraints <[eq(:string target_get_field(t0, "os"), "darwin"), "os must be darwin"]> {
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_is_arch<t0: target>()
kgen.generator @target_is_arch<t0: target>()
  constraints <[target_is_arch(t0, "apple-m1"), "machine must be apple m1"]> {
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_get_field()
kgen.generator @target_get_field() {
  kgen.param.assert<le(1, target_get_field(#kgen<target host>,
                                           "simd_bit_width"))>,
                    "simd_bit_width is always greater than 1"
  kgen.param.assert<eq(:string target_get_field(#kgen<target host>, "os"), "darwin")>,
                    "target os is darwin"
  kgen.return
}

// CHECK-LABEL: kgen.generator @build_param<b: build_info>()
kgen.generator @build_param<b: build_info>() {
  kgen.return
}


// CHECK-LABEL: kgen.generator @build_info_is_debug()
kgen.generator @build_info_is_debug() {
  kgen.param.assert<
    eq(:string build_info_get_field(#kgen<build_info host>, "type"), "debug")>,
    "build is debug"
  kgen.return
}

// CHECK-LABEL: kgen.generator @build_info_profile_level_2()
kgen.generator @build_info_profile_level_2() {
  kgen.param.assert<
    eq(build_info_get_field(#kgen<build_info host>,
                            "llcl_max_profiling_level"), 2)>,
    "build is using llcl profile level 2"
  kgen.return
}

// REGION TYPES
// CHECK-LABEL: kgen.generator @region_params<
kgen.generator @region_params
  // CHECK-SAME: r1: (si32) -> si32,
  <r1: <()>(si32) -> si32,
   // This has an input and output parameter.
   // CHECK-SAME: r2: <() -> result: i1>() -> (),
   r2: <() -> result: i1>() -> (),
   // This uses a different parameter.
   // CHECK-SAME: r3: <dt: dtype>() -> !pop.scalar<dt>
   r3: <dt: dtype>() -> !pop.scalar<dt>
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

lit.struct.decl @A {}
lit.struct.decl @B {}

// CHECK-LABEL: @symbol_exprs
kgen.generator @symbol_exprs() {
  // CHECK: <add(get_sizeof(!kgen.declref<@A>, #kgen.target<{{.*}}>), get_sizeof(!kgen.declref<@B>, #kgen.target<{{.*}}>))>
  %0 = kgen.param.constant = <add(get_sizeof(!kgen.declref<@A>, #kgen<target host>), get_sizeof(!kgen.declref<@B>, #kgen<target host>))>
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

// CHECK-LABEL:  kgen.generator @elaborateFnWithContextualType() -> index {
// CHECK:  kgen.call @takeFnContextualType<ty: type = index, fn: () -> index = @sillyFn>() : () -> index
kgen.generator @elaborateFnWithContextualType() -> index {
  %0 = kgen.call @takeFnContextualType<ty: type = index, fn: ()->index = @sillyFn>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: @elaborateFnWithContextualType2()
kgen.generator @elaborateFnWithContextualType2() -> index {
  kgen.param.declare fn: <ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> = <@takeFnContextualType>

  // CHECK: kgen.param.declare boundFn: () -> index =
  // CHECK-SAME: <bind_signature(:<ty: type, fn: () -> !kgen.paramref<ty>>() -> !kgen.paramref<ty> fn, index, @sillyFn)>
  kgen.param.declare boundFn: ()->index =
    <bind_signature(:<ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> fn,
                    index, @sillyFn)>
  %0 = kgen.call_param[()->index: boundFn]()

  kgen.return %0 : index
}

// CHECK-LABEL: @partialBindSignature
kgen.generator @partialBindSignature() -> index {
  kgen.param.declare fn: <ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> = <@takeFnContextualType>

  // CHECK: kgen.param.declare partiallyBound: <fn: () -> index>() -> index =
  // CHECK-SAME: <bind_signature(:<ty: type, fn: () -> !kgen.paramref<ty>>() -> !kgen.paramref<ty> fn, index, #kgen.unbound)>
  kgen.param.declare
    partiallyBound: <fn: ()->index>()->index =
      <bind_signature(:<ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> fn, index, #kgen.unbound)>
  // CHECK: kgen.call_param[() -> index: bind_signature(:<fn: () -> index>() -> index partiallyBound, @sillyFn)]()
  %0 = kgen.call_param[()->index: bind_signature(:<fn: ()->index>()->index partiallyBound, @sillyFn)]()

  kgen.return %0 : index
}

// CHECK-LABEL: @partialBindSignature2
kgen.generator @partialBindSignature2() -> index {
  // CHECK: kgen.param.declare fn: <fn: () -> index>() -> index = <@takeFnContextualType<ty: type = index, fn: () -> index = #kgen.unbound>>
  kgen.param.declare fn: <fn: ()->index>() -> index =
    <bind_signature(:<ty: type, fn: ()->!kgen.paramref<ty>>() -> !kgen.paramref<ty> @takeFnContextualType, index, #kgen.unbound)>
  // CHECK: kgen.param.declare fullyBound: () -> index = <bind_signature(:<fn: () -> index>() -> index fn, @sillyFn)>
  kgen.param.declare fullyBound: () -> index = <bind_signature(:<fn: ()->index>() -> index fn, @sillyFn)>

  // CHECK: kgen.call_param[() -> index: bind_signature(:<fn: () -> index>() -> index fn, @sillyFn)]()
  %0 = kgen.call_param[()->index: bind_signature(:<fn: ()->index>()->index fn, @sillyFn)]()
  // CHECK: kgen.call_param[() -> index: fullyBound]()
  %1 = kgen.call_param[()->index : fullyBound]()

  kgen.return %0 : index
}

kgen.generator @returnParam<T: type, I>(%arg : !kgen.paramref<T>) -> !kgen.paramref<T> {
 kgen.return %arg : !kgen.paramref<T>
}

// CHECK-LABEL: @partialBindSignature3
kgen.generator @partialBindSignature3<T: type>(%arg : !kgen.paramref<T>) {
 // CHECK-NEXT: kgen.param.declare fn: <T: type>(!kgen.paramref<T>) -> !kgen.paramref<T> = <@returnParam<T: type = #kgen.unbound, I = 32>>
 kgen.param.declare fn: <T: type>(!kgen.paramref<T>) -> !kgen.paramref<T> = <bind_signature(:<T: type, I>(!kgen.paramref<T>) -> !kgen.paramref<T> @returnParam, #kgen.unbound, 32)>
 kgen.return
}

// CHECK-LABEL: @mlirOperationExpr
kgen.generator @mlirOperationExpr() {
  // CHECK: (index, index) -> index = <#kgen.param.mlir_op<"index.add", {}>>
  kgen.param.declare indexAdd: (index, index) -> index =
    <#kgen.param.mlir_op<"index.add", {}>>
  // CHECK: (index, index) -> i1 = <#kgen.param.mlir_op<"index.cmp", {pred = #index<cmp_predicate slt>}>>
  kgen.param.declare indexCmp: (index, index) -> i1 =
    <#kgen.param.mlir_op<"index.cmp", {pred = #index<cmp_predicate slt>}>>
  // CHECK: <*"index">(!pop.array<2, i32>) -> i32 = <#kgen.param.mlir_op<"pop.array.get", {}>>
  kgen.param.declare arrayGet: <*"index">(!pop.array<2, i32>) -> i32 =
    <#kgen.param.mlir_op<"pop.array.get", {}>>
  // CHECK: <size, type: type>(!pop.array<size, type>) -> !kgen.paramref<type> = <#kgen.param.mlir_op<"pop.array.get", {index = 2 : index}>>
  kgen.param.declare arrayGetParam: <size, type: type>(!pop.array<size, type>) -> !kgen.paramref<type> =
    <#kgen.param.mlir_op<"pop.array.get", {index = 2 : index}>>

  // CHECK: (!pop.array<4, i8>) -> i8 = <#kgen.param.mlir_op<"pop.array.get", {{.*}}index = 0 : index}>
  kgen.param.declare boundOp: (!pop.array<4, i8>) -> i8 = <
    bind_signature(
      :<*"index", _size, _type: type>(!pop.array<_size, _type>) -> !kgen.paramref<_type> #kgen.param.mlir_op<"pop.array.get", {}>,
      0, 4, i8)
  >

  // CHECK: cmpResult: i1 = <1>
  kgen.param.declare cmpResult: i1 = <apply(
    :(index, index) -> i1 #kgen.param.mlir_op<"index.cmp", {pred = #index<cmp_predicate eq>}>, 3, 3)>
  kgen.return
}

kgen.generator @evaluator(%funcs: !pop.pointer<!kgen.signature<() -> ()>>, %num: index) -> index {
  %0 = kgen.param.constant = <2>
  kgen.return %0 : index
}

kgen.generator @f1() {
  kgen.return
}

kgen.generator @f2() {
  kgen.return
}

// CHECK-LABEL: @itf
kgen.generator @itf() {
  // CHECK-NEXT: chosenImpl
  // CHECK-SAME: evaluate(:() -> () @f1, @f2,
  // CHECK-SAME:          :(!pop.pointer<!kgen.signature<() -> ()>>, index) -> index @evaluator
  kgen.param.declare chosenImpl : () -> () = <evaluate(:() -> () @f1, @f2, :(!pop.pointer<!kgen.signature<() -> ()>>, index) -> index @evaluator)>
  kgen.call_param[()->(): chosenImpl]()
  kgen.return
}
