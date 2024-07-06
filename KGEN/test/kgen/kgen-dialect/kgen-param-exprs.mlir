// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect -verify-parameters | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

#target = #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target

// CHECK-LABEL: kgen.generator @param_expr
kgen.generator @param_expr<p1, p2, int1: i1, int2: i1, type: dtype, type2: dtype, mlirType: type, fn: (index) -> index>()  {
  // Generic attr syntax in generic ops
  // CHECK: "test.someop"() {
  "test.someop" () {
    // CHECK-SAME: use1 = #kgen.param.expr<add, #kgen.param.decl.ref<"p1"> : index, 42 : index>
    use1 = #kgen.param.expr<add, #kgen.param.decl.ref<"p1"> : index, 42 : index> : index,
    // CHECK-SAME: use2 = #kgen.param.expr<add, #kgen.param.decl.ref<"p1"> : index, 43 : index>
    use2 = #kgen.param.expr<add, 1 : index, #kgen.param.decl.ref<"p1"> : index, 42 : index> : index,
    // CHECK-SAME: use3 = 3 : index
    use3 = #kgen.param.expr<add, 1 : index, 2 : index> : index,

    // Type folding.
    // CHECK-SAME: use4 = #kgen.param.decl.ref<"mlirType"> : !kgen.type
    use4 = #kgen.type<!kgen.paramref<:type mlirType>> : !kgen.type


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

  // CHECK: = kgen.param.constant: !kgen.int_literal = <get_sizeof(mlirType, #kgen.target<{{.*}}>)>
  %23 = kgen.param.constant: !kgen.int_literal = <get_sizeof(mlirType, #target)>

  // CHECK: = kgen.param.constant: !kgen.int_literal = <get_alignof(mlirType, #kgen.target<{{.*}}>)>
  %24 = kgen.param.constant: !kgen.int_literal = <get_alignof(mlirType, #target)>

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

  // CHECK: = kgen.param.constant = <add(mul_nuw(p2, p2), p1, 42)>
  %39 = kgen.param.constant = <add(p1, 42, mul_nuw(p2, p2))>

  // CHECK: = kgen.param.constant = <mul_nuw(mul(p2, 2), p1, 42)>
  %40 = kgen.param.constant = <mul_nuw(p1, 42, add(p2, p2))>

  // CHECK: = kgen.param.constant = <p1>
  %41 = kgen.param.constant = <add(p1, p2, mul_nuw(p2, -1))>

  // CHECK: = kgen.param.constant = <p2>
  %42 = kgen.param.constant = <add(mul(p2, 2), mul_nuw(p2, -1))>

  kgen.param.declare args: variadic<si32> = <[1, 2]>
  // CHECK: constant: si32 = <variadic_get(:variadic<si32> args, 2)>
  kgen.param.constant: si32 = <variadic_get(:variadic<si32> args, 2)>
  // CHECK: constant = <2>
  kgen.param.constant = <variadic_get(:variadic<index> [1, 2], 1)>

  // CHECK: kgen.param.constant = <cond(int1, p1, p2)>
  kgen.param.constant = <cond(int1, p1, p2)>
  // CHECK: kgen.param.constant: i1 = <0>
  kgen.param.constant: i1 = <cond(int1, 0, int1)>
  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <cond(ne(p1, p2), p1, p2)>
  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <cond(int1, p1, p1)>
  // CHECK: constant = <p1>
  kgen.param.constant = <cond(1, p1, p2)>
  // CHECK: constant = <p2>
  kgen.param.constant = <cond(0, p1, p2)>
  // CHECK: constant = <p1>
  kgen.param.constant = <cond(eq(p1, p2), p2, p1)>

  // CHECK: constant = <cond(eq(p1, 1), 4, 5)>
  kgen.param.constant = <cond(eq(p1, 1), add(p1, 3), 5)>

  // COM: Make sure both internal conditionals substitute into add(p1, p2)
  // CHECK: constant = <cond(eq(p1, 1), 4, 5)>
  kgen.param.constant = <cond(eq(p1, 1), cond(eq(p2, 3), add(p1, p2), 4), 5)>

  // CHECK: constant = <1>
  kgen.param.constant = <cond(eq(p1, 1), cond(eq(p2, 2), cond(int1, p1, 1), 1), 1)>

  // COM: This hits the depth limit of recursion (3 ops deep max) but would be <1> if raised
  // CHECK: constant = <cond(eq(p1, 1), cond(eq(p2, 2), cond(int1, cond(not(int2), 0, 1), 1), 1), 1)>
  kgen.param.constant = <cond(eq(p1, 1), cond(eq(p2, 2), cond(int1, cond(not(int2), 0, 1), 1), 1), 1)>

  // CHECK: constant: scalar<index> = <cond(int1, 1, 2)>
  kgen.param.constant: scalar<index> = <cond(int1, #pop.simd<1>, #pop.simd<2>)>

  // CHECK: declare env_test: i1 = <get_env("NDEBUG")>
  kgen.param.declare env_test: i1 = <get_env("NDEBUG")>
  // CHECK: declare env_int = <get_env("OPT_LEVEL")>
  kgen.param.declare env_int = <get_env("OPT_LEVEL")>
  // CHECK: declare env_str: string = <get_env("PROC_NAME")>
  kgen.param.declare env_str: string = <get_env("PROC_NAME")>

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

  // CHECK-NEXT: constant: i32 = <1>
  %5 = kgen.param.constant: i32 = <div(mul_nuw(p1, p2), mul_nuw(p1, p2))>

  // Division by 0 is undefined behavior.
  // CHECK-NEXT: constant: i32 = <0>
  %6 = kgen.param.constant: i32 = <div(12, 0)>

  // Folder only kicks in for constants.
  // CHECK-NEXT: constant: i32 = <div(p1, 0)>
  %7 = kgen.param.constant: i32 = <div(p1, 0)>

  // CHECK-NEXT: constant: i32 = <0>
  %8 = kgen.param.constant: i32 = <div(0, 0)>

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

  // CHECK: = kgen.param.constant = <div(mul(p1, p1, p2, 3), mul(p1, p1, 3))>
  kgen.param.constant = <div(mul(p1, p1, p2, 3), mul(p1, p1, 3))>

  // CHECK: = kgen.param.constant = <mul_nuw(p2, 3)>
  kgen.param.constant = <div(mul_nuw(p1, p1, p2, 3), mul_nuw(p1, p1))>

  // CHECK: = kgen.param.constant = <div(mul_nuw(p1, 3), p2)>
  kgen.param.constant = <div(mul_nuw(p1, p1, p2, 3), mul_nuw(p1, p2, p2))>

  // CHECK: = kgen.param.constant = <p1>
  kgen.param.constant = <div(mul_nuw(p1, p2), p2)>

  // CHECK: = kgen.param.constant = <mul_nuw(p1, 40)>
  kgen.param.constant = <div(mul_nuw(p1, 200000), 5000)>

  // CHECK: = kgen.param.constant = <div(p1, 5000)>
  kgen.param.constant = <div(p1, 5000)>

  // These are too large so the result may overflow on some devices for indices
  // 5B --> too large for 32 bit systems. This may be poisoned so we do
  // CHECK: = kgen.param.constant = <div(mul_nuw(p1, 5000000000), 5000)>
  kgen.param.constant = <div(mul_nuw(p1, 5000000000), 5000)>

  // CHECK: = kgen.param.constant = <div(p1, 10)>
  kgen.param.constant = <div(mul_nuw(50, 2, p1), 1000)>

  kgen.param.constant = <mul(p1, 1)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <mul(p1, 0, p2)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <mul_nuw(p1, 1)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <mul_nuw(p1, 0, p2)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <and(12, 6)>  // CHECK: kgen.param.constant = <4>
  kgen.param.constant = <or(12, 6)>  // CHECK: kgen.param.constant = <14>
  kgen.param.constant = <xor(4, 6)>  // CHECK: kgen.param.constant = <2>
  kgen.param.constant = <shl(p1, 2)>  // CHECK: kgen.param.constant = <mul(p1, 4)>
  kgen.param.constant = <shl(p1, 0)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <shr(p1, 0)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <div(p1, 1)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <mod(p1, 1)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <mod(p1, p1)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <mod(mul(p1, 2), p1)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <mod(mul_nuw(p1, 2), p1)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <mod(add(p1, 2), p1)>  // CHECK: kgen.param.constant = <mod(add(p1, 2), p1)>
  kgen.param.constant = <max(mul_nuw(p1, 2), mul_nuw(2, p2))>  // CHECK: kgen.param.constant = <mul_nuw(max(p1, p2), 2)>
  kgen.param.constant = <max(mul(p1, 2), mul(2, p2))>  // CHECK: kgen.param.constant = <max(mul(p1, 2), mul(p2, 2))>
  kgen.param.constant = <max(mul_nuw(p1, 2), mul_nuw(p2, 3))>  // CHECK: kgen.param.constant = <max(mul_nuw(p1, 2), mul_nuw(p2, 3))>
  kgen.param.constant = <max(mul_nuw(p1, 2), mul_nuw(p2, 4))>  // CHECK: kgen.param.constant = <max(mul_nuw(p1, 2), mul_nuw(p2, 4))>
  kgen.param.constant = <max(add(p1, 2), add(p2, 2))>  // CHECK: kgen.param.constant = <max(add(p1, 2), add(p2, 2))>

  kgen.param.declare square = <mul(p1, p1)>  // CHECK: kgen.param.declare square = <mul(p1, p1)>
  kgen.param.constant = <square>  // CHECK: kgen.param.constant = <square>

  // CHECK = <eq(p1, *?)>
  kgen.param.declare unknown: i1 = <eq(*?, p1)>
  // CHECK: = <0>
  kgen.param.declare unknownEq: i1 = <eq(:dtype *?, f32)>
  // CHECK: = <1>
  kgen.param.declare unknownEqItself: i1 = <eq(:dtype *?, *?)>
  // CHECK: = <0>
  kgen.param.declare unknownEqIndex: i1 = <eq(*?, 1)>
  // CHECK: = <1>
  kgen.param.declare unknownEqItselfIndex: i1 = <eq(*?, *?)>

  // Make sure operand deduplication happens for nested operands too
  kgen.param.declare max = <max(max(p1, 1), p1)>
  // CHECK: kgen.param.declare max = <max(p1, 1)>

  kgen.param.declare min = <min(min(p1, 1), p1)>
  // CHECK: kgen.param.declare min = <min(p1, 1)>

  kgen.return
}

// CHECK-LABEL: kgen.generator @datalayout_operators()
kgen.generator @datalayout_operators() {
  // CHECK-NEXT: <4>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(i32, #target)>
  // CHECK-NEXT: <3>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(i20, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(f64, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(index, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(!kgen.signature<() -> ()>, #target)>
  // CHECK-NEXT: <16>
  kgen.param.constant: !kgen.int_literal = <get_sizeof(!kgen.signature<() capturing -> ()>, #target)>

  // CHECK-NEXT: <4>
  kgen.param.constant: !kgen.int_literal = <get_alignof(i32, #target)>
  // CHECK-NEXT: <4>
  kgen.param.constant: !kgen.int_literal = <get_alignof(i20, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_alignof(f64, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_alignof(index, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: !kgen.int_literal = <get_alignof(!kgen.signature<() -> ()>, #target)>

  // CHECK-NEXT: <1>
  kgen.param.constant: !kgen.int_literal = <get_alignof(!pop.simd<0, f32>, #target)>

  kgen.return
}

// DTYPES
// CHECK-LABEL: kgen.generator @dtype_params<dt: dtype, f32: dtype, ui32: dtype>()
kgen.generator @dtype_params<dt: dtype, f32: dtype, ui32: dtype>() {

  // Make sure that kgen keywords are printed properly escaped.
  kgen.param.declare dt0: dtype = <*"f32">
  kgen.param.declare dt1: dtype = <*"ui32">

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
kgen.generator @type_params<dt: dtype, typeParam: type>() {
  // CHECK: assert <eq(:type typeParam, scalar<f32>)>, "f32 scalarzzz"
  kgen.param.assert <eq(:type typeParam, !pop.scalar<f32>)>, "f32 scalarzzz"
  // CHECK: kgen.param.declare ty1: type = <scalar<f32>>
  kgen.param.declare ty1: type = <scalar<f32>>

  // CHECK: kgen.param.declare ty2: type = <scalar<dt>>
  kgen.param.declare ty2: type = <scalar<dt>>

  // This op returns an SSA value whose type is specified by a type parameter.
  // CHECK: "test.someop"() : () -> !kgen.paramref<ty2>
  "test.someop"() : () -> !kgen.paramref<ty2>

  // kgen.paramref auto-folds non-parameterized types on construction.
  // CHECK: "test.someop"() : () -> !pop.scalar<f32>
  "test.someop"() : () -> !kgen.paramref<!pop.scalar<f32>>

  kgen.return
}

// STRING TYPES
// CHECK-LABEL: kgen.generator @string_params<a: string, b: string>()
kgen.generator @string_params<a: string, b: string>() {
  // CHECK: kgen.param.assert <eq(:string a, b)>, "samesies only"
  kgen.param.assert <eq(:string a, b)>, "samesies only"

  // CHECK: kgen.param.assert <in(:string a, [b, "foo"])>, "samesies or foo"
  kgen.param.assert <in(:string a, [b, "foo"])>, "samesies or foo"

  // CHECK: kgen.param.declare s1: string = <"exciting">
  kgen.param.declare s1: string = <"exciting">

  //kgen.param.declare s2: string = <concat("hello ", "world", "!!11oneone">

  kgen.return
}

// COM: TARGET TYPES

// CHECK-LABEL: kgen.generator @target_params2<t0: target>()
kgen.generator @target_params2<t0: target>() {
  // CHECK: assert <eq(:target t0, #kgen.target<triple = "triple", arch = "cpu", features = "features", data_layout = "p:32:32", simd_bit_width = 4>)>
  kgen.param.assert <eq(:target t0, #kgen.target<triple="triple", arch="cpu", features="features", data_layout="p:32:32", simd_bit_width=4>)>, "must support target!!"
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_has_feature<t0: target>()
kgen.generator @target_has_feature<t0: target>() {
  kgen.param.assert <target_has_feature(t0, "avx")>, "must support avx!"
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_is_cuda_triple<t0: target>()
kgen.generator @target_is_cuda_triple<t0: target>() {
  kgen.param.assert <eq(:string target_get_field(t0, "triple"), "nvptx64-nvidia-cuda")>, "triple must be nvptx64-nvidia-cuda"
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_is_os<t0: target>()
kgen.generator @target_is_os<t0: target>() {
  kgen.param.assert <eq(:string target_get_field(t0, "os"), "darwin")>, "os must be darwin"
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_is_little_endian<t0: target>()
kgen.generator @target_is_little_endian<t0: target>() {
  kgen.param.assert <eq(:string target_get_field(t0, "endianness"), "little")>, "target must be little endian"
  kgen.return
}


// CHECK-LABEL: kgen.generator @target_get_field()
kgen.generator @target_get_field() {
  kgen.param.assert<eq(:!kgen.int_literal #kgen.int_literal<128>, target_get_field(#target, "simd_bit_width"))>,
                    "simd_bit_width is always greater than 1"
  kgen.param.assert<eq(:string target_get_field(#target, "os"), "darwin")>,
                    "target os is darwin"
  kgen.return
}

// CHECK-LABEL: @pointer_param_ops
kgen.generator @pointer_param_ops<ptr: pointer<index, 1>>() {
  // CHECK-NEXT: constant = <load_from_mem(:pointer<index, 1> ptr)>
  kgen.param.constant = <load_from_mem(:pointer<index, 1> ptr)>
  // CHECK-NXET: constant: pointer<i1, 1> = <ptr_bitcast(:pointer<index, 1> ptr)>
  kgen.param.constant: pointer<i1, 1> = <ptr_bitcast(:pointer<index, 1> ptr)>
  kgen.return
}

// REGION TYPES
// CHECK-LABEL: kgen.generator @region_params<
kgen.generator @region_params
  // CHECK-SAME: r1: (si32) -> si32,
  <r1: <[]>(si32) -> si32,
   // This has an input and output parameter.
   // CHECK-SAME: r2: <[] -> i1>() -> (),
   r2: <[] -> i1>() -> (),
   // This uses a different parameter.
   // CHECK-SAME: r3: <dtype>() -> !pop.scalar<*(0,0)>
   r3: <dtype>() -> !pop.scalar<*(0,0)>
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
  // CHECK: kgen.call @takeUnary<:(!pop.scalar<si32>) -> !pop.scalar<si32> @doubleExample>()
  kgen.call @takeUnary<
     :(!pop.scalar<si32>) -> !pop.scalar<si32> @doubleExample>() : () -> ()

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
// CHECK-SAME: <index: type>
// CHECK-SAME: %[[ARG0:.*]]: !kgen.pointer<index>
// CHECK-SAME: %[[ARG1:.*]]: !kgen.pointer<*"index">
kgen.generator @mlir_builtin_types<*"index": type>(
  %arg0: !kgen.pointer<index>, %arg1: !kgen.pointer<*"index">
) -> (index, !kgen.paramref<*"index">) {
  // CHECK: %[[V0:.*]] = pop.load %[[ARG0]] : !kgen.pointer<index>
  %0 = pop.load %arg0 : !kgen.pointer<index>
  // CHECK: %[[V1:.*]] = pop.load %[[ARG1]] : !kgen.pointer<*"index">
  %1 = pop.load %arg1 : !kgen.pointer<*"index">
  // CHECK: return %[[V0]], %[[V1]] : index, !kgen.paramref<*"index">
  kgen.return %0, %1 : index, !kgen.paramref<*"index">
}

lit.struct.decl @A {}
lit.struct.decl @B {}

// CHECK-LABEL: @symbol_exprs
kgen.generator @symbol_exprs() {
  // CHECK: <eq(:!kgen.int_literal get_sizeof(@A, #kgen.target<{{.*}}>),
  // CHECK-SAME: get_sizeof(@B, #kgen.target<{{.*}}>))>
  %0 = kgen.param.constant: i1 = <eq(:!kgen.int_literal get_sizeof(!lit.struct<@A>, #target),
                                  get_sizeof(!lit.struct<@B>, #target))>
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
// CHECK:  kgen.call @takeFnContextualType<:type index, :() -> index @sillyFn>() : () -> index
kgen.generator @elaborateFnWithContextualType() -> index {
  %0 = kgen.call @takeFnContextualType<:type index, :()->index @sillyFn>() : () -> index
  kgen.return %0 : index
}

// CHECK-LABEL: @elaborateFnWithContextualType2()
kgen.generator @elaborateFnWithContextualType2() -> index {
  kgen.param.declare fn: <type, () -> !kgen.paramref<*(1,0)>>() -> !kgen.paramref<*(0,0)> = <@takeFnContextualType>

  // CHECK: kgen.param.declare boundFn: () -> index =
  // CHECK-SAME: <bind_signature(:<type, () -> !kgen.paramref<*(1,0)>>() -> !kgen.paramref<*(0,0)> fn, index, @sillyFn)>
  kgen.param.declare boundFn: () -> index =
    <bind_signature(:<type, () -> !kgen.paramref<*(1,0)>>() -> !kgen.paramref<*(0,0)> fn,
                    index, @sillyFn)>
  %0 = kgen.call_param[()->index: boundFn]()

  kgen.return %0 : index
}

// CHECK-LABEL: @partialBindSignature
kgen.generator @partialBindSignature() -> index {
  kgen.param.declare fn: <type, () -> !kgen.paramref<*(1,0)>>() -> !kgen.paramref<*(0,0)> = <@takeFnContextualType>

  // CHECK: kgen.param.declare partiallyBound: <() -> index>() -> index =
  // CHECK-SAME: <bind_signature(:<type, () -> !kgen.paramref<*(1,0)>>() -> !kgen.paramref<*(0,0)> fn, index, ?)>
  kgen.param.declare
    partiallyBound: <() -> index>() -> index =
      <bind_signature(:<type, () -> !kgen.paramref<*(1,0)>>() -> !kgen.paramref<*(0,0)> fn, index, ?)>
  // CHECK: kgen.call_param[() -> index: bind_signature(:<() -> index>() -> index partiallyBound, @sillyFn)]()
  %0 = kgen.call_param[() -> index: bind_signature(:<() -> index>() -> index partiallyBound, @sillyFn)]()

  kgen.return %0 : index
}

// CHECK-LABEL: @partialBindSignature2
kgen.generator @partialBindSignature2() -> index {
  // CHECK: kgen.param.declare fn: <() -> index>() -> index = <@takeFnContextualType<:type index, :() -> index ?>>
  kgen.param.declare fn: <()->index>() -> index =
    <bind_signature(:<type, ()->!kgen.paramref<*(1,0)>>() -> !kgen.paramref<*(0,0)> @takeFnContextualType, index, ?)>
  // CHECK: kgen.param.declare fullyBound: () -> index = <bind_signature(:<() -> index>() -> index fn, @sillyFn)>
  kgen.param.declare fullyBound: () -> index = <bind_signature(:<()->index>() -> index fn, @sillyFn)>

  // CHECK: kgen.call_param[() -> index: bind_signature(:<() -> index>() -> index fn, @sillyFn)]()
  %0 = kgen.call_param[()->index: bind_signature(:<()->index>()->index fn, @sillyFn)]()
  // CHECK: kgen.call_param[() -> index: fullyBound]()
  %1 = kgen.call_param[()->index : fullyBound]()

  kgen.return %0 : index
}

kgen.generator @returnParam<T: type, I>(%arg : !kgen.paramref<T>) -> !kgen.paramref<T> {
 kgen.return %arg : !kgen.paramref<T>
}

// CHECK-LABEL: @partialBindSignature3
kgen.generator @partialBindSignature3<T: type>(%arg : !kgen.paramref<T>) {
  // CHECK-NEXT: kgen.param.declare fn: <type>(!kgen.paramref<*(0,0)>) -> !kgen.paramref<*(0,0)> = <@returnParam<:type ?, 32>>
  kgen.param.declare fn: <type>(!kgen.paramref<*(0,0)>) -> !kgen.paramref<*(0,0)> =
    <bind_signature(:<type, index>(!kgen.paramref<*(0,0)>) -> !kgen.paramref<*(0,0)> @returnParam, ?, 32)>
  kgen.return
}

// CHECK-LABEL: @mlirOperationExpr
kgen.generator @mlirOperationExpr() {
  // CHECK: (index, index) -> index = <"index.add">
  kgen.param.declare indexAdd: (index, index) -> index = <"index.add">
  // CHECK: (index, index) -> i1 = <"index.cmp"{pred = #index<cmp_predicate slt>}>
  kgen.param.declare indexCmp: (index, index) -> i1 = <"index.cmp"{pred = #index<cmp_predicate slt>}>
  // CHECK: (!pop.array<2, i32>) -> i32 = <"pop.array.get"{index = 0 : index}>
  kgen.param.declare arrayGet: (!pop.array<2, i32>) -> i32 = <"pop.array.get"{index = 0 : index}>

  // CHECK: cmpResult: i1 = <1>
  kgen.param.declare cmpResult: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate eq>}, 3, 3)>
  kgen.return
}

kgen.generator @evaluator(%funcs: !kgen.pointer<!kgen.signature<() -> ()>>, %num: index) -> index {
  %0 = kgen.param.constant = <2>
  kgen.return %0 : index
}

kgen.generator @f1() {
  kgen.return
}

kgen.generator @f2() {
  kgen.return
}

lit.struct.decl @IndexParams0<a, b: f32> {}
lit.struct.decl @IndexParams1<a: i32, b: i64, c: f32, d: i8> {}

// CHECK-LABEL: kgen.generator @indexParamRef
// CHECK-SAME: @IndexParams1<:i32 *(0,0), :i64 *(0,1), :f32 *(1,1), :i8 *(1,0)*>
// CHECK-SAME: @IndexParams0<*(0,0), :f32 *(0,1)>
kgen.generator @indexParamRef<
  fn: <index, f32, <i32, i64>()
      -> !lit.struct<@IndexParams1<:i32 *(0,0), :i64 *(0,1), :f32 *(1,1), :i8 *(1,0)*>>
      -> i8>()
    -> !lit.struct<@IndexParams0<*(0,0), :f32 *(0,1)>>
>() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @partial_bind_index
kgen.generator @partial_bind_index<c>() {
  kgen.param.declare.region fn = <a, b: type>(%arg0: !pop.array<a, b>) {
    kgen.return
  }
  kgen.param.declare callable: <index, type>(!pop.array<*(0,0), *(0,1)>) -> () = <fn>
  // CHECK: declare partial_bound: <type>(!pop.array<c, *(0,0)>) -> ()
  kgen.param.declare partial_bound: <type>(!pop.array<c, *(0,0)>) -> () =
    <bind_signature(:<index, type>(!pop.array<*(0,0), *(0,1)>) -> () callable, c, ?)>
  kgen.return
}

kgen.generator @result_slot(%arg1: index, %arg0: !kgen.pointer<index> byref_result) -> !kgen.none {
  %0 = kgen.param.constant: none = <#kgen.none>
  kgen.return %0 : !kgen.none
}

// CHECK-LABEL: kgen.generator @apply_result_slot
kgen.generator @apply_result_slot() {
  // CHECK-NEXT: constant = <apply_result_slot(:(index, !kgen.pointer<index> byref_result) -> !kgen.none @result_slot, 2)>
  kgen.param.constant: index = <apply_result_slot(:(index, !kgen.pointer<index> byref_result) -> !kgen.none @result_slot, 2)>
  kgen.return
}

// CHECK-LABEL: @int_literal_param
kgen.generator @int_literal_param<abcd: !kgen.int_literal>() {
  // CHECK-NEXT: constant: !kgen.int_literal = <abcd>
  kgen.param.constant: !kgen.int_literal = <abcd>
  kgen.return
}

kgen.generator @kernel() {
  kgen.return
}

// CHECK-LABEL: @compile_assembly
kgen.generator @compile_assembly() {
  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", simd_bit_width = 128>>
  // CHECK: constant: string = <compile_assembly(nvptx, asm, 0, :() -> () @kernel)>
  kgen.param.constant: string = <compile_assembly(nvptx, asm, 0, :() -> () @kernel)>
  // CHECK: constant: string = <compile_assembly(nvptx, llvm, 1, :() -> () @kernel)>
  kgen.param.constant: string = <compile_assembly(nvptx, llvm, 1, :() -> () @kernel)>
  kgen.return
}

// CHECK-LABEL: @get_likage_name
kgen.generator @get_likage_name() {
  // CHECK: constant: string = <get_linkage_name(current_target(), :() -> () @kernel)>
  kgen.param.constant: string = <get_linkage_name(current_target(), :() -> () @kernel)>
  kgen.return
}

// CHECK-LABEL: @unification
kgen.generator @unification() {
  // CHECK: T0: type = <@unification : !metatype.type>
  kgen.param.declare T0: type = <rebind(:!metatype.type #kgen.type<!lit.struct<@unification, !metatype.type>>)>
  kgen.return
}

// CHECK-LABEL: @struct_extract
kgen.generator @struct_extract() {
  // CHECK-NEXT: <2>
  kgen.param.constant = <#kgen.struct.extract<:struct<(index, index)> { 1, 2 }, 1>>
  // CHECK-NEXT: <*?>
  kgen.param.constant = <#kgen.struct.extract<:struct<(index, index)> *?, 0>>
  kgen.return
}
