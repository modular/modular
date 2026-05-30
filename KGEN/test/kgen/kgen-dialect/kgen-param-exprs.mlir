// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect -verify-parameters | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

#target = #kgen.target<triple="", arch="", features="", data_layout="", simd_bit_width=128> : !kgen.target

// CHECK-LABEL: kgen.generator @param_expr
kgen.generator @param_expr<p1, p2, int1: scalar<bool>, int2: scalar<bool>, p1_scalar : scalar<index>, p2_scalar : scalar<index>, type: dtype, type2: dtype, mlirType: type, fn: (index) -> index>()  {
  // Generic attr syntax in generic ops
  // CHECK: "test.someop"() {
  "test.someop" () {
    // CHECK-SAME: use1 = #kgen.cast_to_builtin<#kgen.param.expr<add, #kgen.cast_from_builtin<#kgen.param.decl.ref<"p1"> : index> : !kgen.scalar<index>, #kgen<simd 42> : !kgen.scalar<index>> : !kgen.scalar<index>> : index
    use1 = #kgen.param.expr<add, #kgen.param.decl.ref<"p1"> : index, 42 : index> : index,
    // CHECK-SAME: use2 = #kgen.cast_to_builtin<#kgen.param.expr<add, #kgen.cast_from_builtin<#kgen.param.decl.ref<"p1"> : index> : !kgen.scalar<index>, #kgen<simd 43> : !kgen.scalar<index>> : !kgen.scalar<index>> : index
    use2 = #kgen.param.expr<add, 1 : index, #kgen.param.decl.ref<"p1"> : index, 42 : index> : index,
    // CHECK-SAME: use3 = 3 : index
    use3 = #kgen.param.expr<add, 1 : index, 2 : index> : index,

    // Type folding.
    // CHECK-SAME: use4 = #kgen.param.decl.ref<"mlirType"> : !kgen.type
    use4 = #kgen.type<!kgen.param<:type mlirType>> : !kgen.type


  } : () -> ()
  // Generic syntax in known contexts

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> add(from_builtin(p1), 42))>
  %0 = kgen.param.constant = <#kgen.param.expr<add, #kgen.param.decl.ref<"p1"> : index, 42 : index>>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> add(mul(from_builtin(p2), from_builtin(p2)), from_builtin(p1), 42))>
  %1 = kgen.param.constant = <add(p1, 42, mul(p2, p2))>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> mul(from_builtin(p1), from_builtin(p2), 84))>
  %2 = kgen.param.constant = <mul(p1, 42, add(p2, p2))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(:scalar<index> from_builtin(p1), 42))>
  %3 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(42, p1))>

  // CHECK: = kgen.param.constant: i1 = <0>
  %4 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(41, 42))>

  // CHECK: = kgen.param.constant: i1 = <1>
  %5 = kgen.param.constant: i1 = <1>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(:dtype type, f32))>
  %6 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(:dtype type, f32))>

  // CHECK: = kgen.param.constant: i1 = <0>
  %7 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(:dtype bf16, f16))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(:scalar<index> from_builtin(p1), [add(from_builtin(p2), 1), from_builtin(p2), 3, 1]))>
  %8 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(p1, [3, 1, p2, add(p2, 1), 1]))>

  // CHECK: = kgen.param.constant: i1 = <0>
  %9 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(0, [1, 2]))>

  // CHECK: = kgen.param.constant: i1 = <0>
  %10 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(p1, []))>

  // CHECK: = kgen.param.constant: i1 = <1>
  %11 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(p1, [p1, 1]))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(:scalar<index> from_builtin(p1), 1))>
  %12 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(p1, [1]))>

  // CHECK: <index>i1 = <#kgen.gen<to_builtin(:scalar<bool> eq(:type simd<*(0,0), f32>, simd<2, f32>))>>
  kgen.param.constant: !kgen.generator<<index>i1> = <
    #kgen.gen<to_builtin(:scalar<bool> eq(:type
      #kgen.type<!kgen.simd<*(0,0), f32>>,
      #kgen.type<!kgen.simd<2, f32>>
    ))>>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(:dtype f32, [type, f64]))>
  %13 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(:dtype f32, [f64, type, f64, type]))>

  // CHECK: = kgen.param.constant: i1 = <0>
  %14 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(:dtype f32, [si64, f64]))>

  // CHECK: = kgen.param.constant: i1 = <0>
  %15 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(:dtype type, []))>

  // CHECK: = kgen.param.constant: i1 = <1>
  %16 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(:dtype type, [type, f32]))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(:dtype type, [type2, f32]))>
  %17 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(:dtype type, [type2, f32]))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(:dtype type, f32))>
  %18 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(:dtype type, [f32]))>

  // CHECK: = kgen.param.constant: scalar<bool> = <not(int1)>
  %19 = kgen.param.constant: scalar<bool> = <xor(int1, true)>

  // CHECK-NEXT: = kgen.param.constant: scalar<bool> = <not(int1)>
  %20 = kgen.param.constant: scalar<bool> = <not(int1)>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ne(:dtype type, f32))>
  %21 = kgen.param.constant: i1 = <xor(to_builtin(:scalar<bool> eq(:dtype type, f32)), 1)>

  // CHECK: = kgen.param.constant: i1 = <1>
  %22 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> le(5, 9))>

  // CHECK: = kgen.param.constant = <get_sizeof(mlirType, #kgen.target
  %23 = kgen.param.constant = <get_sizeof(mlirType, #target)>

  // CHECK: = kgen.param.constant = <get_alignof(mlirType, #kgen.target
  %24 = kgen.param.constant = <get_alignof(mlirType, #target)>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> max(from_builtin(p1), 2))>
  %25 = kgen.param.constant = <max(p1, 2)>

  // CHECK: = kgen.param.constant = <4>
  %26 = kgen.param.constant = <max(-2, 4)>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> max(from_builtin(p1), from_builtin(p2), 5))>
  %27 = kgen.param.constant = <max(4, p1, p2, 5, p1, p2)>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> min(from_builtin(p1), 2))>
  %28 = kgen.param.constant = <min(p1, 2)>

  // CHECK: = kgen.param.constant = <-2>
  %29 = kgen.param.constant = <min(-2, 4)>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> min(from_builtin(p1), from_builtin(p2), 4))>
  %30 = kgen.param.constant = <min(4, p1, p2, 5, p1, p2)>

  // CHECK: = kgen.param.constant = <-4>
  %31 = kgen.param.constant = <neg(4)>

  // CHECK: = kgen.param.constant = <-6>
  %32 = kgen.param.constant = <neg(add(2, 4))>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> mul(from_builtin(p1), -1))>
  %33 = kgen.param.constant = <neg(p1)>

  // CHECK: = kgen.param.constant: si64 = <-15>
  kgen.param.constant : si64 = <neg(15)>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> add(mul(from_builtin(p2), -1), from_builtin(p1)))>
  %34 = kgen.param.constant = <sub(p1, p2)>

  // CHECK: = kgen.param.constant = <5>
  %35 = kgen.param.constant = <sub(9, 4)>

  // CHECK: kgen.param.constant: si64 = <5>
  kgen.param.constant : si64 = <sub(9, 4)>

  // CHECK: = kgen.param.constant: scalar<bool> = <true>
  %36 = kgen.param.constant : scalar<bool> = <eq(:scalar<bool> int1, int1)>

  // CHECK: = kgen.param.constant: scalar<bool> = <eq(:scalar<bool> int1, int2)>
  %37 = kgen.param.constant : scalar<bool> = <eq(:scalar<bool> int1, int2)>

  // CHECK: = kgen.param.constant = <apply(:(index) -> index fn, p1)>
  %38 = kgen.param.constant = <apply(:(index) -> index fn, p1)>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> add(mul_no_wrap(from_builtin(p2), from_builtin(p2)), from_builtin(p1), 42))>
  %39 = kgen.param.constant = <add(p1, 42, mul_no_wrap(p2, p2))>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> mul_no_wrap(mul(from_builtin(p2), 2), from_builtin(p1), 42))>
  %40 = kgen.param.constant = <mul_no_wrap(p1, 42, add(p2, p2))>

  // CHECK: = kgen.param.constant = <p1>
  %41 = kgen.param.constant = <add(p1, p2, mul_no_wrap(p2, -1))>

  // CHECK: = kgen.param.constant = <p2>
  %42 = kgen.param.constant = <add(mul(p2, 2), mul_no_wrap(p2, -1))>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> add(mul_no_wrap(from_builtin(p1), 37919), -37919))>
  kgen.param.constant = <mul_no_wrap(add(p1, -1), 37919)>

  kgen.param.declare args: param_list<si32> = <[1, 2]>
  // CHECK: constant: si32 = <#kgen.param_list.get<:param_list<si32> args, 2>>
  kgen.param.constant: si32 = <#kgen.param_list.get<:!kgen.param_list<si32> args, 2>>
  // CHECK: constant = <2>
  kgen.param.constant = <#kgen.param_list.get<:!kgen.param_list<index> [1, 2], 1>>

  // CHECK: kgen.param.constant = <cond(int1, p1, p2)>
  kgen.param.constant = <cond(int1, p1, p2)>
  // CHECK: kgen.param.constant: scalar<bool> = <false>
  kgen.param.constant: scalar<bool> = <cond(int1, false, int1)>
  // CHECK: kgen.param.constant: scalar<index> = <p1_scalar>
  kgen.param.constant :scalar<index> = <cond(ne(:scalar<index> p1_scalar, p2_scalar), p1_scalar, p2_scalar)>
  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <cond(int1, p1, p1)>
  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <cond(true, p1, p2)>
  // CHECK: kgen.param.constant = <p2>
  kgen.param.constant = <cond(false, p1, p2)>
  // CHECK: kgen.param.constant: scalar<index> = <p1_scalar>
  kgen.param.constant :scalar<index> = <cond(eq(:scalar<index> p1_scalar, p2_scalar), p2_scalar, p1_scalar)>

  // CHECK: kgen.param.constant: scalar<index> = <cond(eq(:scalar<index> p1_scalar, 1), 4, 5)>
  kgen.param.constant :scalar<index> = <cond(eq(:scalar<index> p1_scalar, 1), add(:scalar<index> p1_scalar, 3), 5)>

  // COM: Make sure both internal conditionals substitute into add(p1, p2)
  // CHECK: kgen.param.constant: scalar<index> = <cond(eq(:scalar<index> p1_scalar, 1), 4, 5)>
  kgen.param.constant :scalar<index> = <cond(eq(:scalar<index> p1_scalar, 1), cond(eq(:scalar<index> p2_scalar, 3), add(:scalar<index> p1_scalar, p2_scalar), 4), 5)>

  // CHECK: kgen.param.constant: scalar<index> = <1>
  kgen.param.constant :scalar<index> = <cond(eq(:scalar<index> p1_scalar, 1), cond(eq(:scalar<index> p2_scalar, 2), cond(int1, p1_scalar, 1), 1), 1)>

  // COM: This hits the depth limit of recursion (3 ops deep max) but would be <1> if raised
  // CHECK: kgen.param.constant: scalar<index> = <cond(eq(:scalar<index> p1_scalar, 1), cond(eq(:scalar<index> p2_scalar, 2), cond(int1, cond(not(int2), 0, 1), 1), 1), 1)>
  kgen.param.constant:scalar<index> = <cond(eq(:scalar<index> p1_scalar, 1), cond(eq(:scalar<index> p2_scalar, 2), cond(int1, cond(not(int2), 0, 1), 1), 1), 1)>

  // CHECK: constant: scalar<index> = <cond(int1, 1, 2)>
  kgen.param.constant: scalar<index> = <cond(int1, #kgen.simd<1>, #kgen.simd<2>)>

  // CHECK: declare env_test: i1 = <get_env("NDEBUG")>
  kgen.param.declare env_test: i1 = <get_env("NDEBUG")>
  // CHECK: declare env_int = <get_env("OPT_LEVEL")>
  kgen.param.declare env_int = <get_env("OPT_LEVEL")>
  // CHECK: declare env_str: string = <get_env("PROC_NAME")>
  kgen.param.declare env_str: string = <get_env("PROC_NAME")>

  // CHECK: declare concat_str: string = <"hello world">
  kgen.param.declare concat_str: string = <str_concat("hello ", "world")>

  // CHECK: constant: param_list<type> = <[index, f32]>
  kgen.param.constant: param_list<!kgen.type> = <function_get_arg_types(:type (index,f32)->())>

  kgen.return
}

// CHECK-LABEL: @uindex_print_parse
kgen.generator @uindex_print_parse() -> (!kgen.scalar<uindex>, !kgen.scalar<uindex>) {
  // CHECK-DAG: scalar<uindex> = <18446744073709551615>
  %0 = kgen.param.constant: scalar<uindex> = <18446744073709551615>
  // CHECK-DAG: scalar<uindex> = <18446744073709551614>
  %1 = kgen.param.constant: scalar<uindex> = <-2>
  kgen.return %0, %1 : !kgen.scalar<uindex>, !kgen.scalar<uindex>
}


// CHECK-LABEL: @cast_explicit_type_print_parse
kgen.generator @cast_explicit_type_print_parse<p: index, v: !kgen.simd<1, ui32>>() {
  // Inferred matches stored: no explicit type printed.
  // CHECK: kgen.param.constant: scalar<index> = <from_builtin(p)>
  %0 = kgen.param.constant: !kgen.scalar<index> = <from_builtin(:index p) : !kgen.scalar<index>>

  // Otherwise, need an explicit result type.
  // CHECK: kgen.param.constant: scalar<uindex> = <from_builtin(p) : scalar<uindex>>
  %1 = kgen.param.constant: !kgen.scalar<uindex> = <from_builtin(:index p) : !kgen.scalar<uindex>>
  // CHECK: kgen.param.constant: i32 = <to_builtin(:scalar<ui32> v) : i32>
  %2 = kgen.param.constant: i32 = <to_builtin(:!kgen.simd<1, ui32> v) : i32>

  kgen.return
}

lit.struct.decl @StructType0<a: index, b: index> {}
lit.struct.decl @StructType1<a: index, b: index> {}

kgen.struct.generator @StructTypeGen0<a: index, b: index> = !kgen.struct<(index, index)>
kgen.struct.generator @StructTypeGen1<a: index, b: index> = !kgen.struct<(index, index)>

// CHECK-LABEL: @param_expr_type_equality
kgen.generator @param_expr_type_equality<p1>() {
  // COM: Trivial pointer equality.
  // CHECK-NEXT: = kgen.param.constant: i1 = <1>
  kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(
    :!kgen.type #kgen.type<!lit.struct<@StructType0<:index 1, :index p1>>>,
    #kgen.type<!lit.struct<@StructType0<:index 1, :index p1>>>
  ))>

  // COM: Different struct type references.
  // CHECK-NEXT: = kgen.param.constant: i1 = <0>
  kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(
    :!kgen.type #kgen.type<!lit.struct<@StructType0<:index 1, :index p1>>>,
    #kgen.type<!lit.struct<@StructType1<:index 1, :index p1>>>
  ))>

  // COM: One struct type reference and one non-struct type reference.
  // CHECK-NEXT: = kgen.param.constant: i1 = <0>
  kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(
    :!kgen.type #kgen.type<!lit.struct<@StructType0<:index 1, :index p1>>>,
    #kgen.type<index>
  ))>

  // COM: Different struct generator references.
  // CHECK-NEXT: = kgen.param.constant: i1 = <0>
  kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(
    :!kgen.type [typevalue<#kgen.genref<@StructTypeGen0<1, p1>>>, !kgen.struct<(index, index)>],
    [typevalue<#kgen.genref<@StructTypeGen1<1, p1>>>, !kgen.struct<(index, index)>]
  ))>

  // COM: One struct generator reference and one non-struct generator reference.
  // CHECK-NEXT: = kgen.param.constant: i1 = <0>
  kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(
    :!kgen.type [typevalue<#kgen.genref<@StructTypeGen0<1, p1>>>, !kgen.struct<(index, index)>],
    #kgen.type<index>
  ))>
  kgen.return
}

// CHECK-LABEL: @fixed_width_integers
kgen.generator @fixed_width_integers<p1: si32, p2: si32>() {
  // CHECK-NEXT: constant: si32 = <to_builtin(:scalar<si32> add(from_builtin(:si32 p1), from_builtin(:si32 p2)))>
  %0 = kgen.param.constant: si32 = <add(p1, p2)>

  // CHECK-NEXT: constant: si32 = <11>
  %1 = kgen.param.constant: si32 = <add(5, 6)>

  // CHECK-NEXT: constant: si32 = <to_builtin(:scalar<si32> div(from_builtin(:si32 p2), from_builtin(:si32 p1)))>
  %2 = kgen.param.constant: si32 = <div(p2, p1)>

  // CHECK-NEXT: constant: si32 = <2>
  %3 = kgen.param.constant: si32 = <div(12, 5)>

  // CHECK-NEXT: constant: i1 = <to_builtin(:scalar<bool> lt(:scalar<si32> from_builtin(:si32 p2), from_builtin(:si32 p1)))>
  %4 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> lt(:si32 p2, p1))>

  // CHECK-NEXT: constant: si32 = <1>
  %5 = kgen.param.constant: si32 = <div(mul_no_wrap(p1, p2), mul_no_wrap(p1, p2))>

  // Division by 0 is undefined behavior.
  // CHECK-NEXT: constant: si32 = <to_builtin(:scalar<si32> div(12, 0))>
  %6 = kgen.param.constant: si32 = <div(12, 0)>

  // Folder only kicks in for constants.
  // CHECK-NEXT: constant: si32 = <to_builtin(:scalar<si32> div(from_builtin(:si32 p1), 0))>
  %7 = kgen.param.constant: si32 = <div(p1, 0)>

  // CHECK-NEXT: constant: si32 = <to_builtin(:scalar<si32> div(0, 0))>
  %8 = kgen.param.constant: si32 = <div(0, 0)>

  // CHECK-NEXT: constant: si32 = <5>
  %9 = kgen.param.constant: si32 = <div(:si32 10, 2)>

  // CHECK-NEXT: constant: si32 = <-5>
  %10 = kgen.param.constant: si32 = <div(:si32 -10, 2)>

  // CHECK-NEXT: constant: ui32 = <5>
  %11 = kgen.param.constant: ui32 = <div(:ui32 10, 2)>

  // CHECK-NEXT: constant: ui32 = <2147483647>
  %12 = kgen.param.constant: ui32 = <div(:ui32 4294967295, 2)>

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
  %0 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(:f32 1.5, 1.5))>

  // CHECK-NEXT: <0>
  %1 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ne(:f32 1.5, 1.5))>
  kgen.return
}

// CHECK-LABEL: @eq_compare_sub_elements
kgen.generator @eq_compare_sub_elements<a: !kgen.param_list<index>, b: !kgen.param_list<index>, x: index, y: index>() {
  // CHECK-NEXT: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> and(eq(:scalar<index> from_builtin(#kgen.param_list.size<:param_list<index> a>), 2), eq(:scalar<index> from_builtin(#kgen.param_list.size<:param_list<index> b>), 2)))>
  kgen.param.constant: i1 = <and(
    to_builtin(:scalar<bool> eq(:index #kgen.param_list.size<:!kgen.param_list<index> b>, 2)),
    to_builtin(:scalar<bool> eq(:index #kgen.param_list.size<:!kgen.param_list<index> a>, 2)),
    to_builtin(:scalar<bool> eq(:index #kgen.param_list.size<:!kgen.param_list<index> b>, 2))
  )>

  kgen.return
}

// CHECK-LABEL: kgen.generator @int1_aliases
kgen.generator @int1_aliases<p1, p2, int1: i1, type: dtype>()  {

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ne(:dtype type, f32))>
  %0 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ne(:dtype type, f32))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ne(:scalar<index> from_builtin(p1), 42))>
  %1 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ne(p1, 42))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not(from_builtin(:i1 int1)))>
  %2 = kgen.param.constant: i1 = <not(int1)>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ge(:scalar<index> from_builtin(p1), from_builtin(p2)))>
  %3 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ge(p1, p2))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ge(:scalar<index> from_builtin(p1), 43))>
  %4 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> gt(p1, 42))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ge(:scalar<index> from_builtin(p1), 42))>
  %5 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ge(p1, 42))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ge(:scalar<index> from_builtin(p1), 4))>
  %6 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> le(4, p1))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ge(:scalar<index> from_builtin(p1), 5))>
  %7 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> lt(4, p1))>

  // Shouldn't fold `index` constant expressions that differ for 32-/64-bit
  // targets without target info.
  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> div(6000000000, 4))>
  %8 = kgen.param.constant = <div(6000000000, 4)> // 6B/4 differs.

  // CHECK: = kgen.param.constant = <8589934592>
  %9 = kgen.param.constant = <shl(1, 33)>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not(in(:scalar<index> from_builtin(p1), [add(from_builtin(p2), 1), from_builtin(p2), 3, 1])))>
  %10 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not_in(p1, [3, 1, p2, add(p2, 1), 1]))>

  // CHECK: = kgen.param.constant: i1 = <1>
  %11 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not_in(0, [1, 2]))>

  // CHECK: = kgen.param.constant: i1 = <1>
  %12 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not_in(p1, []))>

  // CHECK: = kgen.param.constant: i1 = <0>
  %13 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not_in(p1, [p1, 1]))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ne(:scalar<index> from_builtin(p1), 1))>
  %14 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not_in(p1, [1]))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not(in(:dtype f32, [type, f64])))>
  %15 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not_in(:dtype f32, [f64, type, f64, type]))>

  // CHECK: = kgen.param.constant: i1 = <1>
  %16 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not_in(:dtype f32, [si64, f64]))>

  // CHECK: = kgen.param.constant: i1 = <1>
  %17 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not_in(:dtype type, []))>

  // CHECK: = kgen.param.constant: i1 = <0>
  %18 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not_in(:dtype type, [type, f32]))>

  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> ne(:dtype type, f32))>
  %19 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> not_in(:dtype type, [f32]))>

  // This can't be folded because it is target specific: true on 32-bit and
  // false on 64-bit.
  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(:scalar<index> 0, [4294967296, 8589934592]))>
  %20 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> in(0, [shl(1, 32), shl(2, 32)]))>

  kgen.return
}

// CHECK-LABEL: kgen.generator @param_canonicalize
kgen.generator @param_canonicalize<p1, p2>() {
  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> add(mul(from_builtin(p1), 4), mul(from_builtin(p2), 4)))>
  kgen.param.constant = <mul(add(p1, p2), 4)>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> add(mul(from_builtin(p2), from_builtin(p2)), from_builtin(p1), 42))>
  kgen.param.constant = <add(p1, 42, mul(p2, p2))>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> add(mul(from_builtin(p1), 3), 42))>
  kgen.param.constant = <add(p1, 42, mul(p1, 2))>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> div(mul(from_builtin(p1), from_builtin(p1), from_builtin(p2), 3), mul(from_builtin(p1), from_builtin(p1), 3)))>
  kgen.param.constant = <div(mul(p1, p1, p2, 3), mul(p1, p1, 3))>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> mul_no_wrap(from_builtin(p2), 3))>
  kgen.param.constant = <div(mul_no_wrap(p1, p1, p2, 3), mul_no_wrap(p1, p1))>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> div(mul_no_wrap(from_builtin(p1), 3), from_builtin(p2)))>
  kgen.param.constant = <div(mul_no_wrap(p1, p1, p2, 3), mul_no_wrap(p1, p2, p2))>

  // CHECK: = kgen.param.constant = <p1>
  kgen.param.constant = <div(mul_no_wrap(p1, p2), p2)>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> mul_no_wrap(from_builtin(p1), 40))>
  kgen.param.constant = <div(mul_no_wrap(p1, 200000), 5000)>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> div(from_builtin(p1), 5000))>
  kgen.param.constant = <div(p1, 5000)>

  // These are too large so the result may overflow on some devices for indices
  // 5B --> too large for 32 bit systems. This may be poisoned so we do
  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> div(mul_no_wrap(from_builtin(p1), 5000000000), 5000))>
  kgen.param.constant = <div(mul_no_wrap(p1, 5000000000), 5000)>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> div(from_builtin(p1), 10))>
  kgen.param.constant = <div(mul_no_wrap(50, 2, p1), 1000)>

  // CHECK: = kgen.param.constant = <p1>
  kgen.param.constant = <div(mul_no_wrap(p1, -1), -1)>

  // Division distributes across addition when the denominator cleanly divides
  // every term (GEX-3582).
  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> add(mul_no_wrap(from_builtin(p1), from_builtin(p2)), from_builtin(p1)))>
  kgen.param.constant = <div(add(mul_no_wrap(p1, p2, 1536), mul_no_wrap(p1, 1536)), 1536)>

  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> add(from_builtin(p1), from_builtin(p2)))>
  kgen.param.constant = <div(add(mul_no_wrap(p1, 4), mul_no_wrap(p2, 4)), 4)>

  // Partial cancellation via GCD: (6*p1 + 4*p2) / 2 -> 3*p1 + 2*p2.
  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> add(mul_no_wrap(from_builtin(p1), 3), mul_no_wrap(from_builtin(p2), 2)))>
  kgen.param.constant = <div(add(mul_no_wrap(p1, 6), mul_no_wrap(p2, 4)), 2)>

  // No distribution when one term doesn't cleanly divide.
  // CHECK: = kgen.param.constant = <to_builtin(:scalar<index> div(add(mul_no_wrap(from_builtin(p1), 4), from_builtin(p2)), 4))>
  kgen.param.constant = <div(add(mul_no_wrap(p1, 4), p2), 4)>

  // CHECK: = kgen.param.constant: si64 = <1>
  kgen.param.constant: si64 = <div(-4, -4)>

  // CHECK: = kgen.param.constant: si64 = <3>
  kgen.param.constant: si64 = <div(11, 3)>

  // CHECK: = kgen.param.constant: si64 = <-3>
  kgen.param.constant: si64 = <div(-11, 3)>

  // CHECK: = kgen.param.constant: si64 = <-3>
  kgen.param.constant: si64 = <div(11, -3)>

  // CHECK: = kgen.param.constant: si64 = <3>
  kgen.param.constant: si64 = <div(-11, -3)>

  // CHECK: = kgen.param.constant: si64 = <3>
  kgen.param.constant: si64 = <div(11, 3)>

  // Test that the high-bit is interpreted correctly for unsigned integers
  // CHECK: = kgen.param.constant: ui64 = <4611686018427387904>
  kgen.param.constant: ui64 = <div(9223372036854775808, 2)>

  // CHECK: = kgen.param.constant: ui64 = <9223372036854775807>
  kgen.param.constant: ui64 = <div(18446744073709551615, 2)>

  kgen.param.constant = <mul(p1, 1)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <mul(p1, 0, p2)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <mul_no_wrap(p1, 1)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <mul_no_wrap(p1, 0, p2)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <and(12, 6)>  // CHECK: kgen.param.constant = <4>
  kgen.param.constant = <or(12, 6)>  // CHECK: kgen.param.constant = <14>
  kgen.param.constant = <xor(4, 6)>  // CHECK: kgen.param.constant = <2>
  kgen.param.constant = <shl(p1, 2)>  // CHECK: kgen.param.constant = <to_builtin({{.*}}mul({{.*}}from_builtin({{.*}}p1{{.*}}), 4))>
  kgen.param.constant = <shl(p1, 0)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <shr(p1, 0)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <div(p1, 1)>  // CHECK: kgen.param.constant = <p1>
  kgen.param.constant = <mod(p1, 1)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <mod(p1, 0)>  // CHECK: kgen.param.constant = <to_builtin({{.*}}mod({{.*}}from_builtin({{.*}}p1{{.*}}), 0))>
  kgen.param.constant = <mod(1, 0)>  // CHECK: kgen.param.constant = <to_builtin({{.*}}mod(1, 0))>
  kgen.param.constant = <mod(p1, p1)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <mod(mul(p1, 2), p1)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <mod(mul_no_wrap(p1, 2), p1)>  // CHECK: kgen.param.constant = <0>
  kgen.param.constant = <mod(add(p1, 2), p1)>  // CHECK: kgen.param.constant = <to_builtin({{.*}}mod(add({{.*}}from_builtin({{.*}}p1{{.*}}), 2), from_builtin({{.*}}p1{{.*}})))>
  kgen.param.constant = <max(mul_no_wrap(p1, 2), mul_no_wrap(2, p2))>  // CHECK: kgen.param.constant = <to_builtin({{.*}}mul_no_wrap(max({{.*}}from_builtin({{.*}}p1{{.*}}), from_builtin({{.*}}p2{{.*}})), 2))>
  kgen.param.constant = <max(mul(p1, 2), mul(2, p2))>  // CHECK: kgen.param.constant = <to_builtin({{.*}}max(mul({{.*}}from_builtin({{.*}}p1{{.*}}), 2), mul({{.*}}from_builtin({{.*}}p2{{.*}}), 2)))>
  kgen.param.constant = <max(mul_no_wrap(p1, 2), mul_no_wrap(p2, 3))>  // CHECK: kgen.param.constant = <to_builtin({{.*}}max(mul_no_wrap({{.*}}from_builtin({{.*}}p1{{.*}}), 2), mul_no_wrap({{.*}}from_builtin({{.*}}p2{{.*}}), 3)))>
  kgen.param.constant = <max(mul_no_wrap(p1, 2), mul_no_wrap(p2, 4))>  // CHECK: kgen.param.constant = <to_builtin({{.*}}max(mul_no_wrap({{.*}}from_builtin({{.*}}p1{{.*}}), 2), mul_no_wrap({{.*}}from_builtin({{.*}}p2{{.*}}), 4)))>
  kgen.param.constant = <max(add(p1, 2), add(p2, 2))>  // CHECK: kgen.param.constant = <to_builtin({{.*}}max(add({{.*}}from_builtin({{.*}}p1{{.*}}), 2), add({{.*}}from_builtin({{.*}}p2{{.*}}), 2)))>

  kgen.param.declare square = <mul(p1, p1)>  // CHECK: kgen.param.declare square = <to_builtin({{.*}}mul({{.*}}from_builtin({{.*}}p1{{.*}}), from_builtin({{.*}}p1{{.*}})))>
  kgen.param.constant = <square>  // CHECK: kgen.param.constant = <square>

  // CHECK = <eq(p1, *?)>
  kgen.param.declare unknown: i1 = <to_builtin(:scalar<bool> eq(*?, p1))>
  // CHECK: = <0>
  kgen.param.declare unknownEq: i1 = <to_builtin(:scalar<bool> eq(:dtype *?, f32))>
  // CHECK: = <1>
  kgen.param.declare unknownEqItself: i1 = <to_builtin(:scalar<bool> eq(:dtype *?, *?))>
  // CHECK: = <0>
  kgen.param.declare unknownEqIndex: i1 = <to_builtin(:scalar<bool> eq(*?, 1))>
  // CHECK: = <1>
  kgen.param.declare unknownEqItselfIndex: i1 = <to_builtin(:scalar<bool> eq(*?, *?))>

  // Make sure operand deduplication happens for nested operands too
  kgen.param.declare max = <max(max(p1, 1), p1)>
  // CHECK: kgen.param.declare max = <to_builtin(:scalar<index> max(from_builtin(p1), 1))>

  kgen.param.declare min = <min(min(p1, 1), p1)>
  // CHECK: kgen.param.declare min = <to_builtin(:scalar<index> min(from_builtin(p1), 1))>

  kgen.return
}

// CHECK-LABEL: kgen.generator @datalayout_operators()
kgen.generator @datalayout_operators() {
  // CHECK-NEXT: <4>
  kgen.param.constant: index = <get_sizeof(si32, #target)>
  // CHECK-NEXT: <3>
  kgen.param.constant: index = <get_sizeof(i20, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_sizeof(f64, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_sizeof(index, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_sizeof(!kgen.generator<() -> ()>, #target)>
  // CHECK-NEXT: <16>
  kgen.param.constant: index = <get_sizeof(!kgen.generator<() capturing -> ()>, #target)>

  // CHECK-NEXT: <4>
  kgen.param.constant: index = <get_alignof(si32, #target)>
  // CHECK-NEXT: <4>
  kgen.param.constant: index = <get_alignof(i20, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_alignof(f64, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_alignof(index, #target)>
  // CHECK-NEXT: <8>
  kgen.param.constant: index = <get_alignof(!kgen.generator<() -> ()>, #target)>

  // CHECK-NEXT: <1>
  kgen.param.constant: index = <get_alignof(!kgen.simd<0, f32>, #target)>

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
  // CHECK: kgen.param.constant: dtype = <si256>
  kgen.param.constant: dtype = <si256>
  kgen.return
}

// MLIR TYPES
// CHECK-LABEL: kgen.generator @type_params<dt: dtype, typeParam: type>()
kgen.generator @type_params<dt: dtype, typeParam: type>() {
  // CHECK: kgen.param.assert <to_builtin(:scalar<bool> eq(:type typeParam, scalar<f32>))>
  kgen.param.assert <to_builtin(:scalar<bool> eq(:type typeParam, !kgen.scalar<f32>))>, "f32 scalarzzz"
  // CHECK: kgen.param.declare ty1: type = <scalar<f32>>
  kgen.param.declare ty1: type = <scalar<f32>>

  // CHECK: kgen.param.declare ty2: type = <scalar<dt>>
  kgen.param.declare ty2: type = <scalar<dt>>

  // This op returns an SSA value whose type is specified by a type parameter.
  // CHECK: "test.someop"() : () -> !kgen.param<ty2>
  "test.someop"() : () -> !kgen.param<ty2>

  // kgen.paramref auto-folds non-parameterized types on construction.
  // CHECK: "test.someop"() : () -> !kgen.scalar<f32>
  "test.someop"() : () -> !kgen.param<!kgen.scalar<f32>>

  kgen.return
}

// STRING TYPES
// CHECK-LABEL: kgen.generator @string_params<a: string, b: string>()
kgen.generator @string_params<a: string, b: string>() {
  // CHECK: kgen.param.assert <to_builtin(:scalar<bool> eq(:string a, b))>, "samesies only"
  kgen.param.assert <to_builtin(:scalar<bool> eq(:string a, b))>, "samesies only"

  // CHECK: kgen.param.assert <to_builtin(:scalar<bool> in(:string a, [b, "foo"]))>, "samesies or foo"
  kgen.param.assert <to_builtin(:scalar<bool> in(:string a, [b, "foo"]))>, "samesies or foo"

  // CHECK: kgen.param.declare s1: string = <"exciting">
  kgen.param.declare s1: string = <"exciting">

  //kgen.param.declare s2: string = <concat("hello ", "world", "!!11oneone">

  kgen.return
}

// COM: TARGET TYPES

// CHECK-LABEL: kgen.generator @target_params2<t0: target>()
kgen.generator @target_params2<t0: target>() {
  // CHECK: kgen.param.assert <to_builtin(:scalar<bool> eq(:target t0, #kgen.target<triple = "triple", arch = "cpu", features = "features", data_layout = "p:32:32", simd_bit_width = 4>))>, "must support target!!"
  kgen.param.assert <to_builtin(:scalar<bool> eq(:target t0, #kgen.target<triple="triple", arch="cpu", features="features", data_layout="p:32:32", simd_bit_width=4>))>, "must support target!!"
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_has_feature<t0: target>()
kgen.generator @target_has_feature<t0: target>() {
  kgen.param.assert <target_has_feature(t0, "avx")>, "must support avx!"
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_is_gpu_triple<t0: target>()
kgen.generator @target_is_gpu_triple<t0: target>() {
  kgen.param.assert <to_builtin(:scalar<bool> eq(:string target_get_field(t0, "triple"), "nvptx64-nvidia-cuda"))>, "triple must be nvptx64-nvidia-cuda"
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_is_os<t0: target>()
kgen.generator @target_is_os<t0: target>() {
  kgen.param.assert <to_builtin(:scalar<bool> eq(:string target_get_field(t0, "os"), "darwin"))>, "os must be darwin"
  kgen.return
}

// CHECK-LABEL: kgen.generator @target_is_little_endian<t0: target>()
kgen.generator @target_is_little_endian<t0: target>() {
  kgen.param.assert <to_builtin(:scalar<bool> eq(:string target_get_field(t0, "endianness"), "little"))>, "target must be little endian"
  kgen.return
}


// CHECK-LABEL: kgen.generator @target_get_field()
kgen.generator @target_get_field() {
  kgen.param.assert<to_builtin(:scalar<bool> eq(128, target_get_field(#target, "simd_bit_width")))>,
                    "simd_bit_width is always greater than 1"
  kgen.param.assert<to_builtin(:scalar<bool> eq(:string target_get_field(#target, "os"), "darwin"))>,
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
  <r1: <>(si32) -> si32,
   // This uses a different parameter.
   // CHECK-SAME: r3: <dtype>() -> !kgen.scalar<*(0,0)>
   r3: <dtype>() -> !kgen.scalar<*(0,0)>
   >() {
  // use unaryFn
  kgen.return
}

kgen.generator @takeUnary<unaryFn: (!kgen.scalar<si32>) -> !kgen.scalar<si32>>() {
  // use unaryFn
  kgen.return
}

kgen.func @doubleExample(%arg0: !kgen.scalar<si32>) -> !kgen.scalar<si32> {
  %0 = pop.add %arg0, %arg0: !kgen.scalar<si32>
  kgen.return %0 : !kgen.scalar<si32>
}

kgen.generator @test_region() {
  // CHECK: kgen.call @takeUnary<:(!kgen.scalar<si32>) -> !kgen.scalar<si32> @doubleExample>()
  kgen.call @takeUnary<
     :(!kgen.scalar<si32>) -> !kgen.scalar<si32> @doubleExample>() : () -> ()

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
) -> (index, !kgen.param<*"index">) {
  // CHECK: %[[V0:.*]] = pop.load %[[ARG0]] : !kgen.pointer<index>
  %0 = pop.load %arg0 : !kgen.pointer<index>
  // CHECK: %[[V1:.*]] = pop.load %[[ARG1]] : !kgen.pointer<*"index">
  %1 = pop.load %arg1 : !kgen.pointer<*"index">
  // CHECK: return %[[V0]], %[[V1]] : index, !kgen.param<*"index">
  kgen.return %0, %1 : index, !kgen.param<*"index">
}

lit.struct.decl @A {}
lit.struct.decl @B {}

// CHECK-LABEL: @symbol_exprs
kgen.generator @symbol_exprs() {
  // CHECK: = kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(:scalar<index> from_builtin(get_sizeof(!lit.struct<@A>, #kgen.target<triple = "unknown", arch = "", simd_bit_width = 128>)), from_builtin(get_sizeof(!lit.struct<@B>, #kgen.target<triple = "unknown", arch = "", simd_bit_width = 128>))))>
  %0 = kgen.param.constant: i1 = <to_builtin(:scalar<bool> eq(:index get_sizeof(@A, #target),
                                  get_sizeof(@B, #target)))>
  kgen.return
}

kgen.generator @takeFnContextualType<ty: type, fn: ()->!kgen.param<ty>>() -> !kgen.param<ty> {
  %0 = kgen.call_param[()->!kgen.param<ty>: fn]()
  kgen.return %0: !kgen.param<ty>
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
  kgen.param.declare fn: <type, () -> !kgen.param<*(1,0)>>() -> !kgen.param<*(0,0)> = <@takeFnContextualType>

  // CHECK: kgen.param.declare boundFn: () -> index =
  // CHECK-SAME: <bind_params(:<type, () -> !kgen.param<*(1,0)>>() -> !kgen.param<*(0,0)> fn, :type index, :() -> index @sillyFn)>
  kgen.param.declare boundFn: () -> index =
    <bind_params(:<type, () -> !kgen.param<*(1,0)>>() -> !kgen.param<*(0,0)> fn, :type index, :() -> index @sillyFn)>
  %0 = kgen.call_param[()->index: boundFn]()

  kgen.return %0 : index
}

// CHECK-LABEL: @partialBindSignature
kgen.generator @partialBindSignature() -> index {
  kgen.param.declare fn: <type, () -> !kgen.param<*(1,0)>>() -> !kgen.param<*(0,0)> = <@takeFnContextualType>

  // CHECK: kgen.param.declare partiallyBound: <() -> index>() -> index =
  // CHECK-SAME: <bind_params(:<type, () -> !kgen.param<*(1,0)>>() -> !kgen.param<*(0,0)> fn, :type index, ?)>
  kgen.param.declare
    partiallyBound: <() -> index>() -> index =
      <bind_params(:<type, () -> !kgen.param<*(1,0)>>() -> !kgen.param<*(0,0)> fn, :type index, ?)>
  // CHECK: kgen.call_param[() -> index: bind_params(:<() -> index>() -> index partiallyBound, :() -> index @sillyFn)]()
  %0 = kgen.call_param[() -> index: bind_params(:<() -> index>() -> index partiallyBound, :() -> index @sillyFn)]()

  kgen.return %0 : index
}

// CHECK-LABEL: @partialBindSignature2
kgen.generator @partialBindSignature2() -> index {
  // CHECK: kgen.param.declare fn: <() -> index>() -> index = <@takeFnContextualType<:type index, :() -> index ?>>
  kgen.param.declare fn: <()->index>() -> index =
    <bind_params(:<type, ()->!kgen.param<*(1,0)>>() -> !kgen.param<*(0,0)> @takeFnContextualType, :type index, :() -> index ?)>
  // CHECK: kgen.param.declare fullyBound: () -> index = <bind_params(:<() -> index>() -> index fn, :() -> index @sillyFn)>
  kgen.param.declare fullyBound: () -> index = <bind_params(:<()->index>() -> index fn, :() -> index @sillyFn)>

  // CHECK: kgen.call_param[() -> index: bind_params(:<() -> index>() -> index fn, :() -> index @sillyFn)]()
  %0 = kgen.call_param[()->index: bind_params(:<()->index>()->index fn, :() -> index @sillyFn)]()
  // CHECK: kgen.call_param[() -> index: fullyBound]()
  %1 = kgen.call_param[()->index : fullyBound]()

  kgen.return %0 : index
}

kgen.generator @returnParam<T: type, I>(%arg : !kgen.param<T>) -> !kgen.param<T> {
 kgen.return %arg : !kgen.param<T>
}

// CHECK-LABEL: @partialBindSignature3
kgen.generator @partialBindSignature3<T: type>(%arg : !kgen.param<T>) {
  // CHECK-NEXT: kgen.param.declare fn: <type>(!kgen.param<*(0,0)>) -> !kgen.param<*(0,0)> = <@returnParam<:type ?, 32>>
  kgen.param.declare fn: <type>(!kgen.param<*(0,0)>) -> !kgen.param<*(0,0)> =
    <bind_params(:<type, index>(!kgen.param<*(0,0)>) -> !kgen.param<*(0,0)> @returnParam, ?, 32)>
  kgen.return
}

// CHECK-LABEL: @mlirOperationExpr
kgen.generator @mlirOperationExpr() {
  // CHECK: (index, index) -> index = <"index.add">
  kgen.param.declare indexAdd: (index, index) -> index = <"index.add">
  // CHECK: (index, index) -> i1 = <"index.cmp"{pred = #index<cmp_predicate slt>}>
  kgen.param.declare indexCmp: (index, index) -> i1 = <"index.cmp"{pred = #index<cmp_predicate slt>}>
  // CHECK: (!pop.array<2, si32>) -> si32 = <"pop.array.get"{index = 0 : index}>
  kgen.param.declare arrayGet: (!pop.array<2, si32>) -> si32 = <"pop.array.get"{index = 0 : index}>

  // CHECK: cmpResult: i1 = <1>
  kgen.param.declare cmpResult: i1 = <apply(:(index, index) -> i1 "index.cmp"{pred = #index<cmp_predicate eq>}, 3, 3)>
  kgen.return
}

kgen.generator @evaluator(%funcs: !kgen.pointer<!kgen.generator<() -> ()>>, %num: index) -> index {
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
lit.struct.decl @IndexParams1<a: si32, b: i64, c: f32> {}

// CHECK-LABEL: kgen.generator @indexParamRef
// CHECK-SAME: @IndexParams1<:si32 *(0,0), :i64 *(0,1), :f32 *(1,1)>
// CHECK-SAME: @IndexParams0<*(0,0), :f32 *(0,1)>
kgen.generator @indexParamRef<
  fn: <index, f32, <si32, i64>()
      -> !lit.struct<@IndexParams1<:si32 *(0,0), :i64 *(0,1), :f32 *(1,1)>>>()
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
  // CHECK: declare callable: <index, type>(!pop.array<*(0,0), *(0,1)>) -> () = <fn>
  kgen.param.declare partial_bound: <type>(!pop.array<c, *(0,0)>) -> () =
    <bind_params(:<index, type>(!pop.array<*(0,0), *(0,1)>) -> () callable, c, ?)>
  kgen.return
}

// CHECK-LABEL: @bindParams
kgen.generator @bindParams<c, d: type>() {
  kgen.param.declare.region fn = <a, b: type>(%arg0: !pop.array<a, b>) {
    kgen.return
  }

  // CHECK: declare bind0: <type>(!pop.array<c, *(0,0)>) -> () = <bind_params(:<index, type>(!pop.array<*(0,0), *(0,1)>) -> () fn, c, ?)>
  kgen.param.declare bind0: <type>(!pop.array<c, *(0,0)>) -> () =
    <#kgen.bind_params<:!kgen.generator<<index, type>(!pop.array<*(0,0), *(0,1)>) -> ()> fn, c, ?>>
  // CHECK: declare bind1: <index>(!pop.array<*(0,0), d>) -> () =
  // CHECK-SAME: <bind_params(:<index, type>(!pop.array<*(0,0), *(0,1)>) -> () fn, ?, :type d)>
  kgen.param.declare bind1: <index>(!pop.array<*(0,0), d>) -> () =
    <#kgen.bind_params<:!kgen.generator<<index, type>(!pop.array<*(0,0), *(0,1)>) -> ()> fn, ?, :type d>>
  // CHECK: declare bind_all: (!pop.array<c, d>) -> () =
  // CHECK-SAME: <bind_params(:<index, type>(!pop.array<*(0,0), *(0,1)>) -> () fn, c, :type d)>
  kgen.param.declare bind_all: (!pop.array<c, d>) -> () =
    <#kgen.bind_params<:!kgen.generator<<index, type>(!pop.array<*(0,0), *(0,1)>) -> ()> fn, c, :type d>>
  // CHECK: declare bind_discharged_all = <bind_params(:!lit.generator<<index, {<true, {{.*}}>, <true, {{.*}}>, <true, {{.*}}>}>index> ?, 1 | "111")>
  kgen.param.declare bind_discharged_all: index =
    <#kgen.bind_params<:!lit.generator<<index, {<true, loc("bind_params":1:1)>, <true, loc("bind_params":1:2)>, <true, loc("bind_params":1:3)>}>index> ?, 1 | "111">>
  // CHECK: declare bind_discharged_none: !lit.generator<{{.*}} | "000"
  kgen.param.declare bind_discharged_none: !lit.generator<<{<true, loc("bind_params":2:1)>, <true, loc("bind_params":2:2)>, <true, loc("bind_params":2:3)>}>index> =
    <#kgen.bind_params<:!lit.generator<<index, {<true, loc("bind_params":2:1)>, <true, loc("bind_params":2:2)>, <true, loc("bind_params":2:3)>}>index> ?, 1 | "000">>
  // CHECK: declare bind0_then_bind1: (!pop.array<c, d>) -> () =
  // CHECK-SAME: <bind_params(:<type>(!pop.array<c, *(0,0)>) -> () bind0, :type d)>
  kgen.param.declare bind0_then_bind1: (!pop.array<c, d>) -> () =
    <#kgen.bind_params<:!kgen.generator<<type>(!pop.array<c, *(0,0)>) -> ()> bind0, :type d>>
  // CHECK: declare bind1_then_bind0: (!pop.array<c, d>) -> () =
  // CHECK-SAME: <bind_params(:<index>(!pop.array<*(0,0), d>) -> () bind1, c)>
  kgen.param.declare bind1_then_bind0: (!pop.array<c, d>) -> () =
    <#kgen.bind_params<:!kgen.generator<<index>(!pop.array<*(0,0), d>) -> ()> bind1, c>>

  // NESTED BINDING
  // CHECK: declare nested_bind0_then_bind1: (!pop.array<c, d>) -> () =
  // CHECK-SAME: <bind_params(:<index, type>(!pop.array<*(0,0), *(0,1)>) -> () fn, c, :type d)>
  kgen.param.declare nested_bind0_then_bind1: (!pop.array<c, d>) -> () =
    <#kgen.bind_params<
      :!kgen.generator<<type>(!pop.array<c, *(0,0)>) -> ()>
        #kgen.bind_params<:!kgen.generator<<index, type>(!pop.array<*(0,0), *(0,1)>) -> ()> fn, c>,
      :type d>>
  // CHECK: declare nested_bind1_then_bind0: (!pop.array<c, d>) -> () =
  // CHECK-SAME: <bind_params(:<index, type>(!pop.array<*(0,0), *(0,1)>) -> () fn, c, :type d)>
  kgen.param.declare nested_bind1_then_bind0: (!pop.array<c, d>) -> () =
    <#kgen.bind_params<
      :!kgen.generator<<index>(!pop.array<*(0,0), d>) -> ()>
        #kgen.bind_params<:!kgen.generator<<index, type>(!pop.array<*(0,0), *(0,1)>) -> ()> fn, ?, :type d>,
      c>>
  // The outer mask is indexed over the residual constraints from the inner
  // bind_params. Inner discharged original constraint 1, so residual
  // constraint 0 maps back to original constraint 0. The flattened
  // original-indexed mask is therefore "110".
  // CHECK: declare nested_discharged: !lit.generator<<{<true, {{.*}}}>index> =
  // CHECK-SAME: <bind_params(:!lit.generator<<index, type, {<true, {{.*}}>, <true, {{.*}}>, <true, {{.*}}>}>index> ?, 1, :type i64 | "110")>
  kgen.param.declare nested_discharged: !lit.generator<<{<true, loc("bind_params":3:3)>}>index> =
    <#kgen.bind_params<
      :!lit.generator<<type, {<true, loc("bind_params":3:1)>, <true, loc("bind_params":3:3)>}>index>
        #kgen.bind_params<:!lit.generator<<index, type, {<true, loc("bind_params":3:1)>, <true, loc("bind_params":3:2)>, <true, loc("bind_params":3:3)>}>index> ?, 1, :type ? | "010">,
      :type i64 | "10">>
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

// Helper for apply_result_slot_depth_adj test:
//  1-param generatorwith argument types depend on the param T.
kgen.generator @ptr_slot<T: type>(%arg: !kgen.pointer<T>, %out: !kgen.pointer<!kgen.pointer<T>> byref_result) -> !kgen.none {
  %0 = kgen.param.constant: none = <#kgen.none>
  kgen.return %0 : !kgen.none
}

// CHECK-LABEL: kgen.generator @apply_result_slot_depth_adj
kgen.generator @apply_result_slot_depth_adj() {
  // Tests that IndexDepthAdjuster(-1) is applied in the ApplyResultSlot parser.
  //
  // #kgen.gen provides a ParameterScopeAttrInterface (depth-0) scope with two
  // params: *(0,0): type (T) and *(0,1): !kgen.pointer<*(0,0)> (a T-pointer).
  //
  // bind_params(@ptr_slot, :type *(0,0)) binds T to the gen's first param.
  // Inside the resulting zero-param <> FuncTypeGeneratorType (one extra scope
  // boundary), the bound value lifts from *(0,0) to *(1,0), so arg types
  // become (!kgen.pointer<*(1,0)>, ...).
  //
  // Without the fix, the operand *(0,1) would be parsed with the un-adjusted
  // type !kgen.pointer<*(1,0)> instead of the correct !kgen.pointer<*(0,0)>,
  // and -verify-parameters would reject the round-trip.
  // CHECK-NEXT: constant: <type, pointer<*(0,0)>>pointer<*(0,0)> = <#kgen.gen<apply_result_slot(
  // CHECK-SAME: :(!kgen.pointer<*(1,0)>, !kgen.pointer<pointer<*(1,0)>> byref_result) -> !kgen.none
  // CHECK-SAME: @ptr_slot<:type *(0,0)>, *(0,1))>>
  kgen.param.constant: <type, pointer<*(0,0)>>pointer<*(0,0)> =
    <#kgen.gen<apply_result_slot(
      :(!kgen.pointer<*(1,0)>, !kgen.pointer<pointer<*(1,0)>> byref_result) -> !kgen.none
        bind_params(
          :<type>(!kgen.pointer<*(0,0)>, !kgen.pointer<pointer<*(0,0)>> byref_result) -> !kgen.none
            @ptr_slot,
          :type *(0,0)
        ),
      *(0,1)
    )>>
  kgen.return
}

// CHECK-LABEL: @int_literal_param
kgen.generator @int_literal_param<abcd: !pop.int_literal>() {
  // CHECK-NEXT: constant: !pop.int_literal = <abcd>
  kgen.param.constant: !pop.int_literal = <abcd>
  kgen.return
}

kgen.generator @kernel() {
  kgen.return
}

// CHECK-LABEL: @compile_assembly
kgen.generator @compile_assembly<emission_kind: index>() {
  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>>
  kgen.param.declare amd: target = <#kgen.target<triple = "amdgcn-amd-amdhsa", arch = "", data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", simd_bit_width = 128>>

  // CHECK: constant: string = <#kgen.compile_assembly<nvptx, =asm, "", false, :() -> () @kernel>>
  kgen.param.constant: string = <#kgen.compile_assembly<nvptx, =asm, "", false, :() -> () @kernel>>
  // CHECK: constant: string = <#kgen.compile_assembly<nvptx, =llvm, "", true, :() -> () @kernel>>
  kgen.param.constant: string = <#kgen.compile_assembly<nvptx, =llvm, "", true, :() -> () @kernel>>
  // CHECK: constant: string = <#kgen.compile_assembly<nvptx, =llvm, "option1=value1,option2=value2", true, :() -> () @kernel>>
  kgen.param.constant: string = <#kgen.compile_assembly<nvptx, =llvm, "option1=value1,option2=value2", true, :() -> () @kernel>>

  // CHECK: constant: string = <#kgen.compile_assembly<amd, =object, "", true, :() -> () @kernel>>
  kgen.param.constant: string = <#kgen.compile_assembly<amd, =object, "", true, :() -> () @kernel>>
  // CHECK: constant: string = <#kgen.compile_assembly<nvptx, emission_kind, "", true, :() -> () @kernel>>
  kgen.param.constant: string = <#kgen.compile_assembly<nvptx, emission_kind, "", true, :() -> () @kernel>>

  kgen.return
}

// CHECK-LABEL: @compile_offload_closure
kgen.generator @compile_offload_closure() {
  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>>
  // CHECK: constant: string = <#kgen.compile_offload_closure<nvptx, #kgen.symbol.constant<@kernel> : !kgen.generator<() -> ()>>>
  kgen.param.constant: string = <#kgen.compile_offload_closure<nvptx, #kgen.symbol.constant<@kernel> : !kgen.generator<() -> ()>>>
  kgen.return
}

// CHECK-LABEL: @get_likage_name
kgen.generator @get_likage_name() {
  // CHECK: constant: string = <#kgen.get_linkage_name<current_target(), #kgen.symbol.constant<@kernel> : !kgen.generator<() -> ()>>>
  kgen.param.constant: string = <#kgen.get_linkage_name<current_target(), #kgen.symbol.constant<@kernel> : !kgen.generator<() -> ()>>>
  kgen.return
}

// CHECK-LABEL: @unification
kgen.generator @unification() {
  // CHECK: T0: type = <!lit.struct<@unification>>
  kgen.param.declare T0: type = <rebind(:!metatype.type #kgen.type<!lit.struct<@unification>>)>
  kgen.return
}

// CHECK-LABEL: kgen.generator @rebind_desugar
kgen.generator @rebind_desugar<val: !kgen.param<:type sugar_alias(*"index", index)>>() {
  // CHECK-NEXT: kgen.param.declare desugar = <val>
  kgen.param.declare desugar : index = <rebind(:!kgen.param<:type sugar_alias(*"index", index)> val)>
  kgen.return
}

// CHECK-LABEL: @struct_extract
kgen.generator @struct_extract<idx: index>() {
  // CHECK-NEXT: <2>
  kgen.param.constant = <#kgen.struct.extract<:struct<(index, index)> { 1, 2 }, 1>>
  // CHECK-NEXT: <#interp.uninitmem>
  kgen.param.constant = <#kgen.struct.extract<:struct<(index, index)> #interp.uninitmem, 0>>

  // CHECK-NEXT: <#kgen.struct.extract<:struct<(index, index)> { 1, 2 }, idx>>
  kgen.param.constant = <#kgen.struct.extract<:struct<(index, index)> { 1, 2 }, idx>>
  kgen.return
}

// CHECK-LABEL: @data_to_str
kgen.generator @data_to_str<s1: struct<(pointer<none>, index)>,
                            s2: struct<(pointer<none>, index)>,
                            s3: struct<(pointer<none>, index)>>() {
  // CHECK: = kgen.param.constant: string = <data_to_str(:struct<(pointer<none>, index)> s1, [])>
  %0 = kgen.param.constant: string = <data_to_str(:struct<(pointer<none>, index)> s1, [])>

  // CHECK: = kgen.param.constant: string = <data_to_str(:struct<(pointer<none>, index)> s1, [s2, s3])>
  %1 = kgen.param.constant: string = <data_to_str(:struct<(pointer<none>, index)> s1, [s2, s3])>
  kgen.return
}

// CHECK-LABEL: @string_address
kgen.generator @string_address<s1: string>() {
  // CHECK: %struct = kgen.param.constant: struct<(pointer<none>, index)> = <{ string_address(""), 0 }>
  %0 = kgen.param.constant: struct<(pointer<none>, index)> = <{ string_address(""), 0 }>

  // CHECK: %pointer = kgen.param.constant: pointer<none> = <string_address(s1)>
  %1 = kgen.param.constant: pointer<none> = <string_address(s1)>

  kgen.return
}

// CHECK-LABEL: @gen_attr
kgen.generator @gen_attr<a: index>() {
  // CHECK: = kgen.param.constant: <>index = <#kgen.gen<to_builtin(:scalar<index> add(from_builtin(a), 3))>>
  %0 = kgen.param.constant: !kgen.generator<<>index> = <#kgen.gen<add(a, 3)>>

  // CHECK: = kgen.param.constant: <index>index = <#kgen.gen<to_builtin(:scalar<index> add(from_builtin(*(0,0)), 1))>>
  %1 = kgen.param.constant: !kgen.generator<<index>index> = <#kgen.gen<add(*(0,0), 1)>>

  // CHECK: = kgen.param.constant: <index, index>index = <#kgen.gen<to_builtin(:scalar<index> add(from_builtin(*(0,0)), from_builtin(*(0,1))))>>
  %2 = kgen.param.constant: !kgen.generator<<index, index>index> = <#kgen.gen<add(*(0,0), *(0,1))>>

  kgen.return
}
