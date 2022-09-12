// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.generator @param_expr
kgen.generator @param_expr<p1, p2, int1: i1, type: dtype, type2: dtype>()  {
  // Generic attr syntax in generic ops
  // CHECK: "someop"() {
  "someop" () {
    // CHECK-SAME: use1 = #kgen.param.expr<add, #kgen.param.decl.ref<p1> : index, 42 : index>
    use1 = #kgen.param.expr<add, #kgen.param.decl.ref<*"p1"> : index, 42 : index> : index,
    // CHECK-SAME: use2 = #kgen.param.expr<add, #kgen.param.decl.ref<p1> : index, 43 : index>
    use2 = #kgen.param.expr<add, 1 : index, #kgen.param.decl.ref<p1> : index, 42 : index> : index,
    // CHECK-SAME: use3 = 3 : index
    use3 = #kgen.param.expr<add, 1 : index, 2 : index> : index
  } : () -> ()
  // Generic syntax in known contexts

  // CHECK: = kgen.param.constant = <add(p1, 42)>
  %0 = kgen.param.constant = <#kgen.param.expr<add, #kgen.param.decl.ref<p1> : index, 42 : index>>

  // CHECK: = kgen.param.constant = <add(mul(p2, p2), p1, 42)>
  %1 = kgen.param.constant = <add(p1, 42, mul(p2, p2))>

  // CHECK: = kgen.param.constant = <mul(p1, p2, 84)>
  %2 = kgen.param.constant = <mul(p1, 42, add(p2, p2))>

  // CHECK: = kgen.param.constant : i1 = <eq(p1, 42)>
  %3 = kgen.param.constant: i1 = <eq(42, p1)>

  // CHECK: = kgen.param.constant : i1 = <0>
  %4 = kgen.param.constant: i1 = <eq(41, 42)>

  // CHECK: = kgen.param.constant : i1 = <1>
  %5 = kgen.param.constant: i1 = <1>

  // CHECK: = kgen.param.constant : i1 = <eq(:dtype type, f32)>
  %6 = kgen.param.constant: i1 = <eq(:dtype type, f32)>

  // CHECK: = kgen.param.constant : i1 = <0>
  %7 = kgen.param.constant: i1 = <eq(:dtype bf16, f16)>

  // CHECK: = kgen.param.constant : i1 = <in(p1, [add(p2, 1), p2, 1, 3])>
  %8 = kgen.param.constant: i1 = <in(p1, [3, 1, p2, add(p2, 1), 1])>

  // CHECK: = kgen.param.constant : i1 = <0>
  %9 = kgen.param.constant: i1 = <in(0, [1, 2])>

  // CHECK: = kgen.param.constant : i1 = <0>
  %10 = kgen.param.constant: i1 = <in(p1, [])>

  // CHECK: = kgen.param.constant : i1 = <1>
  %11 = kgen.param.constant: i1 = <in(p1, [p1, 1])>

  // CHECK: = kgen.param.constant : i1 = <eq(p1, 1)>
  %12 = kgen.param.constant: i1 = <in(p1, [1])>

  // CHECK: = kgen.param.constant : i1 = <in(:dtype f32, [type, f64])>
  %13 = kgen.param.constant: i1 = <in(:dtype f32, [f64, type, f64, type])>

  // CHECK: = kgen.param.constant : i1 = <0>
  %14 = kgen.param.constant: i1 = <in(:dtype f32, [si64, f64])>

  // CHECK: = kgen.param.constant : i1 = <0>
  %15 = kgen.param.constant: i1 = <in(:dtype type, [])>

  // CHECK: = kgen.param.constant : i1 = <1>
  %16 = kgen.param.constant: i1 = <in(:dtype type, [type, f32])>

  // CHECK: = kgen.param.constant : i1 = <in(:dtype type, [type2, f32])>
  %17 = kgen.param.constant: i1 = <in(:dtype type, [type2, f32])>

  // CHECK: = kgen.param.constant : i1 = <eq(:dtype type, f32)>
  %18 = kgen.param.constant: i1 = <in(:dtype type, [f32])>

  // The only binary operation that signless i1 supports is xor.
  // CHECK: = kgen.param.constant : i1 = <not(int1)>
  %19 = kgen.param.constant: i1 = <xor(int1, 1)>

  // CHECK: = kgen.param.constant : i1 = <not(int1)>
  %20 = kgen.param.constant: i1 = <not(int1)>

  // CHECK: = kgen.param.constant : i1 = <ne(:dtype type, f32)>
  %21 = kgen.param.constant: i1 = <xor(eq(:dtype type, f32), 1)>

  kgen.return
}

// CHECK-LABEL: kgen.generator @int1_aliases
kgen.generator @int1_aliases<p1, p2, int1: i1, type: dtype>()  {

  // CHECK: = kgen.param.constant : i1 = <ne(:dtype type, f32)>
  %0 = kgen.param.constant: i1 = <ne(:dtype type, f32)>

  // CHECK: = kgen.param.constant : i1 = <ne(p1, 42)>
  %1 = kgen.param.constant: i1 = <ne(p1, 42)>

  // CHECK: = kgen.param.constant : i1 = <not(int1)>
  %2 = kgen.param.constant: i1 = <not(int1)>

  // CHECK: = kgen.param.constant : i1 = <ge(p1, p2)>
  %3 = kgen.param.constant: i1 = <ge(p1, p2)>

  // CHECK: = kgen.param.constant : i1 = <ge(p1, 43)>
  %4 = kgen.param.constant: i1 = <gt(p1, 42)>

  // CHECK: = kgen.param.constant : i1 = <ge(p1, 42)>
  %5 = kgen.param.constant: i1 = <ge(p1, 42)>

  // CHECK: = kgen.param.constant : i1 = <ge(p1, 4)>
  %6 = kgen.param.constant: i1 = <le(4, p1)>

  // CHECK: = kgen.param.constant : i1 = <ge(p1, 5)>
  %7 = kgen.param.constant: i1 = <lt(4, p1)>

  // Shouldn't fold `index` constant expressions that differ for 32-/64-bit
  // targets without target info.
  // CHECK: = kgen.param.constant = <div(6000000000, 4)>
  %8 = kgen.param.constant = <div(6000000000, 4)> // 6B/4 differs.

  // CHECK: = kgen.param.constant = <8589934592>
  %9 = kgen.param.constant = <shl(1, 33)>

  // CHECK: = kgen.param.constant : i1 = <not(in(p1, [add(p2, 1), p2, 1, 3]))>
  %10 = kgen.param.constant: i1 = <not_in(p1, [3, 1, p2, add(p2, 1), 1])>

  // CHECK: = kgen.param.constant : i1 = <1>
  %11 = kgen.param.constant: i1 = <not_in(0, [1, 2])>

  // CHECK: = kgen.param.constant : i1 = <1>
  %12 = kgen.param.constant: i1 = <not_in(p1, [])>

  // CHECK: = kgen.param.constant : i1 = <0>
  %13 = kgen.param.constant: i1 = <not_in(p1, [p1, 1])>

  // CHECK: = kgen.param.constant : i1 = <ne(p1, 1)>
  %14 = kgen.param.constant: i1 = <not_in(p1, [1])>

  // CHECK: = kgen.param.constant : i1 = <not(in(:dtype f32, [type, f64]))>
  %15 = kgen.param.constant: i1 = <not_in(:dtype f32, [f64, type, f64, type])>

  // CHECK: = kgen.param.constant : i1 = <1>
  %16 = kgen.param.constant: i1 = <not_in(:dtype f32, [si64, f64])>

  // CHECK: = kgen.param.constant : i1 = <1>
  %17 = kgen.param.constant: i1 = <not_in(:dtype type, [])>

  // CHECK: = kgen.param.constant : i1 = <0>
  %18 = kgen.param.constant: i1 = <not_in(:dtype type, [type, f32])>

  // CHECK: = kgen.param.constant : i1 = <ne(:dtype type, f32)>
  %19 = kgen.param.constant: i1 = <not_in(:dtype type, [f32])>

  // This can't be folded because it is target specific: true on 32-bit and
  // false on 64-bit.
  // CHECK: = kgen.param.constant : i1 = <in(0, [4294967296, 8589934592])>
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

// CHECK-LABEL: kgen.generator @dtype_params<dt: dtype, *"f32", *"ui32">()
kgen.generator @dtype_params<dt: !kgen.dtype, *"f32", *"ui32">() {

  // Make sure that kgen keywords are printed properly escaped.
  // CHECK: kgen.param.constant = <add(*"f32", *"ui32")>
  kgen.param.constant  = <add(*"f32", *"ui32")>

  // CHECK: kgen.param.constant : dtype = <f32>
  kgen.param.constant : !kgen.dtype = <#kgen.dtype.constant<f32>>

  // CHECK: kgen.param.constant : dtype = <f32>
  kgen.param.constant : !kgen.dtype = <f32>
  // CHECK: kgen.param.constant : dtype = <ui128>
  kgen.param.constant : dtype = <ui128>
  kgen.return
}

// CHECK-LABEL: kgen.generator @type_params<dt: dtype, typeParam: type>()
kgen.generator @type_params<dt: dtype, typeParam: type>()
// CHECK: constraints <[eq(:type typeParam, !meta.scalar<f32>), "f32 scalarzzz", #{{.*}}]> {
   constraints <[eq(:type typeParam, !meta.scalar<f32>), "f32 scalarzzz"]>
 {
  // CHECK: kgen.param.declare ty1: type = <!meta.scalar<f32>>
  kgen.param.declare ty1: type = <!meta.scalar<f32>>

  // CHECK: kgen.param.declare ty2: type = <!meta.scalar<dt>>
  kgen.param.declare ty2: type = <!meta.scalar<dt>>

  // This op returns an SSA value whose type is specified by a type parameter.
  // CHECK: "someop"() : () -> !kgen.paramref<ty2>
  "someop"() : () -> !kgen.paramref<ty2>

  // kgen.paramref auto-folds non-parameterized types on construction.
  // CHECK: "someop"() : () -> !meta.scalar<f32>
  "someop"() : () -> !kgen.paramref<!meta.scalar<f32>>

  kgen.return
}

