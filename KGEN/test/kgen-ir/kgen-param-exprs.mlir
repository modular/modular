// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: kgen.generator @param_expr
kgen.generator @param_expr<p1, p2, int1: i1, type: dtype, type2: dtype>()  {
  // Generic attr syntax in generic ops
  // CHECK: "someop"() {
  "someop" () {
    // CHECK-SAME: use1 = #kgen.param.expr<add, #kgen.param.decl.ref<p1> : index, 42 : index>
    use1 = #kgen.param.expr<add, #kgen.param.decl.ref<"p1"> : index, 42 : index> : index,
    // CHECK-SAME: use2 = #kgen.param.expr<add, #kgen.param.decl.ref<p1> : index, 43 : index>
    use2 = #kgen.param.expr<add, 1 : index, #kgen.param.decl.ref<"p1"> : index, 42 : index> : index,
    // CHECK-SAME: use3 = 3 : index
    use3 = #kgen.param.expr<add, 1 : index, 2 : index> : index
  } : () -> ()
  // Generic syntax in known contexts

  // CHECK: = kgen.param.value = <add(p1, 42)>
  %0 = kgen.param.value = <#kgen.param.expr<add, #kgen.param.decl.ref<"p1"> : index, 42 : index>>

  // CHECK: = kgen.param.value = <add(mul(p2, p2), p1, 42)>
  %1 = kgen.param.value = <add(p1, 42, mul(p2, p2))>

  // CHECK: = kgen.param.value = <mul(p1, p2, 84)>
  %2 = kgen.param.value = <mul(p1, 42, add(p2, p2))>

  // CHECK: = kgen.param.value : i1 = <eq(p1, 42)>
  %3 = kgen.param.value: i1 = <eq(42, p1)>

  // CHECK: = kgen.param.value : i1 = <0>
  %4 = kgen.param.value: i1 = <eq(41, 42)>

  // CHECK: = kgen.param.value : i1 = <1>
  %5 = kgen.param.value: i1 = <1>

  // CHECK: = kgen.param.value : i1 = <eq(:dtype type, f32)>
  %6 = kgen.param.value: i1 = <eq(:dtype type, f32)>

  // CHECK: = kgen.param.value : i1 = <0>
  %7 = kgen.param.value: i1 = <eq(:dtype bf16, f16)>

  // CHECK: = kgen.param.value : i1 = <in(p1, [add(p2, 1), p2, 1, 3])>
  %8 = kgen.param.value: i1 = <in(p1, [3, 1, p2, add(p2, 1), 1])>

  // CHECK: = kgen.param.value : i1 = <0>
  %9 = kgen.param.value: i1 = <in(0, [1, 2])>

  // CHECK: = kgen.param.value : i1 = <0>
  %10 = kgen.param.value: i1 = <in(p1, [])>

  // CHECK: = kgen.param.value : i1 = <1>
  %11 = kgen.param.value: i1 = <in(p1, [p1, 1])>

  // CHECK: = kgen.param.value : i1 = <eq(p1, 1)>
  %12 = kgen.param.value: i1 = <in(p1, [1])>

  // CHECK: = kgen.param.value : i1 = <in(:dtype f32, [type, f64])>
  %13 = kgen.param.value: i1 = <in(:dtype f32, [f64, type, f64, type])>

  // CHECK: = kgen.param.value : i1 = <0>
  %14 = kgen.param.value: i1 = <in(:dtype f32, [si64, f64])>

  // CHECK: = kgen.param.value : i1 = <0>
  %15 = kgen.param.value: i1 = <in(:dtype type, [])>

  // CHECK: = kgen.param.value : i1 = <1>
  %16 = kgen.param.value: i1 = <in(:dtype type, [type, f32])>

  // CHECK: = kgen.param.value : i1 = <in(:dtype type, [type2, f32])>
  %17 = kgen.param.value: i1 = <in(:dtype type, [type2, f32])>

  // CHECK: = kgen.param.value : i1 = <eq(:dtype type, f32)>
  %18 = kgen.param.value: i1 = <in(:dtype type, [f32])>

  // The only binary operation that signless i1 supports is xor.
  // CHECK: = kgen.param.value : i1 = <not(int1)>
  %19 = kgen.param.value: i1 = <xor(int1, 1)>

  // CHECK: = kgen.param.value : i1 = <not(int1)>
  %a19 = kgen.param.value: i1 = <not(int1)>

  // CHECK: = kgen.param.value : i1 = <ne(:dtype type, f32)>
  %20 = kgen.param.value: i1 = <xor(eq(:dtype type, f32), 1)>

  kgen.return
}

// CHECK-LABEL: kgen.generator @int1_aliases
kgen.generator @int1_aliases<p1, p2, int1: i1, type: dtype>()  {

  // CHECK: = kgen.param.value : i1 = <ne(:dtype type, f32)>
  %0 = kgen.param.value: i1 = <ne(:dtype type, f32)>

  // CHECK: = kgen.param.value : i1 = <ne(p1, 42)>
  %1 = kgen.param.value: i1 = <ne(p1, 42)>

  // CHECK: = kgen.param.value : i1 = <not(int1)>
  %2 = kgen.param.value: i1 = <not(int1)>

  // CHECK: = kgen.param.value : i1 = <ge(p1, p2)>
  %3 = kgen.param.value: i1 = <ge(p1, p2)>

  // CHECK: = kgen.param.value : i1 = <ge(p1, 43)>
  %4 = kgen.param.value: i1 = <gt(p1, 42)>

  // CHECK: = kgen.param.value : i1 = <ge(p1, 42)>
  %5 = kgen.param.value: i1 = <ge(p1, 42)>

  // CHECK: = kgen.param.value : i1 = <ge(p1, 4)>
  %6 = kgen.param.value: i1 = <le(4, p1)>

  // CHECK: = kgen.param.value : i1 = <ge(p1, 5)>
  %7 = kgen.param.value: i1 = <lt(4, p1)>

  kgen.return
}

// CHECK-LABEL: kgen.generator @param_canonicalize
kgen.generator @param_canonicalize<p1, p2>()  {
  // CHECK: = kgen.param.value = <add(mul(p1, 4), mul(p2, 4))>
  kgen.param.value = <mul(add(p1, p2), 4)>

  // CHECK: = kgen.param.value = <add(mul(p2, p2), p1, 42)>
  kgen.param.value = <add(p1, 42, mul(p2, p2))>

  // CHECK: = kgen.param.value = <add(mul(p1, 3), 42)>
  kgen.param.value = <add(p1, 42, mul(p1, 2))>

  kgen.param.value = <mul(p1, 1)>  // CHECK: kgen.param.value = <p1>
  kgen.param.value = <mul(p1, 0, p2)>  // CHECK: kgen.param.value = <0>
  kgen.param.value = <and(12, 6)>  // CHECK: kgen.param.value = <4>
  kgen.param.value = <or(12, 6)>  // CHECK: kgen.param.value = <14>
  kgen.param.value = <xor(4, 6)>  // CHECK: kgen.param.value = <2>
  kgen.param.value = <shl(p1, 2)>  // CHECK: kgen.param.value = <mul(p1, 4)>
  kgen.param.value = <shl(p1, 0)>  // CHECK: kgen.param.value = <p1>
  kgen.param.value = <shr(p1, 0)>  // CHECK: kgen.param.value = <p1>
  kgen.param.value = <div(p1, 1)>  // CHECK: kgen.param.value = <p1>
  kgen.param.value = <mod(p1, 1)>  // CHECK: kgen.param.value = <0>

  kgen.param.bind square = <mul(p1, p1)>  // CHECK: kgen.param.bind square = <mul(p1, p1)>
  kgen.param.value = <square>  // CHECK: kgen.param.value = <square>

  kgen.return
}

// CHECK-LABEL: kgen.generator @dtype_params<dt: dtype, "f32", "ui32">()
kgen.generator @dtype_params<dt: !kgen.dtype, "f32", "ui32">() {

  // Make sure that kgen keywords are printed in quotes.
  // CHECK: kgen.param.value = <add("f32", "ui32")>
  kgen.param.value  = <add("f32", "ui32")>

  // CHECK: kgen.param.value : dtype = <f32>
  kgen.param.value : !kgen.dtype = <#kgen.dtype.constant<f32>>

  // CHECK: kgen.param.value : dtype = <f32>
  kgen.param.value : !kgen.dtype = <f32>
  // CHECK: kgen.param.value : dtype = <ui128>
  kgen.param.value : dtype = <ui128>
  kgen.return
}
