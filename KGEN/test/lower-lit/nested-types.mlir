// RUN: kgen-opt -allow-unregistered-dialect -split-input-file -lower-lit %s | FileCheck %s

// CHECK-LABEL: kgen.generator.interface @ptr_itf

kgen.generator.interface @ptr_itf<eltype: type, rettype: type>
    (!pop.pointer<eltype>) -> !kgen.paramref<rettype>

// CHECK: kgen.generator @eltype_inf_thunk
// CHECK-NEXT: eq(:type eltype, !pop.simd<1, f32>)

// Implementation specifies that `eltype` must be `!pop.simd<1, f32>`
lit.func @eltype_inf<rettype: type>
    (%arg0: !pop.pointer<simd<1, f32>>) -> !kgen.paramref<rettype>
    implements @ptr_itf {
  %0 = "a"() : () -> !kgen.paramref<rettype>
  kgen.return %0 : !kgen.paramref<rettype>
}

// CHECK: kgen.generator @rettype_inf_thunk
// CHECK-NEXT: eq(:type rettype, index)

// Implementation specifies that `rettype` must be `index`.
lit.func @rettype_inf<eltype: type>
    (%arg0: !pop.pointer<eltype>) -> index
    implements @ptr_itf {
  %0 = "a"() : () -> index
  kgen.return %0 : index
}

// -----

// CHECK-LABEL: kgen.generator.interface @struct_itf
kgen.generator.interface @struct_itf<eltype: type, dtype: dtype>
    (!pop.struct<eltype, simd<1, dtype>>) -> ()

// CHECK: kgen.generator @struct_inf_thunk
// CHECK-NEXT: constraints <
// CHECK-NEXT: eq(:type eltype, index)
// CHECK-NEXT: eq(:dtype dtype, f32)

lit.func @struct_inf
    (%arg0: !pop.struct<index, simd<1, f32>>) -> ()
    implements @struct_itf {
  kgen.return
}

// -----

kgen.struct.decl @Foo<N> {}

kgen.generator.interface @doFoo<N>(%a: !kgen.ref<@Foo<N = N>>)

// CHECK-LABEL: kgen.generator @doFooImpl_thunk<N>
// CHECK: constraints <[eq(N, 1)
// CHECK: kgen.rebind %arg0 : !kgen.ref<@Foo<N = N>> to !kgen.ref<@Foo<N = 1>>
lit.func @doFooImpl<N>(%a: !kgen.ref<@Foo<N = 1>>) implements @doFoo {
  kgen.return
}
