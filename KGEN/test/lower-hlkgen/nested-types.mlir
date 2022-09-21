// RUN: kgen-opt -allow-unregistered-dialect -split-input-file -lower-hlkgen %s | FileCheck %s

// CHECK-LABEL: kgen.generator.interface @ptr_itf

kgen.generator.interface @ptr_itf<eltype: type, rettype: type>
    (!meta.pointer<eltype>) -> !kgen.paramref<rettype>

// CHECK: kgen.generator @eltype_inf_thunk
// CHECK-NEXT: eq(:type eltype, !meta.scalar<f32>)

// Implementation specifies that `eltype` must be `!meta.scalar<f32>`
hlkgen.generator @eltype_inf<rettype: type>
    (%arg0: !meta.pointer<!meta.scalar<f32>>) -> !kgen.paramref<rettype>
    implements @ptr_itf {
  %0 = "a"() : () -> !kgen.paramref<rettype>
  kgen.return %0 : !kgen.paramref<rettype>
}

// CHECK: kgen.generator @rettype_inf_thunk
// CHECK-NEXT: eq(:type rettype, index)

// Implementation specifies that `rettype` must be `index`.
hlkgen.generator @rettype_inf<eltype: type>
    (%arg0: !meta.pointer<eltype>) -> index
    implements @ptr_itf {
  %0 = "a"() : () -> index
  kgen.return %0 : index
}
